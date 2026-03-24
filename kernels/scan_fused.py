from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from mamba_minimal.selective_scan import selective_scan_ref

try:
    import triton  # type: ignore
    import triton.language as tl  # type: ignore

    TRITON_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    triton = None
    tl = None
    TRITON_AVAILABLE = False


@dataclass(slots=True)
class KernelMetadata:
    backend: str
    used_fallback: bool
    notes: str


@dataclass(frozen=True, slots=True)
class KernelSupportInfo:
    supported: bool
    layout: str
    reason: str


def _next_power_of_two(value: int) -> int:
    if value <= 1:
        return 1
    return 1 << (value - 1).bit_length()


if TRITON_AVAILABLE:

    @triton.jit
    def _softplus(x):
        threshold = 20.0
        return tl.where(x > threshold, x, tl.log(1.0 + tl.exp(x)))


    @triton.jit
    def _selective_scan_chunk_kernel(
        u_ptr,
        delta_ptr,
        a_ptr,
        b_ptr,
        c_ptr,
        d_ptr,
        z_ptr,
        state_in_ptr,
        state_out_ptr,
        y_ptr,
        stride_u_b,
        stride_u_d,
        stride_u_l,
        stride_delta_b,
        stride_delta_d,
        stride_delta_l,
        stride_a_d,
        stride_a_n,
        stride_b_b,
        stride_b_d,
        stride_b_n,
        stride_b_l,
        stride_c_b,
        stride_c_d,
        stride_c_n,
        stride_c_l,
        stride_state_b,
        stride_state_d,
        stride_state_n,
        stride_y_b,
        stride_y_d,
        stride_y_l,
        channels,
        state_size,
        length,
        USE_D: tl.constexpr,
        USE_Z: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_L: tl.constexpr,
    ):
        pid = tl.program_id(0)
        batch_idx = pid // channels
        channel_idx = pid % channels

        offs_n = tl.arange(0, BLOCK_N)
        mask_n = offs_n < state_size

        a = tl.load(
            a_ptr + channel_idx * stride_a_d + offs_n * stride_a_n,
            mask=mask_n,
            other=0.0,
        ).to(tl.float32)
        state = tl.load(
            state_in_ptr
            + batch_idx * stride_state_b
            + channel_idx * stride_state_d
            + offs_n * stride_state_n,
            mask=mask_n,
            other=0.0,
        ).to(tl.float32)

        d_skip = 0.0
        if USE_D:
            d_skip = tl.load(d_ptr + channel_idx).to(tl.float32)

        for token_idx in tl.range(0, BLOCK_L):
            active = token_idx < length
            u_t = tl.load(
                u_ptr + batch_idx * stride_u_b + channel_idx * stride_u_d + token_idx * stride_u_l,
                mask=active,
                other=0.0,
            ).to(tl.float32)
            delta_t = tl.load(
                delta_ptr
                + batch_idx * stride_delta_b
                + channel_idx * stride_delta_d
                + token_idx * stride_delta_l,
                mask=active,
                other=0.0,
            ).to(tl.float32)
            delta_t = _softplus(delta_t)

            b_t = tl.load(
                b_ptr
                + batch_idx * stride_b_b
                + channel_idx * stride_b_d
                + offs_n * stride_b_n
                + token_idx * stride_b_l,
                mask=mask_n & active,
                other=0.0,
            ).to(tl.float32)
            c_t = tl.load(
                c_ptr
                + batch_idx * stride_c_b
                + channel_idx * stride_c_d
                + offs_n * stride_c_n
                + token_idx * stride_c_l,
                mask=mask_n & active,
                other=0.0,
            ).to(tl.float32)

            a_bar = tl.exp(delta_t * a)
            updated_state = a_bar * state + delta_t * b_t * u_t
            state = tl.where(active, updated_state, state)

            y_t = tl.sum(state * c_t, axis=0)
            if USE_D:
                y_t += d_skip * u_t
            if USE_Z:
                z_t = tl.load(
                    z_ptr + batch_idx * stride_u_b + channel_idx * stride_u_d + token_idx * stride_u_l,
                    mask=active,
                    other=0.0,
                ).to(tl.float32)
                y_t *= z_t / (1.0 + tl.exp(-z_t))
            tl.store(
                y_ptr + batch_idx * stride_y_b + channel_idx * stride_y_d + token_idx * stride_y_l,
                y_t,
                mask=active,
            )

        tl.store(
            state_out_ptr
            + batch_idx * stride_state_b
            + channel_idx * stride_state_d
            + offs_n * stride_state_n,
            state,
            mask=mask_n,
        )


def fused_triton_shape_support(
    u: Tensor,
    delta: Tensor,
    A: Tensor,
    B: Tensor,
    C: Tensor,
    D: Optional[Tensor] = None,
    z: Optional[Tensor] = None,
) -> KernelSupportInfo:
    """Return whether inputs fit the current fused Triton kernel boundary.

    This check is intentionally device-agnostic. It validates tensor layout,
    ranks, dtypes, and shape compatibility so it can be unit-tested even on
    CPU-only machines.
    """

    if u.ndim != 3 or delta.shape != u.shape:
        return KernelSupportInfo(False, "invalid", "u and delta must have shape (B, D, L).")
    if A.ndim != 2:
        return KernelSupportInfo(False, "invalid", "A must have shape (D, N).")

    batch, channels, length = u.shape
    state = A.shape[1]
    if A.shape[0] != channels:
        return KernelSupportInfo(False, "invalid", "A.shape[0] must match channel dimension D.")

    if B.ndim != C.ndim:
        return KernelSupportInfo(False, "invalid", "B and C must have the same rank.")

    if B.ndim == 3:
        if B.shape != (batch, state, length) or C.shape != (batch, state, length):
            return KernelSupportInfo(
                False,
                "shared-bc",
                "Rank-3 B and C must have shape (B, N, L).",
            )
        layout = "shared-bc"
    elif B.ndim == 4:
        if B.shape != (batch, channels, state, length) or C.shape != (batch, channels, state, length):
            return KernelSupportInfo(
                False,
                "channel-bc",
                "Rank-4 B and C must have shape (B, D, N, L).",
            )
        layout = "channel-bc"
    else:
        return KernelSupportInfo(
            False,
            "invalid",
            "Current fused Triton path supports only rank-3 or rank-4 B/C tensors.",
        )

    if D is not None and D.shape != (channels,):
        return KernelSupportInfo(False, layout, "D must have shape (D,).")
    if z is not None and z.shape != u.shape:
        return KernelSupportInfo(False, layout, "z must match u shape (B, D, L).")
    if state > 128:
        return KernelSupportInfo(False, layout, "Current Triton kernel supports state size up to 128.")
    if u.dtype not in (torch.float16, torch.float32, torch.bfloat16):
        return KernelSupportInfo(False, layout, f"Unsupported dtype for Triton path: {u.dtype}.")

    return KernelSupportInfo(True, layout, f"Supported fused Triton layout: {layout}.")


def _supported_triton_path(
    u: Tensor,
    delta: Tensor,
    A: Tensor,
    B: Tensor,
    C: Tensor,
    D: Optional[Tensor],
    z: Optional[Tensor],
) -> tuple[bool, str, str]:
    shape_support = fused_triton_shape_support(u, delta, A, B, C, D=D, z=z)
    if not shape_support.supported:
        return False, shape_support.reason, shape_support.layout
    if not TRITON_AVAILABLE:
        return False, "Triton is not installed.", shape_support.layout
    if not u.is_cuda:
        return False, "CUDA tensor required for Triton execution.", shape_support.layout
    if delta.device != u.device or A.device != u.device or B.device != u.device or C.device != u.device:
        return False, "All tensors must live on the same CUDA device.", shape_support.layout
    return True, shape_support.reason, shape_support.layout


def _launch_triton_fused(
    u: Tensor,
    delta: Tensor,
    A: Tensor,
    B: Tensor,
    C: Tensor,
    D: Optional[Tensor],
    z: Optional[Tensor],
    chunk_size: int,
    layout: str,
) -> Tensor:
    batch, channels, length = u.shape
    state_size = A.shape[1]

    u = u.contiguous()
    delta = delta.contiguous()
    A = A.contiguous()
    B = B.contiguous()
    C = C.contiguous()
    D_tensor = D.contiguous() if D is not None else torch.empty(1, device=u.device, dtype=u.dtype)
    z_tensor = z.contiguous() if z is not None else torch.empty(1, device=u.device, dtype=u.dtype)

    state = torch.zeros(batch, channels, state_size, device=u.device, dtype=torch.float32)
    outputs: list[Tensor] = []

    for start in range(0, length, chunk_size):
        end = min(length, start + chunk_size)
        this_length = end - start
        block_l = _next_power_of_two(this_length)
        block_n = _next_power_of_two(state_size)

        u_chunk = u[:, :, start:end]
        delta_chunk = delta[:, :, start:end]
        B_chunk = B[..., start:end]
        C_chunk = C[..., start:end]
        z_chunk = z_tensor[:, :, start:end] if z is not None else z_tensor

        y_chunk = torch.empty(batch, channels, this_length, device=u.device, dtype=u.dtype)
        next_state = torch.empty_like(state)

        stride_b_d = B_chunk.stride(1) if layout == "channel-bc" else 0
        stride_c_d = C_chunk.stride(1) if layout == "channel-bc" else 0
        stride_b_n = B_chunk.stride(2) if layout == "channel-bc" else B_chunk.stride(1)
        stride_c_n = C_chunk.stride(2) if layout == "channel-bc" else C_chunk.stride(1)
        stride_b_l = B_chunk.stride(3) if layout == "channel-bc" else B_chunk.stride(2)
        stride_c_l = C_chunk.stride(3) if layout == "channel-bc" else C_chunk.stride(2)

        _selective_scan_chunk_kernel[(batch * channels,)](
            u_chunk,
            delta_chunk,
            A,
            B_chunk,
            C_chunk,
            D_tensor,
            z_chunk,
            state,
            next_state,
            y_chunk,
            u_chunk.stride(0),
            u_chunk.stride(1),
            u_chunk.stride(2),
            delta_chunk.stride(0),
            delta_chunk.stride(1),
            delta_chunk.stride(2),
            A.stride(0),
            A.stride(1),
            B_chunk.stride(0),
            stride_b_d,
            stride_b_n,
            stride_b_l,
            C_chunk.stride(0),
            stride_c_d,
            stride_c_n,
            stride_c_l,
            state.stride(0),
            state.stride(1),
            state.stride(2),
            y_chunk.stride(0),
            y_chunk.stride(1),
            y_chunk.stride(2),
            channels,
            state_size,
            this_length,
            USE_D=D is not None,
            USE_Z=z is not None,
            BLOCK_N=block_n,
            BLOCK_L=block_l,
        )
        outputs.append(y_chunk)
        state = next_state

    return torch.cat(outputs, dim=-1)


class _FusedScanAutograd(torch.autograd.Function):
    """Autograd wrapper: fused Triton forward, reference backward."""

    @staticmethod
    def forward(
        ctx,
        u: Tensor,
        delta: Tensor,
        A: Tensor,
        B: Tensor,
        C: Tensor,
        D_skip: Optional[Tensor],
        z: Optional[Tensor],
        chunk_size: int,
        layout: str,
    ) -> Tensor:
        ctx.save_for_backward(u, delta, A, B, C, D_skip, z)
        return _launch_triton_fused(u, delta, A, B, C, D_skip, z, chunk_size=chunk_size, layout=layout)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        u, delta, A, B, C, D_skip, z = ctx.saved_tensors
        # Recompute with autograd-enabled reference path to get gradients
        with torch.enable_grad():
            u_ = u.detach().requires_grad_(True)
            delta_ = delta.detach().requires_grad_(True)
            A_ = A.detach().requires_grad_(True)
            B_ = B.detach().requires_grad_(True)
            C_ = C.detach().requires_grad_(True)
            D_ = D_skip.detach().requires_grad_(True) if D_skip is not None else None
            z_ = z.detach().requires_grad_(True) if z is not None else None
            y = selective_scan_ref(u=u_, delta=delta_, A=A_, B=B_, C=C_, D=D_, z=z_)
            y.backward(grad_output)
        return (
            u_.grad,
            delta_.grad,
            A_.grad,
            B_.grad,
            C_.grad,
            D_.grad if D_ is not None else None,
            z_.grad if z_ is not None else None,
            None,  # chunk_size
            None,  # layout
        )


def selective_scan_fused(
    u: Tensor,
    delta: Tensor,
    A: Tensor,
    B: Tensor,
    C: Tensor,
    D: Optional[Tensor] = None,
    z: Optional[Tensor] = None,
    return_metadata: bool = False,
    chunk_size: int = 128,
) -> Tensor | tuple[Tensor, KernelMetadata]:
    """Selective scan entry point with Triton fused-kernel support.

    Supported Triton layouts:
    - shared B/C: ``B`` and ``C`` shape ``(B, N, L)``
    - channel-specific B/C: ``B`` and ``C`` shape ``(B, D, N, L)``

    Common requirements:
    - ``u`` and ``delta`` shape ``(B, D, L)``
    - ``A`` shape ``(D, N)``
    - optional ``D`` skip tensor with shape ``(D,)``
    - optional gate ``z`` with shape ``(B, D, L)``

    Unsupported shapes or CPU-only environments fall back to the trusted
    PyTorch reference path.
    """

    supported, note, layout = _supported_triton_path(u, delta, A, B, C, D, z)
    if supported:
        needs_grad = any(
            t is not None and t.requires_grad for t in (u, delta, A, B, C, D, z)
        )
        if needs_grad:
            output = _FusedScanAutograd.apply(u, delta, A, B, C, D, z, chunk_size, layout)
        else:
            output = _launch_triton_fused(u, delta, A, B, C, D, z, chunk_size=chunk_size, layout=layout)
        metadata = KernelMetadata(
            backend="triton-fused",
            used_fallback=False,
            notes=f"Executed fused Triton chunk kernel ({layout}).",
        )
    else:
        output = selective_scan_ref(u=u, delta=delta, A=A, B=B, C=C, D=D, z=z)
        metadata = KernelMetadata(
            backend="torch-reference",
            used_fallback=True,
            notes=note,
        )
    if return_metadata:
        return output, metadata
    return output
