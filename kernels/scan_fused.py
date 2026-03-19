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
        stride_b_n,
        stride_b_l,
        stride_c_b,
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
                b_ptr + batch_idx * stride_b_b + offs_n * stride_b_n + token_idx * stride_b_l,
                mask=mask_n & active,
                other=0.0,
            ).to(tl.float32)
            c_t = tl.load(
                c_ptr + batch_idx * stride_c_b + offs_n * stride_c_n + token_idx * stride_c_l,
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


def _supported_triton_path(
    u: Tensor,
    delta: Tensor,
    A: Tensor,
    B: Tensor,
    C: Tensor,
    D: Optional[Tensor],
    z: Optional[Tensor],
) -> tuple[bool, str]:
    if not TRITON_AVAILABLE:
        return False, "Triton is not installed."
    if not u.is_cuda:
        return False, "CUDA tensor required for Triton execution."
    if delta.device != u.device or A.device != u.device or B.device != u.device or C.device != u.device:
        return False, "All tensors must live on the same CUDA device."
    if u.ndim != 3 or delta.shape != u.shape:
        return False, "u and delta must have shape (B, D, L)."
    if A.ndim != 2:
        return False, "A must have shape (D, N)."
    if B.ndim != 3 or C.ndim != 3:
        return False, "Current Triton path supports rank-3 B/C tensors with shape (B, N, L)."
    batch, channels, length = u.shape
    state = A.shape[1]
    if A.shape[0] != channels:
        return False, "A.shape[0] must match channel dimension D."
    if B.shape != (batch, state, length) or C.shape != (batch, state, length):
        return False, "B and C must have shape (B, N, L) for the fused Triton path."
    if D is not None and D.shape != (channels,):
        return False, "D must have shape (D,)."
    if z is not None and z.shape != u.shape:
        return False, "z must match u shape."
    if state > 128:
        return False, "Current Triton kernel supports state size up to 128."
    if u.dtype not in (torch.float16, torch.float32, torch.bfloat16):
        return False, f"Unsupported dtype for Triton path: {u.dtype}."
    return True, "supported"


def _launch_triton_fused(
    u: Tensor,
    delta: Tensor,
    A: Tensor,
    B: Tensor,
    C: Tensor,
    D: Optional[Tensor],
    z: Optional[Tensor],
    chunk_size: int,
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

        y_chunk = torch.empty(batch, channels, this_length, device=u.device, dtype=u.dtype)
        next_state = torch.empty_like(state)

        _selective_scan_chunk_kernel[(batch * channels,)](
            u[:, :, start:end],
            delta[:, :, start:end],
            A,
            B[:, :, start:end],
            C[:, :, start:end],
            D_tensor,
            z_tensor[:, :, start:end] if z is not None else z_tensor,
            state,
            next_state,
            y_chunk,
            u.stride(0),
            u.stride(1),
            u.stride(2),
            delta.stride(0),
            delta.stride(1),
            delta.stride(2),
            A.stride(0),
            A.stride(1),
            B.stride(0),
            B.stride(1),
            B.stride(2),
            C.stride(0),
            C.stride(1),
            C.stride(2),
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

    Supported Triton path:
    - ``u`` and ``delta`` shape ``(B, D, L)``
    - ``A`` shape ``(D, N)``
    - ``B`` and ``C`` shape ``(B, N, L)``
    - optional ``D`` skip tensor with shape ``(D,)``
    - optional gate ``z`` with shape ``(B, D, L)``

    Unsupported shapes or CPU-only environments fall back to the trusted
    PyTorch reference path. This keeps the interface stable while enabling a
    progressively faster CUDA implementation.
    """

    supported, note = _supported_triton_path(u, delta, A, B, C, D, z)
    if supported:
        output = _launch_triton_fused(u, delta, A, B, C, D, z, chunk_size=chunk_size)
        metadata = KernelMetadata(
            backend="triton-fused",
            used_fallback=False,
            notes="Executed fused Triton chunk kernel.",
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
