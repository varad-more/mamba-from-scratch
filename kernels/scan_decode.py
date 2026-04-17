"""Triton decode kernel: one selective-scan step (seqlen = 1).

Mamba's autoregressive decode reduces to a trivial recurrence per
(batch, d_inner) element:

    h_new = exp(dt * A) * h_old + (dt * B) * x
    y     = sum_n C[n] * h_new[n]
    y     = y + D * x        (optional)
    y     = y * silu(z)      (optional)

Arithmetic intensity is tiny — ~16 multiplies + a reduction per output
scalar, against ~4 * state_size fp-loads. This kernel is **memory-bound
by construction**, which is the point: we care about MBU, not FLOPs.

Launch shape:
  grid = (d_inner, batch)
  one program handles one output scalar + its state_size-wide state row.

Dtype contract (mirrors :func:`mamba_minimal.scan_naive.selective_scan_naive`):
  * inputs may be fp16 / bf16 / fp32
  * all accumulation runs in fp32 (loaded values are cast on load)
  * state buffer is fp32 throughout (HF convention — matches our step())
  * output ``y`` is written in fp32 and cast to ``x``'s dtype in the wrapper
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

try:
    import triton  # type: ignore
    import triton.language as tl  # type: ignore

    TRITON_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    triton = None
    tl = None
    TRITON_AVAILABLE = False


def _next_power_of_two(value: int) -> int:
    if value <= 1:
        return 1
    return 1 << (value - 1).bit_length()


if TRITON_AVAILABLE:

    @triton.jit
    def _selective_scan_decode_kernel(
        x_ptr, dt_ptr, A_ptr, B_ptr, C_ptr,
        D_ptr, z_ptr,
        state_in_ptr, state_out_ptr,
        y_ptr,
        # x: (B, D)
        stride_x_b, stride_x_d,
        # dt: (B, D)
        stride_dt_b, stride_dt_d,
        # A: (D, N)
        stride_A_d, stride_A_n,
        # B: (B, N)
        stride_B_b, stride_B_n,
        # C: (B, N)
        stride_C_b, stride_C_n,
        # D: (D,)
        stride_D_d,
        # z: (B, D)
        stride_z_b, stride_z_d,
        # state_in/out: (B, D, N)
        stride_state_b, stride_state_d, stride_state_n,
        # y: (B, D)
        stride_y_b, stride_y_d,
        STATE_SIZE: tl.constexpr,
        BLOCK_N: tl.constexpr,
        USE_D: tl.constexpr,
        USE_Z: tl.constexpr,
    ):
        pid_d = tl.program_id(0)  # d_inner index
        pid_b = tl.program_id(1)  # batch index

        n_off = tl.arange(0, BLOCK_N)
        n_mask = n_off < STATE_SIZE

        # Scalar loads (batch, d_inner).
        x_scalar = tl.load(x_ptr + pid_b * stride_x_b + pid_d * stride_x_d).to(tl.float32)
        dt_scalar = tl.load(dt_ptr + pid_b * stride_dt_b + pid_d * stride_dt_d).to(tl.float32)

        # Vector loads over state dim.
        A_row = tl.load(
            A_ptr + pid_d * stride_A_d + n_off * stride_A_n,
            mask=n_mask, other=0.0,
        ).to(tl.float32)
        B_vec = tl.load(
            B_ptr + pid_b * stride_B_b + n_off * stride_B_n,
            mask=n_mask, other=0.0,
        ).to(tl.float32)
        C_vec = tl.load(
            C_ptr + pid_b * stride_C_b + n_off * stride_C_n,
            mask=n_mask, other=0.0,
        ).to(tl.float32)
        h_old = tl.load(
            state_in_ptr + pid_b * stride_state_b + pid_d * stride_state_d + n_off * stride_state_n,
            mask=n_mask, other=0.0,
        ).to(tl.float32)

        # Discretize + recur.
        delta_A = tl.exp(dt_scalar * A_row)
        delta_B = dt_scalar * B_vec
        h_new = delta_A * h_old + delta_B * x_scalar

        # Write new state (fp32).
        tl.store(
            state_out_ptr + pid_b * stride_state_b + pid_d * stride_state_d + n_off * stride_state_n,
            h_new, mask=n_mask,
        )

        # Output projection: y = sum_n C[n] * h_new[n].
        y = tl.sum(tl.where(n_mask, C_vec * h_new, 0.0), axis=0)

        if USE_D:
            D_val = tl.load(D_ptr + pid_d * stride_D_d).to(tl.float32)
            y = y + D_val * x_scalar
        if USE_Z:
            z_val = tl.load(z_ptr + pid_b * stride_z_b + pid_d * stride_z_d).to(tl.float32)
            silu_z = z_val * (1.0 / (1.0 + tl.exp(-z_val)))
            y = y * silu_z

        tl.store(y_ptr + pid_b * stride_y_b + pid_d * stride_y_d, y)


def selective_scan_decode_triton(
    x: Tensor,               # (B, D) fp16/bf16/fp32
    dt: Tensor,              # (B, D) fp32 (post-softplus)
    A: Tensor,               # (D, N) fp32, negative real
    B: Tensor,                # (B, N) fp16/bf16/fp32 — selective B this step
    C: Tensor,                # (B, N) fp16/bf16/fp32 — selective C this step
    ssm_state: Tensor,       # (B, D, N) fp32 — prior state
    D_skip: Optional[Tensor] = None,  # (D,) fp32
    z: Optional[Tensor] = None,        # (B, D) fp16/bf16/fp32
) -> tuple[Tensor, Tensor]:
    """One selective-scan step via Triton.

    Returns ``(y, new_ssm_state)``:
        * ``y`` has shape ``(B, D)`` in ``x``'s dtype.
        * ``new_ssm_state`` has shape ``(B, D, N)`` in fp32.

    The kernel writes state to a freshly allocated buffer — this is an
    allocation per step, which is fine for the tok/s regime we care about
    and keeps the API pure-functional. Callers holding a cache can simply
    rebind to the returned tensor.
    """

    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available in this environment.")
    if not x.is_cuda:
        raise RuntimeError("selective_scan_decode_triton requires CUDA tensors.")

    if x.ndim != 2:
        raise ValueError(f"x must be (B, D), got {tuple(x.shape)}")
    batch, d_inner = x.shape
    d_state = A.shape[1]
    if A.shape[0] != d_inner:
        raise ValueError(f"A first dim must match d_inner {d_inner}, got {A.shape[0]}")
    if B.shape != (batch, d_state):
        raise ValueError(f"B must be (B, N)={(batch, d_state)}, got {tuple(B.shape)}")
    if C.shape != (batch, d_state):
        raise ValueError(f"C must be (B, N)={(batch, d_state)}, got {tuple(C.shape)}")
    if ssm_state.shape != (batch, d_inner, d_state):
        raise ValueError(
            f"ssm_state must be (B, D, N)={(batch, d_inner, d_state)}, "
            f"got {tuple(ssm_state.shape)}"
        )
    if dt.shape != x.shape:
        raise ValueError(f"dt must match x shape {x.shape}, got {tuple(dt.shape)}")

    # Enforce fp32 A and state, matching the kernel and the naive ref.
    if A.dtype != torch.float32:
        A = A.float()
    if ssm_state.dtype != torch.float32:
        ssm_state = ssm_state.float()
    if dt.dtype != torch.float32:
        dt = dt.float()

    new_state = torch.empty_like(ssm_state)
    y = torch.empty(batch, d_inner, device=x.device, dtype=torch.float32)

    use_D = 1 if D_skip is not None else 0
    use_Z = 1 if z is not None else 0
    if D_skip is None:
        D_skip_ptr = ssm_state  # placeholder; kernel ignores when USE_D=0
        stride_D_d = 0
    else:
        if D_skip.dtype != torch.float32:
            D_skip = D_skip.float()
        D_skip_ptr = D_skip
        stride_D_d = D_skip.stride(0)
    if z is None:
        z_ptr = ssm_state
        stride_z_b = 0
        stride_z_d = 0
    else:
        z_ptr = z
        stride_z_b = z.stride(0)
        stride_z_d = z.stride(1)

    BLOCK_N = _next_power_of_two(d_state)
    grid = (d_inner, batch)
    _selective_scan_decode_kernel[grid](
        x, dt, A, B, C,
        D_skip_ptr, z_ptr,
        ssm_state, new_state,
        y,
        x.stride(0), x.stride(1),
        dt.stride(0), dt.stride(1),
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        stride_D_d,
        stride_z_b, stride_z_d,
        ssm_state.stride(0), ssm_state.stride(1), ssm_state.stride(2),
        y.stride(0), y.stride(1),
        STATE_SIZE=d_state,
        BLOCK_N=BLOCK_N,
        USE_D=use_D,
        USE_Z=use_Z,
    )

    return y.to(x.dtype), new_state
