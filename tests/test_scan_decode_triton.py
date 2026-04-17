"""Parity tests for the Triton decode kernel (seqlen = 1).

Bit-exact at fp32 within roundoff; tight tolerances at fp16 / bf16. We
reference against the naive PyTorch path applied to a single-timestep
tensor — which is the authoritative slow-but-correct oracle (validated
against ``mamba_ssm``).

The kernel is CUDA-only; this whole module is skipped on CPU.
"""

from __future__ import annotations

import pytest
import torch

triton = pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("Triton decode kernel requires CUDA", allow_module_level=True)

from kernels.scan_decode import selective_scan_decode_triton  # noqa: E402
from mamba_minimal.scan_naive import selective_scan_naive  # noqa: E402


def _reference_step(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    ssm_state: torch.Tensor,
    D_skip: torch.Tensor | None,
    z: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure-PyTorch reference for one decode step.

    Mirrors the math the Triton kernel implements. Accumulates in fp32,
    casts ``y`` back to ``x``'s dtype at the end (to match the kernel
    wrapper's contract).
    """

    x_f = x.float()
    dt_f = dt.float()
    A_f = A.float()
    B_f = B.float()
    C_f = C.float()
    h_f = ssm_state.float()

    delta_A = torch.exp(dt_f.unsqueeze(-1) * A_f.unsqueeze(0))  # (B, D, N)
    delta_B = dt_f.unsqueeze(-1) * B_f.unsqueeze(1)  # (B, D, N)
    h_new = delta_A * h_f + delta_B * x_f.unsqueeze(-1)  # (B, D, N)
    y = (h_new * C_f.unsqueeze(1)).sum(dim=-1)  # (B, D)
    if D_skip is not None:
        y = y + D_skip.float() * x_f
    if z is not None:
        zf = z.float()
        y = y * (zf * torch.sigmoid(zf))
    return y.to(x.dtype), h_new


def _inputs(
    batch: int,
    d_inner: int,
    d_state: int,
    dtype: torch.dtype,
    with_skip: bool,
    with_gate: bool,
    seed: int = 0,
):
    torch.manual_seed(seed)
    device = "cuda"
    x = torch.randn(batch, d_inner, dtype=dtype, device=device)
    dt = torch.rand(batch, d_inner, device=device) * 0.5  # fp32 delta
    A = -torch.rand(d_inner, d_state, device=device).abs() - 0.1
    B = torch.randn(batch, d_state, dtype=dtype, device=device)
    C = torch.randn(batch, d_state, dtype=dtype, device=device)
    h = torch.randn(batch, d_inner, d_state, device=device)  # fp32 state
    D_skip = torch.randn(d_inner, device=device) if with_skip else None
    z = torch.randn(batch, d_inner, dtype=dtype, device=device) if with_gate else None
    return x, dt, A, B, C, h, D_skip, z


SHAPES = [
    (1, 64, 16),
    (2, 128, 16),
    (1, 1536, 16),  # Mamba-130m shape
    (4, 256, 8),
    (1, 96, 32),  # non-power-of-two d_inner, larger N
]


@pytest.mark.gpu
@pytest.mark.parametrize("batch,d_inner,d_state", SHAPES)
@pytest.mark.parametrize("with_gate", [False, True])
@pytest.mark.parametrize("with_skip", [False, True])
def test_decode_triton_matches_reference_fp32(
    batch: int, d_inner: int, d_state: int, with_gate: bool, with_skip: bool
) -> None:
    x, dt, A, B, C, h, D_skip, z = _inputs(
        batch, d_inner, d_state, torch.float32, with_skip, with_gate
    )
    y_tri, h_tri = selective_scan_decode_triton(
        x=x, dt=dt, A=A, B=B, C=C, ssm_state=h.clone(), D_skip=D_skip, z=z,
    )
    y_ref, h_ref = _reference_step(x, dt, A, B, C, h, D_skip, z)
    torch.testing.assert_close(y_tri, y_ref, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(h_tri, h_ref, atol=1e-5, rtol=1e-5)


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_decode_triton_matches_reference_low_precision(dtype: torch.dtype) -> None:
    batch, d_inner, d_state = 1, 1536, 16
    x, dt, A, B, C, h, D_skip, z = _inputs(
        batch, d_inner, d_state, dtype, with_skip=True, with_gate=True
    )
    y_tri, h_tri = selective_scan_decode_triton(
        x=x, dt=dt, A=A, B=B, C=C, ssm_state=h.clone(), D_skip=D_skip, z=z,
    )
    y_ref, h_ref = _reference_step(x, dt, A, B, C, h, D_skip, z)
    atol = 5e-3 if dtype == torch.float16 else 8e-3
    rtol = 5e-3 if dtype == torch.float16 else 8e-3
    torch.testing.assert_close(y_tri, y_ref, atol=atol, rtol=rtol)
    torch.testing.assert_close(h_tri, h_ref, atol=atol, rtol=rtol)


@pytest.mark.gpu
def test_decode_triton_matches_selective_scan_naive_at_l1() -> None:
    """One decode step must agree with selective_scan_naive run at L=1.

    This ties the kernel back to the naive-scan oracle (``scan_naive``,
    which is bit-exact vs ``mamba_ssm.selective_scan_ref``).
    """

    batch, d_inner, d_state = 2, 128, 16
    x, dt, A, B, C, h, D_skip, z = _inputs(
        batch, d_inner, d_state, torch.float32, with_skip=True, with_gate=True
    )
    # Start from zero state so naive (which doesn't take h0) matches.
    h_zero = torch.zeros_like(h)
    y_tri, _ = selective_scan_decode_triton(
        x=x, dt=dt, A=A, B=B, C=C, ssm_state=h_zero, D_skip=D_skip, z=z,
    )
    # naive expects (B, D, L), (B, N, L)
    u_l1 = x.unsqueeze(-1)
    dt_l1 = dt.unsqueeze(-1)
    B_l1 = B.unsqueeze(-1)
    C_l1 = C.unsqueeze(-1)
    z_l1 = z.unsqueeze(-1)
    y_naive = selective_scan_naive(
        u_l1, dt_l1, A, B_l1, C_l1, D=D_skip, z=z_l1, delta_softplus=False,
    )
    torch.testing.assert_close(y_tri, y_naive.squeeze(-1), atol=1e-5, rtol=1e-5)
