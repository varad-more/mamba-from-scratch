"""Parity tests for the Blelloch parallel selective scan.

The naive scan (``scan_naive.selective_scan_naive``) is the reference oracle.
This module locks the parallel scan to it within fp32 roundoff, across a
mix of shapes, gate/skip/softplus variants, and dtypes.
"""

from __future__ import annotations

import pytest
import torch

from mamba_minimal.parallel_scan import sequential_affine_scan
from mamba_minimal.scan_naive import selective_scan_naive
from mamba_minimal.scan_parallel import (
    blelloch_affine_scan,
    selective_scan_parallel,
)


@pytest.mark.parametrize("length", [1, 2, 3, 4, 5, 7, 8, 15, 16, 17, 32, 37, 64])
def test_blelloch_matches_sequential_fp64(length: int) -> None:
    """Affine scan primitive must be bit-exact in fp64, any length."""
    torch.manual_seed(0)
    a = torch.rand(2, 3, length, 4, dtype=torch.float64) * 0.8 + 0.1
    b = torch.randn(2, 3, length, 4, dtype=torch.float64)
    ref = sequential_affine_scan(a, b)
    par = blelloch_affine_scan(a, b)
    torch.testing.assert_close(ref, par, atol=1e-12, rtol=1e-12)


def _tiny_inputs(
    batch: int = 2,
    dim: int = 16,
    dstate: int = 8,
    seqlen: int = 32,
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
    with_gate: bool = False,
    with_skip: bool = True,
):
    torch.manual_seed(0)
    u = torch.randn(batch, dim, seqlen, dtype=dtype, device=device)
    delta = torch.rand(batch, dim, seqlen, dtype=dtype, device=device) * 0.5
    A = -torch.rand(dim, dstate, dtype=torch.float32, device=device).abs() - 0.1
    B = torch.randn(batch, dstate, seqlen, dtype=dtype, device=device)
    C = torch.randn(batch, dstate, seqlen, dtype=dtype, device=device)
    D = torch.randn(dim, dtype=torch.float32, device=device) if with_skip else None
    z = torch.randn(batch, dim, seqlen, dtype=dtype, device=device) if with_gate else None
    return u, delta, A, B, C, D, z


@pytest.mark.parametrize("with_gate", [False, True])
@pytest.mark.parametrize("with_skip", [False, True])
@pytest.mark.parametrize("delta_softplus", [False, True])
def test_parallel_matches_naive_cpu_fp32(
    with_gate: bool, with_skip: bool, delta_softplus: bool
) -> None:
    u, delta, A, B, C, D, z = _tiny_inputs(with_gate=with_gate, with_skip=with_skip)
    y_naive = selective_scan_naive(
        u, delta, A, B, C, D=D, z=z, delta_softplus=delta_softplus
    )
    y_par = selective_scan_parallel(
        u, delta, A, B, C, D=D, z=z, delta_softplus=delta_softplus
    )
    torch.testing.assert_close(y_naive, y_par, atol=1e-4, rtol=1e-4)


def test_parallel_return_last_state_matches_naive() -> None:
    u, delta, A, B, C, D, z = _tiny_inputs(with_gate=True, with_skip=True)
    y_naive, h_naive = selective_scan_naive(
        u, delta, A, B, C, D=D, z=z, return_last_state=True
    )
    y_par, h_par = selective_scan_parallel(
        u, delta, A, B, C, D=D, z=z, return_last_state=True
    )
    torch.testing.assert_close(y_naive, y_par, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(h_naive, h_par, atol=1e-4, rtol=1e-4)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_parallel_matches_naive_cuda(dtype: torch.dtype) -> None:
    u, delta, A, B, C, D, z = _tiny_inputs(dtype=dtype, device="cuda", with_gate=True)
    y_naive = selective_scan_naive(u, delta, A, B, C, D=D, z=z)
    y_par = selective_scan_parallel(u, delta, A, B, C, D=D, z=z)
    atol = 2e-4 if dtype == torch.float32 else 5e-3
    rtol = 2e-4 if dtype == torch.float32 else 5e-3
    torch.testing.assert_close(y_naive, y_par, atol=atol, rtol=rtol)
