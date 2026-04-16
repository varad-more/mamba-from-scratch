"""Naive selective scan vs the ``mamba_ssm`` oracle.

This is the plan's week-1 correctness anchor: our naive scan must match the
authoritative reference from the ``mamba_ssm`` package. The CPU-only variant
(``selective_scan_ref`` from mamba_ssm) runs everywhere; the CUDA-fused
``selective_scan_fn`` runs only when a GPU is available and is marked
accordingly.
"""

from __future__ import annotations

import pytest
import torch

mamba_ssm = pytest.importorskip("mamba_ssm")

from mamba_ssm.ops.selective_scan_interface import (  # noqa: E402
    selective_scan_fn,
    selective_scan_ref as mamba_ssm_selective_scan_ref,
)

from mamba_minimal.scan_naive import selective_scan_naive  # noqa: E402


def _tiny_inputs(
    batch: int = 2,
    dim: int = 64,
    dstate: int = 16,
    seqlen: int = 32,
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
    variable_bc: bool = True,
    with_gate: bool = False,
    with_skip: bool = True,
):
    torch.manual_seed(0)
    u = torch.randn(batch, dim, seqlen, dtype=dtype, device=device)
    delta = torch.rand(batch, dim, seqlen, dtype=dtype, device=device) * 0.5
    A = -torch.rand(dim, dstate, dtype=torch.float32, device=device).abs() - 0.1
    if variable_bc:
        B = torch.randn(batch, dstate, seqlen, dtype=dtype, device=device)
        C = torch.randn(batch, dstate, seqlen, dtype=dtype, device=device)
    else:
        B = torch.randn(dim, dstate, dtype=dtype, device=device)
        C = torch.randn(dim, dstate, dtype=dtype, device=device)
    D = torch.randn(dim, dtype=torch.float32, device=device) if with_skip else None
    z = torch.randn(batch, dim, seqlen, dtype=dtype, device=device) if with_gate else None
    return u, delta, A, B, C, D, z


@pytest.mark.parametrize("with_gate", [False, True])
@pytest.mark.parametrize("with_skip", [True, False])
@pytest.mark.parametrize("delta_softplus", [False, True])
def test_naive_matches_mamba_ssm_ref_cpu_fp32(
    with_gate: bool, with_skip: bool, delta_softplus: bool
) -> None:
    u, delta, A, B, C, D, z = _tiny_inputs(with_gate=with_gate, with_skip=with_skip)
    ours = selective_scan_naive(
        u, delta, A, B, C, D=D, z=z, delta_softplus=delta_softplus
    )
    ref = mamba_ssm_selective_scan_ref(
        u, delta, A, B, C, D=D, z=z, delta_softplus=delta_softplus
    )
    torch.testing.assert_close(ours, ref, atol=1e-5, rtol=1e-5)


def test_naive_matches_mamba_ssm_ref_return_last_state() -> None:
    u, delta, A, B, C, D, z = _tiny_inputs()
    y_ours, h_ours = selective_scan_naive(
        u, delta, A, B, C, D=D, return_last_state=True
    )
    y_ref, h_ref = mamba_ssm_selective_scan_ref(
        u, delta, A, B, C, D=D, return_last_state=True
    )
    torch.testing.assert_close(y_ours, y_ref, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(h_ours, h_ref, atol=1e-5, rtol=1e-5)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_naive_matches_mamba_ssm_fused_cuda(dtype: torch.dtype) -> None:
    u, delta, A, B, C, D, z = _tiny_inputs(dtype=dtype, device="cuda")
    ours = selective_scan_naive(u, delta, A, B, C, D=D, z=z, delta_softplus=False)
    fused = selective_scan_fn(u, delta, A, B, C, D=D, z=z, delta_softplus=False)
    atol = 1e-5 if dtype == torch.float32 else 2e-3
    rtol = 1e-5 if dtype == torch.float32 else 2e-3
    torch.testing.assert_close(ours, fused, atol=atol, rtol=rtol)
