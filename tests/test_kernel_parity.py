from __future__ import annotations

import pytest
import torch

from kernels.scan_fused import selective_scan_fused
from kernels.scan_naive import selective_scan_naive
from mamba_minimal.selective_scan import selective_scan_ref


def make_inputs(device: str = "cpu"):
    torch.manual_seed(0)
    batch, channels, state, length = 2, 4, 3, 8
    u = torch.randn(batch, channels, length, device=device)
    delta = torch.rand(batch, channels, length, device=device)
    A = -torch.rand(channels, state, device=device)
    B = torch.randn(batch, state, length, device=device)
    C = torch.randn(batch, state, length, device=device)
    D = torch.randn(channels, device=device)
    z = torch.randn(batch, channels, length, device=device)
    return u, delta, A, B, C, D, z


def test_reference_and_kernel_wrappers_match_on_cpu() -> None:
    inputs = make_inputs("cpu")
    reference = selective_scan_ref(*inputs[:5], D=inputs[5], z=inputs[6])
    naive = selective_scan_naive(*inputs[:5], D=inputs[5], z=inputs[6])
    fused = selective_scan_fused(*inputs[:5], D=inputs[5], z=inputs[6])
    assert torch.allclose(reference, naive)
    assert torch.allclose(reference, fused)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for GPU parity checks")
def test_kernel_wrappers_match_on_cuda() -> None:
    inputs = make_inputs("cuda")
    reference = selective_scan_ref(*inputs[:5], D=inputs[5], z=inputs[6])
    naive = selective_scan_naive(*inputs[:5], D=inputs[5], z=inputs[6])
    fused = selective_scan_fused(*inputs[:5], D=inputs[5], z=inputs[6])
    assert torch.allclose(reference, naive, atol=1e-5, rtol=1e-5)
    assert torch.allclose(reference, fused, atol=1e-5, rtol=1e-5)
