from __future__ import annotations

import pytest
import torch

from mamba_minimal.backend import BackendSelectionError, select_scan_backend


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


def test_reference_policy_forces_reference_backend() -> None:
    inputs = make_inputs()
    metadata = select_scan_backend(*inputs[:5], D=inputs[5], z=inputs[6], requested_backend="reference", triton_available=False)
    assert metadata.selected_backend == "torch-reference"
    assert metadata.used_fallback is False


def test_auto_policy_falls_back_cleanly_on_cpu() -> None:
    inputs = make_inputs()
    metadata = select_scan_backend(*inputs[:5], D=inputs[5], z=inputs[6], requested_backend="auto", triton_available=False)
    assert metadata.selected_backend == "torch-reference"
    assert metadata.used_fallback is True
    assert "Triton" in metadata.notes or "CUDA" in metadata.notes


def test_fused_policy_errors_when_unavailable() -> None:
    inputs = make_inputs()
    with pytest.raises(BackendSelectionError):
        select_scan_backend(*inputs[:5], D=inputs[5], z=inputs[6], requested_backend="fused", triton_available=False)
