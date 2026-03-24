from __future__ import annotations

import pytest
import torch

from kernels.scan_fused import fused_triton_shape_support, selective_scan_fused
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


def test_shape_support_accepts_shared_and_channel_specific_bc_layouts() -> None:
    batch, channels, state, length = 2, 4, 3, 8
    u = torch.randn(batch, channels, length)
    delta = torch.rand(batch, channels, length)
    A = -torch.rand(channels, state)

    shared = fused_triton_shape_support(
        u,
        delta,
        A,
        torch.randn(batch, state, length),
        torch.randn(batch, state, length),
    )
    channel = fused_triton_shape_support(
        u,
        delta,
        A,
        torch.randn(batch, channels, state, length),
        torch.randn(batch, channels, state, length),
    )

    assert shared.supported is True
    assert shared.layout == "shared-bc"
    assert channel.supported is True
    assert channel.layout == "channel-bc"


def test_fused_kernel_reports_fallback_metadata_on_cpu() -> None:
    inputs = make_inputs("cpu")
    _, metadata = selective_scan_fused(*inputs[:5], D=inputs[5], z=inputs[6], return_metadata=True)
    assert metadata.used_fallback is True
    assert metadata.backend == "torch-reference"
    assert metadata.notes
    assert ("CUDA" in metadata.notes) or ("Triton" in metadata.notes)


def test_fused_kernel_falls_back_for_channel_specific_bc() -> None:
    batch, channels, state, length = 2, 4, 3, 8
    u = torch.randn(batch, channels, length)
    delta = torch.rand(batch, channels, length)
    A = -torch.rand(channels, state)
    B = torch.randn(batch, channels, state, length)
    C = torch.randn(batch, channels, state, length)

    reference = selective_scan_ref(u, delta, A, B, C)
    fused, metadata = selective_scan_fused(u, delta, A, B, C, return_metadata=True)
    assert torch.allclose(reference, fused)
    assert metadata.used_fallback is True
    assert metadata.backend == "torch-reference"


def test_shape_support_rejects_mismatched_bc_ranks() -> None:
    batch, channels, state, length = 2, 4, 3, 8
    u = torch.randn(batch, channels, length)
    delta = torch.rand(batch, channels, length)
    A = -torch.rand(channels, state)
    B_3d = torch.randn(batch, state, length)
    C_4d = torch.randn(batch, channels, state, length)

    result = fused_triton_shape_support(u, delta, A, B_3d, C_4d)
    assert result.supported is False


def test_shape_support_rejects_large_state_size() -> None:
    batch, channels, length = 1, 2, 4
    state = 256  # exceeds 128 limit
    u = torch.randn(batch, channels, length)
    delta = torch.rand(batch, channels, length)
    A = -torch.rand(channels, state)
    B = torch.randn(batch, state, length)
    C = torch.randn(batch, state, length)

    result = fused_triton_shape_support(u, delta, A, B, C)
    assert result.supported is False
    assert "128" in result.reason


def test_shape_support_rejects_wrong_d_shape() -> None:
    batch, channels, state, length = 2, 4, 3, 8
    u = torch.randn(batch, channels, length)
    delta = torch.rand(batch, channels, length)
    A = -torch.rand(channels, state)
    B = torch.randn(batch, state, length)
    C = torch.randn(batch, state, length)
    D_wrong = torch.randn(channels + 1)

    result = fused_triton_shape_support(u, delta, A, B, C, D=D_wrong)
    assert result.supported is False


def test_fused_scan_matches_ref_without_d_or_z() -> None:
    torch.manual_seed(42)
    batch, channels, state, length = 2, 4, 3, 8
    u = torch.randn(batch, channels, length)
    delta = torch.rand(batch, channels, length)
    A = -torch.rand(channels, state)
    B = torch.randn(batch, state, length)
    C = torch.randn(batch, state, length)

    ref = selective_scan_ref(u, delta, A, B, C)
    fused = selective_scan_fused(u, delta, A, B, C)
    assert torch.allclose(ref, fused)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for GPU parity checks")
def test_kernel_wrappers_match_on_cuda() -> None:
    inputs = make_inputs("cuda")
    reference = selective_scan_ref(*inputs[:5], D=inputs[5], z=inputs[6])
    naive = selective_scan_naive(*inputs[:5], D=inputs[5], z=inputs[6])
    fused = selective_scan_fused(*inputs[:5], D=inputs[5], z=inputs[6])
    assert torch.allclose(reference, naive, atol=1e-5, rtol=1e-5)
    assert torch.allclose(reference, fused, atol=1e-5, rtol=1e-5)
