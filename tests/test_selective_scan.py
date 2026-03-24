from __future__ import annotations

import torch

from mamba_minimal.model import MambaBlock
from mamba_minimal.selective_scan import selective_scan_ref


def test_selective_scan_matches_manual_single_state_case() -> None:
    u = torch.tensor([[[1.0, 2.0, 3.0]]], dtype=torch.float64)
    delta = torch.tensor([[[0.5, 0.5, 0.5]]], dtype=torch.float64)
    A = torch.tensor([[-1.0]], dtype=torch.float64)
    B = torch.tensor([[[2.0, 2.0, 2.0]]], dtype=torch.float64)
    C = torch.tensor([[[1.0, 1.0, 1.0]]], dtype=torch.float64)
    D = torch.tensor([0.25], dtype=torch.float64)

    y, last_state = selective_scan_ref(
        u=u,
        delta=delta,
        A=A,
        B=B,
        C=C,
        D=D,
        delta_softplus=False,
        return_last_state=True,
    )

    x = 0.0
    manual = []
    decay = torch.exp(torch.tensor(-0.5, dtype=torch.float64)).item()
    for u_t in [1.0, 2.0, 3.0]:
        x = decay * x + 0.5 * 2.0 * u_t
        manual.append(x + 0.25 * u_t)

    expected = torch.tensor(manual, dtype=torch.float64).view(1, 1, 3)
    assert torch.allclose(y, expected, atol=1e-10, rtol=1e-10)
    assert torch.allclose(last_state, torch.tensor([[[x]]], dtype=torch.float64))


def test_selective_scan_supports_channel_specific_bc() -> None:
    batch, channels, state, length = 2, 3, 4, 5
    u = torch.randn(batch, channels, length)
    delta = torch.rand(batch, channels, length)
    A = -torch.rand(channels, state)
    B = torch.randn(batch, channels, state, length)
    C = torch.randn(batch, channels, state, length)

    y = selective_scan_ref(u=u, delta=delta, A=A, B=B, C=C)
    assert y.shape == (batch, channels, length)


def test_selective_scan_with_delta_bias() -> None:
    batch, channels, state, length = 1, 2, 3, 4
    u = torch.randn(batch, channels, length, dtype=torch.float64)
    delta = torch.randn(batch, channels, length, dtype=torch.float64)
    A = -torch.rand(channels, state, dtype=torch.float64)
    B = torch.randn(batch, state, length, dtype=torch.float64)
    C = torch.randn(batch, state, length, dtype=torch.float64)
    delta_bias = torch.randn(channels, dtype=torch.float64)

    y_biased = selective_scan_ref(u=u, delta=delta, A=A, B=B, C=C, delta_bias=delta_bias)
    y_manual = selective_scan_ref(
        u=u,
        delta=delta + delta_bias.view(1, -1, 1),
        A=A,
        B=B,
        C=C,
        delta_bias=None,
    )
    assert y_biased.shape == (batch, channels, length)
    assert torch.allclose(y_biased, y_manual, atol=1e-10)


def test_selective_scan_z_gating() -> None:
    batch, channels, state, length = 1, 2, 3, 4
    u = torch.randn(batch, channels, length, dtype=torch.float64)
    delta = torch.rand(batch, channels, length, dtype=torch.float64)
    A = -torch.rand(channels, state, dtype=torch.float64)
    B = torch.randn(batch, state, length, dtype=torch.float64)
    C = torch.randn(batch, state, length, dtype=torch.float64)
    z = torch.randn(batch, channels, length, dtype=torch.float64)

    y_gated = selective_scan_ref(u=u, delta=delta, A=A, B=B, C=C, z=z)
    y_ungated = selective_scan_ref(u=u, delta=delta, A=A, B=B, C=C, z=None)

    # The gated output should differ from ungated
    assert not torch.allclose(y_gated, y_ungated, atol=1e-6)
    # Manually check gating: y_gated = y_ungated * silu(z)
    expected = y_ungated * torch.nn.functional.silu(z)
    assert torch.allclose(y_gated, expected, atol=1e-10)


def test_selective_scan_return_last_state_channel_specific_bc() -> None:
    batch, channels, state, length = 2, 3, 4, 5
    u = torch.randn(batch, channels, length, dtype=torch.float64)
    delta = torch.rand(batch, channels, length, dtype=torch.float64)
    A = -torch.rand(channels, state, dtype=torch.float64)
    B = torch.randn(batch, channels, state, length, dtype=torch.float64)
    C = torch.randn(batch, channels, state, length, dtype=torch.float64)

    y, last_state = selective_scan_ref(
        u=u, delta=delta, A=A, B=B, C=C, delta_softplus=False, return_last_state=True,
    )
    assert y.shape == (batch, channels, length)
    assert last_state.shape == (batch, channels, state)


def test_selective_scan_length_one() -> None:
    batch, channels, state = 1, 2, 3
    u = torch.randn(batch, channels, 1, dtype=torch.float64)
    delta = torch.rand(batch, channels, 1, dtype=torch.float64)
    A = -torch.rand(channels, state, dtype=torch.float64)
    B = torch.randn(batch, state, 1, dtype=torch.float64)
    C = torch.randn(batch, state, 1, dtype=torch.float64)

    y = selective_scan_ref(u=u, delta=delta, A=A, B=B, C=C, delta_softplus=False)
    assert y.shape == (batch, channels, 1)
    assert torch.isfinite(y).all()


def test_mamba_block_forward_shape_and_gradients() -> None:
    block = MambaBlock(d_model=8, d_state=4, d_conv=3, expand=2)
    x = torch.randn(2, 6, 8, requires_grad=True)
    y = block(x)
    assert y.shape == (2, 6, 8)

    loss = y.sum()
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
