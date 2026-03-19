from __future__ import annotations

import math

import torch

from mamba_minimal.discretization import inverse_softplus, zoh_discretize, zoh_discretize_diag


def test_inverse_softplus_round_trip() -> None:
    values = torch.tensor([0.01, 0.1, 1.0, 5.0], dtype=torch.float64)
    recovered = torch.nn.functional.softplus(inverse_softplus(values))
    assert torch.allclose(values, recovered, atol=1e-10, rtol=1e-10)


def test_zoh_discretize_matches_scalar_closed_form() -> None:
    A = torch.tensor([[-1.0]], dtype=torch.float64)
    B = torch.tensor([[1.0]], dtype=torch.float64)
    delta = torch.tensor(0.5, dtype=torch.float64)

    A_bar, B_bar = zoh_discretize(A, B, delta)

    expected_A = math.exp(-0.5)
    expected_B = 1.0 - math.exp(-0.5)
    assert torch.allclose(A_bar.squeeze(), torch.tensor(expected_A, dtype=torch.float64))
    assert torch.allclose(B_bar.squeeze(), torch.tensor(expected_B, dtype=torch.float64))


def test_zoh_discretize_diag_shape_and_values() -> None:
    A = -torch.tensor([[1.0, 2.0], [0.5, 1.5]], dtype=torch.float64)
    delta = torch.tensor([[0.1, 0.2]], dtype=torch.float64)
    discrete = zoh_discretize_diag(A, delta)

    expected = torch.exp(torch.tensor([[[ -0.1, -0.2], [-0.1, -0.3]]], dtype=torch.float64))
    assert discrete.shape == (1, 2, 2)
    assert torch.allclose(discrete, expected)
