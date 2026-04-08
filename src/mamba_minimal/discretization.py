from __future__ import annotations

import torch
from torch import Tensor


def _validate_matrix_shapes(A: Tensor, B: Tensor) -> tuple[int, int]:
    if A.ndim < 2 or A.shape[-1] != A.shape[-2]:
        raise ValueError(f"A must be a square matrix batch, got shape {tuple(A.shape)}")
    if B.ndim < 2 or B.shape[-2] != A.shape[-1]:
        raise ValueError(
            "B must align with A on the state dimension; "
            f"got A shape {tuple(A.shape)} and B shape {tuple(B.shape)}"
        )
    return A.shape[-1], B.shape[-1]


def inverse_softplus(x: Tensor) -> Tensor:
    """Compute the inverse softplus in a numerically stable way."""

    return x + torch.log(-torch.expm1(-x))


def zoh_discretize(A: Tensor, B: Tensor, delta: Tensor) -> tuple[Tensor, Tensor]:
    """Discretize a continuous-time linear system with zero-order hold.

    The continuous system is:
        x'(t) = A x(t) + B u(t)

    This function uses the block-matrix exponential identity instead of the
    unstable explicit inverse formula:

        exp(dt * [[A, B], [0, 0]]) = [[A_bar, B_bar], [0, I]]

    Parameters
    ----------
    A:
        State transition matrix with shape ``(..., N, N)``.
    B:
        Input matrix with shape ``(..., N, M)``.
    delta:
        Time-step with shape broadcastable to ``A.shape[:-2]``.

    Returns
    -------
    (A_bar, B_bar):
        Discrete transition and input matrices.
    """

    n, m = _validate_matrix_shapes(A, B)
    leading = torch.broadcast_shapes(A.shape[:-2], B.shape[:-2], delta.shape)

    A_expanded = torch.broadcast_to(A, (*leading, n, n))
    B_expanded = torch.broadcast_to(B, (*leading, n, m))
    delta_expanded = torch.broadcast_to(delta, leading)

    block = torch.zeros(*leading, n + m, n + m, dtype=A.dtype, device=A.device)
    block[..., :n, :n] = A_expanded
    block[..., :n, n:] = B_expanded

    discrete = torch.matrix_exp(delta_expanded[..., None, None] * block)
    A_bar = discrete[..., :n, :n]
    B_bar = discrete[..., :n, n:]
    return A_bar, B_bar


def zoh_discretize_diag(A: Tensor, delta: Tensor) -> Tensor:
    """Discretize a diagonal state matrix used in Mamba-style SSMs.

    Parameters
    ----------
    A:
        Diagonal continuous-time transition terms with shape ``(D, N)``.
    delta:
        Positive step sizes with shape ``(..., D)``.

    Returns
    -------
    Tensor
        Discretized diagonal factors with shape ``(..., D, N)``.
    """

    if A.ndim != 2:
        raise ValueError(f"A must have shape (D, N), got {tuple(A.shape)}")
    if delta.shape[-1] != A.shape[0]:
        raise ValueError(
            f"delta last dim must match A.shape[0]={A.shape[0]}, got {tuple(delta.shape)}"
        )
    return torch.exp(torch.einsum("...d,dn->...dn", delta, A))
