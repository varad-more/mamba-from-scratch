"""Work-efficient associative parallel selective scan (Blelloch pattern).

The naive scan in :mod:`mamba_minimal.scan_naive` is the correctness oracle;
it walks a Python ``for`` over the sequence dimension. This module replaces
that loop with an associative prefix scan built from torch ops only — pure
PyTorch, no Triton, no custom CUDA.

Core observation: the selective-scan recurrence

    h_t = A_bar_t ⊙ h_{t-1} + B_bar_t ⊙ u_t

is *per-channel, per-state-dim* a scalar affine map ``h -> a_t * h + b_t``.
Affine maps compose associatively, so a prefix scan applied to the sequence
of ``(a_t, b_t)`` pairs yields every ``h_t`` in ``O(log L)`` depth.

We use the Blelloch work-efficient up-sweep / down-sweep pattern:

* Up-sweep: build a binary tree of compositions; at the root sits the full
  prefix composition.
* Down-sweep: walk back down, clearing the root to identity, producing the
  *exclusive* prefix composition at every position.
* Convert exclusive -> inclusive: compose each position's exclusive prefix
  with its own ``(a_t, b_t)``.

Accumulation runs in fp32 regardless of the input dtype (matching the naive
scan contract). The result is cast back to the input dtype at return.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor


def _compose_left_then_right(
    a_l: Tensor, b_l: Tensor, a_r: Tensor, b_r: Tensor
) -> tuple[Tensor, Tensor]:
    """Compose two affine maps: apply left first, then right.

    Left maps ``x -> a_l * x + b_l``, right maps ``y -> a_r * y + b_r``.
    Composition is ``x -> a_r * (a_l * x + b_l) + b_r = (a_r * a_l) x + (a_r * b_l + b_r)``.
    """

    return a_r * a_l, a_r * b_l + b_r


def blelloch_affine_scan(a: Tensor, b: Tensor) -> Tensor:
    """Inclusive affine prefix scan over axis ``-2`` via Blelloch up/down-sweep.

    Computes ``h_t = a_t * h_{t-1} + b_t`` with ``h_{-1} = 0`` for every ``t``
    along the second-to-last axis. All other axes are treated as independent
    batch dims.

    Args:
        a: tensor of shape ``(..., L, F)`` — elementwise scalar multipliers.
        b: tensor of shape ``(..., L, F)`` — elementwise scalar biases.

    Returns:
        Tensor of shape ``(..., L, F)`` holding ``h_0, h_1, ..., h_{L-1}``.
    """

    if a.shape != b.shape:
        raise ValueError(f"a and b must share shape, got {tuple(a.shape)} vs {tuple(b.shape)}")
    if a.ndim < 2:
        raise ValueError(f"a must have at least 2 dims (..., L, F), got shape {tuple(a.shape)}")

    orig_L = a.shape[-2]
    if orig_L == 0:
        return a.new_zeros(a.shape)
    if orig_L == 1:
        return b.clone()

    # Pad L to the next power of 2 using the identity affine map (a=1, b=0).
    Lp = 1 << (orig_L - 1).bit_length()
    if Lp != orig_L:
        pad_shape = (*a.shape[:-2], Lp - orig_L, a.shape[-1])
        a_pad = a.new_ones(pad_shape)
        b_pad = b.new_zeros(pad_shape)
        A = torch.cat([a, a_pad], dim=-2)
        B = torch.cat([b, b_pad], dim=-2)
    else:
        A = a.clone()
        B = b.clone()

    # ----- Up-sweep -----
    # After stride=s, the position (step - 1) of each block of size step = 2*s
    # holds the composition of its two children.
    stride = 1
    while stride < Lp:
        step = 2 * stride
        idx_r = torch.arange(step - 1, Lp, step, device=a.device)
        idx_l = idx_r - stride
        a_l = A.index_select(-2, idx_l)
        b_l = B.index_select(-2, idx_l)
        a_r = A.index_select(-2, idx_r)
        b_r = B.index_select(-2, idx_r)
        comp_a, comp_b = _compose_left_then_right(a_l, b_l, a_r, b_r)
        A = A.index_copy(-2, idx_r, comp_a)
        B = B.index_copy(-2, idx_r, comp_b)
        stride *= 2

    # ----- Down-sweep: build exclusive prefix composition -----
    # Start by replacing the root with the identity affine map.
    root = torch.tensor([Lp - 1], device=a.device)
    A = A.index_copy(-2, root, torch.ones_like(A.index_select(-2, root)))
    B = B.index_copy(-2, root, torch.zeros_like(B.index_select(-2, root)))

    stride = Lp // 2
    while stride >= 1:
        step = 2 * stride
        idx_r = torch.arange(step - 1, Lp, step, device=a.device)
        idx_l = idx_r - stride
        # At this level, parent's exclusive value sits at idx_r; left child's
        # up-sweep value (inclusive composition of its subtree) sits at idx_l.
        a_parent = A.index_select(-2, idx_r)
        b_parent = B.index_select(-2, idx_r)
        a_lchild_up = A.index_select(-2, idx_l)
        b_lchild_up = B.index_select(-2, idx_l)
        # New exclusive values:
        #   left child  <- parent's exclusive value (no prior siblings)
        #   right child <- compose(parent_exclusive, left_child_up_sweep)
        new_a_r, new_b_r = _compose_left_then_right(
            a_parent, b_parent, a_lchild_up, b_lchild_up
        )
        A = A.index_copy(-2, idx_l, a_parent)
        B = B.index_copy(-2, idx_l, b_parent)
        A = A.index_copy(-2, idx_r, new_a_r)
        B = B.index_copy(-2, idx_r, new_b_r)
        stride //= 2

    # A[..., t, :], B[..., t, :] now hold the *exclusive* composition of
    # positions 0..t-1. Applied to h_{-1} = 0 we get h_{t-1} = B[..., t, :].
    # Convert to inclusive by composing with the original (a_t, b_t):
    #     h_t = a_t * h_{t-1} + b_t = a_t * B_excl[t] + b_t
    excl_b = B[..., :orig_L, :]
    return a * excl_b + b


def selective_scan_parallel(
    u: Tensor,
    delta: Tensor,
    A: Tensor,
    B: Tensor,
    C: Tensor,
    D: Optional[Tensor] = None,
    z: Optional[Tensor] = None,
    delta_bias: Optional[Tensor] = None,
    delta_softplus: bool = False,
    return_last_state: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """Selective scan via work-efficient associative prefix scan.

    Drop-in replacement for :func:`mamba_minimal.scan_naive.selective_scan_naive`:
    same argument order, same shapes, same dtype contract (fp32 accumulation).
    The only difference is the scan body — a vectorized Blelloch prefix scan
    replaces the Python for-loop.

    Shapes:
        * ``u``, ``delta``: ``(B, D, L)``
        * ``A``: ``(D, N)`` (negative real)
        * ``B``, ``C``: ``(B, N, L)`` shared, or ``(B, D, N, L)`` channel-specific
        * ``D``: ``(D,)`` optional skip
        * ``z``: ``(B, D, L)`` optional gate (SiLU applied)
    """

    if u.ndim != 3:
        raise ValueError(f"u must have shape (B, D, L), got {tuple(u.shape)}")
    if delta.shape != u.shape:
        raise ValueError(
            f"delta must match u shape {tuple(u.shape)}, got {tuple(delta.shape)}"
        )
    if A.ndim != 2:
        raise ValueError(f"A must have shape (D, N), got {tuple(A.shape)}")

    dtype_in = u.dtype
    batch, channels, length = u.shape
    state = A.shape[1]
    if A.shape[0] != channels:
        raise ValueError(f"A first dim must match channel dim {channels}, got {A.shape[0]}")

    u_f = u.float()
    delta_f = delta.float()
    A_f = A.float()
    B_f = B.float()
    C_f = C.float()

    if delta_bias is not None:
        delta_f = delta_f + delta_bias.float().view(1, -1, 1)
    if delta_softplus:
        delta_f = F.softplus(delta_f)

    # Discretize: a_t = exp(delta_t * A), b_t = delta_t * B_t * u_t.
    delta_A = torch.exp(torch.einsum("bdl,dn->bdln", delta_f, A_f))  # (B, D, L, N)
    if B_f.dim() == 3:
        delta_B_u = torch.einsum("bdl,bnl,bdl->bdln", delta_f, B_f, u_f)
    else:
        delta_B_u = torch.einsum("bdl,bdnl,bdl->bdln", delta_f, B_f, u_f)

    # Run the associative affine scan over the L axis. The scan sees
    # tensors of shape (B, D, L, N); the "feature" axis for the scan is N,
    # broadcast-independent across channels.
    h = blelloch_affine_scan(delta_A, delta_B_u)  # (B, D, L, N)

    # Apply output projection C.
    if C_f.dim() == 3:
        # C_f: (B, N, L), h: (B, D, L, N) -> y: (B, D, L)
        y = torch.einsum("bdln,bnl->bdl", h, C_f)
    else:
        # C_f: (B, D, N, L)
        y = torch.einsum("bdln,bdnl->bdl", h, C_f)

    if D is not None:
        y = y + D.float().view(1, -1, 1) * u_f
    if z is not None:
        y = y * F.silu(z.float())

    y = y.to(dtype=dtype_in)
    if return_last_state:
        last_state = h[..., -1, :]  # (B, D, N), fp32
        return y, last_state
    return y
