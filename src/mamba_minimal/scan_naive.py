"""Naive selective scan — the clarity-first oracle-matching reference.

The signature and defaults intentionally mirror
``mamba_ssm.ops.selective_scan_interface.selective_scan_ref`` so the two are
drop-in interchangeable. The only implementation rule here is: accumulate in
fp32 regardless of input dtype, so numerical drift does not obscure parity.

This lives alongside :func:`mamba_minimal.selective_scan.selective_scan_ref`,
which keeps an older signature (``delta_softplus=True`` default) used by the
rest of the codebase. New code that needs an oracle-compatible reference
should import :func:`selective_scan_naive` from this module.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor


def selective_scan_naive(
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
    """Reference selective scan with fp32 accumulation.

    Shapes follow the Mamba convention:

    * ``u``, ``delta``: ``(B, D, L)``
    * ``A``: ``(D, N)`` (negative real; the caller is responsible for sign)
    * ``B``, ``C``: ``(B, N, L)`` (shared) or ``(B, D, N, L)`` (channel-specific)
    * ``D``: ``(D,)`` skip
    * ``z``: ``(B, D, L)`` gate (SiLU applied)

    Output is returned in the input dtype of ``u``. All internal state is
    accumulated in fp32 even if inputs are fp16/bf16 — this is the point of
    the "naive" variant.
    """

    if u.ndim != 3:
        raise ValueError(f"u must have shape (B, D, L), got {tuple(u.shape)}")
    if delta.shape != u.shape:
        raise ValueError(f"delta must match u shape {tuple(u.shape)}, got {tuple(delta.shape)}")
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

    # Discretize A, B.
    delta_A = torch.exp(torch.einsum("bdl,dn->bdln", delta_f, A_f))
    if B_f.dim() == 3:
        delta_B_u = torch.einsum("bdl,bnl,bdl->bdln", delta_f, B_f, u_f)
    else:
        delta_B_u = torch.einsum("bdl,bdnl,bdl->bdln", delta_f, B_f, u_f)

    x = u_f.new_zeros((batch, channels, state))
    ys: list[Tensor] = []
    last_state: Optional[Tensor] = None

    for t in range(length):
        x = delta_A[:, :, t] * x + delta_B_u[:, :, t]
        if C_f.dim() == 3:
            y_t = torch.einsum("bdn,bn->bd", x, C_f[:, :, t])
        else:
            y_t = torch.einsum("bdn,bdn->bd", x, C_f[:, :, :, t])
        ys.append(y_t)

    y = torch.stack(ys, dim=-1)
    if D is not None:
        y = y + D.float().view(1, -1, 1) * u_f
    if z is not None:
        y = y * F.silu(z.float())

    if return_last_state:
        last_state = x
        return y.to(dtype=dtype_in), last_state
    return y.to(dtype=dtype_in)
