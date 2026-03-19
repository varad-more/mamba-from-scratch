from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass(slots=True)
class SelectiveScanOutput:
    y: Tensor
    last_state: Optional[Tensor] = None


def _validate_bc(name: str, value: Tensor, batch: int, channels: int, state: int, length: int) -> None:
    if value.ndim == 3 and value.shape != (batch, state, length):
        raise ValueError(
            f"{name} with 3 dims must have shape (B, N, L); got {tuple(value.shape)}"
        )
    if value.ndim == 4 and value.shape != (batch, channels, state, length):
        raise ValueError(
            f"{name} with 4 dims must have shape (B, D, N, L); got {tuple(value.shape)}"
        )
    if value.ndim not in (3, 4):
        raise ValueError(f"{name} must have rank 3 or 4, got rank {value.ndim}")


def selective_scan_ref(
    u: Tensor,
    delta: Tensor,
    A: Tensor,
    B: Tensor,
    C: Tensor,
    D: Optional[Tensor] = None,
    z: Optional[Tensor] = None,
    delta_bias: Optional[Tensor] = None,
    delta_softplus: bool = True,
    return_last_state: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """Reference selective scan implementation.

    Parameters
    ----------
    u:
        Input tensor with shape ``(B, D, L)``.
    delta:
        Per-token time-step tensor with shape ``(B, D, L)``.
    A:
        Continuous transition coefficients with shape ``(D, N)``.
    B:
        Input projection coefficients with shape ``(B, N, L)`` or ``(B, D, N, L)``.
    C:
        Output projection coefficients with shape ``(B, N, L)`` or ``(B, D, N, L)``.
    D:
        Optional skip connection with shape ``(D,)``.
    z:
        Optional gating tensor with shape ``(B, D, L)``. A SiLU gate is applied.
    delta_bias:
        Optional bias added to ``delta`` before the softplus.
    delta_softplus:
        Whether to apply softplus to ensure positive time-steps.
    return_last_state:
        Whether to also return the final recurrent state.
    """

    if u.ndim != 3:
        raise ValueError(f"u must have shape (B, D, L), got {tuple(u.shape)}")
    if delta.shape != u.shape:
        raise ValueError(f"delta must match u shape {tuple(u.shape)}, got {tuple(delta.shape)}")
    if A.ndim != 2:
        raise ValueError(f"A must have shape (D, N), got {tuple(A.shape)}")

    batch, channels, length = u.shape
    state = A.shape[1]
    if A.shape[0] != channels:
        raise ValueError(f"A first dim must match channel dim {channels}, got {A.shape[0]}")

    _validate_bc("B", B, batch, channels, state, length)
    _validate_bc("C", C, batch, channels, state, length)

    if D is not None and D.shape != (channels,):
        raise ValueError(f"D must have shape ({channels},), got {tuple(D.shape)}")
    if z is not None and z.shape != u.shape:
        raise ValueError(f"z must match u shape {tuple(u.shape)}, got {tuple(z.shape)}")
    if delta_bias is not None and delta_bias.shape != (channels,):
        raise ValueError(
            f"delta_bias must have shape ({channels},), got {tuple(delta_bias.shape)}"
        )

    delta_work = delta
    if delta_bias is not None:
        delta_work = delta_work + delta_bias.view(1, -1, 1)
    if delta_softplus:
        delta_work = F.softplus(delta_work)

    delta_A = torch.exp(torch.einsum("bdl,dn->bdln", delta_work, A))
    if B.ndim == 3:
        delta_B_u = torch.einsum("bdl,bnl,bdl->bdln", delta_work, B, u)
    else:
        delta_B_u = torch.einsum("bdl,bdnl,bdl->bdln", delta_work, B, u)

    x = torch.zeros(batch, channels, state, device=u.device, dtype=u.dtype)
    ys: list[Tensor] = []

    for index in range(length):
        x = delta_A[:, :, index, :] * x + delta_B_u[:, :, index, :]
        if C.ndim == 3:
            y_t = torch.einsum("bdn,bn->bd", x, C[:, :, index])
        else:
            y_t = torch.einsum("bdn,bdn->bd", x, C[:, :, :, index])
        if D is not None:
            y_t = y_t + D.view(1, -1) * u[:, :, index]
        ys.append(y_t)

    y = torch.stack(ys, dim=-1)
    if z is not None:
        y = y * F.silu(z)

    if return_last_state:
        return y, x
    return y


class SelectiveScan(nn.Module):
    """Thin module wrapper around :func:`selective_scan_ref`."""

    def forward(
        self,
        u: Tensor,
        delta: Tensor,
        A: Tensor,
        B: Tensor,
        C: Tensor,
        D: Optional[Tensor] = None,
        z: Optional[Tensor] = None,
        delta_bias: Optional[Tensor] = None,
        delta_softplus: bool = True,
        return_last_state: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        return selective_scan_ref(
            u=u,
            delta=delta,
            A=A,
            B=B,
            C=C,
            D=D,
            z=z,
            delta_bias=delta_bias,
            delta_softplus=delta_softplus,
            return_last_state=return_last_state,
        )
