from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from .types import ScanSupportInfo


def selective_scan_fused_shape_support(
    u: Tensor,
    delta: Tensor,
    A: Tensor,
    B: Tensor,
    C: Tensor,
    D: Optional[Tensor] = None,
    z: Optional[Tensor] = None,
) -> ScanSupportInfo:
    """Return whether inputs fit the current fused selective-scan boundary."""

    if u.ndim != 3 or delta.shape != u.shape:
        return ScanSupportInfo(False, "invalid", "u and delta must have shape (B, D, L).")
    if A.ndim != 2:
        return ScanSupportInfo(False, "invalid", "A must have shape (D, N).")

    batch, channels, _ = u.shape
    state = A.shape[1]
    if A.shape[0] != channels:
        return ScanSupportInfo(False, "invalid", "A.shape[0] must match channel dimension D.")

    if B.ndim != C.ndim:
        return ScanSupportInfo(False, "invalid", "B and C must have the same rank.")

    if B.ndim == 3:
        if B.shape != (batch, state, u.shape[-1]) or C.shape != (batch, state, u.shape[-1]):
            return ScanSupportInfo(False, "shared-bc", "Rank-3 B and C must have shape (B, N, L).")
        layout = "shared-bc"
    elif B.ndim == 4:
        if B.shape != (batch, channels, state, u.shape[-1]) or C.shape != (
            batch,
            channels,
            state,
            u.shape[-1],
        ):
            return ScanSupportInfo(
                False,
                "channel-bc",
                "Rank-4 B and C must have shape (B, D, N, L).",
            )
        layout = "channel-bc"
    else:
        return ScanSupportInfo(
            False,
            "invalid",
            "Current fused Triton path supports only rank-3 or rank-4 B/C tensors.",
        )

    if D is not None and D.shape != (channels,):
        return ScanSupportInfo(False, layout, "D must have shape (D,).")
    if z is not None and z.shape != u.shape:
        return ScanSupportInfo(False, layout, "z must match u shape (B, D, L).")
    if state > 128:
        return ScanSupportInfo(False, layout, "Current Triton kernel supports state size up to 128.")
    if u.dtype not in (torch.float16, torch.float32, torch.bfloat16):
        return ScanSupportInfo(False, layout, f"Unsupported dtype for Triton path: {u.dtype}.")

    return ScanSupportInfo(True, layout, f"Supported fused Triton layout: {layout}.")


def selective_scan_fused_runtime_support(
    u: Tensor,
    delta: Tensor,
    A: Tensor,
    B: Tensor,
    C: Tensor,
    D: Optional[Tensor] = None,
    z: Optional[Tensor] = None,
    *,
    triton_available: bool,
) -> ScanSupportInfo:
    shape_support = selective_scan_fused_shape_support(u, delta, A, B, C, D=D, z=z)
    if not shape_support.supported:
        return shape_support
    if not triton_available:
        return ScanSupportInfo(False, shape_support.layout, "Triton is not installed.")
    if not u.is_cuda:
        return ScanSupportInfo(False, shape_support.layout, "CUDA tensor required for Triton execution.")
    if delta.device != u.device or A.device != u.device or B.device != u.device or C.device != u.device:
        return ScanSupportInfo(False, shape_support.layout, "All tensors must live on the same CUDA device.")
    return shape_support
