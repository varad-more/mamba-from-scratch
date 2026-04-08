from __future__ import annotations

from typing import Optional

from torch import Tensor

from .capability import selective_scan_fused_runtime_support
from .types import ScanBackend, ScanDispatchMetadata


class BackendSelectionError(RuntimeError):
    """Raised when a requested backend cannot satisfy the current inputs."""


def select_scan_backend(
    u: Tensor,
    delta: Tensor,
    A: Tensor,
    B: Tensor,
    C: Tensor,
    D: Optional[Tensor] = None,
    z: Optional[Tensor] = None,
    *,
    requested_backend: ScanBackend = "auto",
    triton_available: bool,
) -> ScanDispatchMetadata:
    support = selective_scan_fused_runtime_support(
        u,
        delta,
        A,
        B,
        C,
        D=D,
        z=z,
        triton_available=triton_available,
    )

    if requested_backend == "reference":
        return ScanDispatchMetadata(
            requested_backend=requested_backend,
            selected_backend="torch-reference",
            used_fallback=False,
            layout=support.layout,
            notes="Reference backend forced by policy.",
        )

    if requested_backend == "fused":
        if not support.supported:
            raise BackendSelectionError(f"Requested fused backend is unavailable: {support.reason}")
        return ScanDispatchMetadata(
            requested_backend=requested_backend,
            selected_backend="triton-fused",
            used_fallback=False,
            layout=support.layout,
            notes=f"Fused backend selected ({support.layout}).",
        )

    if requested_backend != "auto":
        raise ValueError(f"Unknown scan backend policy: {requested_backend}")

    if support.supported:
        return ScanDispatchMetadata(
            requested_backend=requested_backend,
            selected_backend="triton-fused",
            used_fallback=False,
            layout=support.layout,
            notes=f"Auto-selected fused backend ({support.layout}).",
        )

    return ScanDispatchMetadata(
        requested_backend=requested_backend,
        selected_backend="torch-reference",
        used_fallback=True,
        layout=support.layout,
        notes=support.reason,
    )
