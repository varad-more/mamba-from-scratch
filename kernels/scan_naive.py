from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from torch import Tensor

from mamba_minimal.selective_scan import selective_scan_ref

@dataclass(slots=True)
class KernelMetadata:
    backend: str
    used_fallback: bool
    notes: str


def selective_scan_naive(
    u: Tensor,
    delta: Tensor,
    A: Tensor,
    B: Tensor,
    C: Tensor,
    D: Optional[Tensor] = None,
    z: Optional[Tensor] = None,
    return_metadata: bool = False,
) -> Tensor | tuple[Tensor, KernelMetadata]:
    """Naive kernel entry point.

    This path is intentionally an unfused baseline wrapper around the reference
    implementation. It exists so benchmark and validation code can compare a
    non-fused call site against the fused path without lying about behavior.
    """

    output = selective_scan_ref(u=u, delta=delta, A=A, B=B, C=C, D=D, z=z)
    metadata = KernelMetadata(
        backend="torch-reference-wrapper",
        used_fallback=True,
        notes="Unfused baseline wrapper, delegates to the PyTorch reference implementation.",
    )
    if return_metadata:
        return output, metadata
    return output
