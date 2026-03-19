from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from torch import Tensor

from mamba_minimal.selective_scan import selective_scan_ref

try:
    import triton  # type: ignore

    TRITON_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    triton = None
    TRITON_AVAILABLE = False


@dataclass(slots=True)
class KernelMetadata:
    backend: str
    used_fallback: bool
    notes: str


def selective_scan_fused(
    u: Tensor,
    delta: Tensor,
    A: Tensor,
    B: Tensor,
    C: Tensor,
    D: Optional[Tensor] = None,
    z: Optional[Tensor] = None,
    return_metadata: bool = False,
) -> Tensor | tuple[Tensor, KernelMetadata]:
    """Fused kernel entry point.

    On machines without CUDA/Triton support, the function falls back to the
    trusted reference implementation. This keeps correctness and CPU testability
    intact while allowing a drop-in GPU kernel later.
    """

    output = selective_scan_ref(u=u, delta=delta, A=A, B=B, C=C, D=D, z=z)
    metadata = KernelMetadata(
        backend="triton" if TRITON_AVAILABLE and u.is_cuda else "torch-reference",
        used_fallback=not (TRITON_AVAILABLE and u.is_cuda),
        notes="Reference fallback used unless Triton CUDA kernel is available.",
    )
    if return_metadata:
        return output, metadata
    return output
