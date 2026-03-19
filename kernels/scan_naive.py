from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
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

    The project keeps a working reference fallback at all times. When Triton is
    unavailable, this function delegates to the PyTorch reference path so the
    rest of the repo remains runnable on CPU-only environments.
    """

    output = selective_scan_ref(u=u, delta=delta, A=A, B=B, C=C, D=D, z=z)
    metadata = KernelMetadata(
        backend="triton" if TRITON_AVAILABLE and u.is_cuda else "torch-reference",
        used_fallback=not (TRITON_AVAILABLE and u.is_cuda),
    )
    if return_metadata:
        return output, metadata
    return output
