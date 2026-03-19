from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from .parallel_scan import sequential_affine_scan


def causal_decay_matrix(a_chunk: Tensor) -> Tensor:
    """Build the lower-triangular decay matrix for an affine recurrence chunk.

    Parameters
    ----------
    a_chunk:
        Tensor with shape ``(B, L, F)``.

    Returns
    -------
    Tensor
        Lower-triangular decay weights with shape ``(B, F, L, L)`` where
        ``weights[:, :, i, j]`` equals ``prod_{k=j+1..i} a_k`` for ``j <= i``.
    """

    if a_chunk.ndim != 3:
        raise ValueError(f"a_chunk must have shape (B, L, F), got {tuple(a_chunk.shape)}")

    batch, length, features = a_chunk.shape
    weights = torch.zeros(batch, features, length, length, dtype=a_chunk.dtype, device=a_chunk.device)

    for i in range(length):
        weights[:, :, i, i] = 1.0
        running = torch.ones(batch, features, dtype=a_chunk.dtype, device=a_chunk.device)
        for j in range(i - 1, -1, -1):
            running = running * a_chunk[:, j + 1, :]
            weights[:, :, i, j] = running
    return weights


def ssd_affine_scan(a: Tensor, b: Tensor, chunk_size: int = 128, h0: Optional[Tensor] = None) -> Tensor:
    """Minimal SSD-style chunked scan.

    This prototype keeps the same affine recurrence as the sequential scan but
    expresses the within-chunk computation as a masked matrix multiplication.
    It is intentionally written for clarity rather than speed.
    """

    if a.shape != b.shape:
        raise ValueError(f"a and b must match, got {tuple(a.shape)} vs {tuple(b.shape)}")
    if a.ndim != 3:
        raise ValueError(f"a and b must have shape (B, L, F), got {tuple(a.shape)}")
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    batch, length, features = a.shape
    state = torch.zeros(batch, features, dtype=a.dtype, device=a.device) if h0 is None else h0
    outputs: list[Tensor] = []

    for start in range(0, length, chunk_size):
        end = min(length, start + chunk_size)
        a_chunk = a[:, start:end, :]
        b_chunk = b[:, start:end, :]
        weights = causal_decay_matrix(a_chunk)
        local = torch.einsum("bfij,bjf->bif", weights, b_chunk)

        prefix = torch.cumprod(a_chunk, dim=1)
        local = local + prefix * state.unsqueeze(1)
        outputs.append(local)
        state = local[:, -1, :]

    return torch.cat(outputs, dim=1)


def check_ssd_matches_sequential(a: Tensor, b: Tensor, chunk_size: int = 128) -> float:
    reference = sequential_affine_scan(a, b)
    ssd = ssd_affine_scan(a, b, chunk_size=chunk_size)
    return float((reference - ssd).abs().max().item())
