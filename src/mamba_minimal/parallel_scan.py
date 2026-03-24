from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor


def _validate_inputs(a: Tensor, b: Tensor) -> tuple[int, int]:
    if a.shape != b.shape:
        raise ValueError(f"a and b must have the same shape, got {tuple(a.shape)} vs {tuple(b.shape)}")
    if a.ndim < 2:
        raise ValueError("a and b must have shape (..., L, F)")
    length = a.shape[-2]
    features = a.shape[-1]
    return length, features


def _combine(left_a: Tensor, left_b: Tensor, right_a: Tensor, right_b: Tensor) -> tuple[Tensor, Tensor]:
    """Compose affine transforms in sequence.

    Each pair ``(a, b)`` represents the map ``x -> a * x + b``. The result of
    ``combine(left, right)`` is the affine map that applies ``left`` first and
    ``right`` second.
    """

    return right_a * left_a, right_a * left_b + right_b


def sequential_affine_scan(a: Tensor, b: Tensor, h0: Optional[Tensor] = None) -> Tensor:
    """Sequential reference for the recurrence ``h_t = a_t * h_{t-1} + b_t``."""

    _validate_inputs(a, b)
    leading = a.shape[:-2]
    length, features = a.shape[-2:]

    if h0 is None:
        state = torch.zeros(*leading, features, dtype=a.dtype, device=a.device)
    else:
        state = h0.clone()
    outputs: list[Tensor] = []

    for index in range(length):
        state = a[..., index, :] * state + b[..., index, :]
        outputs.append(state)
    return torch.stack(outputs, dim=-2)


def hillis_steele_affine_scan(a: Tensor, b: Tensor, h0: Optional[Tensor] = None) -> Tensor:
    """Inclusive parallel scan using iterative doubling.

    This is not the most work-efficient scan, but it is compact, easy to read,
    and captures the same associative structure used by GPU prefix-scan kernels.
    """

    length, features = _validate_inputs(a, b)
    prefix_a = a.clone()
    prefix_b = b.clone()

    offset = 1
    while offset < length:
        left_a = prefix_a[..., :-offset, :]
        left_b = prefix_b[..., :-offset, :]
        right_a = prefix_a[..., offset:, :].clone()
        right_b = prefix_b[..., offset:, :].clone()
        combined_a, combined_b = _combine(left_a, left_b, right_a, right_b)
        prefix_a[..., offset:, :] = combined_a
        prefix_b[..., offset:, :] = combined_b
        offset *= 2

    if h0 is None:
        return prefix_b
    return prefix_a * h0.unsqueeze(-2) + prefix_b


def chunked_affine_scan(
    a: Tensor,
    b: Tensor,
    chunk_size: int = 128,
    h0: Optional[Tensor] = None,
) -> Tensor:
    """Chunked scan: sequential within chunks, associative carry across chunks."""

    length, features = _validate_inputs(a, b)
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    leading = a.shape[:-2]
    num_chunks = (length + chunk_size - 1) // chunk_size
    chunk_outputs: list[Tensor] = []
    carry_a: list[Tensor] = []
    carry_b: list[Tensor] = []

    for chunk_idx in range(num_chunks):
        start = chunk_idx * chunk_size
        end = min(length, start + chunk_size)
        a_chunk = a[..., start:end, :]
        b_chunk = b[..., start:end, :]
        local = sequential_affine_scan(a_chunk, b_chunk)
        chunk_outputs.append(local)
        carry_a.append(torch.prod(a_chunk, dim=-2))

        local_b = b_chunk[..., 0, :]
        if end - start > 1:
            local_b = local[..., -1, :]
        carry_b.append(local_b)

    carry_a_tensor = torch.stack(carry_a, dim=-2)
    carry_b_tensor = torch.stack(carry_b, dim=-2)
    chunk_starts = sequential_affine_scan(carry_a_tensor, carry_b_tensor, h0=h0)

    outputs: list[Tensor] = []
    for chunk_idx, local in enumerate(chunk_outputs):
        if chunk_idx == 0 and h0 is None:
            outputs.append(local)
        else:
            if chunk_idx == 0:
                prev = h0
            else:
                prev = chunk_starts[..., chunk_idx - 1, :]
            start = chunk_idx * chunk_size
            end = min(length, start + chunk_size)
            a_chunk = a[..., start:end, :]
            prefix = torch.cumprod(a_chunk, dim=-2)
            outputs.append(local + prefix * prev.unsqueeze(-2))

    return torch.cat(outputs, dim=-2)
