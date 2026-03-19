from __future__ import annotations

import torch

from mamba_minimal.parallel_scan import (
    chunked_affine_scan,
    hillis_steele_affine_scan,
    sequential_affine_scan,
)
from mamba_minimal.ssd import check_ssd_matches_sequential


def test_hillis_steele_matches_sequential() -> None:
    torch.manual_seed(0)
    a = torch.rand(2, 16, 5, dtype=torch.float64)
    b = torch.randn(2, 16, 5, dtype=torch.float64)
    reference = sequential_affine_scan(a, b)
    parallel = hillis_steele_affine_scan(a, b)
    assert torch.allclose(reference, parallel, atol=1e-10, rtol=1e-10)


def test_chunked_scan_matches_sequential_with_initial_state() -> None:
    torch.manual_seed(0)
    a = torch.rand(3, 19, 4)
    b = torch.randn(3, 19, 4)
    h0 = torch.randn(3, 4)
    reference = sequential_affine_scan(a, b, h0=h0)
    chunked = chunked_affine_scan(a, b, chunk_size=6, h0=h0)
    assert torch.allclose(reference, chunked, atol=1e-6, rtol=1e-6)


def test_ssd_chunked_prototype_matches_sequential() -> None:
    torch.manual_seed(0)
    a = torch.rand(2, 12, 3, dtype=torch.float64)
    b = torch.randn(2, 12, 3, dtype=torch.float64)
    max_error = check_ssd_matches_sequential(a, b, chunk_size=4)
    assert max_error < 1e-10
