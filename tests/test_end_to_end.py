from __future__ import annotations

import torch

from mamba_minimal import MambaBlock


def test_end_to_end_forward_is_deterministic_for_same_seed() -> None:
    torch.manual_seed(123)
    block_a = MambaBlock(d_model=16, d_state=8, d_conv=4, expand=2)
    x = torch.randn(2, 10, 16)
    out_a = block_a(x)

    torch.manual_seed(123)
    block_b = MambaBlock(d_model=16, d_state=8, d_conv=4, expand=2)
    out_b = block_b(x)

    assert torch.allclose(out_a, out_b, atol=1e-6, rtol=1e-6)


def test_end_to_end_handles_single_token_sequences() -> None:
    block = MambaBlock(d_model=4, d_state=4, d_conv=2, expand=2)
    x = torch.randn(1, 1, 4)
    y = block(x)
    assert y.shape == (1, 1, 4)
    assert torch.isfinite(y).all()
