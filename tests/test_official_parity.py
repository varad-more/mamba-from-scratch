from __future__ import annotations

import os

import pytest
import torch

from mamba_minimal.model import MambaBlock
from scripts.official_parity import extract_official_mixer, try_build_block_from_official


def _should_run() -> bool:
    return os.getenv("RUN_OFFICIAL_PARITY") == "1"


@pytest.mark.slow
@pytest.mark.skipif(not _should_run(), reason="Set RUN_OFFICIAL_PARITY=1 to run official-model parity")
def test_local_block_matches_official_mixer_layer_zero() -> None:
    transformers = pytest.importorskip("transformers")
    model = transformers.AutoModelForCausalLM.from_pretrained("state-spaces/mamba-130m-hf")
    model.eval()

    block = try_build_block_from_official(model, layer_idx=0)
    assert isinstance(block, MambaBlock)
    block.eval()

    x = torch.randn(1, 8, block.d_model)
    mixer = extract_official_mixer(model, 0)
    with torch.no_grad():
        official = mixer(x)
        local = block(x)

    assert official.shape == local.shape
    assert torch.allclose(official, local, atol=1e-6, rtol=1e-6)
