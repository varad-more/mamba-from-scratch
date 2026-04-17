"""Full Mamba-130m parity: our :class:`MambaModel` vs HuggingFace baseline.

These tests are gated on ``transformers`` being importable. They do a single
download of ``state-spaces/mamba-130m-hf`` (cached on subsequent runs) and
compare:

1. Full-sequence logits on a short prompt.
2. Greedy generation output via :func:`mamba_minimal.generate.generate_native`
   against ``hf.generate(..., do_sample=False, use_cache=False)``.
"""

from __future__ import annotations

import pytest
import torch

transformers = pytest.importorskip("transformers")

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from mamba_minimal.generate import generate_native  # noqa: E402
from mamba_minimal.model import MambaModel  # noqa: E402

MODEL_NAME = "state-spaces/mamba-130m-hf"


@pytest.fixture(scope="module")
def hf_and_ours():
    hf = AutoModelForCausalLM.from_pretrained(MODEL_NAME).eval()
    ours = MambaModel.from_pretrained(MODEL_NAME, scan_impl="parallel", device="cpu").eval()
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    return hf, ours, tok


@pytest.mark.parametrize(
    "prompt",
    ["Mamba is useful because", "Hello world", "The quick brown fox jumps"],
)
def test_full_logits_match_hf(hf_and_ours, prompt: str) -> None:
    hf, ours, tok = hf_and_ours
    ids = tok(prompt, return_tensors="pt").input_ids
    with torch.no_grad():
        y_hf = hf(ids, use_cache=False).logits
        y_ours = ours(ids)
    # fp32 tolerance — HF's slow path and our parallel scan order ops
    # differently, so we expect roundoff but not behavioral drift.
    torch.testing.assert_close(y_ours, y_hf, atol=1e-3, rtol=1e-3)


def test_step_matches_prefill(hf_and_ours) -> None:
    """Prefill-then-step must match a full forward on the extended sequence."""
    _, ours, _ = hf_and_ours
    ids = torch.tensor([[1, 42, 7, 11, 99, 1234, 50, 22]])
    with torch.no_grad():
        _, state = ours(ids[:, :-1], return_state=True)
        full_logits = ours(ids)
        step_logits, _ = ours.step(ids[:, -1], state)
    torch.testing.assert_close(step_logits, full_logits[:, -1, :], atol=1e-3, rtol=1e-3)
    assert full_logits[0, -1].argmax().item() == step_logits[0].argmax().item()


def test_generate_native_matches_hf_greedy(hf_and_ours) -> None:
    hf, ours, tok = hf_and_ours
    ids = tok("Mamba is useful because", return_tensors="pt").input_ids
    out_ours = generate_native(ours, ids, max_new_tokens=20, temperature=0.0)
    with torch.no_grad():
        out_hf = hf.generate(ids, max_new_tokens=20, do_sample=False, use_cache=False)
    assert out_ours.shape == out_hf.shape
    assert (out_ours == out_hf).all(), (
        f"mismatch\n ours: {tok.decode(out_ours[0])}\n hf  : {tok.decode(out_hf[0])}"
    )
