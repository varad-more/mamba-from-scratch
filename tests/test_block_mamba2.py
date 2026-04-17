"""End-to-end parity: our Mamba2Model vs mamba_ssm MambaLMHeadModel on
a real pretrained checkpoint.

We load ``state-spaces/mamba2-130m``, copy weights into our module, and
compare logits on a short prompt. Tolerance is loose at fp32 because our
conv1d path (pure PyTorch) accumulates differently from causal_conv1d and
our RMSNorm is un-fused, but the model output should be functionally
identical (top-1 token match, correlation ≈ 1).
"""

from __future__ import annotations

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("needs CUDA", allow_module_level=True)
pytest.importorskip("mamba_ssm")

from mamba_minimal.block_mamba2 import Mamba2Model  # noqa: E402


@pytest.mark.gpu
@pytest.mark.slow
def test_mamba2_130m_logits_match_reference():
    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
    device = "cuda"
    dtype = torch.float32
    name = "state-spaces/mamba2-130m"

    ref = MambaLMHeadModel.from_pretrained(name, device=device, dtype=dtype).eval()
    ours = Mamba2Model.from_pretrained(name, device=device, dtype=dtype)

    torch.manual_seed(0)
    ids = torch.randint(0, 50276, (1, 32), device=device)

    with torch.no_grad():
        l_ref = ref(ids).logits       # (1, L, V)
        l_ours = ours(ids)             # (1, L, V)

    assert l_ref.shape == l_ours.shape
    # Loose numeric tol (un-fused norm + plain conv1d). Assert argmax parity.
    argmax_match = (l_ref.argmax(-1) == l_ours.argmax(-1)).float().mean().item()
    # Spearman-ish: cosine sim on flattened last-token logits.
    a = l_ref[0, -1].float()
    b = l_ours[0, -1].float()
    cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
    assert argmax_match >= 0.9, f"argmax agreement = {argmax_match:.3f}"
    assert cos >= 0.999, f"cosine(logits) = {cos:.5f}"


@pytest.mark.gpu
@pytest.mark.slow
def test_mamba2_130m_step_matches_prefill():
    """Decode step after prefill should produce same logits as
    full-sequence prefill of the extended prompt."""
    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel  # noqa: F401
    device = "cuda"
    ours = Mamba2Model.from_pretrained("state-spaces/mamba2-130m", device=device, dtype=torch.float32)

    torch.manual_seed(1)
    prompt = torch.randint(0, 50276, (1, 16), device=device)
    next_tok = torch.randint(0, 50276, (1, 1), device=device)
    full = torch.cat([prompt, next_tok], dim=1)

    with torch.no_grad():
        # Reference: prefill the full thing, take logits at position len(prompt).
        logits_full = ours(full)[:, -1, :]

        # Ours: prefill prompt, then step with next_tok.
        _, state = ours(prompt, return_state=True)
        logits_step, _ = ours.step(next_tok[:, 0], state)

    cos = torch.nn.functional.cosine_similarity(
        logits_full[0].float(), logits_step[0].float(), dim=0
    ).item()
    assert cos >= 0.995, f"step vs prefill cosine = {cos:.5f}"
    assert (logits_full.argmax(-1) == logits_step.argmax(-1)).all()
