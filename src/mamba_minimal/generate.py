from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Optional

import torch
from torch import Tensor

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception as exc:  # pragma: no cover - optional dependency
    AutoModelForCausalLM = None
    AutoTokenizer = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None

DEFAULT_MODEL_NAME = "state-spaces/mamba-130m-hf"


# ----------------------------------------------------------------------
# State-carrying generate on our own MambaModel
# ----------------------------------------------------------------------


def _sample_from_logits(
    logits: Tensor,
    temperature: float,
    top_k: Optional[int],
    generator: Optional[torch.Generator],
) -> Tensor:
    """Convert last-step logits ``(B, V)`` into next-token ids ``(B,)``.

    ``temperature <= 0`` selects greedy argmax. ``top_k`` (if given) keeps
    the top-k logits per row before sampling.
    """

    if temperature is None or temperature <= 0:
        return logits.argmax(dim=-1)

    scaled = logits / temperature
    if top_k is not None and top_k > 0 and top_k < scaled.shape[-1]:
        topk_vals, topk_idx = scaled.topk(top_k, dim=-1)
        probs = torch.softmax(topk_vals, dim=-1)
        sampled_idx = torch.multinomial(probs, num_samples=1, generator=generator)
        return topk_idx.gather(-1, sampled_idx).squeeze(-1)

    probs = torch.softmax(scaled, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)


@torch.no_grad()
def generate_native(
    model: Any,
    prompt_ids: Tensor,
    max_new_tokens: int = 32,
    temperature: float = 0.0,
    top_k: Optional[int] = None,
    eos_token_id: Optional[int] = None,
    seed: Optional[int] = None,
) -> Tensor:
    """Autoregressive generation using our :class:`MambaModel` step API.

    The "KV-cache equivalent" is carried as a list of per-layer
    ``(conv_state, ssm_state)`` tuples returned by the model's prefill —
    there is no growing attention cache because Mamba has no attention.

    Args:
        model: a :class:`mamba_minimal.model.MambaModel`.
        prompt_ids: token ids of shape ``(B, L_prompt)``.
        max_new_tokens: number of tokens to decode after the prompt.
        temperature: ``<= 0`` for greedy, positive for sampling.
        top_k: optional top-k truncation for sampling.
        eos_token_id: stop decoding if every batch row has emitted EOS.
        seed: seed for reproducible sampling.

    Returns:
        Tensor of shape ``(B, L_prompt + max_new_tokens_produced)``.
    """

    if prompt_ids.ndim != 2:
        raise ValueError(f"prompt_ids must be (B, L), got {tuple(prompt_ids.shape)}")

    device = prompt_ids.device
    generator: Optional[torch.Generator] = None
    if seed is not None and temperature > 0:
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed))

    # Prefill: run the full prompt, capture per-layer state.
    logits, state = model(prompt_ids, return_state=True)
    next_token = _sample_from_logits(logits[:, -1, :], temperature, top_k, generator)
    emitted = [next_token]

    finished = torch.zeros(prompt_ids.shape[0], dtype=torch.bool, device=device)
    if eos_token_id is not None:
        finished = finished | (next_token == eos_token_id)

    for _ in range(max_new_tokens - 1):
        if finished.all():
            break
        logits, state = model.step(next_token, state)
        next_token = _sample_from_logits(logits, temperature, top_k, generator)
        emitted.append(next_token)
        if eos_token_id is not None:
            finished = finished | (next_token == eos_token_id)

    new_tokens = torch.stack(emitted, dim=1)
    return torch.cat([prompt_ids, new_tokens], dim=1)


@dataclass(slots=True)
class LoadedGenerationModel:
    model_name: str
    device: str
    model: Any
    tokenizer: Any


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


@lru_cache(maxsize=4)
def _cached_load(model_name: str, device: str) -> LoadedGenerationModel:
    if AutoModelForCausalLM is None or AutoTokenizer is None:
        raise RuntimeError(f"transformers is required for generation: {IMPORT_ERROR}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()
    return LoadedGenerationModel(
        model_name=model_name,
        device=device,
        model=model,
        tokenizer=tokenizer,
    )


def load_generation_model(
    model_name: str = DEFAULT_MODEL_NAME,
    device: str = "auto",
) -> LoadedGenerationModel:
    resolved_device = resolve_device(device)
    return _cached_load(model_name, resolved_device)


def generate_with_loaded_model(
    loaded: LoadedGenerationModel,
    prompt: str,
    max_new_tokens: int = 64,
    do_sample: bool = False,
    temperature: float = 1.0,
) -> str:
    encoded = loaded.tokenizer(prompt, return_tensors="pt").to(loaded.device)

    with torch.no_grad():
        output_ids = loaded.model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
        )
    return loaded.tokenizer.decode(output_ids[0], skip_special_tokens=True)


def generate_text(
    model_name: str,
    prompt: str,
    max_new_tokens: int = 64,
    device: str = "cpu",
    do_sample: bool = False,
    temperature: float = 1.0,
) -> str:
    loaded = load_generation_model(model_name=model_name, device=device)
    return generate_with_loaded_model(
        loaded,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text with a HuggingFace causal LM.")
    parser.add_argument("prompt")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args()

    print(
        generate_text(
            model_name=args.model,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            device=args.device,
            do_sample=args.do_sample,
            temperature=args.temperature,
        )
    )


if __name__ == "__main__":
    main()
