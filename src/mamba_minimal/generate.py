from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import torch

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception as exc:  # pragma: no cover - optional dependency
    AutoModelForCausalLM = None
    AutoTokenizer = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None

DEFAULT_MODEL_NAME = "state-spaces/mamba-130m-hf"


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
