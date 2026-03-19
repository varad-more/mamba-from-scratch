from __future__ import annotations

import argparse

import torch

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception as exc:  # pragma: no cover - optional dependency
    AutoModelForCausalLM = None
    AutoTokenizer = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


def generate_text(model_name: str, prompt: str, max_new_tokens: int = 64, device: str = "cpu") -> str:
    if AutoModelForCausalLM is None or AutoTokenizer is None:
        raise RuntimeError(f"transformers is required for generation: {IMPORT_ERROR}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()
    encoded = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(**encoded, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text with a HuggingFace causal LM.")
    parser.add_argument("prompt")
    parser.add_argument("--model", default="state-spaces/mamba-130m-hf")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(generate_text(args.model, args.prompt, args.max_new_tokens, device))


if __name__ == "__main__":
    main()
