from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import torch

try:
    import psutil
except Exception:  # pragma: no cover - optional dependency
    psutil = None

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception as exc:  # pragma: no cover - optional dependency
    AutoModelForCausalLM = None
    AutoTokenizer = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None

DEFAULT_PROMPT_LENGTHS = [8, 32, 128, 512]
PROMPT_SEED = "Mamba selective state spaces enable efficient long-context inference. "


@dataclass(slots=True)
class InferenceResult:
    model_name: str
    device: str
    machine: str
    prompt_tokens: int
    target_prompt_tokens: int
    new_tokens: int
    ttft_ms: float
    inter_token_ms: float
    throughput_tokens_per_s: float
    peak_memory_mb: float


@dataclass(slots=True)
class LoadedModel:
    model_name: str
    model: object
    tokenizer: object


def current_memory_mb(device: str) -> float:
    if device.startswith("cuda"):
        return torch.cuda.max_memory_allocated() / 1024**2
    if psutil is None:
        return 0.0
    return psutil.Process().memory_info().rss / 1024**2


def machine_summary(device: str) -> str:
    if device.startswith("cuda"):
        return torch.cuda.get_device_name(0)
    return f"{platform.system()} {platform.release()} ({platform.machine()})"


def parse_prompt_lengths(value: str | None) -> list[int]:
    if value is None:
        return list(DEFAULT_PROMPT_LENGTHS)
    lengths = [int(chunk.strip()) for chunk in value.split(",") if chunk.strip()]
    if not lengths:
        raise ValueError("No valid prompt lengths were provided.")
    return lengths


def build_prompt(target_tokens: int) -> str:
    words = max(8, target_tokens * 2)
    repeated = (PROMPT_SEED * ((words // len(PROMPT_SEED.split())) + 4)).split()
    return " ".join(repeated[:words])


def load_model(model_name: str, device: str) -> LoadedModel:
    if AutoModelForCausalLM is None or AutoTokenizer is None:
        raise RuntimeError(f"transformers is required for this benchmark: {IMPORT_ERROR}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()
    return LoadedModel(model_name=model_name, model=model, tokenizer=tokenizer)


def measure_loaded_model(loaded: LoadedModel, prompt: str, target_prompt_tokens: int, new_tokens: int, device: str) -> InferenceResult:
    tokenizer = loaded.tokenizer
    model = loaded.model
    encoded = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_tokens = int(encoded["input_ids"].shape[-1])

    if device.startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    start = time.perf_counter()
    with torch.no_grad():
        _ = model.generate(**encoded, max_new_tokens=1, do_sample=False)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    ttft_ms = (time.perf_counter() - start) * 1000.0

    start = time.perf_counter()
    with torch.no_grad():
        _ = model.generate(**encoded, max_new_tokens=new_tokens, do_sample=False)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    full_ms = (time.perf_counter() - start) * 1000.0
    inter_token_ms = max(0.0, (full_ms - ttft_ms) / max(1, new_tokens - 1))
    throughput = 0.0 if inter_token_ms == 0.0 else 1000.0 / inter_token_ms
    peak_memory_mb = current_memory_mb(device)

    return InferenceResult(
        model_name=loaded.model_name,
        device=device,
        machine=machine_summary(device),
        prompt_tokens=prompt_tokens,
        target_prompt_tokens=target_prompt_tokens,
        new_tokens=new_tokens,
        ttft_ms=ttft_ms,
        inter_token_ms=inter_token_ms,
        throughput_tokens_per_s=throughput,
        peak_memory_mb=peak_memory_mb,
    )


def run_inference_benchmarks(
    mamba_model: str,
    baseline_model: str,
    new_tokens: int,
    device: str,
    prompt_lengths: Iterable[int],
) -> list[InferenceResult]:
    loaded_models = [load_model(mamba_model, device), load_model(baseline_model, device)]
    results: list[InferenceResult] = []

    for prompt_length in prompt_lengths:
        prompt = build_prompt(prompt_length)
        for loaded in loaded_models:
            results.append(
                measure_loaded_model(
                    loaded,
                    prompt=prompt,
                    target_prompt_tokens=prompt_length,
                    new_tokens=new_tokens,
                    device=device,
                )
            )
    return results


def save_results(path: Path, results: list[InferenceResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(result) for result in results], indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Mamba vs GPT-2 inference behavior.")
    parser.add_argument("--mamba-model", default="state-spaces/mamba-130m-hf")
    parser.add_argument("--baseline-model", default="gpt2")
    parser.add_argument("--new-tokens", type=int, default=32)
    parser.add_argument("--prompt-lengths", type=str, default=None, help="Comma-separated prompt token targets")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path")
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    results = run_inference_benchmarks(
        args.mamba_model,
        args.baseline_model,
        args.new_tokens,
        device,
        parse_prompt_lengths(args.prompt_lengths),
    )
    payload = [asdict(result) for result in results]

    if args.output is not None:
        save_results(args.output, results)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
