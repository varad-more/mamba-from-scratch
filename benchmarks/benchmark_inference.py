from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

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


PROMPTS = [
    "Mamba is interesting because",
    "The difference between memory-bound and compute-bound workloads is",
]


@dataclass(slots=True)
class InferenceResult:
    model_name: str
    device: str
    prompt_tokens: int
    new_tokens: int
    ttft_ms: float
    inter_token_ms: float
    peak_memory_mb: float


def current_memory_mb(device: str) -> float:
    if device.startswith("cuda"):
        return torch.cuda.max_memory_allocated() / 1024**2
    if psutil is None:
        return 0.0
    return psutil.Process().memory_info().rss / 1024**2


def measure_model(model_name: str, prompt: str, new_tokens: int, device: str) -> InferenceResult:
    if AutoModelForCausalLM is None or AutoTokenizer is None:
        raise RuntimeError(f"transformers is required for this benchmark: {IMPORT_ERROR}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()

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
    peak_memory_mb = current_memory_mb(device)

    return InferenceResult(
        model_name=model_name,
        device=device,
        prompt_tokens=prompt_tokens,
        new_tokens=new_tokens,
        ttft_ms=ttft_ms,
        inter_token_ms=inter_token_ms,
        peak_memory_mb=peak_memory_mb,
    )


def run_inference_benchmarks(mamba_model: str, baseline_model: str, new_tokens: int, device: str) -> list[InferenceResult]:
    results = []
    for prompt in PROMPTS:
        results.append(measure_model(mamba_model, prompt, new_tokens, device))
        results.append(measure_model(baseline_model, prompt, new_tokens, device))
    return results


def save_results(path: Path, results: list[InferenceResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(result) for result in results], indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Mamba vs GPT-2 inference behavior.")
    parser.add_argument("--mamba-model", default="state-spaces/mamba-130m-hf")
    parser.add_argument("--baseline-model", default="gpt2")
    parser.add_argument("--new-tokens", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path")
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    results = run_inference_benchmarks(args.mamba_model, args.baseline_model, args.new_tokens, device)
    payload = [asdict(result) for result in results]

    if args.output is not None:
        save_results(args.output, results)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
