"""Week 2 benchmark: our MambaModel (parallel scan) vs HuggingFace Mamba.

Runs both implementations on ``state-spaces/mamba-130m-hf`` and measures:

* Prefill (time-to-first-token)
* Decode tokens/sec across ``max_new_tokens`` new tokens
* Peak memory

Expect our pure-PyTorch parallel scan to be **roughly 10-50x slower** than
HF's path on a GPU (HF dispatches to the ``mamba_ssm`` CUDA kernels when
available) and **much closer** on CPU (where HF also falls back to a slow
PyTorch loop). The point of this benchmark isn't to win — it's to have a
number we can improve in Phase 3 (fused Triton decode kernel).

Example::

    python benchmarks/week2.py --device cuda --new-tokens 128 \\
        --output benchmarks/results/week2.gpu.json
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from mamba_minimal.generate import generate_native
from mamba_minimal.model import MambaModel


@dataclass(slots=True)
class RunResult:
    impl: str
    device: str
    dtype: str
    prompt_tokens: int
    new_tokens: int
    prefill_ms: float
    decode_ms: float
    total_ms: float
    decode_tok_per_s: float
    peak_memory_mb: float | None


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _reset_peak(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()


def _peak_mb(device: torch.device) -> float | None:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    return None


def bench_ours(
    model: MambaModel,
    prompt_ids: torch.Tensor,
    new_tokens: int,
    device: torch.device,
    warmup: int = 1,
) -> RunResult:
    # Warmup.
    for _ in range(warmup):
        _ = generate_native(model, prompt_ids, max_new_tokens=min(4, new_tokens))
    _sync(device)
    _reset_peak(device)

    # Prefill + first token timed separately.
    t0 = time.perf_counter()
    with torch.no_grad():
        _, state = model(prompt_ids, return_state=True)
    _sync(device)
    t_prefill = time.perf_counter() - t0

    # Generate the remaining tokens.
    last_logits = None
    t0 = time.perf_counter()
    with torch.no_grad():
        logits, state = model(prompt_ids, return_state=True)
        next_token = logits[:, -1, :].argmax(dim=-1)
        tokens_decoded = 1
        for _ in range(new_tokens - 1):
            logits, state = model.step(next_token, state)
            next_token = logits.argmax(dim=-1)
            tokens_decoded += 1
    _sync(device)
    t_total = time.perf_counter() - t0
    t_decode = t_total - t_prefill

    impl_label = "mamba_minimal.parallel+triton" if model.use_triton else "mamba_minimal.parallel"
    return RunResult(
        impl=impl_label,
        device=str(device),
        dtype=str(model.embeddings.weight.dtype).replace("torch.", ""),
        prompt_tokens=int(prompt_ids.shape[-1]),
        new_tokens=tokens_decoded,
        prefill_ms=t_prefill * 1e3,
        decode_ms=t_decode * 1e3,
        total_ms=t_total * 1e3,
        decode_tok_per_s=tokens_decoded / max(t_decode, 1e-9),
        peak_memory_mb=_peak_mb(device),
    )


def bench_hf(
    hf_model: Any,
    prompt_ids: torch.Tensor,
    new_tokens: int,
    device: torch.device,
    warmup: int = 1,
) -> RunResult:
    # Warmup.
    with torch.no_grad():
        _ = hf_model.generate(
            prompt_ids, max_new_tokens=min(4, new_tokens), do_sample=False
        )
    _sync(device)
    _reset_peak(device)

    # Prefill time (forward pass only, no generation).
    t0 = time.perf_counter()
    with torch.no_grad():
        _ = hf_model(prompt_ids)
    _sync(device)
    t_prefill = time.perf_counter() - t0

    # Full generate (HF internally uses cache when available).
    t0 = time.perf_counter()
    with torch.no_grad():
        out = hf_model.generate(prompt_ids, max_new_tokens=new_tokens, do_sample=False)
    _sync(device)
    t_total = time.perf_counter() - t0
    t_decode = max(t_total - t_prefill, 1e-9)

    tokens_decoded = int(out.shape[-1] - prompt_ids.shape[-1])
    return RunResult(
        impl="hf_mamba",
        device=str(device),
        dtype=str(next(hf_model.parameters()).dtype).replace("torch.", ""),
        prompt_tokens=int(prompt_ids.shape[-1]),
        new_tokens=tokens_decoded,
        prefill_ms=t_prefill * 1e3,
        decode_ms=t_decode * 1e3,
        total_ms=t_total * 1e3,
        decode_tok_per_s=tokens_decoded / t_decode,
        peak_memory_mb=_peak_mb(device),
    )


def _render_markdown(results: list[RunResult]) -> str:
    header = (
        "| Impl | Device | Dtype | Prompt | New | Prefill (ms) | "
        "Decode (ms) | Tok/s | Peak MB |"
    )
    sep = "|---|---|---|---:|---:|---:|---:|---:|---:|"
    rows = [header, sep]
    for r in results:
        mb = "-" if r.peak_memory_mb is None else f"{r.peak_memory_mb:.1f}"
        rows.append(
            f"| {r.impl} | {r.device} | {r.dtype} | {r.prompt_tokens} | "
            f"{r.new_tokens} | {r.prefill_ms:.1f} | {r.decode_ms:.1f} | "
            f"{r.decode_tok_per_s:.1f} | {mb} |"
        )
    return "\n".join(rows)


def main() -> None:
    p = argparse.ArgumentParser(description="Week 2 tok/s benchmark: ours vs HF Mamba-130m.")
    p.add_argument("--model", default="state-spaces/mamba-130m-hf")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    p.add_argument("--prompt", default="Mamba is useful because")
    p.add_argument("--prompt-tokens", type=int, default=8,
                   help="If >0, override the prompt to a random sequence of this length.")
    p.add_argument("--new-tokens", type=int, default=128)
    p.add_argument("--use-triton", action="store_true",
                   help="Route our .step() decode path through the Triton kernel.")
    p.add_argument("--output", type=str, default=None)
    args = p.parse_args()

    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto"
        else args.device
    )
    dtype = {"float32": torch.float32, "float16": torch.float16,
             "bfloat16": torch.bfloat16}[args.dtype]

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    if args.prompt_tokens > 0:
        torch.manual_seed(0)
        prompt_ids = torch.randint(
            0, tok.vocab_size, (1, args.prompt_tokens), device=device
        )
    else:
        prompt_ids = tok(args.prompt, return_tensors="pt").input_ids.to(device)

    print(f"Device: {device}  Dtype: {dtype}  Prompt tokens: {prompt_ids.shape[-1]}  "
          f"New tokens: {args.new_tokens}")

    hf_model = AutoModelForCausalLM.from_pretrained(args.model).to(device=device, dtype=dtype).eval()
    ours = MambaModel.from_pretrained(
        args.model, scan_impl="parallel",
        use_triton=args.use_triton,
        device=device, dtype=dtype,
    ).eval()

    results: list[RunResult] = []
    print("\n-- Ours (parallel scan, state-carrying decode) --")
    r = bench_ours(ours, prompt_ids, args.new_tokens, device)
    print(asdict(r))
    results.append(r)

    print("\n-- HF Mamba (transformers.generate) --")
    r = bench_hf(hf_model, prompt_ids, args.new_tokens, device)
    print(asdict(r))
    results.append(r)

    print("\n" + _render_markdown(results))

    ours_tokps = results[0].decode_tok_per_s
    hf_tokps = results[1].decode_tok_per_s
    if ours_tokps > 0:
        print(f"\nHF / ours tok/s ratio: {hf_tokps / ours_tokps:.2f}x")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({
            "config": vars(args),
            "results": [asdict(r) for r in results],
        }, indent=2))
        print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
