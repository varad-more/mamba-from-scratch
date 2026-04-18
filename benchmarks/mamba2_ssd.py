"""Mamba-2 SSD benchmark — Mamba-1 vs Mamba-2, ours vs ``mamba_ssm``.

Four configurations, same prompt, same ``new_tokens``:

  * minimamba Mamba-1 (parallel scan + optional Triton decode)
  * minimamba Mamba-2 (our SSD prefill + pure-PyTorch decode step)
  * mamba_ssm Mamba-1 reference (HF ``state-spaces/mamba-130m-hf``)
  * mamba_ssm Mamba-2 reference (``state-spaces/mamba2-130m``)

The Mamba-2 win (when it shows up) is: SSD turns the scan into a GEMM,
so even our pure-einsum prefill should scale better with seqlen than the
Mamba-1 parallel scan. Decode is Python-bound for both of ours — the
reference will dominate decode tok/s because it uses fused CUDA.
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


@dataclass(slots=True)
class RunResult:
    impl: str
    arch: str
    prompt_tokens: int
    new_tokens: int
    prefill_ms: float
    decode_ms: float
    decode_tok_per_s: float
    peak_mb: float


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def _reset(device):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()


def _peak(device):
    return torch.cuda.max_memory_allocated() / (1024 * 1024) if device.type == "cuda" else 0.0


@torch.no_grad()
def bench_ours(model, arch: str, prompt_ids: torch.Tensor, new_tokens: int, device) -> RunResult:
    # Warmup.
    _ = generate_native(model, prompt_ids, max_new_tokens=min(4, new_tokens))
    _sync(device)
    _reset(device)

    t0 = time.perf_counter()
    _, state = model(prompt_ids, return_state=True)
    _sync(device)
    t_prefill = time.perf_counter() - t0

    t0 = time.perf_counter()
    logits, state = model(prompt_ids, return_state=True)
    next_token = logits[:, -1, :].argmax(dim=-1)
    decoded = 1
    for _ in range(new_tokens - 1):
        logits, state = model.step(next_token, state)
        next_token = logits.argmax(dim=-1)
        decoded += 1
    _sync(device)
    t_total = time.perf_counter() - t0
    t_decode = max(t_total - t_prefill, 1e-9)
    return RunResult(
        impl="minimamba", arch=arch,
        prompt_tokens=int(prompt_ids.shape[-1]),
        new_tokens=decoded,
        prefill_ms=t_prefill * 1e3,
        decode_ms=t_decode * 1e3,
        decode_tok_per_s=decoded / t_decode,
        peak_mb=_peak(device),
    )


@torch.no_grad()
def bench_reference(model: Any, arch: str, prompt_ids: torch.Tensor, new_tokens: int, device) -> RunResult:
    """Reference is either HF transformers Mamba-1 or mamba_ssm MambaLMHeadModel (Mamba-2)."""
    if arch == "mamba2":
        # mamba_ssm MambaLMHeadModel: prefill = forward; generate = .generate(max_length=...)
        _ = model.generate(prompt_ids, max_length=prompt_ids.shape[1] + min(4, new_tokens))
        _sync(device)
        _reset(device)
        t0 = time.perf_counter()
        _ = model(prompt_ids)
        _sync(device)
        t_prefill = time.perf_counter() - t0
        t0 = time.perf_counter()
        out = model.generate(prompt_ids, max_length=prompt_ids.shape[1] + new_tokens)
        _sync(device)
        t_total = time.perf_counter() - t0
        t_decode = max(t_total - t_prefill, 1e-9)
        decoded = out.shape[1] - prompt_ids.shape[1]
    else:
        # HF transformers Mamba-1.
        _ = model.generate(prompt_ids, max_new_tokens=min(4, new_tokens), do_sample=False)
        _sync(device)
        _reset(device)
        t0 = time.perf_counter()
        _ = model(prompt_ids)
        _sync(device)
        t_prefill = time.perf_counter() - t0
        t0 = time.perf_counter()
        out = model.generate(prompt_ids, max_new_tokens=new_tokens, do_sample=False)
        _sync(device)
        t_total = time.perf_counter() - t0
        t_decode = max(t_total - t_prefill, 1e-9)
        decoded = out.shape[1] - prompt_ids.shape[1]
    return RunResult(
        impl="mamba_ssm_ref" if arch == "mamba2" else "hf_mamba",
        arch=arch,
        prompt_tokens=int(prompt_ids.shape[-1]),
        new_tokens=int(decoded),
        prefill_ms=t_prefill * 1e3,
        decode_ms=t_decode * 1e3,
        decode_tok_per_s=decoded / t_decode,
        peak_mb=_peak(device),
    )


def _markdown(rows: list[RunResult]) -> str:
    head = "| Impl | Arch | Prompt | New | Prefill ms | Decode ms | Tok/s | Peak MB |"
    sep = "|---|---|---:|---:|---:|---:|---:|---:|"
    out = [head, sep]
    for r in rows:
        out.append(
            f"| {r.impl} | {r.arch} | {r.prompt_tokens} | {r.new_tokens} | "
            f"{r.prefill_ms:.1f} | {r.decode_ms:.1f} | {r.decode_tok_per_s:.1f} | {r.peak_mb:.1f} |"
        )
    return "\n".join(out)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    p.add_argument("--prompt-tokens", type=int, default=128)
    p.add_argument("--new-tokens", type=int, default=64)
    p.add_argument("--use-triton", action="store_true", help="Route our Mamba-1 .step() through Triton.")
    p.add_argument("--output", default="benchmarks/results/mamba2_ssd.gpu.json")
    args = p.parse_args()

    device = torch.device(args.device)
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    torch.manual_seed(0)
    prompt_ids = torch.randint(0, 50270, (1, args.prompt_tokens), device=device)

    from transformers import AutoModelForCausalLM
    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

    from mamba_minimal.model import load_model

    print(f"Device {device} dtype {dtype} prompt {args.prompt_tokens} new {args.new_tokens}")

    rows: list[RunResult] = []

    # ---- minimamba Mamba-1 ----
    print("\n[1/4] minimamba Mamba-1")
    m1 = load_model("state-spaces/mamba-130m-hf", arch="mamba1",
                    device=device, dtype=dtype, use_triton=args.use_triton)
    rows.append(bench_ours(m1, "mamba1", prompt_ids, args.new_tokens, device))
    print(asdict(rows[-1]))
    del m1
    torch.cuda.empty_cache()

    # ---- minimamba Mamba-2 ----
    print("\n[2/4] minimamba Mamba-2")
    m2 = load_model("state-spaces/mamba2-130m", arch="mamba2", device=device, dtype=dtype)
    rows.append(bench_ours(m2, "mamba2", prompt_ids, args.new_tokens, device))
    print(asdict(rows[-1]))
    del m2
    torch.cuda.empty_cache()

    # ---- HF Mamba-1 ----
    print("\n[3/4] HF Mamba-1")
    r1 = AutoModelForCausalLM.from_pretrained("state-spaces/mamba-130m-hf").to(device=device, dtype=dtype).eval()
    rows.append(bench_reference(r1, "mamba1", prompt_ids, args.new_tokens, device))
    print(asdict(rows[-1]))
    del r1
    torch.cuda.empty_cache()

    # ---- mamba_ssm Mamba-2 ----
    print("\n[4/4] mamba_ssm Mamba-2")
    r2 = MambaLMHeadModel.from_pretrained("state-spaces/mamba2-130m", device=device, dtype=dtype).eval()
    rows.append(bench_reference(r2, "mamba2", prompt_ids, args.new_tokens, device))
    print(asdict(rows[-1]))

    print("\n" + _markdown(rows))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"config": vars(args), "results": [asdict(r) for r in rows]}, indent=2,
    ))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
