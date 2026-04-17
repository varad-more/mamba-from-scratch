"""Week-5 benchmark suite — cross-engine inference harness.

Engines under test:
  * ``minimamba-mamba1``   — our Mamba-1 via parallel scan (+ Triton decode)
  * ``minimamba-mamba2``   — our Mamba-2 via SSD
  * ``mambassm-mamba1``    — ``state-spaces/mamba-130m-hf`` through HF transformers
  * ``mambassm-mamba2``    — ``state-spaces/mamba2-130m`` through ``mamba_ssm``
  * ``hf-pythia-160m``     — dense transformer baseline (same class as 130m mambas)
  * ``hf-pythia-2.8b``     — big-dense-transformer ceiling

Metrics per run (median of N):
  * ``prefill_ms``        — time for the forward over the prompt
  * ``ttft_ms``           — prefill + first decoded token (time-to-first-token)
  * ``decode_ms_per_tok`` — median per-step decode latency
  * ``decode_tok_s``      — 1000 / decode_ms_per_tok
  * ``peak_mb``           — ``torch.cuda.max_memory_allocated``
  * ``params_m``          — model parameter count (M)
  * ``bytes_per_param``   — ``peak_mb * 1e6 / params`` (rough memory efficiency)

Harness pattern is borrowed from a prior inference-benchmark system:
one warmup, median-of-N, CUDA-event timing for the hot loop. OOMs are
caught per-config and recorded rather than crashing the suite.

Default matrix is deliberately small (one batch × two seqlens × one
gen) to keep a full run under ~10 min on an A10G. Use the flags to
expand.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import statistics
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch


# ----------------------------------------------------------------------
# Timer
# ----------------------------------------------------------------------
class Timer:
    """CUDA-event timer with wall-clock fallback for CPU."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self._cuda = device.type == "cuda"
        if self._cuda:
            self._start = torch.cuda.Event(enable_timing=True)
            self._end = torch.cuda.Event(enable_timing=True)

    def __enter__(self):
        if self._cuda:
            torch.cuda.synchronize()
            self._start.record()
        else:
            self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if self._cuda:
            self._end.record()
            torch.cuda.synchronize()
            self.elapsed_ms = self._start.elapsed_time(self._end)
        else:
            self.elapsed_ms = (time.perf_counter() - self._t0) * 1e3


def median_time(fn: Callable[[], None], iters: int, warmups: int, device) -> float:
    for _ in range(warmups):
        fn()
    samples = []
    for _ in range(iters):
        with Timer(device) as t:
            fn()
        samples.append(t.elapsed_ms)
    return statistics.median(samples)


# ----------------------------------------------------------------------
# Runners (one per engine, uniform interface)
# ----------------------------------------------------------------------
@dataclass
class RunResult:
    engine: str
    model: str
    batch: int
    prompt_len: int
    gen_len: int
    dtype: str
    params_m: float = 0.0
    prefill_ms: float = float("nan")
    ttft_ms: float = float("nan")
    decode_ms_per_tok: float = float("nan")
    decode_tok_s: float = float("nan")
    peak_mb: float = float("nan")
    bytes_per_param: float = float("nan")
    ok: bool = True
    error: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


class Runner:
    engine: str = "?"
    model_name: str = "?"

    def __init__(self, device: torch.device, dtype: torch.dtype):
        self.device = device
        self.dtype = dtype

    def load(self) -> None: ...
    def unload(self) -> None:
        if hasattr(self, "model"):
            del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def param_count(self) -> int:
        return sum(p.numel() for p in self.model.parameters())  # type: ignore[attr-defined]

    def prefill(self, ids: torch.Tensor):
        raise NotImplementedError

    def decode_n(self, ids: torch.Tensor, n: int) -> tuple[float, int]:
        """Decode ``n`` tokens after ``ids``. Return (total_decode_ms, actually_decoded)."""
        raise NotImplementedError

    def ttft(self, ids: torch.Tensor) -> float:
        """Prefill + 1 decoded token, end-to-end."""
        raise NotImplementedError


class MinimambaRunner(Runner):
    def __init__(self, arch: str, model_name: str, device, dtype, use_triton: bool = False):
        super().__init__(device, dtype)
        self.engine = f"minimamba-{arch}"
        self.model_name = model_name
        self.arch = arch
        self.use_triton = use_triton

    def load(self):
        from mamba_minimal.model import load_model
        kwargs = {"use_triton": self.use_triton} if self.arch == "mamba1" else {}
        self.model = load_model(
            self.model_name, arch=self.arch, device=self.device, dtype=self.dtype, **kwargs,
        )

    @torch.no_grad()
    def prefill(self, ids):
        return self.model(ids, return_state=True)

    @torch.no_grad()
    def decode_n(self, ids, n):
        logits, state = self.model(ids, return_state=True)
        tok = logits[:, -1, :].argmax(-1)
        with Timer(self.device) as t:
            for _ in range(n):
                logits, state = self.model.step(tok, state)
                tok = logits.argmax(-1)
        return t.elapsed_ms, n

    @torch.no_grad()
    def ttft(self, ids):
        with Timer(self.device) as t:
            logits, state = self.model(ids, return_state=True)
            _ = logits[:, -1, :].argmax(-1)
        return t.elapsed_ms


class MambassmMamba2Runner(Runner):
    def __init__(self, model_name: str, device, dtype):
        super().__init__(device, dtype)
        self.engine = "mambassm-mamba2"
        self.model_name = model_name

    def load(self):
        from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
        self.model = MambaLMHeadModel.from_pretrained(
            self.model_name, device=self.device, dtype=self.dtype,
        ).eval()

    @torch.no_grad()
    def prefill(self, ids):
        return self.model(ids)

    @torch.no_grad()
    def decode_n(self, ids, n):
        with Timer(self.device) as t:
            out = self.model.generate(ids, max_length=ids.shape[1] + n)
        decoded = out.shape[1] - ids.shape[1]
        return t.elapsed_ms, decoded

    @torch.no_grad()
    def ttft(self, ids):
        with Timer(self.device) as t:
            _ = self.model.generate(ids, max_length=ids.shape[1] + 1)
        return t.elapsed_ms


class HFRunner(Runner):
    """HF Transformers causal LM — works for Pythia and HF Mamba-1."""

    def __init__(self, engine: str, model_name: str, device, dtype):
        super().__init__(device, dtype)
        self.engine = engine
        self.model_name = model_name

    def load(self):
        from transformers import AutoModelForCausalLM
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name).to(
            device=self.device, dtype=self.dtype,
        ).eval()

    @torch.no_grad()
    def prefill(self, ids):
        return self.model(ids)

    @torch.no_grad()
    def decode_n(self, ids, n):
        with Timer(self.device) as t:
            out = self.model.generate(
                ids, max_new_tokens=n, do_sample=False,
                pad_token_id=0, use_cache=True,
            )
        decoded = out.shape[1] - ids.shape[1]
        return t.elapsed_ms, decoded

    @torch.no_grad()
    def ttft(self, ids):
        with Timer(self.device) as t:
            _ = self.model.generate(
                ids, max_new_tokens=1, do_sample=False,
                pad_token_id=0, use_cache=True,
            )
        return t.elapsed_ms


ENGINE_REGISTRY: dict[str, Callable[[torch.device, torch.dtype, bool], Runner]] = {
    "minimamba-mamba1-130m": lambda d, t, tri: MinimambaRunner(
        "mamba1", "state-spaces/mamba-130m-hf", d, t, use_triton=tri
    ),
    "minimamba-mamba2-130m": lambda d, t, tri: MinimambaRunner(
        "mamba2", "state-spaces/mamba2-130m", d, t
    ),
    "mambassm-mamba1-130m": lambda d, t, tri: HFRunner(
        "mambassm-mamba1", "state-spaces/mamba-130m-hf", d, t
    ),
    "mambassm-mamba2-130m": lambda d, t, tri: MambassmMamba2Runner(
        "state-spaces/mamba2-130m", d, t
    ),
    "pythia-160m": lambda d, t, tri: HFRunner("hf-pythia-160m", "EleutherAI/pythia-160m", d, t),
    "pythia-2.8b": lambda d, t, tri: HFRunner("hf-pythia-2.8b", "EleutherAI/pythia-2.8b", d, t),
}


# ----------------------------------------------------------------------
# Run a single config
# ----------------------------------------------------------------------
def run_one(
    runner: Runner, batch: int, prompt_len: int, gen_len: int,
    iters: int, warmups: int, vocab: int = 50270,
) -> RunResult:
    r = RunResult(
        engine=runner.engine, model=runner.model_name,
        batch=batch, prompt_len=prompt_len, gen_len=gen_len,
        dtype=str(runner.dtype).replace("torch.", ""),
    )
    try:
        torch.cuda.reset_peak_memory_stats(runner.device) if runner.device.type == "cuda" else None
        ids = torch.randint(0, vocab, (batch, prompt_len), device=runner.device)

        r.params_m = runner.param_count() / 1e6
        # Prefill
        r.prefill_ms = median_time(lambda: runner.prefill(ids), iters, warmups, runner.device)
        # TTFT
        r.ttft_ms = median_time(lambda: runner.ttft(ids), max(1, iters // 2), 1, runner.device)
        # Decode
        dec_ms, dec_n = runner.decode_n(ids, gen_len)
        r.decode_ms_per_tok = dec_ms / max(dec_n, 1)
        r.decode_tok_s = 1000.0 / max(r.decode_ms_per_tok, 1e-9)

        if runner.device.type == "cuda":
            r.peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            r.bytes_per_param = r.peak_mb * 1e6 / max(r.params_m * 1e6, 1)
    except torch.cuda.OutOfMemoryError as e:
        r.ok = False
        r.error = f"OOM: {str(e)[:200]}"
        torch.cuda.empty_cache()
    except Exception as e:
        r.ok = False
        r.error = f"{type(e).__name__}: {str(e)[:200]}"
        traceback.print_exc()
    return r


# ----------------------------------------------------------------------
# CLI / matrix
# ----------------------------------------------------------------------
def build_matrix(args) -> list[tuple[str, int, int, int]]:
    out = []
    for eng in args.engines:
        for b in args.batches:
            for pl in args.prompt_lens:
                for gl in args.gen_lens:
                    out.append((eng, b, pl, gl))
    return out


def write_csv(path: Path, rows: list[RunResult]) -> None:
    if not rows:
        return
    fields = [k for k in asdict(rows[0]).keys() if k != "extra"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            d = asdict(r)
            d.pop("extra", None)
            w.writerow(d)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="float16", choices=["float32", "float16", "bfloat16"])
    p.add_argument("--engines", nargs="+",
                   default=["minimamba-mamba1-130m", "minimamba-mamba2-130m",
                            "mambassm-mamba1-130m", "mambassm-mamba2-130m",
                            "pythia-160m"])
    p.add_argument("--batches", nargs="+", type=int, default=[1])
    p.add_argument("--prompt-lens", nargs="+", type=int, default=[128, 1024])
    p.add_argument("--gen-lens", nargs="+", type=int, default=[128])
    p.add_argument("--iters", type=int, default=3)
    p.add_argument("--warmups", type=int, default=1)
    p.add_argument("--use-triton", action="store_true")
    p.add_argument("--outdir", default="benchmarks/results")
    p.add_argument("--tag", default="suite")
    args = p.parse_args()

    device = torch.device(args.device)
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]

    matrix = build_matrix(args)
    print(f"{len(matrix)} configs to run. dtype={dtype} device={device}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    results: list[RunResult] = []
    # Group by engine: load once, run all (batch, pl, gl) for that engine, then unload.
    by_engine: dict[str, list[tuple[int, int, int]]] = {}
    for eng, b, pl, gl in matrix:
        by_engine.setdefault(eng, []).append((b, pl, gl))

    for eng, configs in by_engine.items():
        print(f"\n=== {eng} ({len(configs)} configs) ===")
        try:
            runner = ENGINE_REGISTRY[eng](device, dtype, args.use_triton)
            runner.load()
        except Exception as e:
            print(f"load failed: {e}")
            for b, pl, gl in configs:
                results.append(RunResult(
                    engine=eng, model="?", batch=b, prompt_len=pl, gen_len=gl,
                    dtype=str(dtype), ok=False, error=f"load failed: {type(e).__name__}: {e}"[:300],
                ))
            continue
        try:
            for b, pl, gl in configs:
                print(f"  [B={b}, pl={pl}, gl={gl}]", end=" ", flush=True)
                r = run_one(runner, b, pl, gl, args.iters, args.warmups)
                results.append(r)
                if r.ok:
                    print(f"prefill={r.prefill_ms:.1f}ms  ttft={r.ttft_ms:.1f}ms  "
                          f"decode={r.decode_tok_s:.1f} tok/s  peak={r.peak_mb:.0f}MB")
                else:
                    print(f"FAIL: {r.error}")
        finally:
            runner.unload()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Dump.
    csv_path = outdir / f"{args.tag}.csv"
    json_path = outdir / f"{args.tag}.json"
    write_csv(csv_path, results)
    json_path.write_text(json.dumps(
        {"config": vars(args), "rows": [asdict(r) for r in results]}, indent=2,
    ))
    print(f"\nWrote {csv_path}\nWrote {json_path}")


if __name__ == "__main__":
    main()
