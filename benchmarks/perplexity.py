"""Perplexity benchmark — quality parity between our MambaModel and the HF reference.

Computes token-level perplexity on WikiText-2 (raw, test split) through two
code paths over the *same* token windows:

  * ``minimamba`` — our ``MambaModel`` (HF checkpoint loaded by our weight
    loader, scan dispatched by our backend policy)
  * ``hf``        — the unpatched ``transformers`` ``MambaForCausalLM``

Both runs share the tokenizer, windows, and dtype, so any perplexity gap is
implementation drift, not data noise. At fp32 the two should agree to several
decimal places — this extends the logit-parity ladder to a corpus-level
quality metric.

Protocol: the test split is joined with ``"\\n\\n"``, tokenized once, and cut
into non-overlapping windows of ``--window`` tokens. Each window predicts
tokens ``1..W-1`` from its own prefix (no cross-window state). Perplexity is
``exp(total_nll / total_predicted_tokens)``.

Example:

    PYTHONPATH=. python benchmarks/perplexity.py \
        --model state-spaces/mamba-130m-hf \
        --window 1024 --max-tokens 65536 --dtype float32 \
        --output benchmarks/results/perplexity.json
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import torch
import torch.nn.functional as F

try:
    from transformers import AutoTokenizer, MambaForCausalLM
except Exception as exc:  # pragma: no cover - optional dependency
    AutoTokenizer = None
    MambaForCausalLM = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


@dataclass(slots=True)
class PerplexityResult:
    engine: str
    model_name: str
    dataset: str
    device: str
    machine: str
    dtype: str
    window: int
    n_windows: int
    total_tokens: int
    predicted_tokens: int
    nll_sum: float
    perplexity: float
    elapsed_s: float


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def machine_summary(device: str) -> str:
    if device.startswith("cuda"):
        return torch.cuda.get_device_name(0)
    return f"{platform.system()} {platform.release()} ({platform.machine()})"


def load_token_windows(
    tokenizer, dataset_name: str, window: int, max_tokens: int | None
) -> torch.Tensor:
    """Tokenize the corpus once and cut it into non-overlapping windows."""

    from datasets import load_dataset

    if dataset_name != "wikitext-2-raw-v1":
        raise ValueError(f"Unsupported dataset {dataset_name!r}")
    raw = load_dataset("Salesforce/wikitext", dataset_name, split="test")
    text = "\n\n".join(row["text"] for row in raw)
    ids = tokenizer(text, return_tensors="pt").input_ids[0]
    if max_tokens is not None:
        ids = ids[:max_tokens]
    n_windows = ids.numel() // window
    if n_windows == 0:
        raise ValueError(
            f"Corpus has {ids.numel()} tokens, fewer than one window of {window}"
        )
    return ids[: n_windows * window].view(n_windows, window)


@torch.no_grad()
def windowed_nll(forward, windows: torch.Tensor, device: str, label: str) -> tuple[float, int]:
    """Sum next-token NLL over all windows. ``forward(ids) -> (1, L, V)`` logits."""

    nll_sum = 0.0
    predicted = 0
    for i, window_ids in enumerate(windows):
        ids = window_ids.unsqueeze(0).to(device)
        logits = forward(ids).float()
        targets = ids[:, 1:]
        log_probs = F.log_softmax(logits[:, :-1], dim=-1)
        token_nll = -log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        nll_sum += token_nll.sum().item()
        predicted += targets.numel()
        print(
            f"  [{label}] window {i + 1}/{len(windows)} "
            f"running ppl={math.exp(nll_sum / predicted):.4f}",
            flush=True,
        )
    return nll_sum, predicted


def run_engine(
    engine: str,
    model_name: str,
    windows: torch.Tensor,
    device: str,
    dtype: torch.dtype,
    args: argparse.Namespace,
) -> PerplexityResult:
    if engine == "minimamba":
        from mamba_minimal.model import MambaModel

        model = MambaModel.from_pretrained(model_name, device=device, dtype=dtype)
        forward = model.forward
    elif engine == "hf":
        hf_model = MambaForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
        hf_model = hf_model.to(device).eval()

        def forward(ids):
            return hf_model(ids).logits
    else:
        raise ValueError(f"Unknown engine {engine!r}")

    start = time.perf_counter()
    nll_sum, predicted = windowed_nll(forward, windows, device, engine)
    elapsed = time.perf_counter() - start

    return PerplexityResult(
        engine=engine,
        model_name=model_name,
        dataset=args.dataset,
        device=device,
        machine=machine_summary(device),
        dtype=str(dtype).removeprefix("torch."),
        window=args.window,
        n_windows=len(windows),
        total_tokens=windows.numel(),
        predicted_tokens=predicted,
        nll_sum=nll_sum,
        perplexity=math.exp(nll_sum / predicted),
        elapsed_s=elapsed,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="state-spaces/mamba-130m-hf")
    parser.add_argument("--dataset", default="wikitext-2-raw-v1")
    parser.add_argument("--engines", nargs="+", default=["minimamba", "hf"],
                        choices=["minimamba", "hf"])
    parser.add_argument("--window", type=int, default=1024)
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="Cap the corpus at this many tokens (default: full test split)")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16"])
    parser.add_argument("--output", default=None, help="Write results JSON here")
    args = parser.parse_args()

    if IMPORT_ERROR is not None:
        raise SystemExit(f"transformers is required: {IMPORT_ERROR}")

    device = resolve_device(args.device)
    dtype = getattr(torch, args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    windows = load_token_windows(tokenizer, args.dataset, args.window, args.max_tokens)
    print(
        f"{args.dataset}: {windows.numel()} tokens → {len(windows)} windows "
        f"of {args.window} on {device} ({args.dtype})"
    )

    results = [
        run_engine(engine, args.model, windows, device, dtype, args)
        for engine in args.engines
    ]

    for r in results:
        print(f"{r.engine:>10}: ppl={r.perplexity:.4f}  ({r.elapsed_s:.1f}s)")
    if len(results) == 2:
        delta = abs(results[0].perplexity - results[1].perplexity)
        print(f"  |Δppl| = {delta:.6f}")

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps([asdict(r) for r in results], indent=2) + "\n")
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
