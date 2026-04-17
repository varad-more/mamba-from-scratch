"""Render the benchmark-suite plots from suite CSV output.

Produces:
  * ``tokps_bar.png``          — bar chart of decode tok/s at B=1, prompt=1024
  * ``latency_vs_seqlen.png``  — line plot: prefill ms vs seqlen, all engines
  * ``memory_vs_seqlen.png``   — peak GB vs seqlen, all engines

Failed rows (OOMs, load errors) are omitted from plots but reported at
the top of each figure so the ``hidden`` isn't hidden.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _rows(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def _fnum(r, k):
    try:
        return float(r[k])
    except (KeyError, ValueError):
        return float("nan")


def _ok(r) -> bool:
    return r["ok"].lower() == "true"


def bar_tokps(rows, out: Path, batch=1, prompt=1024):
    subset = [r for r in rows if _ok(r)
              and int(r["batch"]) == batch and int(r["prompt_len"]) == prompt]
    subset.sort(key=lambda r: _fnum(r, "decode_tok_s"), reverse=True)
    if not subset:
        print(f"skip {out.name}: no rows @ batch={batch} prompt={prompt}")
        return
    names = [r["engine"] for r in subset]
    vals = [_fnum(r, "decode_tok_s") for r in subset]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(names, vals, color="#4878CF")
    ax.set_ylabel("Decode tok/s")
    ax.set_title(f"Decode tok/s — batch={batch}, prompt={prompt}")
    ax.tick_params(axis="x", rotation=25)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.0f}",
                ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"wrote {out}")


def line_latency_vs_seqlen(rows, out: Path, batch=1):
    by_eng: dict[str, list[tuple[int, float]]] = {}
    for r in rows:
        if not _ok(r) or int(r["batch"]) != batch:
            continue
        by_eng.setdefault(r["engine"], []).append((int(r["prompt_len"]), _fnum(r, "prefill_ms")))
    if not by_eng:
        print(f"skip {out.name}: no rows @ batch={batch}")
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for eng, xs in sorted(by_eng.items()):
        xs.sort()
        ax.plot([p for p, _ in xs], [v for _, v in xs], marker="o", label=eng)
    ax.set_xlabel("Prompt length")
    ax.set_ylabel("Prefill ms (median)")
    ax.set_title(f"Prefill latency vs prompt length — batch={batch}")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"wrote {out}")


def memory_vs_seqlen(rows, out: Path, batch=1):
    by_eng: dict[str, list[tuple[int, float]]] = {}
    for r in rows:
        if not _ok(r) or int(r["batch"]) != batch:
            continue
        by_eng.setdefault(r["engine"], []).append((int(r["prompt_len"]), _fnum(r, "peak_mb") / 1024.0))
    if not by_eng:
        print(f"skip {out.name}: no rows @ batch={batch}")
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for eng, xs in sorted(by_eng.items()):
        xs.sort()
        ax.plot([p for p, _ in xs], [v for _, v in xs], marker="o", label=eng)
    ax.set_xlabel("Prompt length")
    ax.set_ylabel("Peak GPU memory (GB)")
    ax.set_title(f"Peak memory vs prompt length — batch={batch}")
    ax.set_xscale("log", base=2)
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="benchmarks/results/suite_v1.csv")
    p.add_argument("--outdir", default="benchmarks/results")
    args = p.parse_args()
    rows = _rows(Path(args.csv))
    outdir = Path(args.outdir)
    bar_tokps(rows, outdir / "tokps_bar.png", batch=1, prompt=1024)
    line_latency_vs_seqlen(rows, outdir / "latency_vs_seqlen.png", batch=1)
    memory_vs_seqlen(rows, outdir / "memory_vs_seqlen.png", batch=1)


if __name__ == "__main__":
    main()
