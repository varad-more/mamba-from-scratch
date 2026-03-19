from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmarks" / "results"
FIGURES = ROOT / "figures"


def load_json(path: Path):
    return json.loads(path.read_text())


def render_scan_results(path: Path, out_path: Path) -> None:
    rows = load_json(path)
    names = [row["name"] for row in rows]
    p50 = [row["latency_ms_p50"] for row in rows]
    bandwidth = [row["achieved_bandwidth_gb_s"] for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(names, p50, color=["#4C78A8", "#F58518", "#54A24B"])
    axes[0].set_title("Selective scan latency (p50)")
    axes[0].set_ylabel("Latency (ms)")

    axes[1].bar(names, bandwidth, color=["#4C78A8", "#F58518", "#54A24B"])
    axes[1].set_title("Estimated achieved bandwidth")
    axes[1].set_ylabel("GB/s")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)


def render_inference_results(path: Path, out_path: Path) -> None:
    rows = load_json(path)
    grouped: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        model = row["model_name"]
        stats = grouped.setdefault(model, {"prompt_tokens": [], "ttft_ms": [], "peak_memory_mb": []})
        stats["prompt_tokens"].append(row["prompt_tokens"])
        stats["ttft_ms"].append(row["ttft_ms"])
        stats["peak_memory_mb"].append(row["peak_memory_mb"])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for model, stats in grouped.items():
        axes[0].plot(stats["prompt_tokens"], stats["ttft_ms"], marker="o", label=model)
        axes[1].plot(stats["prompt_tokens"], stats["peak_memory_mb"], marker="o", label=model)

    axes[0].set_title("TTFT vs prompt length")
    axes[0].set_xlabel("Prompt tokens")
    axes[0].set_ylabel("TTFT (ms)")
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Peak memory vs prompt length")
    axes[1].set_xlabel("Prompt tokens")
    axes[1].set_ylabel("Peak memory (MB)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)

    scan_path = RESULTS / "scan_results.cpu.json"
    if scan_path.exists():
        render_scan_results(scan_path, FIGURES / "scan_benchmark_cpu.png")

    inference_path = RESULTS / "inference_results.cpu.json"
    if inference_path.exists():
        render_inference_results(inference_path, FIGURES / "inference_comparison_cpu.png")

    print("Rendered benchmark figures to", FIGURES)


if __name__ == "__main__":
    main()
