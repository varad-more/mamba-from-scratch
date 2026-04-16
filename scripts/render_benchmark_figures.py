from __future__ import annotations

import argparse
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

    # Check if multi-shape grid (multiple distinct lengths)
    lengths = sorted(set(row["length"] for row in rows))
    backends = sorted(set(row["name"] for row in rows))
    has_multi = len(lengths) > 1

    if has_multi:
        # Group by (name, batch) across lengths — pick batch=1 for the length sweep
        batch_one = [r for r in rows if r["batch"] == 1]
        if not batch_one:
            batch_one = rows

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        colors = {"reference": "#4C78A8", "naive": "#F58518", "fused": "#54A24B"}
        for name in backends:
            subset = sorted([r for r in batch_one if r["name"] == name], key=lambda r: r["length"])
            if not subset:
                continue
            ls = [r["length"] for r in subset]
            p50 = [r["latency_ms_p50"] for r in subset]
            bw = [r["achieved_bandwidth_gb_s"] for r in subset]
            c = colors.get(name, "#999999")
            axes[0].plot(ls, p50, marker="o", label=name, color=c)
            axes[1].plot(ls, bw, marker="o", label=name, color=c)

        axes[0].set_title("Selective scan latency vs sequence length (B=1)")
        axes[0].set_xlabel("Sequence length")
        axes[0].set_ylabel("Latency p50 (ms)")
        axes[0].set_yscale("log")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        axes[1].set_title("Achieved bandwidth vs sequence length (B=1)")
        axes[1].set_xlabel("Sequence length")
        axes[1].set_ylabel("GB/s")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
    else:
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
        stats = grouped.setdefault(
            model,
            {
                "prompt_tokens": [],
                "ttft_ms": [],
                "peak_memory_mb": [],
                "throughput_tokens_per_s": [],
            },
        )
        stats["prompt_tokens"].append(row["prompt_tokens"])
        stats["ttft_ms"].append(row["ttft_ms"])
        stats["peak_memory_mb"].append(row["peak_memory_mb"])
        stats["throughput_tokens_per_s"].append(row.get("throughput_tokens_per_s", 0.0))

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


def render_memory_scaling(path: Path, out_path: Path) -> None:
    rows = load_json(path)
    grouped: dict[str, tuple[list[float], list[float]]] = {}
    for row in rows:
        model_key = row.get("model_name") or row.get("model", "unknown")
        prompt_key = row.get("prompt_tokens") or row.get("actual_prompt_tokens", 0)
        xs, ys = grouped.setdefault(model_key, ([], []))
        xs.append(prompt_key)
        ys.append(row["peak_memory_mb"])

    plt.figure(figsize=(6.5, 4.5))
    for model, (xs, ys) in grouped.items():
        pairs = sorted(zip(xs, ys))
        plt.plot([x for x, _ in pairs], [y for _, y in pairs], marker="o", label=model)

    plt.title("Peak memory vs prompt length")
    plt.xlabel("Prompt tokens")
    plt.ylabel("Peak memory (MB)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)


def render_throughput_comparison(path: Path, out_path: Path) -> None:
    rows = load_json(path)
    grouped: dict[str, tuple[list[float], list[float]]] = {}
    for row in rows:
        xs, ys = grouped.setdefault(row["model_name"], ([], []))
        xs.append(row["prompt_tokens"])
        ys.append(row.get("throughput_tokens_per_s", 0.0))

    plt.figure(figsize=(6.5, 4.5))
    for model, (xs, ys) in grouped.items():
        pairs = sorted(zip(xs, ys))
        plt.plot([x for x, _ in pairs], [y for _, y in pairs], marker="o", label=model)

    plt.title("Decode throughput vs prompt length")
    plt.xlabel("Prompt tokens")
    plt.ylabel("Throughput (tokens/s)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)


def render_default_results(figures_dir: Path) -> None:
    scan_path = RESULTS / "scan_results.cpu.json"
    if scan_path.exists():
        render_scan_results(scan_path, figures_dir / "scan_benchmark_cpu.png")

    inference_path = RESULTS / "inference_results.cpu.json"
    if inference_path.exists():
        render_inference_results(inference_path, figures_dir / "inference_comparison_cpu.png")
        render_memory_scaling(inference_path, figures_dir / "memory_scaling.png")
        render_throughput_comparison(inference_path, figures_dir / "throughput_comparison.png")

    gpu_scan = RESULTS / "scan_results.gpu.json"
    if gpu_scan.exists():
        render_scan_results(gpu_scan, figures_dir / "scan_benchmark_gpu.png")

    gpu_inference = RESULTS / "inference_results.gpu.json"
    if gpu_inference.exists():
        render_inference_results(gpu_inference, figures_dir / "inference_comparison_gpu.png")
        render_throughput_comparison(gpu_inference, figures_dir / "throughput_comparison.gpu.png")

    gpu_memory = RESULTS / "memory_scaling.gpu.json"
    if gpu_memory.exists():
        render_memory_scaling(gpu_memory, figures_dir / "memory_scaling.gpu.png")
    elif gpu_inference.exists():
        render_memory_scaling(gpu_inference, figures_dir / "memory_scaling.gpu.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render benchmark figures from saved JSON results.")
    parser.add_argument("--scan-input", type=Path, default=None)
    parser.add_argument("--scan-output", type=Path, default=None)
    parser.add_argument("--inference-input", type=Path, default=None)
    parser.add_argument("--inference-output", type=Path, default=None)
    args = parser.parse_args()

    FIGURES.mkdir(parents=True, exist_ok=True)

    if args.scan_input and args.scan_output:
        render_scan_results(args.scan_input, args.scan_output)
    if args.inference_input and args.inference_output:
        render_inference_results(args.inference_input, args.inference_output)
    if not any([args.scan_input, args.scan_output, args.inference_input, args.inference_output]):
        render_default_results(FIGURES)

    print("Rendered benchmark figures to", FIGURES)


if __name__ == "__main__":
    main()
