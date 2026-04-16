from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def build_roofline(peak_tflops: float, peak_bandwidth_gb_s: float):
    intensity = np.logspace(-2, 3, 500)
    bandwidth_ceiling = peak_bandwidth_gb_s * intensity
    compute_ceiling = np.full_like(intensity, peak_tflops * 1e3)
    roofline = np.minimum(bandwidth_ceiling, compute_ceiling)
    return intensity, roofline


def load_kernel_points(scan_results_path: Path | None):
    """Extract arithmetic intensity and achieved throughput from benchmark JSON."""
    if scan_results_path is None or not scan_results_path.exists():
        return [], []

    import json
    rows = json.loads(scan_results_path.read_text())
    intensities = []
    throughputs = []
    for row in rows:
        if row.get("name") != "fused" or row.get("used_fallback", True):
            continue
        ai = row.get("arithmetic_intensity_flops_per_byte", 0.0)
        bw = row.get("achieved_bandwidth_gb_s", 0.0)
        if ai > 0 and bw > 0:
            # throughput in GFLOPS = bandwidth * arithmetic_intensity
            gflops = bw * ai
            intensities.append(ai)
            throughputs.append(gflops)
    return intensities, throughputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a simple roofline chart.")
    parser.add_argument("--peak-tflops", type=float, default=31.2, help="Peak device TFLOPs (FP32)")
    parser.add_argument(
        "--peak-bandwidth-gb-s", type=float, default=600.0, help="Peak device bandwidth (GB/s)"
    )
    parser.add_argument("--kernel-intensity", type=float, default=None,
                        help="Manual kernel intensity (used if no --scan-results)")
    parser.add_argument("--kernel-throughput-gflops", type=float, default=None,
                        help="Manual kernel throughput (used if no --scan-results)")
    parser.add_argument("--scan-results", type=Path, default=None,
                        help="Path to scan_results.gpu.json for measured kernel points")
    parser.add_argument("--output", type=Path, default=Path("figures/roofline.png"))
    args = parser.parse_args()

    intensity, roofline = build_roofline(args.peak_tflops, args.peak_bandwidth_gb_s)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.loglog(intensity, roofline, label="Roofline", linewidth=2)

    # Plot measured kernel points from benchmark data
    ki, kt = load_kernel_points(args.scan_results)
    if ki:
        plt.scatter(ki, kt, label="Fused selective scan (measured)", color="crimson", s=50, zorder=5)
    elif args.kernel_intensity is not None and args.kernel_throughput_gflops is not None:
        plt.scatter(
            [args.kernel_intensity],
            [args.kernel_throughput_gflops],
            label="Selective scan kernel",
            color="crimson",
            s=60,
        )

    ridge_point = args.peak_tflops * 1e3 / args.peak_bandwidth_gb_s
    plt.axvline(x=ridge_point, color="gray", linestyle=":", alpha=0.5, label=f"Ridge point ({ridge_point:.1f})")

    plt.xlabel("Arithmetic intensity (FLOPs / byte)")
    plt.ylabel("Performance (GFLOPs / s)")
    plt.title(f"Selective Scan Roofline (A10G: {args.peak_tflops} TFLOPS, {args.peak_bandwidth_gb_s} GB/s)")
    plt.grid(True, which="both", linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output, dpi=180)
    print(f"Saved roofline chart to {args.output}")


if __name__ == "__main__":
    main()
