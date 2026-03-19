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


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a simple roofline chart.")
    parser.add_argument("--peak-tflops", type=float, default=8.1, help="Peak device TFLOPs")
    parser.add_argument(
        "--peak-bandwidth-gb-s", type=float, default=320.0, help="Peak device bandwidth"
    )
    parser.add_argument("--kernel-intensity", type=float, default=2.0)
    parser.add_argument("--kernel-throughput-gflops", type=float, default=600.0)
    parser.add_argument("--output", type=Path, default=Path("figures/roofline.png"))
    args = parser.parse_args()

    intensity, roofline = build_roofline(args.peak_tflops, args.peak_bandwidth_gb_s)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.loglog(intensity, roofline, label="Roofline", linewidth=2)
    plt.scatter(
        [args.kernel_intensity],
        [args.kernel_throughput_gflops],
        label="Selective scan kernel",
        color="crimson",
        s=60,
    )
    plt.xlabel("Arithmetic intensity (FLOPs / byte)")
    plt.ylabel("Performance (GFLOPs / s)")
    plt.title("Selective Scan Roofline")
    plt.grid(True, which="both", linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output, dpi=180)
    print(f"Saved roofline chart to {args.output}")


if __name__ == "__main__":
    main()
