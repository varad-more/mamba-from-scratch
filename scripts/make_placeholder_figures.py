from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def architecture_figure(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")
    boxes = [
        (0.05, 0.35, 0.18, 0.3, "Input\n(B,L,D)"),
        (0.28, 0.35, 0.2, 0.3, "Conv +\nSelective Scan"),
        (0.54, 0.35, 0.18, 0.3, "SiLU Gate\nBranch"),
        (0.77, 0.35, 0.18, 0.3, "Output\n(B,L,D)"),
    ]
    for x, y, w, h, label in boxes:
        ax.add_patch(plt.Rectangle((x, y), w, h, fill=False, linewidth=2))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=11)
    arrows = [(0.23, 0.5, 0.28, 0.5), (0.48, 0.5, 0.54, 0.5), (0.72, 0.5, 0.77, 0.5)]
    for x1, y1, x2, y2 in arrows:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=2))
    ax.set_title("Minimal Mamba Block (high-level)", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=180)


def memory_scaling(path: Path) -> None:
    lengths = np.array([128, 256, 512, 1024, 2048, 4096, 8192])
    mamba = np.full_like(lengths, 420, dtype=float)
    gpt2 = 120 + 0.06 * lengths

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(lengths, mamba, marker="o", label="Mamba (state memory)")
    ax.plot(lengths, gpt2, marker="o", label="GPT-2 (KV cache)")
    ax.set_xlabel("Context length")
    ax.set_ylabel("Approx. peak memory (MB)")
    ax.set_title("Memory vs Context Length")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)


def throughput(path: Path) -> None:
    batches = np.array([1, 4, 8, 16, 32])
    mamba = np.array([120, 410, 760, 1280, 1800])
    gpt2 = np.array([110, 350, 610, 860, 980])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(batches, mamba, marker="o", label="Mamba")
    ax.plot(batches, gpt2, marker="o", label="GPT-2")
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Tokens / second")
    ax.set_title("Throughput scaling (illustrative)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)


def main() -> None:
    out = Path("figures")
    out.mkdir(parents=True, exist_ok=True)
    architecture_figure(out / "architecture.png")
    memory_scaling(out / "memory_scaling.png")
    throughput(out / "throughput_comparison.png")
    print("Generated figures in", out)


if __name__ == "__main__":
    main()
