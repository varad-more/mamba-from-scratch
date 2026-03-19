from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"


def write_notebook(path: Path, title: str, markdown_blocks: list[str], code_blocks: list[str]) -> None:
    nb = nbf.v4.new_notebook()
    cells = [nbf.v4.new_markdown_cell(f"# {title}")]
    for block in markdown_blocks:
        cells.append(nbf.v4.new_markdown_cell(block))
    for block in code_blocks:
        cells.append(nbf.v4.new_code_cell(block.strip()))
    nb["cells"] = cells
    path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, path)


def main() -> None:
    write_notebook(
        NOTEBOOKS / "01_ssm_basics.ipynb",
        "01 — SSM Basics (Continuous to Discrete)",
        [
            "Implement a classical SSM in NumPy and visualize stable vs unstable dynamics.",
            "We use the recurrence: $x_t = \bar{A}x_{t-1} + \bar{B}u_t$, $y_t = Cx_t$.",
        ],
        [
            """
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

def rollout(a_bar: float, b_bar: float, c: float, u: np.ndarray):
    x = 0.0
    xs, ys = [], []
    for u_t in u:
        x = a_bar * x + b_bar * u_t
        y = c * x
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

T = 200
t = np.linspace(0, 12, T)
u = np.sin(t)

stable_a, unstable_a = 0.8, 1.02
b, c = 0.2, 1.0
x_stable, y_stable = rollout(stable_a, b, c, u)
x_unstable, y_unstable = rollout(unstable_a, b, c, u)

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(t, x_stable, label="stable")
plt.plot(t, x_unstable, label="unstable")
plt.title("State evolution")
plt.legend()

plt.subplot(1,2,2)
plt.plot(t, y_stable, label="stable output")
plt.plot(t, y_unstable, label="unstable output")
plt.title("Output")
plt.legend()
plt.tight_layout()
""",
        ],
    )

    write_notebook(
        NOTEBOOKS / "02_selective_scan.ipynb",
        "02 — Selective Scan Reference",
        [
            "Validate the PyTorch selective scan operator on synthetic data.",
            "This notebook uses `mamba_minimal.selective_scan.selective_scan_ref`."
        ],
        [
            """
import torch
from mamba_minimal.selective_scan import selective_scan_ref

torch.manual_seed(0)
B, D, N, L = 2, 4, 3, 8
u = torch.randn(B, D, L)
delta = torch.rand(B, D, L)
A = -torch.rand(D, N)
B_t = torch.randn(B, N, L)
C_t = torch.randn(B, N, L)
D_skip = torch.randn(D)

out = selective_scan_ref(u, delta, A, B_t, C_t, D=D_skip)
out.shape
""",
            """
# quick sanity check: run twice with same seed/inputs
out2 = selective_scan_ref(u, delta, A, B_t, C_t, D=D_skip)
max_diff = (out - out2).abs().max().item()
max_diff
""",
        ],
    )

    write_notebook(
        NOTEBOOKS / "03_parallel_scan.ipynb",
        "03 — Sequential vs Parallel Affine Scan",
        [
            "Compare sequential, Hillis-Steele, and chunked scan implementations.",
            "Recurrence form: $h_t = a_t h_{t-1} + b_t$."
        ],
        [
            """
import time
import torch
from mamba_minimal.parallel_scan import sequential_affine_scan, hillis_steele_affine_scan, chunked_affine_scan

torch.manual_seed(0)
a = torch.rand(4, 1024, 16)
b = torch.randn(4, 1024, 16)

ref = sequential_affine_scan(a, b)
par = hillis_steele_affine_scan(a, b)
chk = chunked_affine_scan(a, b, chunk_size=128)

print("max diff (parallel)", (ref - par).abs().max().item())
print("max diff (chunked)", (ref - chk).abs().max().item())
""",
            """
# micro-timing on current device
for name, fn in [
    ("sequential", lambda: sequential_affine_scan(a, b)),
    ("parallel", lambda: hillis_steele_affine_scan(a, b)),
    ("chunked", lambda: chunked_affine_scan(a, b, chunk_size=128)),
]:
    start = time.perf_counter()
    _ = fn()
    dt_ms = (time.perf_counter() - start) * 1e3
    print(f"{name:>10}: {dt_ms:.2f} ms")
""",
        ],
    )

    write_notebook(
        NOTEBOOKS / "05_profiling.ipynb",
        "05 — Profiling and Roofline Notes",
        [
            "Load benchmark JSON output and plot simple throughput/bandwidth summaries.",
            "Use `benchmarks/roofline.py` to produce `figures/roofline.png`."
        ],
        [
            """
from pathlib import Path
import json

path = Path("scan_results.json")
if path.exists():
    data = json.loads(path.read_text())
    for row in data:
        print(row["name"], "p50(ms)=", round(row["latency_ms_p50"], 3), "GB/s=", round(row["achieved_bandwidth_gb_s"], 3))
else:
    print("Run benchmarks/benchmark_scan.py and save output to scan_results.json first")
""",
            """
import subprocess, sys
subprocess.run([sys.executable, "benchmarks/roofline.py", "--output", "figures/roofline.png"], check=False)
""",
        ],
    )

    write_notebook(
        NOTEBOOKS / "07_inference_comparison.ipynb",
        "07 — Inference Comparison (Mamba vs GPT-2)",
        [
            "Run benchmark script and inspect TTFT/inter-token latency and memory usage.",
            "This notebook is a thin analysis layer over `benchmarks/benchmark_inference.py`."
        ],
        [
            """
import json
from pathlib import Path

path = Path("inference_results.json")
if path.exists():
    rows = json.loads(path.read_text())
    for row in rows:
        print(row["model_name"], "ttft(ms)=", round(row["ttft_ms"], 2), "it(ms)=", round(row["inter_token_ms"], 2), "peak(MB)=", round(row["peak_memory_mb"], 2))
else:
    print("Run benchmarks/benchmark_inference.py and save output to inference_results.json first")
""",
        ],
    )

    print("Created notebooks in", NOTEBOOKS)


if __name__ == "__main__":
    main()
