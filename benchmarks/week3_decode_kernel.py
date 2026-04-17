"""Week 3 microbenchmark: Triton decode kernel vs pure-PyTorch equivalent.

End-to-end decode tok/s is bounded by the four ``nn.Linear`` calls in the
mixer (``in_proj``, ``x_proj``, ``dt_proj``, ``out_proj``) + Python-level
per-step overhead. The Triton kernel replaces only the inner SSM recurrence
+ C projection + D skip + gate, so its true impact shows up in an isolated
microbench — which is also what makes MBU meaningful.

We time 1000 invocations of each path at Mamba-130m decode shapes
(B=1, D=1536, N=16) plus a couple of batch/state sweeps.

Bytes accounting (per invocation, per program):
  * load A_row  : N fp32
  * load B_vec  : N fp32
  * load C_vec  : N fp32
  * load h_old  : N fp32
  * store h_new : N fp32
  * load x, dt, optional D / z : ~5 scalars
  * store y     : 1 scalar
Total per program ≈ (5 * N) fp32 ≈ 20 * N bytes.
Programs per launch = B * D.
So bytes moved per step ≈ B * D * 20 * N bytes.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from kernels.scan_decode import selective_scan_decode_triton


@dataclass(slots=True)
class Row:
    shape: str
    batch: int
    d_inner: int
    d_state: int
    dtype: str
    triton_us: float
    pytorch_us: float
    speedup: float
    triton_gbps: float
    pytorch_gbps: float
    pct_peak_600gbps: float


def _pytorch_reference(x, dt, A, B, C, ssm_state, D_skip, z):
    """Same math as the kernel, implemented in fp32 PyTorch."""
    x_f = x.float()
    dt_f = dt.float()
    delta_A = torch.exp(dt_f.unsqueeze(-1) * A.unsqueeze(0))
    delta_B = dt_f.unsqueeze(-1) * B.float().unsqueeze(1)
    h_new = delta_A * ssm_state + delta_B * x_f.unsqueeze(-1)
    y = (h_new * C.float().unsqueeze(1)).sum(dim=-1)
    if D_skip is not None:
        y = y + D_skip * x_f
    if z is not None:
        zf = z.float()
        y = y * (zf * torch.sigmoid(zf))
    return y.to(x.dtype), h_new


def _bytes_per_step(batch: int, d_inner: int, d_state: int) -> int:
    """Rough bytes moved by the kernel per step — see module docstring."""
    # (A, B, C, h_old, h_new_store) = 5 * N fp32 per program
    vec_bytes = 5 * d_state * 4
    # plus ~5 scalar loads + 1 store ≈ 6 * 4 bytes per program
    scalar_bytes = 6 * 4
    return batch * d_inner * (vec_bytes + scalar_bytes)


def _time_triton(fn, iters: int) -> float:
    # Warmup
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1e3 / iters  # us / call


def benchmark_shape(batch: int, d_inner: int, d_state: int, dtype: torch.dtype, iters: int) -> Row:
    device = "cuda"
    torch.manual_seed(0)
    x = torch.randn(batch, d_inner, dtype=dtype, device=device)
    dt = torch.rand(batch, d_inner, device=device) * 0.5
    A = -torch.rand(d_inner, d_state, device=device).abs() - 0.1
    B = torch.randn(batch, d_state, dtype=dtype, device=device)
    C = torch.randn(batch, d_state, dtype=dtype, device=device)
    ssm_state = torch.randn(batch, d_inner, d_state, device=device)
    D_skip = torch.randn(d_inner, device=device)
    z = torch.randn(batch, d_inner, dtype=dtype, device=device)

    triton_us = _time_triton(
        lambda: selective_scan_decode_triton(
            x=x, dt=dt, A=A, B=B, C=C,
            ssm_state=ssm_state, D_skip=D_skip, z=z,
        ),
        iters,
    )
    pytorch_us = _time_triton(
        lambda: _pytorch_reference(x, dt, A, B, C, ssm_state, D_skip, z),
        iters,
    )

    bytes_moved = _bytes_per_step(batch, d_inner, d_state)
    triton_gbps = bytes_moved / (triton_us * 1e-6) / 1e9
    pytorch_gbps = bytes_moved / (pytorch_us * 1e-6) / 1e9
    peak = 600.0
    return Row(
        shape=f"B={batch}, D={d_inner}, N={d_state}",
        batch=batch,
        d_inner=d_inner,
        d_state=d_state,
        dtype=str(dtype).replace("torch.", ""),
        triton_us=triton_us,
        pytorch_us=pytorch_us,
        speedup=pytorch_us / max(triton_us, 1e-9),
        triton_gbps=triton_gbps,
        pytorch_gbps=pytorch_gbps,
        pct_peak_600gbps=triton_gbps / peak * 100.0,
    )


def _md(rows: list[Row]) -> str:
    head = (
        "| Shape | Dtype | Triton (µs) | PyTorch (µs) | Speedup | "
        "Triton GB/s | PyTorch GB/s | % of 600 GB/s |"
    )
    sep = "|---|---|---:|---:|---:|---:|---:|---:|"
    out = [head, sep]
    for r in rows:
        out.append(
            f"| {r.shape} | {r.dtype} | {r.triton_us:.1f} | {r.pytorch_us:.1f} | "
            f"{r.speedup:.1f}x | {r.triton_gbps:.1f} | {r.pytorch_gbps:.1f} | "
            f"{r.pct_peak_600gbps:.1f}% |"
        )
    return "\n".join(out)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--iters", type=int, default=1000)
    p.add_argument("--output", type=str,
                   default="benchmarks/results/week3_decode_kernel.gpu.json")
    args = p.parse_args()

    shapes = [
        (1, 1536, 16, torch.float32),   # Mamba-130m decode shape
        (4, 1536, 16, torch.float32),
        (8, 1536, 16, torch.float32),
        (1, 1536, 16, torch.float16),
        (1, 1536, 16, torch.bfloat16),
        (1, 3072, 16, torch.float32),   # Mamba-370m-scale
        (1, 1536, 64, torch.float32),   # larger state
    ]

    rows: list[Row] = []
    for b, d, n, dt in shapes:
        row = benchmark_shape(b, d, n, dt, args.iters)
        rows.append(row)
        print(
            f"B={b} D={d} N={n} {str(dt).replace('torch.',''):8} "
            f"triton={row.triton_us:7.1f} us  pytorch={row.pytorch_us:7.1f} us  "
            f"speedup={row.speedup:.2f}x  triton_bw={row.triton_gbps:.1f} GB/s"
        )

    print()
    print(_md(rows))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"iters": args.iters, "rows": [asdict(r) for r in rows]}, indent=2,
    ))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
