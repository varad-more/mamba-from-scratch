from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import torch
from torch import Tensor

from kernels.scan_fused import selective_scan_fused
from kernels.scan_naive import selective_scan_naive
from mamba_minimal.selective_scan import selective_scan_ref


@dataclass(slots=True)
class BenchmarkResult:
    name: str
    device: str
    selected_backend: str
    used_fallback: bool
    notes: str
    batch: int
    channels: int
    state: int
    length: int
    dtype: str
    latency_ms_p50: float
    latency_ms_p95: float
    throughput_tokens_per_s: float
    estimated_bytes: float
    achieved_bandwidth_gb_s: float
    arithmetic_intensity_flops_per_byte: float
    machine: str


def percentile(values: list[float], p: float) -> float:
    values = sorted(values)
    index = max(0, min(len(values) - 1, int(round((len(values) - 1) * p))))
    return values[index]


def make_inputs(batch: int, channels: int, state: int, length: int, device: str, dtype: torch.dtype):
    u = torch.randn(batch, channels, length, device=device, dtype=dtype)
    delta = torch.rand(batch, channels, length, device=device, dtype=dtype)
    A = -torch.rand(channels, state, device=device, dtype=dtype)
    B = torch.randn(batch, state, length, device=device, dtype=dtype)
    C = torch.randn(batch, state, length, device=device, dtype=dtype)
    D = torch.randn(channels, device=device, dtype=dtype)
    z = torch.randn(batch, channels, length, device=device, dtype=dtype)
    return u, delta, A, B, C, D, z


def estimate_bytes(*tensors: Tensor) -> float:
    return float(sum(t.numel() * t.element_size() for t in tensors))


def estimate_scan_flops(batch: int, channels: int, state: int, length: int, has_d: bool, has_z: bool) -> float:
    per_token = 4 * state + 2  # recurrence + output projection, rough but consistent
    if has_d:
        per_token += 1
    if has_z:
        per_token += 4  # cheap SiLU-style gating approximation
    return float(batch * channels * length * per_token)


def machine_summary(device: str) -> str:
    if device.startswith("cuda"):
        return torch.cuda.get_device_name(0)
    return f"{platform.system()} {platform.release()} ({platform.machine()})"


def benchmark(
    name: str,
    fn: Callable[[], Tensor],
    metadata_fn: Callable[[], tuple[str, bool, str]],
    length: int,
    batch: int,
    estimated_bytes: float,
    estimated_flops: float,
    device: str,
    channels: int,
    state: int,
    dtype: torch.dtype,
    warmup: int,
    repeats: int,
) -> BenchmarkResult:
    selected_backend, used_fallback, notes = metadata_fn()
    for _ in range(warmup):
        _ = fn()
    if device.startswith("cuda"):
        torch.cuda.synchronize()

    timings: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        _ = fn()
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        timings.append((time.perf_counter() - start) * 1000.0)

    p50 = percentile(timings, 0.50)
    p95 = percentile(timings, 0.95)
    throughput = (batch * length) / (p50 / 1000.0)
    bandwidth = (estimated_bytes / 1e9) / (p50 / 1000.0)
    arithmetic_intensity = estimated_flops / max(estimated_bytes, 1.0)
    return BenchmarkResult(
        name=name,
        device=device,
        selected_backend=selected_backend,
        used_fallback=used_fallback,
        notes=notes,
        batch=batch,
        channels=channels,
        state=state,
        length=length,
        dtype=str(dtype).replace("torch.", ""),
        latency_ms_p50=p50,
        latency_ms_p95=p95,
        throughput_tokens_per_s=throughput,
        estimated_bytes=estimated_bytes,
        achieved_bandwidth_gb_s=bandwidth,
        arithmetic_intensity_flops_per_byte=arithmetic_intensity,
        machine=machine_summary(device),
    )


def run_scan_benchmarks(
    batch: int,
    channels: int,
    state: int,
    length: int,
    device: str,
    dtype: torch.dtype,
    warmup: int,
    repeats: int,
) -> list[BenchmarkResult]:
    u, delta, A, B, C, D, z = make_inputs(batch, channels, state, length, device, dtype)
    touched = estimate_bytes(u, delta, A, B, C, D, z)
    estimated_flops = estimate_scan_flops(batch, channels, state, length, has_d=True, has_z=True)
    _, fused_metadata = selective_scan_fused(u, delta, A, B, C, D=D, z=z, return_metadata=True)

    return [
        benchmark(
            "reference",
            lambda: selective_scan_ref(u, delta, A, B, C, D=D, z=z),
            lambda: ("torch-reference", False, "Reference backend."),
            length,
            batch,
            touched,
            estimated_flops,
            device,
            channels,
            state,
            dtype,
            warmup,
            repeats,
        ),
        benchmark(
            "naive",
            lambda: selective_scan_naive(u, delta, A, B, C, D=D, z=z),
            lambda: (
                "torch-reference-wrapper",
                True,
                "Unfused baseline wrapper, delegates to the PyTorch reference implementation.",
            ),
            length,
            batch,
            touched,
            estimated_flops,
            device,
            channels,
            state,
            dtype,
            warmup,
            repeats,
        ),
        benchmark(
            "fused",
            lambda: selective_scan_fused(u, delta, A, B, C, D=D, z=z),
            lambda: (
                fused_metadata.selected_backend,
                fused_metadata.used_fallback,
                fused_metadata.notes,
            ),
            length,
            batch,
            touched,
            estimated_flops,
            device,
            channels,
            state,
            dtype,
            warmup,
            repeats,
        ),
    ]


def save_results(path: Path, results: list[BenchmarkResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(result) for result in results], indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark selective scan backends.")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--state", type=int, default=16)
    parser.add_argument("--length", type=int, default=256)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path")
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    results = run_scan_benchmarks(
        batch=args.batch,
        channels=args.channels,
        state=args.state,
        length=args.length,
        device=device,
        dtype=dtype,
        warmup=args.warmup,
        repeats=args.repeats,
    )
    payload = [asdict(result) for result in results]

    if args.output is not None:
        save_results(args.output, results)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
