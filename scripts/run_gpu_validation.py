from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmarks" / "results"
SCRIPTS = ROOT / "scripts"
BENCHMARKS = ROOT / "benchmarks"


def run(cmd: list[str]) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the remaining GPU/Colab validation steps for the Mamba project."
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--state", type=int, default=16)
    parser.add_argument("--length", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--new-tokens", type=int, default=16)
    parser.add_argument("--mamba-model", default="state-spaces/mamba-130m-hf")
    parser.add_argument("--baseline-model", default="gpt2")
    parser.add_argument("--parity-model", default="state-spaces/mamba-130m-hf")
    parser.add_argument("--parity-layers", default="0,5,23")
    parser.add_argument("--skip-parity", action="store_true")
    args = parser.parse_args()

    suffix = "gpu" if args.device in {"auto", "cuda"} else args.device
    RESULTS.mkdir(parents=True, exist_ok=True)

    scan_out = RESULTS / f"scan_results.{suffix}.json"
    inf_out = RESULTS / f"inference_results.{suffix}.json"
    parity_out = RESULTS / f"official_parity.{suffix}.json"

    run(
        [
            sys.executable,
            str(BENCHMARKS / "benchmark_scan.py"),
            "--device",
            args.device,
            "--dtype",
            args.dtype,
            "--batch",
            str(args.batch),
            "--channels",
            str(args.channels),
            "--state",
            str(args.state),
            "--length",
            str(args.length),
            "--warmup",
            str(args.warmup),
            "--repeats",
            str(args.repeats),
            "--output",
            str(scan_out),
        ]
    )

    if not args.skip_parity:
        run(
            [
                sys.executable,
                str(SCRIPTS / "official_parity.py"),
                "--model",
                args.parity_model,
                "--layer",
                args.parity_layers,
                "--seq-len",
                "8",
                "--batch",
                "1",
                "--device",
                args.device,
                "--json",
                "--output",
                str(parity_out),
            ]
        )

    run(
        [
            sys.executable,
            str(BENCHMARKS / "benchmark_inference.py"),
            "--device",
            args.device,
            "--new-tokens",
            str(args.new_tokens),
            "--mamba-model",
            args.mamba_model,
            "--baseline-model",
            args.baseline_model,
            "--output",
            str(inf_out),
        ]
    )

    run([sys.executable, str(SCRIPTS / "render_benchmark_figures.py")])
    print("Saved outputs:")
    print("-", scan_out)
    if not args.skip_parity:
        print("-", parity_out)
    print("-", inf_out)


if __name__ == "__main__":
    main()
