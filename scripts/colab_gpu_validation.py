from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "benchmarks" / "results"
FIGURES_DIR = REPO_ROOT / "figures"


def run(cmd: list[str]) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    print("== Mamba Colab GPU Validation ==")
    print("repo:", REPO_ROOT)

    run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "torch",
            "torchvision",
            "torchaudio",
            "--index-url",
            "https://download.pytorch.org/whl/cu121",
        ]
    )
    run([sys.executable, "-m", "pip", "install", "-e", "." + "[dev,bench,kernel]"])

    run([sys.executable, "-c", "import torch; print('cuda available:', torch.cuda.is_available()); print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"])
    run([sys.executable, "-m", "pytest", "-q"])

    run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "run_gpu_validation.py"),
            "--device",
            "auto",
            "--batch",
            "2",
            "--channels",
            "64",
            "--state",
            "16",
            "--length",
            "256",
            "--warmup",
            "5",
            "--repeats",
            "20",
            "--new-tokens",
            "16",
            "--parity-layers",
            "0,5,23",
        ]
    )

    print("\n== Generated GPU JSON results ==")
    for path in sorted(RESULTS_DIR.glob("*.gpu.json")):
        print(f"\n--- {path.name} ---")
        payload = json.loads(path.read_text())
        print(json.dumps(payload, indent=2)[:4000])

    archive = shutil.make_archive(str(REPO_ROOT / "mamba_gpu_validation_artifacts"), "zip", RESULTS_DIR)
    print("\nArtifacts zip:", archive)
    print("Figures dir:", FIGURES_DIR)


if __name__ == "__main__":
    main()
