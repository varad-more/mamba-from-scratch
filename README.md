# Mamba From Scratch: Math → PyTorch → Kernel → System

A practical, test-first implementation of Mamba concepts from first principles.

This repo is built to answer two questions:
1. **Can we re-implement the core selective SSM mechanics correctly?**
2. **Can we explain the systems tradeoff (memory vs compute) with reproducible evidence?**

---

## What this project proves

- Rebuilt selective scan recurrence with clear tensor contracts and tests.
- Implemented a minimal Mamba block in PyTorch.
- Added parallel/chunked scan implementations and SSD-style prototype.
- Added benchmark harnesses and roofline tooling.
- Kept kernel entry points API-stable with CPU-safe fallback.

> Environment note: this workspace is CPU-only. Triton/CUDA execution is scaffolded and falls back to validated PyTorch reference paths unless GPU support is available.

---

## Status snapshot

- ✅ Core reference modules implemented
- ✅ Unit + parity + end-to-end tests implemented
- ✅ Triton fused forward path added for both shared and channel-specific `B/C` layouts
- ✅ Official HF parity path working for `state-spaces/mamba-130m-hf` mixer extraction
- ✅ `pytest`: **15 passed, 2 skipped** by default (`1` GPU-only, `1` official-parity slow test)
- ✅ Notebooks generated for each major milestone
- ✅ Benchmark scripts + result-driven figure rendering working

---

## Repository layout

```text
mamba-from-scratch/
├── PROJECT_PLAN.md                 # Full implementation plan (developer + showcase)
├── CONTRACTS.md                    # Shape/dtype/tolerance contracts
├── README.md
├── pyproject.toml
├── requirements.txt
├── notebooks/
│   ├── 01_ssm_basics.ipynb
│   ├── 02_selective_scan.ipynb
│   ├── 03_parallel_scan.ipynb
│   ├── 05_profiling.ipynb
│   └── 07_inference_comparison.ipynb
├── src/mamba_minimal/
│   ├── discretization.py
│   ├── selective_scan.py
│   ├── model.py
│   ├── parallel_scan.py
│   ├── ssd.py
│   └── generate.py
├── kernels/
│   ├── scan_naive.py
│   ├── scan_fused.py
│   └── autotune.py
├── benchmarks/
│   ├── benchmark_scan.py
│   ├── benchmark_inference.py
│   └── roofline.py
├── scripts/
│   ├── create_notebooks.py
│   ├── make_placeholder_figures.py
│   ├── official_parity.py
│   ├── render_benchmark_figures.py
│   └── run_gpu_validation.py
├── figures/
└── tests/
```

---

## Quickstart

### 1) Create and activate env

```bash
uv venv .venv
source .venv/bin/activate
```

### 2) Install dependencies (CPU-friendly)

```bash
uv pip install --index-url https://download.pytorch.org/whl/cpu torch
uv pip install -e .[dev]
```

Optional:

```bash
uv pip install -e .[bench]     # inference + benchmark extras
uv pip install -e .[kernel]    # Triton (Linux/CUDA environments)
```

### 3) Run tests

```bash
pytest -q
```

---

## Phase-by-phase execution commands

### Phase 1: core math + recurrence checks

```bash
pytest tests/test_discretization.py tests/test_selective_scan.py -q
```

### Phase 2: scan algorithm checks

```bash
pytest tests/test_parallel_scan.py -q
```

### Phase 3: kernel wrapper parity checks

```bash
pytest tests/test_kernel_parity.py -q
```

### Phase 4: end-to-end block behavior

```bash
pytest tests/test_end_to_end.py -q
```

### Full suite

```bash
pytest -q
```

---

## Benchmarks and profiling

### Scan benchmark

```bash
python benchmarks/benchmark_scan.py --device auto --batch 2 --channels 64 --state 16 --length 256
# save directly to JSON
python benchmarks/benchmark_scan.py --device auto --batch 2 --channels 64 --state 16 --length 256 --output benchmarks/results/scan_results.gpu.json
```

Example output file in this repo:
- `benchmarks/results/scan_results.cpu.json`

### Roofline figure

```bash
python benchmarks/roofline.py --output figures/roofline.png
```

### Inference benchmark (downloads models)

```bash
python benchmarks/benchmark_inference.py --device auto --new-tokens 32
# save directly to JSON
python benchmarks/benchmark_inference.py --device auto --new-tokens 32 --output benchmarks/results/inference_results.gpu.json
```

Example output files in this repo:
- `benchmarks/results/inference_results.cpu.json` (CPU comparison: `mamba-130m-hf` vs `gpt2`)
- `benchmarks/results/inference_results.tiny.cpu.json` (tiny model smoke benchmark)

### Official parity check against HuggingFace Mamba

```bash
python scripts/official_parity.py --model state-spaces/mamba-130m-hf --layer 0 --seq-len 8 --batch 1 --device auto --json
# or sweep multiple layers + save to disk
python scripts/official_parity.py --model state-spaces/mamba-130m-hf --layer 0,5,23 --seq-len 4 --batch 1 --device auto --json --output benchmarks/results/official_parity.gpu.json
```

This verifies that the local `MambaBlock` can load an official mixer state dict and reproduce its output.

Sample saved results:
- `benchmarks/results/official_parity.layer0.cpu.json` → current sample run reports `max_abs_error = 0.0`
- `benchmarks/results/official_parity.sample_layers.cpu.json` → sample multi-layer sweep (`0,5,23`) also reports `max_abs_error = 0.0`

### Text generation smoke test

```bash
python -m mamba_minimal.generate "Mamba is useful because" --model state-spaces/mamba-130m-hf --max-new-tokens 32 --device auto
```

---

## Notebooks

Notebooks are generated and stored under `notebooks/`:
- `01_ssm_basics.ipynb`
- `02_selective_scan.ipynb`
- `03_parallel_scan.ipynb`
- `05_profiling.ipynb`
- `07_inference_comparison.ipynb`

Regenerate notebooks:

```bash
python scripts/create_notebooks.py
```

---

## Figures

Generated figure assets are in `figures/`:
- `roofline.png`
- `memory_scaling.png`
- `throughput_comparison.png`
- `architecture.png`
- `scan_benchmark_cpu.png`
- `inference_comparison_cpu.png`
- `scan_benchmark_gpu.png` (generated after GPU validation)
- `inference_comparison_gpu.png` (generated after GPU validation)

Render result-driven figures from saved benchmark JSON:

```bash
python scripts/render_benchmark_figures.py
```

Run the bundled GPU validation flow (great for Colab):

```bash
python scripts/run_gpu_validation.py --device auto --parity-layers 0,5,23
```

> `scan_benchmark_cpu.png` and `inference_comparison_cpu.png` are generated from actual saved result files in `benchmarks/results/`. `memory_scaling.png` and `throughput_comparison.png` remain illustrative placeholders until replaced with full target-hardware production runs.

---

## Triton fused path: current support boundary

The current fused Triton kernel supports these shape families:
- `u`, `delta`: `(B, D, L)`
- `A`: `(D, N)`
- shared `B`, `C`: `(B, N, L)`
- channel-specific `B`, `C`: `(B, D, N, L)`
- optional `D_skip`: `(D,)`
- optional gate `z`: `(B, D, L)`

If inputs fall outside this boundary, the code automatically falls back to the PyTorch reference implementation.

This means the kernel path is now **real and broader than the initial draft**, while still preserving correctness-first fallback behavior.

---

## Validation strategy

This repo follows a strict validation ladder:
1. Math-level recurrence checks
2. Selective operator parity checks
3. Block-level forward checks
4. Kernel-wrapper parity checks
5. End-to-end behavior checks

Performance claims are only meaningful after this correctness path is green.

---

## Colab / GPU execution

If you want to finish the hardware-dependent validation on Google Colab, use:
- `COLAB_RUNBOOK.md` for step-by-step setup
- `scripts/run_gpu_validation.py` for one-shot execution

That path will produce GPU JSON results and GPU figures under `benchmarks/results/` and `figures/`.

---

## Known limitations

- This workspace is CPU-only, so the new Triton fused kernel path cannot be benchmarked here on real CUDA hardware.
- Even though the fused Triton path now covers both rank-3 and rank-4 `B/C` layouts, it still needs real CUDA execution and benchmarking before making performance claims.
- Official checkpoint layer-by-layer parity is provided as a best-effort script (`scripts/official_parity.py`) and depends on model internals.
- Inference benchmark requires network/model availability.

---

## Next upgrades

- Expand official parity from one mixer/layer to a multi-layer sweep.
- Add CUDA-side parity + benchmark runs for the fused Triton kernels.
- Replace remaining illustrative figures with measured GPU runs on T4/A100.
- Add streaming API demo for side-by-side inference serving.

---

## References

- Mamba paper: https://arxiv.org/abs/2312.00752
- Official Mamba repo: https://github.com/state-spaces/mamba
- Annotated Mamba (Hard Way): https://srush.github.io/annotated-mamba/hard.html
- Mamba-2 algorithm notes: https://tridao.me/blog/2024/mamba2-part3-algorithm/

---

## License

MIT
