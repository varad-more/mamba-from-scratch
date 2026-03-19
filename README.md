# Mamba From Scratch

A from-scratch implementation of the core ideas behind **Mamba**: continuous-time state space models, selective scan, parallel scan, Triton kernel scaffolding, parity checks against the official HuggingFace model, and benchmark / profiling utilities.

This repo is meant to be both:
- a **learning artifact** you can read end-to-end, and
- an **engineering artifact** you can run, test, benchmark, and extend.

---

## What this repo does

This project walks through the Mamba stack in layers:

1. **SSM math**
   - zero-order-hold discretization
   - continuous → discrete recurrence
2. **Reference implementation**
   - selective scan in PyTorch
   - minimal `MambaBlock`
3. **Algorithmic acceleration**
   - sequential scan
   - parallel / chunked affine scan
   - SSD-style prototype
4. **Kernel path**
   - Triton fused selective scan entry point
   - safe fallback to the PyTorch reference implementation
5. **System validation**
   - official model parity checks
   - scan benchmarks
   - inference comparison harness
   - roofline / figure generation

---

## Current status

- ✅ Core math + reference modules implemented
- ✅ Minimal Mamba block implemented in PyTorch
- ✅ Parallel / chunked scan utilities implemented
- ✅ Triton fused forward path added
- ✅ Supports both shared and channel-specific `B/C` layouts in the fused path
- ✅ Official parity path working for `state-spaces/mamba-130m-hf`
- ✅ Benchmarks and result-driven figure generation implemented
- ✅ Colab/GPU runbook + one-shot validation runner added
- ✅ Test suite passing: **15 passed, 2 skipped** by default

> This workspace is CPU-only, so the repository code is complete up to the point where **real CUDA execution and hardware benchmarks** are required. GPU-specific performance claims should be generated on Colab or another CUDA machine using the provided runbook.

---

## Key validation results

### 1) Official model parity

Saved sample parity runs show exact output match for sampled layers from `state-spaces/mamba-130m-hf`:

- `benchmarks/results/official_parity.layer0.cpu.json`
- `benchmarks/results/official_parity.sample_layers.cpu.json`

Current sample result:
- sampled layers: `0, 5, 23`
- max absolute error: **0.0**
- mean absolute error: **0.0**

That means the local `MambaBlock` can successfully load official mixer weights and reproduce the corresponding HuggingFace mixer outputs for the sampled layers.

### 2) CPU scan benchmark sample

From `benchmarks/results/scan_results.cpu.json` for a small sample run (`B=1, D=8, N=4, L=64`):

- reference p50 latency: **3.89 ms**
- naive p50 latency: **4.06 ms**
- fused p50 latency: **3.84 ms**

These are **sanity / plumbing results**, not final performance claims. Real kernel evaluation should be done on GPU.

### 3) CPU inference sample

From `benchmarks/results/inference_results.cpu.json`:

- CPU comparison is wired up and reproducible
- Mamba parity / loading path works
- memory and latency outputs are captured end-to-end

These numbers should be treated as **CPU functional validation**, not the final systems story. The real value of the repo comes from running the GPU validation flow.

---

## Repository layout

```text
mamba-from-scratch/
├── PROJECT_PLAN.md                 # Full implementation plan
├── CONTRACTS.md                    # Shape / dtype / tolerance contracts
├── COLAB_RUNBOOK.md                # How to run the GPU-dependent parts on Colab
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
│   ├── roofline.py
│   └── results/
├── scripts/
│   ├── create_notebooks.py
│   ├── make_placeholder_figures.py
│   ├── official_parity.py
│   ├── render_benchmark_figures.py
│   └── run_gpu_validation.py
├── figures/
└── tests/
```

### Directory guide

- `src/mamba_minimal/` — readable reference implementation
- `kernels/` — Triton / kernel-facing entry points
- `tests/` — correctness, parity, and regression coverage
- `benchmarks/` — reproducible timing / memory scripts
- `scripts/` — notebook generation, parity, figure rendering, validation runners
- `figures/` — generated figures used in the README / analysis

---

## Quickstart

### 1) Create an environment

```bash
uv venv .venv
source .venv/bin/activate
```

### 2) Install dependencies

CPU-friendly setup:

```bash
uv pip install --index-url https://download.pytorch.org/whl/cpu torch
uv pip install -e .[dev]
```

Optional extras:

```bash
uv pip install -e .[bench]     # transformers / accelerate / psutil
uv pip install -e .[kernel]    # Triton (Linux + CUDA environments)
```

### 3) Run tests

```bash
pytest -q
```

---

## Core commands

### Run the full local test suite

```bash
pytest -q
```

### Run phase-specific checks

```bash
pytest tests/test_discretization.py tests/test_selective_scan.py -q
pytest tests/test_parallel_scan.py -q
pytest tests/test_kernel_parity.py -q
pytest tests/test_end_to_end.py -q
```

### Run the scan benchmark

```bash
python benchmarks/benchmark_scan.py \
  --device auto \
  --batch 2 \
  --channels 64 \
  --state 16 \
  --length 256
```

Save output directly to JSON:

```bash
python benchmarks/benchmark_scan.py \
  --device auto \
  --batch 2 \
  --channels 64 \
  --state 16 \
  --length 256 \
  --output benchmarks/results/scan_results.gpu.json
```

### Run the inference benchmark

```bash
python benchmarks/benchmark_inference.py \
  --device auto \
  --new-tokens 32
```

Save output directly to JSON:

```bash
python benchmarks/benchmark_inference.py \
  --device auto \
  --new-tokens 32 \
  --output benchmarks/results/inference_results.gpu.json
```

### Run official parity against HuggingFace Mamba

Single layer:

```bash
python scripts/official_parity.py \
  --model state-spaces/mamba-130m-hf \
  --layer 0 \
  --seq-len 8 \
  --batch 1 \
  --device auto \
  --json
```

Multi-layer sweep:

```bash
python scripts/official_parity.py \
  --model state-spaces/mamba-130m-hf \
  --layer 0,5,23 \
  --seq-len 4 \
  --batch 1 \
  --device auto \
  --json \
  --output benchmarks/results/official_parity.gpu.json
```

### Render figures from saved benchmark results

```bash
python scripts/render_benchmark_figures.py
```

### Generate notebooks

```bash
python scripts/create_notebooks.py
```

### Text generation smoke test

```bash
python -m mamba_minimal.generate \
  "Mamba is useful because" \
  --model state-spaces/mamba-130m-hf \
  --max-new-tokens 32 \
  --device auto
```

---

## GPU / Colab execution

The remaining hardware-dependent part of the project is already wired up.

If you want to run the real CUDA validation flow on Google Colab, use:
- `COLAB_RUNBOOK.md` for setup instructions
- `scripts/run_gpu_validation.py` for one-shot execution

### One-command Colab / GPU validation

```bash
python scripts/run_gpu_validation.py --device auto --parity-layers 0,5,23
```

This generates:
- `benchmarks/results/scan_results.gpu.json`
- `benchmarks/results/official_parity.gpu.json`
- `benchmarks/results/inference_results.gpu.json`
- GPU figure outputs in `figures/`

If you want explicit model control:

```bash
python scripts/run_gpu_validation.py \
  --device auto \
  --mamba-model state-spaces/mamba-130m-hf \
  --baseline-model gpt2 \
  --parity-model state-spaces/mamba-130m-hf \
  --parity-layers 0,5,23
```

---

## Triton fused path: current support boundary

The current fused Triton kernel supports:

- `u`, `delta`: `(B, D, L)`
- `A`: `(D, N)`
- shared `B`, `C`: `(B, N, L)`
- channel-specific `B`, `C`: `(B, D, N, L)`
- optional `D_skip`: `(D,)`
- optional gate `z`: `(B, D, L)`

If inputs fall outside this boundary, execution automatically falls back to the PyTorch reference implementation.

This is intentional: **correctness first, broader kernel coverage second**.

---

## Figures

Generated figure assets live in `figures/`:

- `architecture.png`
- `roofline.png`
- `scan_benchmark_cpu.png`
- `inference_comparison_cpu.png`
- `scan_benchmark_gpu.png` *(generated after GPU validation)*
- `inference_comparison_gpu.png` *(generated after GPU validation)*
- `memory_scaling.png` *(illustrative placeholder)*
- `throughput_comparison.png` *(illustrative placeholder)*

Important note:
- `scan_benchmark_cpu.png` and `inference_comparison_cpu.png` are generated from actual saved results
- `memory_scaling.png` and `throughput_comparison.png` are still illustrative placeholders until replaced with measured GPU production runs

---

## Validation strategy

This repository follows a strict validation ladder:

1. math-level recurrence checks
2. selective operator parity checks
3. block-level forward checks
4. kernel-wrapper parity checks
5. official-model parity checks
6. end-to-end behavior checks

Performance claims should only be made after the correctness path is green.

---

## Known limitations

- This workspace is CPU-only, so true Triton CUDA execution cannot be benchmarked here.
- The fused Triton path is implemented, but final performance claims still require GPU runs.
- Official parity currently focuses on mixer-level validation rather than the full end-to-end pretrained model stack.
- Some figures are still placeholders until GPU results are generated.
- Inference benchmarks depend on model downloads and network availability.

---

## Next steps

- run the full GPU validation flow on Colab or another CUDA machine
- replace illustrative figures with measured GPU figures
- extend parity from sampled layers to broader model sweeps
- tighten the final README results section with real T4 / A100 measurements
- optionally add a lightweight serving demo

---

## References

- Mamba paper: https://arxiv.org/abs/2312.00752
- Official Mamba repo: https://github.com/state-spaces/mamba
- Annotated Mamba (Hard Way): https://srush.github.io/annotated-mamba/hard.html
- Mamba-2 algorithm notes: https://tridao.me/blog/2024/mamba2-part3-algorithm/

---

## License

MIT
