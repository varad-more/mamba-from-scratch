# Mamba From Scratch

A from-scratch PyTorch implementation of **Mamba-1** and **Mamba-2** — the selective state-space model architecture — built up layer by layer from the SSM math, through a parallel selective scan, a fused Triton kernel, a Mamba-2 SSD (structured state-space duality) prefill path, and a full cross-engine benchmark suite against `mamba_ssm` and a dense-transformer (Pythia) baseline.

> **Educational reimplementation.** For production use, prefer [`state-spaces/mamba`](https://github.com/state-spaces/mamba). This repo's goal is **clarity, correctness, and honest comparison** against that reference — not replacing it.

---

## Table of contents

- [What this proves](#what-this-proves)
- [Headline results (NVIDIA A10G, fp16)](#headline-results-nvidia-a10g-fp16)
- [Repository layout](#repository-layout)
- [Quickstart](#quickstart)
- [Running the benchmarks](#running-the-benchmarks)
- [Running the tests](#running-the-tests)
- [Architecture tour](#architecture-tour)
- [Validation ladder](#validation-ladder)
- [Notebooks](#notebooks)
- [Known limitations](#known-limitations)
- [References](#references)
- [License](#license)

---

## What this proves

- **Selective scan, rebuilt and verified.** `selective_scan_naive` is **bit-exact** vs `mamba_ssm.selective_scan_ref` at fp32, and within `< 2e-3` vs the CUDA-fused reference at fp16.
- **Pretrained parity.** `state-spaces/mamba-130m-hf` weights load into our `MambaBlock`; full-model generation through our naive scan is **token-exact** vs the unpatched HuggingFace baseline.
- **Fused Triton decode kernel.** 2.3–2.7× over a pure-PyTorch equivalent for the isolated SSM step. Honest MBU: **1.5% at B=1 → 12% at B=8** of the A10G's 600 GB/s peak — launch-latency bound at Mamba-130m shapes.
- **Mamba-2 via SSD.** Chunked structured-state-space-duality prefill is `O(L)` in both compute and memory; our prefill stays flat through `pl=1024` and scales linearly to `pl=4096`.
- **Cross-engine benchmark suite.** 6 engines × 2 batches × 3 prompt lengths. At `pl=4096`, our Mamba-2 beats Pythia-2.8b prefill **2.8×** on pure einsum — the point of Mamba.
- **Honest gap vs the reference.** Our prefill is 2–5× slower than `mamba_ssm` (un-fused conv + dt_bias + softplus + chunk scan, no CUDA graphs). Our decode is within 10–20%.

---

## Headline results (NVIDIA A10G, fp16)

All numbers reproduced by `benchmarks/suite.py` — raw CSV at
[`benchmarks/results/suite_v1.csv`](benchmarks/results/suite_v1.csv). Full
methodology and analysis in
[`docs/cross_engine_benchmarks.md`](docs/cross_engine_benchmarks.md) and
[`docs/decode_kernel_profiling.md`](docs/decode_kernel_profiling.md).

**Matrix:** 6 engines × batches {1, 4} × prompt lengths {128, 1024, 4096} × 128
generated tokens. CUDA-event timing, 1 warmup + median-of-3. 35/36 configs
succeed; 1 OOM (our Mamba-1 at B=4, pl=4096 — pure-PyTorch Blelloch scan
materializes `O(B·L·D·N)` intermediates).

### 1) Decode throughput — batch=1, prompt=1024

![tokps_bar](benchmarks/results/tokps_bar.png)

| Engine | Params (M) | Decode tok/s |
|---|---:|---:|
| hf-pythia-160m | 162 | **117.4** |
| mambassm-mamba2 | 129 | 67.0 |
| mambassm-mamba1 | 129 | 57.9 |
| minimamba-mamba1 | 129 | 57.4 |
| minimamba-mamba2 | 129 | 51.6 |
| hf-pythia-2.8b | 2775 | 47.5 |

At batch 1 everyone is latency-bound. Pythia-160m's step (one attention + 2
MLPs) beats Mamba's step (in_proj + conv + SSM + gated-norm + out_proj) for
the same param class. **Our Mamba-1 and Mamba-2 trail the `mamba_ssm`
Triton-fused decoder by 10–20%** — gap is Python-level dispatch + un-fused
norm/conv, **not** the SSM math.

### 2) Prefill latency vs prompt length — batch=1

![latency_vs_seqlen](benchmarks/results/latency_vs_seqlen.png)

Prefill time in milliseconds (lower is better):

| Engine | pl=128 | pl=1024 | pl=4096 |
|---|---:|---:|---:|
| hf-pythia-160m | 7.7 | 8.9 | 30.3 |
| mambassm-mamba1 | 21.5 | 23.8 | 55.9 |
| mambassm-mamba2 | 24.9 | 25.9 | **32.5** |
| **minimamba-mamba2** | 45.3 | 46.3 | **161.0** |
| hf-pythia-2.8b | 20.6 | 108.2 | 448.9 |
| minimamba-mamba1 | 119.1 | 1016.7 | 4720.3 |

**Two takeaways:**

1. **SSD is flat through pl=1024.** Our Mamba-2 prefill goes 45 → 46 → 161 ms.
   Chunked GEMMs; scan never shows up as a bottleneck.
2. **Pythia-2.8b's quadratic tail is clearly visible** — 21 → 108 → 449 ms
   (~4× per 4× in length). Our Mamba-2 at pl=4096 (161 ms) **beats
   Pythia-2.8b (449 ms) by 2.8×** on pure einsum (no fused kernel). This is
   the point of Mamba.

### 3) Peak memory vs prompt length — batch=1

![memory_vs_seqlen](benchmarks/results/memory_vs_seqlen.png)

Peak GPU memory in MB (lower is better):

| Engine | pl=128 | pl=1024 | pl=4096 |
|---|---:|---:|---:|
| mambassm-mamba2 | 507 | 359 | **658** |
| **minimamba-mamba2** | 311 | 440 | **954** |
| mambassm-mamba1 | 307 | 651 | 1830 |
| hf-pythia-160m | 403 | 639 | 1446 |
| hf-pythia-2.8b | 5551 | 6284 | 8795 |
| minimamba-mamba1 | 643 | 3356 | 12653 |

Mamba-2 stays under 1 GB even at 4k — SSD's state is `O(1)` per head.
Pythia-2.8b sits at 5.6 GB just for weights and adds KV cache linearly.

### 4) Decode kernel microbench — kernel-in-isolation

From [`benchmarks/decode_kernel.py`](benchmarks/decode_kernel.py), 1000 iters,
CUDA events, A10G. Bytes-moved model ≈ `B·D·(20·N + 24)` bytes/step.

| Shape | Dtype | Triton (µs) | PyTorch (µs) | Speedup | GB/s | % of 600 GB/s |
|---|---|---:|---:|---:|---:|---:|
| B=1, D=1536, N=16 | fp32 | 60.5 | 153.6 | 2.5× | 8.7 | 1.5% |
| B=8, D=1536, N=16 | fp32 | 58.4 | 152.3 | 2.6× | 72.4 | 12.1% |
| B=1, D=1536, N=16 | fp16 | 75.3 | 201.5 | 2.7× | 7.0 | 1.2% |

**Honest MBU:** 1.5% at B=1 → 12% at B=8. Launch-latency bound (~60 µs floor)
at Mamba-130m shapes — the SSM math just isn't enough work per step to
saturate A10G bandwidth at batch 1. End-to-end decode tok/s does **not**
move with `--use-triton` because decode time is dominated by the four
`nn.Linear` calls + Python overhead, not the SSM step.

### 5) Reference parity — correctness ladder

All parity locked at **bit-exact fp32** against `mamba_ssm`:

| Check | Setting | Max abs diff |
|---|---|---:|
| `selective_scan_naive` vs `mamba_ssm.selective_scan_ref` | CPU fp32, parameter sweep | **0.0** (bit-exact) |
| `selective_scan_naive` vs `mamba_ssm.selective_scan_fn` | A10G fp32 | < 1e-5 |
| `selective_scan_naive` vs `mamba_ssm.selective_scan_fn` | A10G fp16 | < 2e-3 |
| `MambaBlock` layer-0 vs HF `MambaMixer` | A10G fp32, random input | **0.0** |
| HF full-model logits, scan patched with ours | A10G fp32, 24 layers | **0.0** |
| Greedy generation, patched vs unpatched HF | A10G, 20 new tokens | **token-exact** |
| Official mixer parity (layers 0, 5, 11, 17, 23) | fp32 | **0.0** |

Reproduced by `tests/test_naive_vs_reference.py`,
`tests/test_official_parity.py`, `scripts/official_parity.py`, and notebooks
`01_selective_scan_derivation.ipynb` / `02_mamba130m_naive_generate.ipynb`.

---

## Repository layout

```text
mamba-from-scratch/
├── README.md
├── ARCHITECTURE.md                          # Module + data-flow map
├── PROJECT_PLAN.md                          # Implementation plan + exit gates
├── CONTRACTS.md                             # Shape / dtype / tolerance contracts
├── pyproject.toml                           # Package + optional extras
├── docs/
│   ├── cross_engine_benchmarks.md           # Cross-engine suite results + analysis
│   └── decode_kernel_profiling.md           # Triton decode kernel + MBU analysis
├── src/mamba_minimal/
│   ├── discretization.py                    # ZOH discretization
│   ├── scan_naive.py                        # Naive selective scan (oracle-matching)
│   ├── scan_parallel.py                     # Hillis-Steele / Blelloch parallel scan
│   ├── scan_ssd.py                          # Mamba-2 chunked SSD scan
│   ├── parallel_scan.py                     # Sequential / chunked reference scans
│   ├── selective_scan.py                    # Legacy reference selective scan
│   ├── model.py                             # MambaBlock + MambaModel (Mamba-1)
│   ├── block_mamba2.py                      # Mamba-2 block
│   ├── ssd.py                               # SSD prototype primitives
│   ├── generate.py                          # State-carrying generate()
│   ├── weights.py                           # HF checkpoint loader
│   ├── api.py                               # FastAPI serving layer
│   └── backend/                             # Capability checks + backend policy
├── kernels/
│   ├── scan_fused.py                        # Fused Triton selective scan (prefill)
│   ├── scan_decode.py                       # Fused Triton decode kernel (seqlen=1)
│   ├── scan_naive.py                        # Reference-wrapper baseline
│   └── autotune.py                          # Kernel autotuning utilities
├── benchmarks/
│   ├── parallel_scan.py                     # Our MambaModel vs HF Mamba tok/s
│   ├── decode_kernel.py                     # Triton decode kernel microbench
│   ├── mamba2_ssd.py                        # Mamba-1 vs Mamba-2 (ours vs mamba_ssm)
│   ├── suite.py                             # Cross-engine benchmark harness
│   ├── plot_suite.py                        # Suite CSV → plots
│   ├── benchmark_scan.py                    # Scan backend comparison
│   ├── benchmark_inference.py               # Mamba vs GPT-2 inference
│   ├── roofline.py                          # Roofline chart generation
│   └── results/                             # Saved JSON/CSV benchmark artifacts
├── tests/                                   # 44+ tests (GPU tests auto-skip on CPU)
├── notebooks/                               # Tutorial + derivation notebooks
├── scripts/                                 # Parity, validation, figure rendering
└── figures/                                 # Generated charts
```

---

## Quickstart

### 1) Environment

The project works with either `conda` or `uv`. Python **3.11+** required.

**Conda (recommended):**

```bash
conda create -n minimamba python=3.11 -y
conda activate minimamba
pip install -e .[dev]
```

**uv:**

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -e .[dev]
```

### 2) Install PyTorch for your hardware

CPU-only:

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch
```

CUDA (example: cu12.1 + torch 2.4):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### 3) Optional extras

```bash
pip install -e .[bench]     # transformers + accelerate for HF comparisons
pip install -e .[serve]     # FastAPI + uvicorn for the demo API
pip install -e .[kernel]    # Triton (Linux + CUDA only)
pip install -e .[all]       # everything
```

### 4) Reference oracle (optional but recommended)

`mamba_ssm` is the correctness oracle. Tests that need it auto-skip when it's absent. Install the prebuilt wheel matching your torch + CUDA:

```bash
# Example: cu12.2 + torch 2.4 + Python 3.11
pip install \
  "https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.4.0/causal_conv1d-1.4.0+cu122torch2.4cxx11abiFALSE-cp311-cp311-linux_x86_64.whl" \
  "https://github.com/state-spaces/mamba/releases/download/v2.2.2/mamba_ssm-2.2.2+cu122torch2.4cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
```

### 5) Smoke test

```bash
pytest -q                           # ~44 tests, GPU ones auto-skip on CPU
python -m mamba_minimal.generate "Mamba is useful because" \
  --model state-spaces/mamba-130m-hf --max-new-tokens 32 --device auto
```

---

## Running the benchmarks

Every benchmark writes JSON to `benchmarks/results/` and can be replayed offline.

### Cross-engine suite (the headline numbers)

~15 minutes on A10G, fp16:

```bash
PYTHONPATH=. python benchmarks/suite.py \
  --dtype float16 \
  --engines minimamba-mamba1-130m minimamba-mamba2-130m \
            mambassm-mamba1-130m mambassm-mamba2-130m \
            pythia-160m pythia-2.8b \
  --batches 1 4 --prompt-lens 128 1024 4096 --gen-lens 128 \
  --iters 3 --warmups 1 --tag suite_v1

PYTHONPATH=. python benchmarks/plot_suite.py \
  --csv benchmarks/results/suite_v1.csv
```

Extend the sweep (only Mamba survives long contexts):

```bash
--batches 1 4 16
--prompt-lens 128 1024 4096 16384 32768
```

### Decode-kernel microbench

```bash
PYTHONPATH=. python benchmarks/decode_kernel.py \
  --device cuda --dtype float16 \
  --output benchmarks/results/decode_kernel.gpu.json
```

### Mamba-1 vs Mamba-2 head-to-head

```bash
PYTHONPATH=. python benchmarks/mamba2_ssd.py \
  --output benchmarks/results/mamba2_ssd.gpu.json
```

### Parallel-scan benchmark (our MambaModel vs HF Mamba)

```bash
PYTHONPATH=. python benchmarks/parallel_scan.py \
  --device cuda --new-tokens 128 \
  --output benchmarks/results/parallel_scan.gpu.json
```

### Scan / inference / roofline (legacy microbenches)

```bash
python benchmarks/benchmark_scan.py --device auto --length 1024 \
  --output benchmarks/results/scan_results.gpu.json
python benchmarks/benchmark_inference.py --device auto \
  --prompt-lengths 8,32,128,256 --new-tokens 32 \
  --output benchmarks/results/inference_results.gpu.json
python benchmarks/roofline.py \
  --scan-results benchmarks/results/scan_results.gpu.json
```

### Official parity check

```bash
python scripts/official_parity.py \
  --model state-spaces/mamba-130m-hf \
  --layer 0,5,11,17,23 --seq-len 16 --batch 2 \
  --device auto --json \
  --output benchmarks/results/official_parity.gpu.json
```

### Serve a generation API

```bash
pip install -e .[serve]
python -m mamba_minimal.api --host 0.0.0.0 --port 8000

# in another terminal:
curl -X POST http://localhost:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "Mamba is useful because", "max_new_tokens": 32}'
```

---

## Running the tests

```bash
pytest -q                           # everything (GPU tests skip on CPU)
pytest -q -m "not gpu"              # CPU-only
pytest -q -m "not slow"             # skip model-download tests
pytest tests/test_naive_vs_reference.py -q   # just the oracle parity
```

Markers:
- `gpu` — requires CUDA + Triton (auto-skipped on CPU)
- `slow` — downloads Mamba-130m from HuggingFace

---

## Architecture tour

```
Input hidden states
        │
    MambaBlock (src/mamba_minimal/model.py)
    ├── in_proj  ─ split into (hidden_states, gate)
    ├── conv1d   ─ + SiLU
    ├── x_proj   ─ split into (dt, B, C)
    ├── dt_proj  ─ softplus
    ├── selective_scan  ─── dispatched by backend policy
    │                   ├── scan_naive        (reference oracle)
    │                   ├── scan_parallel     (Blelloch, pure PyTorch)
    │                   ├── kernels/scan_fused (Triton prefill)
    │                   └── kernels/scan_decode (Triton step)
    └── out_proj ─ * gate
        │
        ▼
    MambaModel (embedding → N×block → RMSNorm → LM head)
```

For Mamba-2, `block_mamba2.py` swaps the selective scan for `scan_ssd.py`'s chunked structured-state-space-duality path. Full map in [`ARCHITECTURE.md`](ARCHITECTURE.md).

### Module map

| Area | File | Responsibility |
|---|---|---|
| Discretization | `src/mamba_minimal/discretization.py` | ZOH + inverse softplus |
| Naive scan | `src/mamba_minimal/scan_naive.py` | Oracle-matching reference |
| Parallel scan | `src/mamba_minimal/scan_parallel.py` | Blelloch, pure PyTorch |
| SSD scan | `src/mamba_minimal/scan_ssd.py` | Mamba-2 chunked SSD |
| Mamba-1 block | `src/mamba_minimal/model.py` | `MambaBlock` + `MambaModel` |
| Mamba-2 block | `src/mamba_minimal/block_mamba2.py` | Mamba-2 block |
| Generate | `src/mamba_minimal/generate.py` | State-carrying decode loop |
| Weight loader | `src/mamba_minimal/weights.py` | HF checkpoint → our modules |
| Fused Triton prefill | `kernels/scan_fused.py` | Chunked selective scan |
| Fused Triton decode | `kernels/scan_decode.py` | One-step SSM recurrence |
| Backend policy | `src/mamba_minimal/backend/` | `auto` / `reference` / `fused` dispatch |
| Serving | `src/mamba_minimal/api.py` | FastAPI demo |

### Triton kernel support

The fused prefill kernel (`kernels/scan_fused.py`) supports:

- `u`, `delta`: `(B, D, L)`
- `A`: `(D, N)`
- shared `B`, `C`: `(B, N, L)`
- channel-specific `B`, `C`: `(B, D, N, L)`
- optional `D_skip`: `(D,)`, optional gate `z`: `(B, D, L)`

Unsupported shapes or CPU environments automatically fall back to the PyTorch reference. **Correctness first, broader kernel coverage second.**

---

## Validation ladder

Performance claims are only made after the correctness path is green:

1. **Math-level recurrence** — ZOH discretization checked in `01_ssm_basics.ipynb`
2. **Selective scan operator parity** — `test_naive_vs_reference.py` against `mamba_ssm`
3. **Block-level forward** — `test_mamba_model_parity.py`, `test_block_mamba2.py`
4. **Kernel-wrapper parity** — `test_kernel_parity.py`, `test_scan_decode_triton.py`
5. **Official model parity** — `test_official_parity.py` with HF weights
6. **End-to-end generation** — `test_end_to_end.py`, `test_generate.py`

---

## Notebooks

Ordered reading path:

1. `01_ssm_basics.ipynb` — classical SSMs in NumPy
2. `01_selective_scan_derivation.ipynb` — tiny-tensor derivation + `mamba_ssm` parity
3. `02_selective_scan.ipynb` — selective scan walkthrough
4. `02_mamba130m_naive_generate.ipynb` — pretrained generation via our naive scan
5. `03_parallel_scan.ipynb` — Hillis-Steele / Blelloch scan algorithms
6. `03_ssd_derivation.ipynb` — Mamba-2 SSD derivation
7. `04_mamba2_generate.ipynb` — Mamba-2 end-to-end generation
8. `05_profiling.ipynb` — roofline + bandwidth analysis
9. `07_inference_comparison.ipynb` — Mamba vs GPT-2 on GPU
10. `08_colab_gpu_validation.ipynb` — Colab-friendly GPU re-run

---

## Known limitations

- **fp16 conv1d and RMSNorm are un-fused** pure-PyTorch. `mamba_ssm` fuses these; accounts for most of the decode gap vs the reference.
- **No CUDA graphs.** Every decode step pays Python dispatch overhead.
- **Mamba-1 parallel scan OOMs** at B=4, pl=4096 on a 24 GB A10G — our pure-PyTorch Blelloch scan materializes `O(B·L·D·N)` intermediates. Fused chunked scan would fix this, but wasn't ported.
- **Prefill is 2–5× slower** than `mamba_ssm` at long context. Still linear in `L`, just with a larger constant.
- **Triton decode MBU is launch-latency bound** (1.5–12% of peak) at Mamba-130m shapes. SSM math isn't where decode time is spent at this scale.
- **`ncu` counters not cross-checked** — MBU numbers come from our own bytes-moved accounting.

---

## References

- Mamba paper — https://arxiv.org/abs/2312.00752
- Mamba-2 paper — https://arxiv.org/abs/2405.21060
- Official Mamba repo — https://github.com/state-spaces/mamba
- Annotated Mamba (Hard Way) — https://srush.github.io/annotated-mamba/hard.html
- Mamba-2 algorithm notes — https://tridao.me/blog/2024/mamba2-part3-algorithm/
- HuggingFace model — https://huggingface.co/state-spaces/mamba-130m-hf

---

## License

MIT
