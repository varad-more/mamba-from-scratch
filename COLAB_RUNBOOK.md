# Colab Runbook

Use this when you want to execute the GPU-dependent validation steps on Google Colab.

## 1) Clone the repo

```bash
git clone https://github.com/varad-more/mamba-from-scratch.git
cd mamba-from-scratch
```

## 2) Create environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

## 3) Install dependencies

For Colab GPU, install PyTorch, project deps, and Triton:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -e .[dev,bench,kernel]
```

If the default CUDA wheel URL changes in Colab, adjust accordingly.

## 4) Sanity checks

```bash
python - <<'PY'
import torch
print('cuda available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('device:', torch.cuda.get_device_name(0))
PY
pytest -q
```

## 5) Run the full GPU validation bundle

```bash
python scripts/run_gpu_validation.py \
  --device auto \
  --batch 2 \
  --channels 64 \
  --state 16 \
  --length 256 \
  --warmup 5 \
  --repeats 20 \
  --new-tokens 16 \
  --parity-layers 0,5,23
```

This will generate:
- `benchmarks/results/scan_results.gpu.json`
- `benchmarks/results/official_parity.gpu.json`
- `benchmarks/results/inference_results.gpu.json`
- refreshed figures in `figures/`

## 6) Optional focused commands

### Scan benchmark only

```bash
python benchmarks/benchmark_scan.py \
  --device auto \
  --batch 2 \
  --channels 64 \
  --state 16 \
  --length 256 \
  --warmup 5 \
  --repeats 20 \
  --output benchmarks/results/scan_results.gpu.json
```

### Official parity only

```bash
python scripts/official_parity.py \
  --model state-spaces/mamba-130m-hf \
  --layer 0,5,23 \
  --seq-len 8 \
  --batch 1 \
  --device auto \
  --json \
  --output benchmarks/results/official_parity.gpu.json
```

### Inference benchmark only

```bash
python benchmarks/benchmark_inference.py \
  --device auto \
  --new-tokens 16 \
  --mamba-model state-spaces/mamba-130m-hf \
  --baseline-model gpt2 \
  --output benchmarks/results/inference_results.gpu.json
```

## 7) What to look for

- `official_parity.gpu.json` should show very low or zero error for supported comparisons.
- `scan_results.gpu.json` should let you compare reference vs naive vs fused latency and bandwidth.
- `inference_results.gpu.json` should expose TTFT / inter-token latency / memory behavior.

## 8) If Triton path falls back

Check:
- CUDA is actually available.
- tensors are on the same CUDA device.
- shapes match supported layouts:
  - `u`, `delta`: `(B, D, L)`
  - `A`: `(D, N)`
  - shared `B`, `C`: `(B, N, L)`
  - or channel-specific `B`, `C`: `(B, D, N, L)`
- state size is <= 128

If needed, start with smaller shapes and increase gradually.
