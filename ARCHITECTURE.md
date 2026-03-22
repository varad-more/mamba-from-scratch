# Mamba-from-Scratch Architecture

This document is the working architecture map for the repository.

The goal is to make the repo easier to reason about before we start deeper cleanup or refactors.

---

## 1. Project goal

This repository is doing two jobs at once:

1. **Educational/reference implementation** of Mamba-style sequence modeling
2. **Practical validation/benchmark harness** against Hugging Face checkpoints and GPU kernels

That combination is useful, but it also means architecture boundaries matter a lot. Right now the repo mostly works, but the responsibilities are spread across a few modules that deserve a clearer map.

---

## 2. Current architecture at a glance

## High-level flow

```text
Hugging Face checkpoint/config
        ↓
utils/hf_loader.py
        ↓
mixer_seq_simple.py
(MambaLMHeadModel / backbone assembly)
        ↓
modules/mamba_simple.py
(core Mamba block / mixer logic)
        ↓
ops/selective_scan_interface.py
(kernel + scan execution path)
        ↓
validation / parity / benchmark scripts
```

## Primary layers

- **Model assembly layer**
  - builds the end-to-end language model stack
  - owns embeddings / backbone / LM head composition
  - current main file: `src/mamba_from_scratch/mixer_seq_simple.py`

- **Core block / mixer layer**
  - defines the main Mamba mixing block logic
  - current main file: `src/mamba_from_scratch/modules/mamba_simple.py`

- **Kernel / scan execution layer**
  - owns selective scan interface and low-level execution path
  - current main file: `src/mamba_from_scratch/ops/selective_scan_interface.py`

- **Interop / checkpoint loading layer**
  - maps HF configs and weights into the local model structure
  - current main file: `src/mamba_from_scratch/utils/hf_loader.py`

- **Experiment / validation layer**
  - GPU validation, parity checks, plots, benchmark outputs
  - current example entry point: `scripts/run_gpu_validation.py`

---

## 3. What each layer should own

## A. Model assembly layer

**Current role**
- constructs the top-level model
- wires blocks together
- defines the user-facing model object for inference/validation

**Should own**
- top-level config normalization
- backbone construction
- hidden-state shape contracts
- forward pass orchestration at the model level

**Should not own**
- backend-specific kernel branching
- checkpoint download logic
- benchmark/reporting logic

---

## B. Core block / mixer layer

**Current role**
- implements the main Mamba block behavior
- handles projection/mixing/state-space related logic

**Should own**
- mathematically meaningful block behavior
- tensor contracts between block inputs/outputs
- residual/block-local behavior

**Should not own**
- model loading from HF
- report generation
- experiment orchestration

---

## C. Ops / selective scan layer

**Current role**
- provides the actual selective scan execution path
- bridges higher-level model logic to lower-level execution

**Should own**
- backend dispatch
- fused/reference scan path selection
- backend capability reporting
- compatibility checks

**Should not own**
- business logic about model loading or experiment layout

---

## D. Loader / HF interop layer

**Current role**
- loads model config / weights from Hugging Face
- translates checkpoint structure into local model structure

**Should own**
- config translation
- state-dict mapping
- checkpoint compatibility checks
- model family/version handling

**Should not own**
- training/inference benchmark logic
- plotting/report generation

---

## E. Validation / benchmark layer

**Current role**
- runs parity checks
- runs GPU validation and performance tests
- emits benchmark artifacts

**Should own**
- experiment config
- report/plot generation
- parity and perf harnesses
- output artifact layout

**Should not own**
- core model implementation
- checkpoint conversion logic

---

## 4. Current architectural pain points

Based on the repo layout, the biggest architecture risks are:

### 1. Reference logic and runtime/backend logic are too close together
The core model files likely carry both:
- conceptual model implementation
- and execution-path assumptions

That makes the repo harder to teach from and harder to optimize safely.

### 2. Backend selection is not explicit enough
For a repo like this, readers should be able to answer:
- are we using reference PyTorch?
- a fused CUDA path?
- a scan wrapper?
- what happens when a backend is unavailable?

That needs to be obvious and inspectable.

### 3. HF loading is important enough to be a first-class boundary
HF checkpoint/config translation is not just plumbing — it is one of the core integration surfaces of the repo.

If it is too tightly coupled to model construction, future changes get harder.

### 4. Validation and benchmarking should sit on top of architecture, not inside it
The repo should feel like:
- **core implementation**
- then **interop**
- then **validation/benchmarking**

If those layers blur together, future maintenance gets messy fast.

### 5. There is room for a stronger “capability matrix” layer
Some model/backends/hardware combos work, some do not.

The repo would benefit from a simple way to express:
- supported backend
- required device
- expected precision
- known incompatibilities
- fallback path

---

## 5. Proposed target architecture

## Target top-level structure

```text
src/mamba_from_scratch/
├── config/
│   ├── model_config.py
│   └── runtime_config.py
├── model/
│   ├── backbone.py
│   ├── lm.py
│   └── blocks/
├── modules/
│   ├── mamba_simple.py
│   └── ...
├── ops/
│   ├── selective_scan_interface.py
│   ├── backends/
│   │   ├── reference.py
│   │   ├── fused.py
│   │   └── registry.py
│   └── capability.py
├── loaders/
│   ├── hf_config.py
│   ├── hf_state_dict.py
│   └── loader.py
├── validation/
│   ├── parity.py
│   ├── perf.py
│   └── reporting.py
└── utils/
```

This does **not** mean the repo needs a giant refactor immediately.
It means we should refactor toward these boundaries over time.

---

## 6. Recommended refactor phases

## Phase 1 — Documentation and boundary cleanup

Goal: make the existing structure understandable without changing behavior too much.

### Actions
- add this `ARCHITECTURE.md`
- document current execution path in README or docs
- clearly mark:
  - model layer
  - ops layer
  - loader layer
  - validation layer
- document known backend/model compatibility issues

### Output
- easier onboarding
- safer future edits
- lower accidental regression risk

---

## Phase 2 — Make backend selection explicit

Goal: reduce hidden behavior around scan/kernel path choice.

### Actions
- introduce a small backend registry or capability module
- make backend choice visible in logs and validation output
- ensure fallback behavior is explicit

### Output
- easier debugging
- clearer GPU/perf reasoning
- better reproducibility

---

## Phase 3 — Separate loader concerns from model concerns

Goal: cleanly isolate HF interop.

### Actions
- move config translation into dedicated loader helpers
- move state-dict conversion/loading into dedicated loader helpers
- keep model constructors simpler and less checkpoint-aware

### Output
- easier model extension
- easier testing
- cleaner architecture for blog/docs

---

## Phase 4 — Make validation/reporting a first-class layer

Goal: make the repo strong as both a learning project and a performance project.

### Actions
- standardize benchmark outputs
- standardize parity reports
- keep plots and validation artifacts in a consistent place
- introduce small config-driven experiment entry points

### Output
- cleaner experimentation workflow
- easier comparison between backends/models
- better reproducibility for blog/demo work

---

## 7. Practical invariants we should preserve

If we refactor, these invariants should stay true:

- **Core model math remains readable**
- **Backend-specific complexity stays out of user-facing model code where possible**
- **HF loading remains reliable and testable**
- **Validation scripts run without deep internal knowledge**
- **Failures are diagnosable at the correct layer**
  - model issue
  - backend issue
  - loader issue
  - validation issue

---

## 8. Best next concrete tasks

If we continue from here, the best next architecture tasks are:

### Option A — Best documentation-first move
Create a short **module map** section in README that links:
- model assembly
- core block logic
- ops/scan path
- HF loader
- validation scripts

### Option B — Best code-structure move
Add a tiny **backend capability/registry module** so scan-path behavior is more explicit.

### Option C — Best maintainability move
Split `hf_loader.py` responsibilities into:
- config translation
- weight/state-dict mapping
- load orchestration

---

## 9. My recommendation

Start with this order:

1. **Document the architecture clearly**
2. **Make backend/runtime selection explicit**
3. **Split loader responsibilities**
4. **Then touch deeper model structure only if needed**

That keeps the repo stable while making it much easier to extend.

---

## 10. Working interpretation of the repo

The best mental model for this repository is:

- **Model core** → the Mamba implementation itself
- **Execution backend** → how selective scan actually runs
- **Interop layer** → how Hugging Face checkpoints/configs enter the system
- **Validation layer** → how correctness and performance are proven

As long as those four layers stay clear, the repo can grow without turning into spaghetti.
