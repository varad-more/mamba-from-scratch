# Mamba-from-Scratch Architecture

This document is the architecture source of truth for the current repository.

It maps the **actual code layout today**, then defines the next refactor steps.

---

## 1) Purpose

This repo has two simultaneous goals:

1. **Readable learning implementation** of Mamba internals
2. **Practical engineering harness** for parity, kernels, and benchmarks

The architecture must keep these concerns separated so we can teach and optimize without turning the codebase into spaghetti.

---

## 2) Current architecture (as implemented)

## High-level runtime flow

```text
Input tokens/hidden states
        ↓
src/mamba_minimal/model.py
(MambaBlock + generation path)
        ↓
src/mamba_minimal/selective_scan.py
(reference scan math)
        ↓
kernels/scan_fused.py
(optional Triton fused path + fallback metadata)
        ↓
tests/ + scripts/ + benchmarks/
(parity, validation, performance)
```

## Layer map

- **Core math + model layer**
  - `src/mamba_minimal/discretization.py`
  - `src/mamba_minimal/selective_scan.py`
  - `src/mamba_minimal/model.py`
  - `src/mamba_minimal/parallel_scan.py`
  - `src/mamba_minimal/ssd.py`
  - `src/mamba_minimal/generate.py`

- **Kernel layer**
  - `kernels/scan_naive.py`
  - `kernels/scan_fused.py`
  - `kernels/autotune.py`

- **Validation + benchmark layer**
  - `tests/*.py`
  - `scripts/official_parity.py`
  - `scripts/run_gpu_validation.py`
  - `benchmarks/benchmark_scan.py`
  - `benchmarks/benchmark_inference.py`
  - `benchmarks/roofline.py`

---

## 3) Ownership boundaries

## A. Core math/model (`src/mamba_minimal/*`)

**Owns**
- numerical correctness of discretization + scan logic
- readable MambaBlock behavior and shape contracts
- deterministic reference path used as truth baseline

**Does not own**
- backend dispatch policy
- benchmark/reporting orchestration

## B. Kernel path (`kernels/*`)

**Owns**
- fused kernel execution path
- kernel eligibility checks (shape/device/dtype)
- explicit metadata about fallback behavior

**Does not own**
- high-level model semantics
- checkpoint loading concerns

## C. Validation/benchmark (`tests`, `scripts`, `benchmarks`)

**Owns**
- parity checks
- performance experiments
- artifact generation (json/figures)

**Does not own**
- core implementation logic

---

## 4) Current strengths

- Reference implementation is readable and easy to test.
- Fused kernel entrypoint already contains support checks and fallback metadata.
- Benchmarks and parity scripts are scriptable and CI-friendly.

---

## 5) Current gaps to address

1. **Backend selection policy is split across call sites**
   - behavior exists, but not centrally represented as a reusable policy module.

2. **Kernel capability reporting is local to one path**
   - we should expose a single capability API for diagnostics and experiment logs.

3. **Model → backend intent is implicit**
   - model code should request a backend policy, not hardcode assumptions.

4. **Validation output lacks standardized backend capability snapshot**
   - results should include “what backend was eligible + why fallback occurred”.

---

## 6) Target next-step architecture (incremental, no giant rewrite)

```text
src/mamba_minimal/
├── backend/
│   ├── capability.py      # canonical support checks
│   ├── policy.py          # backend selection policy (auto/reference/fused)
│   └── types.py           # metadata dataclasses shared across layers
├── selective_scan.py      # reference implementation (truth)
├── model.py               # consumes policy, not backend details
└── ...

kernels/
└── scan_fused.py          # pure fused implementation + kernel-level helpers
```

Key rule: **reference path remains the correctness anchor** and every optimized path must be explainable by metadata.

---

## 7) Refactor phases

## Phase 1 — Capability surface (immediate)

- Introduce `src/mamba_minimal/backend/types.py` for shared metadata dataclasses.
- Move/centralize capability checks into `backend/capability.py`.
- Keep existing behavior; no algorithmic change.

**Definition of done**
- one canonical API for “is fused path supported?”
- no behavior regressions in existing tests

## Phase 2 — Backend policy module

- Add `backend/policy.py` with explicit modes:
  - `auto` (prefer fused if supported)
  - `reference` (force reference)
  - `fused` (require fused, else fail with clear reason)
- Wire policy into scan entry points.

**Definition of done**
- backend choice is explicit and testable
- failures are actionable instead of silent

## Phase 3 — Validation/reporting integration

- Include backend capability + selected policy metadata in benchmark/parity JSON.
- Standardize result schema across scripts.

**Definition of done**
- every benchmark artifact says *what ran and why*

---

## 8) Invariants we must preserve

- Reference scan remains mathematically clear and easy to audit.
- Optimized paths never silently change semantics.
- Every fallback must carry a machine-readable reason.
- Unit tests stay runnable on CPU-only environments.

---

## 9) Immediate implementation plan (this sprint)

1. Add backend metadata + capability module.
2. Add tests for policy/capability behavior on CPU-only hosts.
3. Wire benchmark JSON to include backend selection notes.
4. Update README “Architecture map” to point to the new boundary files.

This keeps momentum high with minimal risk.
