# Build Your Own Mamba — Developer + Showcase Implementation Plan

## TL;DR

Build a minimal but correct Mamba stack from first principles, then prove two things:

1. **Correctness** — your implementation matches trusted references.
2. **Systems value** — your kernel and inference benchmarks explain where Mamba wins and why.

This project should demonstrate that you can move across the full stack:
- math and modeling,
- PyTorch implementation,
- kernel optimization,
- benchmarking and profiling,
- and clear technical communication.

**Timeline:** ~3 weeks part-time  
**Primary hardware:** Colab T4  
**Stretch area:** Mamba-2 SSD

---

## 1. Project Goal

Build a public repo that shows you understand Mamba end-to-end:
- the SSM math,
- the selective recurrence,
- the parallel scan idea,
- the memory-bandwidth bottleneck,
- and the inference tradeoff vs Transformer KV cache.

The final repo should be something a strong engineer can:
- clone,
- run,
- verify,
- benchmark,
- and learn from without guessing what is happening.

---

## 2. What This Project Should Signal to Developers and Recruiters

This is not just a “model reimplementation” project. If done well, it signals:

### Technical depth
- You understand SSMs beyond surface-level summaries.
- You can translate research ideas into working code.
- You know how to validate numerics, not just eyeball outputs.

### Systems thinking
- You understand the difference between compute-bound and memory-bound workloads.
- You can reason about kernel fusion, bandwidth, arithmetic intensity, and roofline models.
- You can connect low-level implementation details to user-visible inference behavior.

### Engineering maturity
- You use reference paths, tests, and benchmark discipline.
- You separate core logic, kernels, tests, and notebooks cleanly.
- You make strong claims only when they are backed by reproducible evidence.

### Communication quality
- You can explain the project to both engineers and hiring managers.
- You can present findings with clear figures, clean repo structure, and readable documentation.

If someone reads the repo for 2–3 minutes, they should immediately understand:
- what you built,
- why it is technically hard,
- how you verified it,
- and what the main systems takeaway is.

---

## 3. Final Deliverables

By the end of the project, the repo should include:

### Core engineering deliverables
- a NumPy notebook for classical SSMs,
- a PyTorch selective scan implementation,
- a minimal Mamba block with parity tests,
- a Triton unfused baseline and fused kernel,
- a profiling notebook with roofline and bandwidth analysis,
- an inference comparison against GPT-2,
- and a reproducible benchmark harness.

### Public-facing deliverables
- a polished README,
- 2–3 strong figures for the README,
- a short architecture diagram,
- clear benchmark summary tables,
- and a reproducible “run this yourself” section.

### Key visuals the repo must contain
At minimum, the final repo should include these visuals:
- **SSM dynamics plot** — to explain the math intuition
- **Sequential vs parallel scan comparison** — to explain the algorithmic trick
- **Roofline chart** — to explain the bottleneck
- **Memory vs context length chart** — the hero image for inference value

---

## 4. Scope

### P0 — Must Ship
These are the core deliverables. Do not skip them.

- SSM fundamentals notebook
- PyTorch selective scan implementation
- Mamba block parity against official model
- Parallel scan implementation and analysis
- Triton scan kernel (baseline + fused)
- Profiling notebook (MBU, roofline, bandwidth)
- Inference comparison vs GPT-2
- Clean README and reproducible repo

### P1 — Nice to Have
Only do these once P0 is stable.

- FastAPI demo
- More benchmark coverage
- Kernel autotuning experiments
- Better plots and presentation polish
- Extra analysis on chunk sizes or state dimension choices

### P2 — Stretch
Only attempt this if core work is already complete.

- Mamba-2 SSD implementation
- Limited parity checks for SSD path
- Extra writeup comparing Mamba-1 and Mamba-2 design tradeoffs

---

## 5. Non-Goals

To keep the project focused, explicitly avoid these unless extra time remains:

- full training pipeline,
- distributed training,
- production-grade serving stack,
- exhaustive Mamba-2 reproduction,
- hardware-specific optimization beyond what is reasonable on T4,
- polishing edge cases that do not improve the learning or showcase value.

---

## 6. Engineering Rules

These rules apply to the entire project:

1. **Correctness before optimization.** A fast wrong kernel is still wrong.
2. **Reference-first development.** NumPy and PyTorch define expected behavior.
3. **One clear contract per module.** Shapes, dtypes, layouts, and outputs must be explicit.
4. **Always keep a fallback path.** Triton can fail; PyTorch reference must still run.
5. **Benchmark honestly.** No cherry-picked numbers, no magical crossover claims.
6. **Make results reproducible.** Pin versions, set seeds, document environment.
7. **Write for future readers.** Code should explain the idea, not hide it.

---

## 7. Repository Layout

```text
mamba-from-scratch/
├── README.md
├── pyproject.toml
├── requirements.txt
├── notebooks/
│   ├── 01_ssm_basics.ipynb
│   ├── 02_selective_scan.ipynb
│   ├── 03_parallel_scan.ipynb
│   ├── 05_profiling.ipynb
│   └── 07_inference_comparison.ipynb
├── src/
│   └── mamba_minimal/
│       ├── __init__.py
│       ├── discretization.py
│       ├── selective_scan.py
│       ├── model.py
│       ├── parallel_scan.py
│       ├── generate.py
│       └── ssd.py
├── kernels/
│   ├── scan_naive.py
│   ├── scan_fused.py
│   └── autotune.py
├── benchmarks/
│   ├── benchmark_scan.py
│   ├── benchmark_inference.py
│   └── roofline.py
├── tests/
│   ├── test_discretization.py
│   ├── test_selective_scan.py
│   ├── test_parallel_scan.py
│   ├── test_kernel_parity.py
│   └── test_end_to_end.py
└── figures/
    ├── roofline.png
    ├── memory_scaling.png
    ├── throughput_comparison.png
    └── architecture.png
```

### Directory responsibilities
- `notebooks/` → explanation and learning artifacts
- `src/mamba_minimal/` → reference implementation code
- `kernels/` → Triton kernels only
- `benchmarks/` → reproducible performance measurements
- `tests/` → correctness and regression coverage
- `figures/` → charts used in README and writeup

---

## 8. Technical Contracts

Define these early and do not let them drift silently.

### Tensor notation
- `B`: batch size
- `L`: sequence length
- `D`: model/channel dimension
- `N`: SSM state dimension

### Expected tensor shapes
- input `u`: `[B, L, D]`
- hidden state `h_t`: `[B, D, N]`
- `A`: `[D, N]` or structured equivalent
- selective `B_t`, `C_t`: `[B, L, D, N]`
- `delta_t`: `[B, L, D]`
- output `y`: `[B, L, D]`

### Dtype policy
- math notebook: `float64`
- PyTorch reference path: `float32`
- kernel inputs: `fp16` / `bf16` allowed
- kernel accumulation: `float32`

### Tolerance policy
- FP64 parity: very tight tolerance
- FP32 parity: `atol=1e-5`, `rtol=1e-4`
- mixed precision kernel parity: loosen only as much as required

### Stability policy
- avoid explicit matrix inverse in ZOH if a stable solve form is available
- clamp or regularize `delta` where needed
- add long-sequence tests for drift, overflow, and underflow

---

## 9. Validation Strategy

Use a strict validation ladder. Do not jump from implementation straight to benchmark charts.

### Level 1 — Math parity
Validate recurrence behavior on synthetic SSM examples.

### Level 2 — Operator parity
Validate selective scan on synthetic tensors.

### Level 3 — Block parity
Validate the full Mamba block against official model intermediates.

### Level 4 — Kernel parity
Validate Triton outputs against the PyTorch reference across shapes and dtypes.

### Level 5 — System parity
Validate generation loop behavior, memory trends, and reproducibility.

### Test categories
- unit tests for discretization and scan math
- parity tests vs reference path
- long-sequence stability tests
- benchmark smoke tests
- end-to-end generation tests

---

## 10. Benchmark Rules

All benchmark results must follow the same measurement protocol.

### Measurement rules
- always report hardware and software versions
- exclude one-time JIT/compile time unless explicitly measuring startup cost
- use warmup runs before measurements
- report p50 and p95 latency
- benchmark on a fixed shape grid, not one best-case shape

### Suggested benchmark grid
- `L = [128, 256, 512, 1024, 2048, 4096, 8192]`
- `B = [1, 4, 8, 16, 32]`

### Metrics to report
- latency
- throughput
- achieved bandwidth (GB/s)
- arithmetic intensity
- model bandwidth utilization (MBU)
- peak memory usage

### Reporting rule
Do **not** say “parallel scan wins after L > 256” as a universal claim.  
Say: **“On tested hardware/software, the crossover appears around X.”**

---

## 11. Execution Plan

## Milestone 0 — Setup and Quality Gates
**Time:** Day 0

### Goal
Create a clean, reproducible project skeleton before writing core logic.

### Build
- initialize repo structure
- add `pyproject.toml` and `requirements.txt`
- configure formatting, linting, and type checking
- add pre-commit hooks
- add baseline `pytest` setup
- document shapes, dtype policy, and tolerances

### Verify
- tests run locally
- lint passes
- imports work from a clean environment

### Output
- project skeleton
- local or CI quality gate setup
- documented implementation contracts

### Showcase value
This milestone does not look flashy, but it signals engineering discipline. It makes the rest of the repo credible.

### Done when
- a fresh clone can install and run the basic test/lint flow without manual fixes

---

## Milestone 1 — Classical SSMs in NumPy
**Time:** Days 1–2

### Goal
Understand and implement the basic SSM recurrence from first principles.

### Build
- continuous-time SSM equations
- ZOH discretization
- sequential recurrence loop
- plots for stable vs unstable systems
- synthetic input example such as a sine wave

### Verify
- stable `A` produces bounded dynamics
- unstable `A` produces expected divergence
- recurrence matches manual sanity checks

### Output
- `notebooks/01_ssm_basics.ipynb`

### Showcase value
This notebook proves you understand the math instead of treating Mamba as a black box.

### Done when
- the notebook is correct, readable, and teaches the core idea clearly

---

## Milestone 2 — Selective Scan in PyTorch
**Time:** Days 3–5

### Goal
Implement the core selective recurrence used by Mamba.

### Build
- token-dependent `B_t`, `C_t`, and `delta_t`
- per-token discretization
- selective scan recurrence
- full Mamba block:
  - input projection
  - conv1d
  - selective scan
  - SiLU gate
  - output projection

### Verify
- operator parity on synthetic inputs
- intermediate parity against `state-spaces/mamba-130m-hf`
- full block output parity within FP32 tolerance

### Output
- `src/mamba_minimal/selective_scan.py`
- `src/mamba_minimal/model.py`
- `tests/test_selective_scan.py`

### Showcase value
This is the first point where the repo proves real implementation depth: not just understanding the idea, but rebuilding the mechanism correctly.

### Done when
- both the operator and the full block match the reference path reliably

---

## Milestone 3 — Parallel Scan
**Time:** Day 6

### Goal
Replace the naive sequential scan with parallelizable formulations and explain the tradeoff.

### Build
- prove associativity of the recurrence transform
- implement Blelloch-style scan
- implement chunked scan as the more practical version

### Verify
- compare against sequential reference on random inputs
- validate correctness in FP64 and FP32
- measure crossover behavior on GPU

### Output
- `src/mamba_minimal/parallel_scan.py`
- `notebooks/03_parallel_scan.ipynb`

### Showcase value
This milestone demonstrates algorithmic reasoning: you are not just writing code, you are explaining why the parallel version exists and when it actually helps.

### Done when
- correctness is established and the tradeoff is clearly explained

---

## Milestone 4 — Triton Kernel
**Time:** Days 7–10

### Goal
Build a Triton implementation that reduces memory traffic and improves performance.

### Phase A — Unfused baseline
Build separate kernels for:
- discretization
- scan
- output projection

This is the debugging baseline.

### Phase B — Fused kernel
Fuse discretization, scan, and output into one kernel.  
Keep intermediates in registers/SRAM whenever possible.

### Phase C — Practical scaling
- parallelize over `D`
- support realistic shapes
- add autotune knobs for chunk/block sizes

### Verify
- parity vs PyTorch across a shape and dtype matrix
- benchmark unfused vs fused versions
- compute achieved bandwidth and compare against T4 peak

### Output
- `kernels/scan_naive.py`
- `kernels/scan_fused.py`
- `benchmarks/benchmark_scan.py`

### Showcase value
This is the highest-signal milestone in the repo. It proves you can move from model code into hardware-aware optimization.

### Done when
- the fused kernel is correct and materially better than the unfused baseline

---

## Milestone 5 — Profiling and Bottleneck Analysis
**Time:** Days 11–12

### Goal
Explain why Mamba decode is memory-bound and show the evidence.

### Build
- compute arithmetic intensity
- compute MBU
- build T4 roofline plot
- compare selective scan vs matmul baseline
- measure scaling across batch size and sequence length

### Verify
- roofline position matches memory-bound expectation
- results are reproducible across repeated runs

### Output
- `notebooks/05_profiling.ipynb`
- `figures/roofline.png`

### Showcase value
This is where the project turns from “I built a kernel” into “I can explain a systems bottleneck with evidence.”

### Done when
- the notebook makes a convincing systems argument, not just a benchmark dump

---

## Milestone 6 — End-to-End Inference Demo
**Time:** Days 13–14

### Goal
Connect the implementation to user-visible inference behavior.

### Build
- autoregressive generation path for Mamba
- GPT-2 baseline for comparison
- same prompt set for both models
- measurements for:
  - time to first token
  - inter-token latency
  - peak memory usage vs context length

### Verify
- generation loop runs reliably
- memory growth chart clearly shows Mamba flat vs GPT-2 growing KV cache

### Output
- `src/mamba_minimal/generate.py`
- `benchmarks/benchmark_inference.py`
- `figures/memory_scaling.png`
- `figures/throughput_comparison.png`

### Showcase value
This is the business-end proof. It connects low-level implementation work to a concrete inference advantage that is easy to explain in interviews.

### Done when
- the comparison clearly explains the practical value of Mamba inference

---

## Milestone 7 — Packaging and Documentation
**Time:** Days 15–16

### Goal
Turn the project into a strong public engineering artifact.

### Build
- clean up naming and module structure
- ensure notebooks run from a fresh environment
- write README with:
  - project motivation
  - implementation path
  - benchmark setup
  - key findings
  - limitations
  - reproduction steps
- create an architecture diagram
- add benchmark summary tables for quick scanning

### Verify
- a fresh user can follow setup and reproduce key outputs
- figures and claims in README map back to scripts/notebooks in repo

### Output
- final repo structure
- polished README
- `figures/architecture.png`
- final figures and summary tables

### Showcase value
This milestone determines whether the project feels like a serious engineering artifact or just a pile of experiments.

### Done when
- the repo is understandable without external explanation

---

## Milestone 8 — Mamba-2 SSD (Stretch)
**Time:** Days 17–18 if available

### Goal
Understand the SSD dual view and build a minimal prototype.

### Build
- minimal PyTorch SSD implementation
- explanation of scan view vs structured matmul view
- limited parity checks

### Verify
- test on small or targeted cases first
- do not spend core-project time chasing full SSD polish

### Output
- `src/mamba_minimal/ssd.py`

### Showcase value
This is bonus depth. It is useful only if the core repo is already clean, correct, and benchmarked.

### Done when
- the implementation is correct enough to demonstrate the idea clearly

---

## 12. Weekly Exit Gates

### End of Week 1
- SSM notebook complete
- selective scan working
- Mamba block parity working
- parallel scan implemented and validated

### End of Week 2
- Triton kernel working
- fused vs unfused benchmark available
- roofline and MBU analysis complete

### End of Week 3
- inference comparison complete
- repo polished
- README public-ready

---

## 13. Portfolio Packaging Requirements

The final repo should be optimized for two reader modes:

### Mode A — Engineer skimming for technical quality
They should find:
- implementation contracts,
- tests,
- benchmark scripts,
- parity evidence,
- and clean separation between reference code and optimized code.

### Mode B — Hiring manager skimming for signal
They should find:
- a one-paragraph summary,
- a hero figure,
- 3–5 key results,
- a short “what this proves” section,
- and a clean README structure.

### Required top-of-README flow
The README should front-load value in this order:
1. one-paragraph project summary
2. hero image: memory scaling chart
3. key results table
4. what was implemented
5. how correctness was verified
6. benchmark highlights
7. reproduction steps
8. limitations and honest tradeoffs

### Required “What this proves” bullets
Add a short section in README with bullets like:
- rebuilt selective scan from scratch,
- matched official model behavior within tolerance,
- implemented and benchmarked a fused Triton kernel,
- demonstrated Mamba’s flat-memory inference behavior vs Transformer KV cache.

---

## 14. Resume and Interview Packaging

The project should make it easy to extract strong resume and interview material.

### Resume-style outcomes the repo should support
By the end, you should be able to truthfully say things like:
- Implemented Mamba selective scan from scratch in PyTorch and validated parity against official checkpoints.
- Built and benchmarked fused Triton kernels for selective scan, analyzing memory bandwidth utilization and roofline limits.
- Compared Mamba and GPT-2 inference behavior, showing flat memory scaling for recurrent state vs linear KV-cache growth.

### Interview talking points the project should enable
You should be able to explain:
- why selective scan is associative,
- why Mamba-1 decode is memory-bound,
- why fusion helps,
- why Mamba inference memory stays flat,
- and why Mamba-2 shifts toward chunked matmuls.

If the repo does not make these points obvious, the project is under-documented.

---

## 15. Risks and Mitigations

### Risk: Triton or Colab environment issues
**Mitigation:** pin versions and keep a known-good PyTorch fallback path.

### Risk: kernel debugging takes too long
**Mitigation:** keep the unfused baseline and small-shape debug cases.

### Risk: numerical drift on long sequences
**Mitigation:** add explicit long-context tests and float32 accumulation policy.

### Risk: benchmarks are noisy or misleading
**Mitigation:** use warmup, repeated runs, and fixed benchmark grids.

### Risk: timeline slips
**Mitigation:** protect P0 scope first and treat SSD as optional.

---

## 16. Final README Structure

The final README should answer these questions in order:

1. What is Mamba?
2. Why build it from scratch?
3. What exactly was implemented?
4. How was correctness verified?
5. What did the kernel benchmarks show?
6. Where does Mamba win over GPT-2?
7. What are the limitations?
8. How can someone reproduce the results?

---

## 17. Recommended References

- Mamba paper: https://arxiv.org/abs/2312.00752
- Annotated Mamba (Hard Way): https://srush.github.io/annotated-mamba/hard.html
- Official Mamba repo: https://github.com/state-spaces/mamba
- mamba.py reference: https://github.com/alxndrTL/mamba.py
- Mamba-2 blog: https://tridao.me/blog/2024/mamba2-part3-algorithm/
- PyTorch SSD fusion blog: https://pytorch.org/blog/accelerating-mamba2-with-kernel-fusion/
- HuggingFace validation model: https://huggingface.co/state-spaces/mamba-130m-hf

---

## 18. Project-Level Definition of Done

The project is done when:
- all P0 items are implemented,
- correctness claims are backed by tests,
- performance claims are backed by scripts and figures,
- the README reflects what the code actually does,
- and a fresh developer can reproduce the main results without asking for help.

### Final quality bar
A strong engineer should be able to read the repo and think:
- this is technically correct,
- this person knows how to validate systems work,
- and this project tells a clean, credible story.

That is the bar.
