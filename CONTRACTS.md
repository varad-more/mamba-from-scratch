# Implementation Contracts

This file defines the shape, dtype, and tolerance contracts used across the project.

## Tensor notation

- `B`: batch size
- `L`: sequence length
- `D`: channel/model dimension
- `N`: SSM state dimension

## Core shape contracts

### Selective scan

- `u`: `(B, D, L)`
- `delta`: `(B, D, L)`
- `A`: `(D, N)`
- `B_t`: `(B, N, L)` or `(B, D, N, L)`
- `C_t`: `(B, N, L)` or `(B, D, N, L)`
- `D_skip`: `(D,)` (optional)
- output `y`: `(B, D, L)`

### Mamba block

- input `hidden_states`: `(B, L, d_model)`
- output `hidden_states`: `(B, L, d_model)`

## Dtype contracts

- Math-level notebooks: `float64`
- Reference implementation: `float32`
- Kernel path may accept mixed precision, but accumulation should stay in `float32`

## Tolerance contracts

- FP64 parity: tight tolerance, typically `atol=1e-10`
- FP32 parity: default `atol=1e-5`, `rtol=1e-4`
- Mixed precision parity: loosen only when justified by kernel precision limits

## Stability contracts

- Prefer stable solve-style discretization approaches over explicit matrix inverse formulas when practical.
- Keep `delta` positive via softplus in selective scan path by default.
- Run long-sequence tests when introducing scan or kernel changes.
