"""Mamba-2 State Space Duality (SSD) scan — block-structured matmul form.

The SSD insight is that the Mamba-2 recurrence (scalar-per-head ``A``,
shared ``B/C`` across heads in a group) can be evaluated as a sequence of
dense matmuls over *chunks* of the sequence instead of a sequential scan.
This exposes GEMM hardware and is what lets Mamba-2 close the gap with
attention on modern GPUs.

Shapes (matching ``mamba_ssm.ops.triton.ssd_combined.ssd_chunk_scan_combined_ref``):
    x:   (B, L, H, P)                  H = nheads, P = headdim
    dt:  (B, L, H)
    A:   (H,)                           scalar per head (negative real)
    B:   (B, L, G, N)                   G = ngroups, N = dstate
    C:   (B, L, G, N)
    D:   (H,) or (H, P)                optional skip
    z:   (B, L, H, P)                  optional SiLU gate

Math (within a chunk of size ``S``, head ``h``, group that ``h`` belongs to):

    dA_i   = dt_i * A_h                        scalar
    dB_i   = dt_i * B_i                        (N,)
    h_i    = exp(dA_i) h_{i-1} + dB_i * x_i
    y_i    = C_i^T h_i

Unrolling the recurrence and writing cumulative decays as

    α_{i→j}  = exp( sum_{u=i+1..j} dA_u )

gives, for the *intra-chunk* contribution,

    y_j = sum_{i<=j}  α_{i→j} * dt_i * (C_j · B_i) * x_i

i.e. ``Y = (L ⊙ (C Bᵀ)) · (dt ⊙ X)`` where ``L`` is the causal
lower-triangular matrix of chunk-local decays. That's a structured matmul.

Across chunks, each chunk summarises its final state

    s_c = sum_{i in chunk} α_{i→end} * dt_i * B_i * x_i     ("chunk_state")

and states are passed via a second structured matmul over the
``nchunks × nchunks`` lower-triangular matrix of inter-chunk decays
(``state_passing``). The carry-in contribution to ``y`` in chunk ``c`` is
then ``exp(dA_cumsum_in_chunk) * (C · s_{c-1})``.

Three einsums total:
    1. chunk_state       — (B, G, N) and (B, H, P) → per-chunk states
    2. state_passing     — structured matmul across chunks
    3. chunk_scan        — intra-chunk output + carry-in

This file is pure PyTorch. Parity vs ``mamba_ssm``'s own reference is
checked in ``tests/test_scan_ssd.py``; a Triton fused version can later
replace the hot einsum — the algebra is identical.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import Tensor


def _broadcast_groups_to_heads(B: Tensor, C: Tensor, nheads: int) -> tuple[Tensor, Tensor]:
    """Expand (B, L, G, N) → (B, L, H, N) by repeating each group across its heads."""
    ngroups = B.shape[2]
    if nheads == ngroups:
        return B, C
    if nheads % ngroups != 0:
        raise ValueError(f"nheads ({nheads}) must be a multiple of ngroups ({ngroups})")
    rep = nheads // ngroups
    B = repeat(B, "b l g n -> b l (g r) n", r=rep)
    C = repeat(C, "b l g n -> b l (g r) n", r=rep)
    return B, C


def _chunk_state(
    B_exp: Tensor,       # (B, L, H, N)
    x: Tensor,           # (B, L, H, P)
    dt: Tensor,          # (B, H, C, S)
    dA_cumsum: Tensor,   # (B, H, C, S)
) -> Tensor:
    """Per-chunk final state. Returns (B, C, H, P, N)."""
    chunk_size = dt.shape[-1]
    # Decay weight for token i within chunk = exp(sum_{u=i+1..end} dA_u)
    # = exp(dA_cumsum[end] - dA_cumsum[i])
    decay_states = torch.exp(dA_cumsum[:, :, :, -1:] - dA_cumsum)  # (B, H, C, S)
    B_c = rearrange(B_exp, "b (c l) h n -> b c l h n", l=chunk_size)
    x_c = rearrange(x, "b (c l) h p -> b c l h p", l=chunk_size)
    dtype = x.dtype
    return torch.einsum(
        "bclhn,bhcl,bhcl,bclhp->bchpn",
        B_c.to(dtype), decay_states.to(dtype), dt.to(dtype), x_c,
    )


def _state_passing(
    states: Tensor,            # (B, C, H, P*N)
    dA_chunk_sum: Tensor,      # (B, H, C) — total decay per chunk
    initial_states: Optional[Tensor] = None,  # (B, H, P*N)
) -> tuple[Tensor, Tensor]:
    """Propagate per-chunk states. Returns (per-chunk carry-in, final state)."""
    if initial_states is None:
        initial_states = torch.zeros_like(states[:, 0])
    states = torch.cat([initial_states.unsqueeze(1), states], dim=1)   # (B, C+1, H, D)
    dA = F.pad(dA_chunk_sum, (1, 0))                                    # (B, H, C+1)
    dA = torch.cumsum(dA, dim=-1)                                       # (B, H, C+1)
    # Structured matmul over chunks: (C+1, C+1) lower-triangular decay.
    seg = dA.unsqueeze(-1) - dA.unsqueeze(-2)                           # (B, H, C+1, C+1)
    decay = torch.exp(seg)
    nchunks = dA.shape[-1]
    mask = torch.tril(torch.ones(nchunks, nchunks, device=states.device, dtype=torch.bool))
    decay = decay.masked_fill(~mask, 0.0)
    out = torch.einsum("bhzc,bchd->bzhd", decay.to(states.dtype), states)
    # out[:, :-1] → carry-in for chunk c (state at its left edge)
    # out[:, -1]  → final state
    return out[:, :-1], out[:, -1]


def _chunk_scan(
    B_exp: Tensor,       # (B, L, H, N)
    C_exp: Tensor,       # (B, L, H, N)
    x: Tensor,           # (B, L, H, P)
    dt: Tensor,          # (B, H, C, S)
    dA_cumsum: Tensor,   # (B, H, C, S)
    prev_states: Tensor, # (B, C, H, P, N) — carry-in per chunk
    D: Optional[Tensor], # (H,) or (H, P)
    z: Optional[Tensor], # (B, L, H, P)
) -> Tensor:
    """Compute y within all chunks (intra-chunk structured matmul + carry-in)."""
    chunk_size = dt.shape[-1]
    nchunks = dt.shape[-2]
    dtype = x.dtype
    C_c = rearrange(C_exp, "b (c l) h n -> b c l h n", c=nchunks)
    B_c = rearrange(B_exp, "b (c s) h n -> b c s h n", c=nchunks)
    x_c = rearrange(x, "b (c s) h p -> b c s h p", c=nchunks)

    # 1) Intra-chunk: Y = (L ⊙ (C Bᵀ)) · (dt ⊙ X).
    CB = torch.einsum("bclhn,bcshn->bchls", C_c, B_c)                   # (B, C, H, S, S)
    # L: causal lower-tri with exp(dA_cumsum[l] - dA_cumsum[s]) for s<=l.
    seg = dA_cumsum.unsqueeze(-1) - dA_cumsum.unsqueeze(-2)             # (B, H, C, S, S)
    L = torch.exp(seg)
    L = rearrange(L, "b h c l s -> b c h l s")
    scores = CB * L
    causal = torch.tril(torch.ones(chunk_size, chunk_size, device=x.device, dtype=torch.bool))
    scores = scores.masked_fill(~causal, 0.0)
    y_intra = torch.einsum(
        "bchls,bhcs,bcshp->bclhp",
        scores.to(dtype), dt.to(dtype), x_c,
    )

    # 2) Carry-in from prev chunk: y += exp(dA_cumsum) * (C · prev_state)
    state_decay = torch.exp(rearrange(dA_cumsum, "b h c l -> b c l h 1"))
    y_carry = torch.einsum(
        "bclhn,bchpn->bclhp",
        C_c, prev_states.to(C_c.dtype),
    ) * state_decay
    y = y_intra + y_carry
    y = rearrange(y, "b c l h p -> b (c l) h p")

    if D is not None:
        D_ = D.unsqueeze(-1) if D.dim() == 1 else D           # (H, P)
        y = y + x * D_
    if z is not None:
        y = y * F.silu(z)
    return y


def selective_scan_ssd(
    x: Tensor,              # (B, L, H, P)
    dt: Tensor,             # (B, L, H)
    A: Tensor,              # (H,)
    B: Tensor,              # (B, L, G, N)
    C: Tensor,              # (B, L, G, N)
    chunk_size: int,
    D: Optional[Tensor] = None,
    z: Optional[Tensor] = None,
    dt_bias: Optional[Tensor] = None,
    dt_softplus: bool = False,
    initial_states: Optional[Tensor] = None,   # (B, H, P, N)
    return_final_state: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """Mamba-2 SSD scan as three structured matmuls + a causal mask.

    Matches the semantics of ``mamba_ssm.ops.triton.ssd_combined
    .ssd_chunk_scan_combined_ref``. Bit-exact parity is verified in
    ``tests/test_scan_ssd.py``.
    """
    batch, seqlen, nheads, headdim = x.shape
    dstate = B.shape[-1]

    # Right-pad to a multiple of chunk_size.
    pad = (-seqlen) % chunk_size
    if pad:
        x = F.pad(x, (0, 0, 0, 0, 0, pad))
        dt = F.pad(dt, (0, 0, 0, pad))
        B = F.pad(B, (0, 0, 0, 0, 0, pad))
        C = F.pad(C, (0, 0, 0, 0, 0, pad))
        if z is not None:
            z = F.pad(z, (0, 0, 0, 0, 0, pad))
    L_padded = seqlen + pad

    # dt preprocessing (bias + softplus). Cast to fp32 *before* cumsum.
    dt_f = rearrange(dt, "b (c l) h -> b h c l", l=chunk_size).float()
    if dt_bias is not None:
        dt_f = dt_f + dt_bias.view(1, -1, 1, 1)
    if dt_softplus:
        dt_f = F.softplus(dt_f)

    dA = dt_f * A.view(1, -1, 1, 1)                            # (B, H, C, S)
    dA_cumsum = torch.cumsum(dA, dim=-1)                        # (B, H, C, S)

    # Broadcast group-shared B/C over heads.
    B_exp, C_exp = _broadcast_groups_to_heads(B, C, nheads)

    # 1. Per-chunk states.
    states = _chunk_state(B_exp, x, dt_f, dA_cumsum)            # (B, C, H, P, N)

    # 2. State passing across chunks.
    states_flat = rearrange(states, "b c h p n -> b c h (p n)")
    init_flat = (
        rearrange(initial_states, "b h p n -> b h (p n)")
        if initial_states is not None else None
    )
    carry_flat, final_flat = _state_passing(
        states_flat, dA_cumsum[:, :, :, -1], initial_states=init_flat,
    )
    carry = rearrange(carry_flat, "b c h (p n) -> b c h p n", n=dstate)
    final_state = rearrange(final_flat, "b h (p n) -> b h p n", n=dstate)

    # 3. Per-chunk output (intra-chunk matmul + carry-in).
    y = _chunk_scan(B_exp, C_exp, x, dt_f, dA_cumsum, carry, D=D, z=z)

    y = y[:, :seqlen]
    if return_final_state:
        return y, final_state
    return y
