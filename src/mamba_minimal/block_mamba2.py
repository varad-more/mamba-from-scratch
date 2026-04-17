"""Mamba-2 block + model — pure PyTorch + our SSD scan.

Architecture difference from Mamba-1:
  * ``A`` is a **scalar per head** (not a per-(D, N) matrix).
  * ``B`` / ``C`` are shared across heads within a *group* (``ngroups`` <<
    ``nheads`` in general; 1 group for mamba2-130m, 8 for the 2.7b).
  * ``in_proj`` produces a concatenated ``[z, x, B, C, dt]`` in one GEMM.
  * Depthwise conv runs over ``[x, B, C]`` together.
  * A gated RMSNorm sits between the SSM output ``y`` and ``out_proj``.

We load weights from the ``mamba_ssm`` ``MambaLMHeadModel`` checkpoint
directly — the on-disk layout is the same as the HF release.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from mamba_minimal.scan_ssd import selective_scan_ssd


@dataclass
class Mamba2Config:
    d_model: int
    n_layer: int
    vocab_size: int
    d_state: int = 128
    d_conv: int = 4
    expand: int = 2
    headdim: int = 64
    ngroups: int = 1
    chunk_size: int = 256
    rms_norm_eps: float = 1e-5
    tie_embeddings: bool = True

    @property
    def d_inner(self) -> int:
        return self.expand * self.d_model

    @property
    def nheads(self) -> int:
        return self.d_inner // self.headdim


def _rms_norm(x: Tensor, weight: Tensor, eps: float) -> Tensor:
    dtype = x.dtype
    xf = x.float()
    rms = xf.pow(2).mean(-1, keepdim=True).add(eps).rsqrt()
    return (xf * rms).to(dtype) * weight


def _gated_rms_norm(y: Tensor, z: Tensor, weight: Tensor, eps: float) -> Tensor:
    """Mamba-2 default: norm(x * silu(z)) — norm_before_gate=False."""
    return _rms_norm(y * F.silu(z), weight, eps)


class Mamba2Block(nn.Module):
    def __init__(self, cfg: Mamba2Config):
        super().__init__()
        self.cfg = cfg
        d_in_proj = 2 * cfg.d_inner + 2 * cfg.ngroups * cfg.d_state + cfg.nheads
        self.in_proj = nn.Linear(cfg.d_model, d_in_proj, bias=False)

        d_conv_channels = cfg.d_inner + 2 * cfg.ngroups * cfg.d_state
        self.conv1d = nn.Conv1d(
            d_conv_channels, d_conv_channels, kernel_size=cfg.d_conv,
            groups=d_conv_channels, padding=cfg.d_conv - 1, bias=True,
        )
        self.dt_bias = nn.Parameter(torch.zeros(cfg.nheads))
        self.A_log = nn.Parameter(torch.zeros(cfg.nheads))
        self.D = nn.Parameter(torch.ones(cfg.nheads))
        self.norm_weight = nn.Parameter(torch.ones(cfg.d_inner))
        self.out_proj = nn.Linear(cfg.d_inner, cfg.d_model, bias=False)

    # ----- shared helpers -----
    def _split(self, zxbcdt: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Split in_proj output into (z, xBC, dt)."""
        cfg = self.cfg
        return torch.split(
            zxbcdt,
            [cfg.d_inner, cfg.d_inner + 2 * cfg.ngroups * cfg.d_state, cfg.nheads],
            dim=-1,
        )

    # ----- prefill -----
    def forward(
        self, u: Tensor, return_state: bool = False
    ) -> Tensor | tuple[Tensor, tuple[Tensor, Tensor]]:
        """u: (B, L, d_model). Returns out (B, L, d_model)."""
        cfg = self.cfg
        B_, L, _ = u.shape
        zxbcdt = self.in_proj(u)
        z, xBC, dt = self._split(zxbcdt)                        # z:(B,L,D), xBC:(B,L,D+2GN), dt:(B,L,H)

        # Depthwise causal conv: pad on the left, run full conv, trim right overhang.
        xBC_t = xBC.transpose(1, 2)                             # (B, C, L)
        xBC_conv = self.conv1d(xBC_t)[:, :, :L].transpose(1, 2)  # (B, L, C)
        xBC_act = F.silu(xBC_conv)

        x, B, C = torch.split(
            xBC_act,
            [cfg.d_inner, cfg.ngroups * cfg.d_state, cfg.ngroups * cfg.d_state],
            dim=-1,
        )
        x = rearrange(x, "b l (h p) -> b l h p", p=cfg.headdim)
        B = rearrange(B, "b l (g n) -> b l g n", g=cfg.ngroups)
        C = rearrange(C, "b l (g n) -> b l g n", g=cfg.ngroups)
        A = -torch.exp(self.A_log.float())                      # (H,)

        y = selective_scan_ssd(
            x=x, dt=dt, A=A, B=B, C=C, chunk_size=cfg.chunk_size,
            D=self.D, dt_bias=self.dt_bias, dt_softplus=True,
            return_final_state=return_state,
        )
        if return_state:
            y, ssm_state = y
        y = rearrange(y, "b l h p -> b l (h p)").to(u.dtype)
        y = _gated_rms_norm(y, z, self.norm_weight, cfg.rms_norm_eps)
        out = self.out_proj(y)
        if return_state:
            # conv_state = last d_conv - 1 cols of xBC (pre-conv input), left-padded.
            # We return the raw xBC so the decoder can roll it.
            conv_state = xBC[:, -cfg.d_conv + 1:, :].transpose(1, 2).contiguous()
            if conv_state.shape[-1] < cfg.d_conv - 1:
                pad = cfg.d_conv - 1 - conv_state.shape[-1]
                conv_state = F.pad(conv_state, (pad, 0))
            return out, (conv_state, ssm_state)
        return out

    # ----- decode step -----
    def step(
        self, u_t: Tensor, state: tuple[Tensor, Tensor],
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        """u_t: (B, d_model). Returns (out (B, d_model), new_state)."""
        cfg = self.cfg
        conv_state, ssm_state = state                            # conv_state:(B, C, d_conv-1), ssm:(B, H, P, N)
        zxbcdt = self.in_proj(u_t)                               # (B, d_in_proj)
        z, xBC, dt = self._split(zxbcdt)                         # z:(B,D); xBC:(B,C); dt:(B,H)

        # Roll conv state: append xBC, take last d_conv samples.
        xBC = xBC.unsqueeze(-1)                                  # (B, C, 1)
        conv_buf = torch.cat([conv_state, xBC], dim=-1)          # (B, C, d_conv)
        # Depthwise conv: (conv_buf * weight).sum over kernel
        w = self.conv1d.weight.squeeze(1)                        # (C, d_conv)
        xBC_conv = (conv_buf * w).sum(-1) + self.conv1d.bias     # (B, C)
        xBC_act = F.silu(xBC_conv)
        new_conv_state = conv_buf[:, :, 1:]                      # (B, C, d_conv-1)

        x, B, C = torch.split(
            xBC_act,
            [cfg.d_inner, cfg.ngroups * cfg.d_state, cfg.ngroups * cfg.d_state],
            dim=-1,
        )
        # Broadcast groups → heads
        x_h = x.view(x.shape[0], cfg.nheads, cfg.headdim)        # (B, H, P)
        B_g = B.view(B.shape[0], cfg.ngroups, cfg.d_state)       # (B, G, N)
        C_g = C.view(C.shape[0], cfg.ngroups, cfg.d_state)
        B_h = B_g.repeat_interleave(cfg.nheads // cfg.ngroups, dim=1)  # (B, H, N)
        C_h = C_g.repeat_interleave(cfg.nheads // cfg.ngroups, dim=1)

        A = -torch.exp(self.A_log.float())                       # (H,)
        dt_f = F.softplus(dt.float() + self.dt_bias)             # (B, H)
        dA = torch.exp(dt_f * A)                                 # (B, H)
        dB = dt_f.unsqueeze(-1) * B_h.float()                    # (B, H, N)

        # ssm_state: (B, H, P, N). Update in fp32.
        s = ssm_state.float()
        s = dA[..., None, None] * s + dB.unsqueeze(-2) * x_h.unsqueeze(-1).float()
        y = (s * C_h.unsqueeze(-2).float()).sum(dim=-1)          # (B, H, P)
        y = y + self.D.view(1, -1, 1).float() * x_h.float()      # (B, H, P)
        y = y.to(u_t.dtype)
        y = y.reshape(y.shape[0], cfg.d_inner)                   # (B, D)

        y = _gated_rms_norm(y, z, self.norm_weight, cfg.rms_norm_eps)
        out = self.out_proj(y)
        return out, (new_conv_state, s)

    # ----- state helpers -----
    def empty_state(self, batch: int, device, dtype=torch.float32) -> tuple[Tensor, Tensor]:
        cfg = self.cfg
        d_conv_channels = cfg.d_inner + 2 * cfg.ngroups * cfg.d_state
        conv_state = torch.zeros(batch, d_conv_channels, cfg.d_conv - 1, device=device, dtype=dtype)
        ssm_state = torch.zeros(batch, cfg.nheads, cfg.headdim, cfg.d_state, device=device, dtype=torch.float32)
        return conv_state, ssm_state


class Mamba2ResidualBlock(nn.Module):
    def __init__(self, cfg: Mamba2Config):
        super().__init__()
        self.cfg = cfg
        self.norm_weight = nn.Parameter(torch.ones(cfg.d_model))
        self.mixer = Mamba2Block(cfg)

    def forward(self, u: Tensor, return_state: bool = False):
        h = _rms_norm(u, self.norm_weight, self.cfg.rms_norm_eps)
        if return_state:
            y, state = self.mixer(h, return_state=True)
            return u + y, state
        return u + self.mixer(h)

    def step(self, u_t: Tensor, state):
        h = _rms_norm(u_t, self.norm_weight, self.cfg.rms_norm_eps)
        y, new_state = self.mixer.step(h, state)
        return u_t + y, new_state


class Mamba2Model(nn.Module):
    def __init__(self, cfg: Mamba2Config):
        super().__init__()
        self.cfg = cfg
        self.embeddings = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList([Mamba2ResidualBlock(cfg) for _ in range(cfg.n_layer)])
        self.norm_f_weight = nn.Parameter(torch.ones(cfg.d_model))
        if not cfg.tie_embeddings:
            self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        else:
            self.lm_head = None

    def forward(self, input_ids: Tensor, return_state: bool = False):
        h = self.embeddings(input_ids)
        states = [] if return_state else None
        for layer in self.layers:
            if return_state:
                h, s = layer(h, return_state=True)
                states.append(s)
            else:
                h = layer(h)
        h = _rms_norm(h, self.norm_f_weight, self.cfg.rms_norm_eps)
        logits = h @ self.embeddings.weight.T if self.lm_head is None else self.lm_head(h)
        if return_state:
            return logits, states
        return logits

    def step(self, token: Tensor, states):
        h = self.embeddings(token)
        new_states = []
        for layer, s in zip(self.layers, states):
            h, ns = layer.step(h, s)
            new_states.append(ns)
        h = _rms_norm(h, self.norm_f_weight, self.cfg.rms_norm_eps)
        logits = h @ self.embeddings.weight.T if self.lm_head is None else self.lm_head(h)
        return logits, new_states

    def empty_state(self, batch: int, device, dtype=torch.float32):
        return [layer.mixer.empty_state(batch, device, dtype) for layer in self.layers]

    # ----- loader -----
    @classmethod
    def from_pretrained(
        cls, name: str = "state-spaces/mamba2-130m",
        device: str | torch.device = "cuda", dtype: torch.dtype = torch.float32,
    ) -> "Mamba2Model":
        from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
        ref = MambaLMHeadModel.from_pretrained(name, device=device, dtype=dtype)
        rcfg = ref.config
        # Grab structural params from the first mixer.
        m0 = ref.backbone.layers[0].mixer
        cfg = Mamba2Config(
            d_model=rcfg.d_model,
            n_layer=rcfg.n_layer,
            vocab_size=m0.out_proj.weight.shape[0] if False else ref.lm_head.weight.shape[0],
            d_state=m0.d_state,
            d_conv=m0.d_conv,
            expand=2,
            headdim=m0.headdim,
            ngroups=m0.ngroups,
            chunk_size=m0.chunk_size,
            tie_embeddings=getattr(rcfg, "tie_embeddings", True),
        )
        model = cls(cfg).to(device=device, dtype=dtype)
        model._load_from_ref(ref)
        return model.eval()

    def _load_from_ref(self, ref) -> None:
        """Copy weights out of a mamba_ssm MambaLMHeadModel into this module."""
        self.embeddings.weight.data.copy_(ref.backbone.embedding.weight.data)
        self.norm_f_weight.data.copy_(ref.backbone.norm_f.weight.data)
        if self.lm_head is not None:
            self.lm_head.weight.data.copy_(ref.lm_head.weight.data)

        for i, (dst, src) in enumerate(zip(self.layers, ref.backbone.layers)):
            dst.norm_weight.data.copy_(src.norm.weight.data)
            m = dst.mixer
            sm = src.mixer
            m.in_proj.weight.data.copy_(sm.in_proj.weight.data)
            m.conv1d.weight.data.copy_(sm.conv1d.weight.data)
            m.conv1d.bias.data.copy_(sm.conv1d.bias.data)
            m.dt_bias.data.copy_(sm.dt_bias.data)
            m.A_log.data.copy_(sm.A_log.data)
            m.D.data.copy_(sm.D.data)
            m.norm_weight.data.copy_(sm.norm.weight.data)
            m.out_proj.weight.data.copy_(sm.out_proj.weight.data)
