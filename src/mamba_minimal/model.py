from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .backend import ScanBackend, select_scan_backend
from .discretization import inverse_softplus
from .scan_naive import selective_scan_naive
from .scan_parallel import selective_scan_parallel
from .selective_scan import selective_scan_ref

try:
    from kernels.scan_fused import selective_scan_fused

    _FUSED_AVAILABLE = True
except Exception:
    _FUSED_AVAILABLE = False


@dataclass(slots=True)
class MambaBlockConfig:
    d_model: int
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    dt_rank: int | str = "auto"
    bias: bool = False
    conv_bias: bool = True

    @property
    def d_inner(self) -> int:
        return self.expand * self.d_model

    def resolved_dt_rank(self) -> int:
        if self.dt_rank == "auto":
            return math.ceil(self.d_model / 16)
        return int(self.dt_rank)


class MambaBlock(nn.Module):
    """Minimal Mamba block close to the original mixer design.

    This module intentionally prioritizes readability and parity with the
    reference implementation over raw performance.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: int | str = "auto",
        bias: bool = False,
        conv_bias: bool = True,
        use_fused_kernel: bool = False,
        scan_backend: ScanBackend = "auto",
    ) -> None:
        super().__init__()
        self.config = MambaBlockConfig(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dt_rank=dt_rank,
            bias=bias,
            conv_bias=conv_bias,
        )
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = self.config.d_inner
        self.dt_rank = self.config.resolved_dt_rank()
        if use_fused_kernel and scan_backend != "auto":
            raise ValueError("use_fused_kernel and scan_backend cannot both be set explicitly")
        if use_fused_kernel:
            scan_backend = "fused"
        self.scan_backend: ScanBackend = scan_backend

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            bias=conv_bias,
        )
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        a_init = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(a_init))
        self.D = nn.Parameter(torch.ones(self.d_inner, dtype=torch.float32))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.xavier_uniform_(self.x_proj.weight)
        nn.init.xavier_uniform_(self.dt_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.in_proj.bias is not None:
            nn.init.zeros_(self.in_proj.bias)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)
        if self.conv1d.bias is not None:
            nn.init.zeros_(self.conv1d.bias)

        # Keep the initial time steps positive but small.
        dt_init = torch.full((self.d_inner,), 0.1, dtype=torch.float32)
        with torch.no_grad():
            self.dt_proj.bias.copy_(inverse_softplus(dt_init))
            self.conv1d.weight.zero_()
            self.conv1d.weight[:, 0, -1] = 1.0

    def _causal_conv(self, x: Tensor) -> Tensor:
        x_conv = self.conv1d(x.transpose(1, 2))
        x_conv = x_conv[..., : x.shape[1]]
        return x_conv.transpose(1, 2)

    def forward(self, hidden_states: Tensor) -> Tensor:
        if hidden_states.ndim != 3:
            raise ValueError(
                f"hidden_states must have shape (B, L, D), got {tuple(hidden_states.shape)}"
            )
        if hidden_states.shape[-1] != self.d_model:
            raise ValueError(
                f"Expected hidden size {self.d_model}, got {hidden_states.shape[-1]}"
            )

        xz = self.in_proj(hidden_states)
        x, z = xz.chunk(2, dim=-1)

        x = self._causal_conv(x)
        x = F.silu(x)

        x_proj = self.x_proj(x)
        dt, B, C = torch.split(x_proj, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj(dt)

        A = -torch.exp(self.A_log.float())
        u = x.transpose(1, 2)
        delta = dt.transpose(1, 2)
        B_scan = B.transpose(1, 2)
        C_scan = C.transpose(1, 2)
        z_scan = z.transpose(1, 2)
        dispatch = select_scan_backend(
            u,
            delta,
            A,
            B_scan,
            C_scan,
            D=self.D.float(),
            z=z_scan,
            requested_backend=self.scan_backend,
            triton_available=_FUSED_AVAILABLE,
        )

        if dispatch.selected_backend == "triton-fused":
            y = selective_scan_fused(
                u=u,
                delta=delta,
                A=A,
                B=B_scan,
                C=C_scan,
                D=self.D.float(),
                z=z_scan,
            )
        else:
            y = selective_scan_ref(
                u=u,
                delta=delta,
                A=A,
                B=B_scan,
                C=C_scan,
                D=self.D.float(),
                z=z_scan,
            )
        y = y.transpose(1, 2)
        return self.out_proj(y)

    def load_official_mixer_state_dict(self, state_dict: dict[str, Tensor]) -> None:
        """Load a state dict with the same parameter names as the official mixer.

        This is useful when extracting a single mixer block from a HuggingFace or
        `state-spaces/mamba` checkpoint and mapping keys manually.
        """

        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        if missing:
            raise ValueError(f"Missing parameters while loading official mixer state: {missing}")
        if unexpected:
            raise ValueError(
                f"Unexpected parameters while loading official mixer state: {unexpected}"
            )

    @classmethod
    def from_official_config(cls, config: Any) -> "MambaBlock":
        """Create a block from a config object exposing Mamba-like attributes."""

        return cls(
            d_model=int(getattr(config, "hidden_size")),
            d_state=int(getattr(config, "state_size", getattr(config, "d_state", 16))),
            d_conv=int(getattr(config, "conv_kernel", getattr(config, "d_conv", 4))),
            expand=int(getattr(config, "expand", 2)),
            dt_rank=getattr(config, "time_step_rank", getattr(config, "dt_rank", "auto")),
            bias=bool(getattr(config, "use_bias", False)),
            conv_bias=bool(getattr(config, "use_conv_bias", True)),
        )

    # ------------------------------------------------------------------
    # State-carrying inference path (Phase 2).
    #
    # The public ``forward`` above stays backward-compatible (legacy scan,
    # stateless, softplus-inside-scan). Everything below is the new path
    # used by :class:`MambaModel` and the state-carrying generate loop.
    # ------------------------------------------------------------------

    def allocate_inference_state(
        self,
        batch_size: int,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> tuple[Tensor, Tensor]:
        """Return zero-initialized ``(conv_state, ssm_state)`` for decode."""

        conv_state = torch.zeros(
            batch_size, self.d_inner, self.d_conv, device=device, dtype=dtype
        )
        ssm_state = torch.zeros(
            batch_size, self.d_inner, self.d_state, device=device, dtype=torch.float32
        )
        return conv_state, ssm_state

    def _apply_scan(
        self,
        scan_impl: str,
        u: Tensor,
        delta: Tensor,
        A: Tensor,
        B: Tensor,
        C: Tensor,
        D_skip: Tensor,
        z: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Run the selected selective scan and return ``(y, last_state)``."""

        if scan_impl == "parallel":
            return selective_scan_parallel(
                u, delta, A, B, C, D=D_skip, z=z,
                delta_softplus=False, return_last_state=True,
            )
        if scan_impl == "naive":
            return selective_scan_naive(
                u, delta, A, B, C, D=D_skip, z=z,
                delta_softplus=False, return_last_state=True,
            )
        raise ValueError(f"unknown scan_impl: {scan_impl!r}")

    def forward_with_state(
        self,
        hidden_states: Tensor,
        conv_state: Tensor | None = None,
        ssm_state: Tensor | None = None,
        scan_impl: str = "parallel",
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Run the block and return (output, new_conv_state, new_ssm_state).

        Matches HuggingFace's slow_forward conventions: softplus is applied
        to ``dt_proj`` output before the scan. If states are ``None`` they
        are allocated as zeros. ``conv_state`` stores the last ``d_conv``
        post-in_proj inputs (left-padded with zeros after prefill) so the
        next ``step()`` can continue the causal conv without replay.
        """

        if hidden_states.ndim != 3:
            raise ValueError(
                f"hidden_states must have shape (B, L, D), got {tuple(hidden_states.shape)}"
            )
        batch, seq_len, d_model = hidden_states.shape
        if d_model != self.d_model:
            raise ValueError(f"Expected hidden size {self.d_model}, got {d_model}")

        device = hidden_states.device
        if conv_state is None or ssm_state is None:
            alloc_conv, alloc_ssm = self.allocate_inference_state(
                batch, device=device, dtype=hidden_states.dtype
            )
            conv_state = alloc_conv if conv_state is None else conv_state
            ssm_state = alloc_ssm if ssm_state is None else ssm_state

        xz = self.in_proj(hidden_states)  # (B, L, 2 * d_inner)
        x, z = xz.chunk(2, dim=-1)  # each (B, L, d_inner)
        # Post-in_proj stream goes through conv in (B, d_inner, L) layout.
        x_t = x.transpose(1, 2)  # (B, d_inner, L)

        # Causal conv over the full sequence.
        x_conv = self.conv1d(x_t)[..., :seq_len]  # (B, d_inner, L)
        x_conv = F.silu(x_conv)

        # Build new conv_state: last d_conv inputs, left-padded with zeros.
        pad_amount = max(self.d_conv - seq_len, 0)
        new_conv_state = F.pad(x_t[..., -self.d_conv:], (pad_amount, 0))

        # Selective scan inputs: all in (B, d_inner, L) / (B, d_state, L).
        x_conv_bld = x_conv.transpose(1, 2)  # (B, L, d_inner) for x_proj
        x_proj_out = self.x_proj(x_conv_bld)
        dt, B_ssm, C_ssm = torch.split(
            x_proj_out, [self.dt_rank, self.d_state, self.d_state], dim=-1,
        )
        dt = F.softplus(self.dt_proj(dt))  # (B, L, d_inner)

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        delta = dt.transpose(1, 2)  # (B, d_inner, L)
        B_scan = B_ssm.transpose(1, 2).contiguous()  # (B, d_state, L)
        C_scan = C_ssm.transpose(1, 2).contiguous()  # (B, d_state, L)
        z_scan = z.transpose(1, 2)  # (B, d_inner, L)

        y, new_ssm_state = self._apply_scan(
            scan_impl, x_conv, delta, A, B_scan, C_scan, self.D.float(), z_scan,
        )
        # y: (B, d_inner, L); new_ssm_state: (B, d_inner, d_state) fp32.
        out = self.out_proj(y.transpose(1, 2))  # (B, L, d_model)
        return out, new_conv_state, new_ssm_state

    def step(
        self,
        hidden_state_t: Tensor,
        conv_state: Tensor,
        ssm_state: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Advance one token. ``hidden_state_t`` is ``(B, d_model)`` or ``(B, 1, d_model)``.

        Returns ``(output_t, new_conv_state, new_ssm_state)`` with
        ``output_t`` shape ``(B, d_model)``. This is the ``O(1)``-per-step
        decode path; the SSM state and conv buffer are the full "KV-cache
        equivalent" — nothing else needs to be remembered.
        """

        if hidden_state_t.ndim == 2:
            x = hidden_state_t.unsqueeze(1)  # (B, 1, d_model)
        elif hidden_state_t.ndim == 3 and hidden_state_t.shape[1] == 1:
            x = hidden_state_t
        else:
            raise ValueError(
                f"step expects (B, d_model) or (B, 1, d_model), got {tuple(hidden_state_t.shape)}"
            )

        xz = self.in_proj(x)  # (B, 1, 2 * d_inner)
        x_t, z = xz.chunk(2, dim=-1)
        x_t = x_t.squeeze(1)  # (B, d_inner)
        z = z.squeeze(1)  # (B, d_inner)

        # Roll conv_state left and append current x_t at the rightmost slot.
        new_conv_state = torch.roll(conv_state, shifts=-1, dims=-1)
        new_conv_state[:, :, -1] = x_t
        # Depthwise conv via elementwise multiply + sum.
        conv_w = self.conv1d.weight.squeeze(1)  # (d_inner, d_conv)
        x_conv_t = (new_conv_state * conv_w).sum(dim=-1)
        if self.conv1d.bias is not None:
            x_conv_t = x_conv_t + self.conv1d.bias
        x_conv_t = F.silu(x_conv_t)  # (B, d_inner)

        # SSM projections.
        x_proj_out = self.x_proj(x_conv_t)  # (B, dt_rank + 2*d_state)
        dt, B_vec, C_vec = torch.split(
            x_proj_out, [self.dt_rank, self.d_state, self.d_state], dim=-1,
        )
        dt = F.softplus(self.dt_proj(dt))  # (B, d_inner)

        # Discretize for a single timestep.
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        dt_f = dt.float()
        delta_A = torch.exp(dt_f.unsqueeze(-1) * A.unsqueeze(0))  # (B, d_inner, d_state)
        delta_B_u = (
            dt_f.unsqueeze(-1) * B_vec.float().unsqueeze(1) * x_conv_t.float().unsqueeze(-1)
        )  # (B, d_inner, d_state)

        new_ssm_state = delta_A * ssm_state + delta_B_u  # fp32
        y_t = (new_ssm_state * C_vec.float().unsqueeze(1)).sum(dim=-1)  # (B, d_inner)
        y_t = y_t + self.D.float() * x_conv_t.float()
        y_t = y_t * F.silu(z.float())
        y_t = y_t.to(dtype=x.dtype)

        out_t = self.out_proj(y_t)  # (B, d_model)
        return out_t, new_conv_state, new_ssm_state


class RMSNorm(nn.Module):
    """Root-mean-square layernorm (Mamba's norm choice).

    ``y = x / rms(x) * weight`` where ``rms(x) = sqrt(mean(x^2) + eps)``.
    fp32 reduction regardless of input dtype — matches HF's implementation.
    """

    def __init__(self, d: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: Tensor) -> Tensor:
        dtype_in = x.dtype
        x_f = x.float()
        rms = x_f.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (x_f * rms).to(dtype_in) * self.weight


class MambaResidualBlock(nn.Module):
    """Pre-norm + mixer + residual, matching HF's ``MambaBlock`` wrapper."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: int | str = "auto",
        bias: bool = False,
        conv_bias: bool = True,
        norm_eps: float = 1e-5,
        scan_impl: str = "parallel",
    ) -> None:
        super().__init__()
        self.norm = RMSNorm(d_model, eps=norm_eps)
        self.mixer = MambaBlock(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dt_rank=dt_rank,
            bias=bias,
            conv_bias=conv_bias,
        )
        self.scan_impl = scan_impl

    def forward_with_state(
        self,
        hidden_states: Tensor,
        conv_state: Tensor | None = None,
        ssm_state: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        residual = hidden_states
        normed = self.norm(hidden_states)
        out, conv_state, ssm_state = self.mixer.forward_with_state(
            normed, conv_state=conv_state, ssm_state=ssm_state, scan_impl=self.scan_impl,
        )
        return residual + out, conv_state, ssm_state

    def step(
        self,
        hidden_state_t: Tensor,
        conv_state: Tensor,
        ssm_state: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        residual = hidden_state_t.squeeze(1) if hidden_state_t.ndim == 3 else hidden_state_t
        normed = self.norm(residual)
        out_t, conv_state, ssm_state = self.mixer.step(normed, conv_state, ssm_state)
        return residual + out_t, conv_state, ssm_state


@dataclass(slots=True)
class MambaModelConfig:
    vocab_size: int
    d_model: int
    n_layer: int
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    dt_rank: int | str = "auto"
    bias: bool = False
    conv_bias: bool = True
    norm_eps: float = 1e-5
    pad_vocab_size_multiple: int = 1
    tie_word_embeddings: bool = True

    @classmethod
    def from_hf_config(cls, config: Any) -> "MambaModelConfig":
        vocab_size = int(getattr(config, "vocab_size"))
        pad_mult = int(getattr(config, "pad_vocab_size_multiple", 1))
        if pad_mult > 1 and (vocab_size % pad_mult) != 0:
            vocab_size = ((vocab_size // pad_mult) + 1) * pad_mult
        return cls(
            vocab_size=vocab_size,
            d_model=int(getattr(config, "hidden_size")),
            n_layer=int(getattr(config, "num_hidden_layers")),
            d_state=int(getattr(config, "state_size", 16)),
            d_conv=int(getattr(config, "conv_kernel", 4)),
            expand=int(getattr(config, "expand", 2)),
            dt_rank=getattr(config, "time_step_rank", "auto"),
            bias=bool(getattr(config, "use_bias", False)),
            conv_bias=bool(getattr(config, "use_conv_bias", True)),
            norm_eps=float(getattr(config, "layer_norm_epsilon", 1e-5)),
            pad_vocab_size_multiple=pad_mult,
            tie_word_embeddings=bool(getattr(config, "tie_word_embeddings", True)),
        )


class MambaModel(nn.Module):
    """Full Mamba LM: embedding → N × (RMSNorm + MambaBlock + residual) → RMSNorm → LM head.

    Matches the HuggingFace ``state-spaces/mamba-130m-hf`` architecture, but
    runs selective scan through our own implementation (``scan_impl`` picks
    between ``"parallel"`` and ``"naive"``).
    """

    def __init__(self, config: MambaModelConfig, scan_impl: str = "parallel") -> None:
        super().__init__()
        self.config = config
        self.scan_impl = scan_impl
        self.embeddings = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList(
            [
                MambaResidualBlock(
                    d_model=config.d_model,
                    d_state=config.d_state,
                    d_conv=config.d_conv,
                    expand=config.expand,
                    dt_rank=config.dt_rank,
                    bias=config.bias,
                    conv_bias=config.conv_bias,
                    norm_eps=config.norm_eps,
                    scan_impl=scan_impl,
                )
                for _ in range(config.n_layer)
            ]
        )
        self.norm_f = RMSNorm(config.d_model, eps=config.norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embeddings.weight

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def allocate_inference_state(
        self, batch_size: int, device: torch.device | str | None = None
    ) -> list[tuple[Tensor, Tensor]]:
        """Return a list of per-layer ``(conv_state, ssm_state)`` tuples."""

        if device is None:
            device = self.device
        return [layer.mixer.allocate_inference_state(batch_size, device=device) for layer in self.layers]

    def forward(
        self,
        input_ids: Tensor,
        state: list[tuple[Tensor, Tensor]] | None = None,
        return_state: bool = False,
    ) -> Tensor | tuple[Tensor, list[tuple[Tensor, Tensor]]]:
        """Run the full model on a token sequence.

        If ``state`` is ``None`` we start from zeros. After prefill, if
        ``return_state=True`` we return the per-layer ``(conv_state, ssm_state)``
        list so the caller can continue with ``step()``.
        """

        h = self.embeddings(input_ids)
        new_state: list[tuple[Tensor, Tensor]] = []
        for i, layer in enumerate(self.layers):
            conv_s, ssm_s = (None, None) if state is None else state[i]
            h, conv_s, ssm_s = layer.forward_with_state(h, conv_s, ssm_s)
            new_state.append((conv_s, ssm_s))
        h = self.norm_f(h)
        logits = self.lm_head(h)
        if return_state:
            return logits, new_state
        return logits

    def step(
        self, token_ids: Tensor, state: list[tuple[Tensor, Tensor]]
    ) -> tuple[Tensor, list[tuple[Tensor, Tensor]]]:
        """Advance one token per batch element. ``token_ids`` shape ``(B,)``.

        Returns ``(logits, new_state)`` where ``logits`` shape is ``(B, vocab)``.
        """

        h = self.embeddings(token_ids)  # (B, d_model)
        new_state: list[tuple[Tensor, Tensor]] = []
        for i, layer in enumerate(self.layers):
            conv_s, ssm_s = state[i]
            h, conv_s, ssm_s = layer.step(h, conv_s, ssm_s)
            new_state.append((conv_s, ssm_s))
        h = self.norm_f(h)
        logits = self.lm_head(h)
        return logits, new_state

    # ------------------------------------------------------------------
    # Checkpoint loading
    # ------------------------------------------------------------------

    def load_hf_state_dict(self, state_dict: dict[str, Tensor]) -> None:
        """Load a ``state-spaces/mamba-*-hf`` state dict into this model.

        HF keys have the shape:
          * ``backbone.embeddings.weight``
          * ``backbone.layers.{i}.norm.weight``
          * ``backbone.layers.{i}.mixer.<mixer_param>``
          * ``backbone.norm_f.weight``
          * ``lm_head.weight`` (usually tied to embeddings)
        """

        from .weights import MIXER_PARAM_NAMES

        own = {k: p for k, p in self.named_parameters()}
        missing: list[str] = []

        def _copy(dst_key: str, src_tensor: Tensor) -> None:
            if dst_key not in own:
                raise KeyError(f"target param missing: {dst_key}")
            with torch.no_grad():
                own[dst_key].copy_(src_tensor.to(own[dst_key].dtype))

        # Embedding.
        emb_key = "backbone.embeddings.weight"
        if emb_key not in state_dict:
            missing.append(emb_key)
        else:
            _copy("embeddings.weight", state_dict[emb_key])

        # Per-layer: norm + mixer.
        for i, layer in enumerate(self.layers):
            norm_key = f"backbone.layers.{i}.norm.weight"
            if norm_key not in state_dict:
                missing.append(norm_key)
            else:
                _copy(f"layers.{i}.norm.weight", state_dict[norm_key])
            for name in MIXER_PARAM_NAMES:
                src = f"backbone.layers.{i}.mixer.{name}"
                if src not in state_dict:
                    missing.append(src)
                    continue
                _copy(f"layers.{i}.mixer.{name}", state_dict[src])

        # Final norm.
        final_key = "backbone.norm_f.weight"
        if final_key not in state_dict:
            missing.append(final_key)
        else:
            _copy("norm_f.weight", state_dict[final_key])

        # LM head — if tied, it's already bound to embeddings; otherwise copy.
        if not self.config.tie_word_embeddings:
            if "lm_head.weight" not in state_dict:
                missing.append("lm_head.weight")
            else:
                _copy("lm_head.weight", state_dict["lm_head.weight"])

        if missing:
            raise KeyError(f"missing keys while loading Mamba state dict: {missing}")

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = "state-spaces/mamba-130m-hf",
        scan_impl: str = "parallel",
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> "MambaModel":
        """Load an HF Mamba checkpoint into this module tree."""

        from .weights import load_mamba_hf_checkpoint

        ckpt = load_mamba_hf_checkpoint(model_name)
        config = MambaModelConfig.from_hf_config(ckpt.config)
        model = cls(config, scan_impl=scan_impl)
        model.load_hf_state_dict(ckpt.state_dict)
        model = model.to(device=device, dtype=dtype)
        model.eval()
        return model
