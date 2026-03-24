from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .discretization import inverse_softplus
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
        self.use_fused_kernel = use_fused_kernel and _FUSED_AVAILABLE

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
        scan_fn = selective_scan_ref
        if self.use_fused_kernel:
            scan_fn = selective_scan_fused
        y = scan_fn(
            u=x.transpose(1, 2),
            delta=dt.transpose(1, 2),
            A=A,
            B=B.transpose(1, 2),
            C=C.transpose(1, 2),
            D=self.D.float(),
            z=z.transpose(1, 2),
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
