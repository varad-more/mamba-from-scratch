"""Mamba checkpoint loading.

Loads a ``state-spaces/mamba-*-hf`` checkpoint via HuggingFace ``transformers``
and maps per-layer mixer weights into the key layout expected by
:class:`mamba_minimal.model.MambaBlock`. This is the oracle-agnostic path: it
does not depend on ``mamba_ssm``, only on ``transformers``.

Design: keep the surface area small. Two things matter here —
  1. You can rehydrate one mixer layer from a real pretrained checkpoint.
  2. The key mapping is explicit and tested, so a silent rename upstream
     in ``transformers`` becomes a loud failure here rather than at inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

MIXER_PARAM_NAMES = (
    "in_proj.weight",
    "conv1d.weight",
    "conv1d.bias",
    "x_proj.weight",
    "dt_proj.weight",
    "dt_proj.bias",
    "A_log",
    "D",
    "out_proj.weight",
)


@dataclass(slots=True)
class MambaCheckpoint:
    model_name: str
    config: Any
    state_dict: dict[str, Tensor]
    num_layers: int


def load_mamba_hf_checkpoint(model_name: str = "state-spaces/mamba-130m-hf") -> MambaCheckpoint:
    """Load a HuggingFace Mamba checkpoint and return its config + state dict."""

    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    state_dict = {k: v.detach().clone() for k, v in model.state_dict().items()}
    del model

    num_layers = int(getattr(config, "num_hidden_layers"))
    return MambaCheckpoint(
        model_name=model_name,
        config=config,
        state_dict=state_dict,
        num_layers=num_layers,
    )


def extract_mixer_state(state_dict: dict[str, Tensor], layer_idx: int) -> dict[str, Tensor]:
    """Return one layer's mixer params keyed by :class:`MambaBlock` parameter names.

    Raises KeyError if any expected parameter is missing — an upstream rename
    should fail loudly here, not at inference time.
    """

    prefix = f"backbone.layers.{layer_idx}.mixer."
    mapped: dict[str, Tensor] = {}
    for name in MIXER_PARAM_NAMES:
        src_key = prefix + name
        if src_key not in state_dict:
            raise KeyError(f"Missing mixer param {src_key} in checkpoint")
        mapped[name] = state_dict[src_key]
    return mapped


def build_block_from_config(config: Any) -> "torch.nn.Module":
    """Create a :class:`MambaBlock` whose shapes match the HF config."""

    from .model import MambaBlock

    return MambaBlock.from_official_config(config)


def load_layer_into_block(
    checkpoint: MambaCheckpoint,
    layer_idx: int,
) -> "torch.nn.Module":
    """Build a fresh :class:`MambaBlock` and load one HF mixer layer into it."""

    if not (0 <= layer_idx < checkpoint.num_layers):
        raise IndexError(
            f"layer_idx {layer_idx} out of range for checkpoint with "
            f"{checkpoint.num_layers} layers"
        )
    block = build_block_from_config(checkpoint.config)
    mixer_state = extract_mixer_state(checkpoint.state_dict, layer_idx)
    block.load_official_mixer_state_dict(mixer_state)
    return block


def layer_norm_weight(state_dict: dict[str, Tensor], layer_idx: int) -> Tensor:
    """Return the pre-mixer RMSNorm weight for a given layer."""

    key = f"backbone.layers.{layer_idx}.norm.weight"
    if key not in state_dict:
        raise KeyError(f"Missing norm weight {key}")
    return state_dict[key]


def mixer_param_count(state_dict: dict[str, Tensor], layer_idx: int) -> int:
    """Sum the element counts of all mixer params for a given layer (for sanity)."""

    mixer = extract_mixer_state(state_dict, layer_idx)
    return sum(int(t.numel()) for t in mixer.values())
