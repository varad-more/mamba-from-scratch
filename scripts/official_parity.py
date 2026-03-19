from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any

import torch

from mamba_minimal.model import MambaBlock


@dataclass(slots=True)
class ParityStats:
    max_abs_error: float
    mean_abs_error: float


def compare_tensors(a: torch.Tensor, b: torch.Tensor) -> ParityStats:
    diff = (a - b).abs()
    return ParityStats(max_abs_error=float(diff.max().item()), mean_abs_error=float(diff.mean().item()))


def try_build_block_from_official(model: Any, layer_idx: int) -> MambaBlock:
    # This script intentionally uses a permissive strategy because HF model internals
    # can vary across versions. We discover the mixer module dynamically.
    backbone = getattr(model, "backbone", None)
    if backbone is None:
        raise RuntimeError("Official model does not expose `.backbone`; inspect architecture manually.")

    layers = getattr(backbone, "layers", None)
    if layers is None:
        raise RuntimeError("Official model does not expose `.backbone.layers`.")

    layer = layers[layer_idx]
    mixer = getattr(layer, "mixer", None)
    if mixer is None:
        raise RuntimeError("Layer does not expose `.mixer`.")

    block = MambaBlock(
        d_model=int(mixer.d_model),
        d_state=int(mixer.d_state),
        d_conv=int(mixer.d_conv),
        expand=int(mixer.expand),
        dt_rank=int(mixer.dt_rank),
        bias=bool(getattr(mixer, "bias", False)),
        conv_bias=bool(getattr(mixer, "conv_bias", True)),
    )

    # Attempt direct state load first.
    incompatible = block.load_state_dict(mixer.state_dict(), strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "State mapping mismatch between official mixer and local MambaBlock. "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )
    return block


def main() -> None:
    parser = argparse.ArgumentParser(description="Best-effort parity check against HF Mamba model")
    parser.add_argument("--model", default="state-spaces/mamba-130m-hf")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    try:
        from transformers import AutoModelForCausalLM
    except Exception as exc:
        raise RuntimeError("transformers is required for parity script") from exc

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModelForCausalLM.from_pretrained(args.model).to(device)
    model.eval()

    block = try_build_block_from_official(model, args.layer).to(device)
    block.eval()

    hidden_size = block.d_model
    x = torch.randn(args.batch, args.seq_len, hidden_size, device=device)

    with torch.no_grad():
        official = model.backbone.layers[args.layer].mixer(x)
        local = block(x)

    stats = compare_tensors(official, local)
    print(
        {
            "model": args.model,
            "layer": args.layer,
            "device": device,
            "max_abs_error": stats.max_abs_error,
            "mean_abs_error": stats.mean_abs_error,
        }
    )


if __name__ == "__main__":
    main()
