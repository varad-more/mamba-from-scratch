from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from mamba_minimal.model import MambaBlock


@dataclass(slots=True)
class ParityStats:
    model: str
    layer: int
    device: str
    max_abs_error: float
    mean_abs_error: float
    local_shape: tuple[int, ...]
    official_shape: tuple[int, ...]


def compare_tensors(a: torch.Tensor, b: torch.Tensor, model_name: str, layer_idx: int, device: str) -> ParityStats:
    diff = (a - b).abs()
    return ParityStats(
        model=model_name,
        layer=layer_idx,
        device=device,
        max_abs_error=float(diff.max().item()),
        mean_abs_error=float(diff.mean().item()),
        local_shape=tuple(b.shape),
        official_shape=tuple(a.shape),
    )


def extract_layers(model: Any):
    backbone = getattr(model, "backbone", None)
    if backbone is None:
        raise RuntimeError("Official model does not expose `.backbone`; inspect architecture manually.")

    layers = getattr(backbone, "layers", None)
    if layers is None:
        raise RuntimeError("Official model does not expose `.backbone.layers`.")
    return layers


def extract_official_mixer(model: Any, layer_idx: int) -> Any:
    layers = extract_layers(model)
    if not (0 <= layer_idx < len(layers)):
        raise IndexError(f"layer_idx={layer_idx} out of range for {len(layers)} layers")

    layer = layers[layer_idx]
    mixer = getattr(layer, "mixer", None)
    if mixer is None:
        raise RuntimeError("Layer does not expose `.mixer`.")
    return mixer


def try_build_block_from_official(model: Any, layer_idx: int) -> MambaBlock:
    mixer = extract_official_mixer(model, layer_idx)

    hidden_size = int(getattr(mixer, "hidden_size"))
    d_state = int(getattr(mixer, "ssm_state_size"))
    d_conv = int(getattr(mixer, "conv_kernel_size"))
    d_inner = int(getattr(mixer, "intermediate_size"))
    if d_inner % hidden_size != 0:
        raise RuntimeError(
            f"Official intermediate_size={d_inner} is not divisible by hidden_size={hidden_size}."
        )
    expand = d_inner // hidden_size
    dt_rank = int(getattr(mixer, "time_step_rank"))
    use_bias = bool(getattr(mixer, "use_bias", False))
    use_conv_bias = bool(getattr(mixer, "use_conv_bias", True))

    block = MambaBlock(
        d_model=hidden_size,
        d_state=d_state,
        d_conv=d_conv,
        expand=expand,
        dt_rank=dt_rank,
        bias=use_bias,
        conv_bias=use_conv_bias,
    )

    incompatible = block.load_state_dict(mixer.state_dict(), strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "State mapping mismatch between official mixer and local MambaBlock. "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )
    return block


def parse_layers(layer_arg: str | None, all_layers: bool, total_layers: int) -> list[int]:
    if all_layers:
        return list(range(total_layers))
    if layer_arg is None:
        return [0]
    values = []
    for chunk in layer_arg.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        values.append(int(chunk))
    if not values:
        raise ValueError("No valid layer indices were provided.")
    return values


def run_parity_for_layer(model: Any, model_name: str, layer_idx: int, batch: int, seq_len: int, device: str) -> ParityStats:
    block = try_build_block_from_official(model, layer_idx).to(device)
    block.eval()

    hidden_size = block.d_model
    x = torch.randn(batch, seq_len, hidden_size, device=device)
    mixer = extract_official_mixer(model, layer_idx)
    with torch.no_grad():
        official = mixer(x)
        local = block(x)
    return compare_tensors(official, local, model_name, layer_idx, device)


def main() -> None:
    parser = argparse.ArgumentParser(description="Best-effort parity check against HF Mamba model")
    parser.add_argument("--model", default="state-spaces/mamba-130m-hf")
    parser.add_argument("--layer", type=str, default=None, help="Single layer index or comma-separated list")
    parser.add_argument("--all-layers", action="store_true", help="Run parity for every mixer layer")
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of repr dict")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path")
    args = parser.parse_args()

    try:
        from transformers import AutoModelForCausalLM
    except Exception as exc:
        raise RuntimeError("transformers is required for parity script") from exc

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(args.seed)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device)
    model.eval()

    total_layers = len(extract_layers(model))
    layer_indices = parse_layers(args.layer, args.all_layers, total_layers)
    results = [
        asdict(run_parity_for_layer(model, args.model, layer_idx, args.batch, args.seq_len, device))
        for layer_idx in layer_indices
    ]

    payload: object
    if len(results) == 1:
        payload = results[0]
    else:
        payload = {
            "model": args.model,
            "device": device,
            "layers": layer_indices,
            "max_abs_error": max(row["max_abs_error"] for row in results),
            "mean_abs_error": sum(row["mean_abs_error"] for row in results) / len(results),
            "results": results,
        }

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2))

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(payload)


if __name__ == "__main__":
    main()
