from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class KernelConfig:
    block_l: int
    num_warps: int
    num_stages: int


DEFAULT_CONFIGS = [
    KernelConfig(block_l=64, num_warps=4, num_stages=2),
    KernelConfig(block_l=128, num_warps=4, num_stages=2),
    KernelConfig(block_l=128, num_warps=8, num_stages=3),
    KernelConfig(block_l=256, num_warps=8, num_stages=3),
]


def describe_configs() -> list[str]:
    return [
        f"block_l={cfg.block_l}, num_warps={cfg.num_warps}, num_stages={cfg.num_stages}"
        for cfg in DEFAULT_CONFIGS
    ]
