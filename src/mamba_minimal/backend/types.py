from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


ScanBackend = Literal["auto", "reference", "fused"]


@dataclass(frozen=True, slots=True)
class ScanSupportInfo:
    supported: bool
    layout: str
    reason: str


@dataclass(frozen=True, slots=True)
class ScanDispatchMetadata:
    requested_backend: ScanBackend
    selected_backend: str
    used_fallback: bool
    layout: str
    notes: str

    @property
    def backend(self) -> str:
        """Backward-compatible alias for older metadata consumers."""

        return self.selected_backend
