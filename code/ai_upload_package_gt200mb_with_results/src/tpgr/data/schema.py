from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class CommandOutput:
    frame_index: int
    timestamp_sec: float
    active_track_id: int | None
    gesture: str
    gesture_confidence: float
    intent: str
    command: str
    command_confidence: float
    safe_fallback: bool
    latency_ms: float
    fps: float
    notes: list[str] = field(default_factory=list)
    debug: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

