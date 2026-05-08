from __future__ import annotations

from typing import Dict

from tpgr.data.labels import gesture_to_command, gesture_to_intent


class IntentParser:
    def parse(self, gesture: str, confidence: float) -> Dict[str, str | float]:
        return {
            "gesture": gesture,
            "gesture_confidence": confidence,
            "intent": gesture_to_intent(gesture),
            "command": gesture_to_command(gesture),
            "command_confidence": confidence,
        }
