from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class PlannerHint:
    command: str
    target_speed_mps: float
    note: str


def command_to_planner_hint(command: str) -> PlannerHint:
    if command in {"STOP", "KEEP_WAIT", "LEFT_TURN_WAIT"}:
        return PlannerHint(command=command, target_speed_mps=0.0, note="停车/等待")
    if command == "SLOW_DOWN":
        return PlannerHint(command=command, target_speed_mps=3.0, note="降速通过")
    if command in {"GO_STRAIGHT", "TURN_LEFT", "TURN_RIGHT", "CHANGE_LANE", "PULL_OVER"}:
        return PlannerHint(command=command, target_speed_mps=6.0, note="交由高层规划器生成轨迹")
    return PlannerHint(command="NO_COMMAND", target_speed_mps=0.0, note="无有效指挥")


class CarlaCommandBridge:
    def __init__(self, vehicle: Optional[Any] = None) -> None:
        self.vehicle = vehicle

    def apply(self, command: str) -> PlannerHint:
        hint = command_to_planner_hint(command)
        if self.vehicle is None:
            return hint
        try:
            control = self.vehicle.get_control()
            if hint.target_speed_mps <= 0.1:
                control.throttle = 0.0
                control.brake = 1.0
            elif command == "SLOW_DOWN":
                control.throttle = min(control.throttle, 0.2)
                control.brake = max(control.brake, 0.2)
            self.vehicle.apply_control(control)
        except Exception:
            pass
        return hint
