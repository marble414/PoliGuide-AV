from __future__ import annotations

from typing import Iterable


CANONICAL_GESTURES = (
    "NO_GESTURE",
    "STOP",
    "GO_STRAIGHT",
    "TURN_LEFT",
    "LEFT_TURN_WAIT",
    "TURN_RIGHT",
    "CHANGE_LANE",
    "SLOW_DOWN",
    "PULL_OVER",
)

BASE_COMMANDS = CANONICAL_GESTURES[1:]
DIRECTIONS = ("F", "B", "L", "R")
COMMAND_CLASSES = (
    "NO_COMMAND",
    "STOP",
    "GO_STRAIGHT",
    "LEFT_TURN_WAIT",
    "TURN_LEFT",
    "TURN_RIGHT",
    "CHANGE_LANE",
    "SLOW_DOWN",
    "PULL_OVER",
)
CANONICAL_COMMANDS = COMMAND_CLASSES
DIRECTION_CLASSES = ("NO_DIRECTION", *DIRECTIONS)

GESTURE_TO_INDEX = {name: idx for idx, name in enumerate(CANONICAL_GESTURES)}
INDEX_TO_GESTURE = {idx: name for name, idx in GESTURE_TO_INDEX.items()}

_CTPGESTURE_V1 = {
    0: "NO_GESTURE",
    1: "STOP",
    2: "GO_STRAIGHT",
    3: "TURN_LEFT",
    4: "LEFT_TURN_WAIT",
    5: "TURN_RIGHT",
    6: "CHANGE_LANE",
    7: "SLOW_DOWN",
    8: "PULL_OVER",
}

_STRING_ALIASES = {
    "": "NO_GESTURE",
    "none": "NO_GESTURE",
    "no_gesture": "NO_GESTURE",
    "background": "NO_GESTURE",
    "inactive": "NO_GESTURE",
    "stop": "STOP",
    "halt": "STOP",
    "forward": "GO_STRAIGHT",
    "straight": "GO_STRAIGHT",
    "go": "GO_STRAIGHT",
    "go_straight": "GO_STRAIGHT",
    "left": "TURN_LEFT",
    "turn_left": "TURN_LEFT",
    "left_turn": "TURN_LEFT",
    "left_turn_wait": "LEFT_TURN_WAIT",
    "wait_left": "LEFT_TURN_WAIT",
    "right": "TURN_RIGHT",
    "turn_right": "TURN_RIGHT",
    "right_turn": "TURN_RIGHT",
    "change_lane": "CHANGE_LANE",
    "lane_change": "CHANGE_LANE",
    "slow": "SLOW_DOWN",
    "slow_down": "SLOW_DOWN",
    "pull_over": "PULL_OVER",
}

_DIRECTION_ALIASES = {
    "f": "F",
    "front": "F",
    "forward": "F",
    "ahead": "F",
    "0": "F",
    "b": "B",
    "back": "B",
    "backward": "B",
    "behind": "B",
    "1": "B",
    "l": "L",
    "left": "L",
    "2": "L",
    "r": "R",
    "right": "R",
    "3": "R",
}


def _canonicalize_text(value: str) -> str:
    key = str(value).strip().replace("-", "_").replace(" ", "_").lower()
    if key in _STRING_ALIASES:
        return _STRING_ALIASES[key]
    upper = key.upper()
    if upper in CANONICAL_GESTURES:
        return upper
    parts = upper.split("_", 1)
    if len(parts) == 2 and parts[0] in DIRECTIONS and parts[1] in CANONICAL_GESTURES:
        return parts[1]
    raise ValueError(f"unknown gesture label: {value!r}")


def direction_class_name(direction: str | int | None) -> str:
    if direction is None:
        return "NO_DIRECTION"
    key = str(direction).strip().replace("-", "_").replace(" ", "_").lower()
    return _DIRECTION_ALIASES.get(key, "NO_DIRECTION")


def directional_gesture_name(direction: str | int | None, gesture: str) -> str:
    base = _canonicalize_text(gesture)
    if base == "NO_GESTURE":
        return "NO_GESTURE"
    direction_name = direction_class_name(direction)
    if direction_name == "NO_DIRECTION":
        direction_name = "F"
    return f"{direction_name}_{base}"


def _ctpv2_raw_to_directional(raw_label: int) -> str:
    if raw_label == 0:
        return "NO_GESTURE"
    if not 1 <= raw_label <= 32:
        raise ValueError(f"CTPGesture v2 label out of range: {raw_label}")
    offset = raw_label - 1
    direction = DIRECTIONS[offset // len(BASE_COMMANDS)]
    gesture = BASE_COMMANDS[offset % len(BASE_COMMANDS)]
    return directional_gesture_name(direction, gesture)


def normalize_raw_label(
    raw_label: str | int,
    dataset_name: str | None = None,
    label_space: str = "canonical",
    direction: str | int | None = None,
) -> str:
    space = _normalize_label_space(label_space)
    dataset = (dataset_name or "").lower()
    if isinstance(raw_label, str):
        text = raw_label.strip()
        try:
            raw_int = int(text)
        except ValueError:
            if "_" in text and text.split("_", 1)[0].upper() in DIRECTIONS:
                directional = text.upper()
                base = gesture_to_command(directional)
                return directional if space == "ctpv2_directional" else base
            base = _canonicalize_text(text)
            return directional_gesture_name(direction, base) if space == "ctpv2_directional" else base
        raw_label = raw_int

    raw_int = int(raw_label)
    if "v2" in dataset:
        directional = _ctpv2_raw_to_directional(raw_int)
        return directional if space == "ctpv2_directional" else gesture_to_command(directional)
    mapped = _CTPGESTURE_V1.get(raw_int)
    if mapped is None:
        raise ValueError(f"CTPGesture v1 label out of range: {raw_int}")
    return directional_gesture_name(direction, mapped) if space == "ctpv2_directional" else mapped


def _normalize_label_space(label_space: str | None) -> str:
    key = str(label_space or "canonical").strip().lower()
    if key in {"canonical", "base", "gesture", "9class", "9_class"}:
        return "canonical"
    if key in {"ctpv2_directional", "ctpgesture_v2_directional", "directional", "33class", "33_class"}:
        return "ctpv2_directional"
    raise ValueError(f"unknown label space: {label_space!r}")


def get_label_space_classes(label_space: str = "canonical") -> tuple[str, ...]:
    if _normalize_label_space(label_space) == "canonical":
        return CANONICAL_GESTURES
    return ("NO_GESTURE", *(f"{direction}_{gesture}" for direction in DIRECTIONS for gesture in BASE_COMMANDS))


def get_label_to_index(label_space: str = "canonical") -> dict[str, int]:
    return {name: idx for idx, name in enumerate(get_label_space_classes(label_space))}


def get_index_to_label(label_space: str = "canonical") -> dict[int, str]:
    return {idx: name for name, idx in get_label_to_index(label_space).items()}


def get_task_classes(task: str, label_space: str = "canonical") -> tuple[str, ...]:
    task_name = str(task).lower()
    if task_name in {"gesture", "label"}:
        return get_label_space_classes(label_space)
    if task_name == "command":
        return COMMAND_CLASSES
    if task_name == "direction":
        return DIRECTION_CLASSES
    raise ValueError(f"unknown task: {task!r}")


def get_task_label_to_index(task: str, label_space: str = "canonical") -> dict[str, int]:
    return {name: idx for idx, name in enumerate(get_task_classes(task, label_space=label_space))}


def get_task_index_to_label(task: str, label_space: str = "canonical") -> dict[int, str]:
    return {idx: name for name, idx in get_task_label_to_index(task, label_space=label_space).items()}


def gesture_to_command(gesture: str) -> str:
    text = str(gesture).strip().upper()
    if text == "NO_GESTURE":
        return "NO_COMMAND"
    parts = text.split("_", 1)
    if len(parts) == 2 and parts[0] in DIRECTIONS:
        text = parts[1]
    if text in BASE_COMMANDS:
        return text
    if text in COMMAND_CLASSES:
        return text
    return _canonicalize_text(text)


def gesture_to_direction(gesture: str) -> str:
    text = str(gesture).strip().upper()
    parts = text.split("_", 1)
    if len(parts) == 2 and parts[0] in DIRECTIONS and parts[1] in BASE_COMMANDS:
        return parts[0]
    return "NO_DIRECTION"


def gesture_to_intent(gesture: str) -> str:
    command = gesture_to_command(gesture)
    return {
        "NO_COMMAND": "UNKNOWN",
        "STOP": "HALT",
        "GO_STRAIGHT": "PROCEED",
        "TURN_LEFT": "TURN_LEFT",
        "LEFT_TURN_WAIT": "WAIT_LEFT_TURN",
        "TURN_RIGHT": "TURN_RIGHT",
        "CHANGE_LANE": "CHANGE_LANE",
        "SLOW_DOWN": "DECELERATE",
        "PULL_OVER": "PULL_OVER",
        "KEEP_WAIT": "KEEP_WAIT",
    }.get(command, "UNKNOWN")


def safe_hold_command(command: str) -> str:
    command_name = str(command).strip().upper()
    if command_name in {"STOP", "LEFT_TURN_WAIT", "KEEP_WAIT"}:
        return command_name
    if command_name == "SLOW_DOWN":
        return "SLOW_DOWN"
    if command_name in {"GO_STRAIGHT", "TURN_LEFT", "TURN_RIGHT", "CHANGE_LANE", "PULL_OVER"}:
        return "KEEP_WAIT"
    return "NO_COMMAND"


def labels_to_indices(labels: Iterable[str], label_space: str = "canonical") -> list[int]:
    mapping = get_label_to_index(label_space)
    return [mapping[label] for label in labels]
