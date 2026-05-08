from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tpgr.data.labels import (
    direction_class_name,
    gesture_to_command,
    gesture_to_direction,
    get_label_space_classes,
    get_label_to_index,
    get_task_classes,
    get_task_label_to_index,
    normalize_raw_label,
)
from tpgr.data.manifests import filter_by_split, read_jsonl, resolve_path


@dataclass
class VideoRecord:
    sample_id: str
    feature_path: str
    video_path: str
    features: np.ndarray
    gesture_targets: np.ndarray
    command_targets: np.ndarray
    direction_targets: np.ndarray
    gesture_labels: list[str]
    command_labels: list[str]
    direction_labels: list[str]


def load_video_records(
    manifest_path: str | Path,
    split: str,
    dataset_name: str,
    label_space: str = "ctpv2_directional",
    direction_target_source: str = "gesture",
) -> list[VideoRecord]:
    manifest_path = Path(manifest_path)
    items = filter_by_split(read_jsonl(manifest_path), split=split)
    gesture_to_index = get_label_to_index(label_space)
    command_to_index = get_task_label_to_index("command", label_space=label_space)
    direction_to_index = get_task_label_to_index("direction", label_space=label_space)
    records: list[VideoRecord] = []

    for item in items:
        feature_path = resolve_path(item["feature_path"], manifest_path)
        payload = np.load(feature_path, allow_pickle=True)
        features = payload["features"].astype(np.float32)
        raw_labels = payload["raw_labels"]
        directions = payload["directions"].astype(str)
        gesture_labels = [
            normalize_raw_label(
                int(raw_label),
                dataset_name=dataset_name,
                label_space=label_space,
                direction=str(direction),
            )
            for raw_label, direction in zip(raw_labels.tolist(), directions.tolist())
        ]
        command_labels = [gesture_to_command(label) for label in gesture_labels]
        if str(direction_target_source).lower() in {"payload", "raw", "orientation"}:
            direction_labels = [direction_class_name(str(direction)) for direction in directions.tolist()]
        else:
            direction_labels = [gesture_to_direction(label) for label in gesture_labels]
        records.append(
            VideoRecord(
                sample_id=str(item["sample_id"]),
                feature_path=str(feature_path),
                video_path=str(payload["video_path"]) if "video_path" in payload else str(item.get("video_path", "")),
                features=features,
                gesture_targets=np.asarray([gesture_to_index[label] for label in gesture_labels], dtype=np.int64),
                command_targets=np.asarray([command_to_index[label] for label in command_labels], dtype=np.int64),
                direction_targets=np.asarray([direction_to_index[label] for label in direction_labels], dtype=np.int64),
                gesture_labels=gesture_labels,
                command_labels=command_labels,
                direction_labels=direction_labels,
            )
        )
    return records


def build_frame_class_weights(records: list[VideoRecord], task: str, label_space: str, mode: str = "inverse_sqrt") -> np.ndarray | None:
    mode = str(mode or "none").lower()
    if mode in {"none", "off", "false"}:
        return None
    if task == "gesture":
        num_classes = len(get_label_space_classes(label_space))
    else:
        num_classes = len(get_task_classes(task, label_space=label_space))
    counts = np.zeros((num_classes,), dtype=np.float64)
    for record in records:
        targets = getattr(record, f"{task}_targets")
        bincount = np.bincount(targets, minlength=num_classes)
        counts += bincount.astype(np.float64)
    counts = np.clip(counts, 1.0, None)
    if mode in {"inverse", "inv"}:
        weights = 1.0 / counts
    elif mode in {"inverse_sqrt", "inv_sqrt"}:
        weights = 1.0 / np.sqrt(counts)
    else:
        raise ValueError(f"未知 class_weighting: {mode}")
    weights = weights / np.clip(weights.mean(), 1e-8, None)
    return weights.astype(np.float32)


def moving_average_probs(probs: np.ndarray, window: int) -> np.ndarray:
    window = max(1, int(window))
    if window <= 1:
        return probs
    pad = window - 1
    padded = np.pad(probs, ((pad, 0), (0, 0)), mode="edge")
    cumsum = np.concatenate(
        [np.zeros((1, padded.shape[1]), dtype=np.float64), np.cumsum(padded, axis=0, dtype=np.float64)],
        axis=0,
    )
    smoothed = cumsum[window:] - cumsum[:-window]
    return (smoothed / float(window)).astype(np.float32)


def fuse_multitask_probs(
    gesture_probs: np.ndarray,
    label_space: str,
    command_probs: np.ndarray | None,
    direction_probs: np.ndarray | None,
    command_weight: float,
    direction_weight: float,
    background_agnostic_direction: bool = False,
) -> np.ndarray:
    gesture_classes = list(get_label_space_classes(label_space))
    command_to_index = get_task_label_to_index("command", label_space=label_space)
    direction_to_index = get_task_label_to_index("direction", label_space=label_space)
    fused = np.log(np.clip(gesture_probs, 1e-8, 1.0))
    if command_probs is not None and command_weight > 0.0:
        fused += float(command_weight) * np.asarray(
            [
                np.log(np.clip(command_probs[:, command_to_index[gesture_to_command(label)]], 1e-8, 1.0))
                for label in gesture_classes
            ],
            dtype=np.float32,
        ).T
    if direction_probs is not None and direction_weight > 0.0:
        fused += float(direction_weight) * np.asarray(
            [
                (
                    np.zeros((direction_probs.shape[0],), dtype=np.float32)
                    if background_agnostic_direction and label == "NO_GESTURE"
                    else np.log(np.clip(direction_probs[:, direction_to_index[gesture_to_direction(label)]], 1e-8, 1.0))
                )
                for label in gesture_classes
            ],
            dtype=np.float32,
        ).T
    fused = fused - np.max(fused, axis=-1, keepdims=True)
    probs = np.exp(fused)
    return probs / np.clip(np.sum(probs, axis=-1, keepdims=True), 1e-8, None)


def apply_gesture_bias(
    gesture_probs: np.ndarray,
    label_space: str,
    bias_map: dict[str, float] | None = None,
) -> np.ndarray:
    if not bias_map:
        return gesture_probs
    class_names = list(get_label_space_classes(label_space))
    bias = np.asarray([float(bias_map.get(name, 0.0)) for name in class_names], dtype=np.float32)
    if not np.any(np.abs(bias) > 1e-8):
        return gesture_probs
    fused = np.log(np.clip(gesture_probs, 1e-8, 1.0)) + bias[None, :]
    fused = fused - np.max(fused, axis=-1, keepdims=True)
    probs = np.exp(fused)
    return probs / np.clip(np.sum(probs, axis=-1, keepdims=True), 1e-8, None)
