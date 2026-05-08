from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tpgr.data.labels import (
    get_label_space_classes,
    get_task_classes,
    gesture_to_command,
    gesture_to_direction,
    normalize_raw_label,
)
from tpgr.data.manifests import filter_by_split, read_jsonl, resolve_path


def estimate_transition_statistics(
    video_manifest_path: str | Path,
    dataset_name: str | None,
    label_space: str,
    split: str = "train",
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    class_names = list(get_label_space_classes(label_space))
    label_to_index = {name: idx for idx, name in enumerate(class_names)}
    num_classes = len(class_names)
    start_counts = np.ones((num_classes,), dtype=np.float64)
    transition_counts = np.ones((num_classes, num_classes), dtype=np.float64)
    items = filter_by_split(read_jsonl(video_manifest_path), split=split)

    for item in items:
        feature_path = resolve_path(item["feature_path"], video_manifest_path)
        payload = np.load(feature_path, allow_pickle=True)
        raw_labels = payload["raw_labels"]
        directions = payload["directions"].astype(str)
        labels = [
            normalize_raw_label(
                int(raw_label),
                dataset_name=dataset_name,
                label_space=label_space,
                direction=str(direction),
            )
            for raw_label, direction in zip(raw_labels.tolist(), directions.tolist())
        ]
        indices = [label_to_index[label] for label in labels]
        if not indices:
            continue
        start_counts[indices[0]] += 1.0
        for src, dst in zip(indices[:-1], indices[1:]):
            transition_counts[src, dst] += 1.0
    return start_counts, transition_counts, class_names


def build_online_transition_matrix(
    class_names: list[str],
    start_counts: np.ndarray,
    transition_counts: np.ndarray,
    stay_bias: float = 18.0,
    same_direction_bonus: float = 2.5,
    same_command_bonus: float = 1.5,
    cross_direction_scale: float = 0.15,
    background_bonus: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    counts = np.asarray(transition_counts, dtype=np.float64).copy()
    num_classes = len(class_names)
    background = class_names[0]
    for idx in range(num_classes):
        counts[idx, idx] += float(stay_bias)
    for src_idx, src_name in enumerate(class_names):
        for dst_idx, dst_name in enumerate(class_names):
            if src_idx == dst_idx:
                continue
            if src_name == background or dst_name == background:
                counts[src_idx, dst_idx] += float(background_bonus)
                continue
            src_dir = gesture_to_direction(src_name)
            dst_dir = gesture_to_direction(dst_name)
            src_cmd = gesture_to_command(src_name)
            dst_cmd = gesture_to_command(dst_name)
            if src_dir == dst_dir:
                counts[src_idx, dst_idx] += float(same_direction_bonus)
            else:
                counts[src_idx, dst_idx] *= float(cross_direction_scale)
            if src_cmd == dst_cmd:
                counts[src_idx, dst_idx] += float(same_command_bonus)

    row_sums = counts.sum(axis=1, keepdims=True).clip(min=1e-8)
    transition = counts / row_sums
    start = np.asarray(start_counts, dtype=np.float64)
    start = start / start.sum().clip(min=1e-8)
    return np.log(start.clip(min=1e-8)), np.log(transition.clip(min=1e-8))


def estimate_task_transition_statistics(
    video_manifest_path: str | Path,
    dataset_name: str | None,
    label_space: str,
    task: str,
    split: str = "train",
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    class_names = list(get_task_classes(task, label_space=label_space))
    label_to_index = {name: idx for idx, name in enumerate(class_names)}
    num_classes = len(class_names)
    start_counts = np.ones((num_classes,), dtype=np.float64)
    transition_counts = np.ones((num_classes, num_classes), dtype=np.float64)
    items = filter_by_split(read_jsonl(video_manifest_path), split=split)

    for item in items:
        feature_path = resolve_path(item["feature_path"], video_manifest_path)
        payload = np.load(feature_path, allow_pickle=True)
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
        if task == "command":
            labels = [gesture_to_command(label) for label in gesture_labels]
        elif task == "direction":
            labels = [gesture_to_direction(label) for label in gesture_labels]
        else:
            raise ValueError(f"未知 task: {task}")
        indices = [label_to_index[label] for label in labels]
        if not indices:
            continue
        start_counts[indices[0]] += 1.0
        for src, dst in zip(indices[:-1], indices[1:]):
            transition_counts[src, dst] += 1.0
    return start_counts, transition_counts, class_names


def build_simple_online_transition_matrix(
    start_counts: np.ndarray,
    transition_counts: np.ndarray,
    stay_bias: float = 8.0,
) -> tuple[np.ndarray, np.ndarray]:
    counts = np.asarray(transition_counts, dtype=np.float64).copy()
    num_classes = counts.shape[0]
    for idx in range(num_classes):
        counts[idx, idx] += float(stay_bias)
    row_sums = counts.sum(axis=1, keepdims=True).clip(min=1e-8)
    transition = counts / row_sums
    start = np.asarray(start_counts, dtype=np.float64)
    start = start / start.sum().clip(min=1e-8)
    return np.log(start.clip(min=1e-8)), np.log(transition.clip(min=1e-8))


@dataclass
class OnlineViterbiDecoder:
    class_names: list[str]
    log_start: np.ndarray
    log_transition: np.ndarray
    emission_scale: float = 1.0

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._state: np.ndarray | None = None

    def step(self, probs: np.ndarray) -> tuple[str, np.ndarray]:
        probs = np.asarray(probs, dtype=np.float64)
        if probs.ndim != 1 or probs.shape[0] != len(self.class_names):
            raise ValueError("decoder 输入维度与类别数不匹配")
        log_emission = np.log(np.clip(probs, 1e-8, 1.0)) * float(self.emission_scale)
        if self._state is None:
            self._state = self.log_start + log_emission
        else:
            candidates = self._state[:, None] + self.log_transition
            self._state = log_emission + np.max(candidates, axis=0)
        self._state = self._state - np.max(self._state)
        idx = int(np.argmax(self._state))
        posterior = np.exp(self._state)
        posterior = posterior / posterior.sum().clip(min=1e-8)
        return self.class_names[idx], posterior.astype(np.float32)
