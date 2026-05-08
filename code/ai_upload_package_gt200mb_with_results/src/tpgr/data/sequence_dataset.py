from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from tpgr.data.features import adapt_sequence_length, sequence_to_graph_input, sequence_to_tcn_features
from tpgr.data.labels import (
    get_label_to_index,
    get_task_label_to_index,
    gesture_to_command,
    gesture_to_direction,
    normalize_raw_label,
)
from tpgr.data.manifests import filter_by_split, read_jsonl, resolve_path


class SequenceDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        split: str = "train",
        clip_len: int = 48,
        representation: str = "features",
        training: bool = False,
        dataset_name: str | None = None,
        feature_variant: str = "base",
        augment: dict | None = None,
        temporal_mode: str = "crop_pad",
        label_space: str = "canonical",
        tasks: Iterable[str] | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.samples = filter_by_split(read_jsonl(self.manifest_path), split=split)
        self.clip_len = int(clip_len)
        self.representation = str(representation)
        self.training = bool(training)
        self.dataset_name = dataset_name
        self.feature_variant = str(feature_variant or "base")
        self.augment = dict(augment or {})
        self.temporal_mode = str(temporal_mode or "crop_pad")
        self.label_space = str(label_space or "canonical")
        self.tasks = [str(task).lower() for task in (tasks or ["gesture"])]
        if "gesture" not in self.tasks:
            self.tasks.insert(0, "gesture")
        self.label_to_index = get_label_to_index(self.label_space)
        self.task_label_to_index = {
            task: get_task_label_to_index(task, label_space=self.label_space)
            for task in self.tasks
        }

    def __len__(self) -> int:
        return len(self.samples)

    def _load_npz(self, sample: dict):
        key = "feature_path" if "feature_path" in sample else "keypoints_path"
        path = resolve_path(sample[key], self.manifest_path)
        return np.load(path, allow_pickle=True)

    def _load_array(self, sample: dict) -> np.ndarray:
        payload = self._load_npz(sample)
        if self.representation == "features":
            if "features" in payload:
                arr = payload["features"].astype(np.float32, copy=False)
            else:
                arr = sequence_to_tcn_features(
                    payload["keypoints"].astype(np.float32, copy=False),
                    feature_variant=self.feature_variant,
                    bbox_track=payload["bbox_track"].astype(np.float32, copy=False) if "bbox_track" in payload else None,
                )
            return adapt_sequence_length(arr, self.clip_len, training=self.training, mode=self.temporal_mode)
        if self.representation == "graph":
            arr = sequence_to_graph_input(payload["keypoints"].astype(np.float32, copy=False))
            arr = np.transpose(arr, (1, 2, 0))
            arr = adapt_sequence_length(arr, self.clip_len, training=self.training, mode=self.temporal_mode)
            return np.transpose(arr, (2, 0, 1)).astype(np.float32, copy=False)
        raise ValueError(f"unsupported representation: {self.representation}")

    def _label_name(self, sample: dict) -> str:
        attrs = sample.get("attributes") or {}
        return normalize_raw_label(
            sample["label"],
            dataset_name=self.dataset_name,
            label_space=self.label_space,
            direction=sample.get("direction") or attrs.get("direction"),
        )

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        x = self._load_array(sample)
        gesture_label = self._label_name(sample)
        y = self.label_to_index[gesture_label]
        targets: dict[str, torch.Tensor] = {}
        if "gesture" in self.tasks:
            targets["gesture"] = torch.tensor(y, dtype=torch.long)
        if "command" in self.tasks:
            targets["command"] = torch.tensor(
                self.task_label_to_index["command"][gesture_to_command(gesture_label)],
                dtype=torch.long,
            )
        if "direction" in self.tasks:
            targets["direction"] = torch.tensor(
                self.task_label_to_index["direction"][gesture_to_direction(gesture_label)],
                dtype=torch.long,
            )
        return {
            "x": torch.from_numpy(x).float(),
            "y": torch.tensor(y, dtype=torch.long),
            "targets": targets,
            "sample_id": str(sample.get("sample_id", index)),
            "label": gesture_label,
        }


def collate_batch(batch: list[dict]) -> dict:
    x = torch.stack([item["x"] for item in batch], dim=0)
    y = torch.stack([item["y"] for item in batch], dim=0)
    tasks = sorted({task for item in batch for task in item.get("targets", {})})
    targets = {
        task: torch.stack([item["targets"][task] for item in batch], dim=0)
        for task in tasks
    }
    return {
        "x": x,
        "y": y,
        "targets": targets,
        "sample_id": [item.get("sample_id") for item in batch],
        "label": [item.get("label") for item in batch],
    }

