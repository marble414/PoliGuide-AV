from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from tpgr.data.labels import (
    CANONICAL_COMMANDS,
    CANONICAL_GESTURES,
    get_label_space_classes,
    gesture_to_command,
    normalize_raw_label,
)
from tpgr.data.manifests import resolve_path
from tpgr.metrics.classification import classification_report_dict
from tpgr.metrics.sequence import (
    classwise_jaccard,
    command_switch_delay,
    macro_jaccard,
    sequence_edit_similarity,
    stable_output_latency,
)


def group_segments_by_video(
    items: Iterable[dict],
    manifest_path: str | Path,
) -> List[Tuple[Path, List[dict]]]:
    grouped: Dict[Path, List[dict]] = defaultdict(list)
    for item in items:
        video_path = resolve_path(item["video_path"], manifest_path)
        grouped[video_path].append(item)
    return sorted(grouped.items(), key=lambda pair: pair[0].name)


def build_dense_targets(
    segments: Sequence[dict],
    frame_count: int,
    dataset_name: str | None,
    label_space: str = "canonical",
) -> tuple[list[str], list[str], list[str]]:
    max_end = max((int(item.get("frame_end", -1)) for item in segments), default=-1)
    total_frames = max(int(frame_count), max_end + 1, 0)

    gestures = ["NO_GESTURE"] * total_frames
    commands = ["NO_COMMAND"] * total_frames
    directions = ["unknown"] * total_frames

    for item in sorted(segments, key=lambda record: (int(record.get("frame_start", 0)), int(record.get("frame_end", 0)))):
        start = max(0, int(item.get("frame_start", 0)))
        end = min(total_frames - 1, int(item.get("frame_end", -1)))
        if end < start:
            continue
        attrs = item.get("attributes", {}) or {}
        label = normalize_raw_label(
            item.get("label", "NO_GESTURE"),
            dataset_name=dataset_name,
            label_space=label_space,
            direction=item.get("direction") or attrs.get("direction"),
        )
        command = gesture_to_command(label)
        direction = str(item.get("direction") or attrs.get("direction") or "unknown")
        for frame_index in range(start, end + 1):
            gestures[frame_index] = label
            commands[frame_index] = command
            directions[frame_index] = direction

    return gestures, commands, directions


def labels_to_indices(labels: Sequence[str], class_names: Sequence[str]) -> List[int]:
    mapping = {name: idx for idx, name in enumerate(class_names)}
    return [mapping[item] for item in labels]


def summarize_sequence_metrics(
    gt: Sequence[str],
    pred: Sequence[str],
    class_names: Sequence[str],
    background_label: str,
) -> dict:
    if not gt:
        return {
            "frames": 0,
            "foreground_frames": 0,
            "accuracy": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
            "macro_jaccard": 0.0,
            "foreground_frame_accuracy": 0.0,
            "foreground_macro_jaccard": 0.0,
            "sequence_edit_similarity": 0.0,
            "switch_delay_frames": 0.0,
            "stable_output_latency_frames": 0.0,
        }

    y_true = labels_to_indices(gt, class_names)
    y_pred = labels_to_indices(pred, class_names)
    summary = classification_report_dict(y_true, y_pred, class_names=list(class_names))
    jaccard_scores = classwise_jaccard(pred, gt, class_names)
    foreground_classes = [name for name in class_names if name != background_label]
    foreground_scores = classwise_jaccard(pred, gt, foreground_classes)
    foreground_mask = [label != background_label for label in gt]
    foreground_total = sum(1 for flag in foreground_mask if flag)
    foreground_correct = sum(
        1 for flag, true_label, pred_label in zip(foreground_mask, gt, pred) if flag and true_label == pred_label
    )

    summary.update({
        "frames": int(len(gt)),
        "foreground_frames": int(foreground_total),
        "macro_jaccard": macro_jaccard(pred, gt, class_names),
        "foreground_macro_jaccard": (
            float(sum(foreground_scores.values()) / len(foreground_scores))
            if foreground_scores else 0.0
        ),
        "foreground_frame_accuracy": float(foreground_correct / foreground_total) if foreground_total else 0.0,
        "sequence_edit_similarity": sequence_edit_similarity(pred, gt),
        "switch_delay_frames": command_switch_delay(gt, pred),
        "stable_output_latency_frames": stable_output_latency(
            gt,
            pred,
            background_label=background_label,
        ),
    })
    for name, value in jaccard_scores.items():
        summary[f"{name}_jaccard"] = float(value)
    return summary


def subset_summary(
    gt: Sequence[str],
    pred: Sequence[str],
    class_names: Sequence[str],
) -> dict:
    if not gt:
        return {"frames": 0, "accuracy": 0.0, "macro_f1": 0.0, "macro_jaccard": 0.0}
    y_true = labels_to_indices(gt, class_names)
    y_pred = labels_to_indices(pred, class_names)
    metrics = classification_report_dict(y_true, y_pred, class_names=list(class_names))
    metrics["frames"] = int(len(gt))
    metrics["macro_jaccard"] = macro_jaccard(pred, gt, class_names)
    return metrics


__all__ = [
    "CANONICAL_COMMANDS",
    "CANONICAL_GESTURES",
    "get_label_space_classes",
    "build_dense_targets",
    "group_segments_by_video",
    "labels_to_indices",
    "subset_summary",
    "summarize_sequence_metrics",
]
