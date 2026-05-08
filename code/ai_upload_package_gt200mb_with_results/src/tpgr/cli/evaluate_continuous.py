from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from time import perf_counter
from typing import Dict, List

from tqdm import tqdm

from tpgr.cli.common import configure_runtime, save_json
from tpgr.config import ensure_dir, load_config
from tpgr.data.manifests import filter_by_split, read_jsonl
from tpgr.eval_continuous import (
    CANONICAL_COMMANDS,
    build_dense_targets,
    get_label_space_classes,
    group_segments_by_video,
    subset_summary,
    summarize_sequence_metrics,
)
from tpgr.io.json_io import JsonlWriter
from tpgr.io.video_io import iter_video
from tpgr.pipeline.system import TrafficPoliceGestureSystem


def _frame_count_from_video(path: Path) -> int:
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频文件: {path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return frame_count


def main() -> None:
    parser = argparse.ArgumentParser(description="整视频连续评估，尽量与 CTPGesture v2 连续识别口径对齐")
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", default=None, help="默认读取 config.dataset.manifest_path")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-dir", default="runs/continuous_eval")
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--limit-videos", type=int, default=0)
    parser.add_argument("--save-jsonl-dir", default=None, help="若提供，则按视频保存逐帧 gt/pred jsonl")
    parser.add_argument("--override", nargs="*", default=[], help="形如 runtime.pose_backend.device=cuda 的覆盖项")
    args = parser.parse_args()

    configure_runtime(1)
    cfg = load_config(args.config, *args.override)
    dataset_cfg = cfg.get("dataset", {})
    manifest_path = args.manifest or dataset_cfg.get("manifest_path")
    if manifest_path is None:
        raise ValueError("请通过 --manifest 提供原始 manifest，或在 config.dataset.manifest_path 中配置")
    split_key = f"{args.split}_split"
    split = dataset_cfg.get(split_key, args.split)
    dataset_name = args.dataset_name or dataset_cfg.get("dataset_name")
    label_space = str(dataset_cfg.get("label_space", cfg.get("classifier", {}).get("label_space", "canonical")))
    gesture_classes = list(get_label_space_classes(label_space))

    items = filter_by_split(read_jsonl(manifest_path), split=split)
    videos = group_segments_by_video(items, manifest_path)
    if args.limit_videos > 0:
        videos = videos[: int(args.limit_videos)]
    if not videos:
        raise ValueError(f"在 {manifest_path} 中未找到 split={split!r} 的视频")

    out_dir = ensure_dir(args.output_dir)
    jsonl_dir = ensure_dir(args.save_jsonl_dir) if args.save_jsonl_dir else None
    system = TrafficPoliceGestureSystem(cfg)

    all_gt_gesture: List[str] = []
    all_pred_gesture: List[str] = []
    all_gt_command: List[str] = []
    all_pred_command: List[str] = []
    direction_groups: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: {"gt": [], "pred": []})
    latency_values: List[float] = []
    video_summaries: List[dict] = []

    wall_tic = perf_counter()

    for video_path, segments in tqdm(videos, desc="continuous_eval_videos"):
        frame_count = _frame_count_from_video(video_path)
        gt_gesture, gt_command, gt_direction = build_dense_targets(
            segments,
            frame_count,
            dataset_name=dataset_name,
            label_space=label_space,
        )
        pred_gesture: List[str] = []
        pred_command: List[str] = []

        system.reset_runtime_state()
        writer = JsonlWriter(jsonl_dir / f"{video_path.stem}.jsonl") if jsonl_dir is not None else None
        try:
            for frame_index, frame, fps in tqdm(iter_video(str(video_path)), desc=video_path.stem, leave=False):
                result, detections = system.process_frame(frame, fps=fps)
                frame_gt_gesture = gt_gesture[frame_index] if frame_index < len(gt_gesture) else "NO_GESTURE"
                frame_gt_command = gt_command[frame_index] if frame_index < len(gt_command) else "NO_COMMAND"
                frame_direction = gt_direction[frame_index] if frame_index < len(gt_direction) else "unknown"

                pred_gesture.append(result.gesture)
                pred_command.append(result.command)
                latency_values.append(float(result.latency_ms))

                if frame_gt_gesture != "NO_GESTURE":
                    direction_groups[frame_direction]["gt"].append(frame_gt_gesture)
                    direction_groups[frame_direction]["pred"].append(result.gesture)

                if writer is not None:
                    writer.write({
                        **result.to_dict(),
                        "detections": detections,
                        "gt_gesture": frame_gt_gesture,
                        "gt_command": frame_gt_command,
                        "gt_direction": frame_direction,
                    })
        finally:
            if writer is not None:
                writer.close()

        valid_len = min(len(pred_gesture), len(gt_gesture))
        gt_gesture = gt_gesture[:valid_len]
        gt_command = gt_command[:valid_len]
        pred_gesture = pred_gesture[:valid_len]
        pred_command = pred_command[:valid_len]

        all_gt_gesture.extend(gt_gesture)
        all_pred_gesture.extend(pred_gesture)
        all_gt_command.extend(gt_command)
        all_pred_command.extend(pred_command)

        per_video_gesture = summarize_sequence_metrics(
            gt_gesture,
            pred_gesture,
            class_names=gesture_classes,
            background_label=gesture_classes[0],
        )
        per_video_command = summarize_sequence_metrics(
            gt_command,
            pred_command,
            class_names=CANONICAL_COMMANDS,
            background_label="NO_COMMAND",
        )
        mean_latency_ms = float(sum(latency_values[-valid_len:]) / valid_len) if valid_len else 0.0
        video_summaries.append({
            "video": video_path.name,
            "frames": int(valid_len),
            "mean_latency_ms": mean_latency_ms,
            "system_fps_estimate": float(1000.0 / mean_latency_ms) if mean_latency_ms > 1e-6 else 0.0,
            "gesture": per_video_gesture,
            "command": per_video_command,
        })

    wall_elapsed = max(perf_counter() - wall_tic, 1e-6)

    gesture_summary = summarize_sequence_metrics(
        all_gt_gesture,
        all_pred_gesture,
        class_names=gesture_classes,
        background_label=gesture_classes[0],
    )
    command_summary = summarize_sequence_metrics(
        all_gt_command,
        all_pred_command,
        class_names=CANONICAL_COMMANDS,
        background_label="NO_COMMAND",
    )
    direction_summary = {
        direction: subset_summary(values["gt"], values["pred"], class_names=gesture_classes)
        for direction, values in sorted(direction_groups.items())
        if values["gt"]
    }

    summary = {
        "config": args.config,
        "manifest_path": str(Path(manifest_path)),
        "dataset_name": dataset_name,
        "label_space": label_space,
        "split": split,
        "num_videos": len(videos),
        "total_frames": len(all_gt_gesture),
        "total_foreground_frames": int(sum(1 for item in all_gt_gesture if item != gesture_classes[0])),
        "mean_latency_ms": float(sum(latency_values) / len(latency_values)) if latency_values else 0.0,
        "system_fps_estimate": float(1000.0 / (sum(latency_values) / len(latency_values))) if latency_values else 0.0,
        "end_to_end_wall_fps": float(len(all_gt_gesture) / wall_elapsed),
        "gesture": gesture_summary,
        "command": command_summary,
        "by_direction_gesture": direction_summary,
        "videos": video_summaries,
        "notes": [
            "该评测直接跑整视频连续流，不再按单个 clip 分类。",
            f"当前 gesture label_space={label_space}，前景类别数={max(len(gesture_classes) - 1, 0)}。",
            "direction 分组结果用于观察朝向影响；若使用 33 类方向条件标签，它与原论文的 32 类前景任务更接近。",
        ],
    }
    save_json(summary, out_dir / "metrics.json")
    print(summary)


if __name__ == "__main__":
    main()
