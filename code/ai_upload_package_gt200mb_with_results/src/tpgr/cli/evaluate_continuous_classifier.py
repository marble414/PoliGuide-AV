from __future__ import annotations

import argparse
from collections import defaultdict, deque
from pathlib import Path
from time import perf_counter

import numpy as np
from tqdm import tqdm

from tpgr.cli.common import configure_runtime, save_json
from tpgr.config import ensure_dir, load_config
from tpgr.data.labels import get_label_space_classes, gesture_to_command, gesture_to_direction, normalize_raw_label
from tpgr.data.manifests import filter_by_split, read_jsonl, resolve_path
from tpgr.decoding import (
    OnlineViterbiDecoder,
    build_online_transition_matrix,
    build_simple_online_transition_matrix,
    estimate_task_transition_statistics,
    estimate_transition_statistics,
)
from tpgr.eval_continuous import CANONICAL_COMMANDS, subset_summary, summarize_sequence_metrics
from tpgr.io.json_io import JsonlWriter
from tpgr.pipeline.classifier import build_classifier


def _window(features: np.ndarray, frame_index: int, window_len: int, mode: str = "causal") -> np.ndarray:
    window_len = max(1, int(window_len))
    mode = str(mode or "causal").lower()
    if mode == "center":
        half = window_len // 2
        start = max(0, int(frame_index) - half)
        end = min(len(features), start + window_len)
        start = max(0, end - window_len)
        return features[start:end]
    start = max(0, int(frame_index) - window_len + 1)
    return features[start:int(frame_index) + 1]


def _scores_to_array(scores: dict[str, float], class_names: list[str]) -> np.ndarray:
    return np.asarray([float(scores.get(name, 0.0)) for name in class_names], dtype=np.float32)


def _fuse_task_probs(
    gesture_probs: np.ndarray,
    class_names: list[str],
    command_probs: np.ndarray | None,
    command_label_to_index: dict[str, int],
    direction_probs: np.ndarray | None,
    direction_label_to_index: dict[str, int],
    command_weight: float,
    direction_weight: float,
) -> np.ndarray:
    fused = np.log(np.clip(gesture_probs, 1e-8, 1.0))
    if command_probs is not None and command_weight > 0.0:
        command_log = np.asarray([
            np.log(np.clip(command_probs[command_label_to_index[gesture_to_command(label)]], 1e-8, 1.0))
            for label in class_names
        ], dtype=np.float32)
        fused += command_weight * command_log
    if direction_probs is not None and direction_weight > 0.0:
        direction_log = np.asarray([
            np.log(np.clip(direction_probs[direction_label_to_index[gesture_to_direction(label)]], 1e-8, 1.0))
            for label in class_names
        ], dtype=np.float32)
        fused += direction_weight * direction_log
    fused = fused - np.max(fused)
    probs = np.exp(fused)
    return probs / np.clip(np.sum(probs), 1e-8, None)


def main() -> None:
    parser = argparse.ArgumentParser(description="直接在 VIBE/特征流上做 CTPGesture v2 连续评测")
    parser.add_argument("--config", required=True)
    parser.add_argument("--video-manifest", default=None, help="默认读取 config.dataset.video_manifest_path")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-dir", default="runs/continuous_eval_classifier")
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--limit-videos", type=int, default=0)
    parser.add_argument("--save-jsonl-dir", default=None)
    parser.add_argument("--smooth-window", type=int, default=None)
    parser.add_argument("--context-len", type=int, default=None)
    parser.add_argument("--window-mode", choices=["causal", "center"], default=None)
    parser.add_argument("--override", nargs="*", default=[], help="形如 classifier.checkpoint=... 的覆盖项")
    args = parser.parse_args()

    configure_runtime(1)
    cfg = load_config(args.config, *args.override)
    dataset_cfg = cfg.get("dataset", {})
    video_manifest_path = args.video_manifest or dataset_cfg.get("video_manifest_path")
    if video_manifest_path is None:
        raise ValueError("请通过 --video-manifest 提供逐视频特征清单，或在 config.dataset.video_manifest_path 中配置")

    label_space = str(dataset_cfg.get("label_space", cfg.get("classifier", {}).get("label_space", "canonical")))
    class_names = list(get_label_space_classes(label_space))
    clip_len = int(cfg.get("classifier", {}).get("clip_len", dataset_cfg.get("clip_len", 64)))
    split_key = f"{args.split}_split"
    split = dataset_cfg.get(split_key, args.split)
    dataset_name = args.dataset_name or dataset_cfg.get("dataset_name")
    sequence_source = str(dataset_cfg.get("sequence_source", "features")).lower()

    items = filter_by_split(read_jsonl(video_manifest_path), split=split)
    if args.limit_videos > 0:
        items = items[: int(args.limit_videos)]
    if not items:
        raise ValueError(f"在 {video_manifest_path} 中未找到 split={split!r} 的视频")

    classifier = build_classifier(cfg["classifier"])
    out_dir = ensure_dir(args.output_dir)
    jsonl_dir = ensure_dir(args.save_jsonl_dir) if args.save_jsonl_dir else None
    eval_cfg = cfg.get("evaluation", {}) or {}
    smooth_window = args.smooth_window
    if smooth_window is None:
        smooth_window = int(eval_cfg.get("smooth_window", 5))
    smooth_window = max(1, int(smooth_window))
    context_len = args.context_len
    if context_len is None:
        context_len = int(eval_cfg.get("context_len", clip_len))
    context_len = max(int(clip_len), int(context_len))
    window_mode = str(args.window_mode or eval_cfg.get("window_mode", "causal")).lower()
    decoder_cfg = cfg.get("decoder", {}) or {}
    decoder_enabled = bool(decoder_cfg.get("enabled", False))
    task_smoothing_cfg = dict((eval_cfg.get("task_smoothing", {}) or {}))
    task_smoothing_enabled = bool(task_smoothing_cfg.get("enabled", False))
    gesture_smooth_window = max(1, int(task_smoothing_cfg.get("gesture_window", smooth_window)))
    command_smooth_window = max(1, int(task_smoothing_cfg.get("command_window", smooth_window)))
    direction_smooth_window = max(1, int(task_smoothing_cfg.get("direction_window", smooth_window)))
    classifier_cfg = cfg.get("classifier", {}) or {}
    task_fusion_cfg = classifier_cfg.get("task_fusion", {}) or {}
    command_weight = float(task_fusion_cfg.get("command_weight", 0.0))
    direction_weight = float(task_fusion_cfg.get("direction_weight", 0.0))
    task_decoder_cfg = dict((eval_cfg.get("task_decoder", {}) or {}))
    task_decoder_enabled = bool(task_decoder_cfg.get("enabled", False))
    decoder_start = None
    decoder_transition = None
    command_decoder_template = None
    direction_decoder_template = None
    if decoder_enabled:
        fit_split = str(decoder_cfg.get("fit_split", dataset_cfg.get("train_split", "train")))
        start_counts, transition_counts, _ = estimate_transition_statistics(
            video_manifest_path=video_manifest_path,
            dataset_name=dataset_name,
            label_space=label_space,
            split=fit_split,
        )
        decoder_start, decoder_transition = build_online_transition_matrix(
            class_names=class_names,
            start_counts=start_counts,
            transition_counts=transition_counts,
            stay_bias=float(decoder_cfg.get("stay_bias", 18.0)),
            same_direction_bonus=float(decoder_cfg.get("same_direction_bonus", 2.5)),
            same_command_bonus=float(decoder_cfg.get("same_command_bonus", 1.5)),
            cross_direction_scale=float(decoder_cfg.get("cross_direction_scale", 0.15)),
            background_bonus=float(decoder_cfg.get("background_bonus", 1.0)),
        )
    if task_decoder_enabled:
        fit_split = str(task_decoder_cfg.get("fit_split", dataset_cfg.get("train_split", "train")))
        command_start_counts, command_transition_counts, command_class_names = estimate_task_transition_statistics(
            video_manifest_path=video_manifest_path,
            dataset_name=dataset_name,
            label_space=label_space,
            task="command",
            split=fit_split,
        )
        direction_start_counts, direction_transition_counts, direction_class_names = estimate_task_transition_statistics(
            video_manifest_path=video_manifest_path,
            dataset_name=dataset_name,
            label_space=label_space,
            task="direction",
            split=fit_split,
        )
        command_start, command_transition = build_simple_online_transition_matrix(
            command_start_counts,
            command_transition_counts,
            stay_bias=float(task_decoder_cfg.get("command_stay_bias", 8.0)),
        )
        direction_start, direction_transition = build_simple_online_transition_matrix(
            direction_start_counts,
            direction_transition_counts,
            stay_bias=float(task_decoder_cfg.get("direction_stay_bias", 12.0)),
        )
        task_emission_scale = float(task_decoder_cfg.get("emission_scale", 1.0))
        command_decoder_template = (command_class_names, command_start, command_transition, task_emission_scale)
        direction_decoder_template = (direction_class_names, direction_start, direction_transition, task_emission_scale)

    all_gt_gesture: list[str] = []
    all_pred_gesture: list[str] = []
    all_gt_command: list[str] = []
    all_pred_command: list[str] = []
    direction_groups: dict[str, dict[str, list[str]]] = defaultdict(lambda: {"gt": [], "pred": []})
    latency_values: list[float] = []
    video_summaries: list[dict] = []
    wall_tic = perf_counter()

    for item in tqdm(items, desc="continuous_eval_classifier"):
        feature_path = resolve_path(item["feature_path"], video_manifest_path)
        payload = np.load(feature_path, allow_pickle=True)
        if sequence_source == "keypoints":
            if "keypoints" not in payload:
                raise KeyError(f"{feature_path} 中没有 keypoints，无法按 dataset.sequence_source=keypoints 评测")
            sequence = payload["keypoints"].astype(np.float32)
            bbox_track = payload["bbox_track"].astype(np.float32) if "bbox_track" in payload else None
        elif sequence_source == "features":
            if "features" not in payload:
                raise KeyError(f"{feature_path} 中没有 features，无法按 dataset.sequence_source=features 评测")
            sequence = payload["features"].astype(np.float32)
            bbox_track = None
        else:
            raise ValueError(f"未知 dataset.sequence_source: {sequence_source}")
        raw_labels = payload["raw_labels"]
        directions = payload["directions"].astype(str)
        video_path = str(payload["video_path"]) if "video_path" in payload else item.get("video_path", "")

        gt_gesture = [
            normalize_raw_label(
                int(raw_label),
                dataset_name=dataset_name,
                label_space=label_space,
                direction=str(direction),
            )
            for raw_label, direction in zip(raw_labels.tolist(), directions.tolist())
        ]
        gt_command = [gesture_to_command(label) for label in gt_gesture]
        pred_gesture: list[str] = []
        pred_command: list[str] = []
        score_buffer: deque[np.ndarray] = deque(maxlen=smooth_window)
        gesture_task_buffer: deque[np.ndarray] = deque(maxlen=gesture_smooth_window)
        command_task_buffer: deque[np.ndarray] = deque(maxlen=command_smooth_window)
        direction_task_buffer: deque[np.ndarray] = deque(maxlen=direction_smooth_window)
        decoder = None
        command_decoder = None
        direction_decoder = None
        if decoder_enabled:
            decoder = OnlineViterbiDecoder(
                class_names=class_names,
                log_start=decoder_start,
                log_transition=decoder_transition,
                emission_scale=float(decoder_cfg.get("emission_scale", 1.0)),
            )
        if task_decoder_enabled and command_decoder_template is not None and direction_decoder_template is not None:
            command_decoder = OnlineViterbiDecoder(
                class_names=command_decoder_template[0],
                log_start=command_decoder_template[1],
                log_transition=command_decoder_template[2],
                emission_scale=command_decoder_template[3],
            )
            direction_decoder = OnlineViterbiDecoder(
                class_names=direction_decoder_template[0],
                log_start=direction_decoder_template[1],
                log_transition=direction_decoder_template[2],
                emission_scale=direction_decoder_template[3],
            )
        writer = JsonlWriter(jsonl_dir / f"{item['sample_id']}.jsonl") if jsonl_dir is not None else None
        start_idx = len(latency_values)

        try:
            for frame_index in range(len(sequence)):
                tic = perf_counter()
                bbox_window = None
                if bbox_track is not None:
                    bbox_window = _window(bbox_track, frame_index, context_len, mode=window_mode)
                pred = classifier.predict_sequence(
                    _window(sequence, frame_index, context_len, mode=window_mode),
                    bbox_track=bbox_window,
                )
                latency_values.append((perf_counter() - tic) * 1000.0)

                raw_scores = _scores_to_array(pred.scores, class_names)
                raw_label = class_names[int(np.argmax(raw_scores))]
                extras = pred.extras or {}
                if (task_smoothing_enabled or task_decoder_enabled) and "gesture_scores_raw" in extras:
                    gesture_probs = _scores_to_array(extras["gesture_scores_raw"], class_names)
                    if task_smoothing_enabled:
                        gesture_task_buffer.append(gesture_probs)
                        gesture_probs = np.mean(np.stack(list(gesture_task_buffer), axis=0), axis=0)
                    command_probs = None
                    direction_probs = None
                    if "command_scores" in extras:
                        command_names = list(classifier.command_index_to_label.values())
                        command_probs = _scores_to_array(extras["command_scores"], command_names)
                        if task_smoothing_enabled:
                            command_task_buffer.append(command_probs)
                            command_probs = np.mean(np.stack(list(command_task_buffer), axis=0), axis=0)
                        if task_decoder_enabled and command_decoder is not None:
                            _, command_probs = command_decoder.step(command_probs)
                    if "direction_scores" in extras:
                        direction_names = list(classifier.direction_index_to_label.values())
                        direction_probs = _scores_to_array(extras["direction_scores"], direction_names)
                        if task_smoothing_enabled:
                            direction_task_buffer.append(direction_probs)
                            direction_probs = np.mean(np.stack(list(direction_task_buffer), axis=0), axis=0)
                        if task_decoder_enabled and direction_decoder is not None:
                            _, direction_probs = direction_decoder.step(direction_probs)
                    raw_scores = _fuse_task_probs(
                        gesture_probs=gesture_probs,
                        class_names=class_names,
                        command_probs=command_probs,
                        command_label_to_index=classifier.command_label_to_index,
                        direction_probs=direction_probs,
                        direction_label_to_index=classifier.direction_label_to_index,
                        command_weight=command_weight,
                        direction_weight=direction_weight,
                    )
                    raw_label = class_names[int(np.argmax(raw_scores))]
                else:
                    score_buffer.append(raw_scores)
                    avg_scores = np.mean(np.stack(list(score_buffer), axis=0), axis=0)
                    raw_label = class_names[int(np.argmax(avg_scores))]
                    raw_scores = avg_scores / np.clip(np.sum(avg_scores), 1e-8, None)
                pred_label = raw_label
                decoded_scores = raw_scores
                if decoder is not None:
                    pred_label, decoded_scores = decoder.step(raw_scores)
                pred_gesture.append(pred_label)
                pred_command.append(str((pred.extras or {}).get("command_label", gesture_to_command(pred_label))))

                direction = str(directions[frame_index])
                if gt_gesture[frame_index] != class_names[0]:
                    direction_groups[direction]["gt"].append(gt_gesture[frame_index])
                    direction_groups[direction]["pred"].append(pred_label)

                if writer is not None:
                    writer.write({
                        "frame_index": int(frame_index),
                        "video_path": video_path,
                        "gt_gesture": gt_gesture[frame_index],
                        "pred_gesture": pred_label,
                        "pred_gesture_raw": raw_label,
                        "gt_command": gt_command[frame_index],
                        "pred_command": pred_command[-1],
                        "direction": direction,
                        "scores": {name: float(value) for name, value in zip(class_names, decoded_scores.tolist())},
                        "scores_raw": {name: float(value) for name, value in zip(class_names, raw_scores.tolist())},
                        "extras": pred.extras or {},
                    })
        finally:
            if writer is not None:
                writer.close()

        all_gt_gesture.extend(gt_gesture)
        all_pred_gesture.extend(pred_gesture)
        all_gt_command.extend(gt_command)
        all_pred_command.extend(pred_command)

        per_video_gesture = summarize_sequence_metrics(
            gt_gesture,
            pred_gesture,
            class_names=class_names,
            background_label=class_names[0],
        )
        per_video_command = summarize_sequence_metrics(
            gt_command,
            pred_command,
            class_names=CANONICAL_COMMANDS,
            background_label="NO_COMMAND",
        )
        mean_latency_ms = float(sum(latency_values[start_idx:]) / max(len(sequence), 1))
        video_summaries.append({
            "video": item["sample_id"],
            "video_path": video_path,
            "frames": int(len(sequence)),
            "mean_latency_ms": mean_latency_ms,
            "system_fps_estimate": float(1000.0 / mean_latency_ms) if mean_latency_ms > 1e-6 else 0.0,
            "gesture": per_video_gesture,
            "command": per_video_command,
        })

    wall_elapsed = max(perf_counter() - wall_tic, 1e-6)
    gesture_summary = summarize_sequence_metrics(
        all_gt_gesture,
        all_pred_gesture,
        class_names=class_names,
        background_label=class_names[0],
    )
    command_summary = summarize_sequence_metrics(
        all_gt_command,
        all_pred_command,
        class_names=CANONICAL_COMMANDS,
        background_label="NO_COMMAND",
    )
    direction_summary = {
        direction: subset_summary(values["gt"], values["pred"], class_names=class_names)
        for direction, values in sorted(direction_groups.items())
        if values["gt"]
    }

    summary = {
        "config": args.config,
        "video_manifest_path": str(Path(video_manifest_path)),
        "dataset_name": dataset_name,
        "label_space": label_space,
        "split": split,
        "num_videos": len(items),
        "total_frames": len(all_gt_gesture),
        "total_foreground_frames": int(sum(1 for item in all_gt_gesture if item != class_names[0])),
        "mean_latency_ms": float(sum(latency_values) / len(latency_values)) if latency_values else 0.0,
        "system_fps_estimate": float(1000.0 / (sum(latency_values) / len(latency_values))) if latency_values else 0.0,
        "end_to_end_wall_fps": float(len(all_gt_gesture) / wall_elapsed),
        "gesture": gesture_summary,
        "command": command_summary,
        "by_direction_gesture": direction_summary,
        "videos": video_summaries,
        "notes": [
            (
                "该评测直接在预计算特征流上做因果滑窗预测。"
                if sequence_source == "features"
                else "该评测直接在逐帧 keypoints 流上做因果滑窗预测。"
            ),
            f"gesture label_space={label_space}，前景类别数={max(len(class_names) - 1, 0)}。",
            f"对每帧预测做最近 {smooth_window} 帧均值平滑，以更贴近连续识别口径。",
            f"窗口模式={window_mode}，评测上下文长度={context_len}。",
            (
                f"已启用任务级平滑：gesture={gesture_smooth_window}, "
                f"command={command_smooth_window}, direction={direction_smooth_window}。"
                if task_smoothing_enabled else "未启用任务级平滑。"
            ),
            "已启用任务级在线解码。" if task_decoder_enabled else "未启用任务级在线解码。",
            "已启用 online decoder。" if decoder_enabled else "未启用 online decoder。",
        ],
    }
    save_json(summary, out_dir / "metrics.json")
    print(summary)


if __name__ == "__main__":
    main()
