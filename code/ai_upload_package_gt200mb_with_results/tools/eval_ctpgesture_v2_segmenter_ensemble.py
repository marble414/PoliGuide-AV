#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from tpgr.cli.common import configure_runtime, resolve_device, save_json
from tpgr.config import ensure_dir, load_config
from tpgr.ctpgv2_segmenter import apply_gesture_bias, fuse_multitask_probs, load_video_records, moving_average_probs
from tpgr.data.labels import get_label_space_classes, get_task_classes
from tpgr.decoding import OnlineViterbiDecoder, build_online_transition_matrix, estimate_transition_statistics
from tpgr.eval_continuous import subset_summary, summarize_sequence_metrics
from tpgr.models.builder import build_model


def _predict_probs(
    model: torch.nn.Module,
    features: np.ndarray,
    device: torch.device,
    amp_enabled: bool,
    chunk_len: int,
    overlap: int,
) -> dict[str, np.ndarray]:
    total = len(features)
    step = max(1, int(chunk_len) - int(overlap))
    task_sums: dict[str, np.ndarray] = {}
    counts = np.zeros((total, 1), dtype=np.float32)
    autocast = torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled) \
        if str(device).startswith("cuda") else nullcontext()
    with torch.inference_mode():
        for start in range(0, total, step):
            end = min(total, start + int(chunk_len))
            chunk = features[start:end]
            valid_len = len(chunk)
            x = torch.from_numpy(chunk).unsqueeze(0).float().to(device, non_blocking=True)
            with autocast:
                logits = model(x)
            for task, task_logits in logits.items():
                probs = torch.softmax(task_logits, dim=1)[0].transpose(0, 1).cpu().numpy().astype(np.float32)
                probs = probs[:valid_len]
                if task not in task_sums:
                    task_sums[task] = np.zeros((total, probs.shape[1]), dtype=np.float32)
                task_sums[task][start:end] += probs
            counts[start:end, 0] += 1.0
            if end >= total:
                break
    counts = np.clip(counts, 1.0, None)
    return {task: value / counts for task, value in task_sums.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="评测 CTPGesture v2 frame-segmenter 多模型集成")
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--override", nargs="*", default=[])
    args = parser.parse_args()

    configure_runtime(1)
    cfg = load_config(args.config, *args.override)
    device = torch.device(resolve_device(str(cfg.get("device", "auto"))))
    label_space = str(cfg["dataset"].get("label_space", "ctpv2_directional"))
    dataset_name = str(cfg["dataset"].get("dataset_name", "ctpgesture_v2"))
    split = args.split or str(cfg["dataset"].get("eval_split", cfg["dataset"].get("train_split", "train")))
    records = load_video_records(
        manifest_path=cfg["dataset"]["video_manifest_path"],
        split=split,
        dataset_name=dataset_name,
        label_space=label_space,
        direction_target_source=str(cfg["dataset"].get("direction_target_source", "gesture")),
    )
    out_dir = ensure_dir(args.output_dir)

    gesture_classes = list(get_label_space_classes(label_space))
    command_classes = list(get_task_classes("command", label_space=label_space))
    eval_cfg = cfg.get("evaluation", {}) or {}
    smooth_window = int(eval_cfg.get("smooth_window", 1))
    gesture_smooth_window = int(eval_cfg.get("gesture_smooth_window", smooth_window))
    command_smooth_window = int(eval_cfg.get("command_smooth_window", smooth_window))
    direction_smooth_window = int(eval_cfg.get("direction_smooth_window", smooth_window))
    fusion_cfg = eval_cfg.get("task_fusion", {}) or {}
    gesture_bias = dict(eval_cfg.get("gesture_bias", {}) or {})

    members_cfg = list(cfg.get("members", []))
    if not members_cfg:
        raise ValueError("ensemble 配置需要 members")

    members: list[dict] = []
    for member_cfg in members_cfg:
        member_source = load_config(member_cfg["config"])
        model = build_model(member_source["model"]).to(device)
        state = torch.load(member_cfg["checkpoint"], map_location=device, weights_only=False)
        model.load_state_dict(state["model"] if "model" in state else state, strict=True)
        model.eval()
        member_dataset = member_source.get("dataset", {}) or {}
        member_manifest_path = member_dataset.get("video_manifest_path", cfg["dataset"]["video_manifest_path"])
        member_dataset_name = str(member_dataset.get("dataset_name", dataset_name))
        member_label_space = str(member_dataset.get("label_space", label_space))
        if str(member_manifest_path) == str(cfg["dataset"]["video_manifest_path"]) and member_label_space == label_space:
            record_map = None
        else:
            member_records = load_video_records(
                manifest_path=member_manifest_path,
                split=split,
                dataset_name=member_dataset_name,
                label_space=member_label_space,
                direction_target_source=str(member_dataset.get("direction_target_source", "gesture")),
            )
            record_map = {record.sample_id: record for record in member_records}
        members.append({
            "name": str(member_cfg.get("name", Path(str(member_cfg["checkpoint"])).parent.name)),
            "weight": float(member_cfg.get("weight", 1.0)),
            "model": model,
            "amp": bool(member_cfg.get("amp", True)),
            "chunk_len": int(member_cfg.get("chunk_len", member_source.get("evaluation", {}).get("chunk_len", 4096))),
            "overlap": int(member_cfg.get("overlap", member_source.get("evaluation", {}).get("overlap", 512))),
            "record_map": record_map,
        })

    decoder = None
    if bool((cfg.get("decoder", {}) or {}).get("enabled", False)):
        decoder_cfg = cfg.get("decoder", {}) or {}
        fit_split = str(decoder_cfg.get("fit_split", cfg["dataset"].get("train_split", "train")))
        start_counts, transition_counts, _ = estimate_transition_statistics(
            video_manifest_path=cfg["dataset"]["video_manifest_path"],
            dataset_name=dataset_name,
            label_space=label_space,
            split=fit_split,
        )
        log_start, log_transition = build_online_transition_matrix(
            class_names=gesture_classes,
            start_counts=start_counts,
            transition_counts=transition_counts,
            stay_bias=float(decoder_cfg.get("stay_bias", 18.0)),
            same_direction_bonus=float(decoder_cfg.get("same_direction_bonus", 2.5)),
            same_command_bonus=float(decoder_cfg.get("same_command_bonus", 1.5)),
            cross_direction_scale=float(decoder_cfg.get("cross_direction_scale", 0.15)),
            background_bonus=float(decoder_cfg.get("background_bonus", 1.0)),
        )
        decoder = OnlineViterbiDecoder(
            class_names=gesture_classes,
            log_start=log_start,
            log_transition=log_transition,
            emission_scale=float(decoder_cfg.get("emission_scale", 1.0)),
        )

    all_gt_gesture: list[str] = []
    all_pred_gesture: list[str] = []
    all_gt_command: list[str] = []
    all_pred_command: list[str] = []
    latencies: list[float] = []

    wall_tic = perf_counter()
    for record in records:
        tic = perf_counter()
        gesture_probs = None
        command_probs = None
        direction_probs = None
        total_weight = 0.0
        for member in members:
            member_record = member["record_map"].get(record.sample_id, None) if member["record_map"] is not None else record
            if member_record is None:
                raise KeyError(f"成员 {member['name']} 缺少样本 {record.sample_id} 的特征")
            if len(member_record.features) != len(record.features):
                raise ValueError(
                    f"成员 {member['name']} 的样本 {record.sample_id} 长度不一致: "
                    f"{len(member_record.features)} != {len(record.features)}"
                )
            probs = _predict_probs(
                member["model"],
                member_record.features,
                device,
                amp_enabled=str(device).startswith("cuda") and bool(member["amp"]),
                chunk_len=int(member["chunk_len"]),
                overlap=int(member["overlap"]),
            )
            weight = float(member["weight"])
            total_weight += weight
            gesture_probs = probs["gesture"] * weight if gesture_probs is None else gesture_probs + probs["gesture"] * weight
            command_probs = probs["command"] * weight if command_probs is None else command_probs + probs["command"] * weight
            direction_probs = probs["direction"] * weight if direction_probs is None else direction_probs + probs["direction"] * weight
        gesture_probs = gesture_probs / max(total_weight, 1e-8)
        command_probs = command_probs / max(total_weight, 1e-8)
        direction_probs = direction_probs / max(total_weight, 1e-8)

        if gesture_smooth_window > 1:
            gesture_probs = moving_average_probs(gesture_probs, gesture_smooth_window)
        if command_smooth_window > 1:
            command_probs = moving_average_probs(command_probs, command_smooth_window)
        if direction_smooth_window > 1:
            direction_probs = moving_average_probs(direction_probs, direction_smooth_window)

        fused = fuse_multitask_probs(
            gesture_probs=gesture_probs,
            label_space=label_space,
            command_probs=command_probs,
            direction_probs=direction_probs,
            command_weight=float(fusion_cfg.get("command_weight", 0.0)),
            direction_weight=float(fusion_cfg.get("direction_weight", 0.0)),
            background_agnostic_direction=bool(fusion_cfg.get("background_agnostic_direction", False)),
        )
        fused = apply_gesture_bias(fused, label_space=label_space, bias_map=gesture_bias)
        pred_labels = []
        for frame_index in range(len(fused)):
            frame_probs = fused[frame_index]
            if decoder is not None:
                pred_label, _ = decoder.step(frame_probs)
            else:
                pred_label = gesture_classes[int(np.argmax(frame_probs))]
            pred_labels.append(pred_label)
        pred_command = [command_classes[int(np.argmax(command_probs[frame_idx]))] for frame_idx in range(len(pred_labels))]
        latencies.append((perf_counter() - tic) * 1000.0 / max(len(record.features), 1))

        all_gt_gesture.extend(record.gesture_labels)
        all_pred_gesture.extend(pred_labels)
        all_gt_command.extend(record.command_labels)
        all_pred_command.extend(pred_command)
        if decoder is not None:
            decoder.reset()

    wall_ms = (perf_counter() - wall_tic) * 1000.0
    metrics = {
        "config": args.config,
        "split": split,
        "num_videos": len(records),
        "total_frames": len(all_gt_gesture),
        "members": [
            {
                "name": member["name"],
                "weight": member["weight"],
                "chunk_len": member["chunk_len"],
                "overlap": member["overlap"],
            }
            for member in members
        ],
        "evaluation": {
            "gesture_smooth_window": gesture_smooth_window,
            "command_smooth_window": command_smooth_window,
            "direction_smooth_window": direction_smooth_window,
            "task_fusion": {
                "command_weight": float(fusion_cfg.get("command_weight", 0.0)),
                "direction_weight": float(fusion_cfg.get("direction_weight", 0.0)),
            },
            "gesture_bias": gesture_bias,
            "decoder": cfg.get("decoder", {}) or {"enabled": False},
        },
        "gesture": summarize_sequence_metrics(
            gt=all_gt_gesture,
            pred=all_pred_gesture,
            class_names=gesture_classes,
            background_label=gesture_classes[0],
        ),
        "command": subset_summary(
            gt=all_gt_command,
            pred=all_pred_command,
            class_names=command_classes,
        ),
        "mean_latency_ms": float(np.mean(latencies)) if latencies else 0.0,
        "system_fps_estimate": float(1000.0 / np.mean(latencies)) if latencies and np.mean(latencies) > 0 else 0.0,
        "end_to_end_wall_fps": float(len(all_gt_gesture) / (wall_ms / 1000.0)) if wall_ms > 0 else 0.0,
    }
    save_json(metrics, out_dir / "metrics.json")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
