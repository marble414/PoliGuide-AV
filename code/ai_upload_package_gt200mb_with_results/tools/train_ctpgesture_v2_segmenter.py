from __future__ import annotations

import argparse
import json
import random
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from tpgr.cli.common import configure_runtime, resolve_device, save_json, set_seed
from tpgr.config import ensure_dir, load_config, save_config
from tpgr.ctpgv2_segmenter import (
    build_frame_class_weights,
    fuse_multitask_probs,
    load_video_records,
    moving_average_probs,
)
from tpgr.data.labels import get_label_space_classes, get_task_classes
from tpgr.eval_continuous import subset_summary, summarize_sequence_metrics
from tpgr.logging_utils import setup_logger
from tpgr.models.builder import build_model


class RandomCropVideoDataset(Dataset):
    def __init__(
        self,
        records,
        crop_len: int,
        samples_per_video: int = 8,
        foreground_ratio: float = 0.0,
        transition_ratio: float = 0.0,
        transition_radius: int = 64,
    ) -> None:
        self.records = list(records)
        self.crop_len = int(crop_len)
        self.samples_per_video = int(samples_per_video)
        self.foreground_ratio = float(min(max(foreground_ratio, 0.0), 1.0))
        self.transition_ratio = float(min(max(transition_ratio, 0.0), 1.0))
        self.transition_radius = max(1, int(transition_radius))
        self._foreground_indices: list[np.ndarray] = []
        self._transition_indices: list[np.ndarray] = []
        for record in self.records:
            gesture = np.asarray(record.gesture_targets, dtype=np.int64)
            self._foreground_indices.append(np.flatnonzero(gesture != 0))
            if len(gesture) <= 1:
                self._transition_indices.append(np.zeros((0,), dtype=np.int64))
                continue
            transitions = np.flatnonzero(gesture[1:] != gesture[:-1]) + 1
            self._transition_indices.append(transitions.astype(np.int64, copy=False))

    def __len__(self) -> int:
        return len(self.records) * self.samples_per_video

    def _sample_window(self, total: int, index: int) -> tuple[int, int]:
        if total <= self.crop_len:
            return 0, total
        transition_indices = self._transition_indices[index]
        foreground_indices = self._foreground_indices[index]
        mode = random.random()
        if len(transition_indices) > 0 and mode < self.transition_ratio:
            anchor = int(random.choice(transition_indices.tolist()))
            jitter = random.randint(-self.transition_radius, self.transition_radius)
            start = anchor - self.crop_len // 2 + jitter
        elif len(foreground_indices) > 0 and mode < self.transition_ratio + self.foreground_ratio:
            anchor = int(random.choice(foreground_indices.tolist()))
            start_min = max(0, anchor - self.crop_len + 1)
            start_max = min(anchor, total - self.crop_len)
            start = random.randint(start_min, start_max) if start_max >= start_min else start_min
        else:
            start = random.randint(0, total - self.crop_len)
        start = max(0, min(start, total - self.crop_len))
        return start, start + self.crop_len

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record_index = index % len(self.records)
        record = self.records[record_index]
        total = len(record.features)
        start, end = self._sample_window(total, record_index)
        features = record.features[start:end]
        gesture = record.gesture_targets[start:end]
        command = record.command_targets[start:end]
        direction = record.direction_targets[start:end]
        return {
            "x": torch.from_numpy(features).float(),
            "gesture": torch.from_numpy(gesture).long(),
            "command": torch.from_numpy(command).long(),
            "direction": torch.from_numpy(direction).long(),
        }


def _collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {
        "x": torch.stack([item["x"] for item in batch], dim=0),
        "gesture": torch.stack([item["gesture"] for item in batch], dim=0),
        "command": torch.stack([item["command"] for item in batch], dim=0),
        "direction": torch.stack([item["direction"] for item in batch], dim=0),
    }


def _task_loss(
    logits: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    task_weights: dict[str, float],
    class_weights: dict[str, torch.Tensor | None],
    label_smoothing: float,
    task_loss_types: dict[str, str],
    focal_gamma: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    total = None
    stats: dict[str, float] = {}
    for task, task_logits in logits.items():
        target = batch[task]
        loss_type = str(task_loss_types.get(task, "cross_entropy")).lower()
        if loss_type in {"focal", "focal_ce", "focal_cross_entropy"}:
            log_probs = F.log_softmax(task_logits, dim=1)
            target_log_probs = log_probs.gather(dim=1, index=target.unsqueeze(1)).squeeze(1)
            pt = torch.exp(target_log_probs)
            ce = F.cross_entropy(
                task_logits,
                target,
                weight=class_weights.get(task),
                label_smoothing=label_smoothing,
                reduction="none",
            )
            loss = ((1.0 - pt).clamp(min=0.0) ** float(focal_gamma) * ce).mean()
        else:
            loss = F.cross_entropy(
                task_logits,
                target,
                weight=class_weights.get(task),
                label_smoothing=label_smoothing,
            )
        stats[f"{task}_loss"] = float(loss.detach().item())
        weighted = loss * float(task_weights.get(task, 1.0))
        total = weighted if total is None else total + weighted
    if total is None:
        raise ValueError("没有可用的任务 loss")
    return total, stats


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


def evaluate_records(
    model: torch.nn.Module,
    records,
    cfg: dict,
    device: torch.device,
    amp_enabled: bool,
) -> dict:
    label_space = str(cfg["dataset"].get("label_space", "ctpv2_directional"))
    gesture_classes = list(get_label_space_classes(label_space))
    command_classes = list(get_task_classes("command", label_space=label_space))
    fusion_cfg = cfg.get("evaluation", {}).get("task_fusion", {}) or {}
    smooth_window = int(cfg.get("evaluation", {}).get("smooth_window", 1))
    chunk_len = int(cfg.get("evaluation", {}).get("chunk_len", 4096))
    overlap = int(cfg.get("evaluation", {}).get("overlap", 512))

    all_gt_gesture: list[str] = []
    all_pred_gesture: list[str] = []
    all_gt_command: list[str] = []
    all_pred_command: list[str] = []

    for record in records:
        probs = _predict_probs(model, record.features, device, amp_enabled, chunk_len=chunk_len, overlap=overlap)
        gesture_probs = probs["gesture"]
        command_probs = probs.get("command")
        direction_probs = probs.get("direction")
        if smooth_window > 1:
            gesture_probs = moving_average_probs(gesture_probs, smooth_window)
            if command_probs is not None:
                command_probs = moving_average_probs(command_probs, smooth_window)
            if direction_probs is not None:
                direction_probs = moving_average_probs(direction_probs, smooth_window)
        fused = fuse_multitask_probs(
            gesture_probs=gesture_probs,
            label_space=label_space,
            command_probs=command_probs,
            direction_probs=direction_probs,
            command_weight=float(fusion_cfg.get("command_weight", 0.0)),
            direction_weight=float(fusion_cfg.get("direction_weight", 0.0)),
            background_agnostic_direction=bool(fusion_cfg.get("background_agnostic_direction", False)),
        )
        pred_indices = np.argmax(fused, axis=-1).tolist()
        pred_labels = [gesture_classes[idx] for idx in pred_indices]
        pred_command = [
            command_classes[int(np.argmax(command_probs[frame_idx]))] if command_probs is not None else "NO_COMMAND"
            for frame_idx in range(len(pred_labels))
        ]

        all_gt_gesture.extend(record.gesture_labels)
        all_pred_gesture.extend(pred_labels)
        all_gt_command.extend(record.command_labels)
        all_pred_command.extend(pred_command)

    return {
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
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="训练 CTPGesture v2 整视频逐帧分割模型")
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", nargs="*", default=[])
    args = parser.parse_args()

    configure_runtime(1)
    cfg = load_config(args.config, *args.override)
    set_seed(int(cfg.get("seed", 42)))
    device = torch.device(resolve_device(str(cfg.get("device", "auto"))))

    run_dir = ensure_dir(cfg["trainer"]["output_dir"])
    logger = setup_logger(log_file=str(run_dir / "train.log"))
    save_config(cfg, run_dir / "resolved_config.yaml")

    label_space = str(cfg["dataset"].get("label_space", "ctpv2_directional"))
    dataset_name = str(cfg["dataset"].get("dataset_name", "ctpgesture_v2"))
    train_records = load_video_records(
        manifest_path=cfg["dataset"]["video_manifest_path"],
        split=str(cfg["dataset"].get("train_split", "train")),
        dataset_name=dataset_name,
        label_space=label_space,
        direction_target_source=str(cfg["dataset"].get("direction_target_source", "gesture")),
    )
    eval_records = load_video_records(
        manifest_path=cfg["dataset"]["video_manifest_path"],
        split=str(cfg["dataset"].get("eval_split", cfg["dataset"].get("train_split", "train"))),
        dataset_name=dataset_name,
        label_space=label_space,
        direction_target_source=str(cfg["dataset"].get("direction_target_source", "gesture")),
    )

    train_set = RandomCropVideoDataset(
        records=train_records,
        crop_len=int(cfg["dataset"].get("crop_len", 2048)),
        samples_per_video=int(cfg["dataset"].get("samples_per_video", 8)),
        foreground_ratio=float(cfg["dataset"].get("foreground_ratio", 0.0)),
        transition_ratio=float(cfg["dataset"].get("transition_ratio", 0.0)),
        transition_radius=int(cfg["dataset"].get("transition_radius", 64)),
    )
    train_loader = DataLoader(
        train_set,
        batch_size=int(cfg["trainer"].get("batch_size", 4)),
        shuffle=True,
        num_workers=int(cfg["trainer"].get("num_workers", 0)),
        pin_memory=bool(cfg["trainer"].get("pin_memory", True)),
        persistent_workers=bool(cfg["trainer"].get("persistent_workers", True)) and int(cfg["trainer"].get("num_workers", 0)) > 0,
        collate_fn=_collate,
    )

    model = build_model(cfg["model"]).to(device)
    if bool(cfg["trainer"].get("compile", False)) and hasattr(torch, "compile"):
        model = torch.compile(model, mode=str(cfg["trainer"].get("compile_mode", "reduce-overhead")))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["trainer"].get("lr", 3e-4)),
        weight_decay=float(cfg["trainer"].get("weight_decay", 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(int(cfg["trainer"].get("epochs", 8)), 1),
    )
    amp_enabled = str(device).startswith("cuda") and bool(cfg["trainer"].get("amp", True))
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled) if str(device).startswith("cuda") else None

    class_weights = {
        "gesture": build_frame_class_weights(train_records, "gesture", label_space, mode=cfg["trainer"].get("class_weighting", "inverse_sqrt")),
        "command": build_frame_class_weights(train_records, "command", label_space, mode=cfg["trainer"].get("class_weighting", "inverse_sqrt")),
        "direction": build_frame_class_weights(train_records, "direction", label_space, mode=cfg["trainer"].get("class_weighting", "inverse_sqrt")),
    }
    class_weights = {
        task: (torch.from_numpy(weight).float().to(device) if weight is not None else None)
        for task, weight in class_weights.items()
    }
    task_weights = dict(cfg["trainer"].get("task_weights", {"gesture": 1.0, "command": 0.5, "direction": 0.25}))
    label_smoothing = float(cfg["trainer"].get("label_smoothing", 0.0))
    task_loss_types = {
        str(task): str(loss_type)
        for task, loss_type in dict(cfg["trainer"].get("task_loss_types", {})).items()
    }
    focal_gamma = float(cfg["trainer"].get("focal_gamma", 2.0))

    best_score = float("-inf")
    history: list[dict] = []
    init_checkpoint = cfg["trainer"].get("init_checkpoint")
    if init_checkpoint:
        state = torch.load(init_checkpoint, map_location=device, weights_only=False)
        if "model" in state:
            state = state["model"]
        model.load_state_dict(state, strict=bool(cfg["trainer"].get("init_strict", True)))
        logger.info("load init checkpoint=%s", init_checkpoint)

    autocast = torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled) \
        if str(device).startswith("cuda") else nullcontext()

    epochs = int(cfg["trainer"].get("epochs", 8))
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        steps = 0
        for batch in train_loader:
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with autocast:
                logits = model(batch["x"])
                loss, loss_stats = _task_loss(
                    logits=logits,
                    batch=batch,
                    task_weights=task_weights,
                    class_weights=class_weights,
                    label_smoothing=label_smoothing,
                    task_loss_types=task_loss_types,
                    focal_gamma=focal_gamma,
                )
            if scaler is not None:
                scaler.scale(loss).backward()
                if float(cfg["trainer"].get("grad_clip", 0.0)) > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["trainer"].get("grad_clip", 1.0)))
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if float(cfg["trainer"].get("grad_clip", 0.0)) > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["trainer"].get("grad_clip", 1.0)))
                optimizer.step()
            total_loss += float(loss.item())
            steps += 1
        scheduler.step()

        metrics = evaluate_records(model, eval_records, cfg, device=device, amp_enabled=amp_enabled)
        train_loss = total_loss / max(steps, 1)
        epoch_summary = {
            "epoch": epoch,
            "train_loss": train_loss,
            "eval": metrics,
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(epoch_summary)
        score = float(metrics["gesture"]["foreground_macro_jaccard"])
        checkpoint = {
            "model": model.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
            "config": cfg,
        }
        torch.save(checkpoint, run_dir / "last.pt")
        if score >= best_score:
            best_score = score
            torch.save(checkpoint, run_dir / "best.pt")

        logger.info(
            "epoch=%d train_loss=%.4f gesture_jaccard=%.4f foreground_jaccard=%.4f command_jaccard=%.4f",
            epoch,
            train_loss,
            float(metrics["gesture"]["macro_jaccard"]),
            float(metrics["gesture"]["foreground_macro_jaccard"]),
            float(metrics["command"]["macro_jaccard"]),
        )

    summary = {
        "best_foreground_macro_jaccard": best_score,
        "history": history,
    }
    save_json(summary, run_dir / "summary.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
