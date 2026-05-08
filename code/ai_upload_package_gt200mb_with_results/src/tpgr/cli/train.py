from __future__ import annotations

import argparse
import copy
import json
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from tpgr.cli.common import configure_runtime, resolve_device, save_json, set_seed
from tpgr.config import ensure_dir, load_config, save_config
from tpgr.data.labels import (
    get_label_to_index,
    get_label_space_classes,
    get_task_classes,
    normalize_raw_label,
)
from tpgr.data.sequence_dataset import SequenceDataset, collate_batch
from tpgr.logging_utils import setup_logger
from tpgr.metrics.classification import classification_report_dict
from tpgr.models.builder import build_model


class FocalCrossEntropyLoss(torch.nn.Module):
    def __init__(
        self,
        gamma: float = 2.0,
        weight: torch.Tensor | None = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = float(gamma)
        self.reduction = reduction
        if weight is not None:
            self.register_buffer("weight", weight.clone().detach().float())
        else:
            self.weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = torch.nn.functional.cross_entropy(
            logits,
            targets,
            weight=self.weight,
            reduction="none",
        )
        pt = torch.exp(-ce)
        loss = torch.pow(1.0 - pt, self.gamma) * ce
        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        return loss.mean()


def update_ema_model(ema_model: torch.nn.Module, model: torch.nn.Module, decay: float) -> None:
    with torch.no_grad():
        ema_params = dict(ema_model.named_parameters())
        model_params = dict(model.named_parameters())
        for name, param in model_params.items():
            if name in ema_params:
                ema_params[name].mul_(decay).add_(param.detach(), alpha=1.0 - decay)
        ema_buffers = dict(ema_model.named_buffers())
        for name, buffer in model.named_buffers():
            if name in ema_buffers:
                ema_buffers[name].copy_(buffer.detach())


def _normalize_logits(logits) -> dict[str, torch.Tensor]:
    return logits if isinstance(logits, dict) else {"gesture": logits}


def _task_metrics(task_logits: dict[str, torch.Tensor], task_targets: dict[str, torch.Tensor], task_classes: dict[str, list[str]]) -> dict[str, dict]:
    metrics = {}
    for task, logits in task_logits.items():
        if task not in task_targets:
            continue
        pred = torch.argmax(logits, dim=-1)
        metrics[task] = classification_report_dict(
            task_targets[task].cpu().tolist(),
            pred.cpu().tolist(),
            class_names=task_classes[task],
        )
    return metrics


def _compute_multitask_loss(
    task_logits: dict[str, torch.Tensor],
    task_targets: dict[str, torch.Tensor],
    criterions: dict[str, torch.nn.Module],
    task_weights: dict[str, float],
) -> tuple[torch.Tensor, dict[str, float]]:
    losses = {}
    total = None
    for task, logits in task_logits.items():
        if task not in task_targets or task not in criterions:
            continue
        loss = criterions[task](logits, task_targets[task])
        losses[task] = float(loss.detach().item())
        weighted = loss * float(task_weights.get(task, 1.0))
        total = weighted if total is None else total + weighted
    if total is None:
        raise ValueError("没有可用的任务 loss")
    return total, losses


def _flatten_metrics(metrics: dict[str, dict]) -> dict[str, float]:
    flat = {}
    for task, task_metric in metrics.items():
        prefix = "" if task == "gesture" else f"{task}_"
        for key, value in task_metric.items():
            flat[f"{prefix}{key}"] = value
    return flat


def evaluate(
    model,
    loader,
    device,
    task_classes: dict[str, list[str]],
    criterions: dict[str, torch.nn.Module],
    task_weights: dict[str, float],
    amp_enabled: bool = False,
):
    model.eval()
    all_targets = {task: [] for task in task_classes}
    all_preds = {task: [] for task in task_classes}
    total_loss = 0.0
    autocast = torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled) \
        if str(device).startswith("cuda") else nullcontext()
    with torch.inference_mode():
        for batch in loader:
            x = batch["x"].float().to(device, non_blocking=True)
            task_targets = {
                task: target.to(device, non_blocking=True)
                for task, target in batch.get("targets", {}).items()
            }
            with autocast:
                logits = _normalize_logits(model(x))
                loss, _ = _compute_multitask_loss(logits, task_targets, criterions, task_weights)
            batch_size = len(batch["y"])
            total_loss += float(loss.item()) * batch_size
            for task, task_logit in logits.items():
                if task not in task_targets:
                    continue
                pred = torch.argmax(task_logit, dim=-1)
                all_targets[task].extend(task_targets[task].cpu().tolist())
                all_preds[task].extend(pred.cpu().tolist())
    metrics = _flatten_metrics({
        task: classification_report_dict(all_targets[task], all_preds[task], class_names=task_classes[task])
        for task in task_classes
        if all_targets[task]
    })
    metrics["loss"] = total_loss / max(len(all_targets.get("gesture", [])), 1)
    return metrics


def build_class_weights(
    train_set: SequenceDataset,
    dataset_name: str | None,
    mode: str,
    label_space: str,
    beta: float = 0.999,
) -> torch.Tensor | None:
    mode = (mode or "none").lower()
    if mode in {"none", "off", "false"}:
        return None

    class_names = list(get_label_space_classes(label_space))
    label_to_index = get_label_to_index(label_space)
    counts = torch.zeros(len(class_names), dtype=torch.float32)
    for sample in train_set.samples:
        label_name = normalize_raw_label(
            sample["label"],
            dataset_name=dataset_name,
            label_space=label_space,
            direction=sample.get("direction") or (sample.get("attributes") or {}).get("direction"),
        )
        counts[label_to_index[label_name]] += 1.0
    counts = torch.clamp(counts, min=1.0)

    if mode in {"inverse", "inv"}:
        weights = 1.0 / counts
    elif mode in {"inverse_sqrt", "inv_sqrt"}:
        weights = 1.0 / torch.sqrt(counts)
    elif mode in {"effective_num", "cb"}:
        beta = min(max(float(beta), 0.0), 0.999999)
        weights = (1.0 - beta) / torch.clamp(1.0 - torch.pow(beta, counts), min=1e-8)
    else:
        raise ValueError(f"未知 class_weighting 模式: {mode}")

    weights = weights / weights.mean()
    return weights


def main() -> None:
    parser = argparse.ArgumentParser(description="训练交警手势时序分类器")
    parser.add_argument("--config", required=True, help="配置文件路径")
    parser.add_argument("--override", nargs="*", default=[], help="形如 trainer.epochs=8 的覆盖项")
    args = parser.parse_args()

    configure_runtime(1)
    cfg = load_config(args.config, *args.override)
    set_seed(int(cfg.get("seed", 42)))
    label_space = str(cfg["dataset"].get("label_space", "canonical"))
    model_task_dims = cfg["model"].get("task_dims") or {}
    task_names = [str(task).lower() for task in (cfg["dataset"].get("tasks") or list(model_task_dims.keys()) or ["gesture"])]
    if "gesture" not in task_names:
        task_names = ["gesture", *task_names]
    task_classes = {task: list(get_task_classes(task, label_space=label_space)) for task in task_names}
    class_names = task_classes["gesture"]

    run_dir = ensure_dir(cfg["trainer"]["output_dir"])
    logger = setup_logger(log_file=str(run_dir / "train.log"))
    save_config(cfg, run_dir / "resolved_config.yaml")

    representation = cfg["dataset"].get("representation", "features")
    train_set = SequenceDataset(
        manifest_path=cfg["dataset"]["manifest_path"],
        split=cfg["dataset"].get("train_split", "train"),
        clip_len=int(cfg["dataset"].get("clip_len", 48)),
        representation=representation,
        training=True,
        dataset_name=cfg["dataset"].get("dataset_name"),
        feature_variant=cfg["dataset"].get("feature_variant", "base"),
        augment=cfg["dataset"].get("augment"),
        temporal_mode=cfg["dataset"].get("temporal_mode", "crop_pad"),
        label_space=label_space,
        tasks=task_names,
    )
    val_set = SequenceDataset(
        manifest_path=cfg["dataset"]["manifest_path"],
        split=cfg["dataset"].get("val_split", "val"),
        clip_len=int(cfg["dataset"].get("clip_len", 48)),
        representation=representation,
        training=False,
        dataset_name=cfg["dataset"].get("dataset_name"),
        feature_variant=cfg["dataset"].get("feature_variant", "base"),
        temporal_mode=cfg["dataset"].get("temporal_mode", "crop_pad"),
        label_space=label_space,
        tasks=task_names,
    )
    train_loader_kwargs = {
        "batch_size": int(cfg["trainer"].get("batch_size", 16)),
        "shuffle": True,
        "num_workers": int(cfg["trainer"].get("num_workers", 0)),
        "pin_memory": bool(cfg["trainer"].get("pin_memory", True)),
        "persistent_workers": bool(cfg["trainer"].get("persistent_workers", True)) and int(cfg["trainer"].get("num_workers", 0)) > 0,
        "collate_fn": collate_batch,
    }
    val_loader_kwargs = {
        "batch_size": int(cfg["trainer"].get("batch_size", 16)),
        "shuffle": False,
        "num_workers": int(cfg["trainer"].get("num_workers", 0)),
        "pin_memory": bool(cfg["trainer"].get("pin_memory", True)),
        "persistent_workers": bool(cfg["trainer"].get("persistent_workers", True)) and int(cfg["trainer"].get("num_workers", 0)) > 0,
        "collate_fn": collate_batch,
    }
    if int(cfg["trainer"].get("num_workers", 0)) > 0:
        prefetch_factor = int(cfg["trainer"].get("prefetch_factor", 2))
        train_loader_kwargs["prefetch_factor"] = prefetch_factor
        val_loader_kwargs["prefetch_factor"] = prefetch_factor

    train_loader = DataLoader(train_set, **train_loader_kwargs)
    val_loader = DataLoader(val_set, **val_loader_kwargs)

    model = build_model(cfg["model"])
    device = resolve_device(cfg.get("device", "auto"))
    logger.info("use device=%s", device)
    model.to(device)
    init_checkpoint = cfg["trainer"].get("init_checkpoint")
    if init_checkpoint:
        logger.info("load init checkpoint=%s", init_checkpoint)
        state = torch.load(init_checkpoint, map_location=device, weights_only=False)
        state_dict = state["model"] if isinstance(state, dict) and "model" in state else state
        model.load_state_dict(state_dict, strict=bool(cfg["trainer"].get("init_strict", True)))
    train_model = model
    use_ema = bool(cfg["trainer"].get("ema", True))
    ema_decay = float(cfg["trainer"].get("ema_decay", 0.999))
    ema_eval_start_epoch = int(cfg["trainer"].get("ema_eval_start_epoch", 1))
    ema_model = None
    if use_ema:
        ema_model = copy.deepcopy(model).to(device).eval()
        for param in ema_model.parameters():
            param.requires_grad_(False)
    compile_enabled = bool(cfg["trainer"].get("compile", False)) and hasattr(torch, "compile")
    if compile_enabled:
        compile_mode = str(cfg["trainer"].get("compile_mode", "reduce-overhead"))
        logger.info("compile model mode=%s", compile_mode)
        train_model = torch.compile(model, mode=compile_mode)
    amp_enabled = bool(cfg["trainer"].get("amp", True)) and str(device).startswith("cuda")
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    class_weights = build_class_weights(
        train_set,
        dataset_name=cfg["dataset"].get("dataset_name"),
        mode=str(cfg["trainer"].get("class_weighting", "none")),
        label_space=label_space,
        beta=float(cfg["trainer"].get("class_weight_beta", 0.999)),
    )
    if class_weights is not None:
        logger.info(
            "class weights: %s",
            {name: round(float(value), 4) for name, value in zip(class_names, class_weights.tolist())},
        )
        class_weights = class_weights.to(device)

    label_smoothing = float(cfg["trainer"].get("label_smoothing", 0.0))
    gesture_loss_name = str(cfg["trainer"].get("gesture_loss", "ce")).lower()
    if gesture_loss_name in {"ce", "cross_entropy"}:
        gesture_criterion = torch.nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=label_smoothing,
        )
    elif gesture_loss_name in {"focal", "focal_ce", "focal_cross_entropy"}:
        if label_smoothing > 0:
            logger.warning("gesture_loss=focal 时忽略 label_smoothing=%.4f", label_smoothing)
        gesture_criterion = FocalCrossEntropyLoss(
            gamma=float(cfg["trainer"].get("focal_gamma", 2.0)),
            weight=class_weights,
            reduction="mean",
        )
    else:
        raise ValueError(f"未知 gesture_loss: {gesture_loss_name}")

    criterions = {
        "gesture": gesture_criterion,
    }
    for task in task_names:
        if task == "gesture":
            continue
        criterions[task] = torch.nn.CrossEntropyLoss(label_smoothing=0.0)
    task_weights = {
        task: float((cfg["trainer"].get("task_weights", {}) or {}).get(task, 1.0))
        for task in task_names
    }
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["trainer"].get("lr", 1e-3)),
        weight_decay=float(cfg["trainer"].get("weight_decay", 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(int(cfg["trainer"].get("epochs", 10)), 1)
    )

    best_f1 = -1.0
    history = []
    epochs = int(cfg["trainer"].get("epochs", 10))
    grad_clip = float(cfg["trainer"].get("grad_clip", 1.0))
    for epoch in range(1, epochs + 1):
        train_model.train()
        total_loss = 0.0
        all_targets = {task: [] for task in task_names}
        all_preds = {task: [] for task in task_names}
        for batch_idx, batch in enumerate(train_loader, start=1):
            x = batch["x"].float().to(device, non_blocking=True)
            task_targets = {
                task: target.to(device, non_blocking=True)
                for task, target in batch.get("targets", {}).items()
            }

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled) \
                    if str(device).startswith("cuda") else nullcontext():
                logits = _normalize_logits(train_model(x))
                loss, _ = _compute_multitask_loss(logits, task_targets, criterions, task_weights)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            if ema_model is not None:
                update_ema_model(ema_model, model, decay=ema_decay)

            batch_size = len(batch["y"])
            total_loss += float(loss.item()) * batch_size
            for task, task_logit in logits.items():
                if task not in task_targets:
                    continue
                pred = torch.argmax(task_logit, dim=-1)
                all_targets[task].extend(task_targets[task].cpu().tolist())
                all_preds[task].extend(pred.cpu().tolist())
            if batch_idx % 10 == 0 or batch_idx == len(train_loader):
                logger.info("epoch=%s step=%s/%s loss=%.4f", epoch, batch_idx, len(train_loader), float(loss.item()))

        scheduler.step()
        train_metrics = _flatten_metrics({
            task: classification_report_dict(all_targets[task], all_preds[task], class_names=task_classes[task])
            for task in task_names
            if all_targets[task]
        })
        train_metrics["loss"] = total_loss / max(len(all_targets.get("gesture", [])), 1)

        use_ema_for_eval = ema_model is not None and epoch >= ema_eval_start_epoch
        eval_model = ema_model if use_ema_for_eval else model
        val_metrics = evaluate(
            eval_model,
            val_loader,
            device,
            task_classes=task_classes,
            criterions=criterions,
            task_weights=task_weights,
            amp_enabled=amp_enabled,
        )
        record = {"epoch": epoch, "train": train_metrics, "val": val_metrics, "lr": scheduler.get_last_lr()[0]}
        history.append(record)
        logger.info("epoch=%s train_loss=%.4f val_macro_f1=%.4f val_acc=%.4f",
                    epoch, train_metrics["loss"], val_metrics["macro_f1"], val_metrics["accuracy"])

        checkpoint = {
            "model": eval_model.state_dict(),
            "epoch": epoch,
            "config": cfg,
        }
        if ema_model is not None:
            checkpoint["ema_model"] = ema_model.state_dict()
            checkpoint["raw_model"] = model.state_dict()
            checkpoint["model_source"] = "ema" if use_ema_for_eval else "raw"
        else:
            checkpoint["model_source"] = "raw"
        torch.save(checkpoint, run_dir / "last.pt")
        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            torch.save(checkpoint, run_dir / "best.pt")

        with (run_dir / "history.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    save_json({"best_macro_f1": best_f1, "history": history}, run_dir / "summary.json")


if __name__ == "__main__":
    main()
