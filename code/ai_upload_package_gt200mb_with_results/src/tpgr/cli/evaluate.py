from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from tpgr.cli.common import configure_runtime, resolve_device, save_json
from tpgr.config import ensure_dir, load_config
from tpgr.data.labels import get_index_to_label, get_label_space_classes
from tpgr.data.sequence_dataset import SequenceDataset, collate_batch
from tpgr.metrics.classification import classification_report_dict, save_confusion
from tpgr.models.builder import build_model


def main() -> None:
    parser = argparse.ArgumentParser(description="评估时序分类器")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-dir", default="runs/eval")
    parser.add_argument("--override", nargs="*", default=[], help="形如 trainer.batch_size=64 的覆盖项")
    args = parser.parse_args()

    configure_runtime(1)
    cfg = load_config(args.config, *args.override)
    out_dir = ensure_dir(args.output_dir)
    label_space = str(cfg["dataset"].get("label_space", "canonical"))
    class_names = list(get_label_space_classes(label_space))
    index_to_label = get_index_to_label(label_space)
    representation = cfg["dataset"].get("representation", "features")
    split_key = f"{args.split}_split"

    dataset = SequenceDataset(
        manifest_path=cfg["dataset"]["manifest_path"],
        split=cfg["dataset"].get(split_key, args.split),
        clip_len=int(cfg["dataset"].get("clip_len", 48)),
        representation=representation,
        training=False,
        dataset_name=cfg["dataset"].get("dataset_name"),
        feature_variant=cfg["dataset"].get("feature_variant", "base"),
        temporal_mode=cfg["dataset"].get("temporal_mode", "crop_pad"),
        label_space=label_space,
    )
    loader_kwargs = {
        "batch_size": int(cfg["trainer"].get("batch_size", 16)),
        "shuffle": False,
        "num_workers": int(cfg["trainer"].get("num_workers", 0)),
        "pin_memory": bool(cfg["trainer"].get("pin_memory", True)),
        "persistent_workers": bool(cfg["trainer"].get("persistent_workers", True)) and int(cfg["trainer"].get("num_workers", 0)) > 0,
        "collate_fn": collate_batch,
    }
    if int(cfg["trainer"].get("num_workers", 0)) > 0:
        loader_kwargs["prefetch_factor"] = int(cfg["trainer"].get("prefetch_factor", 2))
    loader = DataLoader(dataset, **loader_kwargs)

    device = resolve_device(cfg.get("device", "auto"))
    model = build_model(cfg["model"]).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model"] if "model" in state else state)
    if bool(cfg["trainer"].get("compile", False)) and hasattr(torch, "compile"):
        model = torch.compile(model, mode=str(cfg["trainer"].get("compile_mode", "reduce-overhead")))
    model.eval()
    amp_enabled = bool(cfg["trainer"].get("amp", True)) and str(device).startswith("cuda")

    all_true, all_pred = [], []
    grouped = defaultdict(lambda: {"true": [], "pred": []})
    outputs = []

    with torch.inference_mode():
        for batch in tqdm(loader, desc="eval"):
            x = batch["x"].float().to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
                logits = model(x)
            probs = torch.softmax(logits, dim=-1)
            pred = torch.argmax(probs, dim=-1).cpu().tolist()
            y_true = batch["y"].cpu().tolist()
            for idx, sample_id in enumerate(batch["sample_ids"]):
                true_label = index_to_label[y_true[idx]]
                pred_label = index_to_label[pred[idx]]
                meta = batch["metas"][idx]
                outputs.append({
                    "sample_id": sample_id,
                    "true_label": true_label,
                    "pred_label": pred_label,
                    "scores": {index_to_label[i]: float(probs[idx, i].cpu()) for i in range(probs.shape[1])},
                    "meta": meta,
                })
                for key, value in meta.items():
                    grouped[f"{key}={value}"]["true"].append(y_true[idx])
                    grouped[f"{key}={value}"]["pred"].append(pred[idx])
            all_true.extend(y_true)
            all_pred.extend(pred)

    metrics = classification_report_dict(all_true, all_pred, class_names=class_names)
    grouped_metrics = {
        name: classification_report_dict(items["true"], items["pred"], class_names=class_names)
        for name, items in grouped.items()
        if items["true"]
    }

    save_json(metrics, out_dir / "metrics.json")
    save_json(grouped_metrics, out_dir / "grouped_metrics.json")
    save_json({"predictions": outputs}, out_dir / "predictions.json")
    save_confusion(all_true, all_pred, out_dir / "confusion", class_names=class_names)


if __name__ == "__main__":
    main()
