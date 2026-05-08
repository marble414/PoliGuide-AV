from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
from tqdm import tqdm

from tpgr.cli.common import configure_runtime, save_json
from tpgr.config import ensure_dir, load_config
from tpgr.data.labels import get_label_space_classes, get_label_to_index, normalize_raw_label
from tpgr.data.manifests import filter_by_split, read_jsonl, resolve_path
from tpgr.metrics.classification import classification_report_dict, save_confusion
from tpgr.pipeline.classifier import build_classifier


def _maybe_slice_features(arr: np.ndarray, item: dict) -> np.ndarray:
    frame_start = item.get("frame_start")
    frame_end = item.get("frame_end")
    if arr.ndim >= 2 and frame_start is not None and frame_end is not None:
        start = max(0, int(frame_start))
        end = max(start, int(frame_end))
        if arr.shape[0] > end:
            return arr[start:end + 1]
    return arr


def main() -> None:
    parser = argparse.ArgumentParser(description="按 classifier 配置评估序列分类器")
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-dir", default="runs/eval_classifier")
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--override", nargs="*", default=[], help="形如 classifier.checkpoint=... 的覆盖项")
    args = parser.parse_args()

    configure_runtime(1)
    cfg = load_config(args.config, *args.override)
    dataset_cfg = cfg.get("dataset", {})
    manifest_path = args.manifest or dataset_cfg.get("manifest_path")
    if manifest_path is None:
        raise ValueError("请通过 --manifest 提供数据清单，或在 config.dataset.manifest_path 中配置")
    split_key = f"{args.split}_split"
    split = dataset_cfg.get(split_key, args.split)
    dataset_name = args.dataset_name or dataset_cfg.get("dataset_name")
    label_space = str(dataset_cfg.get("label_space", cfg.get("classifier", {}).get("label_space", "canonical")))
    class_names = list(get_label_space_classes(label_space))
    label_to_index = get_label_to_index(label_space)

    items = filter_by_split(read_jsonl(manifest_path), split=split)
    classifier = build_classifier(cfg["classifier"])
    out_dir = ensure_dir(args.output_dir)

    all_true, all_pred = [], []
    grouped = defaultdict(lambda: {"true": [], "pred": []})
    outputs = []

    for item in tqdm(items, desc="eval_classifier"):
        path = item.get("keypoints_path") or item.get("feature_path")
        if path is None:
            raise KeyError(f"sample {item.get('sample_id')} 缺少 keypoints_path/feature_path")
        resolved = resolve_path(path, manifest_path)
        data = np.load(resolved)
        keypoints = data["keypoints"].astype(np.float32) if "keypoints" in data else data["features"].astype(np.float32)
        keypoints = _maybe_slice_features(keypoints, item)

        pred = classifier.predict_sequence(keypoints)
        true_label = normalize_raw_label(
            item["label"],
            dataset_name=dataset_name,
            label_space=label_space,
            direction=item.get("direction") or (item.get("attributes") or {}).get("direction"),
        )
        true_idx = label_to_index[true_label]
        pred_idx = label_to_index[pred.label]
        all_true.append(true_idx)
        all_pred.append(pred_idx)

        meta = item.get("attributes", {})
        outputs.append({
            "sample_id": item.get("sample_id"),
            "true_label": true_label,
            "pred_label": pred.label,
            "scores": {key: float(value) for key, value in pred.scores.items()},
            "extras": pred.extras or {},
            "meta": meta,
        })
        for key, value in meta.items():
            grouped[f"{key}={value}"]["true"].append(true_idx)
            grouped[f"{key}={value}"]["pred"].append(pred_idx)

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
