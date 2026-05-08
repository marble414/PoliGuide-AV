#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from tpgr.cli.common import configure_runtime, save_json
from tpgr.config import ensure_dir, load_config
from tpgr.data.labels import CANONICAL_GESTURES, GESTURE_TO_INDEX, normalize_raw_label
from tpgr.data.manifests import filter_by_split, read_jsonl, resolve_path
from tpgr.metrics.classification import classification_report_dict
from tpgr.pipeline.classifier import build_classifier


def main() -> None:
    parser = argparse.ArgumentParser(description="在验证集上搜索双模型融合权重")
    parser.add_argument("--config", required=True, help="ensemble classifier 配置文件")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--split", default="val")
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--steps", type=int, default=21, help="搜索步数，21 表示 0.00 到 1.00 每 0.05 一步")
    parser.add_argument("--metric", default="macro_f1", choices=["macro_f1", "accuracy"])
    parser.add_argument("--output-dir", default="runs/ensemble_tuning")
    args = parser.parse_args()

    configure_runtime(1)
    cfg = load_config(args.config)
    if cfg.get("classifier", {}).get("type") != "ensemble":
        raise ValueError("配置中的 classifier.type 必须为 ensemble")
    members = cfg["classifier"].get("members", [])
    if len(members) != 2:
        raise ValueError("当前仅支持两个成员的融合权重搜索")

    dataset_cfg = cfg.get("dataset", {})
    manifest_path = args.manifest or dataset_cfg.get("manifest_path")
    if manifest_path is None:
        raise ValueError("请通过 --manifest 提供数据清单，或在 config.dataset.manifest_path 中配置")
    split_key = f"{args.split}_split"
    split = dataset_cfg.get(split_key, args.split)
    dataset_name = args.dataset_name or dataset_cfg.get("dataset_name")
    items = filter_by_split(read_jsonl(manifest_path), split=split)
    samples = []
    for item in items:
        path = item.get("keypoints_path") or item.get("feature_path")
        if path is None:
            continue
        keypoints_path = resolve_path(path, manifest_path)
        samples.append({
            "sample_id": item.get("sample_id"),
            "label_idx": GESTURE_TO_INDEX[normalize_raw_label(item["label"], dataset_name=dataset_name)],
            "path": keypoints_path,
        })

    member_classifiers = []
    for member_cfg in members:
        sub_cfg = {k: v for k, v in member_cfg.items() if k != "weight"}
        member_classifiers.append(build_classifier(sub_cfg))

    cached_probs = [[] for _ in member_classifiers]
    y_true = []
    for sample in tqdm(samples, desc="cache_member_probs"):
        data = np.load(sample["path"])
        keypoints = data["keypoints"].astype(np.float32) if "keypoints" in data else data["features"].astype(np.float32)
        y_true.append(sample["label_idx"])
        for idx, classifier in enumerate(member_classifiers):
            pred = classifier.predict_sequence(keypoints)
            cached_probs[idx].append(np.array([pred.scores[label] for label in CANONICAL_GESTURES], dtype=np.float32))

    y_true = np.array(y_true, dtype=np.int64)
    probs_a = np.stack(cached_probs[0], axis=0)
    probs_b = np.stack(cached_probs[1], axis=0)

    results = []
    best = None
    for alpha in np.linspace(0.0, 1.0, num=max(2, int(args.steps))):
        probs = (1.0 - alpha) * probs_a + alpha * probs_b
        y_pred = np.argmax(probs, axis=1).tolist()
        metrics = classification_report_dict(y_true.tolist(), y_pred)
        record = {
            "alpha_member2": float(alpha),
            "alpha_member1": float(1.0 - alpha),
            "metrics": metrics,
        }
        results.append(record)
        score = float(metrics[args.metric])
        if best is None or score > float(best["metrics"][args.metric]):
            best = record

    out_dir = ensure_dir(args.output_dir)
    save_json({
        "best": best,
        "results": results,
    }, out_dir / "ensemble_weight_search.json")

    tuned_cfg = copy.deepcopy(cfg)
    tuned_cfg["classifier"]["members"][0]["weight"] = float(best["alpha_member1"])
    tuned_cfg["classifier"]["members"][1]["weight"] = float(best["alpha_member2"])
    tuned_path = Path(out_dir) / "tuned_ensemble_config.json"
    tuned_path.write_text(json.dumps(tuned_cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    print({"best": best, "tuned_config": str(tuned_path)})


if __name__ == "__main__":
    main()
