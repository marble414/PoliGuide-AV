#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from tpgr.data.features import bbox_sequence_to_track_features
from tpgr.data.manifests import write_jsonl


def _load_track(track_path: Path, usable_len: int) -> np.ndarray:
    with track_path.open("rb") as f:
        payload = pickle.load(f)
    bbox = np.asarray(payload["bbox"], dtype=np.float32)
    if bbox.ndim != 2 or bbox.shape[1] != 4:
        raise ValueError(f"{track_path} 的 bbox 形状异常: {bbox.shape}")
    if len(bbox) < usable_len:
        raise ValueError(f"{track_path} 帧数不足: {len(bbox)} < {usable_len}")
    return bbox[:usable_len]


def _rewrite_manifest(items: list[dict], out_root: Path, recipe: str) -> list[dict]:
    rewritten = []
    for item in items:
        new_item = dict(item)
        feature_path = Path(str(item["feature_path"]))
        new_item["feature_path"] = str(Path("videos") / feature_path.name)
        attrs = dict(item.get("attributes", {}) or {})
        attrs["feature_recipe"] = recipe
        new_item["attributes"] = attrs
        rewritten.append(new_item)
    return rewritten


def main() -> None:
    parser = argparse.ArgumentParser(description="为 CTPGesture v2 wholebody 33类构建 bbox track 增强特征")
    parser.add_argument("--source-dir", default="data/processed/ctpgesture_v2_wholebody33")
    parser.add_argument("--dataset-root", default="data/raw/police_gesture_v2")
    parser.add_argument("--output-dir", default="data/processed/ctpgesture_v2_wholebody33_trackaug")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    src_root = Path(args.source_dir).resolve()
    dataset_root = Path(args.dataset_root).resolve()
    out_root = Path(args.output_dir).resolve()
    out_videos = out_root / "videos"
    out_videos.mkdir(parents=True, exist_ok=True)
    track_dir = dataset_root / "track_single"

    sample_payload = np.load(next((src_root / "videos").glob("*.npz")), allow_pickle=True)
    input_dim = int(np.asarray(sample_payload["features"]).shape[1])
    recipe = f"{src_root.name}_trackaug_v1"

    for manifest_name in ["manifest_videos.jsonl", "manifest_videos_trainval.jsonl"]:
        src_manifest = src_root / manifest_name
        if not src_manifest.exists():
            continue
        items = [json.loads(line) for line in src_manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
        for item in items:
            src_feature = src_root / item["feature_path"]
            dst_feature = out_videos / Path(str(item["feature_path"])).name
            if args.skip_existing and dst_feature.exists():
                continue
            payload = np.load(src_feature, allow_pickle=True)
            data = {key: payload[key] for key in payload.files}
            stem = str(data["stem"])
            features = np.asarray(data["features"], dtype=np.float32)
            bbox = _load_track(track_dir / f"{stem}.pkl", usable_len=len(features))
            track_features = bbox_sequence_to_track_features(bbox)
            data["bbox_track"] = bbox.astype(np.float32, copy=False)
            data["track_features"] = track_features
            data["features"] = np.concatenate([features, track_features], axis=-1).astype(np.float32, copy=False)
            np.savez_compressed(dst_feature, **data)
        write_jsonl(_rewrite_manifest(items, out_root, recipe), out_root / manifest_name)

    print(
        json.dumps(
            {
                "source_dir": str(src_root),
                "dataset_root": str(dataset_root),
                "output_dir": str(out_root),
                "feature_recipe": recipe,
                "input_dim": input_dim + 14,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
