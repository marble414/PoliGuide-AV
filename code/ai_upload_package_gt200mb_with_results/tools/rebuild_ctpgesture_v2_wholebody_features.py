#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tpgr.data.features import infer_tcn_feature_dim, sequence_to_tcn_features
from tpgr.data.manifests import write_jsonl


def _load_manifest_items(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="基于已提取 wholebody keypoints 重建 CTPGesture v2 特征")
    parser.add_argument("--source-dir", default="data/processed/ctpgesture_v2_wholebody33")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--feature-variant", default="wholebody_full")
    parser.add_argument("--include-velocity", action="store_true", default=True)
    parser.add_argument("--limit-videos", type=int, default=0)
    args = parser.parse_args()

    source_root = Path(args.source_dir).resolve()
    out_root = Path(args.output_dir).resolve()
    out_videos = out_root / "videos"
    out_videos.mkdir(parents=True, exist_ok=True)

    manifest_names = ["manifest_videos.jsonl", "manifest_videos_trainval.jsonl"]
    feature_variant = str(args.feature_variant)
    rewritten_manifests: dict[str, list[dict]] = {}

    for manifest_name in manifest_names:
        src_manifest = source_root / manifest_name
        if not src_manifest.exists():
            continue
        items = _load_manifest_items(src_manifest)
        if args.limit_videos > 0:
            items = items[: int(args.limit_videos)]
        rewritten: list[dict] = []
        for item in items:
            stem = str(item["sample_id"])
            src_npz = source_root / str(item["feature_path"])
            if not src_npz.exists():
                raise FileNotFoundError(src_npz)
            dst_npz = out_videos / f"{stem}.npz"
            payload = np.load(src_npz, allow_pickle=True)
            keypoints = payload["keypoints"].astype(np.float32, copy=False)
            features = sequence_to_tcn_features(
                keypoints,
                include_velocity=bool(args.include_velocity),
                feature_variant=feature_variant,
            )
            save_payload = {key: payload[key] for key in payload.files}
            save_payload["features"] = features.astype(np.float32, copy=False)
            np.savez_compressed(dst_npz, **save_payload)

            new_item = dict(item)
            new_item["feature_path"] = str(Path("videos") / dst_npz.name)
            attrs = dict(item.get("attributes", {}) or {})
            attrs["feature_recipe"] = f"{feature_variant}_v1"
            attrs["feature_dim"] = int(features.shape[-1])
            new_item["attributes"] = attrs
            rewritten.append(new_item)
        rewritten_manifests[manifest_name] = rewritten

    for manifest_name, items in rewritten_manifests.items():
        write_jsonl(items, out_root / manifest_name)

    print(
        json.dumps(
            {
                "source_dir": str(source_root),
                "output_dir": str(out_root),
                "feature_variant": feature_variant,
                "include_velocity": bool(args.include_velocity),
                "feature_dim": infer_tcn_feature_dim(133, include_velocity=bool(args.include_velocity), feature_variant=feature_variant),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
