#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tpgr.data.manifests import write_jsonl

SMPL24_EDGES = [
    (0, 1), (0, 2), (0, 3),
    (1, 4), (2, 5), (3, 6),
    (4, 7), (5, 8), (6, 9),
    (7, 10), (8, 11), (9, 12),
    (12, 13), (12, 14), (12, 15),
    (13, 16), (14, 17),
    (16, 18), (17, 19),
    (18, 20), (19, 21),
    (20, 22), (21, 23),
]


def _velocity(x: np.ndarray) -> np.ndarray:
    out = np.zeros_like(x, dtype=np.float32)
    out[1:] = x[1:] - x[:-1]
    return out


def _acceleration(x: np.ndarray) -> np.ndarray:
    out = np.zeros_like(x, dtype=np.float32)
    out[2:] = x[2:] - 2.0 * x[1:-1] + x[:-2]
    return out


def build_rich_features(payload: dict[str, np.ndarray]) -> np.ndarray:
    joints24 = np.asarray(payload["smpl_joints24"], dtype=np.float32)
    pose24 = np.asarray(payload["pose_axis_angle"], dtype=np.float32)
    root_orient = np.asarray(payload["root_orient"], dtype=np.float32)
    center_rel = np.asarray(payload["center_rel"], dtype=np.float32)
    center_vel = np.asarray(payload["center_vel"], dtype=np.float32)

    root = joints24[:, :1, :]
    rel = joints24 - root
    bone = np.stack([rel[:, dst] - rel[:, src] for src, dst in SMPL24_EDGES], axis=1)

    scale = np.linalg.norm(bone, axis=-1).mean(axis=1, keepdims=True).astype(np.float32)
    scale = np.clip(scale, 1e-6, None)
    rel = rel / scale[:, None, :]
    bone = bone / scale[:, None, :]

    rel_vel = _velocity(rel)
    bone_vel = _velocity(bone)
    pose_vel = _velocity(pose24)
    pose_acc = _acceleration(pose24)
    center_acc = _acceleration(center_vel)

    root_sin = np.sin(root_orient)
    root_cos = np.cos(root_orient)

    features = np.concatenate(
        [
            rel.reshape(rel.shape[0], -1),
            rel_vel.reshape(rel_vel.shape[0], -1),
            bone.reshape(bone.shape[0], -1),
            bone_vel.reshape(bone_vel.shape[0], -1),
            pose24.reshape(pose24.shape[0], -1),
            pose_vel.reshape(pose_vel.shape[0], -1),
            pose_acc.reshape(pose_acc.shape[0], -1),
            root_sin.reshape(root_sin.shape[0], -1),
            root_cos.reshape(root_cos.shape[0], -1),
            center_rel,
            center_vel,
            center_acc,
        ],
        axis=-1,
    ).astype(np.float32, copy=False)
    return features


def _rewrite_manifest(items: list[dict], out_root: Path) -> list[dict]:
    rewritten = []
    for item in items:
        new_item = dict(item)
        feature_path = Path(str(item["feature_path"]))
        new_item["feature_path"] = str(Path("videos") / feature_path.name)
        attrs = dict(item.get("attributes", {}) or {})
        attrs["feature_recipe"] = "vibe33_rich_v1"
        new_item["attributes"] = attrs
        rewritten.append(new_item)
    return rewritten


def main() -> None:
    parser = argparse.ArgumentParser(description="从现有 VIBE33 视频特征生成 rich feature 版本")
    parser.add_argument("--source-dir", default="data/processed/ctpgesture_v2_vibe33")
    parser.add_argument("--output-dir", default="data/processed/ctpgesture_v2_vibe33_rich")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    src_root = Path(args.source_dir).resolve()
    out_root = Path(args.output_dir).resolve()
    out_videos = out_root / "videos"
    out_videos.mkdir(parents=True, exist_ok=True)

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
            data["features"] = build_rich_features(data)
            np.savez_compressed(dst_feature, **data)
        write_jsonl(_rewrite_manifest(items, out_root), out_root / manifest_name)

    print(
        json.dumps(
            {
                "source_dir": str(src_root),
                "output_dir": str(out_root),
                "feature_recipe": "vibe33_rich_v1",
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
