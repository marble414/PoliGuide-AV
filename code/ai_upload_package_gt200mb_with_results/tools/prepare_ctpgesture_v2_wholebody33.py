#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import cv2
import numpy as np

from tpgr.config import load_config
from tpgr.data.features import infer_tcn_feature_dim, sequence_to_tcn_features
from tpgr.data.manifests import write_jsonl
from tpgr.pipeline.system import build_pose_backend


def _load_manifest_items(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _merge_manifest_items(existing_path: Path, items: list[dict]) -> list[dict]:
    merged: dict[str, dict] = {}
    if existing_path.exists():
        for item in _load_manifest_items(existing_path):
            merged[str(item["sample_id"])] = item
    for item in items:
        merged[str(item["sample_id"])] = item
    return [merged[key] for key in sorted(merged)]


def _centerwh_to_xyxy(bbox: np.ndarray, width: int, height: int, scale: float) -> tuple[int, int, int, int]:
    cx, cy, bw, bh = [float(v) for v in bbox.tolist()]
    half_w = max(1.0, bw * float(scale) / 2.0)
    half_h = max(1.0, bh * float(scale) / 2.0)
    x1 = max(0, int(round(cx - half_w)))
    y1 = max(0, int(round(cy - half_h)))
    x2 = min(width, int(round(cx + half_w)))
    y2 = min(height, int(round(cy + half_h)))
    return x1, y1, max(x1 + 1, x2), max(y1 + 1, y2)


def _pick_best_detection(detections: list[dict]) -> dict | None:
    if not detections:
        return None
    return max(detections, key=lambda det: (float(det.get("score", 0.0)), np.mean(det.get("keypoint_scores", [0.0]))))


def _project_detection_to_full_frame(det: dict, x1: int, y1: int, num_keypoints: int) -> np.ndarray:
    kpts = np.asarray(det.get("keypoints", []), dtype=np.float32)
    scores = np.asarray(det.get("keypoint_scores", np.ones((len(kpts),), dtype=np.float32)), dtype=np.float32)
    if kpts.ndim != 2 or kpts.shape[1] != 2:
        return np.zeros((num_keypoints, 3), dtype=np.float32)
    if scores.ndim == 0:
        scores = np.full((kpts.shape[0],), float(scores), dtype=np.float32)
    full = np.zeros((kpts.shape[0], 3), dtype=np.float32)
    full[:, :2] = kpts + np.asarray([[float(x1), float(y1)]], dtype=np.float32)
    full[:, 2] = scores
    if full.shape[0] != num_keypoints:
        out = np.zeros((num_keypoints, 3), dtype=np.float32)
        usable = min(num_keypoints, full.shape[0])
        out[:usable] = full[:usable]
        return out
    return full


def main() -> None:
    parser = argparse.ArgumentParser(description="构建 CTPGesture v2 wholebody 33类全视频特征")
    parser.add_argument("--pose-config", default="configs/deploy_openmmlab_wholebody_local.yaml")
    parser.add_argument("--source-dir", default="data/processed/ctpgesture_v2_vibe33")
    parser.add_argument("--dataset-root", default="data/raw/police_gesture_v2")
    parser.add_argument("--output-dir", default="data/processed/ctpgesture_v2_wholebody33")
    parser.add_argument("--crop-scale", type=float, default=1.45)
    parser.add_argument("--limit-videos", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--include-stems", nargs="*", default=None)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.pose_config)
    pose_backend = build_pose_backend(cfg["runtime"]["pose_backend"])
    num_keypoints = int(getattr(pose_backend, "num_keypoints", 133))

    source_root = Path(args.source_dir).resolve()
    dataset_root = Path(args.dataset_root).resolve()
    out_root = Path(args.output_dir).resolve()
    out_videos = out_root / "videos"
    out_videos.mkdir(parents=True, exist_ok=True)

    label_dir = dataset_root / "label_combine_frame"
    direction_dir = dataset_root / "label_ori_frame"
    track_dir = dataset_root / "track_single"

    for manifest_name in ["manifest_videos.jsonl", "manifest_videos_trainval.jsonl"]:
        src_manifest = source_root / manifest_name
        if not src_manifest.exists():
            continue
        items = _load_manifest_items(src_manifest)
        if args.include_stems:
            include = {str(stem) for stem in args.include_stems}
            items = [item for item in items if str(item["sample_id"]) in include]
        if args.limit_videos > 0:
            items = items[: int(args.limit_videos)]

        rewritten = []
        for item in items:
            stem = str(item["sample_id"])
            video_path = Path(str(item["video_path"])).resolve()
            dst_feature = out_videos / f"{stem}.npz"
            new_item = dict(item)
            new_item["feature_path"] = str(Path("videos") / dst_feature.name)
            attrs = dict(item.get("attributes", {}) or {})
            attrs["feature_recipe"] = "wholebody_focus_v1"
            new_item["attributes"] = attrs
            rewritten.append(new_item)
            if args.skip_existing and dst_feature.exists():
                continue

            raw_labels = np.asarray(json.loads((label_dir / f"{stem}.json").read_text(encoding="utf-8")), dtype=np.int64)
            directions = np.asarray(json.loads((direction_dir / f"{stem}.json").read_text(encoding="utf-8")), dtype=object)
            track_payload = pickle.load((track_dir / f"{stem}.pkl").open("rb"))
            track_bbox = np.asarray(track_payload["bbox"], dtype=np.float32)

            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise FileNotFoundError(video_path)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if hasattr(cv2, "CAP_PROP_FRAME_COUNT") else 0
            usable_len = min(len(raw_labels), len(directions), len(track_bbox))
            if frame_count > 0:
                usable_len = min(usable_len, frame_count)
            if args.max_frames > 0:
                usable_len = min(usable_len, int(args.max_frames))

            keypoints = np.zeros((usable_len, num_keypoints, 3), dtype=np.float32)
            try:
                for frame_idx in range(usable_len):
                    ok, frame = cap.read()
                    if not ok:
                        usable_len = frame_idx
                        keypoints = keypoints[:usable_len]
                        break
                    x1, y1, x2, y2 = _centerwh_to_xyxy(track_bbox[frame_idx], width, height, scale=float(args.crop_scale))
                    crop = frame[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue
                    detections = pose_backend.estimate(crop, frame_idx)
                    det = _pick_best_detection(detections)
                    if det is not None:
                        keypoints[frame_idx] = _project_detection_to_full_frame(det, x1, y1, num_keypoints)
            finally:
                cap.release()

            raw_labels = raw_labels[:usable_len]
            directions = directions[:usable_len]
            track_bbox = track_bbox[:usable_len]
            features = sequence_to_tcn_features(keypoints, include_velocity=True, feature_variant="wholebody_focus")
            np.savez_compressed(
                dst_feature,
                features=features.astype(np.float32, copy=False),
                keypoints=keypoints.astype(np.float32, copy=False),
                bbox_track=track_bbox.astype(np.float32, copy=False),
                raw_labels=raw_labels,
                directions=directions.astype(str),
                video_path=str(video_path),
                split=str(item["split"]),
                stem=stem,
            )

        manifest_path = out_root / manifest_name
        if args.include_stems:
            write_jsonl(_merge_manifest_items(manifest_path, rewritten), manifest_path)
        else:
            write_jsonl(rewritten, manifest_path)

    print(
        json.dumps(
            {
                "source_dir": str(source_root),
                "dataset_root": str(dataset_root),
                "output_dir": str(out_root),
                "feature_recipe": "wholebody_focus_v1",
                "feature_dim": infer_tcn_feature_dim(133, include_velocity=True, feature_variant="wholebody_focus"),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
