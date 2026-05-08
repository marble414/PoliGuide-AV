#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from tpgr.config import load_config
from tpgr.data.manifests import read_jsonl, resolve_path, write_jsonl
from tpgr.pipeline.detector import HOGPersonDetector
from tpgr.pipeline.runtime_utils import resolve_pose_candidates
from tpgr.pipeline.system import build_pose_backend


def main() -> None:
    parser = argparse.ArgumentParser(description="从 manifest 视频片段中提取关键点序列")
    parser.add_argument("--config", required=True, help="含 runtime.pose_backend 的配置")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    pose_backend = build_pose_backend(cfg["runtime"]["pose_backend"])
    detector = HOGPersonDetector(**cfg["runtime"].get("detector", {})) if pose_backend.needs_detector else None
    records = read_jsonl(args.manifest)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir = Path(args.output_manifest).parent.resolve()
    default_num_keypoints = int(getattr(pose_backend, "num_keypoints", 17))

    processed = []
    grouped_records = defaultdict(list)
    for item in records:
        grouped_records[resolve_path(item["video_path"], args.manifest)].append(item)

    for video_path, items in tqdm(grouped_records.items(), desc="videos"):
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(video_path)
        next_frame_index = -1
        sorted_items = sorted(items, key=lambda item: (int(item["frame_start"]), int(item["frame_end"]), str(item["sample_id"])))
        try:
            for item in sorted_items:
                npz_path = out_dir / f"{item['sample_id']}.npz"
                rel_npz_path = os.path.relpath(npz_path, manifest_dir)
                if args.skip_existing and npz_path.exists():
                    new_item = dict(item)
                    new_item["keypoints_path"] = rel_npz_path
                    processed.append(new_item)
                    continue

                frame_start = int(item["frame_start"])
                frame_end = int(item["frame_end"])
                if frame_end < frame_start:
                    raise ValueError(f"{item['sample_id']} 的 frame_end 小于 frame_start")

                keypoints = []
                if next_frame_index != frame_start:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)
                    next_frame_index = frame_start

                for frame_idx in range(frame_start, frame_end + 1):
                    if next_frame_index != frame_idx:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                        next_frame_index = frame_idx

                    ok, frame = cap.read()
                    if not ok:
                        break
                    next_frame_index = frame_idx + 1

                    candidates = resolve_pose_candidates(frame, pose_backend, detector)
                    detections = pose_backend.estimate(frame, frame_idx, candidates=candidates)
                    if detections:
                        best = max(detections, key=lambda det: det.get("score", 0.0))
                        kpts = np.asarray(best["keypoints"], dtype=np.float32)
                        scores = np.asarray(best.get("keypoint_scores", np.ones(len(kpts))), dtype=np.float32)
                        if kpts.shape[-1] == 2:
                            kpts = np.concatenate([kpts, scores[:, None]], axis=-1)
                        default_num_keypoints = int(kpts.shape[0])
                    else:
                        kpts = np.zeros((default_num_keypoints, 3), dtype=np.float32)
                    keypoints.append(kpts)

                keypoints = np.stack(keypoints, axis=0) if keypoints else np.zeros((0, default_num_keypoints, 3), dtype=np.float32)
                np.savez_compressed(npz_path, keypoints=keypoints)
                new_item = dict(item)
                new_item["keypoints_path"] = rel_npz_path
                processed.append(new_item)
        finally:
            cap.release()

    write_jsonl(processed, args.output_manifest)
    print({"processed": len(processed), "output_manifest": args.output_manifest})


if __name__ == "__main__":
    main()
