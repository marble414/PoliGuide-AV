#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

from tpgr.data.labels import CANONICAL_GESTURES, GESTURE_TO_INDEX
from tpgr.data.manifests import write_jsonl

GESTURES = [
    "STOP",
    "GO_STRAIGHT",
    "TURN_LEFT",
    "LEFT_TURN_WAIT",
    "TURN_RIGHT",
    "SLOW_DOWN",
    "PULL_OVER",
    "NO_GESTURE",
]


def base_skeleton(cx: float = 320.0, cy: float = 220.0, scale: float = 140.0) -> np.ndarray:
    pts = np.array([
        [0.0, -0.95],
        [-0.08, -1.00], [0.08, -1.00],
        [-0.16, -0.92], [0.16, -0.92],
        [-0.20, -0.55], [0.20, -0.55],
        [-0.36, -0.20], [0.36, -0.20],
        [-0.48,  0.10], [0.48,  0.10],
        [-0.12,  0.05], [0.12,  0.05],
        [-0.14,  0.55], [0.14,  0.55],
        [-0.16,  1.00], [0.16,  1.00],
    ], dtype=np.float32)
    pts *= scale
    pts[:, 0] += cx
    pts[:, 1] += cy
    scores = np.ones((17, 1), dtype=np.float32)
    return np.concatenate([pts, scores], axis=-1)


def set_arm_pose(kpts: np.ndarray, left_target: Tuple[float, float], right_target: Tuple[float, float]) -> np.ndarray:
    kpts = kpts.copy()
    ls, rs = kpts[5, :2], kpts[6, :2]
    kpts[7, :2] = ls + 0.5 * np.array(left_target)
    kpts[9, :2] = ls + np.array(left_target)
    kpts[8, :2] = rs + 0.5 * np.array(right_target)
    kpts[10, :2] = rs + np.array(right_target)
    return kpts


def gesture_pose(gesture: str, t: int, T: int, base: np.ndarray) -> np.ndarray:
    phase = math.sin(2 * math.pi * t / max(T, 1))
    spread = 70 + 8 * phase
    up = -90 + 10 * phase
    down = 85 + 10 * phase
    diag = 65 + 8 * phase

    if gesture == "STOP":
        return set_arm_pose(base, (-spread, 0), (spread, 0))
    if gesture == "GO_STRAIGHT":
        return set_arm_pose(base, (-20, up), (20, up))
    if gesture == "TURN_LEFT":
        return set_arm_pose(base, (-spread, 0), (25, down))
    if gesture == "LEFT_TURN_WAIT":
        return set_arm_pose(base, (-spread, 0), (15, up))
    if gesture == "TURN_RIGHT":
        return set_arm_pose(base, (-25, down), (spread, 0))
    if gesture == "SLOW_DOWN":
        return set_arm_pose(base, (-35, down), (35, down))
    if gesture == "PULL_OVER":
        return set_arm_pose(base, (-diag, diag * 0.25), (diag, diag * 0.35))
    return base


def jitter(kpts: np.ndarray, sigma: float = 3.0) -> np.ndarray:
    arr = kpts.copy()
    arr[:, :2] += np.random.normal(scale=sigma, size=arr[:, :2].shape)
    return arr


def render_skeleton_video(frames: List[np.ndarray], video_path: Path, pose_jsonl_path: Path, fps: int = 15) -> None:
    h, w = 480, 640
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    pose_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with pose_jsonl_path.open("w", encoding="utf-8") as f:
        for i, kpts in enumerate(frames):
            canvas = np.full((h, w, 3), 30, dtype=np.uint8)
            cv2.rectangle(canvas, (210, 40), (430, 440), (60, 60, 60), 2)
            bbox = [
                float(np.min(kpts[:, 0]) - 25),
                float(np.min(kpts[:, 1]) - 25),
                float(np.max(kpts[:, 0]) + 25),
                float(np.max(kpts[:, 1]) + 25),
            ]
            edges = [
                (0, 1), (0, 2), (1, 3), (2, 4),
                (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
                (5, 11), (6, 12), (11, 12),
                (11, 13), (13, 15), (12, 14), (14, 16),
            ]
            for a, b in edges:
                p1 = tuple(kpts[a, :2].astype(int))
                p2 = tuple(kpts[b, :2].astype(int))
                cv2.line(canvas, p1, p2, (0, 255, 255), 3)
            for p in kpts:
                cv2.circle(canvas, tuple(p[:2].astype(int)), 4, (255, 255, 255), -1)
            writer.write(canvas)
            f.write(json.dumps({
                "frame_index": i,
                "detections": [{
                    "bbox": bbox,
                    "score": 0.99,
                    "keypoints": kpts[:, :2].tolist(),
                    "keypoint_scores": kpts[:, 2].tolist(),
                    "occluded": False,
                }]
            }, ensure_ascii=False) + "\n")
    writer.release()


def create_sequence(gesture: str, clip_len: int = 48) -> np.ndarray:
    base = base_skeleton(
        cx=320.0 + random.uniform(-25, 25),
        cy=220.0 + random.uniform(-10, 10),
        scale=140.0 + random.uniform(-8, 8),
    )
    frames = []
    for t in range(clip_len):
        kpts = gesture_pose(gesture, t, clip_len, base)
        kpts = jitter(kpts, sigma=2.0)
        frames.append(kpts)
    return np.stack(frames, axis=0).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="生成可快速跑通的 toy 数据集与演示视频")
    parser.add_argument("--output-dir", default="data/processed/toy")
    parser.add_argument("--num-train", type=int, default=14)
    parser.add_argument("--num-val", type=int, default=4)
    parser.add_argument("--num-test", type=int, default=4)
    parser.add_argument("--clip-len", type=int, default=48)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    seq_dir = out_dir / "sequences"
    demo_dir = out_dir / "demo"
    seq_dir.mkdir(parents=True, exist_ok=True)
    demo_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for split, count in [("train", args.num_train), ("val", args.num_val), ("test", args.num_test)]:
        for gesture in GESTURES:
            for idx in range(count):
                seq = create_sequence(gesture, clip_len=args.clip_len)
                sample_id = f"{split}_{gesture}_{idx:03d}"
                npz_path = seq_dir / f"{sample_id}.npz"
                np.savez_compressed(npz_path, keypoints=seq)
                records.append({
                    "sample_id": sample_id,
                    "keypoints_path": str(npz_path.relative_to(out_dir)),
                    "label": gesture,
                    "split": split,
                    "attributes": {
                        "lighting": random.choice(["day", "night"]),
                        "weather": random.choice(["clear", "rain_aug"]),
                        "occlusion": random.choice(["none", "light"]),
                        "distance": random.choice(["near", "mid"]),
                        "people": "single",
                    },
                })

    manifest_path = out_dir / "manifest.jsonl"
    write_jsonl(records, manifest_path)

    # 生成 demo 视频：STOP -> LEFT_TURN_WAIT -> GO_STRAIGHT
    demo_frames = []
    gt_commands = []
    timeline = [("STOP", 30), ("LEFT_TURN_WAIT", 30), ("GO_STRAIGHT", 30)]
    for gesture, length in timeline:
        for t in range(length):
            kpts = gesture_pose(gesture, t, length, base_skeleton())
            kpts = jitter(kpts, sigma=1.5)
            demo_frames.append(kpts)
            gt_commands.append(gesture if gesture != "NO_GESTURE" else "NO_COMMAND")

    video_path = demo_dir / "toy_demo.mp4"
    pose_jsonl_path = demo_dir / "toy_demo_detections.jsonl"
    render_skeleton_video(demo_frames, video_path, pose_jsonl_path, fps=15)

    gt_jsonl = demo_dir / "toy_demo_gt.jsonl"
    with gt_jsonl.open("w", encoding="utf-8") as f:
        for idx, cmd in enumerate(gt_commands):
            f.write(json.dumps({"frame_index": idx, "gt_command": cmd}, ensure_ascii=False) + "\n")

    print({
        "manifest": str(manifest_path),
        "demo_video": str(video_path),
        "demo_detections": str(pose_jsonl_path),
        "classes": GESTURES,
        "num_samples": len(records),
    })


if __name__ == "__main__":
    main()
