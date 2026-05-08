#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Iterable

import numpy as np

from tpgr.data.labels import normalize_raw_label
from tpgr.data.manifests import write_jsonl


def split_video_stems(stems: list[str], val_ratio: float, test_ratio: float, seed: int) -> tuple[set[str], set[str]]:
    import zlib

    ordered = sorted(stems, key=lambda stem: (zlib.crc32(f"{seed}:{stem}".encode("utf-8")), stem))
    count_test = int(round(len(ordered) * test_ratio))
    count_val = int(round(len(ordered) * val_ratio))
    if len(ordered) >= 5:
        count_test = min(max(count_test, 1), len(ordered) - 2)
        count_val = min(max(count_val, 1), len(ordered) - count_test - 1)
    else:
        count_test = 0
        count_val = 0
    test_stems = set(ordered[:count_test])
    val_stems = set(ordered[count_test:count_test + count_val])
    return val_stems, test_stems


def _load_json_list(path: Path) -> list[int] | list[str]:
    return json.loads(path.read_text(encoding="utf-8"))


def _pick_person(frame_record: dict) -> dict | None:
    if not frame_record:
        return None
    person_id = sorted(frame_record)[0]
    return frame_record[person_id]


def _safe_array(value, shape: tuple[int, ...], dtype=np.float32) -> np.ndarray:
    arr = np.asarray(value, dtype=dtype)
    if arr.size == 0:
        return np.zeros(shape, dtype=dtype)
    return arr.reshape(shape).astype(dtype, copy=False)


def build_video_payload(vibe_records: list[dict]) -> dict[str, np.ndarray]:
    num_frames = len(vibe_records)
    joints_seq = np.zeros((num_frames, 49, 3), dtype=np.float32)
    pose_seq = np.zeros((num_frames, 72), dtype=np.float32)

    for frame_idx, frame_record in enumerate(vibe_records):
        person = _pick_person(frame_record)
        if person is None:
            continue
        joints = _safe_array(person.get("joints3d"), (49, 3))
        pose = _safe_array(person.get("pose"), (72,))
        joints_seq[frame_idx] = joints
        pose_seq[frame_idx] = pose

    centers = joints_seq.mean(axis=1)
    center_rel = centers - centers[:1]
    center_vel = np.vstack([np.zeros((1, 3), dtype=np.float32), np.diff(centers, axis=0)])

    joints_centered = joints_seq - centers[:, None, :]
    scales = np.linalg.norm(joints_centered, axis=-1).mean(axis=1, keepdims=True)
    scales = np.clip(scales, 1e-6, None)
    joints_normalized = joints_centered / scales[:, None, :]

    features = np.concatenate(
        [
            joints_normalized.reshape(num_frames, -1),
            pose_seq,
            center_rel,
            center_vel,
        ],
    axis=-1,
    ).astype(np.float32)
    return {
        "features": features,
        "joints3d_49": joints_seq.astype(np.float32, copy=False),
        "smpl_joints24": joints_seq[:, -24:, :].astype(np.float32, copy=False),
        "pose_axis_angle": pose_seq.reshape(num_frames, 24, 3).astype(np.float32, copy=False),
        "root_orient": pose_seq[:, :3].astype(np.float32, copy=False),
        "center_rel": center_rel.astype(np.float32, copy=False),
        "center_vel": center_vel.astype(np.float32, copy=False),
    }


def _iter_window_end_indices(labels: np.ndarray, split: str, train_stride: int, eval_stride: int, background_stride_multiplier: int) -> Iterable[int]:
    base_stride = train_stride if split == "train" else eval_stride
    background_stride = max(1, base_stride * max(1, background_stride_multiplier))
    gesture_stride = max(1, base_stride)
    for frame_end, raw_label in enumerate(labels):
        stride = background_stride if int(raw_label) == 0 and split == "train" else gesture_stride
        if frame_end % stride == 0:
            yield frame_end


def _window_features(features: np.ndarray, frame_end: int, clip_len: int) -> np.ndarray:
    frame_end = int(frame_end)
    start = max(0, frame_end - clip_len + 1)
    window = features[start:frame_end + 1].astype(np.float32, copy=False)
    if window.shape[0] >= clip_len:
        return window[-clip_len:]
    pad = np.repeat(window[:1], clip_len - window.shape[0], axis=0) if window.shape[0] > 0 else np.zeros((clip_len, features.shape[1]), dtype=np.float32)
    return np.concatenate([pad, window], axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="为 CTPGesture v2 构建 33 类 VIBE 连续训练窗口")
    parser.add_argument("--dataset-root", default="data/raw/police_gesture_v2")
    parser.add_argument("--output-dir", default="data/processed/ctpgesture_v2_vibe33")
    parser.add_argument("--clip-len", type=int, default=64)
    parser.add_argument("--train-stride", type=int, default=4)
    parser.add_argument("--eval-stride", type=int, default=1)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--background-stride-multiplier", type=int, default=4)
    parser.add_argument("--limit-videos", type=int, default=0)
    parser.add_argument("--materialize-windows", action="store_true", help="将每个窗口单独保存为 npz；默认复用逐视频特征文件")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).resolve()
    out_dir = Path(args.output_dir).resolve()
    window_dir = out_dir / "windows"
    video_feature_dir = out_dir / "videos"
    if args.materialize_windows:
        window_dir.mkdir(parents=True, exist_ok=True)
    video_feature_dir.mkdir(parents=True, exist_ok=True)

    video_dir = dataset_root / "video"
    label_dir = dataset_root / "label_combine_frame"
    direction_dir = dataset_root / "label_ori_frame"
    vibe_dir = dataset_root / "vibe"

    stems = [path.stem for path in sorted(video_dir.glob("*")) if path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".m4v"}]
    if args.limit_videos > 0:
        stems = stems[: int(args.limit_videos)]
    val_stems, test_stems = split_video_stems(stems, args.val_ratio, args.test_ratio, args.seed)

    window_records = []
    video_records = []
    total_frames = 0

    for stem in stems:
        split = "test" if stem in test_stems else ("val" if stem in val_stems else "train")
        labels = np.asarray(_load_json_list(label_dir / f"{stem}.json"), dtype=np.int64)
        directions = np.asarray(_load_json_list(direction_dir / f"{stem}.json"), dtype=object)
        with (vibe_dir / f"{stem}.pkl").open("rb") as f:
            vibe_records = pickle.load(f)
        video_payload = build_video_payload(vibe_records)
        features = video_payload["features"]
        usable_len = min(len(features), len(labels), len(directions))
        if usable_len <= 0:
            raise ValueError(f"{stem} 没有可用帧")
        if not (len(features) == len(labels) == len(directions)):
            print({
                "warning": "length_mismatch_truncated",
                "stem": stem,
                "features": len(features),
                "labels": len(labels),
                "directions": len(directions),
                "usable_len": usable_len,
            })
        features = features[:usable_len]
        labels = labels[:usable_len]
        directions = directions[:usable_len]

        total_frames += int(len(labels))
        video_path = next(video_dir.glob(f"{stem}.*"))
        video_feature_path = video_feature_dir / f"{stem}.npz"
        if not (args.skip_existing and video_feature_path.exists()):
            np.savez_compressed(
                video_feature_path,
                **video_payload,
                raw_labels=labels,
                directions=directions.astype(str),
                video_path=str(video_path),
                split=split,
                stem=stem,
            )
        video_records.append({
            "sample_id": stem,
            "video_path": str(video_path),
            "feature_path": str(video_feature_path.relative_to(out_dir)),
            "split": split,
            "attributes": {
                "source_video": stem,
                "num_frames": int(len(labels)),
                "label_space": "ctpv2_directional",
            },
        })

        for frame_end in _iter_window_end_indices(
            labels=labels,
            split=split,
            train_stride=int(args.train_stride),
            eval_stride=int(args.eval_stride),
            background_stride_multiplier=int(args.background_stride_multiplier),
        ):
            raw_label = int(labels[frame_end])
            direction = str(directions[frame_end])
            label_name = normalize_raw_label(
                raw_label,
                dataset_name="ctpgesture_v2",
                label_space="ctpv2_directional",
                direction=direction,
            )
            sample_id = f"{stem}_{frame_end:06d}"
            frame_start = max(0, int(frame_end) - int(args.clip_len) + 1)
            if args.materialize_windows:
                feature_path = window_dir / f"{sample_id}.npz"
                if not (args.skip_existing and feature_path.exists()):
                    np.savez_compressed(feature_path, features=_window_features(features, frame_end, int(args.clip_len)))
                relative_feature_path = str(feature_path.relative_to(out_dir))
            else:
                relative_feature_path = str(video_feature_path.relative_to(out_dir))
            window_records.append({
                "sample_id": sample_id,
                "video_path": str(video_path),
                "frame_start": frame_start,
                "frame_end": int(frame_end),
                "label": raw_label,
                "direction": direction,
                "split": split,
                "feature_path": relative_feature_path,
                "attributes": {
                    "source_video": stem,
                    "raw_label": raw_label,
                    "direction": direction,
                    "label_name": label_name,
                    "label_space": "ctpv2_directional",
                    "window_type": "causal",
                    "feature_source": "window_npz" if args.materialize_windows else "video_npz",
                },
            })

    manifest_path = out_dir / "manifest_windows.jsonl"
    video_manifest_path = out_dir / "manifest_videos.jsonl"
    write_jsonl(window_records, manifest_path)
    write_jsonl(video_records, video_manifest_path)

    print({
        "output_dir": str(out_dir),
        "window_manifest": str(manifest_path),
        "video_manifest": str(video_manifest_path),
        "num_videos": len(video_records),
        "num_windows": len(window_records),
        "total_frames": total_frames,
        "feature_dim": 225,
        "clip_len": int(args.clip_len),
    })


if __name__ == "__main__":
    main()
