#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import zlib
from pathlib import Path

import pandas as pd

from tpgr.data.labels import normalize_raw_label
from tpgr.data.manifests import write_jsonl


def find_video(videos_dir: Path, stem: str) -> Path:
    for ext in [".mp4", ".avi", ".mov", ".mkv"]:
        path = videos_dir / f"{stem}{ext}"
        if path.exists():
            return path
    matches = list(videos_dir.glob(f"{stem}.*"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"未在 {videos_dir} 中找到与 {stem} 对应的视频文件")


def load_framewise_labels(path: Path) -> list[int]:
    raw = path.read_text(encoding="utf-8").strip().strip(",")
    if not raw:
        return []
    return [int(token) for token in raw.split(",") if token]


def iter_runs(labels: list[int]) -> list[tuple[int, int, int]]:
    runs: list[tuple[int, int, int]] = []
    index = 0
    while index < len(labels):
        value = labels[index]
        end = index + 1
        while end < len(labels) and labels[end] == value:
            end += 1
        runs.append((index, end - 1, value))
        index = end
    return runs


def pick_validation_videos(stems: list[str], val_ratio: float, seed: int) -> set[str]:
    if not stems:
        return set()
    ordered = sorted(stems, key=lambda stem: (zlib.crc32(f"{seed}:{stem}".encode("utf-8")), stem))
    count = int(round(len(ordered) * val_ratio))
    if len(ordered) >= 3:
        count = min(max(count, 1), len(ordered) - 1)
    else:
        count = 0
    return set(ordered[:count])


def build_manifest_from_dataset_root(
    dataset_root: Path,
    output_manifest: str,
    val_ratio: float,
    background_len: int,
    background_stride: int,
    seed: int,
) -> None:
    dataset_root = dataset_root.resolve()
    train_videos = sorted((dataset_root / "train").glob("*"))
    train_stems = [path.stem for path in train_videos if path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}]
    val_stems = pick_validation_videos(train_stems, val_ratio=val_ratio, seed=seed)

    records = []
    gesture_count = 0
    background_count = 0
    for split_dir in [dataset_root / "train", dataset_root / "test"]:
        if not split_dir.exists():
            continue
        for video_path in sorted(split_dir.glob("*")):
            if video_path.suffix.lower() not in {".mp4", ".avi", ".mov", ".mkv"}:
                continue
            label_path = video_path.with_suffix(".csv")
            if not label_path.exists():
                raise FileNotFoundError(label_path)
            labels = load_framewise_labels(label_path)
            split = "test" if split_dir.name == "test" else ("val" if video_path.stem in val_stems else "train")

            for frame_start, frame_end, raw_label in iter_runs(labels):
                if raw_label == 0:
                    window_start = frame_start
                    while window_start + background_len - 1 <= frame_end:
                        sample_id = f"{video_path.stem}_{window_start:06d}_{window_start + background_len - 1:06d}"
                        records.append({
                            "sample_id": sample_id,
                            "video_path": str(video_path),
                            "frame_start": int(window_start),
                            "frame_end": int(window_start + background_len - 1),
                            "label": "NO_GESTURE",
                            "split": split,
                            "attributes": {
                                "source_video": video_path.stem,
                                "segment_type": "background",
                                "raw_label": 0,
                            },
                        })
                        background_count += 1
                        window_start += background_stride
                    continue

                records.append({
                    "sample_id": f"{video_path.stem}_{frame_start:06d}_{frame_end:06d}",
                    "video_path": str(video_path),
                    "frame_start": int(frame_start),
                    "frame_end": int(frame_end),
                    "label": normalize_raw_label(raw_label, dataset_name="ctpgesture_v1"),
                    "split": split,
                    "attributes": {
                        "source_video": video_path.stem,
                        "segment_type": "gesture",
                        "raw_label": int(raw_label),
                    },
                })
                gesture_count += 1

    write_jsonl(records, output_manifest)
    print({
        "samples": len(records),
        "gesture_segments": gesture_count,
        "background_segments": background_count,
        "output_manifest": output_manifest,
    })


def main() -> None:
    parser = argparse.ArgumentParser(description="将 CTPGesture v1 标签整理为统一 manifest")
    parser.add_argument("--dataset-root", help="原始解压目录，如 data/raw/police_gesture_v1")
    parser.add_argument("--videos-dir")
    parser.add_argument("--labels-csv", help="包含 video/frame_start/frame_end/label 或等价列")
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--split-column", default="split")
    parser.add_argument("--video-column", default="video")
    parser.add_argument("--start-column", default="frame_start")
    parser.add_argument("--end-column", default="frame_end")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--background-len", type=int, default=48)
    parser.add_argument("--background-stride", type=int, default=240)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.dataset_root:
        build_manifest_from_dataset_root(
            dataset_root=Path(args.dataset_root),
            output_manifest=args.output_manifest,
            val_ratio=float(args.val_ratio),
            background_len=int(args.background_len),
            background_stride=int(args.background_stride),
            seed=int(args.seed),
        )
        return

    if not args.videos_dir or not args.labels_csv:
        raise ValueError("legacy 模式需要同时提供 --videos-dir 和 --labels-csv")

    df = pd.read_csv(args.labels_csv)
    records = []
    for idx, row in df.iterrows():
        video_name = str(row[args.video_column])
        video_path = find_video(Path(args.videos_dir), Path(video_name).stem)
        label = normalize_raw_label(row[args.label_column], dataset_name="ctpgesture_v1")
        record = {
            "sample_id": f"{Path(video_name).stem}_{idx:06d}",
            "video_path": str(video_path),
            "frame_start": int(row[args.start_column]),
            "frame_end": int(row[args.end_column]),
            "label": label,
            "split": str(row.get(args.split_column, "train")),
            "attributes": {},
        }
        for key in ["lighting", "weather", "occlusion", "distance", "people"]:
            if key in row:
                record["attributes"][key] = str(row[key])
        records.append(record)

    write_jsonl(records, args.output_manifest)
    print({"samples": len(records), "output_manifest": args.output_manifest})


if __name__ == "__main__":
    main()
