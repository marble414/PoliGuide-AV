#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import zlib
from pathlib import Path

import pandas as pd

from tpgr.data.labels import normalize_raw_label
from tpgr.data.manifests import write_jsonl


def parse_label(value, label_space: str = "canonical", direction: str | None = None):
    try:
        return normalize_raw_label(
            int(value),
            dataset_name="ctpgesture_v2",
            label_space=label_space,
            direction=direction,
        )
    except Exception:
        return normalize_raw_label(
            str(value),
            dataset_name="ctpgesture_v2",
            label_space=label_space,
            direction=direction,
        )


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


def split_video_stems(stems: list[str], val_ratio: float, test_ratio: float, seed: int) -> tuple[set[str], set[str]]:
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


def dominant_direction(tokens: list[str]) -> str:
    values = [str(token) for token in tokens if str(token).strip()]
    if not values:
        return "unknown"
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return max(sorted(counts), key=lambda key: counts[key])


def build_manifest_from_dataset_root(
    dataset_root: Path,
    output_manifest: str,
    label_space: str,
    val_ratio: float,
    test_ratio: float,
    background_len: int,
    background_stride: int,
    seed: int,
) -> None:
    dataset_root = dataset_root.resolve()
    videos_dir = dataset_root / "video"
    labels_dir = dataset_root / "label_combine_frame"
    directions_dir = dataset_root / "label_ori_frame"
    stems = [path.stem for path in sorted(videos_dir.glob("*")) if path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".m4v"}]
    val_stems, test_stems = split_video_stems(stems, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed)

    records = []
    gesture_count = 0
    background_count = 0
    for stem in stems:
        video_candidates = list(videos_dir.glob(f"{stem}.*"))
        if not video_candidates:
            raise FileNotFoundError(f"未找到 {stem} 对应的视频文件")
        video_path = video_candidates[0]
        label_path = labels_dir / f"{stem}.json"
        direction_path = directions_dir / f"{stem}.json"
        labels = [int(value) for value in json.loads(label_path.read_text(encoding="utf-8"))]
        directions = [str(value) for value in json.loads(direction_path.read_text(encoding="utf-8"))] if direction_path.exists() else ["unknown"] * len(labels)

        split = "test" if stem in test_stems else ("val" if stem in val_stems else "train")
        for frame_start, frame_end, raw_label in iter_runs(labels):
            direction = dominant_direction(directions[frame_start:frame_end + 1])
            if raw_label == 0:
                window_start = frame_start
                while window_start + background_len - 1 <= frame_end:
                    sample_id = f"{stem}_{window_start:06d}_{window_start + background_len - 1:06d}"
                    records.append({
                        "sample_id": sample_id,
                        "video_path": str(video_path),
                        "frame_start": int(window_start),
                        "frame_end": int(window_start + background_len - 1),
                        "label": parse_label(0, label_space=label_space, direction=direction),
                        "direction": direction,
                        "split": split,
                        "attributes": {
                            "direction": direction,
                            "source_video": stem,
                            "segment_type": "background",
                            "raw_label": 0,
                        },
                    })
                    background_count += 1
                    window_start += background_stride
                continue

            label = parse_label(raw_label, label_space=label_space, direction=direction)
            records.append({
                "sample_id": f"{stem}_{frame_start:06d}_{frame_end:06d}",
                "video_path": str(video_path),
                "frame_start": int(frame_start),
                "frame_end": int(frame_end),
                "label": label,
                "direction": direction,
                "split": split,
                "attributes": {
                    "direction": direction,
                    "source_video": stem,
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
    parser = argparse.ArgumentParser(description="整理 CTPGesture v2 manifest")
    parser.add_argument("--dataset-root", help="原始解压目录，如 data/raw/police_gesture_v2")
    parser.add_argument("--labels-csv", help="legacy 模式下的 CSV 标签文件")
    parser.add_argument("--videos-dir")
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--label-space", default="canonical", help="canonical(9类) 或 ctpv2_directional(33类)")
    parser.add_argument("--video-column", default="video")
    parser.add_argument("--start-column", default="frame_start")
    parser.add_argument("--end-column", default="frame_end")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--direction-column", default="direction")
    parser.add_argument("--split-column", default="split")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--background-len", type=int, default=48)
    parser.add_argument("--background-stride", type=int, default=240)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.dataset_root:
        build_manifest_from_dataset_root(
            dataset_root=Path(args.dataset_root),
            output_manifest=args.output_manifest,
            label_space=str(args.label_space),
            val_ratio=float(args.val_ratio),
            test_ratio=float(args.test_ratio),
            background_len=int(args.background_len),
            background_stride=int(args.background_stride),
            seed=int(args.seed),
        )
        return

    if not args.labels_csv or not args.videos_dir:
        raise ValueError("legacy 模式需要同时提供 --labels-csv 和 --videos-dir")

    videos_dir = Path(args.videos_dir)
    df = pd.read_csv(args.labels_csv)

    records = []
    for idx, row in df.iterrows():
        video_stem = Path(str(row[args.video_column])).stem
        candidates = list(videos_dir.glob(f"{video_stem}.*"))
        if not candidates:
            raise FileNotFoundError(f"无法在 {videos_dir} 中找到视频 {video_stem}")
        video_path = candidates[0]
        direction = str(row.get(args.direction_column, "unknown"))
        label = parse_label(
            row[args.label_column],
            label_space=str(args.label_space),
            direction=direction,
        )
        record = {
            "sample_id": f"{video_stem}_{idx:06d}",
            "video_path": str(video_path),
            "frame_start": int(row[args.start_column]),
            "frame_end": int(row[args.end_column]),
            "label": label,
            "direction": direction,
            "split": str(row.get(args.split_column, "train")),
            "attributes": {"direction": direction},
        }
        for key in ["lighting", "weather", "occlusion", "distance", "people"]:
            if key in row:
                record["attributes"][key] = str(row[key])
        records.append(record)

    write_jsonl(records, args.output_manifest)
    print({"samples": len(records), "output_manifest": args.output_manifest})


if __name__ == "__main__":
    main()
