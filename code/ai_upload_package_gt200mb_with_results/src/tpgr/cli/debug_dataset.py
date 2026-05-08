from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from tpgr.cli.common import configure_runtime, save_json
from tpgr.config import load_config
from tpgr.data.labels import get_label_space_classes, get_label_to_index, normalize_raw_label
from tpgr.data.manifests import filter_by_split, read_jsonl, resolve_path
from tpgr.io.json_io import JsonlWriter
from tpgr.io.video_io import get_video_writer
from tpgr.io.visualization import draw_detections, draw_result_panel
from tpgr.metrics.classification import classification_report_dict, save_confusion
from tpgr.pipeline.system import TrafficPoliceGestureSystem


def iter_video_segment(video_path: str | Path, frame_start: int, frame_end: int):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = float(fps) if fps and fps > 1e-3 else 25.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_start))
    try:
        for frame_index in range(int(frame_start), int(frame_end) + 1):
            ok, frame = cap.read()
            if not ok:
                break
            yield frame_index, frame, fps
    finally:
        cap.release()


def aggregate_scores(score_sum: np.ndarray, class_names: list[str]) -> tuple[str, float]:
    if float(score_sum.sum()) <= 0.0:
        return class_names[0], 0.0
    probs = score_sum / max(float(score_sum.sum()), 1e-6)
    pred_idx = int(np.argmax(probs))
    return class_names[pred_idx], float(probs[pred_idx])


def main() -> None:
    parser = argparse.ArgumentParser(description="在整个数据集 split 上进行弹窗式可视化 debug")
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-video", default="outputs/test_debug.mp4")
    parser.add_argument("--output-jsonl", default="outputs/test_debug.jsonl")
    parser.add_argument("--output-summary", default="outputs/test_debug_summary.json")
    parser.add_argument("--debug-topk", type=int, default=5)
    parser.add_argument("--debug-panel-width", type=int, default=440)
    parser.add_argument("--limit", type=int, default=0, help="仅处理前 N 个 clip，0 表示全部")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--sample-id", default=None)
    parser.add_argument("--wait-ms", type=int, default=1)
    parser.add_argument("--window-name", default="tpgr-test-debug")
    parser.add_argument("--no-display", dest="display", action="store_false")
    parser.set_defaults(display=True)
    args = parser.parse_args()

    configure_runtime(1)
    cfg = load_config(args.config)
    dataset_cfg = cfg.get("dataset", {})
    manifest_path = args.manifest or dataset_cfg.get("manifest_path")
    if manifest_path is None:
        raise ValueError("请在 config.dataset.manifest_path 中配置清单，或通过 --manifest 指定")
    split_key = f"{args.split}_split"
    split = dataset_cfg.get(split_key, args.split)
    dataset_name = args.dataset_name or dataset_cfg.get("dataset_name")
    label_space = str(dataset_cfg.get("label_space", cfg.get("classifier", {}).get("label_space", "canonical")))
    class_names = list(get_label_space_classes(label_space))
    label_to_index = get_label_to_index(label_space)

    items = filter_by_split(read_jsonl(manifest_path), split=split)
    if args.sample_id:
        items = [item for item in items if item.get("sample_id") == args.sample_id]
    if args.start_index > 0:
        items = items[args.start_index:]
    if args.limit > 0:
        items = items[:args.limit]
    if not items:
        raise ValueError("没有匹配到任何样本")

    system = TrafficPoliceGestureSystem(cfg)
    writer = None
    paused = False
    stop_all = False

    all_true: list[int] = []
    all_pred: list[int] = []
    per_clip = []
    frame_matches = 0
    frame_total = 0

    if args.display:
        cv2.namedWindow(args.window_name, cv2.WINDOW_NORMAL)

    with JsonlWriter(args.output_jsonl) as jsonl_writer:
        for clip_index, sample in enumerate(tqdm(items, desc="debug_dataset"), start=1):
            system.reset_runtime_state()
            gt_label = normalize_raw_label(
                sample["label"],
                dataset_name=dataset_name,
                label_space=label_space,
                direction=sample.get("direction") or (sample.get("attributes") or {}).get("direction"),
            )
            gt_idx = label_to_index[gt_label]
            video_path = resolve_path(sample["video_path"], manifest_path)
            frame_start = int(sample["frame_start"])
            frame_end = int(sample["frame_end"])
            clip_score_sum = np.zeros((len(class_names),), dtype=np.float32)
            clip_frames = 0
            skip_clip = False

            for source_frame_index, frame, fps in iter_video_segment(video_path, frame_start, frame_end):
                result, detections = system.process_frame(frame, fps=fps)
                clip_frames += 1

                raw_prediction = (result.debug or {}).get("raw_prediction") or {}
                raw_scores = raw_prediction.get("scores") or {}
                if raw_scores:
                    for label, score in raw_scores.items():
                        if label in label_to_index:
                            clip_score_sum[label_to_index[label]] += float(score)
                else:
                    clip_score_sum[label_to_index.get(result.gesture, 0)] += max(float(result.gesture_confidence), 1e-3)

                frame_total += 1
                if result.gesture == gt_label:
                    frame_matches += 1

                provisional_pred, provisional_conf = aggregate_scores(clip_score_sum, class_names)
                running_correct = sum(int(t == p) for t, p in zip(all_true, all_pred)) + int(provisional_pred == gt_label)
                running_total = len(all_true) + 1
                running_accuracy = running_correct / max(running_total, 1)

                eval_debug = {
                    "clip_index": clip_index,
                    "num_clips": len(items),
                    "sample_id": sample.get("sample_id", f"{clip_index:06d}"),
                    "ground_truth": gt_label,
                    "frame_prediction": result.gesture,
                    "clip_prediction": provisional_pred,
                    "clip_correct": provisional_pred == gt_label,
                    "clip_frames": clip_frames,
                    "running_accuracy": float(running_accuracy),
                    "frame_accuracy": float(frame_matches / max(frame_total, 1)),
                    "source_frame_index": int(source_frame_index),
                }
                result.debug = {**(result.debug or {}), "evaluation": eval_debug}

                vis = frame.copy()
                vis = draw_detections(vis, detections, result.active_track_id, debug=True)
                vis = draw_result_panel(
                    vis,
                    result,
                    debug=True,
                    topk=args.debug_topk,
                    panel_width=args.debug_panel_width,
                )

                if writer is None:
                    writer = get_video_writer(args.output_video, fps, (vis.shape[1], vis.shape[0]))
                writer.write(vis)

                jsonl_writer.write({
                    "sample_id": sample.get("sample_id"),
                    "source_frame_index": int(source_frame_index),
                    "ground_truth": gt_label,
                    "provisional_clip_prediction": provisional_pred,
                    "running_accuracy": float(running_accuracy),
                    **result.to_dict(),
                    "detections": detections,
                })

                if args.display:
                    cv2.imshow(args.window_name, vis)
                    wait_ms = 0 if paused else max(1, int(args.wait_ms))
                    key = cv2.waitKey(wait_ms) & 0xFF
                    if key == 27:
                        stop_all = True
                        break
                    if key == ord(" "):
                        paused = not paused
                    if key == ord("n"):
                        skip_clip = True
                        break

            clip_pred, clip_conf = aggregate_scores(clip_score_sum, class_names)
            all_true.append(gt_idx)
            all_pred.append(label_to_index[clip_pred])
            per_clip.append({
                "sample_id": sample.get("sample_id"),
                "ground_truth": gt_label,
                "prediction": clip_pred,
                "confidence": float(clip_conf),
                "correct": bool(clip_pred == gt_label),
                "video_path": str(video_path),
                "frame_start": frame_start,
                "frame_end": frame_end,
                "attributes": sample.get("attributes", {}),
            })

            if stop_all:
                break
            if skip_clip:
                continue

    if writer is not None:
        writer.release()
    if args.display:
        cv2.destroyAllWindows()

    metrics = classification_report_dict(all_true, all_pred, class_names=class_names)
    metrics["frame_accuracy"] = float(frame_matches / max(frame_total, 1))
    metrics["num_clips"] = len(all_true)
    metrics["num_frames"] = int(frame_total)
    save_json({
        "metrics": metrics,
        "clips": per_clip,
    }, args.output_summary)
    save_confusion(all_true, all_pred, Path(args.output_summary).with_suffix(""), class_names=class_names)


if __name__ == "__main__":
    main()
