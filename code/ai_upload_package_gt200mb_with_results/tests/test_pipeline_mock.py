import json
from pathlib import Path

import cv2
import numpy as np

from tpgr.pipeline.classifier import BaseClassifier, Prediction
from tpgr.pipeline.system import TrafficPoliceGestureSystem


def test_pipeline_with_precomputed_backend(tmp_path: Path):
    video_path = tmp_path / "demo.mp4"
    jsonl_path = tmp_path / "demo.jsonl"

    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (640, 480))
    for i in range(15):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()

    with jsonl_path.open("w", encoding="utf-8") as f:
        for i in range(15):
            kpts = np.zeros((17, 3), dtype=float)
            kpts[:, 0] = 320
            kpts[:, 1] = 240
            kpts[:, 2] = 1.0
            f.write(json.dumps({
                "frame_index": i,
                "detections": [{
                    "bbox": [250, 100, 390, 420],
                    "score": 0.99,
                    "keypoints": kpts[:, :2].tolist(),
                    "keypoint_scores": kpts[:, 2].tolist(),
                    "occluded": False,
                }]
            }) + "\n")

    cfg = {
        "runtime": {
            "sequence_len": 32,
            "min_predict_len": 4,
            "pose_backend": {"type": "precomputed", "jsonl_path": str(jsonl_path)},
            "tracker": {"iou_threshold": 0.3, "max_age": 5, "min_hits": 1},
            "selector": {},
            "state_machine": {"window_size": 5, "min_consensus": 3, "conf_threshold": 0.5, "hold_frames": 2, "expiry_frames": 6},
        },
        "classifier": {"type": "rule_based"},
    }

    system = TrafficPoliceGestureSystem(cfg)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = None
    for i in range(15):
        result, detections = system.process_frame(frame, fps=10.0)
    assert result is not None
    assert result.active_track_id == 1


class _SequenceClassifier(BaseClassifier):
    def __init__(self) -> None:
        self.calls = 0
        self.class_names = ["NO_GESTURE", "STOP"]
        self.label_space = "canonical"

    def predict_sequence(self, keypoints: np.ndarray, bbox_track: np.ndarray | None = None) -> Prediction:
        self.calls += 1
        if self.calls <= 2:
            scores = {"NO_GESTURE": 0.1, "STOP": 0.9}
            label = "STOP"
        else:
            scores = {"NO_GESTURE": 0.9, "STOP": 0.1}
            label = "NO_GESTURE"
        return Prediction(label=label, confidence=scores[label], scores=scores)


class _SequenceMultitaskClassifier(BaseClassifier):
    def __init__(self) -> None:
        self.calls = 0
        self.class_names = ["NO_GESTURE", "STOP"]
        self.label_space = "canonical"
        self.command_index_to_label = {0: "NO_COMMAND", 1: "STOP"}
        self.command_label_to_index = {"NO_COMMAND": 0, "STOP": 1}
        self.direction_index_to_label = {0: "UNKNOWN", 1: "FRONT"}
        self.direction_label_to_index = {"UNKNOWN": 0, "FRONT": 1}
        self.task_fusion = {"command_weight": 0.0, "direction_weight": 0.0}

    def predict_sequence(self, keypoints: np.ndarray, bbox_track: np.ndarray | None = None) -> Prediction:
        self.calls += 1
        if self.calls <= 2:
            scores = {"NO_GESTURE": 0.1, "STOP": 0.9}
            gesture_scores_raw = {"NO_GESTURE": 0.2, "STOP": 0.8}
            command_scores = {"NO_COMMAND": 0.1, "STOP": 0.9}
            label = "STOP"
        else:
            scores = {"NO_GESTURE": 0.9, "STOP": 0.1}
            gesture_scores_raw = {"NO_GESTURE": 0.85, "STOP": 0.15}
            command_scores = {"NO_COMMAND": 0.85, "STOP": 0.15}
            label = "NO_GESTURE"
        return Prediction(
            label=label,
            confidence=scores[label],
            scores=scores,
            extras={
                "gesture_scores_raw": gesture_scores_raw,
                "command_scores": command_scores,
                "direction_scores": {"UNKNOWN": 0.2, "FRONT": 0.8},
                "command_label": max(command_scores, key=command_scores.get),
                "direction_label": "FRONT",
            },
        )


def test_prediction_smoothing_keeps_short_track_consistent(tmp_path: Path):
    jsonl_path = tmp_path / "demo.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for i in range(4):
            kpts = np.zeros((17, 3), dtype=float)
            kpts[:, 0] = 320
            kpts[:, 1] = 240
            kpts[:, 2] = 1.0
            f.write(json.dumps({
                "frame_index": i,
                "detections": [{
                    "bbox": [250, 100, 390, 420],
                    "score": 0.99,
                    "keypoints": kpts[:, :2].tolist(),
                    "keypoint_scores": kpts[:, 2].tolist(),
                    "occluded": False,
                }]
            }) + "\n")

    cfg = {
        "runtime": {
            "sequence_len": 8,
            "min_predict_len": 1,
            "pose_backend": {"type": "precomputed", "jsonl_path": str(jsonl_path)},
            "tracker": {"iou_threshold": 0.3, "max_age": 5, "min_hits": 1},
            "selector": {},
            "prediction_smoothing": {"window_size": 3},
            "state_machine": {"window_size": 3, "min_consensus": 1, "conf_threshold": 0.0, "hold_frames": 1, "expiry_frames": 2},
        },
        "classifier": {"type": "rule_based"},
    }

    system = TrafficPoliceGestureSystem(cfg)
    system.classifier = _SequenceClassifier()
    system.class_names = list(system.classifier.class_names)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    result1, _ = system.process_frame(frame, fps=10.0)
    result2, _ = system.process_frame(frame, fps=10.0)
    result3, _ = system.process_frame(frame, fps=10.0)

    assert result1.gesture == "STOP"
    assert result2.gesture == "STOP"
    assert result3.debug["raw_prediction"]["label"] == "NO_GESTURE"
    assert result3.debug["raw_prediction"]["label_smoothed"] == "STOP"
    assert result3.gesture == "STOP"


def test_task_prediction_smoothing_uses_multitask_raw_scores(tmp_path: Path):
    jsonl_path = tmp_path / "demo_task.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for i in range(4):
            kpts = np.zeros((17, 3), dtype=float)
            kpts[:, 0] = 320
            kpts[:, 1] = 240
            kpts[:, 2] = 1.0
            f.write(json.dumps({
                "frame_index": i,
                "detections": [{
                    "bbox": [250, 100, 390, 420],
                    "score": 0.99,
                    "keypoints": kpts[:, :2].tolist(),
                    "keypoint_scores": kpts[:, 2].tolist(),
                    "occluded": False,
                }]
            }) + "\n")

    cfg = {
        "runtime": {
            "sequence_len": 8,
            "min_predict_len": 1,
            "pose_backend": {"type": "precomputed", "jsonl_path": str(jsonl_path)},
            "tracker": {"iou_threshold": 0.3, "max_age": 5, "min_hits": 1},
            "selector": {},
            "prediction_smoothing": {"gesture_window": 3, "command_window": 3, "direction_window": 3},
            "state_machine": {"window_size": 3, "min_consensus": 1, "conf_threshold": 0.0, "hold_frames": 1, "expiry_frames": 2},
        },
        "classifier": {"type": "rule_based"},
    }

    system = TrafficPoliceGestureSystem(cfg)
    system.classifier = _SequenceMultitaskClassifier()
    system.class_names = list(system.classifier.class_names)
    system.command_index_to_label = dict(system.classifier.command_index_to_label)
    system.command_label_to_index = dict(system.classifier.command_label_to_index)
    system.direction_index_to_label = dict(system.classifier.direction_index_to_label)
    system.direction_label_to_index = dict(system.classifier.direction_label_to_index)
    system.command_labels = list(system.classifier.command_index_to_label.values())
    system.direction_labels = list(system.classifier.direction_index_to_label.values())
    system.command_weight = 0.0
    system.direction_weight = 0.0
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    _, _ = system.process_frame(frame, fps=10.0)
    _, _ = system.process_frame(frame, fps=10.0)
    result3, _ = system.process_frame(frame, fps=10.0)

    assert result3.debug["raw_prediction"]["label"] == "NO_GESTURE"
    assert result3.debug["raw_prediction"]["label_smoothed"] == "STOP"
    assert "gesture_scores_raw_smoothed" in result3.debug["raw_prediction"]
    assert result3.gesture == "STOP"
