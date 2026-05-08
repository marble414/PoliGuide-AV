from __future__ import annotations

import time
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np

from tpgr.data.labels import get_label_space_classes, gesture_to_command, gesture_to_direction, gesture_to_intent
from tpgr.data.schema import CommandOutput
from tpgr.pipeline.classifier import BaseClassifier, Prediction, build_classifier
from tpgr.pipeline.detector import HOGPersonDetector
from tpgr.pipeline.intent import IntentParser
from tpgr.pipeline.pose_backends import (
    BasePoseBackend,
    DummyBoxPoseBackend,
    MediaPipePoseBackend,
    OpenMMLabPoseBackend,
    PrecomputedPoseBackend,
)
from tpgr.pipeline.runtime_utils import resolve_pose_candidates
from tpgr.pipeline.selector import OfficerSelector
from tpgr.pipeline.sequence_buffer import SequenceBuffer
from tpgr.pipeline.state_machine import TemporalCommandFilter
from tpgr.pipeline.tracker import GreedyIoUTracker


def build_pose_backend(cfg: dict) -> BasePoseBackend:
    kind = cfg.get("type", "precomputed").lower()
    if kind == "precomputed":
        return PrecomputedPoseBackend(cfg["jsonl_path"])
    if kind == "dummy":
        return DummyBoxPoseBackend()
    if kind == "mediapipe":
        return MediaPipePoseBackend(
            static_image_mode=bool(cfg.get("static_image_mode", False)),
            model_complexity=int(cfg.get("model_complexity", 1)),
        )
    if kind == "openmmlab":
        return OpenMMLabPoseBackend(
            model_alias=cfg.get("model_alias", "human"),
            pose_model=cfg.get("pose_model"),
            device=cfg.get("device"),
            pose2d_weights=cfg.get("pose2d_weights"),
            det_model=cfg.get("det_model"),
            det_weights=cfg.get("det_weights"),
            scope=cfg.get("scope", "mmpose"),
        )
    raise ValueError(f"未知姿态后端类型: {kind}")


class TrafficPoliceGestureSystem:
    def __init__(self, cfg: dict) -> None:
        runtime_cfg = cfg["runtime"]
        tracker_cfg = runtime_cfg.get("tracker", {})
        selector_cfg = runtime_cfg.get("selector", {})
        state_cfg = runtime_cfg.get("state_machine", {})
        classifier_cfg = cfg["classifier"]

        self.pose_backend = build_pose_backend(runtime_cfg["pose_backend"])
        self.detector = HOGPersonDetector(**runtime_cfg.get("detector", {})) if self.pose_backend.needs_detector else None
        self.tracker = GreedyIoUTracker(
            iou_threshold=float(tracker_cfg.get("iou_threshold", 0.3)),
            max_age=int(tracker_cfg.get("max_age", 15)),
            min_hits=int(tracker_cfg.get("min_hits", 1)),
        )
        self.selector = OfficerSelector(**selector_cfg)
        self.buffer = SequenceBuffer(
            max_len=int(runtime_cfg.get("sequence_len", 64)),
            num_keypoints=int(getattr(self.pose_backend, "num_keypoints", 17)),
        )
        self.classifier: BaseClassifier = build_classifier(classifier_cfg)
        self.intent_parser = IntentParser()
        self.state_machine = TemporalCommandFilter(
            window_size=int(state_cfg.get("window_size", 7)),
            min_consensus=int(state_cfg.get("min_consensus", 5)),
            conf_threshold=float(state_cfg.get("conf_threshold", 0.60)),
            hold_frames=int(state_cfg.get("hold_frames", 8)),
            expiry_frames=int(state_cfg.get("expiry_frames", 20)),
        )
        smoothing_cfg = runtime_cfg.get("prediction_smoothing", {}) or {}
        self.prediction_smoothing_window = max(1, int(smoothing_cfg.get("window_size", 1)))
        self.min_predict_len = int(runtime_cfg.get("min_predict_len", 12))
        self.frame_index = 0
        self.last_result: Optional[CommandOutput] = None
        self.class_names = list(
            getattr(
                self.classifier,
                "class_names",
                get_label_space_classes(getattr(self.classifier, "label_space", "canonical")),
            )
        )
        self.command_index_to_label = dict(getattr(self.classifier, "command_index_to_label", {}) or {})
        self.direction_index_to_label = dict(getattr(self.classifier, "direction_index_to_label", {}) or {})
        self.command_label_to_index = dict(getattr(self.classifier, "command_label_to_index", {}) or {})
        self.direction_label_to_index = dict(getattr(self.classifier, "direction_label_to_index", {}) or {})
        self.command_labels = list(self.command_index_to_label.values())
        self.direction_labels = list(self.direction_index_to_label.values())
        task_fusion = dict(getattr(self.classifier, "task_fusion", {}) or {})
        self.command_weight = float(task_fusion.get("command_weight", 0.0))
        self.direction_weight = float(task_fusion.get("direction_weight", 0.0))
        self.gesture_smoothing_window = max(1, int(smoothing_cfg.get("gesture_window", 1)))
        self.command_smoothing_window = max(1, int(smoothing_cfg.get("command_window", 1)))
        self.direction_smoothing_window = max(1, int(smoothing_cfg.get("direction_window", 1)))
        self.task_smoothing_enabled = any(
            window > 1 for window in (
                self.gesture_smoothing_window,
                self.command_smoothing_window,
                self.direction_smoothing_window,
            )
        )
        self._score_history: Dict[int, deque[np.ndarray]] = {}
        self._gesture_history: Dict[int, deque[np.ndarray]] = {}
        self._command_history: Dict[int, deque[np.ndarray]] = {}
        self._direction_history: Dict[int, deque[np.ndarray]] = {}

    def reset_runtime_state(self) -> None:
        self.tracker.reset()
        self.selector.reset()
        self.buffer.reset()
        self.state_machine.reset()
        self.frame_index = 0
        self.last_result = None
        self._score_history.clear()
        self._gesture_history.clear()
        self._command_history.clear()
        self._direction_history.clear()

    def _clear_missing_score_history(self, active_track_ids: List[int]) -> None:
        active = {int(track_id) for track_id in active_track_ids if track_id is not None}
        stale = [track_id for track_id in self._score_history.keys() if track_id not in active]
        for track_id in stale:
            self._score_history.pop(track_id, None)
            self._gesture_history.pop(track_id, None)
            self._command_history.pop(track_id, None)
            self._direction_history.pop(track_id, None)

    def _smooth_prediction_scores(self, track_id: int, scores: Dict[str, float]) -> np.ndarray:
        vector = np.asarray([float(scores.get(label, 0.0)) for label in self.class_names], dtype=np.float32)
        if self.prediction_smoothing_window <= 1:
            return vector
        history = self._score_history.setdefault(track_id, deque(maxlen=self.prediction_smoothing_window))
        history.append(vector)
        return np.mean(np.stack(list(history), axis=0), axis=0)

    @staticmethod
    def _score_map_to_vector(score_map: Dict[str, float], labels: List[str]) -> np.ndarray:
        return np.asarray([float(score_map.get(label, 0.0)) for label in labels], dtype=np.float32)

    def _smooth_task_prediction(self, track_id: int, pred: Prediction) -> Tuple[np.ndarray, Dict[str, object]] | None:
        extras = pred.extras or {}
        if not self.task_smoothing_enabled or "gesture_scores_raw" not in extras:
            return None

        gesture_raw = self._score_map_to_vector(extras["gesture_scores_raw"], self.class_names)
        gesture_history = self._gesture_history.setdefault(track_id, deque(maxlen=self.gesture_smoothing_window))
        gesture_history.append(gesture_raw)
        gesture_probs = np.mean(np.stack(list(gesture_history), axis=0), axis=0)
        gesture_probs = gesture_probs / np.clip(np.sum(gesture_probs), 1e-8, None)

        command_probs = None
        if self.command_labels and "command_scores" in extras:
            command_raw = self._score_map_to_vector(extras["command_scores"], self.command_labels)
            command_history = self._command_history.setdefault(track_id, deque(maxlen=self.command_smoothing_window))
            command_history.append(command_raw)
            command_probs = np.mean(np.stack(list(command_history), axis=0), axis=0)
            command_probs = command_probs / np.clip(np.sum(command_probs), 1e-8, None)

        direction_probs = None
        if self.direction_labels and "direction_scores" in extras:
            direction_raw = self._score_map_to_vector(extras["direction_scores"], self.direction_labels)
            direction_history = self._direction_history.setdefault(track_id, deque(maxlen=self.direction_smoothing_window))
            direction_history.append(direction_raw)
            direction_probs = np.mean(np.stack(list(direction_history), axis=0), axis=0)
            direction_probs = direction_probs / np.clip(np.sum(direction_probs), 1e-8, None)

        fused = np.log(np.clip(gesture_probs, 1e-8, 1.0))
        if command_probs is not None and self.command_weight > 0.0:
            command_log = np.asarray([
                np.log(np.clip(command_probs[self.command_label_to_index[gesture_to_command(label)]], 1e-8, 1.0))
                for label in self.class_names
            ], dtype=np.float32)
            fused += self.command_weight * command_log
        if direction_probs is not None and self.direction_weight > 0.0:
            direction_log = np.asarray([
                np.log(np.clip(direction_probs[self.direction_label_to_index[gesture_to_direction(label)]], 1e-8, 1.0))
                for label in self.class_names
            ], dtype=np.float32)
            fused += self.direction_weight * direction_log
        fused -= np.max(fused)
        probs = np.exp(fused)
        probs = probs / np.clip(np.sum(probs), 1e-8, None)

        debug: Dict[str, object] = {
            "gesture_scores_raw_smoothed": {
                self.class_names[i]: float(gesture_probs[i]) for i in range(len(self.class_names))
            }
        }
        if command_probs is not None:
            command_idx = int(np.argmax(command_probs))
            debug["command_scores_smoothed"] = {
                self.command_labels[i]: float(command_probs[i]) for i in range(len(self.command_labels))
            }
            debug["command_label_smoothed"] = self.command_labels[command_idx]
        if direction_probs is not None:
            direction_idx = int(np.argmax(direction_probs))
            debug["direction_scores_smoothed"] = {
                self.direction_labels[i]: float(direction_probs[i]) for i in range(len(self.direction_labels))
            }
            debug["direction_label_smoothed"] = self.direction_labels[direction_idx]
        return probs, debug

    @staticmethod
    def _visible_keypoint_count(scores: np.ndarray, threshold: float = 0.05) -> int:
        if scores.size == 0:
            return 0
        return int(np.sum(scores >= threshold))

    def _build_debug_detections(self, detections: List[dict]) -> List[dict]:
        debug_dets = []
        for det in detections:
            scores = np.asarray(det.get("keypoint_scores", []), dtype=np.float32)
            debug_dets.append({
                "track_id": None if det.get("track_id") is None else int(det["track_id"]),
                "bbox": [float(x) for x in det.get("bbox", [])],
                "score": float(det.get("score", 0.0)),
                "track_hits": int(det.get("track_hits", 0)),
                "occluded": bool(det.get("occluded", False)),
                "selector_score": float(det.get("selector_score", 0.0)),
                "selector_components": {
                    key: float(value)
                    for key, value in det.get("selector_components", {}).items()
                },
                "visible_keypoints": self._visible_keypoint_count(scores),
                "mean_keypoint_score": float(scores.mean()) if scores.size else 0.0,
            })
        debug_dets.sort(key=lambda item: item["selector_score"], reverse=True)
        return debug_dets

    def _detect(self, frame: np.ndarray) -> List[dict]:
        if self.detector is None:
            return []
        return self.detector.detect(frame)

    def process_frame(self, frame: np.ndarray, fps: float = 0.0) -> tuple[CommandOutput, List[dict]]:
        tic = time.perf_counter()
        candidates = resolve_pose_candidates(frame, self.pose_backend, self.detector)
        detections = self.pose_backend.estimate(frame, self.frame_index, candidates=candidates)
        detections = self.tracker.update(detections)
        active_track_id = self.selector.select(detections, frame)
        self.buffer.clear_missing(list(self.tracker.get_confirmed_tracks().keys()))
        self._clear_missing_score_history(list(self.tracker.get_confirmed_tracks().keys()))

        gesture = "NO_GESTURE"
        gesture_conf = 0.0
        intent = "UNKNOWN"
        command = "NO_COMMAND"
        command_conf = 0.0
        safe_fallback = False
        pred_debug = None

        target_det = None
        for det in detections:
            if det.get("track_id") == active_track_id:
                target_det = det
                break

        if target_det is not None:
            kpts = np.asarray(target_det["keypoints"], dtype=np.float32)
            scores = np.asarray(target_det.get("keypoint_scores", np.ones(len(kpts))), dtype=np.float32)
            if kpts.ndim == 2:
                kpts = np.concatenate([kpts, scores[:, None]], axis=-1)
            self.buffer.append(active_track_id, self.frame_index, kpts, bbox=np.asarray(target_det.get("bbox", []), dtype=np.float32))
            seq = self.buffer.get(active_track_id)
            bbox_seq = self.buffer.get_bboxes(active_track_id)
            pred = self.classifier.predict_sequence(seq, bbox_track=bbox_seq) if len(seq) >= self.min_predict_len else None
            if pred is not None:
                pred_debug = {
                    "label": pred.label,
                    "confidence": float(pred.confidence),
                    "scores": {key: float(value) for key, value in pred.scores.items()},
                }
                task_smoothed = self._smooth_task_prediction(int(active_track_id), pred)
                if task_smoothed is not None:
                    smoothed_scores, task_debug = task_smoothed
                    pred_debug.update(task_debug)
                else:
                    smoothed_scores = self._smooth_prediction_scores(int(active_track_id), pred.scores)
                    smoothed_scores = smoothed_scores / np.clip(np.sum(smoothed_scores), 1e-8, None)
                gesture_idx = int(np.argmax(smoothed_scores))
                gesture = self.class_names[gesture_idx]
                gesture_conf = float(smoothed_scores[gesture_idx])
                if self.prediction_smoothing_window > 1 or task_smoothed is not None:
                    pred_debug["label_smoothed"] = gesture
                    pred_debug["confidence_smoothed"] = gesture_conf
                    pred_debug["scores_smoothed"] = {
                        self.class_names[i]: float(smoothed_scores[i]) for i in range(len(self.class_names))
                    }
                parsed = self.intent_parser.parse(gesture, gesture_conf)
                intent = str(parsed["intent"])
                command = str(parsed["command"])
                command_conf = float(parsed["command_confidence"])
                command, command_conf, safe_fallback = self.state_machine.step(
                    command, command_conf, occluded=bool(target_det.get("occluded", False))
                )
            else:
                command, command_conf, safe_fallback = self.state_machine.step("NO_COMMAND", 0.0, occluded=True)
        else:
            command, command_conf, safe_fallback = self.state_machine.step("NO_COMMAND", 0.0, occluded=True)

        if command != "NO_COMMAND":
            if gesture == "NO_GESTURE":
                # 处于保持/回退状态时，将命令反推为更稳定的文本
                gesture = command if command != "KEEP_WAIT" else "LEFT_TURN_WAIT"
                gesture_conf = command_conf
            intent = gesture_to_intent(gesture)
        else:
            intent = "UNKNOWN"

        latency_ms = (time.perf_counter() - tic) * 1000.0
        result = CommandOutput(
            frame_index=self.frame_index,
            timestamp_sec=self.frame_index / fps if fps > 0 else float(self.frame_index),
            active_track_id=active_track_id,
            gesture=gesture,
            gesture_confidence=float(gesture_conf),
            intent=intent,
            command=command,
            command_confidence=float(command_conf),
            safe_fallback=bool(safe_fallback),
            latency_ms=float(latency_ms),
            fps=float(fps),
            notes=[],
            debug={
                "num_detections": len(detections),
                "active_track_id": active_track_id,
                "min_predict_len": int(self.min_predict_len),
                "tracks": self._build_debug_detections(detections),
                "active_track": None if target_det is None else {
                    "track_id": int(active_track_id) if active_track_id is not None else None,
                    "bbox": [float(x) for x in target_det.get("bbox", [])],
                    "track_hits": int(target_det.get("track_hits", 0)),
                    "selector_score": float(target_det.get("selector_score", 0.0)),
                    "selector_components": {
                        key: float(value)
                        for key, value in target_det.get("selector_components", {}).items()
                    },
                    "occluded": bool(target_det.get("occluded", False)),
                    "buffer_len": int(self.buffer.length(active_track_id)) if active_track_id is not None else 0,
                    "visible_keypoints": self._visible_keypoint_count(scores) if target_det is not None else 0,
                    "mean_keypoint_score": float(scores.mean()) if target_det is not None and scores.size else 0.0,
                    "ready_for_prediction": bool(target_det is not None and self.buffer.length(active_track_id) >= self.min_predict_len) if active_track_id is not None else False,
                },
                "raw_prediction": pred_debug,
                "state_machine": self.state_machine.get_debug_state(),
            },
        )
        self.last_result = result
        self.frame_index += 1
        return result, detections
