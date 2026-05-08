from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

from tpgr.ctpgv2_segmenter import apply_gesture_bias
from tpgr.data.features import adapt_sequence_length, normalize_keypoints_sequence, sequence_to_graph_input, sequence_to_tcn_features
from tpgr.data.labels import (
    get_index_to_label,
    get_label_space_classes,
    get_task_index_to_label,
    get_task_label_to_index,
    gesture_to_command,
    gesture_to_direction,
)
from tpgr.models.builder import build_model


@dataclass
class Prediction:
    label: str
    confidence: float
    scores: Dict[str, float]
    extras: Dict[str, object] | None = None


class BaseClassifier:
    def predict_sequence(self, keypoints: np.ndarray, bbox_track: np.ndarray | None = None) -> Prediction:
        raise NotImplementedError


class RuleBasedGestureClassifier(BaseClassifier):
    def __init__(self, label_space: str = "canonical") -> None:
        self.label_space = label_space
        self.labels = list(get_label_space_classes(label_space))
        if label_space not in {"canonical", "base", "gesture", "9class"}:
            raise ValueError("RuleBasedGestureClassifier 仅支持 canonical 9 类标签空间")

    @staticmethod
    def _arm_vector(frame: np.ndarray, side: str) -> np.ndarray:
        shoulder = 5 if side == "left" else 6
        wrist = 9 if side == "left" else 10
        return frame[wrist, :2] - frame[shoulder, :2]

    @staticmethod
    def _arm_y(frame: np.ndarray, side: str) -> float:
        shoulder = 5 if side == "left" else 6
        wrist = 9 if side == "left" else 10
        return float(frame[wrist, 1] - frame[shoulder, 1])

    def predict_sequence(self, keypoints: np.ndarray, bbox_track: np.ndarray | None = None) -> Prediction:
        if len(keypoints) == 0:
            return Prediction("NO_GESTURE", 0.0, {"NO_GESTURE": 1.0})
        seq = normalize_keypoints_sequence(keypoints)
        frame = seq[-5:].mean(axis=0) if len(seq) >= 5 else seq.mean(axis=0)
        left = self._arm_vector(frame, "left")
        right = self._arm_vector(frame, "right")
        left_h = abs(left[1]) < 0.25 and abs(left[0]) > 0.25
        right_h = abs(right[1]) < 0.25 and abs(right[0]) > 0.25
        left_up = self._arm_y(frame, "left") < -0.25
        right_up = self._arm_y(frame, "right") < -0.25
        left_down = self._arm_y(frame, "left") > 0.20
        right_down = self._arm_y(frame, "right") > 0.20

        scores = {name: 0.01 for name in self.labels}
        if left_h and right_h:
            scores["STOP"] = 0.90
        elif left_h and right_up:
            scores["LEFT_TURN_WAIT"] = 0.80
        elif left_h:
            scores["TURN_LEFT"] = 0.70
        elif right_h:
            scores["TURN_RIGHT"] = 0.70
        elif left_up and right_up:
            scores["GO_STRAIGHT"] = 0.75
        elif left_down and right_down:
            scores["SLOW_DOWN"] = 0.70
        else:
            scores["NO_GESTURE"] = 0.60

        total = sum(scores.values())
        scores = {k: float(v / total) for k, v in scores.items()}
        label = max(scores, key=scores.get)
        return Prediction(label=label, confidence=scores[label], scores=scores)


class TorchGestureClassifier(BaseClassifier):
    def __init__(
        self,
        model_cfg: dict,
        checkpoint: str | Path,
        representation: str = "features",
        clip_len: int = 48,
        device: str = "cpu",
        feature_variant: str = "base",
        keypoint_subset: str = "all",
        tta_num_clips: int = 1,
        temporal_mode: str = "crop_pad",
        label_space: str = "canonical",
        task_fusion: dict | None = None,
        gesture_bias: dict | None = None,
    ) -> None:
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model = build_model(model_cfg).to(self.device)
        state = torch.load(checkpoint, map_location=self.device, weights_only=False)
        if "model" in state:
            state = state["model"]
        self.model.load_state_dict(state, strict=True)
        if bool(model_cfg.get("compile", False)) and hasattr(torch, "compile"):
            self.model = torch.compile(self.model, mode=str(model_cfg.get("compile_mode", "reduce-overhead")))
        self.model.eval()
        self.representation = representation
        self.clip_len = clip_len
        self.feature_variant = feature_variant
        self.amp_enabled = str(self.device).startswith("cuda") and bool(model_cfg.get("amp", True))
        self.keypoint_subset = keypoint_subset
        self.tta_num_clips = max(1, int(tta_num_clips))
        self.temporal_mode = str(temporal_mode or "crop_pad")
        self.label_space = label_space
        self.class_names = list(get_label_space_classes(label_space))
        self.index_to_label = get_index_to_label(label_space)
        self.command_index_to_label = get_task_index_to_label("command", label_space=label_space)
        self.direction_index_to_label = get_task_index_to_label("direction", label_space=label_space)
        self.command_label_to_index = get_task_label_to_index("command", label_space=label_space)
        self.direction_label_to_index = get_task_label_to_index("direction", label_space=label_space)
        self.task_fusion = {
            "command_weight": float((task_fusion or {}).get("command_weight", 0.0)),
            "direction_weight": float((task_fusion or {}).get("direction_weight", 0.0)),
        }
        self.gesture_bias = dict(gesture_bias or {})

    @staticmethod
    def _sequence_logits_to_probs(logits: torch.Tensor) -> np.ndarray:
        # 支持两类模型输出：
        # 1. clip 分类器: [B, C]
        # 2. 逐帧分割器: [B, C, T]，在线场景取当前缓冲区最后一帧
        if logits.ndim == 3:
            logits = logits[..., -1]
        if logits.ndim != 2:
            raise ValueError(f"不支持的 logits 形状: {tuple(logits.shape)}")
        return torch.softmax(logits, dim=1).mean(dim=0).cpu().numpy()

    def _select_keypoints(self, keypoints: np.ndarray) -> np.ndarray:
        if self.keypoint_subset == "body17" and keypoints.ndim == 3 and keypoints.shape[1] >= 17:
            return keypoints[:, :17]
        return keypoints

    def _prepare_input(self, keypoints: np.ndarray, bbox_track: np.ndarray | None = None) -> torch.Tensor:
        keypoints = self._select_keypoints(keypoints)
        if self.representation == "features":
            arr = keypoints.astype(np.float32) if keypoints.ndim == 2 else sequence_to_tcn_features(
                keypoints,
                feature_variant=self.feature_variant,
                bbox_track=bbox_track,
            )
            arr = adapt_sequence_length(arr, self.clip_len, training=False, mode=self.temporal_mode)
            tensor = torch.from_numpy(arr).unsqueeze(0)
        elif self.representation == "graph":
            arr = sequence_to_graph_input(keypoints)
            arr = np.transpose(arr, (1, 2, 0))
            arr = adapt_sequence_length(arr, self.clip_len, training=False, mode=self.temporal_mode)
            arr = np.transpose(arr, (2, 0, 1))
            tensor = torch.from_numpy(arr).unsqueeze(0)
        else:
            raise ValueError(f"不支持的 representation: {self.representation}")
        return tensor.float().to(self.device)

    def _temporal_views(self, keypoints: np.ndarray, bbox_track: np.ndarray | None = None) -> list[tuple[np.ndarray, np.ndarray | None]]:
        keypoints = self._select_keypoints(keypoints)
        if keypoints.shape[0] <= self.clip_len or self.tta_num_clips <= 1:
            return [(keypoints, bbox_track)]
        max_start = keypoints.shape[0] - self.clip_len
        starts = np.linspace(0, max_start, num=self.tta_num_clips, dtype=np.int32)
        views = []
        for start in starts.tolist():
            end = start + self.clip_len
            bbox_view = None if bbox_track is None else bbox_track[start:end]
            views.append((keypoints[start:end], bbox_view))
        return views

    @torch.no_grad()
    def predict_sequence(self, keypoints: np.ndarray, bbox_track: np.ndarray | None = None) -> Prediction:
        if len(keypoints) == 0:
            background = self.class_names[0]
            return Prediction(background, 0.0, {background: 1.0})
        views = self._temporal_views(keypoints, bbox_track=bbox_track)
        x = torch.cat([self._prepare_input(view, bbox_view) for view, bbox_view in views], dim=0)
        autocast = torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.amp_enabled) \
            if str(self.device).startswith("cuda") else nullcontext()
        with autocast:
            logits = self.model(x)
        if isinstance(logits, dict):
            gesture_probs = self._sequence_logits_to_probs(logits["gesture"])
            command_probs = None
            direction_probs = None
            if "command" in logits:
                command_probs = self._sequence_logits_to_probs(logits["command"])
            if "direction" in logits:
                direction_probs = self._sequence_logits_to_probs(logits["direction"])
            fused = np.log(np.clip(gesture_probs, 1e-8, 1.0))
            if command_probs is not None and self.task_fusion["command_weight"] > 0:
                command_log = np.asarray([
                    np.log(np.clip(command_probs[self.command_label_to_index[gesture_to_command(label)]], 1e-8, 1.0))
                    for label in self.class_names
                ], dtype=np.float32)
                fused += self.task_fusion["command_weight"] * command_log
            if direction_probs is not None and self.task_fusion["direction_weight"] > 0:
                direction_log = np.asarray([
                    np.log(np.clip(direction_probs[self.direction_label_to_index[gesture_to_direction(label)]], 1e-8, 1.0))
                    for label in self.class_names
                ], dtype=np.float32)
                fused += self.task_fusion["direction_weight"] * direction_log
            fused = fused - np.max(fused)
            probs = np.exp(fused)
            probs = probs / np.clip(np.sum(probs), 1e-8, None)
            if self.gesture_bias:
                probs = apply_gesture_bias(
                    probs[None, :],
                    label_space=self.label_space,
                    bias_map=self.gesture_bias,
                )[0]
            extras = {}
            extras["gesture_scores_raw"] = {
                self.index_to_label[i]: float(gesture_probs[i])
                for i in range(len(gesture_probs))
            }
            if command_probs is not None:
                command_idx = int(np.argmax(command_probs))
                extras["command_label"] = self.command_index_to_label[command_idx]
                extras["command_scores"] = {
                    self.command_index_to_label[i]: float(command_probs[i])
                    for i in range(len(command_probs))
                }
            if direction_probs is not None:
                direction_idx = int(np.argmax(direction_probs))
                extras["direction_label"] = self.direction_index_to_label[direction_idx]
                extras["direction_scores"] = {
                    self.direction_index_to_label[i]: float(direction_probs[i])
                    for i in range(len(direction_probs))
                }
        else:
            probs = self._sequence_logits_to_probs(logits)
            if self.gesture_bias:
                probs = apply_gesture_bias(
                    probs[None, :],
                    label_space=self.label_space,
                    bias_map=self.gesture_bias,
                )[0]
            extras = None
        label_idx = int(np.argmax(probs))
        scores = {self.index_to_label[i]: float(probs[i]) for i in range(len(probs))}
        return Prediction(
            label=self.index_to_label[label_idx],
            confidence=float(probs[label_idx]),
            scores=scores,
            extras=extras,
        )


class EnsembleGestureClassifier(BaseClassifier):
    def __init__(self, members: list[BaseClassifier], weights: list[float] | None = None) -> None:
        if not members:
            raise ValueError("集成分类器至少需要一个成员")
        self.members = members
        self.weights = weights or [1.0] * len(members)
        if len(self.weights) != len(self.members):
            raise ValueError("集成分类器的 weights 数量必须与成员数量一致")
        self.class_names = list(getattr(members[0], "class_names", get_label_space_classes("canonical")))
        self.label_space = getattr(members[0], "label_space", "canonical")
        self.index_to_label = getattr(members[0], "index_to_label", None)
        self.command_index_to_label = getattr(members[0], "command_index_to_label", {})
        self.direction_index_to_label = getattr(members[0], "direction_index_to_label", {})
        self.command_label_to_index = getattr(members[0], "command_label_to_index", {})
        self.direction_label_to_index = getattr(members[0], "direction_label_to_index", {})
        self.task_fusion = dict(getattr(members[0], "task_fusion", {}) or {})
        self.gesture_bias = dict(getattr(members[0], "gesture_bias", {}) or {})

    @staticmethod
    def _weighted_average_dict(
        predictions: list[tuple[Prediction, float]],
        extra_key: str,
        label_names: list[str],
    ) -> dict[str, float] | None:
        total_weight = 0.0
        probs = np.zeros((len(label_names),), dtype=np.float32)
        for pred, weight in predictions:
            extras = pred.extras or {}
            if extra_key not in extras:
                continue
            total_weight += float(weight)
            score_map = extras[extra_key]
            for idx, label in enumerate(label_names):
                probs[idx] += float(weight) * float(score_map.get(label, 0.0))
        if total_weight <= 0.0:
            return None
        probs = probs / total_weight
        probs = probs / np.clip(np.sum(probs), 1e-8, None)
        return {label_names[idx]: float(probs[idx]) for idx in range(len(label_names))}

    def predict_sequence(self, keypoints: np.ndarray, bbox_track: np.ndarray | None = None) -> Prediction:
        class_names = self.class_names or list(get_label_space_classes("canonical"))
        probs = np.zeros((len(class_names),), dtype=np.float32)
        total_weight = 0.0
        predictions: list[tuple[Prediction, float]] = []
        for member, weight in zip(self.members, self.weights):
            pred = member.predict_sequence(keypoints, bbox_track=bbox_track)
            predictions.append((pred, float(weight)))
            total_weight += float(weight)
            for idx, label in enumerate(class_names):
                probs[idx] += float(weight) * float(pred.scores.get(label, 0.0))
        probs = probs / max(total_weight, 1e-6)
        label_idx = int(np.argmax(probs))
        scores = {class_names[i]: float(probs[i]) for i in range(len(probs))}
        extras: Dict[str, object] | None = None

        gesture_scores_raw = self._weighted_average_dict(predictions, "gesture_scores_raw", class_names)
        command_labels = list(self.command_index_to_label.values())
        command_scores = (
            self._weighted_average_dict(predictions, "command_scores", command_labels)
            if command_labels else None
        )
        direction_labels = list(self.direction_index_to_label.values())
        direction_scores = (
            self._weighted_average_dict(predictions, "direction_scores", direction_labels)
            if direction_labels else None
        )
        if gesture_scores_raw is not None or command_scores is not None or direction_scores is not None:
            extras = {}
            if gesture_scores_raw is not None:
                extras["gesture_scores_raw"] = gesture_scores_raw
            if command_scores is not None:
                extras["command_scores"] = command_scores
                command_label = max(command_scores, key=command_scores.get)
                extras["command_label"] = command_label
            if direction_scores is not None:
                extras["direction_scores"] = direction_scores
                direction_label = max(direction_scores, key=direction_scores.get)
                extras["direction_label"] = direction_label
        return Prediction(
            label=class_names[label_idx],
            confidence=float(probs[label_idx]),
            scores=scores,
            extras=extras,
        )


def build_classifier(cfg: dict) -> BaseClassifier:
    kind = cfg.get("type", "rule_based").lower()
    label_space = cfg.get("label_space", "canonical")
    if kind == "rule_based":
        return RuleBasedGestureClassifier(label_space=label_space)
    if kind == "torch":
        return TorchGestureClassifier(
            model_cfg=cfg["model"],
            checkpoint=cfg["checkpoint"],
            representation=cfg.get("representation", "features"),
            clip_len=int(cfg.get("clip_len", 48)),
            device=cfg.get("device", "cpu"),
            feature_variant=cfg.get("feature_variant", "base"),
            keypoint_subset=cfg.get("keypoint_subset", "all"),
            tta_num_clips=int(cfg.get("tta_num_clips", 1)),
            temporal_mode=cfg.get("temporal_mode", "crop_pad"),
            label_space=label_space,
            task_fusion=cfg.get("task_fusion"),
            gesture_bias=cfg.get("gesture_bias"),
        )
    if kind == "ensemble":
        members = []
        weights = []
        for item in cfg.get("members", []):
            member_cfg = {k: v for k, v in item.items() if k != "weight"}
            members.append(build_classifier(member_cfg))
            weights.append(float(item.get("weight", 1.0)))
        return EnsembleGestureClassifier(members=members, weights=weights)
    raise ValueError(f"不支持的 classifier.type: {kind}")
