from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch


class BasePoseBackend:
    needs_detector: bool = False
    num_keypoints: int = 17

    def estimate(
        self,
        frame: np.ndarray,
        frame_index: int,
        candidates: Optional[List[dict]] = None,
    ) -> List[dict]:
        raise NotImplementedError


class PrecomputedPoseBackend(BasePoseBackend):
    needs_detector = False

    def __init__(self, jsonl_path: str | Path) -> None:
        self.jsonl_path = Path(jsonl_path)
        self.frame_map: Dict[int, List[dict]] = {}
        self.num_keypoints = 17
        with self.jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                detections = item["detections"]
                if detections and "keypoints" in detections[0]:
                    self.num_keypoints = int(np.asarray(detections[0]["keypoints"]).shape[0])
                self.frame_map[int(item["frame_index"])] = detections

    def estimate(self, frame: np.ndarray, frame_index: int, candidates: Optional[List[dict]] = None) -> List[dict]:
        return [dict(det) for det in self.frame_map.get(frame_index, [])]


class DummyBoxPoseBackend(BasePoseBackend):
    needs_detector = True
    num_keypoints = 17

    def estimate(self, frame: np.ndarray, frame_index: int, candidates: Optional[List[dict]] = None) -> List[dict]:
        detections: List[dict] = []
        for cand in candidates or []:
            x1, y1, x2, y2 = cand["bbox"]
            w = max(x2 - x1, 1.0)
            h = max(y2 - y1, 1.0)
            keypoints = np.array([
                [x1 + 0.50 * w, y1 + 0.10 * h],
                [x1 + 0.45 * w, y1 + 0.09 * h],
                [x1 + 0.55 * w, y1 + 0.09 * h],
                [x1 + 0.40 * w, y1 + 0.11 * h],
                [x1 + 0.60 * w, y1 + 0.11 * h],
                [x1 + 0.35 * w, y1 + 0.28 * h],
                [x1 + 0.65 * w, y1 + 0.28 * h],
                [x1 + 0.25 * w, y1 + 0.45 * h],
                [x1 + 0.75 * w, y1 + 0.45 * h],
                [x1 + 0.15 * w, y1 + 0.60 * h],
                [x1 + 0.85 * w, y1 + 0.60 * h],
                [x1 + 0.42 * w, y1 + 0.58 * h],
                [x1 + 0.58 * w, y1 + 0.58 * h],
                [x1 + 0.38 * w, y1 + 0.78 * h],
                [x1 + 0.62 * w, y1 + 0.78 * h],
                [x1 + 0.35 * w, y1 + 0.98 * h],
                [x1 + 0.65 * w, y1 + 0.98 * h],
            ], dtype=np.float32)
            detections.append({
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "score": float(cand.get("score", 0.5)),
                "keypoints": keypoints.tolist(),
                "keypoint_scores": [0.5] * len(keypoints),
                "occluded": False,
            })
        return detections


class MediaPipePoseBackend(BasePoseBackend):
    needs_detector = True
    num_keypoints = 17

    _MP_TO_COCO = {
        0: 0,
        2: 2,
        5: 1,
        7: 3,
        8: 4,
        11: 5,
        12: 6,
        13: 7,
        14: 8,
        15: 9,
        16: 10,
        23: 11,
        24: 12,
        25: 13,
        26: 14,
        27: 15,
        28: 16,
    }

    def __init__(self, static_image_mode: bool = False, model_complexity: int = 1) -> None:
        try:
            import mediapipe as mp
        except ImportError as e:
            raise RuntimeError("MediaPipe 未安装，请安装 requirements-optional.txt 中的 mediapipe。") from e
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            enable_segmentation=False,
            min_detection_confidence=0.4,
            min_tracking_confidence=0.4,
        )

    def estimate(self, frame: np.ndarray, frame_index: int, candidates: Optional[List[dict]] = None) -> List[dict]:
        outputs: List[dict] = []
        h, w = frame.shape[:2]
        for cand in candidates or []:
            x1, y1, x2, y2 = map(int, cand["bbox"])
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            result = self.pose.process(rgb)
            if not result.pose_landmarks:
                continue
            kpts = np.zeros((17, 2), dtype=np.float32)
            scores = np.zeros((17,), dtype=np.float32)
            for mp_idx, coco_idx in self._MP_TO_COCO.items():
                lm = result.pose_landmarks.landmark[mp_idx]
                kpts[coco_idx, 0] = x1 + lm.x * (x2 - x1)
                kpts[coco_idx, 1] = y1 + lm.y * (y2 - y1)
                scores[coco_idx] = float(getattr(lm, "visibility", 0.5))
            outputs.append({
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "score": float(cand.get("score", 0.5)),
                "keypoints": kpts.tolist(),
                "keypoint_scores": scores.tolist(),
                "occluded": float(scores.mean()) < 0.2,
            })
        return outputs


class OpenMMLabPoseBackend(BasePoseBackend):
    needs_detector = False

    @staticmethod
    def _register_safe_checkpoint_globals() -> None:
        add_safe_globals = getattr(torch.serialization, "add_safe_globals", None)
        if add_safe_globals is None:
            return
        safe_globals = [
            np.core.multiarray._reconstruct,
            np.ndarray,
            np.dtype,
            np.float32,
            np.float64,
            np.int64,
            type(np.dtype(np.float32)),
            type(np.dtype(np.float64)),
            type(np.dtype(np.int64)),
        ]
        add_safe_globals(safe_globals)

    @staticmethod
    def _build_inferencer(factory, **kwargs):
        original_torch_load = torch.load

        def load_with_legacy_default(*args, **load_kwargs):
            load_kwargs.setdefault("weights_only", False)
            return original_torch_load(*args, **load_kwargs)

        torch.load = load_with_legacy_default
        try:
            return factory(**kwargs)
        finally:
            torch.load = original_torch_load

    def __init__(
        self,
        model_alias: str = "human",
        device: Optional[str] = None,
        pose2d_weights: Optional[str] = None,
        det_model: Optional[str] = None,
        det_weights: Optional[str] = None,
        scope: str = "mmpose",
        pose_model: Optional[str] = None,
    ) -> None:
        try:
            from mmpose.apis import MMPoseInferencer
            from mmpose.apis.inferencers.utils.default_det_models import default_det_models
            from mmpose.utils import register_all_modules as register_mmpose_modules
            from mmdet.utils import register_all_modules as register_mmdet_modules
        except ImportError as e:
            raise RuntimeError(
                "OpenMMLab 依赖缺失，请按照 README 安装 mmpose/mmdet/mmengine/mmcv。"
            ) from e
        register_mmpose_modules(init_default_scope=True)
        register_mmdet_modules(init_default_scope=True)
        pose_model = pose_model or model_alias
        if det_weights and not det_model:
            det_model = default_det_models.get(model_alias, {}).get("model") or default_det_models.get(pose_model, {}).get("model")
        self._register_safe_checkpoint_globals()
        self.inferencer = self._build_inferencer(
            MMPoseInferencer,
            pose2d=pose_model,
            pose2d_weights=pose2d_weights,
            device=device,
            scope=scope,
            det_model=det_model,
            det_weights=det_weights,
            show_progress=False,
        )
        dataset_meta = getattr(getattr(self.inferencer, "model", None), "dataset_meta", {}) or {}
        self.num_keypoints = int(dataset_meta.get("num_keypoints", 133 if "wholebody" in str(pose_model).lower() else 17))

    @staticmethod
    def _normalize_bbox(bbox: object, keypoints: np.ndarray) -> List[float]:
        if bbox is None:
            mins = keypoints.min(axis=0)
            maxs = keypoints.max(axis=0)
            return [float(mins[0]), float(mins[1]), float(maxs[0]), float(maxs[1])]

        bbox_arr = np.asarray(bbox, dtype=np.float32).reshape(-1)
        if bbox_arr.size < 4:
            mins = keypoints.min(axis=0)
            maxs = keypoints.max(axis=0)
            return [float(mins[0]), float(mins[1]), float(maxs[0]), float(maxs[1])]
        return [float(v) for v in bbox_arr[:4]]

    def _parse_instances(self, result: dict) -> List[dict]:
        predictions = result.get("predictions", [])
        if isinstance(predictions, list) and predictions and isinstance(predictions[0], list):
            instances = predictions[0]
        else:
            instances = predictions
        outputs: List[dict] = []
        for inst in instances:
            kpts = np.asarray(inst.get("keypoints"), dtype=np.float32)
            scores = np.asarray(inst.get("keypoint_scores"), dtype=np.float32)
            if kpts.ndim == 3 and kpts.shape[0] == 1:
                kpts = kpts[0]
            if scores.ndim == 2 and scores.shape[0] == 1:
                scores = scores[0]
            if kpts.ndim != 2 or kpts.shape[1] != 2:
                continue
            if scores.ndim == 0:
                scores = np.full((kpts.shape[0],), float(scores), dtype=np.float32)
            bbox = self._normalize_bbox(inst.get("bbox"), kpts)
            bbox_score = np.asarray(inst.get("bbox_score", np.mean(scores)), dtype=np.float32).reshape(-1)
            det_score = float(bbox_score[0]) if bbox_score.size else float(np.mean(scores))
            outputs.append({
                "bbox": bbox,
                "score": det_score,
                "keypoints": kpts.tolist(),
                "keypoint_scores": scores.tolist(),
                "occluded": float(np.mean(scores)) < 0.2,
            })
        return outputs

    def estimate(self, frame: np.ndarray, frame_index: int, candidates: Optional[List[dict]] = None) -> List[dict]:
        result = next(self.inferencer(frame, show=False, return_vis=False, draw_bbox=False))
        return self._parse_instances(result)
