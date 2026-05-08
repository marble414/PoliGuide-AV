from __future__ import annotations

from typing import Dict, List, Optional

import cv2
import numpy as np


class OfficerSelector:
    def __init__(
        self,
        center_weight: float = 0.35,
        size_weight: float = 0.15,
        visibility_weight: float = 0.25,
        hand_weight: float = 0.15,
        track_age_weight: float = 0.15,
        stability_weight: float = 0.08,
        vest_weight: float = 0.10,
        sticky_bonus: float = 0.10,
    ) -> None:
        self.center_weight = center_weight
        self.size_weight = size_weight
        self.visibility_weight = visibility_weight
        self.hand_weight = hand_weight
        self.track_age_weight = track_age_weight
        self.stability_weight = stability_weight
        self.vest_weight = vest_weight
        self.sticky_bonus = sticky_bonus
        self.last_track_id: Optional[int] = None

    def _center_score(self, bbox: list[float], frame_shape: tuple[int, int, int]) -> float:
        h, w = frame_shape[:2]
        cx = 0.5 * (bbox[0] + bbox[2])
        cy = 0.5 * (bbox[1] + bbox[3])
        dist = np.linalg.norm(np.array([cx / w - 0.5, cy / h - 0.45], dtype=np.float32))
        return float(max(0.0, 1.0 - dist * 1.8))

    def _size_score(self, bbox: list[float], frame_shape: tuple[int, int, int]) -> float:
        h, w = frame_shape[:2]
        area = max(bbox[2] - bbox[0], 1.0) * max(bbox[3] - bbox[1], 1.0)
        return float(min(area / (w * h * 0.25), 1.0))

    def _visibility_score(self, det: dict) -> float:
        scores = np.asarray(det.get("keypoint_scores", []), dtype=np.float32)
        if scores.size == 0:
            return 0.0
        idx = [5, 6, 7, 8, 9, 10]
        idx = [i for i in idx if i < len(scores)]
        if not idx:
            return float(scores.mean())
        return float(scores[idx].mean())

    def _track_age_score(self, det: dict) -> float:
        hits = float(det.get("track_hits", 1))
        return float(min(hits / 10.0, 1.0))

    def _hand_visibility_score(self, det: dict) -> float:
        scores = np.asarray(det.get("keypoint_scores", []), dtype=np.float32)
        if scores.size == 0:
            return 0.0
        if scores.size >= 133:
            idx = [9, 10] + list(range(91, 112)) + list(range(112, 133))
        else:
            idx = [7, 8, 9, 10]
        idx = [i for i in idx if i < len(scores)]
        return float(scores[idx].mean()) if idx else float(scores.mean())

    def _stability_score(self, det: dict) -> float:
        motion = float(det.get("track_motion", 0.0))
        return float(max(0.0, 1.0 - motion * 2.5))

    def _vest_score(self, frame: np.ndarray, bbox: list[float]) -> float:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = map(int, bbox)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        roi = frame[y1:y1 + max((y2 - y1) // 2, 1), x1:x2]
        if roi.size == 0:
            return 0.0
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([20, 60, 80]), np.array([90, 255, 255]))
        return float(mask.mean() / 255.0)

    def _score_components(self, det: dict, frame: np.ndarray) -> dict:
        sticky = self.sticky_bonus if det.get("track_id") == self.last_track_id else 0.0
        center = self._center_score(det["bbox"], frame.shape)
        size = self._size_score(det["bbox"], frame.shape)
        visibility = self._visibility_score(det)
        hand = self._hand_visibility_score(det)
        track_age = self._track_age_score(det)
        stability = self._stability_score(det)
        vest = self._vest_score(frame, det["bbox"])
        total = (
            self.center_weight * center
            + self.size_weight * size
            + self.visibility_weight * visibility
            + self.hand_weight * hand
            + self.track_age_weight * track_age
            + self.stability_weight * stability
            + self.vest_weight * vest
            + sticky
        )
        return {
            "center": float(center),
            "size": float(size),
            "visibility": float(visibility),
            "hand": float(hand),
            "track_age": float(track_age),
            "stability": float(stability),
            "vest": float(vest),
            "sticky_bonus": float(sticky),
            "total": float(total),
        }

    def select(self, detections: List[dict], frame: np.ndarray) -> Optional[int]:
        if not detections:
            self.last_track_id = None
            return None

        best_track = None
        best_score = -1.0
        for det in detections:
            track_id = det.get("track_id")
            if track_id is None:
                continue
            components = self._score_components(det, frame)
            det["selector_components"] = components
            det["selector_score"] = float(components["total"])
            if components["total"] > best_score:
                best_score = components["total"]
                best_track = track_id

        self.last_track_id = best_track
        return best_track

    def reset(self) -> None:
        self.last_track_id = None
