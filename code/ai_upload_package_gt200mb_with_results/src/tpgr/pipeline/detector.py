from __future__ import annotations

from dataclasses import dataclass
from typing import List

import cv2
import numpy as np


@dataclass
class PersonProposal:
    bbox: list[float]
    score: float


class BaseDetector:
    def detect(self, frame: np.ndarray) -> List[dict]:
        raise NotImplementedError


class HOGPersonDetector(BaseDetector):
    def __init__(
        self,
        win_stride: tuple[int, int] = (8, 8),
        padding: tuple[int, int] = (8, 8),
        scale: float = 1.05,
        score_threshold: float = 0.2,
        max_detections: int = 10,
    ) -> None:
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self.win_stride = win_stride
        self.padding = padding
        self.scale = scale
        self.score_threshold = score_threshold
        self.max_detections = max_detections

    def detect(self, frame: np.ndarray) -> List[dict]:
        boxes, weights = self.hog.detectMultiScale(
            frame, winStride=self.win_stride, padding=self.padding, scale=self.scale
        )
        detections: List[dict] = []
        for (x, y, w, h), score in zip(boxes, weights):
            score = float(score)
            if score < self.score_threshold:
                continue
            detections.append({
                "bbox": [float(x), float(y), float(x + w), float(y + h)],
                "score": score,
            })
        detections.sort(key=lambda item: item["score"], reverse=True)
        return detections[: self.max_detections]
