from __future__ import annotations

from typing import List, Optional

import numpy as np

from tpgr.pipeline.detector import BaseDetector
from tpgr.pipeline.pose_backends import BasePoseBackend


def resolve_pose_candidates(
    frame: np.ndarray,
    pose_backend: BasePoseBackend,
    detector: Optional[BaseDetector] = None,
) -> Optional[List[dict]]:
    if not pose_backend.needs_detector:
        return None
    candidates = detector.detect(frame) if detector is not None else []
    if candidates:
        return candidates
    height, width = frame.shape[:2]
    return [{
        "bbox": [0.0, 0.0, float(width), float(height)],
        "score": 1.0,
    }]
