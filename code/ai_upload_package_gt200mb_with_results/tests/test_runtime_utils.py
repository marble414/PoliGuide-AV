import numpy as np

from tpgr.pipeline.runtime_utils import resolve_pose_candidates


class _Backend:
    def __init__(self, needs_detector: bool):
        self.needs_detector = needs_detector


class _Detector:
    def __init__(self, detections):
        self._detections = detections

    def detect(self, frame):
        return list(self._detections)


def test_resolve_pose_candidates_skips_detector_for_full_frame_backends():
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    assert resolve_pose_candidates(frame, _Backend(needs_detector=False), detector=None) is None


def test_resolve_pose_candidates_falls_back_to_full_frame_box():
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    candidates = resolve_pose_candidates(frame, _Backend(needs_detector=True), detector=_Detector([]))
    assert candidates == [{"bbox": [0.0, 0.0, 320.0, 240.0], "score": 1.0}]


def test_resolve_pose_candidates_prefers_detector_output():
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    detections = [{"bbox": [1.0, 2.0, 3.0, 4.0], "score": 0.8}]
    candidates = resolve_pose_candidates(frame, _Backend(needs_detector=True), detector=_Detector(detections))
    assert candidates == detections
