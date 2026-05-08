from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return float(inter / union)


@dataclass
class Tracklet:
    track_id: int
    bbox: np.ndarray
    score: float
    hits: int = 1
    misses: int = 0
    last_seen: int = 0
    extras: Dict[str, float] = field(default_factory=dict)


class GreedyIoUTracker:
    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_age: int = 15,
        min_hits: int = 1,
    ) -> None:
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self.tracks: Dict[int, Tracklet] = {}
        self._next_id = 1
        self.frame_index = 0

    @staticmethod
    def _bbox_center_area(bbox: list[float] | np.ndarray) -> tuple[np.ndarray, float]:
        bbox = np.asarray(bbox, dtype=np.float32)
        center = np.array([(bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5], dtype=np.float32)
        area = float(max(bbox[2] - bbox[0], 1.0) * max(bbox[3] - bbox[1], 1.0))
        return center, area

    def _match(self, detections: List[dict]) -> List[Tuple[int, int]]:
        track_ids = list(self.tracks.keys())
        if not track_ids or not detections:
            return []
        ious = np.zeros((len(track_ids), len(detections)), dtype=np.float32)
        for i, track_id in enumerate(track_ids):
            for j, det in enumerate(detections):
                ious[i, j] = iou_xyxy(self.tracks[track_id].bbox, np.array(det["bbox"], dtype=np.float32))
        matches: List[Tuple[int, int]] = []
        used_tracks = set()
        used_dets = set()
        while True:
            max_idx = np.unravel_index(np.argmax(ious), ious.shape)
            score = ious[max_idx]
            if score < self.iou_threshold:
                break
            ti, dj = int(max_idx[0]), int(max_idx[1])
            if ti in used_tracks or dj in used_dets:
                ious[ti, dj] = -1.0
                continue
            matches.append((track_ids[ti], dj))
            used_tracks.add(ti)
            used_dets.add(dj)
            ious[ti, :] = -1.0
            ious[:, dj] = -1.0
        return matches

    def update(self, detections: List[dict]) -> List[dict]:
        self.frame_index += 1
        matches = self._match(detections)
        matched_tracks = set()
        matched_dets = set()

        for track_id, det_idx in matches:
            det = detections[det_idx]
            track = self.tracks[track_id]
            prev_center = np.array(track.extras.get("center", self._bbox_center_area(track.bbox)[0]), dtype=np.float32)
            center, area = self._bbox_center_area(det["bbox"])
            motion = float(np.linalg.norm(center - prev_center) / max(np.sqrt(area), 1.0))
            ema_motion = 0.6 * float(track.extras.get("ema_motion", motion)) + 0.4 * motion
            track.bbox = np.array(det["bbox"], dtype=np.float32)
            track.score = float(det.get("score", 1.0))
            track.hits += 1
            track.misses = 0
            track.last_seen = self.frame_index
            track.extras["center"] = center.tolist()
            track.extras["area"] = area
            track.extras["ema_motion"] = float(ema_motion)
            matched_tracks.add(track_id)
            matched_dets.add(det_idx)
            det["track_id"] = track_id
            det["track_hits"] = track.hits
            det["track_misses"] = track.misses
            det["track_motion"] = float(ema_motion)

        unmatched_tracks = [tid for tid in list(self.tracks.keys()) if tid not in matched_tracks]
        for track_id in unmatched_tracks:
            track = self.tracks[track_id]
            track.misses += 1
            if track.misses > self.max_age:
                del self.tracks[track_id]

        for det_idx, det in enumerate(detections):
            if det_idx in matched_dets:
                continue
            track_id = self._next_id
            self._next_id += 1
            self.tracks[track_id] = Tracklet(
                track_id=track_id,
                bbox=np.array(det["bbox"], dtype=np.float32),
                score=float(det.get("score", 1.0)),
                hits=1,
                misses=0,
                last_seen=self.frame_index,
                extras={
                    "center": self._bbox_center_area(det["bbox"])[0].tolist(),
                    "area": self._bbox_center_area(det["bbox"])[1],
                    "ema_motion": 0.0,
                },
            )
            det["track_id"] = track_id
            det["track_hits"] = 1
            det["track_misses"] = 0
            det["track_motion"] = 0.0

        return detections

    def get_confirmed_tracks(self) -> Dict[int, Tracklet]:
        return {tid: trk for tid, trk in self.tracks.items() if trk.hits >= self.min_hits}

    def reset(self) -> None:
        self.tracks.clear()
        self._next_id = 1
        self.frame_index = 0
