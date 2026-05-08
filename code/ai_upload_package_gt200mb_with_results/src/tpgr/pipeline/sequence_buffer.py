from __future__ import annotations

from collections import defaultdict, deque
from typing import Deque, Dict, List, Tuple

import numpy as np


class SequenceBuffer:
    def __init__(self, max_len: int = 64, num_keypoints: int = 17) -> None:
        self.max_len = max_len
        self.num_keypoints = num_keypoints
        self.buffers: Dict[int, Deque[Tuple[int, np.ndarray]]] = defaultdict(lambda: deque(maxlen=max_len))
        self.bbox_buffers: Dict[int, Deque[Tuple[int, np.ndarray]]] = defaultdict(lambda: deque(maxlen=max_len))

    def append(self, track_id: int, frame_index: int, keypoints: np.ndarray, bbox: np.ndarray | None = None) -> None:
        keypoints = np.asarray(keypoints, dtype=np.float32)
        self.num_keypoints = int(keypoints.shape[0])
        self.buffers[track_id].append((frame_index, keypoints))
        if bbox is not None:
            self.bbox_buffers[track_id].append((frame_index, np.asarray(bbox, dtype=np.float32)))

    def get(self, track_id: int) -> np.ndarray:
        items = list(self.buffers.get(track_id, []))
        if not items:
            return np.zeros((0, self.num_keypoints, 3), dtype=np.float32)
        items.sort(key=lambda item: item[0])
        arr = []
        for _, keypoints in items:
            keypoints = np.asarray(keypoints, dtype=np.float32)
            if keypoints.shape[-1] == 2:
                scores = np.ones((*keypoints.shape[:1], 1), dtype=np.float32)
                keypoints = np.concatenate([keypoints, scores], axis=-1)
            arr.append(keypoints)
        return np.stack(arr, axis=0)

    def get_bboxes(self, track_id: int) -> np.ndarray:
        items = list(self.bbox_buffers.get(track_id, []))
        if not items:
            return np.zeros((0, 4), dtype=np.float32)
        items.sort(key=lambda item: item[0])
        arr = [np.asarray(bbox, dtype=np.float32) for _, bbox in items]
        return np.stack(arr, axis=0)

    def length(self, track_id: int) -> int:
        return len(self.buffers.get(track_id, []))

    def clear_missing(self, alive_track_ids: List[int]) -> None:
        alive_set = set(alive_track_ids)
        dead = [track_id for track_id in self.buffers if track_id not in alive_set]
        for track_id in dead:
            del self.buffers[track_id]
        dead_bbox = [track_id for track_id in self.bbox_buffers if track_id not in alive_set]
        for track_id in dead_bbox:
            del self.bbox_buffers[track_id]

    def reset(self) -> None:
        self.buffers.clear()
        self.bbox_buffers.clear()
