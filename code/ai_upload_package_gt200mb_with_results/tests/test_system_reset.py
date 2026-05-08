import json
from pathlib import Path

import numpy as np

from tpgr.pipeline.system import TrafficPoliceGestureSystem


def test_system_reset_runtime_state(tmp_path: Path):
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
            "min_predict_len": 2,
            "pose_backend": {"type": "precomputed", "jsonl_path": str(jsonl_path)},
            "tracker": {"iou_threshold": 0.3, "max_age": 5, "min_hits": 1},
            "selector": {},
            "state_machine": {"window_size": 5, "min_consensus": 3, "conf_threshold": 0.5, "hold_frames": 2, "expiry_frames": 6},
        },
        "classifier": {"type": "rule_based"},
    }

    system = TrafficPoliceGestureSystem(cfg)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    system.process_frame(frame, fps=10.0)
    system.process_frame(frame, fps=10.0)
    assert system.frame_index == 2
    assert system.buffer.length(1) >= 1

    system.reset_runtime_state()

    assert system.frame_index == 0
    assert system.buffer.length(1) == 0
    assert system.buffer.get_bboxes(1).shape[0] == 0
    assert system.selector.last_track_id is None
    assert not system.tracker.tracks
