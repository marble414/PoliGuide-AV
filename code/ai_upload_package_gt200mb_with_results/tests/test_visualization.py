import numpy as np

from tpgr.data.schema import CommandOutput
from tpgr.io.visualization import draw_result_panel


def test_draw_result_panel_debug_widens_frame():
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    result = CommandOutput(
        frame_index=3,
        timestamp_sec=0.1,
        active_track_id=1,
        gesture="STOP",
        gesture_confidence=0.9,
        intent="HALT",
        command="STOP",
        command_confidence=0.9,
        safe_fallback=False,
        latency_ms=8.5,
        fps=30.0,
        debug={
            "min_predict_len": 12,
            "active_track": {
                "buffer_len": 16,
                "ready_for_prediction": True,
                "track_hits": 5,
                "occluded": False,
                "visible_keypoints": 17,
                "mean_keypoint_score": 0.88,
                "selector_score": 0.91,
            },
            "raw_prediction": {
                "scores": {"STOP": 0.9, "NO_GESTURE": 0.1},
            },
            "state_machine": {
                "current_command": "STOP",
                "current_confidence": 0.9,
                "conf_threshold": 0.6,
                "stable_vote": {"command": "STOP", "confidence": 0.9},
                "history": [{"command": "STOP", "confidence": 0.9}],
                "window_size": 9,
                "votes": {"STOP": 1},
            },
            "tracks": [
                {
                    "track_id": 1,
                    "selector_score": 0.91,
                    "track_hits": 5,
                    "visible_keypoints": 17,
                    "selector_components": {
                        "center": 0.9,
                        "size": 0.4,
                        "visibility": 0.95,
                        "track_age": 0.5,
                        "vest": 0.1,
                    },
                }
            ],
            "evaluation": {
                "clip_index": 2,
                "num_clips": 10,
                "sample_id": "clip_0002",
                "ground_truth": "STOP",
                "frame_prediction": "STOP",
                "clip_prediction": "STOP",
                "clip_correct": True,
                "clip_frames": 16,
                "running_accuracy": 0.75,
            },
        },
    )

    vis = draw_result_panel(frame, result, debug=True, topk=3, panel_width=360)
    assert vis.shape[0] == frame.shape[0]
    assert vis.shape[1] == frame.shape[1] + 360
