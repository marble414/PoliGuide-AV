from tpgr.eval_continuous import build_dense_targets, summarize_sequence_metrics


def test_build_dense_targets_defaults_to_background_between_segments():
    segments = [
        {"frame_start": 2, "frame_end": 4, "label": "STOP", "direction": "F"},
        {"frame_start": 7, "frame_end": 8, "label": "TURN_LEFT", "direction": "L"},
    ]
    gestures, commands, directions = build_dense_targets(segments, frame_count=10, dataset_name="ctpgesture_v2")
    assert gestures[:2] == ["NO_GESTURE", "NO_GESTURE"]
    assert gestures[2:5] == ["STOP", "STOP", "STOP"]
    assert commands[7:9] == ["TURN_LEFT", "TURN_LEFT"]
    assert directions[0] == "unknown"
    assert directions[7] == "L"


def test_summarize_sequence_metrics_has_continuous_fields():
    gt = ["NO_GESTURE", "STOP", "STOP", "TURN_LEFT", "TURN_LEFT"]
    pred = ["NO_GESTURE", "STOP", "NO_GESTURE", "TURN_LEFT", "TURN_LEFT"]
    summary = summarize_sequence_metrics(
        gt,
        pred,
        class_names=["NO_GESTURE", "STOP", "TURN_LEFT"],
        background_label="NO_GESTURE",
    )
    assert summary["frames"] == 5
    assert "macro_jaccard" in summary
    assert "foreground_macro_jaccard" in summary
    assert "sequence_edit_similarity" in summary
    assert "stable_output_latency_frames" in summary
