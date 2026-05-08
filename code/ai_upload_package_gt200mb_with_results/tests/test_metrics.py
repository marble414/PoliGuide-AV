from tpgr.metrics.sequence import (
    classwise_jaccard,
    command_switch_delay,
    edit_distance,
    macro_jaccard,
    sequence_edit_similarity,
    stable_output_latency,
)


def test_edit_distance():
    assert edit_distance(["A", "B"], ["A", "C"]) == 1


def test_sequence_metrics():
    gt = ["NO_COMMAND"] * 3 + ["STOP"] * 5 + ["GO_STRAIGHT"] * 4
    pred = ["NO_COMMAND"] * 4 + ["STOP"] * 4 + ["GO_STRAIGHT"] * 4
    assert 0.0 <= sequence_edit_similarity(pred, gt) <= 1.0
    assert command_switch_delay(gt, pred) >= 0.0
    assert stable_output_latency(gt, pred) >= 0.0


def test_stable_output_latency_supports_custom_background():
    gt = ["NO_GESTURE"] * 2 + ["STOP"] * 4
    pred = ["NO_GESTURE"] * 3 + ["STOP"] * 3
    assert stable_output_latency(gt, pred, background_label="NO_GESTURE") >= 0.0


def test_jaccard_metrics():
    gt = ["NO_GESTURE", "STOP", "STOP", "TURN_LEFT"]
    pred = ["NO_GESTURE", "STOP", "TURN_LEFT", "TURN_LEFT"]
    scores = classwise_jaccard(pred, gt, ["NO_GESTURE", "STOP", "TURN_LEFT"])
    assert scores["NO_GESTURE"] == 1.0
    assert 0.0 <= scores["STOP"] <= 1.0
    assert 0.0 <= macro_jaccard(pred, gt, ["NO_GESTURE", "STOP", "TURN_LEFT"]) <= 1.0
