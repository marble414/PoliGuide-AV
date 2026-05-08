from tpgr.data.labels import (
    get_task_classes,
    directional_gesture_name,
    gesture_to_command,
    gesture_to_direction,
    get_label_space_classes,
    normalize_raw_label,
    safe_hold_command,
)


def test_label_mapping_basic():
    assert normalize_raw_label(1, "ctpgesture_v1") == "STOP"
    assert normalize_raw_label("forward", "ctpgesture_v1") == "GO_STRAIGHT"
    assert normalize_raw_label(9, "ctpgesture_v2") == "STOP"
    assert gesture_to_command("LEFT_TURN_WAIT") == "LEFT_TURN_WAIT"
    assert safe_hold_command("GO_STRAIGHT") == "KEEP_WAIT"


def test_ctpv2_directional_label_space():
    classes = get_label_space_classes("ctpv2_directional")
    assert len(classes) == 33
    assert len(get_task_classes("command", "ctpv2_directional")) == 9
    assert len(get_task_classes("direction", "ctpv2_directional")) == 5
    assert normalize_raw_label(0, "ctpgesture_v2", label_space="ctpv2_directional") == "NO_GESTURE"
    assert normalize_raw_label(1, "ctpgesture_v2", label_space="ctpv2_directional") == "F_STOP"
    assert normalize_raw_label(32, "ctpgesture_v2", label_space="ctpv2_directional") == "R_PULL_OVER"
    assert directional_gesture_name("L", "TURN_LEFT") == "L_TURN_LEFT"
    assert gesture_to_command("B_LEFT_TURN_WAIT") == "LEFT_TURN_WAIT"
    assert gesture_to_direction("B_LEFT_TURN_WAIT") == "B"
