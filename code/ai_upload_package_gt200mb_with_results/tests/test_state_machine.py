from tpgr.pipeline.state_machine import TemporalCommandFilter


def test_state_machine_stabilizes():
    f = TemporalCommandFilter(window_size=5, min_consensus=3, conf_threshold=0.5, hold_frames=3, expiry_frames=8)
    cmd = "NO_COMMAND"
    for _ in range(3):
        cmd, conf, safe = f.step("STOP", 0.9, occluded=False)
    assert cmd == "STOP"
    cmd, conf, safe = f.step("NO_COMMAND", 0.0, occluded=True)
    assert cmd in {"STOP", "KEEP_WAIT"}
