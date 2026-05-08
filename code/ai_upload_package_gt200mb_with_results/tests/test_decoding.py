import numpy as np

from tpgr.decoding import OnlineViterbiDecoder


def test_online_viterbi_decoder_prefers_stay_on_brief_noise():
    class_names = ["NO_GESTURE", "F_STOP", "L_STOP"]
    log_start = np.log(np.array([0.1, 0.8, 0.1], dtype=np.float64))
    transition = np.array(
        [
            [0.85, 0.10, 0.05],
            [0.05, 0.93, 0.02],
            [0.05, 0.02, 0.93],
        ],
        dtype=np.float64,
    )
    decoder = OnlineViterbiDecoder(
        class_names=class_names,
        log_start=log_start,
        log_transition=np.log(transition),
        emission_scale=1.0,
    )
    sequence = [
        np.array([0.05, 0.90, 0.05], dtype=np.float32),
        np.array([0.05, 0.85, 0.10], dtype=np.float32),
        np.array([0.05, 0.20, 0.75], dtype=np.float32),
        np.array([0.05, 0.90, 0.05], dtype=np.float32),
    ]
    labels = [decoder.step(probs)[0] for probs in sequence]
    assert labels == ["F_STOP", "F_STOP", "F_STOP", "F_STOP"]
