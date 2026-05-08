import numpy as np

from tpgr.data.features import adapt_sequence_length, bbox_sequence_to_track_features, infer_tcn_feature_dim, resample_sequence, sequence_to_tcn_features


def test_resample_sequence_preserves_endpoints():
    arr = np.arange(5, dtype=np.float32)[:, None]
    out = resample_sequence(arr, 9)

    assert out.shape == (9, 1)
    assert np.isclose(out[0, 0], 0.0)
    assert np.isclose(out[-1, 0], 4.0)


def test_adapt_sequence_length_resample_handles_empty():
    arr = np.zeros((0, 3), dtype=np.float32)
    out = adapt_sequence_length(arr, 7, training=False, mode="resample")

    assert out.shape == (7, 3)
    assert np.allclose(out, 0.0)


def test_adapt_sequence_length_hybrid_eval_uses_resample():
    arr = np.arange(10, dtype=np.float32)[:, None]
    out = adapt_sequence_length(arr, 6, training=False, mode="hybrid")

    assert out.shape == (6, 1)
    assert np.isclose(out[0, 0], 0.0)
    assert np.isclose(out[-1, 0], 9.0)


def test_adapt_sequence_length_tail_pad_uses_latest_window():
    arr = np.arange(10, dtype=np.float32)[:, None]
    out = adapt_sequence_length(arr, 6, training=False, mode="tail_pad")

    assert out.shape == (6, 1)
    assert np.allclose(out[:, 0], np.arange(4, 10, dtype=np.float32))


def test_bbox_sequence_to_track_features_shape():
    bbox = np.asarray(
        [
            [10, 20, 30, 40],
            [11, 22, 30, 40],
            [13, 25, 31, 41],
        ],
        dtype=np.float32,
    )
    feats = bbox_sequence_to_track_features(bbox)

    assert feats.shape == (3, 14)


def test_sequence_to_tcn_features_wholebody_focus_track_appends_track_dims():
    keypoints = np.zeros((5, 133, 3), dtype=np.float32)
    keypoints[..., 2] = 1.0
    bbox = np.asarray(
        [
            [10, 20, 30, 40],
            [11, 20, 30, 40],
            [12, 21, 31, 41],
            [13, 23, 32, 41],
            [14, 25, 32, 42],
        ],
        dtype=np.float32,
    )
    feats = sequence_to_tcn_features(keypoints, feature_variant="wholebody_focus_track", bbox_track=bbox)

    assert feats.shape[0] == 5
    assert feats.shape[1] == infer_tcn_feature_dim(133, feature_variant="wholebody_focus_track")
