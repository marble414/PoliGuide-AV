import numpy as np

from tpgr.data.features import infer_tcn_feature_dim, sequence_to_tcn_features


def test_tcn_feature_dims_for_base_and_rich():
    keypoints = np.zeros((8, 17, 3), dtype=np.float32)
    keypoints[..., 2] = 1.0

    base = sequence_to_tcn_features(keypoints, feature_variant="base")
    rich = sequence_to_tcn_features(keypoints, feature_variant="rich")

    assert base.shape == (8, infer_tcn_feature_dim(17, feature_variant="base"))
    assert rich.shape == (8, infer_tcn_feature_dim(17, feature_variant="rich"))
    assert rich.shape[1] > base.shape[1]


def test_wholebody_focus_features_have_expected_dim():
    keypoints = np.zeros((8, 133, 3), dtype=np.float32)
    keypoints[..., 2] = 1.0

    focus = sequence_to_tcn_features(keypoints, feature_variant="wholebody_focus")

    assert focus.shape == (8, infer_tcn_feature_dim(133, feature_variant="wholebody_focus"))
    assert focus.shape[1] > infer_tcn_feature_dim(59, feature_variant="base")
