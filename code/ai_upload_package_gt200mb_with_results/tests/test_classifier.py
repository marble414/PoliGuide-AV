import numpy as np
import torch

from tpgr.pipeline.classifier import BaseClassifier, EnsembleGestureClassifier, Prediction, TorchGestureClassifier


class _MockClassifier(BaseClassifier):
    def __init__(self, label: str, scores: dict[str, float]) -> None:
        self.label = label
        self.scores = scores

    def predict_sequence(self, keypoints: np.ndarray, bbox_track: np.ndarray | None = None) -> Prediction:
        return Prediction(label=self.label, confidence=self.scores[self.label], scores=self.scores)


class _MockMultitaskClassifier(BaseClassifier):
    def __init__(
        self,
        label: str,
        scores: dict[str, float],
        gesture_scores_raw: dict[str, float],
        command_scores: dict[str, float],
        direction_scores: dict[str, float],
    ) -> None:
        self.label = label
        self.scores = scores
        self.class_names = list(scores.keys())
        self.label_space = "canonical"
        self.command_index_to_label = {idx: name for idx, name in enumerate(command_scores.keys())}
        self.direction_index_to_label = {idx: name for idx, name in enumerate(direction_scores.keys())}
        self.command_label_to_index = {name: idx for idx, name in self.command_index_to_label.items()}
        self.direction_label_to_index = {name: idx for idx, name in self.direction_index_to_label.items()}
        self._extras = {
            "gesture_scores_raw": gesture_scores_raw,
            "command_scores": command_scores,
            "direction_scores": direction_scores,
            "command_label": max(command_scores, key=command_scores.get),
            "direction_label": max(direction_scores, key=direction_scores.get),
        }

    def predict_sequence(self, keypoints: np.ndarray, bbox_track: np.ndarray | None = None) -> Prediction:
        return Prediction(
            label=self.label,
            confidence=self.scores[self.label],
            scores=self.scores,
            extras=self._extras,
        )


def test_ensemble_classifier_weighted_average():
    keypoints = np.zeros((16, 17, 3), dtype=np.float32)
    clf = EnsembleGestureClassifier(
        members=[
            _MockClassifier("STOP", {"STOP": 0.8, "NO_GESTURE": 0.2}),
            _MockClassifier("NO_GESTURE", {"STOP": 0.3, "NO_GESTURE": 0.7}),
        ],
        weights=[0.7, 0.3],
    )

    pred = clf.predict_sequence(keypoints)

    assert pred.label == "STOP"
    assert pred.scores["STOP"] > pred.scores["NO_GESTURE"]


def test_ensemble_classifier_preserves_weighted_multitask_extras():
    keypoints = np.zeros((16, 17, 3), dtype=np.float32)
    clf = EnsembleGestureClassifier(
        members=[
            _MockMultitaskClassifier(
                "STOP",
                {"STOP": 0.8, "NO_GESTURE": 0.2},
                {"STOP": 0.7, "NO_GESTURE": 0.3},
                {"STOP": 0.9, "NO_COMMAND": 0.1},
                {"UNKNOWN": 0.2, "FRONT": 0.8},
            ),
            _MockMultitaskClassifier(
                "NO_GESTURE",
                {"STOP": 0.4, "NO_GESTURE": 0.6},
                {"STOP": 0.2, "NO_GESTURE": 0.8},
                {"STOP": 0.3, "NO_COMMAND": 0.7},
                {"UNKNOWN": 0.6, "FRONT": 0.4},
            ),
        ],
        weights=[0.75, 0.25],
    )

    pred = clf.predict_sequence(keypoints)

    assert pred.extras is not None
    assert pred.extras["gesture_scores_raw"]["STOP"] > pred.extras["gesture_scores_raw"]["NO_GESTURE"]
    assert abs(pred.extras["gesture_scores_raw"]["STOP"] - 0.575) < 1e-6
    assert abs(pred.extras["command_scores"]["STOP"] - 0.75) < 1e-6
    assert pred.extras["command_label"] == "STOP"
    assert abs(pred.extras["direction_scores"]["FRONT"] - 0.7) < 1e-6
    assert pred.extras["direction_label"] == "FRONT"


def test_torch_classifier_tta_smoke(tmp_path):
    checkpoint = tmp_path / "model.pt"
    model_cfg = {
        "name": "skeleton_tcn",
        "input_dim": 85,
        "num_classes": 9,
        "hidden_dim": 32,
        "num_blocks": 2,
        "dropout": 0.1,
        "amp": False,
    }
    from tpgr.models.builder import build_model

    model = build_model(model_cfg)
    torch.save({"model": model.state_dict()}, checkpoint)

    clf = TorchGestureClassifier(
        model_cfg=model_cfg,
        checkpoint=checkpoint,
        representation="features",
        clip_len=48,
        device="cpu",
        feature_variant="base",
        tta_num_clips=3,
    )
    keypoints = np.zeros((64, 17, 3), dtype=np.float32)
    keypoints[..., 2] = 1.0

    pred = clf.predict_sequence(keypoints)

    assert pred.label in pred.scores
    assert abs(sum(pred.scores.values()) - 1.0) < 1e-4


def test_torch_classifier_resample_temporal_mode_smoke(tmp_path):
    checkpoint = tmp_path / "model.pt"
    model_cfg = {
        "name": "skeleton_tcn",
        "input_dim": 85,
        "num_classes": 9,
        "hidden_dim": 32,
        "num_blocks": 2,
        "dropout": 0.1,
        "amp": False,
    }
    from tpgr.models.builder import build_model

    model = build_model(model_cfg)
    torch.save({"model": model.state_dict()}, checkpoint)

    clf = TorchGestureClassifier(
        model_cfg=model_cfg,
        checkpoint=checkpoint,
        representation="features",
        clip_len=64,
        device="cpu",
        feature_variant="base",
        tta_num_clips=1,
        temporal_mode="resample",
    )
    keypoints = np.zeros((17, 17, 3), dtype=np.float32)
    keypoints[..., 2] = 1.0

    pred = clf.predict_sequence(keypoints)

    assert pred.label in pred.scores
    assert abs(sum(pred.scores.values()) - 1.0) < 1e-4


def test_torch_classifier_tail_pad_temporal_mode_smoke(tmp_path):
    checkpoint = tmp_path / "model.pt"
    model_cfg = {
        "name": "skeleton_tcn",
        "input_dim": 85,
        "num_classes": 9,
        "hidden_dim": 32,
        "num_blocks": 2,
        "dropout": 0.1,
        "amp": False,
    }
    from tpgr.models.builder import build_model

    model = build_model(model_cfg)
    torch.save({"model": model.state_dict()}, checkpoint)

    clf = TorchGestureClassifier(
        model_cfg=model_cfg,
        checkpoint=checkpoint,
        representation="features",
        clip_len=48,
        device="cpu",
        feature_variant="base",
        tta_num_clips=1,
        temporal_mode="tail_pad",
    )
    keypoints = np.zeros((96, 17, 3), dtype=np.float32)
    keypoints[..., 2] = 1.0

    pred = clf.predict_sequence(keypoints)

    assert pred.label in pred.scores
    assert abs(sum(pred.scores.values()) - 1.0) < 1e-4


def test_torch_classifier_precomputed_feature_sequence_and_directional_labels(tmp_path):
    checkpoint = tmp_path / "model.pt"
    model_cfg = {
        "name": "skeleton_tcn",
        "input_dim": 219,
        "num_classes": 33,
        "hidden_dim": 32,
        "num_blocks": 2,
        "dropout": 0.1,
        "amp": False,
    }
    from tpgr.models.builder import build_model

    model = build_model(model_cfg)
    torch.save({"model": model.state_dict()}, checkpoint)

    clf = TorchGestureClassifier(
        model_cfg=model_cfg,
        checkpoint=checkpoint,
        representation="features",
        clip_len=64,
        device="cpu",
        tta_num_clips=2,
        label_space="ctpv2_directional",
    )
    features = np.zeros((96, 219), dtype=np.float32)

    pred = clf.predict_sequence(features)

    assert pred.label in pred.scores
    assert len(pred.scores) == 33
    assert abs(sum(pred.scores.values()) - 1.0) < 1e-4


def test_torch_classifier_multitask_fusion_smoke(tmp_path):
    checkpoint = tmp_path / "model.pt"
    model_cfg = {
        "name": "multitask_skeleton_tcn",
        "input_dim": 225,
        "num_classes": 33,
        "task_dims": {"gesture": 33, "command": 9, "direction": 5},
        "hidden_dim": 32,
        "num_blocks": 2,
        "dropout": 0.1,
        "amp": False,
    }
    from tpgr.models.builder import build_model

    model = build_model(model_cfg)
    torch.save({"model": model.state_dict()}, checkpoint)

    clf = TorchGestureClassifier(
        model_cfg=model_cfg,
        checkpoint=checkpoint,
        representation="features",
        clip_len=64,
        device="cpu",
        label_space="ctpv2_directional",
        task_fusion={"command_weight": 0.3, "direction_weight": 0.2},
    )
    features = np.zeros((96, 225), dtype=np.float32)

    pred = clf.predict_sequence(features)

    assert pred.label in pred.scores
    assert pred.extras is not None
    assert "command_label" in pred.extras
    assert "direction_label" in pred.extras


def test_torch_classifier_frame_segmenter_multitask_smoke(tmp_path):
    checkpoint = tmp_path / "segmenter.pt"
    model_cfg = {
        "name": "frame_segmenter_multitask",
        "input_dim": 85,
        "task_dims": {"gesture": 9, "command": 9, "direction": 5},
        "hidden_dim": 32,
        "num_blocks": 3,
        "dropout": 0.1,
        "amp": False,
    }
    from tpgr.models.builder import build_model

    model = build_model(model_cfg)
    torch.save({"model": model.state_dict()}, checkpoint)

    clf = TorchGestureClassifier(
        model_cfg=model_cfg,
        checkpoint=checkpoint,
        representation="features",
        clip_len=64,
        device="cpu",
        label_space="canonical",
        task_fusion={"command_weight": 0.2, "direction_weight": 0.0},
    )
    features = np.zeros((96, 85), dtype=np.float32)
    pred = clf.predict_sequence(features)

    assert pred.label in pred.scores
    assert pred.extras is not None
    assert "gesture_scores_raw" in pred.extras
    assert "command_scores" in pred.extras


def test_torch_classifier_frame_segmenter_wholebody_keypoints_smoke(tmp_path):
    checkpoint = tmp_path / "segmenter_wholebody.pt"
    model_cfg = {
        "name": "frame_segmenter_multitask",
        "input_dim": 456,
        "task_dims": {"gesture": 33, "command": 9, "direction": 5},
        "hidden_dim": 32,
        "num_blocks": 3,
        "dropout": 0.1,
        "amp": False,
    }
    from tpgr.models.builder import build_model

    model = build_model(model_cfg)
    torch.save({"model": model.state_dict()}, checkpoint)

    clf = TorchGestureClassifier(
        model_cfg=model_cfg,
        checkpoint=checkpoint,
        representation="features",
        clip_len=96,
        device="cpu",
        feature_variant="wholebody_focus",
        label_space="ctpv2_directional",
        task_fusion={"command_weight": 0.2, "direction_weight": 0.1},
    )
    keypoints = np.zeros((120, 133, 3), dtype=np.float32)
    keypoints[..., 2] = 1.0

    pred = clf.predict_sequence(keypoints)

    assert pred.label in pred.scores
    assert len(pred.scores) == 33
    assert pred.extras is not None
    assert "gesture_scores_raw" in pred.extras
    assert "direction_scores" in pred.extras


def test_torch_classifier_gesture_bias_changes_prediction(tmp_path):
    checkpoint = tmp_path / "segmenter.pt"
    model_cfg = {
        "name": "frame_segmenter_multitask",
        "input_dim": 85,
        "task_dims": {"gesture": 9, "command": 9, "direction": 5},
        "hidden_dim": 32,
        "num_blocks": 3,
        "dropout": 0.1,
        "amp": False,
    }
    from tpgr.models.builder import build_model

    model = build_model(model_cfg)
    torch.save({"model": model.state_dict()}, checkpoint)

    baseline = TorchGestureClassifier(
        model_cfg=model_cfg,
        checkpoint=checkpoint,
        representation="features",
        clip_len=64,
        device="cpu",
        label_space="canonical",
    )
    biased = TorchGestureClassifier(
        model_cfg=model_cfg,
        checkpoint=checkpoint,
        representation="features",
        clip_len=64,
        device="cpu",
        label_space="canonical",
        gesture_bias={"STOP": 5.0},
    )
    features = np.zeros((96, 85), dtype=np.float32)
    base_pred = baseline.predict_sequence(features)
    biased_pred = biased.predict_sequence(features)

    assert biased_pred.scores["STOP"] > base_pred.scores["STOP"]
