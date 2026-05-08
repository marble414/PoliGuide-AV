import numpy as np
import pytest

from tpgr.pipeline.pose_backends import OpenMMLabPoseBackend


def test_openmmlab_parse_instances_flattens_bbox_and_uses_bbox_score():
    backend = OpenMMLabPoseBackend.__new__(OpenMMLabPoseBackend)
    result = {
        "predictions": [[{
            "bbox": ([1.0, 2.0, 30.0, 40.0],),
            "bbox_score": np.float32(0.9),
            "keypoints": [[10.0, 11.0], [12.0, 13.0]],
            "keypoint_scores": [0.2, 0.3],
        }]]
    }

    parsed = backend._parse_instances(result)

    assert parsed[0]["bbox"] == [1.0, 2.0, 30.0, 40.0]
    assert parsed[0]["score"] == pytest.approx(0.9)
    assert parsed[0]["keypoints"] == [[10.0, 11.0], [12.0, 13.0]]
    assert parsed[0]["keypoint_scores"] == pytest.approx([0.2, 0.3])
    assert parsed[0]["occluded"] is False
