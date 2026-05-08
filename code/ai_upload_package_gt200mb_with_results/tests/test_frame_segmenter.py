from __future__ import annotations

import torch

from tpgr.models.frame_segmenter import FrameSegmenterMultiTask


def test_frame_segmenter_multitask_output_shapes() -> None:
    model = FrameSegmenterMultiTask(
        input_dim=225,
        task_dims={"gesture": 33, "command": 9, "direction": 5},
        hidden_dim=64,
        num_blocks=4,
        dropout=0.1,
    )
    x = torch.randn(2, 128, 225)
    outputs = model(x)
    assert outputs["gesture"].shape == (2, 33, 128)
    assert outputs["command"].shape == (2, 9, 128)
    assert outputs["direction"].shape == (2, 5, 128)
