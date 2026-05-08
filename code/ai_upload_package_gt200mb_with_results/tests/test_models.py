import torch

from tpgr.models.frame_segmenter import FrameSegmenterMultiTask
from tpgr.models.skeleton_stgcn import SkeletonSTGCN
from tpgr.models.skeleton_tcn import SkeletonTCN
from tpgr.models.skeleton_transformer import SkeletonTransformer
from tpgr.models.multitask_temporal import MultiTaskSkeletonTCN
from tpgr.models.prgraph_eps import PRGraphEPSMultiTask
from tpgr.models.frame_unet_segmenter import FrameUNetSegmenterMultiTask


def test_tcn_forward():
    model = SkeletonTCN(input_dim=85, num_classes=9)
    x = torch.randn(2, 48, 85)
    y = model(x)
    assert y.shape == (2, 9)


def test_stgcn_forward():
    model = SkeletonSTGCN(in_channels=3, num_classes=9, num_nodes=17)
    x = torch.randn(2, 3, 48, 17)
    y = model(x)
    assert y.shape == (2, 9)


def test_transformer_forward():
    model = SkeletonTransformer(input_dim=665, num_classes=9, d_model=128, depth=2, num_heads=4, max_len=64)
    x = torch.randn(2, 48, 665)
    y = model(x)
    assert y.shape == (2, 9)


def test_multitask_tcn_forward():
    model = MultiTaskSkeletonTCN(
        input_dim=225,
        task_dims={"gesture": 33, "command": 9, "direction": 5},
        hidden_dim=64,
        num_blocks=2,
        dropout=0.1,
    )
    x = torch.randn(2, 64, 225)
    out = model(x)
    assert set(out.keys()) == {"gesture", "command", "direction"}
    assert out["gesture"].shape == (2, 33)
    assert out["command"].shape == (2, 9)
    assert out["direction"].shape == (2, 5)


def test_prgraph_multitask_forward():
    model = PRGraphEPSMultiTask(
        input_dim=225,
        task_dims={"gesture": 33, "command": 9, "direction": 5},
        hidden_dim=64,
        num_blocks=2,
        dropout=0.1,
    )
    x = torch.randn(2, 64, 225)
    out = model(x)
    assert set(out.keys()) == {"gesture", "command", "direction"}
    assert out["gesture"].shape == (2, 33)
    assert out["command"].shape == (2, 9)
    assert out["direction"].shape == (2, 5)


def test_frame_unet_segmenter_forward():
    model = FrameUNetSegmenterMultiTask(
        input_dim=225,
        task_dims={"gesture": 33, "command": 9, "direction": 5},
        hidden_dim=64,
        num_levels=3,
        dropout=0.1,
    )
    x = torch.randn(2, 128, 225)
    out = model(x)
    assert set(out.keys()) == {"gesture", "command", "direction"}
    assert out["gesture"].shape == (2, 33, 128)
    assert out["command"].shape == (2, 9, 128)
    assert out["direction"].shape == (2, 5, 128)


def test_frame_segmenter_causal_forward():
    model = FrameSegmenterMultiTask(
        input_dim=225,
        task_dims={"gesture": 33, "command": 9, "direction": 5},
        hidden_dim=64,
        num_blocks=3,
        dropout=0.1,
        causal=True,
    )
    x = torch.randn(2, 128, 225)
    out = model(x)
    assert set(out.keys()) == {"gesture", "command", "direction"}
    assert out["gesture"].shape == (2, 33, 128)
    assert out["command"].shape == (2, 9, 128)
    assert out["direction"].shape == (2, 5, 128)


def test_frame_segmenter_causal_is_prefix_invariant():
    torch.manual_seed(0)
    model = FrameSegmenterMultiTask(
        input_dim=8,
        task_dims={"gesture": 4},
        hidden_dim=16,
        num_blocks=2,
        dropout=0.0,
        causal=True,
    )
    model.eval()
    x = torch.randn(1, 24, 8)
    x_future_changed = x.clone()
    x_future_changed[:, 12:, :] = torch.randn_like(x_future_changed[:, 12:, :])
    y1 = model(x)["gesture"]
    y2 = model(x_future_changed)["gesture"]
    assert torch.allclose(y1[..., :12], y2[..., :12], atol=1e-6)
