from __future__ import annotations

from typing import Any, Dict

from .skeleton_stgcn import SkeletonSTGCN
from .skeleton_tcn import SkeletonTCN
from .skeleton_transformer import SkeletonTransformer
from .multitask_temporal import MultiTaskSkeletonTCN, MultiTaskSkeletonTransformer
from .prgraph_eps import PRGraphEPSMultiTask
from .frame_segmenter import FrameSegmenterMultiTask
from .frame_unet_segmenter import FrameUNetSegmenterMultiTask
from .frame_gru_segmenter import FrameGRUSegmenterMultiTask


def build_model(model_cfg: Dict[str, Any]):
    name = model_cfg.get("name", "skeleton_tcn").lower()
    task_dims = model_cfg.get("task_dims") or model_cfg.get("heads")
    if name in {"multitask_skeleton_tcn", "skeleton_tcn_multitask"}:
        if not task_dims:
            raise ValueError("多任务 TCN 需要 model.task_dims")
        return MultiTaskSkeletonTCN(
            input_dim=int(model_cfg["input_dim"]),
            task_dims={str(k): int(v) for k, v in task_dims.items()},
            hidden_dim=int(model_cfg.get("hidden_dim", 128)),
            num_blocks=int(model_cfg.get("num_blocks", 4)),
            dropout=float(model_cfg.get("dropout", 0.2)),
        )
    if name in {"multitask_skeleton_transformer", "skeleton_transformer_multitask"}:
        if not task_dims:
            raise ValueError("多任务 Transformer 需要 model.task_dims")
        return MultiTaskSkeletonTransformer(
            input_dim=int(model_cfg["input_dim"]),
            task_dims={str(k): int(v) for k, v in task_dims.items()},
            d_model=int(model_cfg.get("d_model", model_cfg.get("hidden_dim", 256))),
            depth=int(model_cfg.get("depth", 4)),
            num_heads=int(model_cfg.get("num_heads", 8)),
            mlp_ratio=float(model_cfg.get("mlp_ratio", 4.0)),
            dropout=float(model_cfg.get("dropout", 0.2)),
            max_len=int(model_cfg.get("max_len", 128)),
        )
    if name in {"prgraph_eps_multitask", "multitask_prgraph_eps", "position_rotation_graph_multitask"}:
        if not task_dims:
            raise ValueError("PRGraph 多任务模型需要 model.task_dims")
        return PRGraphEPSMultiTask(
            task_dims={str(k): int(v) for k, v in task_dims.items()},
            input_dim=int(model_cfg.get("input_dim", 225)),
            hidden_dim=int(model_cfg.get("hidden_dim", 160)),
            num_blocks=int(model_cfg.get("num_blocks", 4)),
            dropout=float(model_cfg.get("dropout", 0.2)),
            elevation_threshold=float(model_cfg.get("elevation_threshold", 0.15)),
        )
    if name in {"frame_segmenter_multitask", "multitask_frame_segmenter", "temporal_segmenter_multitask"}:
        if not task_dims:
            raise ValueError("FrameSegmenter 多任务模型需要 model.task_dims")
        return FrameSegmenterMultiTask(
            input_dim=int(model_cfg["input_dim"]),
            task_dims={str(k): int(v) for k, v in task_dims.items()},
            hidden_dim=int(model_cfg.get("hidden_dim", 256)),
            num_blocks=int(model_cfg.get("num_blocks", 10)),
            dropout=float(model_cfg.get("dropout", 0.1)),
            causal=bool(model_cfg.get("causal", False)),
        )
    if name in {"frame_segmenter_causal_multitask", "multitask_frame_segmenter_causal", "causal_temporal_segmenter_multitask"}:
        if not task_dims:
            raise ValueError("Causal FrameSegmenter 多任务模型需要 model.task_dims")
        return FrameSegmenterMultiTask(
            input_dim=int(model_cfg["input_dim"]),
            task_dims={str(k): int(v) for k, v in task_dims.items()},
            hidden_dim=int(model_cfg.get("hidden_dim", 256)),
            num_blocks=int(model_cfg.get("num_blocks", 10)),
            dropout=float(model_cfg.get("dropout", 0.1)),
            causal=True,
        )
    if name in {"frame_unet_segmenter_multitask", "temporal_unet_multitask", "multitask_frame_unet_segmenter"}:
        if not task_dims:
            raise ValueError("FrameUNetSegmenter 多任务模型需要 model.task_dims")
        return FrameUNetSegmenterMultiTask(
            input_dim=int(model_cfg["input_dim"]),
            task_dims={str(k): int(v) for k, v in task_dims.items()},
            hidden_dim=int(model_cfg.get("hidden_dim", 192)),
            num_levels=int(model_cfg.get("num_levels", 3)),
            dropout=float(model_cfg.get("dropout", 0.1)),
        )
    if name in {"frame_gru_segmenter_multitask", "temporal_gru_multitask", "multitask_frame_gru_segmenter"}:
        if not task_dims:
            raise ValueError("FrameGRUSegmenter 多任务模型需要 model.task_dims")
        return FrameGRUSegmenterMultiTask(
            input_dim=int(model_cfg["input_dim"]),
            task_dims={str(k): int(v) for k, v in task_dims.items()},
            hidden_dim=int(model_cfg.get("hidden_dim", 384)),
            num_blocks=int(model_cfg.get("num_blocks", 4)),
            gru_layers=int(model_cfg.get("gru_layers", 2)),
            dropout=float(model_cfg.get("dropout", 0.1)),
            bidirectional=bool(model_cfg.get("bidirectional", True)),
        )
    num_classes = int(model_cfg["num_classes"])
    if name == "skeleton_tcn":
        return SkeletonTCN(
            input_dim=int(model_cfg["input_dim"]),
            num_classes=num_classes,
            hidden_dim=int(model_cfg.get("hidden_dim", 128)),
            num_blocks=int(model_cfg.get("num_blocks", 4)),
            dropout=float(model_cfg.get("dropout", 0.2)),
        )
    if name == "skeleton_stgcn":
        return SkeletonSTGCN(
            in_channels=int(model_cfg.get("in_channels", 3)),
            num_classes=num_classes,
            hidden_dim=int(model_cfg.get("hidden_dim", 64)),
            num_nodes=int(model_cfg.get("num_nodes", 17)),
            dropout=float(model_cfg.get("dropout", 0.2)),
        )
    if name == "skeleton_transformer":
        return SkeletonTransformer(
            input_dim=int(model_cfg["input_dim"]),
            num_classes=num_classes,
            d_model=int(model_cfg.get("d_model", model_cfg.get("hidden_dim", 256))),
            depth=int(model_cfg.get("depth", 4)),
            num_heads=int(model_cfg.get("num_heads", 8)),
            mlp_ratio=float(model_cfg.get("mlp_ratio", 4.0)),
            dropout=float(model_cfg.get("dropout", 0.2)),
            max_len=int(model_cfg.get("max_len", 128)),
        )
    raise ValueError(f"不支持的模型名称: {name}")
