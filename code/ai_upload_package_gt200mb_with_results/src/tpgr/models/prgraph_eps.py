from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .skeleton_tcn import AttentionPool1D

# VIBE / SPIN commonly expose 49 joints as [25 OpenPose | 24 SMPL].
# For position-rotation coupling we use the trailing 24 SMPL joints so they
# align with the 24 axis-angle pose vectors in `pose(72)`.
SMPL24_EDGES = [
    (0, 1), (0, 2), (0, 3),
    (1, 4), (2, 5), (3, 6),
    (4, 7), (5, 8), (6, 9),
    (7, 10), (8, 11), (9, 12),
    (12, 13), (12, 14), (12, 15),
    (13, 16), (14, 17),
    (16, 18), (17, 19),
    (18, 20), (19, 21),
    (20, 22), (21, 23),
]


def build_smpl24_adjacency(num_nodes: int = 24) -> torch.Tensor:
    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    for node in range(num_nodes):
        adj[node, node] = 1.0
    for src, dst in SMPL24_EDGES:
        adj[src, dst] = 1.0
        adj[dst, src] = 1.0
    deg = adj.sum(dim=-1, keepdim=True).clamp(min=1.0)
    return adj / deg


def _split_vibe_features(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if x.ndim != 3:
        raise ValueError(f"PRGraph 输入必须为 [B, T, F]，实际为 {x.shape}")
    if x.shape[-1] < 225:
        raise ValueError(f"PRGraph 需要至少 225 维 VIBE 特征，实际为 {x.shape[-1]}")
    joints49 = x[..., :147].reshape(x.shape[0], x.shape[1], 49, 3)
    pose24 = x[..., 147:219].reshape(x.shape[0], x.shape[1], 24, 3)
    center_rel = x[..., 219:222]
    center_vel = x[..., 222:225]
    smpl24 = joints49[..., -24:, :]
    return smpl24, pose24, center_rel, center_vel


class ElevationPartitionContext(nn.Module):
    def __init__(self, channels: int, threshold: float = 0.15, dropout: float = 0.1) -> None:
        super().__init__()
        self.threshold = float(threshold)
        self.proj = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, features: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        # features: [B, C, T, V]
        # positions: [B, T, V, 3]
        rel_y = positions[..., 1] - positions[..., :1, 1]
        bins = torch.zeros_like(rel_y, dtype=torch.long)
        bins = torch.where(rel_y > self.threshold, torch.full_like(bins, 2), bins)
        bins = torch.where(rel_y.abs() <= self.threshold, torch.full_like(bins, 1), bins)
        one_hot = F.one_hot(bins, num_classes=3).float()

        feat_btvc = features.permute(0, 2, 3, 1).contiguous()
        counts = one_hot.sum(dim=2, keepdim=False).clamp(min=1.0).unsqueeze(-1)
        contexts = torch.einsum("btvc,btvp->btpc", feat_btvc, one_hot) / counts
        contexts = self.proj(contexts)
        lifted = torch.einsum("btpc,btvp->btvc", contexts, one_hot)
        return features + lifted.permute(0, 3, 1, 2).contiguous()


class GraphTemporalBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.self_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.neigh_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.norm = nn.BatchNorm2d(out_channels)
        self.temporal = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.residual = (
            nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        neigh = torch.einsum("nctv,vw->nctw", x, adj)
        y = self.self_proj(x) + self.neigh_proj(neigh)
        y = F.gelu(self.norm(y))
        y = self.temporal(y)
        return F.gelu(y + self.residual(x))


class CrossStreamFusion(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.pos_to_rot = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.rot_to_pos = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels * 2, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, pos: torch.Tensor, rot: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gates = self.gate(torch.cat([pos, rot], dim=1))
        pos_gate, rot_gate = gates.chunk(2, dim=1)
        pos = pos + pos_gate * self.rot_to_pos(rot)
        rot = rot + rot_gate * self.pos_to_rot(pos)
        return pos, rot


class PositionRotationGraphBlock(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.2, elevation_threshold: float = 0.15) -> None:
        super().__init__()
        self.pos_block = GraphTemporalBlock(channels, channels, dropout=dropout)
        self.rot_block = GraphTemporalBlock(channels, channels, dropout=dropout)
        self.pos_eps = ElevationPartitionContext(channels, threshold=elevation_threshold, dropout=dropout)
        self.rot_eps = ElevationPartitionContext(channels, threshold=elevation_threshold, dropout=dropout)
        self.fusion = CrossStreamFusion(channels)

    def forward(self, pos: torch.Tensor, rot: torch.Tensor, adj: torch.Tensor, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pos = self.pos_block(pos, adj)
        rot = self.rot_block(rot, adj)
        pos = self.pos_eps(pos, positions)
        rot = self.rot_eps(rot, positions)
        pos, rot = self.fusion(pos, rot)
        return pos, rot


class MotionHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )
        self.block = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )
        self.pool = AttentionPool1D(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.proj(x)
        x = self.block(x)
        return self.pool(x)


class PRGraphEPSMultiTask(nn.Module):
    def __init__(
        self,
        task_dims: dict[str, int],
        input_dim: int = 225,
        hidden_dim: int = 160,
        num_blocks: int = 4,
        dropout: float = 0.2,
        elevation_threshold: float = 0.15,
    ) -> None:
        super().__init__()
        if int(input_dim) < 225:
            raise ValueError("PRGraphEPSMultiTask 需要 225 维 VIBE 特征输入")
        self.register_buffer("adj", build_smpl24_adjacency())
        self.pos_input = nn.Sequential(
            nn.Conv2d(7, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
        )
        self.rot_input = nn.Sequential(
            nn.Conv2d(9, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList([
            PositionRotationGraphBlock(hidden_dim, dropout=dropout, elevation_threshold=elevation_threshold)
            for _ in range(num_blocks)
        ])
        self.motion_head = MotionHead(input_dim=9, hidden_dim=hidden_dim, dropout=dropout)
        self.partition_head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 3),
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        fused_dim = hidden_dim * 4
        self.shared = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.heads = nn.ModuleDict({
            task: nn.Linear(hidden_dim, int(num_classes))
            for task, num_classes in task_dims.items()
        })

    def _partition_summary(self, fused: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        rel_y = positions[..., 1] - positions[..., :1, 1]
        bins = torch.zeros_like(rel_y, dtype=torch.long)
        bins = torch.where(rel_y > 0.15, torch.full_like(bins, 2), bins)
        bins = torch.where(rel_y.abs() <= 0.15, torch.full_like(bins, 1), bins)
        one_hot = F.one_hot(bins, num_classes=3).float()
        feat_btvc = fused.permute(0, 2, 3, 1).contiguous()
        counts = one_hot.sum(dim=2).clamp(min=1.0).unsqueeze(-1)
        contexts = torch.einsum("btvc,btvp->btpc", feat_btvc, one_hot) / counts
        contexts = contexts.mean(dim=1).reshape(fused.shape[0], -1)
        return self.partition_head(contexts)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        smpl24, pose24, center_rel, center_vel = _split_vibe_features(x)
        pos_vel = torch.zeros_like(smpl24)
        pos_vel[:, 1:] = smpl24[:, 1:] - smpl24[:, :-1]
        root_rel = smpl24 - smpl24[..., :1, :]
        rot_vel = torch.zeros_like(pose24)
        rot_vel[:, 1:] = pose24[:, 1:] - pose24[:, :-1]
        root_rot = pose24[..., :1, :].expand(-1, -1, pose24.shape[2], -1)

        pos_feat = torch.cat([root_rel, pos_vel, torch.linalg.norm(root_rel, dim=-1, keepdim=True)], dim=-1)
        rot_feat = torch.cat([pose24, rot_vel, root_rot], dim=-1)

        pos = self.pos_input(pos_feat.permute(0, 3, 1, 2).contiguous())
        rot = self.rot_input(rot_feat.permute(0, 3, 1, 2).contiguous())

        for block in self.blocks:
            pos, rot = block(pos, rot, self.adj, smpl24)

        fused = 0.5 * (pos + rot)
        global_pool = fused.mean(dim=-1)
        global_pool = global_pool.mean(dim=-1)
        pos_pool = pos.mean(dim=-1).mean(dim=-1)
        rot_pool = rot.mean(dim=-1).mean(dim=-1)
        motion = self.motion_head(torch.cat([center_rel, center_vel, pose24[..., 0, :]], dim=-1))
        partition = self._partition_summary(fused, smpl24)

        summary = torch.cat([
            0.5 * (pos_pool + rot_pool),
            global_pool,
            motion,
            partition,
        ], dim=-1)
        summary = self.shared(summary)
        return {task: head(summary) for task, head in self.heads.items()}
