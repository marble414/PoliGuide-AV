from __future__ import annotations

import torch
import torch.nn as nn

from .skeleton_tcn import AttentionPool1D, ResidualTemporalBlock
from .skeleton_transformer import AttentionPool


class MultiTaskSkeletonTCN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        task_dims: dict[str, int],
        hidden_dim: int = 128,
        num_blocks: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            *[ResidualTemporalBlock(hidden_dim, kernel_size=3, dropout=dropout) for _ in range(num_blocks)]
        )
        self.pool = AttentionPool1D(hidden_dim)
        self.shared = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.heads = nn.ModuleDict({
            task: nn.Linear(hidden_dim, int(num_classes))
            for task, num_classes in task_dims.items()
        })

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError(f"MultiTaskSkeletonTCN 输入必须为 [B, T, F]，实际为 {x.shape}")
        x = x.transpose(1, 2)
        x = self.input_proj(x)
        x = self.blocks(x)
        x = self.pool(x)
        x = self.shared(x)
        return {task: head(x) for task, head in self.heads.items()}


class MultiTaskSkeletonTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        task_dims: dict[str, int],
        d_model: int = 256,
        depth: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.2,
        max_len: int = 128,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=int(d_model * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.pool = AttentionPool(d_model)
        self.shared = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.heads = nn.ModuleDict({
            task: nn.Linear(d_model, int(num_classes))
            for task, num_classes in task_dims.items()
        })

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError(f"MultiTaskSkeletonTransformer 输入必须为 [B, T, F]，实际为 {x.shape}")
        if x.shape[1] > self.pos_embed.shape[1]:
            raise ValueError(f"序列长度 {x.shape[1]} 超过模型最大长度 {self.pos_embed.shape[1]}")
        x = self.input_proj(x)
        x = x + self.pos_embed[:, :x.shape[1]]
        x = self.encoder(x)
        x = self.pool(x)
        x = self.shared(x)
        return {task: head(x) for task, head in self.heads.items()}
