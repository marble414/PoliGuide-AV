from __future__ import annotations

import torch
import torch.nn as nn

COCO17_ADJACENCY = [
    (0, 0), (1, 1), (2, 2), (3, 3), (4, 4),
    (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10),
    (11, 11), (12, 12), (13, 13), (14, 14), (15, 15), (16, 16),
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
]


def build_adjacency(num_nodes: int = 17) -> torch.Tensor:
    A = torch.zeros(num_nodes, num_nodes, dtype=torch.float32)
    for i, j in COCO17_ADJACENCY:
        A[i, j] = 1.0
        A[j, i] = 1.0
    deg = A.sum(dim=-1, keepdim=True).clamp(min=1.0)
    A = A / deg
    return A


class STGCNBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.gcn = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.tcn = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(out_channels),
        )
        self.residual = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        y = torch.einsum("nctv,vw->nctw", x, A)
        y = self.gcn(y)
        y = self.tcn(y)
        res = self.residual(x)
        return self.relu(y + res)


class SkeletonSTGCN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        hidden_dim: int = 64,
        num_nodes: int = 17,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.register_buffer("A", build_adjacency(num_nodes))
        self.data_bn = nn.BatchNorm1d(in_channels * num_nodes)
        self.blocks = nn.ModuleList([
            STGCNBlock(in_channels, hidden_dim, dropout=dropout),
            STGCNBlock(hidden_dim, hidden_dim, dropout=dropout),
            STGCNBlock(hidden_dim, hidden_dim * 2, dropout=dropout),
        ])
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"SkeletonSTGCN 输入必须为 [B, C, T, V]，实际为 {x.shape}")
        n, c, t, v = x.shape
        x = x.permute(0, 3, 1, 2).contiguous().view(n, v * c, t)
        x = self.data_bn(x)
        x = x.view(n, v, c, t).permute(0, 2, 3, 1).contiguous()
        for block in self.blocks:
            x = block(x, self.A)
        x = x.mean(dim=-1).mean(dim=-1)
        return self.head(x)
