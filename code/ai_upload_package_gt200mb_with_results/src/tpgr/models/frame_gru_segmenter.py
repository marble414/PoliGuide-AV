from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ResidualDilatedConvBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            bias=False,
        )
        self.norm1 = nn.GroupNorm(num_groups=8, num_channels=channels)
        self.conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            bias=False,
        )
        self.norm2 = nn.GroupNorm(num_groups=8, num_channels=channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(x)
        y = F.gelu(self.norm1(y))
        y = self.dropout(y)
        y = self.conv2(y)
        y = self.norm2(y)
        return F.gelu(x + self.dropout(y))


class FrameGRUSegmenterMultiTask(nn.Module):
    def __init__(
        self,
        input_dim: int,
        task_dims: dict[str, int],
        hidden_dim: int = 384,
        num_blocks: int = 4,
        gru_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        if hidden_dim % 2 != 0 and bidirectional:
            raise ValueError("bidirectional GRU 需要偶数 hidden_dim")
        self.input_proj = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=hidden_dim),
            nn.GELU(),
        )
        self.local_blocks = nn.ModuleList(
            [
                _ResidualDilatedConvBlock(
                    hidden_dim,
                    dilation=2 ** (idx % max(1, num_blocks)),
                    dropout=dropout,
                )
                for idx in range(num_blocks)
            ]
        )
        gru_hidden = hidden_dim // 2 if bidirectional else hidden_dim
        self.pre_gru_norm = nn.LayerNorm(hidden_dim)
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=gru_hidden,
            num_layers=max(1, int(gru_layers)),
            batch_first=True,
            dropout=dropout if int(gru_layers) > 1 else 0.0,
            bidirectional=bool(bidirectional),
        )
        self.gru_proj = nn.Linear(hidden_dim, hidden_dim)
        self.post_gru_norm = nn.LayerNorm(hidden_dim)
        self.post_gru_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.heads = nn.ModuleDict(
            {task: nn.Conv1d(hidden_dim, int(num_classes), kernel_size=1) for task, num_classes in task_dims.items()}
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError(f"FrameGRUSegmenterMultiTask 需要 [B, T, F]，实际为 {tuple(x.shape)}")
        y = x.transpose(1, 2).contiguous()
        y = self.input_proj(y)
        for block in self.local_blocks:
            y = block(y)

        seq = y.transpose(1, 2).contiguous()
        seq = self.pre_gru_norm(seq)
        gru_out, _ = self.gru(seq)
        seq = seq + self.dropout(self.gru_proj(gru_out))
        seq = seq + self.dropout(self.post_gru_ffn(self.post_gru_norm(seq)))
        y = seq.transpose(1, 2).contiguous()
        return {task: head(y) for task, head in self.heads.items()}
