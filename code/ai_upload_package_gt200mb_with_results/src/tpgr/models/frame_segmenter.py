from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TimewiseLayerNorm1D(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.transpose(1, 2)).transpose(1, 2).contiguous()


class ResidualDilatedBlock1D(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float = 0.1, causal: bool = False) -> None:
        super().__init__()
        self.dilation = int(dilation)
        self.causal = bool(causal)
        self.conv1 = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            padding=0 if self.causal else self.dilation,
            dilation=dilation,
            bias=False,
        )
        self.norm1 = TimewiseLayerNorm1D(channels) if self.causal else nn.GroupNorm(num_groups=8, num_channels=channels)
        self.conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            padding=0 if self.causal else self.dilation,
            dilation=dilation,
            bias=False,
        )
        self.norm2 = TimewiseLayerNorm1D(channels) if self.causal else nn.GroupNorm(num_groups=8, num_channels=channels)
        self.dropout = nn.Dropout(dropout)

    def _apply_conv(self, conv: nn.Conv1d, x: torch.Tensor) -> torch.Tensor:
        if self.causal:
            x = F.pad(x, (2 * self.dilation, 0))
        return conv(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self._apply_conv(self.conv1, x)
        y = F.gelu(self.norm1(y))
        y = self.dropout(y)
        y = self._apply_conv(self.conv2, y)
        y = self.norm2(y)
        return F.gelu(x + self.dropout(y))


class FrameSegmenterMultiTask(nn.Module):
    def __init__(
        self,
        input_dim: int,
        task_dims: dict[str, int],
        hidden_dim: int = 256,
        num_blocks: int = 10,
        dropout: float = 0.1,
        causal: bool = False,
    ) -> None:
        super().__init__()
        input_norm: nn.Module = TimewiseLayerNorm1D(hidden_dim) if causal else nn.GroupNorm(num_groups=8, num_channels=hidden_dim)
        self.input_proj = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=1, bias=False),
            input_norm,
            nn.GELU(),
        )
        dilations = [2 ** (idx % max(1, num_blocks)) for idx in range(num_blocks)]
        self.blocks = nn.ModuleList(
            [
                ResidualDilatedBlock1D(
                    hidden_dim,
                    dilation=d,
                    dropout=dropout,
                    causal=causal,
                )
                for d in dilations
            ]
        )
        self.heads = nn.ModuleDict(
            {task: nn.Conv1d(hidden_dim, int(num_classes), kernel_size=1) for task, num_classes in task_dims.items()}
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError(f"FrameSegmenterMultiTask 需要 [B, T, F]，实际为 {tuple(x.shape)}")
        y = x.transpose(1, 2).contiguous()
        y = self.input_proj(y)
        for block in self.blocks:
            y = block(y)
        return {task: head(y) for task, head in self.heads.items()}
