from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ResidualConvBlock(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(num_groups=8, num_channels=channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(num_groups=8, num_channels=channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(x)
        y = F.gelu(self.norm1(y))
        y = self.dropout(y)
        y = self.conv2(y)
        y = self.norm2(y)
        return F.gelu(x + self.dropout(y))


class _DownsampleStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(x)


class _UpsampleStage(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv1d(in_channels + skip_channels, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=out_channels),
            nn.GELU(),
        )
        self.block1 = _ResidualConvBlock(out_channels, dropout=dropout)
        self.block2 = _ResidualConvBlock(out_channels, dropout=dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-1], mode="nearest")
        x = torch.cat([x, skip], dim=1)
        x = self.proj(x)
        x = self.block1(x)
        x = self.block2(x)
        return x


class FrameUNetSegmenterMultiTask(nn.Module):
    def __init__(
        self,
        input_dim: int,
        task_dims: dict[str, int],
        hidden_dim: int = 192,
        num_levels: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if num_levels < 2:
            raise ValueError("FrameUNetSegmenterMultiTask 至少需要 2 个 level")
        self.input_proj = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=hidden_dim),
            nn.GELU(),
        )

        channels = [hidden_dim * (2 ** level) for level in range(num_levels)]
        self.enc_blocks = nn.ModuleList()
        self.down_blocks = nn.ModuleList()
        for level, ch in enumerate(channels):
            self.enc_blocks.append(nn.Sequential(
                _ResidualConvBlock(ch, dropout=dropout),
                _ResidualConvBlock(ch, dropout=dropout),
            ))
            if level < num_levels - 1:
                self.down_blocks.append(_DownsampleStage(ch, channels[level + 1]))

        self.bottleneck = nn.Sequential(
            _ResidualConvBlock(channels[-1], dropout=dropout),
            _ResidualConvBlock(channels[-1], dropout=dropout),
        )

        self.up_blocks = nn.ModuleList()
        for level in range(num_levels - 1, 0, -1):
            self.up_blocks.append(
                _UpsampleStage(
                    in_channels=channels[level],
                    skip_channels=channels[level - 1],
                    out_channels=channels[level - 1],
                    dropout=dropout,
                )
            )

        self.heads = nn.ModuleDict(
            {task: nn.Conv1d(channels[0], int(num_classes), kernel_size=1) for task, num_classes in task_dims.items()}
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError(f"FrameUNetSegmenterMultiTask 需要 [B, T, F]，实际为 {tuple(x.shape)}")
        y = x.transpose(1, 2).contiguous()
        y = self.input_proj(y)

        skips: list[torch.Tensor] = []
        for level, block in enumerate(self.enc_blocks):
            y = block(y)
            skips.append(y)
            if level < len(self.down_blocks):
                y = self.down_blocks[level](y)

        y = self.bottleneck(y)
        for level, up in enumerate(self.up_blocks):
            skip = skips[-(level + 2)]
            y = up(y, skip)

        return {task: head(y) for task, head in self.heads.items()}
