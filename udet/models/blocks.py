"""U-Net 基础构件（1D 序列版 / 2D 图像版）。

所有构件都保持极简、可读，方便后续魔改：
    * ``DoubleConv`` —— 两层卷积 + BN + 激活；
    * ``Down``       —— 下采样（池化）+ DoubleConv；
    * ``Up``         —— 上采样 + 与跳连拼接 + DoubleConv；
    * ``OutConv``    —— 1x1 卷积输出头。
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_norm(num_features: int, use_batchnorm: bool) -> nn.Module:
    return nn.BatchNorm1d(num_features) if use_batchnorm else nn.Identity()


def _make_norm2d(num_features: int, use_batchnorm: bool) -> nn.Module:
    return nn.BatchNorm2d(num_features) if use_batchnorm else nn.Identity()


# --------------------------------------------------------------------------- #
# 1D（序列）构件
# --------------------------------------------------------------------------- #
class DoubleConv1d(nn.Module):
    """(Conv1d -> [BN] -> ReLU) x 2"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        use_batchnorm: bool = True,
        dropout: float = 0.0,
        mid_channels: Optional[int] = None,
    ) -> None:
        super().__init__()
        mid_channels = mid_channels or out_channels
        padding = kernel_size // 2

        layers = [
            nn.Conv1d(in_channels, mid_channels, kernel_size, padding=padding, bias=not use_batchnorm),
            _make_norm(mid_channels, use_batchnorm),
            nn.ReLU(inplace=True),
            nn.Conv1d(mid_channels, out_channels, kernel_size, padding=padding, bias=not use_batchnorm),
            _make_norm(out_channels, use_batchnorm),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down1d(nn.Module):
    """下采样：MaxPool1d(2) + DoubleConv1d。"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, use_batchnorm: bool = True, dropout: float = 0.0) -> None:
        super().__init__()
        self.pool = nn.MaxPool1d(kernel_size=2, ceil_mode=True)  # ceil_mode 兼容奇数长度
        self.conv = DoubleConv1d(in_channels, out_channels, kernel_size, use_batchnorm, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class Up1d(nn.Module):
    """上采样 + 跳连拼接 + DoubleConv1d。

    先上采样 2 倍，再插值对齐 skip 的实际长度（兼容奇数长度序列），最后拼接。
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        use_batchnorm: bool = True,
        dropout: float = 0.0,
        mode: str = "nearest",
    ) -> None:
        super().__init__()
        self.mode = mode
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = DoubleConv1d(in_channels + skip_channels, out_channels, kernel_size, use_batchnorm, dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.size(-1) != skip.size(-1):  # 对齐长度
            x = F.interpolate(x, size=skip.size(-1), mode=self.mode)
        return self.conv(torch.cat([skip, x], dim=1))


class OutConv1d(nn.Module):
    """1x1 卷积输出头。"""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# --------------------------------------------------------------------------- #
# 2D（图像）构件
# --------------------------------------------------------------------------- #
class DoubleConv2d(nn.Module):
    """(Conv2d -> [BN] -> ReLU) x 2"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        use_batchnorm: bool = True,
        dropout: float = 0.0,
        mid_channels: Optional[int] = None,
    ) -> None:
        super().__init__()
        mid_channels = mid_channels or out_channels
        padding = kernel_size // 2

        layers = [
            nn.Conv2d(in_channels, mid_channels, kernel_size, padding=padding, bias=not use_batchnorm),
            _make_norm2d(mid_channels, use_batchnorm),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size, padding=padding, bias=not use_batchnorm),
            _make_norm2d(out_channels, use_batchnorm),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down2d(nn.Module):
    """下采样：MaxPool2d(2) + DoubleConv2d。"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, use_batchnorm: bool = True, dropout: float = 0.0) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, ceil_mode=True)
        self.conv = DoubleConv2d(in_channels, out_channels, kernel_size, use_batchnorm, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class Up2d(nn.Module):
    """上采样（双线性插值或转置卷积）+ 跳连拼接 + DoubleConv2d。"""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        use_batchnorm: bool = True,
        dropout: float = 0.0,
        bilinear: bool = False,
    ) -> None:
        super().__init__()
        if bilinear:
            self.up: nn.Module = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv2d(in_channels + skip_channels, out_channels, kernel_size, use_batchnorm, dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:  # 对齐尺寸
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=True)
        return self.conv(torch.cat([skip, x], dim=1))


class OutConv2d(nn.Module):
    """1x1 卷积输出头。"""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)
