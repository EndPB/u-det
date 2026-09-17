"""2D U-Net：经典图像分割版本（Ronneberger et al., 2015 风格）。

输入 ``(B, C, H, W)``，输出 ``(B, num_classes, H, W)``，尺寸自动对齐。
保留标准结构，方便后续在此基础上魔改（例如换成注意力跳连、条件调制等）。
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import DoubleConv2d, Down2d, OutConv2d, Up2d


class UNet2D(nn.Module):
    """经典 2D U-Net。

    Args:
        in_channels: 输入通道数。
        out_channels: 输出类别数。
        features: 各层级通道数，例如 ``(64, 128, 256, 512)``。
        kernel_size: 卷积核大小。
        use_batchnorm: 是否使用 BatchNorm。
        dropout: dropout 概率。
        bilinear: True 使用双线性插值上采样，False 使用转置卷积。
        deep_supervision: 是否返回深监督 logits 列表。
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 2,
        features: Sequence[int] = (64, 128, 256, 512),
        kernel_size: int = 3,
        use_batchnorm: bool = True,
        dropout: float = 0.0,
        bilinear: bool = False,
        deep_supervision: bool = False,
    ) -> None:
        super().__init__()
        features = tuple(int(f) for f in features)
        if len(features) < 2:
            raise ValueError("features 至少需要 2 个层级")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.features = features
        self.depth = len(features) - 1
        self.bilinear = bilinear
        self.deep_supervision = deep_supervision

        # ---- 编码器路径 ----
        self.inc = DoubleConv2d(in_channels, features[0], kernel_size, use_batchnorm, dropout)
        self.downs = nn.ModuleList(
            [
                Down2d(features[i], features[i + 1], kernel_size, use_batchnorm, dropout)
                for i in range(self.depth)
            ]
        )

        # ---- 解码器路径 ----
        self.ups = nn.ModuleList(
            [
                Up2d(
                    features[i],
                    features[i - 1],
                    features[i - 1],
                    kernel_size,
                    use_batchnorm,
                    dropout,
                    bilinear=bilinear,
                )
                for i in range(self.depth, 0, -1)
            ]
        )

        # ---- 输出头 ----
        self.outc = OutConv2d(features[0], out_channels)
        if deep_supervision:
            self.aux_heads = nn.ModuleList(
                [OutConv2d(features[i], out_channels) for i in range(self.depth, 0, -1)]
            )

        self._init_weights()

    # ------------------------------------------------------------------ #
    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d,)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------ #
    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
    ) -> Union[torch.Tensor, List[torch.Tensor], tuple]:
        """前向计算，返回 ``(B, out_channels, H, W)``（或深监督 logits 列表）。"""
        input_size = x.shape[-2:]
        decoder_feats: List[torch.Tensor] = []

        x = self.inc(x)
        skips = []
        for down in self.downs:
            skips.append(x)
            x = down(x)

        for up, skip in zip(self.ups, reversed(skips)):
            x = up(x, skip)
            decoder_feats.append(x)

        logits = self.outc(x)
        if logits.shape[-2:] != input_size:
            logits = F.interpolate(logits, size=input_size, mode="bilinear", align_corners=True)

        if self.deep_supervision:
            aux_logits = []
            for feat, head in zip(decoder_feats, self.aux_heads):
                aux = head(feat)
                if aux.shape[-2:] != input_size:
                    aux = F.interpolate(aux, size=input_size, mode="bilinear", align_corners=True)
                aux_logits.append(aux)
            aux_logits.append(logits)
            return (aux_logits, decoder_feats) if return_features else aux_logits

        return (logits, decoder_feats) if return_features else logits


def build_unet2d(**kwargs) -> UNet2D:
    """便捷构建函数。"""
    return UNet2D(**kwargs)
