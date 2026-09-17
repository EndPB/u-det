"""U-Net（单文件实现，1D / 2D 通用）。

    dim=1：输入 (B, C, L) —— 用于 token 序列（C 通常为编码器 hidden_size）
    dim=2：输入 (B, C, H, W) —— 用于图像 / 特征图

典型用法（U-Det：逐 token 编码器 + U-Net + 双任务头）::

    feats = encoder(input_ids, attention_mask)                        # (B, L, 768)
    logits, scales, bottleneck = unet(feats.transpose(1, 2), mask=attention_mask, return_features=True)
    # scales：各解码层特征，由粗到细 [(B,256,L/4), (B,128,L/2), (B,64,L)]，分辨率不回拉
    # bottleneck：最底部特征（可选 mid Transformer 交互之后）(B, 512, L/8)

结构：DoubleConv -> [Down x N] -> [mid] -> [Up x N] -> 1x1 卷积
魔改入口：DoubleConv（卷积块）、Down（下采样）、Up（跳连）、mid（瓶颈模块，如 transformer）、UNet.forward（整体）
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# 按维度选择基础算子（1D / 2D 共用同一套结构）
# --------------------------------------------------------------------------- #
def _conv_cls(dim: int):
    return nn.Conv1d if dim == 1 else nn.Conv2d


def _norm_cls(dim: int):
    return nn.BatchNorm1d if dim == 1 else nn.BatchNorm2d


def _drop_cls(dim: int):
    return nn.Dropout if dim == 1 else nn.Dropout2d


def _max_pool(dim: int, kernel_size: int = 2) -> nn.Module:
    # ceil_mode：兼容奇数长度序列 / 非 2 次幂尺寸
    return nn.MaxPool1d(kernel_size, ceil_mode=True) if dim == 1 else nn.MaxPool2d(kernel_size, ceil_mode=True)


def _resize_like(x: torch.Tensor, size, dim: int, bilinear: bool = False) -> torch.Tensor:
    """把 x 插值到目标尺寸（1D 用长度，2D 用 (H, W)）。"""
    if dim == 1:
        if x.shape[-1] == size:
            return x
        return F.interpolate(x, size=size, mode="nearest")
    if tuple(x.shape[-2:]) == tuple(size):
        return x
    return F.interpolate(x, size=size, mode="bilinear", align_corners=True)


def downsample_mask(mask: torch.Tensor, dim: int, depth: int) -> torch.Tensor:
    """把 (B, L) 有效位 mask 按 depth 次 2 倍池化下采样（与各 Down 对齐，ceil_mode）。"""
    m = mask.float()
    for _ in range(depth):
        m = (F.max_pool1d if dim == 1 else F.max_pool2d)(m, 2, 2, ceil_mode=True)
    return m


# --------------------------------------------------------------------------- #
# 构件
# --------------------------------------------------------------------------- #
class DoubleConv(nn.Module):
    """两层卷积 + [BN] + ReLU（U-Net 的基础块）。"""

    def __init__(
        self,
        dim: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        use_batchnorm: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        Conv, Norm = _conv_cls(dim), _norm_cls(dim)
        padding = kernel_size // 2
        layers: List[nn.Module] = []
        for ch_in, ch_out in ((in_channels, out_channels), (out_channels, out_channels)):
            layers.append(Conv(ch_in, ch_out, kernel_size, padding=padding, bias=not use_batchnorm))
            layers.append(Norm(ch_out) if use_batchnorm else nn.Identity())
            layers.append(nn.ReLU(inplace=True))
        if dropout > 0:
            layers.append(_drop_cls(dim)(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    """下采样：池化 + DoubleConv。"""

    def __init__(self, dim: int, in_channels: int, out_channels: int, kernel_size: int = 3,
                 use_batchnorm: bool = True, dropout: float = 0.0):
        super().__init__()
        self.pool = _max_pool(dim, 2)
        self.conv = DoubleConv(dim, in_channels, out_channels, kernel_size, use_batchnorm, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class Up(nn.Module):
    """上采样 + 跳连拼接 + DoubleConv，自动对齐跳连尺寸。"""

    def __init__(self, dim: int, in_channels: int, skip_channels: int, out_channels: int,
                 kernel_size: int = 3, use_batchnorm: bool = True, dropout: float = 0.0,
                 bilinear: bool = False):
        super().__init__()
        self.dim = dim
        # 2D 且非双线性时用转置卷积；其余情况统一用插值（对奇数尺寸更稳）
        if dim == 2 and not bilinear:
            self.up: nn.Module = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        else:
            self.up = nn.Identity()
        self.conv = DoubleConv(dim, in_channels + skip_channels, out_channels, kernel_size, use_batchnorm, dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        size = skip.shape[-1] if self.dim == 1 else skip.shape[-2:]
        if isinstance(self.up, nn.Identity):
            x = _resize_like(x, size, self.dim)
        else:
            x = self.up(x)
            if tuple(x.shape[-2:]) != tuple(size):
                x = F.interpolate(x, size=size, mode="bilinear", align_corners=True)
        return self.conv(torch.cat([skip, x], dim=1))


# --------------------------------------------------------------------------- #
# U-Net
# --------------------------------------------------------------------------- #
class UNet(nn.Module):
    """U-Net（1D 序列 / 2D 图像通用）。

    Args:
        dim: 1 = 序列（输入 (B,C,L)）；2 = 图像（输入 (B,C,H,W)）。
        in_channels: 输入通道数（1D 时一般等于编码器 hidden_size，如 CodeT5-base 768）。
        out_channels: 输出通道数 / 类别数；None = 不建输出头（只当特征提取器用，必须 return_features=True）。
        features: 各层级通道数，如 [64, 128, 256, 512]（len-1 = 下采样次数）。
        kernel_size: 卷积核大小。
        use_batchnorm: 是否使用 BatchNorm。
        dropout: Dropout 概率。
        bilinear: 2D 时是否用双线性插值上采样（False 用转置卷积）。
        deep_supervision: True 时返回各解码层 logits 列表（多尺度监督）。
        mid: 瓶颈模块（如 models/transformer.py 的 MidTransformer，1D 专用），None 则不插。
    """

    def __init__(
        self,
        dim: int = 1,
        in_channels: int = 768,
        out_channels: Optional[int] = 2,
        features: Sequence[int] = (64, 128, 256, 512),
        kernel_size: int = 3,
        use_batchnorm: bool = True,
        dropout: float = 0.0,
        bilinear: bool = False,
        deep_supervision: bool = False,
        mid: Optional[nn.Module] = None,
    ):
        super().__init__()
        features = tuple(int(f) for f in features)
        if dim not in (1, 2):
            raise ValueError("dim 只能是 1 或 2")
        if len(features) < 2:
            raise ValueError("features 至少需要 2 个层级")
        if mid is not None and dim != 1:
            raise ValueError("mid 瓶颈模块目前只支持 1D 序列")

        self.dim = dim
        self.depth = len(features) - 1
        self.out_channels = out_channels
        self.features = features
        self.deep_supervision = deep_supervision
        self.mid = mid
        Conv = _conv_cls(dim)

        self.inc = DoubleConv(dim, in_channels, features[0], kernel_size, use_batchnorm, dropout)
        self.downs = nn.ModuleList(
            [Down(dim, features[i], features[i + 1], kernel_size, use_batchnorm, dropout)
             for i in range(self.depth)]
        )
        self.ups = nn.ModuleList(
            [Up(dim, features[i], features[i - 1], features[i - 1], kernel_size, use_batchnorm, dropout, bilinear)
             for i in range(self.depth, 0, -1)]
        )
        self.outc = Conv(features[0], out_channels, kernel_size=1) if out_channels else None
        if out_channels and deep_supervision:
            self.aux_heads = nn.ModuleList(
                [Conv(features[i], out_channels, kernel_size=1) for i in range(self.depth - 1, -1, -1)]
            )

        self._init_weights()

    def _init_weights(self) -> None:
        Conv = _conv_cls(self.dim)
        Norm = _norm_cls(self.dim)
        for m in self.modules():
            if isinstance(m, Conv):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, Norm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_features: bool = False,
    ) -> Union[torch.Tensor, List[torch.Tensor], Tuple]:
        """前向计算。

        Args:
            x: (B, C, L)（dim=1）或 (B, C, H, W)（dim=2）。
            mask: (B, L) 可选，1D 时把 padding 位置的输出置零，并传给 mid 模块。
            return_features: True 时额外返回各解码层特征与瓶颈特征。

        Returns:
            logits (B, out_channels, L/H/W)；deep_supervision=True 时为其列表；
            return_features=True 时在最后追加 (scales, bottleneck)，
            scales 为各解码层特征（由粗到细、保持原生分辨率），bottleneck 为最底部特征；
            若 out_channels=None 则 logits 为 None（只看特征）。
        """
        if self.outc is None and not return_features:
            raise ValueError("out_channels=None 的 U-Net 只能配合 return_features=True 使用")
        size = x.shape[-1] if self.dim == 1 else x.shape[-2:]

        x = self.inc(x)
        skips = []
        for down in self.downs:
            skips.append(x)
            x = down(x)

        if self.mid is not None:  # 瓶颈：双向语义交互（在压缩后的序列上做全局注意力）
            bottleneck = self.mid(x, downsample_mask(mask, self.dim, self.depth) if mask is not None else None)
        else:
            bottleneck = x
        x = bottleneck

        decoder_feats: List[torch.Tensor] = []
        for up, skip in zip(self.ups, reversed(skips)):
            x = up(x, skip)
            decoder_feats.append(x)

        logits = None
        if self.outc is not None:
            logits = _resize_like(self.outc(x), size, self.dim)
            if mask is not None and self.dim == 1:
                logits = logits * mask.unsqueeze(1).to(logits.dtype)

        if self.deep_supervision:
            outs = [_resize_like(head(f), size, self.dim) for f, head in zip(decoder_feats, self.aux_heads)]
            if mask is not None and self.dim == 1:
                outs = [o * mask.unsqueeze(1).to(o.dtype) for o in outs]
            outs.append(logits)
            return (outs, decoder_feats, bottleneck) if return_features else outs

        return (logits, decoder_feats, bottleneck) if return_features else logits


def build_unet(name: str = "unet1d", **kwargs) -> UNet:
    """按名称构建 U-Net（'unet1d' 序列 / 'unet2d' 图像），与编码器切换方式一致。"""
    variants = {"unet1d": 1, "unet2d": 2}
    if name not in variants:
        raise KeyError(f"未知网络 {name!r}，可选：{sorted(variants)}")
    return UNet(dim=variants[name], **kwargs)
