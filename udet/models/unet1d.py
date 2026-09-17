"""1D U-Net：面向 **token 序列** 的 U-Net（编码器输出 -> 逐 token 预测）。

典型用法（与编码器串联，后续在此基础魔改）::

    encoder_out = encoder(input_ids, attention_mask)          # (B, L, D)
    x = encoder_out.last_hidden_state.transpose(1, 2)         # (B, D, L)
    logits = unet(x, mask=encoder_out.attention_mask)         # (B, num_classes, L)

设计说明：
    * 输入 ``(B, C, L)``，输出 ``(B, num_classes, L)``，长度与输入对齐；
    * 奇数长度序列通过 ``ceil_mode`` 池化 + 插值对齐，不会报错；
    * 支持 ``deep_supervision``（深监督），返回多尺度 logits 列表；
    * ``mask`` 传入时（可选）会把 padding 位置的 logits 置零，避免影响后续处理。
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import DoubleConv1d, Down1d, OutConv1d, Up1d


class UNet1D(nn.Module):
    """序列版 U-Net。

    Args:
        in_channels: 输入特征维度（通常等于编码器 ``hidden_size``，如 CodeT5-base 为 768）。
        out_channels: 输出类别数。
        features: 各层级通道数，例如 ``(64, 128, 256, 512)`` 表示 3 次下采样。
        kernel_size: 卷积核大小（默认 3）。
        use_batchnorm: 是否使用 BatchNorm。
        dropout: dropout 概率。
        deep_supervision: 是否启用深监督（返回各解码层 logits 列表）。
        apply_mask: 传入 mask 时是否将 padding 位置输出置零。
    """

    def __init__(
        self,
        in_channels: int = 768,
        out_channels: int = 2,
        features: Sequence[int] = (64, 128, 256, 512),
        kernel_size: int = 3,
        use_batchnorm: bool = True,
        dropout: float = 0.0,
        deep_supervision: bool = False,
        apply_mask: bool = True,
    ) -> None:
        super().__init__()
        features = tuple(int(f) for f in features)
        if len(features) < 2:
            raise ValueError("features 至少需要 2 个层级")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.features = features
        self.depth = len(features) - 1
        self.deep_supervision = deep_supervision
        self.apply_mask = apply_mask

        # ---- 编码器（下采样）路径 ----
        self.inc = DoubleConv1d(in_channels, features[0], kernel_size, use_batchnorm, dropout)
        self.downs = nn.ModuleList(
            [
                Down1d(features[i], features[i + 1], kernel_size, use_batchnorm, dropout)
                for i in range(self.depth)
            ]
        )

        # ---- 解码器（上采样）路径 ----
        self.ups = nn.ModuleList(
            [
                Up1d(features[i], features[i - 1], features[i - 1], kernel_size, use_batchnorm, dropout)
                for i in range(self.depth, 0, -1)
            ]
        )

        # ---- 输出头 ----
        self.outc = OutConv1d(features[0], out_channels)
        if deep_supervision:
            # 每个解码层一个辅助头（层索引与 features 对齐）
            self.aux_heads = nn.ModuleList(
                [OutConv1d(features[i], out_channels) for i in range(self.depth, 0, -1)]
            )

        self._init_weights()

    # ------------------------------------------------------------------ #
    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d,)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------ #
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_features: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """前向计算。

        Args:
            x: ``(B, C, L)`` 输入序列特征。
            mask: ``(B, L)`` 可选 mask（1 为有效 token）。仅在 ``apply_mask=True`` 时用于置零输出。
            return_features: 为 True 时同时返回各解码层特征（参考多尺度特征用）。

        Returns:
            ``logits`` (B, out_channels, L)；
            若 ``deep_supervision=True``，返回 ``List[logits_scale...]``（第一个为最终输出）；
            若 ``return_features=True``，返回 ``(logits, features)``。
        """
        input_len = x.size(-1)
        decoder_feats: List[torch.Tensor] = []

        # ---- 编码器路径（保留下采样前的特征作为跳连）----
        x = self.inc(x)
        skips = []
        for down in self.downs:
            skips.append(x)
            x = down(x)

        # ---- 解码器路径 ----
        for up, skip in zip(self.ups, reversed(skips)):
            x = up(x, skip)
            decoder_feats.append(x)

        # ---- 输出 ----
        logits = self.outc(x)

        # 长度对齐（奇数长度经过 ceil_mode 池化后可能差 1 个位置）
        if logits.size(-1) != input_len:
            logits = F.interpolate(logits, size=input_len, mode="nearest")

        if self.apply_mask and mask is not None:
            logits = logits * mask.unsqueeze(1).to(logits.dtype)

        if self.deep_supervision:
            aux_logits = []
            for feat, head in zip(decoder_feats, self.aux_heads):
                aux = head(feat)
                if aux.size(-1) != input_len:
                    aux = F.interpolate(aux, size=input_len, mode="nearest")
                if self.apply_mask and mask is not None:
                    aux = aux * mask.unsqueeze(1).to(aux.dtype)
                aux_logits.append(aux)
            aux_logits.append(logits)
            return (aux_logits, decoder_feats) if return_features else aux_logits

        return (logits, decoder_feats) if return_features else logits


def build_unet1d(**kwargs) -> UNet1D:
    """便捷构建函数（供配置文件统一调用）。"""
    return UNet1D(**kwargs)
