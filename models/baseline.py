"""直接二分类基线（单文件实现）：编码器 + masked 池化 + 线性分类头。

与 U-Det 的差别只在"是否用 U-Net + 双任务"：这里把编码器输出直接池化成句向量再分类，
是最常规的 CodeT5 二分类基线（编码器可冻结、也可微调）。

用法::

    from models import PooledClassifier

    clf = PooledClassifier(encoder, pooling="mean", hidden=None, out=2)
    logits, _ = clf(input_ids, attention_mask)      # (B, 2)；第二个返回值恒为 None（无 token 头）
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class PooledClassifier(nn.Module):
    """编码器 + masked 池化 + 分类头（无 U-Net、无 token 级任务）。

    Args:
        encoder: 任意满足 ``forward(input_ids, attention_mask) -> (B, L, D)`` 的编码器。
        dim: 特征维度（默认取 ``encoder.hidden_size``）。
        pooling: ``mean``（masked 平均）/ ``max``（masked 最大）/ ``first``（取首 token）。
        hidden: 分类头隐层维度，None 表示直接线性投影。
        out: 类别数。
        dropout: dropout 概率。
    """

    #: 序列长度补齐倍数（与 UDet 的接口保持一致，基线不做多尺度池化，取 8 即可）
    align_multiple = 8

    def __init__(
        self,
        encoder: nn.Module,
        dim: Optional[int] = None,
        pooling: str = "mean",
        hidden: Optional[int] = None,
        out: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        if pooling not in ("mean", "max", "first"):
            raise ValueError(f"未知池化方式 {pooling!r}，可选：mean / max / first")
        self.encoder = encoder
        self.pooling = pooling
        dim = int(dim or encoder.hidden_size)
        self.norm = nn.LayerNorm(dim)
        layer = [] if not hidden else [nn.Linear(dim, int(hidden)), nn.GELU(), nn.Dropout(dropout)]
        layer += [nn.Linear(int(hidden) if hidden else dim, out)]
        self.net = nn.Sequential(*layer)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        hidden = self.encoder(input_ids, attention_mask)                 # (B, L, D)
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)             # (B, L, 1)
        if self.pooling == "mean":
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        elif self.pooling == "max":
            pooled = (hidden * mask + (mask - 1) * 1e4).max(1).values    # padding 位置压到 -1e4
        else:
            pooled = hidden[:, 0]
        return self.net(self.norm(pooled)), None                         # 与 UDet 保持同接口
