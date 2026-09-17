"""任务头（单文件实现）：样本级 + token 级（多尺度）。

    SampleHead  接在 U-Net 瓶颈（最底部），masked 平均池化 -> MLP -> 二分类 logits
    TokenHeads  接在各解码层 + 瓶颈特征，每尺度 1x1 卷积 -> 每 token 1 个 logit

用法::

    from models import SampleHead, TokenHeads

    sample_head = SampleHead(dim=512, hidden=256)                    # (B,512,L/8) -> (B,2)
    token_heads = TokenHeads(list(reversed(features)))               # [(B,512,L/8), ..., (B,64,L)] -> [(B,1,·)]
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch
import torch.nn as nn


class SampleHead(nn.Module):
    """样本级二分类头：masked 平均池化 + MLP。

    Args:
        dim: 输入通道数（= U-Net features[-1]，瓶颈通道数）。
        hidden: MLP 隐层维度（None 取 dim）。
        out: 类别数。
        dropout: dropout 概率。
    """

    def __init__(self, dim: int, hidden: Optional[int] = None, out: int = 2, dropout: float = 0.0):
        super().__init__()
        hidden = int(hidden or dim)
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, out)
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """(B, C, L) -> (B, out)；mask (B, L) 为 1/0 有效位标记（可选）。"""
        if mask is None:
            pooled = x.mean(dim=-1)
        else:
            w = mask.unsqueeze(1).to(x.dtype)
            pooled = (x * w).sum(dim=-1) / w.sum(dim=-1).clamp(min=1.0)
        return self.net(self.norm(pooled))


class TokenHeads(nn.Module):
    """token 级分类头：每个尺度一个 1x1 卷积（多尺度深监督用）。

    Args:
        channels: 各尺度输入通道数，**由粗到细**，如 [512, 256, 128, 64]。
        out: 输出通道数（二分类取 1，配合 BCEWithLogits）。
    """

    def __init__(self, channels: Sequence[int], out: int = 1):
        super().__init__()
        self.channels = tuple(int(c) for c in channels)
        self.heads = nn.ModuleList([nn.Conv1d(c, out, kernel_size=1) for c in self.channels])

    def forward(self, feats: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        """feats: 由粗到细的特征列表 [(B,C,L_k), ...] -> 同序的 logits 列表 [(B,1,L_k), ...]。"""
        if len(feats) != len(self.heads):
            raise ValueError(f"期望 {len(self.heads)} 个尺度特征，收到 {len(feats)} 个")
        return [head(f) for head, f in zip(self.heads, feats)]
