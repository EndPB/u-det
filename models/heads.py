"""任务头（单文件实现）：样本级 + token 级（多尺度）。

约定与 v0.2 主干一致：特征为 **(B, L, D)**（序列优先）。

    SampleHead  接在瓶颈特征上：序列维平均池化 -> MLP -> 二分类 logits
    TokenHeads  接在各尺度特征上：每尺度一个 Linear(D -> out)，输出转成 (B, out, L) 供池化损失

用法::

    from models import SampleHead, TokenHeads

    sample_head = SampleHead(768, hidden=256)            # (B, L/16, 768) -> (B, 2)
    token_heads = TokenHeads([768] * 5)                  # 5 个尺度 -> [(B, 1, L/16), ..., (B, 1, L)]
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch
import torch.nn as nn


class SampleHead(nn.Module):
    """样本级二分类头：序列维池化 + MLP。

    Args:
        dim: 输入特征维度（= 主干隐藏维度）。
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
        """(B, L, D) -> (B, out)；mask (B, L) 为 1/0 有效位标记（可选）。"""
        if mask is None:
            pooled = x.mean(dim=1)
        else:
            w = mask.unsqueeze(-1).to(x.dtype)
            pooled = (x * w).sum(dim=1) / w.sum(dim=1).clamp(min=1.0)
        return self.net(self.norm(pooled))


class TokenHeads(nn.Module):
    """token 级分类头：每个尺度一个线性投影（多尺度深监督）。

    Args:
        dims: 各尺度输入维度（v0.2 隐藏维度恒定，通常都是同一个值）。
        out: 输出通道数（二分类取 1，配合 BCEWithLogits）。
    """

    def __init__(self, dims: Sequence[int], out: int = 1):
        super().__init__()
        self.dims = tuple(int(d) for d in dims)
        self.heads = nn.ModuleList([nn.Linear(d, out) for d in self.dims])

    def forward(self, feats: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        """feats: 由粗到细的特征列表 [(B, L_k, D), ...] -> [(B, out, L_k), ...]。"""
        if len(feats) != len(self.heads):
            raise ValueError(f"期望 {len(self.heads)} 个尺度特征，收到 {len(feats)} 个")
        return [head(f).transpose(1, 2) for head, f in zip(self.heads, feats)]
