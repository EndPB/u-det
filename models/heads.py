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
    """样本级二分类头：序列维池化（可多尺度）+ MLP。

    Args:
        dim: **单层**特征维度（= 主干隐藏维度）。
        hidden: MLP 隐层维度（None 取 dim）。
        out: 类别数。
        dropout: dropout 概率。
        n_levels: **从几个尺度取特征**（默认 1 = 只用瓶颈，与历史行为逐位一致）。

            主干 forward 返回的 ``levels`` 是**由粗到细**的多尺度列表
            （跨度 64 / 32 / 16 / 4 / 1）。只取 ``levels[0]``（瓶颈）时，
            一篇 293 token 的文档会被压成 ``ceil(293/64) = 5`` 个向量再做均值池化 ——
            而同一篇文档在整段编码器里是完整的 293 个 token。
            取多层则把各尺度的池化结果**拼接**，让样本头同时拿到
            “瓶颈的全局视角” 与 “细层的细节”。
    """

    def __init__(self, dim: int, hidden: Optional[int] = None, out: int = 2, dropout: float = 0.0,
                 n_levels: int = 1):
        super().__init__()
        hidden = int(hidden or dim)
        self.n_levels = max(1, int(n_levels))
        width = int(dim) * self.n_levels
        self.dim, self.width = int(dim), width
        self.norm = nn.LayerNorm(width)
        self.net = nn.Sequential(
            nn.Linear(width, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, out)
        )

    @staticmethod
    def _pool(x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """(B, L, D) -> (B, D)：序列维平均池化（可带 mask）。"""
        if mask is None:
            return x.mean(dim=1)
        w = mask.unsqueeze(-1).to(x.dtype)
        return (x * w).sum(dim=1) / w.sum(dim=1).clamp(min=1.0)

    def forward(self, x, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """返回 (B, out)。

        ``x`` 可以是单尺度 (B, L, D)，也可以是 ``[(B, L_k, D)] * n_levels`` 的多尺度列表。
        注意 ``mask`` 只适用于**单尺度**（各尺度长度不同，无法共用一个 mask）；
        U-Det 全流程 batch=1 且不补 padding，所以一直是 mask=None。
        """
        if isinstance(x, (list, tuple)):
            xs = list(x[: self.n_levels])
            if len(xs) != self.n_levels:
                raise ValueError(f"SampleHead 配置了 {self.n_levels} 个尺度，实际收到 {len(xs)} 个")
            if mask is not None:
                raise ValueError("多尺度 SampleHead 不支持共用 mask（各尺度长度不同）")
            parts = [self._pool(t) for t in xs]
            pooled = parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)
        else:
            pooled = self._pool(x, mask)
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
