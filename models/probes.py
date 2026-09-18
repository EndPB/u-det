"""辅助探针（单文件）：从"每个位置代表 span 个 token"的特征里解码窗口内**逐 token** 的 logits。

动机（v0.4 的 B2「跳跃连接引导下采样」）
------------------------------------------------
主干的下采样把 ``span`` 个 token 压成一个向量。现有 ``token_loss`` 只要求这个向量能预测
"窗口内 AI **比例**"（一个数）——**信息论上这并不要求它保留窗口内的位置结构**：
一个 64 长的窗口里有 1 个 AI token 和 32 个，目标分别是 0.016 / 0.5，
而窗口里那段 AI 落在哪、边界在哪，完全没被约束。位置/边界恰恰是行级、片段级评测真正吃的信号。

``PositionProbe`` 把要求改成"回答 ``span`` 个问题"（窗口内第 t 个 token 是不是 AI）：

    logit[b, j, t] = <proj(f[b, j]), W[t]> / sqrt(hidden)

于是下采样必须**自己**保住窗口内逐 token 的位置信息。注意这条约束是挂在**下采样路径**上的
（不是上采样/skip 那一侧），所以 skip 无法替它"兜底"——这正是"跳跃引导下采样"：
细层的 token 标签反过来教下采样该保留什么。

为什么用双线性分解而不是 gather
--------------------------------
朴素做法是把每个窗口的 ``span`` 个特征 gather 成 ``(B, L_k, span, D)`` 再喂 MLP，
在 L=24576、span=64 时是 3.9 GB 级别的中间张量。这里改成**先投影再点积**：

    u = proj(f)            (B, L_k, hidden)
    logit = einsum('bjh,sh->bjs', u, W) * scale

位置码 ``W ∈ R^{span×hidden}`` 是一张**可学的位置码表**（等价于给每个偏移一个专属读出方向），
算力只有 ``L_k · span · hidden``，L=24576 时总共 2.4 万个 logit，显存可忽略。

用法::

    from models import PositionProbe, PositionProbes, position_targets

    probe = PositionProbe(768, span=64, hidden=256)              # 瓶颈：1 个位置代表 64 个 token
    probs = probe(f)                                             # (B, L_k, 64)
    target, keep = position_targets(tok_labels, span=64, ignore=-100)
    loss = F.binary_cross_entropy_with_logits(probs, target, weight=keep, reduction="sum") / keep.sum()

    probes = PositionProbes(768, spans=(4, 16, 32, 64))          # 挂在下采样路径的 4 级上
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def position_targets(tok_labels: torch.Tensor, span: int, ignore: int = -100
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(B, L)`` 的逐 token 标签 -> ``(B, L_k, span)`` 的目标与有效位掩码。

    ``L_k = ceil(L / span)``，第 j 个位置的第 t 个偏移对应原始下标 ``j * span + t``；
    序列尾端不足一个 span 的部分补 ``ignore`` 并掩掉。
    """
    batch, length = tok_labels.shape
    length_out = math.ceil(length / span)
    want = length_out * span
    padded = F.pad(tok_labels, (0, want - length), value=ignore) if want > length else tok_labels
    target = padded.reshape(batch, length_out, span)
    keep = (target != ignore).float()
    return target.clamp(min=0).float(), keep


class PositionProbe(nn.Module):
    """单级位置探针：``(B, L_k, D) -> (B, L_k, span)`` 的逐偏移 logits。

    Args:
        dim: 输入特征维度（= 主干隐藏维度）。
        span: 每个位置代表多少个原始 token（= 主干该级的 ``stride``）。
        hidden: 分解维度（位置码与投影向量的公共维度）。
        dropout: dropout 概率（作用在投影后的读出向量上）。
    """

    def __init__(self, dim: int = 768, span: int = 4, hidden: int = 256, dropout: float = 0.0):
        super().__init__()
        self.dim, self.span, self.hidden = int(dim), int(span), int(hidden)
        self.norm = nn.LayerNorm(self.dim)
        self.proj = nn.Linear(self.dim, self.hidden)
        self.pos = nn.Parameter(torch.randn(self.span, self.hidden) * 0.02)
        self.drop = nn.Dropout(dropout)
        self.scale = self.hidden ** -0.5
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        nn.init.normal_(self.pos, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L_k, D) -> (B, L_k, span)。"""
        u = self.drop(self.proj(self.norm(x)))                    # (B, L_k, H)
        return torch.einsum("bjh,sh->bjs", u, self.pos) * self.scale


class PositionProbes(nn.Module):
    """一组位置探针（**每级独立权重**），按 ``spans`` 顺序与输入特征一一对应。

    ``span=1`` 时退化为普通的逐 token 线性头（rank = hidden 的受限线性分类器）。

    Args:
        dim: 输入特征维度。
        spans: 各级的跨度列表。
        hidden: 分解维度。
        dropout: dropout 概率。
    """

    def __init__(self, dim: int = 768, spans: Sequence[int] = (4, 16, 32, 64),
                 hidden: int = 256, dropout: float = 0.0):
        super().__init__()
        self.spans: Tuple[int, ...] = tuple(int(s) for s in spans)
        self.probes = nn.ModuleList(
            [PositionProbe(dim, span, hidden, dropout) for span in self.spans]
        )

    def forward(self, feats: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        if len(feats) != len(self.probes):
            raise ValueError(f"期望 {len(self.probes)} 级特征，收到 {len(feats)} 级")
        return [probe(f) for probe, f in zip(self.probes, feats)]


__all__ = ["PositionProbe", "PositionProbes", "position_targets"]
