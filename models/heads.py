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
                 n_levels: int = 1, report_dim: int = 0, report_proj: int = 32):
        super().__init__()
        hidden = int(hidden or dim)
        self.n_levels = max(1, int(n_levels))
        self.report_dim = max(0, int(report_dim))
        # v0.4.6：手工统计报告作为**文档级全局向量**注入，不再作为前缀 token 进序列。
        #   report_dim = 0 ⇒ 整条分支不存在，权重形状与历史**逐位一致**（老 ckpt 可直接载）。
        self.report_width = 0
        if self.report_dim > 0:
            self.report_width = max(1, int(report_proj))
            self.report_mlp = nn.Sequential(
                nn.Linear(self.report_dim, self.report_width), nn.GELU())
        width = int(dim) * self.n_levels + self.report_width
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

    def forward(self, x, mask: Optional[torch.Tensor] = None,
                report: Optional[torch.Tensor] = None) -> torch.Tensor:
        """返回 (B, out)。

        ``x`` 可以是单尺度 (B, L, D)，也可以是 ``[(B, L_k, D)] * n_levels`` 的多尺度列表。
        注意 ``mask`` 只适用于**单尺度**（各尺度长度不同，无法共用一个 mask）；
        U-Det 全流程 batch=1 且不补 padding，所以一直是 mask=None。

        ``report``：v0.4.6 的文档级手工统计向量 (B, report_dim)。
        配了 report_dim 时**必需**；未配时必须为空 —— 两边都用报错而非静默忽略，
        避免出现"以为接上了其实没接"这种只有看指标才能发现的错误。
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
        if self.report_dim > 0:
            if report is None:
                raise ValueError(f"SampleHead 配了 report_dim={self.report_dim}，但没有收到 report 向量")
            r = torch.as_tensor(report, dtype=pooled.dtype, device=pooled.device)
            if r.dim() == 1:
                r = r.unsqueeze(0)
            if r.shape[-1] != self.report_dim:
                raise ValueError(f"report 维度 {r.shape[-1]} != report_dim {self.report_dim}")
            pooled = torch.cat([pooled, self.report_mlp(r)], dim=-1)
        elif report is not None:
            raise ValueError("SampleHead 未配 report_dim，却收到了 report 向量（配置不一致）")
        return self.net(self.norm(pooled))


class TokenHeads(nn.Module):
    """token 级分类头：每个尺度一个线性投影（多尺度深监督）。

    Args:
        dims: 各尺度输入维度（v0.2 隐藏维度恒定，通常都是同一个值）。
        out: 输出通道数（二分类取 1，配合 BCEWithLogits）。
        bypass_dim: **最细尺度的旁路**（v0.4.5）。非空时，最细尺度的头额外拼接一份
            维度为 ``bypass_dim`` 的特征（U-Det 传的是**编码器的全长逐 token 输出**）。

            动机（与 v0.4.4 给样本头加多尺度是同一个原理）：主干所有尺度都是
            **瓶颈之后**的产物，而瓶颈跨度是 64 个 token —— 一篇 300 token 的短文档
            只賸 ⌈300/64⌉ = 5 个位置。实测（§8.10.2）显示 U-Det 的弱项恰好集中在
            **短文档的 token 级指标**（`[0,512)` 桶 line −5.22 / chunk −6.23），
            与这个瓶颈诊断吻合。旁路让最细尺度直接看到未经压缩的编码器特征。

            ★ 为什么不做“把各尺度 logits 融合起来”：粗尺度的目标经过
            `lse`/`mean` 聚合，它的语义是“窗口内是否有 AI”而非逐 token 判断，
            升采样后当成逐 token 预测融合会**模糊**细尺度结果。
    """

    def __init__(self, dims: Sequence[int], out: int = 1, bypass_dim: Optional[int] = None):
        super().__init__()
        self.dims = tuple(int(d) for d in dims)
        self.bypass_dim = None if bypass_dim is None else int(bypass_dim)
        head_dims = list(self.dims)
        if self.bypass_dim:                          # 只有**最细**尺度拼接旁路
            head_dims[-1] = self.dims[-1] + self.bypass_dim
        self.heads = nn.ModuleList([nn.Linear(d, out) for d in head_dims])

    def forward(self, feats: Sequence[torch.Tensor],
                bypass: Optional[torch.Tensor] = None) -> List[torch.Tensor]:
        """feats: 由粗到细 [(B, L_k, D), ...]；bypass: (B, L, D_by)（仅最细尺度用）。

        返回 [(B, out, L_k), ...]。
        """
        if len(feats) != len(self.heads):
            raise ValueError(f"期望 {len(self.heads)} 个尺度特征，收到 {len(feats)} 个")
        xs = list(feats)
        if self.bypass_dim:
            if bypass is None:
                raise ValueError("TokenHeads 配置了 bypass_dim，但 forward 未收到 bypass 特征")
            if tuple(bypass.shape[:2]) != tuple(xs[-1].shape[:2]):
                raise ValueError(
                    f"旁路特征必须与最细尺度同长：{tuple(bypass.shape)} vs {tuple(xs[-1].shape)}")
            xs[-1] = torch.cat([xs[-1], bypass], dim=-1)
        return [head(f).transpose(1, 2) for head, f in zip(self.heads, xs)]
