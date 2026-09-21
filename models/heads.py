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


class GatedAttentionPool(nn.Module):
    """ABMIL 的门控注意力池化（Ilse et al. 2018）：把序列维的**平均**换成**注意力加权和**。

    $$a_i = w^\\top\\big(\\tanh(V h_i)\\odot\\sigma(U h_i)\\big),\\quad
       \\alpha = \\mathrm{softmax}(a),\\quad z = \\sum_i \\alpha_i h_i$$

    ★ **`w` 零初始化** ⇒ 起始时所有 $a_i = 0$ ⇒ $\\alpha$ 均匀 ⇒ **输出与平均池化逐位相同**。
    这和 peft 的 `lora_B` 零初始化、主干门控的 `gate_init=-1` 是同一个套路：
    让新算子从"旧行为"起步，既便于对比，也避免冷启动。
    （副作用：第一步 $V/U$ 的梯度为 0（$\\partial a/\\partial V \\propto w = 0$），
    只有 $w$ 先动；$w$ 一非零 $V/U$ 就开始学 —— 这是标准的"末层零初始化"行为。）

    Args:
        dim: 输入特征维度。
        attn_dim: 注意力隐层维度 $M$（同时决定 $V,U$ 的参数量：$2\\cdot D\\cdot M$）。
        dropout: 注意力分支的 dropout。
    """

    def __init__(self, dim: int, attn_dim: int = 128, dropout: float = 0.0):
        super().__init__()
        self.V = nn.Linear(int(dim), int(attn_dim))
        self.U = nn.Linear(int(dim), int(attn_dim))
        self.w = nn.Linear(int(attn_dim), 1, bias=False)
        self.drop = nn.Dropout(dropout)
        nn.init.zeros_(self.w.weight)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """(B, L, D) -> (B, D)。``mask`` 为 (B, L) 的 0/1，0 的位置不参与注意力。"""
        a = self.w(torch.tanh(self.V(x)) * torch.sigmoid(self.U(x))).squeeze(-1)   # (B, L)
        if mask is not None:
            a = a.masked_fill(mask.to(torch.bool) == 0, float("-inf"))
        alpha = self.drop(torch.softmax(a, dim=1))                                 # (B, L)
        return (alpha.unsqueeze(-1) * x).sum(dim=1)


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
                 n_levels: int = 1, report_dim: int = 0, report_proj: int = 32,
                 pool: str = "mean", attn_dim: int = 128):
        super().__init__()
        hidden = int(hidden or dim)
        self.n_levels = max(1, int(n_levels))
        self.pool = str(pool)
        if self.pool not in ("mean", "abmil"):
            raise ValueError(f"未知 pool {pool!r}，可选：mean / abmil")
        # v0.4.8：把**逐尺度平均池化**换成 ABMIL（门控注意力）。
        #   pool="mean"（默认）⇒ 不建任何注意力模块，权重形状与历史**逐位一致**。
        self.attn = None
        if self.pool == "abmil":
            self.attn = nn.ModuleList(
                [GatedAttentionPool(int(dim), attn_dim, dropout) for _ in range(self.n_levels)])
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
            if self.attn is not None:                       # v0.4.8：逐尺度各自做 ABMIL
                parts = [m(t) for m, t in zip(self.attn, xs)]
            else:
                parts = [self._pool(t) for t in xs]
            pooled = parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)
        else:
            pooled = self.attn[0](x, mask) if self.attn is not None else self._pool(x, mask)
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
