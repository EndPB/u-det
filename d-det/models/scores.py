"""d-det 核心模型：双分数读出（s1 众数坍缩度 / s2 偏好位移度）。

设计见 ``docx/d-det.md`` §2/§3/§5（推导）与附录 A（符号 ↔ 代码映射）：

    s1 = α1 · w1ᵀ g1(z)     人机分离（m4 的 BCE 监督）
    s2 = α2 · w2ᵀ g2(z)     偏好位移（base/instruct 配对的 hinge margin 监督）

三处"工程落点"与原文的差异（改代码前先读这里）：

1. ★ **零初始化只落在一个因子上**（修正原文的一处梯度死锁）：
   原文写 "α1=α2=0（初始）" 且 "新通道零初始化" —— 若 α 与 w 同时为 0，则
   ∂L/∂w = α·(…) = 0 且 ∂L/∂α = w·h = 0，两头都学不动（一阶优化的乘法死锁）。
   本实现固定 α=1、只零初始化读出向量 w1（s2_rank>1 时 w2 例外地用小随机初始化——docx/d-det-v0.2.md §3 死锁第 3 例）：起点 s1≡s2≡0（可做"起点等价"自检，
   见 scripts/selfcheck.py），且第一个 step 读出向量就有非零梯度。
   （u-det 的门控用 bias=-1 制造非零起点，是同一类"冷启动"问题的另一种解法。）

2. **正交正则用归一化 cos²(w1, w2)**：裸内积 (w1ᵀw2)² 可被模长缩放规避、对量纲敏感；
   cos² 才是"读出方向正交"的直接度量（实现见 train.py: orth_reg；零向量处 = 0 且梯度为 0）。

3. **s2 支持低秩**：w2 形状 (D, r)。r=1 → 标量（与原文公式逐字对应）；
   r>1 → r 维向量（设计文档注记 1 的"稳健版子空间"；训练侧用 (s2₊−s2₋) 的 L2 范数做
   hinge，语义 = "到人类先验子空间的距离"）。首版配置 r=1。

参数量：读出层 2×768（r=1）；pool=abmil 时另加约 0.2M（V/U/w），mean 时为零。
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class GatedAttentionPool(nn.Module):
    """门控注意力池化（ABMIL）：(B, L, D) -> (B, D)。

    ``w`` 零初始化 ⇒ 起点注意力均匀，输出与平均池化**逐位相同**
    （u-det 的同一套路：让新算子从"旧行为"起步，便于对比与冷启动）。
    副作用：第一步 V/U 无梯度（∂a/∂V ∝ w = 0），w 先动，之后 V/U 才学。
    """

    def __init__(self, dim: int, attn_dim: int = 128, dropout: float = 0.0):
        super().__init__()
        self.V = nn.Linear(int(dim), int(attn_dim))
        self.U = nn.Linear(int(dim), int(attn_dim))
        self.w = nn.Linear(int(attn_dim), 1, bias=False)
        self.drop = nn.Dropout(dropout)
        nn.init.zeros_(self.w.weight)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """``mask`` 为 (B, L) 的 0/1；0 的位置不参与注意力。"""
        a = self.w(torch.tanh(self.V(x)) * torch.sigmoid(self.U(x))).squeeze(-1)   # (B, L)
        if mask is not None:
            a = a.masked_fill(mask.to(torch.bool) == 0, float("-inf"))
        alpha = self.drop(torch.softmax(a, dim=1))                                 # (B, L)
        return (alpha.unsqueeze(-1) * x).sum(dim=1)


class DualScoreModel(nn.Module):
    """编码器 → 池化 → 双读出。``forward`` 返回 ``(s1, s2)``：

        s1: (B,)      众数坍缩分数（越高越"贴 base 众数流形"）
        s2: (B, r)    偏好位移分数（越高越"被偏好奖励推远"；r=1 时即标量）
    """

    def __init__(
        self,
        encoder: nn.Module,
        dim: Optional[int] = None,
        pool: str = "mean",
        attn_dim: int = 128,
        dropout: float = 0.0,
        mlp_layers: int = 0,
        mlp_hidden: Optional[int] = None,
        s2_rank: int = 1,
        s2_detach: bool = False,
    ):
        super().__init__()
        self.encoder = encoder
        dim = int(dim or encoder.hidden_size)
        self.dim = dim
        self.s2_rank = max(1, int(s2_rank))
        self.s2_detach = bool(s2_detach)      # 归因实验：s2 只更新读出 w2（隔离表征冲突）

        self.pool = str(pool)
        if self.pool not in ("mean", "abmil"):
            raise ValueError(f"未知 pool {self.pool!r}，可选：mean / abmil")
        self.attn = (GatedAttentionPool(dim, attn_dim, dropout)
                     if self.pool == "abmil" else None)

        # 共享 trunk（可选；默认 0 层 = 恒等 —— 设计文档 g1/g2 的线性特殊情形）
        self.trunk = None
        if int(mlp_layers) > 0:
            hidden = int(mlp_hidden or dim)
            layers, width = [], dim
            for _ in range(int(mlp_layers)):
                layers += [nn.Linear(width, hidden), nn.GELU(), nn.Dropout(dropout)]
                width = hidden
            self.trunk = nn.Sequential(*layers)

        # ★ 新通道读出：w1 / w2 零初始化（α 恒 1，见模块 docstring 注记 1）
        #   例外：s2_rank>1 时 w2 改用小随机——L2 铰链 relu(m−‖Δ‖) 在 Δ=0 处零梯度，
        #   零初始化会永久冻结 w2（死锁第 3 例，C2 实测 loss_s2≡1.0；见 docx/d-det-v0.2.md §3）
        self.w1 = nn.Linear(dim, 1, bias=False)
        self.w2 = nn.Linear(dim, self.s2_rank, bias=False)
        nn.init.zeros_(self.w1.weight)
        if self.s2_rank == 1:
            nn.init.zeros_(self.w2.weight)          # rank=1：起点等价自检依赖 s2≡0
        else:
            nn.init.normal_(self.w2.weight, std=1e-3)

        # token 级人机读出（hybrid 的 token BCE；零初始化，与样本读出约定一致）
        self.tok_head = nn.Linear(dim, 1)
        nn.init.zeros_(self.tok_head.weight)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _pool_mean(x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if mask is None:
            return x.mean(dim=1)
        w = mask.unsqueeze(-1).to(x.dtype)
        return (x * w).sum(dim=1) / w.sum(dim=1).clamp(min=1.0)

    def features(self, input_ids: torch.Tensor,
                 attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """编码 + 池化 → (B, D)。**唯一特征入口**：敏感性 / 分析脚本也走这里，
        保证与训练完全同一条前向路径。"""
        feats = self.encoder(input_ids, attention_mask)
        h = (self.attn(feats, attention_mask) if self.attn is not None
             else self._pool_mean(feats, attention_mask))
        if self.trunk is not None:
            h = self.trunk(h)
        return h

    def forward(self, input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None):
        s1, s2, _ = self.forward_feat(input_ids, attention_mask)
        return s1, s2

    def forward_feat(self, input_ids: torch.Tensor,
                     attention_mask: Optional[torch.Tensor] = None):
        """同 ``forward``，但额外返回池化特征 h —— 几何辅助损失（Fisher / 去冗余 / 跨族）
        复用同一次前向，避免重复编码。"""
        h = self.features(input_ids, attention_mask)
        s1 = self.w1(h).squeeze(-1)          # (B,)
        # s2_detach=True 时：s2 的梯度不回传编码器（隔离「表征冲突 vs 读出冲突」用）
        h2 = h.detach() if self.s2_detach else h
        s2 = self.w2(h2)                     # (B, r)
        return s1, s2, h

    def token_logits(self, input_ids: torch.Tensor,
                     attention_mask: Optional[torch.Tensor] = None):
        """(B, L) 逐 token 人机分数（token 级辅助监督 / hybrid 评测用）。"""
        feats = self.encoder(input_ids, attention_mask)
        return self.tok_head(feats).squeeze(-1)

    def readout_vectors(self):
        """返回 (w1 (D,), w2 (D, r))：正交度分析（cos²）与归因用。"""
        return self.w1.weight.reshape(-1), self.w2.weight.reshape(self.s2_rank, self.dim).T
