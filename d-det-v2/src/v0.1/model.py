# -*- coding: utf-8 -*-
"""d-det-v2 v0.1 · 模型层

- EncoderPool：CodeT5 编码器 + 分块前向（≤4×512，首2+尾2）+ 有效 token 加权池化；
- Stage1Model：阶段一直接监督（B0/B1）；
- FHead：家族头（低秩 8 维 + GELU + 零初始化末层，γ=1）；
- DHead：检测头（b=[z;z⊙z] 或 signed/magnitude 读出 + 零初始化末层）。

约定：基线 logits 以 stopgrad 加入阶段二输出；末层零初始化 ⇒ 初始预测等于基线。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5EncoderModel


class EncoderPool(nn.Module):
    """CodeT5 编码器（可微调/冻结）+ 分块池化 → h ∈ R^d。"""

    def __init__(self, path, grad_checkpointing=False):
        super().__init__()
        self.enc = T5EncoderModel.from_pretrained(path)
        if grad_checkpointing:
            self.enc.gradient_checkpointing_enable()
        self.hidden = int(self.enc.config.d_model)

    def forward(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # ids/mask: [B, C, L] → h: [B, D]
        B, C, L = ids.shape
        flat_ids = ids.reshape(B * C, L)
        flat_mask = mask.reshape(B * C, L).to(torch.long)
        out = self.enc(input_ids=flat_ids, attention_mask=flat_mask).last_hidden_state  # [BC, L, D]
        w = flat_mask.unsqueeze(-1).to(out.dtype)
        pooled = (out * w).sum(1) / w.sum(1).clamp(min=1.0)          # [BC, D]
        pooled = pooled.reshape(B, C, -1)
        tok_w = flat_mask.sum(1).to(out.dtype).reshape(B, C)          # [B, C]
        h = (pooled * tok_w.unsqueeze(-1)).sum(1) / tok_w.sum(1, keepdim=True).clamp(min=1.0)
        return h


class Stage1Model(nn.Module):
    """阶段一：编码器 + 线性头（F: K+1 类；D: 标量）。"""

    def __init__(self, encoder_path, n_out, grad_checkpointing=False):
        super().__init__()
        self.encoder = EncoderPool(encoder_path, grad_checkpointing=grad_checkpointing)
        self.head = nn.Linear(self.encoder.hidden, n_out)

    def forward(self, ids, mask):
        h = self.encoder(ids, mask)
        return self.head(h), h


class FHead(nn.Module):
    """家族校正头：z = B·x ∈ R^rank；ℓ = γ·W·GELU(z)，W 零初始化（γ=1）。

    in_dim = 2d（Ours/B2/B3）或 d（机制头 r-only / ref-only）。
    """

    def __init__(self, in_dim, rank, n_cls):
        super().__init__()
        self.B = nn.Linear(in_dim, rank, bias=False)
        self.W = nn.Linear(rank, n_cls)
        nn.init.zeros_(self.W.weight)
        nn.init.zeros_(self.W.bias)

    def forward(self, x):
        return self.W(F.gelu(self.B(x)))


class DHead(nn.Module):
    """检测校正头：z = B·x ∈ R^rank；b = [z; z⊙z]（full）/ [z]（signed）/ [z⊙z]（mag）。

    ℓ = γ·w·GELU(V·b)，w 零初始化（γ=1）；外部与 stopgrad(s1) 相加。
    """

    def __init__(self, in_dim, rank, hidden, mode="full"):
        super().__init__()
        assert mode in ("full", "signed", "mag")
        self.mode = mode
        self.B = nn.Linear(in_dim, rank, bias=False)
        b_dim = rank * 2 if mode == "full" else rank
        self.V = nn.Linear(b_dim, hidden)
        self.w = nn.Linear(hidden, 1)
        nn.init.zeros_(self.w.weight)
        nn.init.zeros_(self.w.bias)

    def forward(self, x):
        z = self.B(x)
        if self.mode == "full":
            b = torch.cat([z, z * z], dim=-1)
        elif self.mode == "signed":
            b = z
        else:
            b = z * z
        return self.w(F.gelu(self.V(b))).squeeze(-1)


# --------------------------------------------------------------------------- #
# 阶段二特征构造（臂 → 头输入）
# --------------------------------------------------------------------------- #

def feat_full(h, m):
    """Ours/B3：x = [h − m; h ⊙ m]（m 为预测参考或全局均值，可广播 [1,d]）。"""
    if m.ndim == 2 and m.shape[0] == 1:
        m = m.expand_as(h)
    return torch.cat([h - m, h * m], dim=-1)


def feat_b2(h):
    """B2：x = [h; h ⊙ h]（等容量普通校正，无内容参考）。"""
    return torch.cat([h, h * h], dim=-1)


def feat_r(h, m):
    """机制头 r-only：x = h − m。"""
    if m.ndim == 2 and m.shape[0] == 1:
        m = m.expand_as(h)
    return h - m


def feat_ref(m):
    """机制头 ref-only：x = m。"""
    return m


def freeze_(module: nn.Module):
    for p in module.parameters():
        p.requires_grad_(False)


def n_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())
