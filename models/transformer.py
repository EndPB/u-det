"""U-Net 瓶颈处的双向语义交互（单文件实现）。

放在 U-Net 最底部（下采样之后、上采样之前），在压缩后的序列上做全局双向注意力：
    * 序列长度只有 L/2^depth，注意力开销可忽略；
    * 用正弦位置编码，长度无关（配合逐 token 编码器可处理任意长输入）。

用法::

    from models import build_mid

    mid = build_mid("transformer", dim=512, layers=2, heads=8)   # 或 build_mid("none")
    x = mid(x, mask)                                             # (B, C, L) 进出同形
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalPositionEncoding(nn.Module):
    """正弦位置编码（固定、不占参数、长度无关）。"""

    def __init__(self, dim: int, max_positions: int = 8192):
        super().__init__()
        self.dim = dim
        position = torch.arange(max_positions, dtype=torch.float32)[:, None]
        div = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
        pe = torch.zeros(max_positions, dim)
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe[None], persistent=False)  # (1, P, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, L, D)
        length = x.shape[1]
        if length > self.pe.shape[1]:  # 超出预计算范围则临时扩展
            self.pe = self._make(length).to(self.pe.device, self.pe.dtype)
        return x + self.pe[:, :length].to(x.dtype)

    def _make(self, length: int) -> torch.Tensor:
        pos = torch.arange(length, dtype=torch.float32)[:, None]
        div = torch.exp(torch.arange(0, self.dim, 2, dtype=torch.float32) * (-math.log(10000.0) / self.dim))
        pe = torch.zeros(length, self.dim)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
        return pe[None]


class Block(nn.Module):
    """pre-LN Transformer 编码器层（自注意力 + FFN）。"""

    def __init__(self, dim: int, heads: int, dropout: float = 0.0, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, dim)
        )

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + h
        return x + self.mlp(self.norm2(x))


class MidTransformer(nn.Module):
    """瓶颈双向交互：若干层 pre-LN Transformer，对 (B, C, L) 序列做全局注意力。

    Args:
        dim: 通道数（与 U-Net 最深层的 features[-1] 一致，如 512）。
        layers: 层数。
        heads: 注意力头数。
        dropout: dropout 概率。
        mlp_ratio: FFN 隐层相对 dim 的倍率。
        max_positions: 正弦位置编码预计算长度（超出会自动扩展）。
    """

    def __init__(
        self,
        dim: int = 512,
        layers: int = 2,
        heads: int = 8,
        dropout: float = 0.0,
        mlp_ratio: float = 4.0,
        max_positions: int = 8192,
    ):
        super().__init__()
        self.dim = dim
        self.pos = SinusoidalPositionEncoding(dim, max_positions)
        self.blocks = nn.ModuleList([Block(dim, heads, dropout, mlp_ratio) for _ in range(layers)])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """(B, C, L) -> (B, C, L)；mask (B, L) 为 1/0 的有效位标记（可选）。"""
        if x.dim() != 3:
            raise ValueError(f"MidTransformer 只处理 (B, C, L) 序列，收到 {tuple(x.shape)}")
        h = x.transpose(1, 2)                       # (B, L, C)
        h = self.pos(h)
        key_padding_mask = None if mask is None else ~mask.bool()
        for block in self.blocks:
            h = block(h, key_padding_mask)
        return self.norm(h).transpose(1, 2)


def build_mid(name: str = "none", **kwargs) -> nn.Module | None:
    """按名称构建瓶颈模块（'none' 表示不插，直接走 U-Net 深支路）。"""
    if name in (None, "", "none", "identity"):
        return None
    if name != "transformer":
        raise KeyError(f"未知瓶颈模块 {name!r}，可选：['none', 'transformer']")
    kwargs.pop("name", None)
    return MidTransformer(**kwargs)
