"""序列级 Transformer 编解码器（v0.3，单文件实现）。

与 v0.2（频域 U-Net）的唯一区别：**上下采样不再用 FFT 核，而是"窗口化跨注意力"的
Transformer 层**；隐藏维度 D 恒定，只有 token 数量在变。

结构（下采样 -> 瓶颈 -> 上采样）::

    下采样 x4   第 i 级把 L 降到 ceil(L / div_i)：用 ceil(L / div_i) 个 query，
                每个 query 只在**细层**序列上取一个宽度 win_i 的窗口当 key/value。
                **四级共享同一个 Block**（像卷积核一样拿同一个块在不同层上跑）。
    瓶颈   x4   最粗分辨率上的自注意力（独立权重）。
    上采样 x4   逐级镜像。query 来自**粗层**（升维后的捷径），
                key/value 来自**下采样前存下的细层特征（skip）** ——
                也就是说**上采样的跨注意力本身就是跳连**，不再需要额外的融合模块。
                上采样的四级共享**另一个** Block（与下采样不共享）。

两种采样都带**无参算子捷径**残差（对应 v0.2 的成对均值池化 / 最近邻重复）::

    下采样捷径 = 同窗口均值池化      dst = mean(窗口)          L -> ceil(L/div)
    上采样捷径 = 最近邻重复          dst = repeat_interleave   L -> L*div
    每个 Block 的输出 = dst + o(attn(q(dst), k/v(窗口))) 再过 FFN

query 一律由捷径经线性投影得到（内容驱动），不用可学 query 表 -> 天然支持任意长度。

填充：窗口越界部分两侧统一补 **[PAD]（零向量）**，与卷积的 padding 同语义，
因此不需要任何注意力掩码。长度记账仍走 v0.2 那套闭环（补到 div 整数倍再裁回真实长度）。

用法::

    net = build_codec(dim=768, depth=4, mid=4, heads=12)
    levels, _ = net(hidden)          # levels 由粗到细
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .hier import SelfBlock, SinusoidalPE


def window_pad(seq: torch.Tensor, stride: int, window: int) -> Tuple[torch.Tensor, int, torch.Tensor]:
    """按"窗口化采样"的需要给序列两侧补 [PAD]（零向量），返回 (padded, L_out, idx)。

    输出位置 j 的锚点在**原序列**坐标里是 ``j * stride``；窗口是以锚点为中心、宽 ``window`` 的一段。
    左补 ``left=(window-1)//2``、右侧补到最后一个窗口能完整放下，于是 padded 坐标里下标就是
    ``锚点 + t``，**不需要任何掩码**（越界位置的值就是 [PAD] 的零向量，与卷积 padding 同语义）。

    Args:
        seq: (B, L, D)。
        stride: 抽取向量的步长（下采样时 = 除数；上采样的 kv 侧固定为 1）。
        window: 感受野宽度。

    Returns:
        padded (B, L+left+right, D)；L_out = ceil(L/stride)；idx (L_out, window) 的 padded 下标。
    """
    length_in = seq.size(1)
    length_out = math.ceil(length_in / stride)
    anchor = torch.arange(length_out, device=seq.device) * stride
    left = (window - 1) // 2
    need = (int(anchor[-1].item()) + window) if length_out else 0
    right = max(0, need - (length_in + left))
    if left or right:
        seq = F.pad(seq, (0, 0, left, right))                  # [PAD]（零向量）
    idx = anchor[:, None] + torch.arange(window, device=seq.device)[None, :]
    assert idx.numel() == 0 or int(idx.max().item()) < seq.size(1), "窗口越出补齐范围"
    return seq, length_out, idx


def _gather(seq: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """(B, L, D) + (L_out, win) -> (B, L_out, win, D)。"""
    batch, _, dim = seq.shape
    return seq[:, idx.reshape(-1)].reshape(batch, idx.size(0), idx.size(1), dim)


class WindowBlock(nn.Module):
    """一次窗口化跨注意力采样（下采样与上采样共用的结构，各自持有独立权重）。

    ``out = dst + o(attn(q(norm(dst)), k/v(norm(窗口))))``，再过 FFN。

    Args:
        dim: 隐藏维度（恒定）。
        heads: 注意力头数。
        mlp_ratio: FFN 隐层倍率。
        dropout: dropout 概率。
    """

    def __init__(self, dim: int = 768, heads: int = 12, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        if dim % heads:
            raise ValueError(f"dim={dim} 必须能被 heads={heads} 整除")
        self.heads, self.head_dim = heads, dim // heads
        self.scale = self.head_dim ** -0.5
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.o = nn.Linear(dim, dim, bias=False)
        self.drop = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, dim)
        )

    def forward(self, src: torch.Tensor, dst: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """src: (B, L_kv, D) 窗口的来源；dst: (B, L_out, D) 捷径（同时充当 query 来源）。"""
        batch, length_out, dim = dst.shape
        win = idx.shape[1]
        # 先在紧凑的 (L_kv, D) 上算 k/v 再 gather，省掉一份 (L_out, win, D) 中间张量（长序列省 ~1GB+）
        h = self.norm_kv(src)
        kk = _gather(self.k(h), idx)                                      # (B, L_out, win, D)
        vv = _gather(self.v(h), idx)
        q = self.q(self.norm_q(dst)).view(batch, length_out, self.heads, self.head_dim).transpose(1, 2)
        kk = kk.view(batch, length_out, win, self.heads, self.head_dim).permute(0, 3, 1, 2, 4)
        vv = vv.view(batch, length_out, win, self.heads, self.head_dim).permute(0, 3, 1, 2, 4)
        alpha = self.drop(torch.softmax(torch.einsum("bhod,bhowd->bhow", q, kk) * self.scale, dim=-1))
        ctx = torch.einsum("bhow,bhowd->bhod", alpha, vv).transpose(1, 2).reshape(batch, length_out, dim)
        x = dst + self.o(ctx)                                             # 算子捷径 + 注意力校正
        return x + self.mlp(self.norm2(x))


class TransformerCodec(nn.Module):
    """v0.3 主干：共享权重的窗口注意力上下采样 + 4 层瓶颈自注意力。

    Args:
        dim: 隐藏维度（恒定，默认 768）。
        depth: 上下采样级数（默认 4）。
        mid: 瓶颈自注意力层数（默认 4）。
        heads: 注意力头数。
        mlp_ratio: FFN 隐层倍率。
        dropout: dropout 概率。
        divs: 各级下采样除数（默认 ``(4, 4, 2, 2)``，累计 64x）。
        wins: 各级窗口大小（默认 ``(16, 16, 16, 32)``）。
        share: True = 同一栈内四级共享一个 Block（像卷积核在层上跑）；
               False（默认）= 每级独立 Block（上下各 4 个，共 8 个）。
        gate: v0.4 跳跃融合门控。True = `_up` 末尾再过一个每级独立的门
              ``out = g ⊙ u + (1-g) ⊙ skip``，``g = σ(Linear([u, skip]))``；
              位置对齐的融合交给逐元素门控，跨注意力只负责"扩张算子"本身。
        gate_init: 门控偏置初值（权重零初始化）。默认 -1.0 ⇒ 起点 g≈0.27，偏向 skip。

    位置/跨度（供位置探针使用）::

        down_strides  = (4, 16, 32, 64)      # downs[k] 每个位置代表多少个原始 token
        level_strides = (64, 32, 16, 4, 1)   # levels  每个位置代表多少个原始 token（由粗到细）
    """

    def __init__(self, dim: int = 768, depth: int = 4, mid: int = 4, heads: int = 12,
                 mlp_ratio: float = 4.0, dropout: float = 0.0,
                 divs: Sequence[int] = (4, 4, 2, 2), wins: Sequence[int] = (16, 16, 16, 32),
                 share: bool = False, gate: bool = False, gate_init: float = -1.0, **ignored):
        super().__init__()
        divs, wins = tuple(int(d) for d in divs), tuple(int(w) for w in wins)
        if not (len(divs) == len(wins) == depth):
            raise ValueError(f"divs/wins 长度必须等于 depth={depth}（收到 {divs} / {wins}）")
        self.dim, self.depth, self.mid_layers = dim, depth, mid
        self.divs, self.wins, self.share = divs, wins, bool(share)
        # "一个输出位置代表多少个原始 token"：ceil 的复合是精确的，故等于各级除数连乘
        self.down_strides = tuple(int(math.prod(divs[: i + 1])) for i in range(depth))
        self.level_strides = tuple(int(math.prod(divs[: depth - i])) for i in range(depth)) + (1,)
        n = 1 if self.share else depth
        self.down_blocks = nn.ModuleList([WindowBlock(dim, heads, mlp_ratio, dropout) for _ in range(n)])
        self.up_blocks = nn.ModuleList([WindowBlock(dim, heads, mlp_ratio, dropout) for _ in range(n)])
        self.mid_blocks = nn.ModuleList([SelfBlock(dim, heads, mlp_ratio, dropout) for _ in range(mid)])
        self.pe = SinusoidalPE(dim)
        self.gate_init = float(gate_init)
        self.gate = None
        if gate:
            self.gate = nn.ModuleList([nn.Linear(2 * dim, dim) for _ in range(depth)])
            for lin in self.gate:
                nn.init.zeros_(lin.weight)                    # 起点 g 与内容无关 = σ(gate_init)
                nn.init.constant_(lin.bias, self.gate_init)

    # ------------------------------------------------------------------ #
    def _down(self, x: torch.Tensor, level: int) -> Tuple[torch.Tensor, int]:
        """一级下采样：两侧补 [PAD] -> 窗口均值捷径 -> 窗口跨注意力（kv = 细层自己）。"""
        stride, window = self.divs[level], self.wins[level]
        seq, length_out, idx = window_pad(x, stride, window)
        dst = _gather(seq, idx).mean(dim=2)                    # 无参捷径：窗口均值池化
        block = self.down_blocks[0 if self.share else level]
        return block(seq, dst, idx), length_out

    def _up(self, x: torch.Tensor, skip: torch.Tensor, level: int) -> torch.Tensor:
        """一级上采样：最近邻重复当捷径与 query，kv = **下采样前存下的细层（skip）**。

        v0.4 起末尾多一道**门控融合**（``gate=True`` 时）：跨注意力的 kv 是 skip，
        它同时承担"扩张"和"跳跃融合"两件事；门控把后者单独拿出来，
        让 skip 以"逐通道加权残差"的形式参与，而不是必须穿过 softmax 对齐映射。
        """
        stride, window = self.divs[level], self.wins[level]
        seq, _, idx = window_pad(skip, 1, window)                # kv 侧：细层，步长 1
        dst = x.repeat_interleave(stride, dim=1)[:, : skip.size(1)]
        block = self.up_blocks[0 if self.share else level]
        out = block(seq, dst, idx)                               # 扩张算子（跨注意力）
        if self.gate is None:
            return out
        g = torch.sigmoid(self.gate[level](torch.cat([out, skip], dim=-1)))
        return g * out + (1.0 - g) * skip

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor, return_features: bool = True, return_down: bool = False):
        """x: (B, L, D)（整段序列，长度任意）；返回 levels 由粗到细。

        return_down=True 时额外返回**下采样路径**的各级输出（供位置探针使用），
        第 k 个元素的下标对应 ``down_strides[k]`` 前多少个原始 token。
        """
        lengths = [x.size(1)]
        skips: List[torch.Tensor] = []
        downs: List[torch.Tensor] = []
        for level in range(self.depth):                          # ===== 下采样 =====
            skips.append(x)                                      # tap：下采样前的细层特征
            x, length_out = self._down(x, level)
            lengths.append(length_out)
            x = self.pe(x)                                       # 每级入口重加正弦 PE
            downs.append(x)
        for block in self.mid_blocks:                            # ===== 瓶颈 =====
            x = block(x)
        levels = [x]
        for level in reversed(range(self.depth)):                # ===== 上采样 =====
            x = self._up(x, skips[level], level)                 # 跨注意力（+ v0.4 门控）即跳连
            x = self.pe(x)
            levels.append(x)
        if return_down:
            return levels, x, downs
        return (levels, x) if return_features else levels


def build_codec(name: str = "codec", **kwargs) -> TransformerCodec:
    """构建 v0.3 主干。"""
    if name not in ("codec", "hier2", "transformer"):
        raise KeyError(f"未知主干 {name!r}，可选：['codec']")
    kwargs.pop("name", None)
    kwargs.pop("in_channels", None)
    return TransformerCodec(**kwargs)
