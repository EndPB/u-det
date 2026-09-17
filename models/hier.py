"""频域 U-Net（单文件实现，v0.2 主干）。

对应"频域核版"规范：

    * 隐藏维度恒定 D=768；改变 token 数量的只有两个**固定形状**的算子：
        - KernelDown 下采样：奇数补齐 → rfft → 截低半频（抗混叠）→ 逐频率低秩 D×D 混合
          → irfft 半长 → 加"成对均值池化"残差（盒式低通+抽取，同样抗混叠）→ LayerNorm；
        - KernelUp   上采样：rfft（压缩域）→ 逐频率混合 → 谱零延拓（截断的伴随 = 带限插值）
          → irfft 双倍长 → 按 Ls 裁剪 → LayerNorm。
    * skip 在**下采样前**引出（满分辨率），上采样后经 SkipFuse（concat + DW k3 + PW + 门控）注入；
    * 注意力层：编码每级 1 层 + 瓶颈 8 层 + 解码每级 1 层（默认 4+8+4=16 个 block）；
    * 正弦位置编码在每级入口**重加**（核操作平移等变，PE 会随层数衰减）；
    * FFT 一律在 ≥ 真实长度的 **2 的幂**上做（尾部补零），算完裁回真实长度：
      既保证"不截断"，又让 cuFFT plan 只与 ~10 种尺寸相关（显存有界，防 plan 泄漏）；
    * 长度对调用者透明：Ls 记账 + 裁剪闭环；训练/评测一律 batch=1
      （FFT 是全局算子，批内 padding 会改变频谱语义）。

    用法::

        net = build_hier(dim=768, depth=4, mid=8, heads=12)
        levels, _ = net(hidden)          # levels 由粗到细：[L/16, L/8, L/4, L/2, L]
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _pow2(n: int) -> int:
    """≥ n 的最小 2 的幂（FFT 内部分析/合成长度，保证 cuFFT plan 数量有限）。"""
    return 1 << max(0, (n - 1).bit_length())


class SinusoidalPE(nn.Module):
    """正弦位置编码（固定、无参数、长度无关）；每级入口重加。"""

    def __init__(self, dim: int, max_positions: int = 1 << 16):
        super().__init__()
        self.dim = dim
        self.register_buffer("pe", self._make(max_positions), persistent=False)

    def _make(self, length: int) -> torch.Tensor:
        pos = torch.arange(length, dtype=torch.float32)[:, None]
        div = torch.exp(torch.arange(0, self.dim, 2, dtype=torch.float32) * (-math.log(10000.0) / self.dim))
        pe = torch.zeros(length, self.dim)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
        return pe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) > self.pe.size(0):
            self.pe = self._make(int(x.size(1))).to(self.pe.device)
        return x + self.pe[: x.size(1)].to(x.dtype)


class SelfBlock(nn.Module):
    """pre-LN 双向自注意力 + FFN。"""

    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + h
        return x + self.mlp(self.norm2(x))


class _KernelMixin:
    """频域算子共用的复数低秩混合：V(r,d)、U(d,r) 与逐频率调制 m(k,r)。

    复数参数存 real/imag 两个实数 Parameter（优化器兼容），前向再合成 complex。
    FFT 一律在**内部的 2 的幂长度**上做（不足处补零），运算完再裁回真实长度：
    这样 cuFFT plan 只与 2 的幂尺寸有关（全局 ~10 种），显存有界、不会反复重建。
    """

    def _init_mix(self, dim: int, rank: int, control: int) -> None:
        self.rank, self.control = rank, control
        self.Vr = nn.Parameter(torch.randn(rank, dim) / (2 * dim) ** 0.5)
        self.Vi = nn.Parameter(torch.randn(rank, dim) / (2 * dim) ** 0.5)
        self.Ur = nn.Parameter(torch.randn(dim, rank) / (2 * rank) ** 0.5)
        self.Ui = nn.Parameter(torch.randn(dim, rank) / (2 * rank) ** 0.5)
        self.mr = nn.Parameter(torch.ones(control, rank) + 0.02 * torch.randn(control, rank))
        self.mi = nn.Parameter(torch.zeros(control, rank))

    def _mod(self, bins: int) -> torch.Tensor:
        """控制点（control 个）线性插值到 bins 个频率，返回复数调制 (bins, r)。"""
        pos = torch.linspace(0, self.control - 1, bins, device=self.mr.device)
        lo = pos.floor().long().clamp(0, self.control - 1)
        hi = (lo + 1).clamp(0, self.control - 1)
        t = (pos - lo).unsqueeze(-1)
        return torch.complex(self.mr[lo] * (1 - t) + self.mr[hi] * t,
                             self.mi[lo] * (1 - t) + self.mi[hi] * t)

    def _mix(self, spec: torch.Tensor) -> torch.Tensor:
        """(B, K, D) 复数谱 -> 逐频率低秩混合 (B, K, D)。"""
        v = torch.complex(self.Vr, self.Vi)                  # (r, d)
        u = torch.complex(self.Ur, self.Ui)                  # (d, r)
        return (spec @ v.t() * self._mod(spec.shape[1])) @ u.t()


class KernelDown(_KernelMixin, nn.Module):
    """下采样一步：rfft -> 截低半频 -> 逐频率混合 -> irfft 半长；残差为成对均值池化。"""

    def __init__(self, dim: int, rank: int = 64, control: int = 33):
        super().__init__()
        self._init_mix(dim, rank, control)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, dim = x.shape
        if length % 2:                                       # ① 奇数补齐（复制末 token）
            x = torch.cat([x, x[:, -1:]], dim=1)
            length += 1
        out = length // 2
        p_in = _pow2(length)                                 # 分析长度（2 的幂，显存/plan 有界）
        p_out = max(_pow2(out), p_in // 2)                   # 合成长度（保证低半频能装下）
        xp = F.pad(x, (0, 0, 0, p_in - length)) if p_in > length else x
        spec = torch.fft.rfft(xp, dim=1)                     # ② (B, p_in/2+1, D) 复数
        bins = p_in // 4 + 1                                 # ③ 保低半频（抗混叠）
        mixed = self._mix(spec[:, :bins])                    # ④ 逐频率低秩混合
        spec_out = torch.zeros(batch, p_out // 2 + 1, dim, dtype=mixed.dtype, device=mixed.device)
        spec_out[:, :bins] = mixed                           #   带限插值到合成采样率
        y = torch.fft.irfft(spec_out, n=p_out, dim=1)[:, :out]   # ⑤ 裁回 L/2
        res = x.reshape(batch, out, 2, dim).mean(2)          # ⑥ 成对均值池化残差
        return self.norm(y + res)


class KernelUp(_KernelMixin, nn.Module):
    """上采样一步（KernelDown 的伴随）：压缩域混合 -> 谱零延拓 -> irfft 双倍 -> 裁剪。"""

    def __init__(self, dim: int, rank: int = 64, control: int = 33, residual: bool = False):
        super().__init__()
        self._init_mix(dim, rank, control)
        self.residual = residual
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, length_out: int) -> torch.Tensor:
        batch, length, dim = x.shape
        p_in = _pow2(length)                                 # 压缩域分析长度（2 的幂）
        p_out = max(_pow2(2 * length), p_in)                 # 合成长度（≥ 2L 的 2 的幂）
        xp = F.pad(x, (0, 0, 0, p_in - length)) if p_in > length else x
        spec = torch.fft.rfft(xp, dim=1)                     # ① (B, p_in/2+1, D)
        mixed = self._mix(spec)                              #    先混合（压缩域最便宜）
        spec_out = torch.zeros(batch, p_out // 2 + 1, dim, dtype=mixed.dtype, device=mixed.device)
        spec_out[:, : mixed.shape[1]] = mixed                # ② 谱零延拓 = 带限插值（截断的伴随）
        y = torch.fft.irfft(spec_out, n=p_out, dim=1)        # ③ (B, p_out, D)
        if self.residual:                                    # 可选：最近邻残差
            y = y + x.repeat_interleave(2, dim=1)
        return self.norm(y[:, :length_out])                  # ④ 裁回真实长度


class SkipFuse(nn.Module):
    """skip 融合：concat -> 深度可分离 k3（局部对账）-> PW(2D->D) -> 门控残差注入。"""

    def __init__(self, dim: int):
        super().__init__()
        self.dw = nn.Conv1d(2 * dim, 2 * dim, 3, padding=1, groups=2 * dim)
        self.pw = nn.Linear(2 * dim, dim, bias=False)
        self.gate = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.Sigmoid())
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        z = torch.cat([x, skip], dim=-1)                     # ① 拼接 (B, Ls, 2D)
        z = self.dw(z.transpose(1, 2)).transpose(1, 2)       # ② 局部对账
        fused = self.norm(self.pw(z))                        # ③ 通道融合
        return x + self.gate(x) * fused                      # ④ 门控注入


class FrequencyUNet(nn.Module):
    """频域 U-Net：隐藏维度不变，token 数量由频域核改变。

    Args:
        dim: 隐藏维度（恒定，默认 768）。
        depth: 上下采样次数（默认 4）。
        mid: 瓶颈注意力层数（默认 8）。
        heads: 注意力头数。
        mlp_ratio: FFN 隐层倍率。
        dropout: dropout 概率。
        rank: 频域混合的低秩维 r。
        control: 逐频率调制的控制点数 m（线性插值到实际 bin 数）。
    """

    def __init__(self, dim: int = 768, depth: int = 4, mid: int = 8, heads: int = 12,
                 mlp_ratio: float = 4.0, dropout: float = 0.0, rank: int = 64, control: int = 33,
                 up_residual: bool = False):
        super().__init__()
        self.dim, self.depth, self.mid_layers = dim, depth, mid
        self.enc_blocks = nn.ModuleList([SelfBlock(dim, heads, mlp_ratio, dropout) for _ in range(depth)])
        self.downs = nn.ModuleList([KernelDown(dim, rank, control) for _ in range(depth)])
        self.mid_blocks = nn.ModuleList([SelfBlock(dim, heads, mlp_ratio, dropout) for _ in range(mid)])
        self.ups = nn.ModuleList([KernelUp(dim, rank, control, up_residual) for _ in range(depth)])
        self.fuses = nn.ModuleList([SkipFuse(dim) for _ in range(depth)])
        self.dec_blocks = nn.ModuleList([SelfBlock(dim, heads, mlp_ratio, dropout) for _ in range(depth)])
        self.pe = SinusoidalPE(dim)

    def forward(self, x: torch.Tensor, return_features: bool = True) -> Sequence[torch.Tensor] | Tuple:
        """x: (B, L, D)（整段序列，长度任意）；返回 levels 由粗到细 [L/2^depth, ..., L]。"""
        lengths = [x.size(1)]
        skips: List[torch.Tensor] = []
        for enc, down in zip(self.enc_blocks, self.downs):    # ===== 编码 =====
            x = enc(x)
            skips.append(x)                                  # tap：下采样前，满分辨率
            x = down(x)
            lengths.append(x.size(1))                        # 记账：真实长度
            x = self.pe(x)
        for block in self.mid_blocks:                        # ===== 瓶颈（8 层）=====
            x = block(x)
        levels = [x]
        for i in reversed(range(self.depth)):                # ===== 解码 =====
            x = self.ups[i](x, lengths[i])                   # 2L -> 裁剪到 Ls[i]
            x = self.fuses[i](x, skips[i])
            x = self.dec_blocks[i](x)
            x = self.pe(x)
            levels.append(x)
        return (levels, x) if return_features else levels


def build_hier(name: str = "hier", **kwargs) -> FrequencyUNet:
    """按名称构建主干（v0.2 为频域 U-Net）。"""
    if name not in ("hier", "frequnet"):
        raise KeyError(f"未知主干 {name!r}，可选：['hier']")
    kwargs.pop("name", None)
    kwargs.pop("in_channels", None)
    return FrequencyUNet(**kwargs)
