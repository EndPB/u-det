"""直接二分类基线（单文件实现）：编码器 + 池化/滑窗池化 + 分类头。

与 U-Det 的差别只在"是否用 U-Net + 双任务"。两种基线：

    PooledClassifier          按 ``max_length`` **截断**后 masked 池化（原始基线，m4 0.9756）
    WindowedContextClassifier **滑动窗口全长覆盖** + 样本头 + 可选逐 token 头
                              （CodeT5 只有 512 位置上限，截断只能看到开头一小段，
                              而 hybrid 中位 1274 / max 10475 ⇒ 行级/片段级必须全长覆盖）

用法::

    from models import PooledClassifier, WindowedContextClassifier

    clf = PooledClassifier(encoder, pooling="mean", hidden=None, out=2)
    logits, _ = clf(input_ids, attention_mask)      # (B, 2)；第二个返回值恒为 None（无 token 头）

    clf = WindowedContextClassifier(encoder, window=512, stride=256, token_head=True)
    logits, (tok,) = clf(input_ids, attention_mask)  # (B, 2) 与 (B, L)，L = 真实长度
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class PooledClassifier(nn.Module):
    """编码器 + masked 池化 + 分类头（无 U-Net、无 token 级任务）。

    Args:
        encoder: 任意满足 ``forward(input_ids, attention_mask) -> (B, L, D)`` 的编码器。
        dim: 特征维度（默认取 ``encoder.hidden_size``）。
        pooling: ``mean``（masked 平均）/ ``max``（masked 最大）/ ``first``（取首 token）。
        hidden: 分类头隐层维度，None 表示直接线性投影。
        out: 类别数。
        dropout: dropout 概率。
    """

    #: 序列长度补齐倍数（与 UDet 的接口保持一致，基线不做多尺度池化，取 8 即可）
    align_multiple = 8

    def __init__(
        self,
        encoder: nn.Module,
        dim: Optional[int] = None,
        pooling: str = "mean",
        hidden: Optional[int] = None,
        out: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        if pooling not in ("mean", "max", "first"):
            raise ValueError(f"未知池化方式 {pooling!r}，可选：mean / max / first")
        self.encoder = encoder
        self.pooling = pooling
        dim = int(dim or encoder.hidden_size)
        self.norm = nn.LayerNorm(dim)
        layer = [] if not hidden else [nn.Linear(dim, int(hidden)), nn.GELU(), nn.Dropout(dropout)]
        layer += [nn.Linear(int(hidden) if hidden else dim, out)]
        self.net = nn.Sequential(*layer)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        if attention_mask is None:                                      # batch=1 整段时无 padding
            attention_mask = torch.ones_like(input_ids)
        hidden = self.encoder(input_ids, attention_mask)                 # (B, L, D)
        mask = attention_mask[:, : hidden.shape[1]].unsqueeze(-1).to(hidden.dtype)   # 编码器可能已截断
        if self.pooling == "mean":
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        elif self.pooling == "max":
            pooled = (hidden * mask + (mask - 1) * 1e4).max(1).values    # padding 位置压到 -1e4
        else:
            pooled = hidden[:, 0]
        return self.net(self.norm(pooled)), None                         # 与 UDet 保持同接口


class WindowedContextClassifier(nn.Module):
    """整段覆盖（滑动窗口）的上下文分类器：样本头 + 可选的逐 token 头。

    与 `PooledClassifier` 的差别只有一点：**不截断**。CodeT5 的位置上限是 512，
    `CodeT5Encoder.forward` 会硬截断，而 hybrid 长度中位 1274 / max 10475、
    m4 `max` 32948 —— 截断只能看到开头一小段。偏偏 `evaluate()` 是按**全文件**
    算行级/片段级 F1 的（`zip(probs, tok_labels, line_of_token)` 逐位对齐），
    所以截断的结果与 U-Det 不能比。

    窗口在**模型内部**攒批：`evaluate()` 逐样本调用（恒 batch=1），
    逐样本跑几十个窗口会慢到不可用；这里把所有窗口拼成 `(N_win, W)` 分批过编码器。

    三个正确性要点（错一个都会**静默**出错）::

        1. 返回的 token logits 长度必须**恰好等于输入长度 L**
           （evaluate 用 zip 逐位对齐，长度不符是整体错位，不会报错）；
        2. 重叠区对 **logits** 取平均，不是对特征平均（对特征平均会改变尺度）；
        3. 尾部不足一窗时补 0 并用 attention_mask 屏蔽，不能当真实 token。

    参数
    ----
    encoder
        `CodeT5Encoder`（它自己按 ``max_length`` 截断，正好当每个窗口的护栏）。
    dim
        特征维度（默认取 ``encoder.hidden_size``）。
    window
        窗口长度。``<= 0`` 表示**不切窗**（退化成单窗口 + 截断）；配合
        ``token_head=False`` 可逐位复现 `PooledClassifier` 的结果（回归测试用）。
    stride
        窗口步长，默认 ``window // 2``（50% 重叠，接缝平滑）。设成 ``window``
        即不重叠（只做池化时够用、省一半算力）。
    window_batch
        一次过编码器的窗口数（控制显存）。
    token_head
        是否加逐 token 头；True 时返回 ``[token_logits]``（长度 = L），
        False 时返回 ``None``（与 `PooledClassifier` 同接口）。
    pooling / hidden / out / dropout
        与 `PooledClassifier` 同义（只作用于样本头）。

    按流覆盖步长
    ------------
    类的属性 ``accepts_stride = True`` 声明本模块接受 ``forward(..., stride=)``。
    样本级流（m4）不需要重叠 —— 只做池化时接缝毫无意义，而 50% 重叠会让窗口数翻倍。
    因此 `train.py` 会按 ``model.baseline.stride_by_stream`` 逐流传入：
    ``m4 -> 512``（不重叠，省一半）、``hybrid -> 256``（50% 重叠，逐 token 接缝平滑）。
    做成**显式参数**而非可变的内部状态：忘传时只会回退到默认步长（结果仍然正确、
    只是更慢），不会出现"训练用一种步长、评测用另一种"的静默不一致。
    """

    #: 声明接受 forward(..., stride=)（train.py 按此属性决定要不要传）
    accepts_stride = True

    def __init__(self, encoder: nn.Module, dim: Optional[int] = None, window: int = 512,
                 stride: int = 0, window_batch: int = 32, token_head: bool = True,
                 pooling: str = "mean", hidden: Optional[int] = None, out: int = 2,
                 dropout: float = 0.0):
        super().__init__()
        if pooling not in ("mean", "max"):
            raise ValueError(f"窗口版只支持 pooling=mean/max（收到 {pooling!r}）")
        self.encoder = encoder                                  # 必须叫 encoder：LR 分组与 ckpt 过滤都按这个名字
        self.window = int(window or 0)
        self.stride = int(stride) if stride else max(1, self.window // 2)
        self.window_batch = max(1, int(window_batch))
        self.pooling = pooling
        dim = int(dim or encoder.hidden_size)
        self.norm = nn.LayerNorm(dim)
        layer = [] if not hidden else [nn.Linear(dim, int(hidden)), nn.GELU(), nn.Dropout(dropout)]
        layer += [nn.Linear(int(hidden) if hidden else dim, out)]
        self.net = nn.Sequential(*layer)
        self.token_head = nn.Linear(dim, 1) if token_head else None

    # ------------------------------------------------------------------ #
    def _starts(self, length: int, stride: int | None = None) -> list[int]:
        """窗口起点：首窗贴左端、末窗贴右端、无重复（保证每个位置至少被覆盖一次）。

        ``stride`` 给定时覆盖 ``self.stride``（用于按流区分：样本级流不需要重叠）。
        """
        stride = int(stride) if stride else self.stride
        if self.window > 0:
            stride = max(1, min(stride, self.window))       # 步长 > 窗口会漏掉位置，直接夹住
        if self.window <= 0 or length <= self.window:
            return [0]
        last = length - self.window
        starts = list(range(0, last + 1, stride))
        if starts[-1] != last:
            starts.append(last)
        return starts

    def _encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                stride: int | None = None):
        """切窗 -> 分批过编码器，返回窗口级特征与原序列下标。

        Returns:
            ``(h, batch_idx, adv, valid, length)``：
            ``h`` 是 ``(N_win, W_e, D)``（``W_e`` 是编码器实际输出的窗口长，可能被 max_length 截断）；
            ``adv`` 是 ``(N_win, W_e)`` 的**原序列**下标；``valid`` 标出真实 token。
        """
        batch, length = input_ids.shape
        device = input_ids.device
        starts = self._starts(length, stride)
        width = self.window if self.window > 0 else length
        n_win = batch * len(starts)
        batch_idx = torch.arange(batch, device=device).repeat_interleave(len(starts))
        adv = (torch.tensor(starts, device=device).repeat(batch)[:, None]
               + torch.arange(width, device=device)[None, :])               # (N_win, width)
        valid = adv < length
        safe = adv.clamp(max=max(length - 1, 0))
        ids = input_ids[batch_idx[:, None], safe] * valid                   # 补 [PAD] = 0
        mask = attention_mask[batch_idx[:, None], safe] * valid

        feats = [self.encoder(ids[i:i + self.window_batch], mask[i:i + self.window_batch])
                 for i in range(0, n_win, self.window_batch)]
        h = torch.cat(feats, 0) if len(feats) > 1 else feats[0]
        width_enc = h.shape[1]                                             # 可能被 max_length 截断
        return h, batch_idx, adv[:, :width_enc], valid[:, :width_enc], length

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                stride: Optional[int] = None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        batch = input_ids.shape[0]
        h, batch_idx, adv, valid, length = self._encode(input_ids, attention_mask, stride)
        n_win = h.shape[0]

        token_logits = None
        if self.token_head is not None:
            tok = self.token_head(h).squeeze(-1)                            # (N_win, W_e)
            flat = (batch_idx[:, None] * length + adv)[valid]               # 展平下标（去重前）
            total = torch.zeros(batch * length, device=h.device, dtype=h.dtype)
            total = total.index_add(0, flat, tok[valid])                    # 重叠区求和
            count = torch.zeros(batch * length, device=h.device, dtype=h.dtype)
            count = count.index_add(0, flat, torch.ones_like(tok[valid]))
            token_logits = (total / count.clamp(min=1)).view(batch, length)  # 重叠区取平均

        w = valid.unsqueeze(-1).to(h.dtype)                                 # (N_win, W_e, 1)
        if self.pooling == "mean":
            pooled_win = (h * w).sum(1) / w.sum(1).clamp(min=1)             # 逐窗池化
        else:
            pooled_win = (h * w + (w - 1) * 1e4).max(1).values
        pooled_sum = torch.zeros(batch, h.shape[-1], device=h.device, dtype=h.dtype)
        pooled_sum = pooled_sum.index_add(0, batch_idx, pooled_win)         # 同一(样本)的多个窗求和
        pooled_cnt = torch.zeros(batch, device=h.device, dtype=h.dtype)
        pooled_cnt = pooled_cnt.index_add(
            0, batch_idx, torch.ones(n_win, device=h.device, dtype=h.dtype))
        pooled = pooled_sum / pooled_cnt.clamp(min=1).unsqueeze(-1)         # 再对窗口取平均
        sample_logits = self.net(self.norm(pooled))
        return sample_logits, ([token_logits] if token_logits is not None else None)
