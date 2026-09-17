"""编码器统一接口。

任何新编码器只需继承 :class:`BaseEncoder` 并实现 ``forward`` / ``hidden_size``，
再用 ``@register_encoder("name")`` 注册，即可通过 ``build_encoder("name", ...)`` 构建，
从而在配置文件中一键切换。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import torch
import torch.nn as nn


@dataclass
class EncoderOutput:
    """编码器统一输出。

    Attributes:
        last_hidden_state: ``(B, L, D)`` 最后一层（或选定层）的 token 级表示。
        hidden_states: 选定的各层隐状态列表，每个为 ``(B, L, D)``；未开启则为 ``None``。
        attention_mask: ``(B, L)``，1 表示有效 token，0 表示 padding。
        pooled_output: ``(B, D)`` 句级表示（默认按 mask 做平均池化），可选。
    """

    last_hidden_state: torch.Tensor
    hidden_states: Optional[Sequence[torch.Tensor]] = None
    attention_mask: Optional[torch.Tensor] = None
    pooled_output: Optional[torch.Tensor] = field(default=None)

    @property
    def sequence_length(self) -> int:
        return self.last_hidden_state.size(1)

    @property
    def hidden_size(self) -> int:
        return self.last_hidden_state.size(-1)


class BaseEncoder(nn.Module, ABC):
    """所有编码器的抽象基类。

    子类需要实现：

    * ``forward(input_ids, attention_mask=None, **kwargs) -> EncoderOutput``
    * ``hidden_size`` 属性（输出维度 D）
    * ``num_layers`` 属性（可选的层数信息）
    """

    #: 编码器可接受的最长输入长度（用于数据侧截断策略）
    max_length: int = 512

    def __init__(self) -> None:
        super().__init__()
        self._frozen = False

    # ------------------------------------------------------------------ #
    # 子类必须实现
    # ------------------------------------------------------------------ #
    @property
    @abstractmethod
    def hidden_size(self) -> int:
        """输出隐状态维度 D。"""

    @abstractmethod
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> EncoderOutput:
        """前向计算，返回 :class:`EncoderOutput`。"""

    # ------------------------------------------------------------------ #
    # 通用能力
    # ------------------------------------------------------------------ #
    @property
    def num_layers(self) -> int:
        """Transformer 层数；未知时返回 0。"""
        return 0

    def freeze(self) -> "BaseEncoder":
        """冻结全部参数（不参与梯度更新）。"""
        for p in self.parameters():
            p.requires_grad = False
        self._frozen = True
        return self

    def unfreeze(self) -> "BaseEncoder":
        """解冻全部参数。"""
        for p in self.parameters():
            p.requires_grad = True
        self._frozen = False
        return self

    def freeze_layers(self, n: int) -> "BaseEncoder":
        """冻结前 ``n`` 层（含词嵌入）。子类可覆盖以实现更细粒度控制。

        默认按参数名匹配 ``layer.{i}.``（BERT 系）或 ``block.{i}.``（T5 系）。
        """
        if n <= 0:
            return self
        patterns = []
        for i in range(n):
            patterns.extend([f"layer.{i}.", f"block.{i}."])
        for name, p in self.named_parameters():
            if any(pat in name for pat in patterns):
                p.requires_grad = False
        return self

    def trainable_parameters(self):
        """返回可训练参数（用于优化器分组）。"""
        return (p for p in self.parameters() if p.requires_grad)

    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @staticmethod
    def _masked_mean(hidden: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """按 attention mask 做平均池化，返回 ``(B, D)``。"""
        if mask is None:
            return hidden.mean(dim=1)
        mask = mask.to(hidden.dtype).unsqueeze(-1)  # (B, L, 1)
        denom = mask.sum(dim=1).clamp(min=1e-6)
        return (hidden * mask).sum(dim=1) / denom


def freeze_module(module: nn.Module) -> None:
    """工具函数：冻结任意 nn.Module 的全部参数。"""
    for p in module.parameters():
        p.requires_grad = False
