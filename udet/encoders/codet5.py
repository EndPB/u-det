"""CodeT5 编码器（基于 transformers 的 ``T5EncoderModel``）。

CodeT5 基于 T5 架构，本模块只使用其 **encoder** 部分作为特征提取器：

* 支持从 Hugging Face Hub（``Salesforce/codet5-base``）或本地目录加载权重；
* 支持冻结 / 部分冻结、梯度检查点、混合精度加载；
* 支持输出全部层隐状态（供多尺度跳连 / 后续魔改使用）；
* 输出统一的 :class:`~udet.encoders.base.EncoderOutput`。
"""

from __future__ import annotations

import os
from typing import Any, List, Optional, Sequence, Union

import torch
import torch.nn as nn

try:  # transformers 版本兼容
    from transformers import T5EncoderModel
except ImportError as exc:  # pragma: no cover
    raise ImportError("请先安装 transformers: pip install transformers") from exc

from .base import BaseEncoder, EncoderOutput
from .registry import register_encoder

_DTYPE_MAP = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "half": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}


def resolve_dtype(dtype: Union[str, torch.dtype, None]) -> Optional[torch.dtype]:
    """把字符串 / torch.dtype 统一成 torch.dtype；None 表示保持默认。"""
    if dtype is None or isinstance(dtype, torch.dtype):
        return dtype
    key = str(dtype).lower()
    if key not in _DTYPE_MAP:
        raise ValueError(f"不支持的 dtype: {dtype!r}，可选 {sorted(_DTYPE_MAP)}")
    return _DTYPE_MAP[key]


def _load_pretrained(model_cls, path: str, torch_dtype: Optional[torch.dtype], **kwargs):
    """兼容不同 transformers 版本的权重加载（torch_dtype / dtype 迁移）。"""
    if torch_dtype is None:
        return model_cls.from_pretrained(path, **kwargs)
    base_kwargs = dict(kwargs)
    try:
        # transformers >= 4.56 推荐写法
        return model_cls.from_pretrained(path, dtype=torch_dtype, **base_kwargs)
    except (TypeError, ValueError):
        try:
            # 旧版本写法（新版会有 FutureWarning，但可用）
            return model_cls.from_pretrained(path, torch_dtype=torch_dtype, **base_kwargs)
        except (TypeError, ValueError):
            # 兜底：以默认精度加载后再转换
            model = model_cls.from_pretrained(path, **base_kwargs)
            return model.to(torch_dtype)


@register_encoder("codet5")
class CodeT5Encoder(BaseEncoder):
    """CodeT5 编码器封装。

    Args:
        model_name_or_path: 本地权重目录（推荐，如 ``checkpoints/codet5-base``）
            或 Hub 上的名字（``Salesforce/codet5-base``）。
        torch_dtype: 加载精度，``float32`` / ``float16`` / ``bfloat16``。
        freeze: 是否冻结全部参数。
        freeze_embeddings: 是否冻结词嵌入。
        freeze_layers: 冻结前 N 层 Transformer。
        output_hidden_states: 是否返回每一层的隐状态列表。
        gradient_checkpointing: 是否开启梯度检查点（省显存、略慢）。
        layer_selection: ``"last"``、``"all"`` 或层索引列表（0 表示嵌入层输出，1..N 表示第 i 层输出）。
        pooling: 句级池化方式，``"mean"`` 或 ``"max"``。
        max_length: 模型最大输入长度。
    """

    def __init__(
        self,
        model_name_or_path: str = "checkpoints/codet5-base",
        torch_dtype: Union[str, torch.dtype, None] = "float32",
        freeze: bool = False,
        freeze_embeddings: bool = False,
        freeze_layers: int = 0,
        output_hidden_states: bool = True,
        gradient_checkpointing: bool = False,
        layer_selection: Union[str, Sequence[int]] = "last",
        pooling: str = "mean",
        max_length: int = 512,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        self._pooling = pooling
        self.max_length = int(max_length)
        self.layer_selection = layer_selection
        self.output_hidden_states = bool(output_hidden_states)

        dtype = resolve_dtype(torch_dtype)
        self._torch_dtype = dtype or torch.float32

        load_kwargs: dict = {}
        if os.path.sep in model_name_or_path or model_name_or_path.startswith("."):
            # 本地路径：避免联网检查
            load_kwargs["local_files_only"] = True
        load_kwargs.update(kwargs)

        self.encoder = _load_pretrained(T5EncoderModel, model_name_or_path, dtype, **load_kwargs)
        self.config = self.encoder.config

        if gradient_checkpointing:
            self.encoder.gradient_checkpointing_enable()
            self.gradient_checkpointing = True
        else:
            self.gradient_checkpointing = False

        # ---- 冻结策略 ----
        if freeze_embeddings:
            self.encoder.shared.requires_grad_(False)
        if freeze or (freeze_layers and freeze_layers >= self.num_layers):
            self.freeze()
        elif freeze_layers > 0:
            self.freeze_layers(freeze_layers)
            self._frozen = False

    # ------------------------------------------------------------------ #
    # 属性
    # ------------------------------------------------------------------ #
    @property
    def hidden_size(self) -> int:
        return int(self.config.d_model)

    @property
    def num_layers(self) -> int:
        return int(self.config.num_layers)

    @property
    def num_heads(self) -> int:
        return int(self.config.num_heads)

    @property
    def dtype(self) -> torch.dtype:
        return self._torch_dtype

    # ------------------------------------------------------------------ #
    # 层选择
    # ------------------------------------------------------------------ #
    def _select_layers(self, all_states: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        sel = self.layer_selection
        if sel is None or sel == "last":
            return [all_states[-1]]
        if sel == "all":
            return list(all_states)
        if isinstance(sel, (list, tuple)):
            idxs = [int(i) for i in sel]
            for i in idxs:
                if not 0 <= i < len(all_states):
                    raise IndexError(
                        f"layer_selection 索引 {i} 越界（共 {len(all_states)} 个状态，0=嵌入层）"
                    )
            return [all_states[i] for i in idxs]
        raise ValueError(f"不支持的 layer_selection: {sel!r}")

    # ------------------------------------------------------------------ #
    # 前向
    # ------------------------------------------------------------------ #
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        return_hidden_states: Optional[bool] = None,
        return_pooled: bool = True,
        **kwargs: Any,
    ) -> EncoderOutput:
        """前向计算。

        Args:
            input_ids: ``(B, L)`` token id。
            attention_mask: ``(B, L)``，1 为有效 token。
            inputs_embeds: 直接传入嵌入向量（与 ``input_ids`` 二选一）。
            return_hidden_states: 覆盖实例默认设置，是否返回各层隐状态。
            return_pooled: 是否计算句级池化输出。
        """
        need_hidden = self.output_hidden_states if return_hidden_states is None else bool(return_hidden_states)

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=need_hidden,
            return_dict=True,
        )

        all_states: Sequence[torch.Tensor]
        if need_hidden and getattr(outputs, "hidden_states", None) is not None:
            all_states = outputs.hidden_states
            selected = self._select_layers(all_states)
            hidden_states = selected if len(selected) > 1 else selected
            last_hidden = selected[-1]
        else:
            last_hidden = outputs.last_hidden_state
            hidden_states = None

        pooled = None
        if return_pooled:
            if self._pooling == "max" and attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
                pooled = (last_hidden * mask + (1.0 - mask) * torch.finfo(last_hidden.dtype).min).max(dim=1).values
            else:
                pooled = self._masked_mean(last_hidden, attention_mask)

        return EncoderOutput(
            last_hidden_state=last_hidden,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            pooled_output=pooled,
        )

    # ------------------------------------------------------------------ #
    # 便捷方法
    # ------------------------------------------------------------------ #
    def get_input_embeddings(self) -> nn.Module:
        """返回词嵌入层（供拼接手工特征等魔改使用）。"""
        return self.encoder.get_input_embeddings()

    def enable_gradient_checkpointing(self) -> None:
        self.encoder.gradient_checkpointing_enable()
        self.gradient_checkpointing = True

    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.hidden_size}, num_layers={self.num_layers}, "
            f"dtype={self._torch_dtype}, pooling={self._pooling}, "
            f"layer_selection={self.layer_selection}"
        )
