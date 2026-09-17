"""通用 HuggingFace 编码器：任何 ``AutoModel`` 可加载的编码器都能直接接入。

用途：在不改代码的前提下切换编码器（BERT / RoBERTa / GraphCodeBERT / CodeBERT / T5 系 ...）。

示例::

    build_encoder("hf", model_name_or_path="microsoft/graphcodebert-base")
    build_encoder("hf", model_name_or_path="Salesforce/codet5p-220m")
"""

from __future__ import annotations

import os
from typing import Any, Optional, Sequence, Union

import torch

try:
    from transformers import AutoConfig, AutoModel, AutoModelForSeq2SeqLM
except ImportError as exc:  # pragma: no cover
    raise ImportError("请先安装 transformers: pip install transformers") from exc

from .base import BaseEncoder, EncoderOutput
from .codet5 import _load_pretrained, resolve_dtype
from .registry import register_encoder


@register_encoder("hf")
class HuggingFaceEncoder(BaseEncoder):
    """任意 HuggingFace 编码器的统一封装。

    对于 encoder-decoder 模型（如 T5 系列）会自动只取 encoder 部分。

    Args:
        model_name_or_path: 本地目录或 Hub 名称。
        torch_dtype: 加载精度。
        freeze / freeze_embeddings / freeze_layers: 冻结策略。
        output_hidden_states: 是否返回各层隐状态。
        layer_selection: ``"last"`` / ``"all"`` / 层索引列表。
        pooling: ``"mean"`` 或 ``"max"``。
        trust_remote_code: 是否信任 Hub 上的自定义代码。
        max_length: 模型最大输入长度。
    """

    def __init__(
        self,
        model_name_or_path: str,
        torch_dtype: Union[str, torch.dtype, None] = "float32",
        freeze: bool = False,
        freeze_embeddings: bool = False,
        freeze_layers: int = 0,
        output_hidden_states: bool = True,
        gradient_checkpointing: bool = False,
        layer_selection: Union[str, Sequence[int]] = "last",
        pooling: str = "mean",
        trust_remote_code: bool = False,
        max_length: int = 512,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        self._pooling = pooling
        self.max_length = int(max_length)
        self.layer_selection = layer_selection
        self.output_hidden_states = bool(output_hidden_states)
        self._torch_dtype = resolve_dtype(torch_dtype) or torch.float32

        load_kwargs: dict = {"trust_remote_code": trust_remote_code}
        if os.path.sep in model_name_or_path or model_name_or_path.startswith("."):
            load_kwargs["local_files_only"] = True
        load_kwargs.update(kwargs)

        config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)

        if getattr(config, "is_encoder_decoder", False):
            seq2seq = _load_pretrained(AutoModelForSeq2SeqLM, model_name_or_path, self._torch_dtype, **load_kwargs)
            self.encoder = seq2seq.get_encoder()
        else:
            self.encoder = _load_pretrained(AutoModel, model_name_or_path, self._torch_dtype, **load_kwargs)

        self.config = config

        if gradient_checkpointing:
            self.encoder.gradient_checkpointing_enable()

        if freeze_embeddings:
            emb = self.encoder.get_input_embeddings()
            if emb is not None:
                emb.requires_grad_(False)
        if freeze or (freeze_layers and freeze_layers >= self.num_layers):
            self.freeze()
        elif freeze_layers > 0:
            self.freeze_layers(freeze_layers)

    # ------------------------------------------------------------------ #
    @property
    def hidden_size(self) -> int:
        for key in ("d_model", "hidden_size", "dim"):
            if hasattr(self.config, key):
                return int(getattr(self.config, key))
        raise AttributeError("无法从 config 推断 hidden_size")

    @property
    def num_layers(self) -> int:
        for key in ("num_layers", "num_hidden_layers", "n_layer"):
            if hasattr(self.config, key):
                return int(getattr(self.config, key))
        return 0

    # ------------------------------------------------------------------ #
    def _select_layers(self, all_states: Sequence[torch.Tensor]):
        sel = self.layer_selection
        if sel is None or sel == "last":
            return [all_states[-1]]
        if sel == "all":
            return list(all_states)
        if isinstance(sel, (list, tuple)):
            return [all_states[int(i)] for i in sel]
        raise ValueError(f"不支持的 layer_selection: {sel!r}")

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        return_hidden_states: Optional[bool] = None,
        return_pooled: bool = True,
        **kwargs: Any,
    ) -> EncoderOutput:
        need_hidden = self.output_hidden_states if return_hidden_states is None else bool(return_hidden_states)

        call_kwargs: dict = {
            "attention_mask": attention_mask,
            "output_hidden_states": need_hidden,
            "return_dict": True,
        }
        # 部分模型（BERT 系）不接受 inputs_embeds=None 之外的输入，做最小兼容
        if input_ids is not None:
            call_kwargs["input_ids"] = input_ids
        if inputs_embeds is not None:
            call_kwargs["inputs_embeds"] = inputs_embeds

        outputs = self.encoder(**call_kwargs)

        if need_hidden and getattr(outputs, "hidden_states", None) is not None:
            selected = self._select_layers(outputs.hidden_states)
            last_hidden = selected[-1]
            hidden_states: Optional[Sequence[torch.Tensor]] = selected
        else:
            last_hidden = outputs.last_hidden_state
            hidden_states = None

        pooled = None
        if return_pooled:
            pooled = self._masked_mean(last_hidden, attention_mask)

        return EncoderOutput(
            last_hidden_state=last_hidden,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            pooled_output=pooled,
        )

    def get_input_embeddings(self):
        return self.encoder.get_input_embeddings()

    def extra_repr(self) -> str:
        return f"hidden_size={self.hidden_size}, num_layers={self.num_layers}, dtype={self._torch_dtype}"
