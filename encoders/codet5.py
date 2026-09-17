"""CodeT5 编码器（单文件实现）。

只使用 CodeT5 的 encoder 部分作为特征提取器：
    * 默认从 checkpoints/codet5-base 加载（由 scripts/prepare.py 下载）；
    * 返回 token 级表示 (B, L, D)，可直接喂给 models/unet.py；
    * 支持冻结、混合精度加载、输出各层隐状态（多尺度跳连备用）。
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
from transformers import T5EncoderModel

_DTYPES = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}


def _dtype_kwarg() -> str:
    """transformers >= 4.56 用 ``dtype=``，旧版本用 ``torch_dtype=``。"""
    import transformers

    try:
        major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
    except Exception:  # noqa: BLE001
        return "torch_dtype"
    return "dtype" if (major, minor) >= (4, 56) else "torch_dtype"


class CodeT5Encoder(nn.Module):
    """CodeT5 编码器。

    Args:
        path: 权重目录（默认 checkpoints/codet5-base）或 HF 模型名（Salesforce/codet5-base）。
        dtype: 加载精度，float32 / float16 / bfloat16。
        freeze: 是否冻结全部参数。
        gradient_checkpointing: 是否开启梯度检查点（省显存）。
        max_length: 最大输入长度。
    """

    def __init__(
        self,
        path: str = "checkpoints/codet5-base",
        dtype: str = "float32",
        freeze: bool = False,
        gradient_checkpointing: bool = False,
        max_length: int = 512,
    ):
        super().__init__()
        self.path = path
        self.max_length = max_length

        torch_dtype = _DTYPES.get(dtype, dtype) if isinstance(dtype, str) else dtype
        load_kwargs = {"local_files_only": os.path.isdir(path)}
        try:
            self.model = T5EncoderModel.from_pretrained(
                path, **{_dtype_kwarg(): torch_dtype}, **load_kwargs
            )
        except (TypeError, ValueError):
            # 兜底：按默认精度加载后转换
            self.model = T5EncoderModel.from_pretrained(path, **load_kwargs).to(torch_dtype)

        if gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        if freeze:
            self.freeze()

    # ------------------------------------------------------------------ #
    @property
    def hidden_size(self) -> int:
        """输出维度 D（CodeT5-base 为 768）。"""
        return self.model.config.d_model

    @property
    def num_layers(self) -> int:
        return self.model.config.num_layers

    @property
    def vocab_size(self) -> int:
        return self.model.config.vocab_size

    def freeze(self):
        """冻结全部参数。"""
        for p in self.model.parameters():
            p.requires_grad = False

    def unfreeze(self):
        """解冻全部参数。"""
        for p in self.model.parameters():
            p.requires_grad = True

    # ------------------------------------------------------------------ #
    def forward(
        self,
        input_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        inputs_embeds: torch.Tensor = None,
        output_hidden_states: bool = False,
    ) -> torch.Tensor:
        """前向计算。

        Args:
            input_ids: (B, L) token id。
            attention_mask: (B, L)，1 为有效 token（可选）。
            inputs_embeds: 直接传入嵌入（与 input_ids 二选一）。
            output_hidden_states: True 时返回各层隐状态元组（含嵌入层，共 num_layers+1 个）。

        Returns:
            (B, L, D) 的 token 级表示；或 hidden_states 元组。
        """
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        if output_hidden_states:
            return out.hidden_states
        return out.last_hidden_state

    # ------------------------------------------------------------------ #
    def get_input_embeddings(self) -> nn.Module:
        """词嵌入层（拼接手工特征等改动用）。"""
        return self.model.get_input_embeddings()
