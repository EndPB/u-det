"""Qwen2.5-Coder（decoder-only）的"编码器式"包装：causal 模型 → (B, L, D) 隐藏态。

按 d-det 编码器约定（见 encoders/__init__.py）：
    - 属性 hidden_size / num_layers；
    - forward(input_ids, attention_mask) -> (B, L, D)，padding 位置置零；
    - LoRA 路径：底座全冻结 + peft 适配器；``requires_grad_`` 被覆写为只作用于
      lora_ 参数（防训练脚本的全局解冻把底座打开）；
    - 梯度检查点开关 ckpt；冻结底座+ckpt 时自动打开输入梯度（peft 要求）。

注意：这是"把 decoder 当编码器用"的标准做法（取隐藏态池化做分类）；块大小/布 padding
由上层 pad_collate 控制（右 padding + mask，对 causal + mean-pooling 安全）。
"""

from __future__ import annotations

import inspect
import os

import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModel

_DTYPES = {"float32": torch.float32, "float16": torch.float16,
           "bfloat16": torch.bfloat16, "bf16": torch.bfloat16}


def _dtype_kwarg() -> str:
    try:
        params = inspect.signature(AutoModel.from_pretrained).parameters
        return "torch_dtype" if "torch_dtype" in params else "dtype"
    except (TypeError, ValueError):
        return "torch_dtype"


class QwenCoderEncoder(nn.Module):
    """Qwen2.5-Coder-1.5B（或任意 Qwen2 系）→ 隐藏态 (B, L, D)。

    Args:
        path: 本地模型目录（checkpoints/qwen2.5-coder-1.5b-instruct）。
        dtype: 底座权重精度（默认 bfloat16，省显存；配 autocast 使用）。
        layer: 取第几层隐状态（-1 最后一层）。
        r / alpha / dropout: peft LoRA 配置（dropout 建议 0——包装器训练时底座恒 eval）。
        targets: LoRA 注入模块名（Qwen2 注意力 + MLP 投影）。
        full_ft: 全参微调（不注入 LoRA、底座解冻；12GB 下不建议）。
        ckpt: 梯度检查点。
    """

    def __init__(
        self,
        path: str = "checkpoints/qwen2.5-coder-1.5b-instruct",
        dtype: str = "bfloat16",
        layer: int = -1,
        r: int = 32,
        alpha: int = 64,
        dropout: float = 0.0,
        targets=("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
        full_ft: bool = False,
        ckpt: bool = True,
        quant: str = "none",
        **ignored,          # 方便 cfg 里保留 codet5 系参数直接切 name
    ):
        super().__init__()
        torch_dtype = _DTYPES.get(dtype, dtype) if isinstance(dtype, str) else dtype
        load_kwargs = {"local_files_only": os.path.isdir(path)}
        quant = str(quant).lower()
        if quant in ("gptq", "int4", "4bit"):
            import importlib.util
            if not any(importlib.util.find_spec(m) for m in ("gptqmodel", "auto_gptq")):
                raise RuntimeError("GPTQ 量化加载需要 gptqmodel 或 auto-gptq（未安装）")
            base = AutoModel.from_pretrained(path, device_map={"": 0}, **load_kwargs)
        else:
            try:
                base = AutoModel.from_pretrained(path, **{_dtype_kwarg(): torch_dtype},
                                                 **load_kwargs)
            except (TypeError, ValueError):
                base = AutoModel.from_pretrained(path, **load_kwargs).to(torch_dtype)

        self._quant = quant in ("gptq", "int4", "4bit")
        self.full_ft = bool(full_ft)
        self.ckpt = bool(ckpt)
        if self.full_ft:
            base.requires_grad_(True)
            self.model = base
        else:
            base.requires_grad_(False)
            lora_cfg = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=int(r), lora_alpha=int(alpha), lora_dropout=float(dropout),
                bias="none", target_modules=[str(t) for t in targets],
            )
            self.model = get_peft_model(base, lora_cfg)
            if self.ckpt and hasattr(self.model, "enable_input_require_grads"):
                self.model.enable_input_require_grads()   # 冻结底座 + ckpt 的必要条件
            # ★ 不强制 eval：Qwen2 无内置 dropout（attention_dropout=0），train/eval 数值等价；
            #   保持 train 态才能让 HF 梯度检查点在训练时生效（与 CodeT5 包装不同）。
        if self.ckpt:
            try:
                self.model.gradient_checkpointing_enable()
                self.model.config.use_cache = False
            except Exception as e:  # 不阻断；一般只在特殊 peft/transformers 组合下告警
                print(f"[qwen] gradient_checkpointing_enable 告警：{e}")

        self.n_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        if self.n_trainable == 0:
            raise ValueError("Qwen 编码器无可训练参数（LoRA 未命中 targets？）")
        self._d_model = int(base.config.hidden_size)
        self._n_layers = int(base.config.num_hidden_layers)
        self.layer = int(layer)
        print(f"[qwen] Qwen2.5-Coder {'全参微调' if self.full_ft else f'LoRA(r={r}, alpha={alpha})'}："
              f"可训练 {self.n_trainable / 1e6:.3f}M，{self._n_layers} 层"
              f"{'解冻' if self.full_ft else '冻结'}（causal → (B,L,D)）；ckpt={self.ckpt}")

    # ------------------------------------------------------------------ #
    @property
    def hidden_size(self) -> int:
        return self._d_model

    @property
    def num_layers(self) -> int:
        return self._n_layers

    def train(self, mode: bool = True):
        """LoRA 路径与全参路径都直接跟随（Qwen2 dropout=0，语义安全；ckpt 需要 train 态）。"""
        super().train(mode)
        return self

    def requires_grad_(self, requires_grad: bool = True):
        """★ 覆写：LoRA 路径只作用于 lora_ 参数，底座恒冻。"""
        if getattr(self, "full_ft", False):
            return super().requires_grad_(requires_grad)
        for name, p in self.model.named_parameters():
            if "lora_" in name:
                p.requires_grad = bool(requires_grad)
        return self

    def to(self, *args, **kwargs):
        """★ 覆写：GPTQ 量化底座固定在 device_map 指定卡上，不参与搬家。"""
        if getattr(self, "_quant", False):
            for name, mod in self.named_children():
                if name != "model":
                    mod.to(*args, **kwargs)
            return self
        return super().to(*args, **kwargs)

    def forward(self, input_ids: torch.Tensor = None,
                attention_mask: torch.Tensor = None, **kwargs) -> torch.Tensor:
        """(B, L) -> (B, L, D)，padding 位置置零。"""
        batch, length = input_ids.shape
        if length == 0:
            return input_ids.new_zeros(batch, 0, self._d_model).to(torch.float32)
        if attention_mask is None:
            attention_mask = input_ids.new_ones(batch, length)
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=self.layer != -1,
            return_dict=True,
        )
        hidden = out.last_hidden_state if self.layer == -1 else out.hidden_states[self.layer]
        return hidden * attention_mask.unsqueeze(-1).to(hidden.dtype)
