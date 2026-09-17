"""CodeT5 编码器（单文件实现）。

两种模式（对应 u-det 的消融开关 ``encoder.name``）：

    codet5     上下文编码：整个序列进编码器，返回 (B, L, D)（受 512 位置/attention 限制）。
    codet5tok  逐 token 编码：每个 token 独立过冻结编码器（长度=1），可预算成查表；
               训练零编码开销、长度无上限（U-Det 默认）。

两者都只使用 CodeT5 的 encoder 部分：
    * 默认从 checkpoints/codet5-base 加载（由 scripts/prepare.py 下载）；
    * 返回 token 级表示 (B, L, D)，可直接嗂给 models/unet.py。
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
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


class CodeT5TokenEncoder(nn.Module):
    """逐 token 独立编码（长度=1）的 CodeT5。

    每个 token 单独送进编码器（序列长度 1、无上下文），等价于一个固定函数
    f(token)：attention 无交互，只剩逐 token 的非线性变换。因此可以预先算好
    (vocab, D) 查表（LUT）缓存到磁盘，前向退化为一次 ``F.embedding``：

        * 训练零编码开销（不用加载 850MB 权重、不用跑 transformer）；
        * 输入长度无上限（不受 n_positions / attention O(L^2) 限制），
          长短程上下文完全交给后面的 U-Net 与瓶颈 Transformer。

    两种模式（``freeze``）：
        freeze=True （默认）LUT 作为固定 buffer（非持久，不进 state_dict）；
        freeze=False       LUT 作为可训练嵌入表（用编码器逐 token 特征初始化），
                          等价于“不冻结的编码器”，但参数量只有 24.7M 且无 O(L²) 开销。

    Args:
        path: 权重目录（默认 checkpoints/codet5-base）或 HF 模型名。
        dtype: 构建 LUT 时加载权重用的精度（缓存后失效）。
        lut: LUT 缓存路径；存在则直接加载（此时不加载编码器权重）。
        layer: 取第几层隐状态（-1 = 最后一层，0 = 词嵌入本身）。
        chunk: 构建 LUT 时的分块大小（vocab 逐块前向，控制显存）。
        freeze: True = LUT 固定；False = LUT 可训练（微调）。
    """

    def __init__(
        self,
        path: str = "checkpoints/codet5-base",
        dtype: str = "float32",
        lut: str = "data/processed/codet5_lut.pt",
        layer: int = -1,
        chunk: int = 8192,
        freeze: bool = True,
        **ignored,
    ):
        # max_length / gradient_checkpointing 等全局配置项在这里无意义（逐 token 查表无长度上限、不跑 attention），
        # 统一接收并忽略，方便在 cfg 里直接切换 encoder.name。
        super().__init__()
        self.path = path
        self.layer = layer
        self.chunk = chunk
        self.freeze = bool(freeze)

        cache = Path(lut)
        if cache.exists():
            blob = torch.load(cache, map_location="cpu", weights_only=True)
            self.hidden_size = int(blob["hidden_size"])
            self.vocab_size = int(blob["vocab_size"])
            table = blob["lut"].float()
            print(f"[codet5tok] 加载 LUT 缓存：{cache}（{tuple(table.shape)}，来自 {blob.get('path', path)}）")
        else:
            table = self._build_lut(dtype)
            cache.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"lut": table, "hidden_size": self.hidden_size, "vocab_size": self.vocab_size, "path": path},
                cache,
            )
            print(f"[codet5tok] LUT 已缓存：{cache}（{tuple(table.shape)}）")

        # 固定特征或可训练嵌入表（persistent=False 时不进 state_dict）
        if self.freeze:
            self.register_buffer("lut", table, persistent=False)
        else:
            self.lut = nn.Parameter(table.clone())
            print(f"[codet5tok] 可训练模式：LUT {tuple(table.shape)} 将参与微调")

    # ------------------------------------------------------------------ #
    def _build_lut(self, dtype: str) -> torch.Tensor:
        """逐块前向冻结编码器，得到 (vocab, D) 查表。"""
        model = CodeT5Encoder(path=self.path, dtype=dtype, freeze=True, max_length=1)
        model.eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)

        self.hidden_size = int(model.hidden_size)
        self.vocab_size = int(model.vocab_size)
        table = torch.zeros(self.vocab_size, self.hidden_size, dtype=torch.float32)
        print(f"[codet5tok] 构建 LUT：vocab={self.vocab_size} dim={self.hidden_size} layer={self.layer} device={device}")
        with torch.no_grad():
            for start in range(0, self.vocab_size, self.chunk):
                ids = torch.arange(start, min(start + self.chunk, self.vocab_size), device=device)[:, None]
                states = model(input_ids=ids, attention_mask=torch.ones_like(ids), output_hidden_states=True)
                h = states[-1] if self.layer == -1 else states[self.layer]
                table[start:start + ids.shape[0]] = h[:, 0].float().cpu()
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
        return table

    # ------------------------------------------------------------------ #
    @property
    def num_layers(self) -> int:
        return 0  # 无上下文堆叠，逐 token 查表

    def forward(
        self,
        input_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        """(B, L) token id -> (B, L, D) 逐 token 特征（padding 位置置零）。"""
        out = F.embedding(input_ids, self.lut)
        if attention_mask is not None:
            out = out * attention_mask.unsqueeze(-1).to(out.dtype)
        return out
