"""CodeT5 编码器（单文件实现）。

三种模式（对应 u-det 的消融开关 ``encoder.name``）：

    codet5     上下文编码：整个序列进编码器，返回 (B, L, D)（受 512 位置/attention 限制）。
    codet5tok  逐 token 编码：每个 token 独立过冻结编码器（长度=1），可预算成查表；
               训练零编码开销、长度无上限（U-Det 默认）。
    codet5lora 逐 token 编码 + peft LoRA：同样长度=1（无上下文），但每步现算，
               梯度真正回流进 CodeT5 的 LoRA 适配器。

三者都只使用 CodeT5 的 encoder 部分：
    * 默认从 checkpoints/codet5-base 加载（由 scripts/prepare.py 下载）；
    * 返回 token 级表示 (B, L, D)，可直接嗂给 models/hier.py。
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.checkpoint import checkpoint
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
        # 上下文编码受 CodeT5 的 512 位置限制：这里按 max_length 硬截断。
        # （v0.2 主干用的 codet5tok 查表不受此限，长度无上限、不截断。）
        if input_ids is not None and self.max_length and input_ids.shape[1] > self.max_length:
            input_ids = input_ids[:, : self.max_length]
            if attention_mask is not None:
                attention_mask = attention_mask[:, : self.max_length]
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


class CodeT5LoRAEncoder(nn.Module):
    """逐 token（长度=1）过 **LoRA 版 CodeT5**：保持“无上下文”，但梯度真正回流进 CodeT5。

    与 ``codet5tok`` 的关系：两者都把每个 token 单独送进编码器（无跨 token attention），
    差别只在“表”从哪来——

        codet5tok  预计算并缓存 (vocab, D) 查表；训练零编码开销、长度无上限。
        codet5lora 每个 step 现算 (L, 1) 前向；peft LoRA 的 A/B 参与训练，
                   CodeT5 本体（含 LayerNorm / 词嵌入 / 相对位置偏置）全部冻结。

    peft 的 ``lora_A`` 零初始化 ⇒ **训练起点与 ``codet5tok``（freeze=true）逐位相同**，
    所以这是“在冻结特征上再加一点可学修正”的严格推广。

    长度=1 时 attention 无跨 token 交互，逐 token 结果与分块顺序无关，
    因此用 ``chunk`` + 梯度检查点把显存压到与总长度无关的 O(chunk·D)。

    Args:
        path / dtype: 与 ``CodeT5Encoder`` 一致。
        layer: 取第几层隐状态（-1 最后一层）。
        r / alpha / dropout: 传给 peft ``LoraConfig``。
        targets: 注入 LoRA 的线性层名（T5: ``q/k/v/o`` 与 ``wi/wo``）。
        chunk: 每块处理的 token 数（显存/速度折中）。
        ckpt: 是否对每块做梯度检查点（L 很大时必须开）。
    """

    def __init__(
        self,
        path: str = "checkpoints/codet5-base",
        dtype: str = "float32",
        layer: int = -1,
        r: int = 16,
        alpha: int = 32,
        dropout: float = 0.0,
        targets=("q", "v"),
        chunk: int = 4096,
        ckpt: bool = True,
        compile: bool = False,
        pad_id: int = 0,
        **ignored,
    ):
        # max_length / lut / freeze 等在“逐 token”下无意义，统一接收并忽略，方便切 name。
        super().__init__()
        torch_dtype = _DTYPES.get(dtype, dtype) if isinstance(dtype, str) else dtype
        load_kwargs = {"local_files_only": os.path.isdir(path)}
        try:
            base = T5EncoderModel.from_pretrained(
                path, **{_dtype_kwarg(): torch_dtype}, **load_kwargs
            )
        except (TypeError, ValueError):
            base = T5EncoderModel.from_pretrained(path, **load_kwargs).to(torch_dtype)

        base.requires_grad_(False)                       # 底座先全冻，再由 peft 开适配器
        lora_cfg = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=int(r),
            lora_alpha=int(alpha),
            lora_dropout=float(dropout),
            bias="none",
            target_modules=[str(t) for t in targets],
        )
        self.model = get_peft_model(base, lora_cfg)
        self.n_lora = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        if self.n_lora == 0:
            raise ValueError(f"LoRA 未命中任何层（targets={targets}）")

        self._d_model = int(base.config.d_model)
        self._n_layers = int(base.config.num_layers)
        self.layer = int(layer)
        self.chunk = max(1, int(chunk))
        self.ckpt = bool(ckpt)
        self.pad_id = int(pad_id)
        self.model.eval()
        if compile:
            # 逐 token 前向的固定开销很大（短序列时占 2/3 以上），编译后 L=293 快 ~2.9x。
            # 编译要求形状固定，配合下面的"补齐到 chunk"拿到单一静态图。
            self._tokens = torch.compile(self._tokens, dynamic=False)
        print(f"[codet5lora] 逐 token + peft LoRA(r={r}, alpha={alpha}, targets={list(targets)}, "
              f"dropout={dropout})：可训练 {self.n_lora / 1e6:.3f}M，底座 {self._n_layers} 层冻结；"
              f"chunk={self.chunk} ckpt={self.ckpt} compile={compile}")

    # ------------------------------------------------------------------ #
    @property
    def hidden_size(self) -> int:
        return self._d_model

    @property
    def num_layers(self) -> int:
        return self._n_layers

    def train(self, mode: bool = True):
        """底座恒为 eval（关掉 CodeT5 内部 dropout），逐 token 特征保持确定。"""
        super().train(mode)
        self.model.eval()
        return self

    def _tokens(self, ids: torch.Tensor) -> torch.Tensor:
        """(c, 1) -> (c, 1, D)：逐 token 独立前向。"""
        out = self.model(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            output_hidden_states=self.layer != -1,
            return_dict=True,
        )
        return out.last_hidden_state if self.layer == -1 else out.hidden_states[self.layer]

    def forward(
        self,
        input_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        batch, length = input_ids.shape
        flat = input_ids.reshape(-1)
        if flat.numel() == 0:                                # 空序列（理论边界，保证不报错）
            return input_ids.new_zeros(batch, 0, self._d_model).to(torch.float32)
        pieces = []
        for start in range(0, flat.numel(), self.chunk):
            piece = flat[start:start + self.chunk]
            keep = piece.numel()
            if keep < self.chunk:                        # 补齐到固定长度：形状静态（便于编译）
                piece = torch.cat([piece, piece.new_full((self.chunk - keep,), self.pad_id)])
            ids = piece[:, None]                         # (chunk, 1)
            if self.training and self.ckpt:
                hidden = checkpoint(self._tokens, ids, use_reentrant=False)
            else:
                hidden = self._tokens(ids)
            pieces.append(hidden[:keep].reshape(-1, hidden.shape[-1]))
        out = torch.cat(pieces, 0).view(batch, length, -1)
        if attention_mask is not None:
            out = out * attention_mask.unsqueeze(-1).to(out.dtype)
        return out


def _build_lora_t5(path: str, dtype, r: int, alpha: int, dropout: float, targets):
    """加载**全冻结**的 CodeT5 encoder 并注入 peft LoRA。

    ``codet5lora`` 与 ``codet5blk`` 共用同一套底座构造（差别只在 forward 怎么切序列）。
    返回 ``(peft_model, base_config)``；两者的 ``state_dict`` 键结构完全一致，
    因此可以互相 ``load_state_dict``（自检里用来做逐位回归）。
    """
    torch_dtype = _DTYPES.get(dtype, dtype) if isinstance(dtype, str) else dtype
    load_kwargs = {"local_files_only": os.path.isdir(path)}
    try:
        base = T5EncoderModel.from_pretrained(path, **{_dtype_kwarg(): torch_dtype}, **load_kwargs)
    except (TypeError, ValueError):
        base = T5EncoderModel.from_pretrained(path, **load_kwargs).to(torch_dtype)

    base.requires_grad_(False)                       # 底座先全冻，再由 peft 开适配器
    lora_cfg = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=int(r),
        lora_alpha=int(alpha),
        lora_dropout=float(dropout),
        bias="none",
        target_modules=[str(t) for t in targets],
    )
    return get_peft_model(base, lora_cfg), base.config


class CodeT5BlockEncoder(nn.Module):
    """**分块**过 LoRA 版 CodeT5：块内 token 之间有真正的 attention，块间彼此独立。

    与 ``codet5lora`` 的**唯一**区别是"一次喂几个 token"：

        codet5lora  每个 token 单独前向（序列长度 = **1**）⇒ softmax 只有一个元素恒为 1，
                    **attention 完全退化，q / k 是死参数**（所以它的 LoRA 目标里没有 q/k）。
        codet5blk   每 ``block`` 个 token 作为一个**块**一起前向（序列长度 = **K**）⇒ 块内
                    attention 真实存在，每个 token 能看到同块内其余 K-1 个 token 的上下文。

    ★ **块的输入 / 输出向量数严格 1:1**（K 进 K 出，不池化、不重叠、不跨样本边界），
    所以对下游完全透明：返回的仍是 ``(B, L, D)``，``models/hier2.py`` 一行都不用改。
    长度也仍然无上限 —— 块与块相互独立，既不受 ``n_positions=512`` 限制，
    也没有 O(L²) 的整段 attention。

    长度处理：每条的序列按 K 切块，**只有末块**用 ``pad_id`` 补齐并在 attention 里屏蔽
    （末块之前的所有块都是满的）；输出里所有 padding 位置最后被 ``attention_mask`` 置零。

    Args:
        block: 块大小 K（默认 128）。
        block_batch: 一次前向并多少个块（默认 16 ⇒ 16×128 = 2048 个 token）。
        path / dtype / layer: 与 ``CodeT5Encoder`` 一致。
        r / alpha / dropout / targets: 传给 peft ``LoraConfig``（attention 已"复活"，q/k 可以放进来了）。
        ckpt: 是否对每个前向分片做梯度检查点。
        pad_id: 补齐用的 token id（0 = ``<pad>``）。
        static: 是否把**块数**补齐到 ``block_batch`` 的整数倍。
            开启后每次前向的形状恒为 ``(block_batch, K)``，形状静态 ⇒ 可以安全地
            ``compile``（不会因块数变化而反复重编译）。代价是短样本会多算几个全 pad 的块
            （被 mask 屏蔽并最后裁掉，不影响任何有效输出）。
        compile: 是否对分片前向做 ``torch.compile``。
            ★ 强烈建议与 ``static=True`` 一起用：动态形状下 compile 会频繁重编译。
            本项目实测：逐 token 前向不开 compile 时，固定开销能占去 2/3 以上。
    """

    def __init__(
        self,
        path: str = "checkpoints/codet5-base",
        dtype: str = "float32",
        layer: int = -1,
        block: int = 128,
        block_batch: int = 16,
        r: int = 16,
        alpha: int = 32,
        dropout: float = 0.0,
        targets=("q", "k", "v", "o"),
        ckpt: bool = True,
        pad_id: int = 0,
        static: bool = False,
        compile: bool = False,
        **ignored,
    ):
        # lut / freeze / max_length / chunk 等在分块语义下无意义，统一接收并忽略，
        # 方便在 cfg 里直接改一行 encoder.name 就切过来。
        super().__init__()
        self.model, base_cfg = _build_lora_t5(path, dtype, r, alpha, dropout, targets)
        self.n_lora = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        if self.n_lora == 0:
            raise ValueError(f"LoRA 未命中任何层（targets={targets}）")

        self._d_model = int(base_cfg.d_model)
        self._n_layers = int(base_cfg.num_layers)
        self.layer = int(layer)
        self.block = max(1, int(block))
        self.block_batch = max(1, int(block_batch))
        self.ckpt = bool(ckpt)
        self.pad_id = int(pad_id)
        self.static = bool(static)
        self.model.eval()                            # 底座恒 eval，关掉 CodeT5 内部 dropout
        if compile:
            # static=True 时形状恒定 ⇒ dynamic=False 拿到单一静态图（最快）；
            # 否则块数随长度变化，只能退而用 dynamic=True，会有重编译开销。
            self._blocks = torch.compile(self._blocks, dynamic=not self.static)
        print(f"[codet5blk] 分块过 LoRA CodeT5：block={self.block} block_batch={self.block_batch}"
              f"（每前向最多 {self.block * self.block_batch} token）r={r} alpha={alpha} "
              f"targets={list(targets)}；可训练 {self.n_lora / 1e6:.3f}M，底座 {self._n_layers} 层冻结；"
              f"ckpt={self.ckpt} static={self.static} compile={compile}")

    # ------------------------------------------------------------------ #
    @property
    def hidden_size(self) -> int:
        return self._d_model

    @property
    def num_layers(self) -> int:
        return self._n_layers

    def train(self, mode: bool = True):
        """底座恒为 eval（关掉 CodeT5 内部 dropout），块内特征保持确定。"""
        super().train(mode)
        self.model.eval()
        return self

    def _blocks(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """(nb, K) -> (nb, K, D)：块内带 attention 的前向。"""
        out = self.model(
            input_ids=ids,
            attention_mask=mask,
            output_hidden_states=self.layer != -1,
            return_dict=True,
        )
        return out.last_hidden_state if self.layer == -1 else out.hidden_states[self.layer]

    def forward(
        self,
        input_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        """(B, L) token id -> (B, L, D) 分块上下文特征（padding 位置置零）。"""
        batch, length = input_ids.shape
        if length == 0:                                     # 空序列（理论边界，保证不报错）
            return input_ids.new_zeros(batch, 0, self._d_model).to(torch.float32)
        if attention_mask is None:
            attention_mask = input_ids.new_ones(batch, length)
        keep = attention_mask                               # 原始 mask（B, L），最后用来清零

        k = self.block
        nblk = (length + k - 1) // k
        if self.static:
            # 把块数抬到 block_batch 的整数倍 ⇒ 每次前向形状恒为 (block_batch, K)。
            # 多出来的块全是 pad（mask 全 0），被下面的兜底逻辑保住不产生 NaN，
            # 最后又在 [:, :length] 被裁掉，因此不影响任何有效输出。
            nblk = -(-nblk // self.block_batch) * self.block_batch
        pad_len = nblk * k - length
        ids, mask = input_ids, attention_mask
        if pad_len:
            ids = F.pad(ids, (0, pad_len), value=self.pad_id)
            mask = F.pad(mask, (0, pad_len), value=0)

        ids = ids.reshape(batch * nblk, k)                  # (B·nblk, K)：块与块不跨样本
        mask = mask.reshape(batch * nblk, k)
        # 整块全 pad 在正常路径下不会出现（末块之前都是满块），但全 0 的 attention mask 会让
        # softmax 全 -inf 得到 NaN，所以兜一手：给这种块留一个"可看"的位置。
        # 这些位置随后仍会被 keep 置零，因此不改变任何有效输出。
        dead = mask.sum(-1) == 0
        if bool(dead.any()):
            mask = mask.clone()
            mask[dead, 0] = 1
        pieces = []
        for start in range(0, ids.shape[0], self.block_batch):
            ids_b = ids[start:start + self.block_batch]
            mask_b = mask[start:start + self.block_batch]
            if self.training and self.ckpt:
                hidden = checkpoint(self._blocks, ids_b, mask_b, use_reentrant=False)
            else:
                hidden = self._blocks(ids_b, mask_b)
            pieces.append(hidden)
        out = pieces[0] if len(pieces) == 1 else torch.cat(pieces, 0)
        out = out.reshape(batch, nblk * k, -1)[:, :length]
        return out * keep.unsqueeze(-1).to(out.dtype)


# --------------------------------------------------------------------------- #
# 预训练层 → 瓶颈：把 CodeT5 的若干层**按位置拷进 SelfBlock**（v0.4.9）
# --------------------------------------------------------------------------- #
def _parse_block_spec(spec: str) -> tuple[int, int]:
    """``"codet5@8-11"`` -> ``(8, 11)``（含两端，CodeT5 encoder 的 0-based block 索引）。"""
    text = str(spec).strip()
    if "@" not in text:
        raise ValueError(f"mid_init 格式应为 'codet5@8-11'，收到 {spec!r}")
    head, rng = text.split("@", 1)
    if head.strip() not in ("codet5", "t5"):
        raise ValueError(f"mid_init 只支持 codet5@a-b，收到 {spec!r}")
    lo, _, hi = rng.partition("-")
    lo, hi = int(lo), int(hi or lo)
    if lo > hi:
        raise ValueError(f"mid_init 区间写反了：{spec!r}")
    return lo, hi


def init_selfblocks_from_codet5(selfblocks, path: str = "checkpoints/codet5-base",
                                spec: str = "codet5@8-11", verbose: bool = True) -> list:
    """把 CodeT5 第 ``[lo, hi]`` 层的权重**按位置拷贝**进瓶颈的 ``SelfBlock``。

    ★ **不改结构、不改参数量** —— 只换初始化来源。这是 §8.16.9 选定的做法：
      「叠加」会 +28.3M 参数且正好加在 §8.15.1 证明有害的文档级路径上；
      「减到 2/3 层」会同时动深度，而 §8.16.6 的 P6 已说明深度不是瓶颈。

    ⚠️ **这是近似移植**，两处语义对不上（函数会打印一个保真度读数）：
      * T5 用 ``T5LayerNorm``（RMS、**不减均值**、无偏置），``SelfBlock`` 用 ``nn.LayerNorm``
        （减均值 + 偏置）⇒ **只拷 weight、bias 置零**；
      * CodeT5-base 的 FFN 激活是 **ReLU**（``feed_forward_proj: "relu"``），
        ``SelfBlock`` 用 **GELU** ⇒ 保持 GELU（改它就是改结构）。
    **被忠实拷贝的是全部线性映射**（``q/k/v/o`` 与 ``wi/wo``）—— 它们承载了绝大部分预训练知识。
    另外 ``SelfBlock.attn`` 带 bias 而 T5 的 ``q/k/v/o`` 不带 ⇒ 所有 bias 置零。

    ⚠️ 预期收益是**加快收敛，不是抬高上限**（本项目 v0.3.1 vs v0.3.2 的直接结论：
    LoRA 微调在 epoch 0 大幅领先，到周期末端收敛到同一水平）。
    ⇒ 判据必须看**周期末端**（lessons A1/A2），不能拿前两轮说事。

    Returns:
        拷贝过的 ``T5Block`` 列表（供保真度诊断使用）。
    """
    from transformers import T5EncoderModel

    lo, hi = _parse_block_spec(spec)
    blocks = list(selfblocks)
    want = hi - lo + 1
    if len(blocks) != want:
        raise ValueError(f"mid_init={spec!r} 要拷 {want} 层，但瓶颈有 {len(blocks)} 层 "
                         f"（本项目的做法是「替换」而不是「叠加」，两者必须相等）")
    load_kwargs = {"local_files_only": os.path.isdir(path)}
    model = T5EncoderModel.from_pretrained(path, **load_kwargs)
    model.eval()

    def _zero_bias(mod) -> None:
        """``SelfBlock`` 默认的 ``nn.LayerNorm`` 带 bias；``RMSNorm``（T5 语义）不带。"""
        bias = getattr(mod, "bias", None)
        if bias is not None:
            bias.zero_()

    torch.set_grad_enabled(False)
    try:
        srcs = []
        for i, b in enumerate(range(lo, hi + 1)):
            blk = model.encoder.block[b]
            attn = blk.layer[0].SelfAttention
            ffn = blk.layer[1].DenseReluDense
            dst = blocks[i]
            dst.norm1.weight.copy_(blk.layer[0].layer_norm.weight)
            _zero_bias(dst.norm1)
            dst.attn.in_proj_weight.copy_(
                torch.cat([attn.q.weight, attn.k.weight, attn.v.weight], dim=0))
            dst.attn.in_proj_bias.zero_()
            dst.attn.out_proj.weight.copy_(attn.o.weight)
            dst.attn.out_proj.bias.zero_()
            dst.norm2.weight.copy_(blk.layer[1].layer_norm.weight)
            _zero_bias(dst.norm2)
            dst.mlp[0].weight.copy_(ffn.wi.weight)
            dst.mlp[0].bias.zero_()
            dst.mlp[3].weight.copy_(ffn.wo.weight)
            dst.mlp[3].bias.zero_()
            srcs.append(blk)
    finally:
        torch.set_grad_enabled(True)

    n = sum(p.numel() for p in blocks[0].parameters()) if blocks else 0
    if verbose:
        print(f"[mid_init] 已把 CodeT5 blocks {lo}..{hi} 的权重**按位置**拷进瓶颈的 "
              f"{len(blocks)} 个 SelfBlock（结构不变、参数不变）")
        print(f"[mid_init] ⚠️ 近似移植：T5LayerNorm(RMS)→nn.LayerNorm(减均值) 与 ReLU→GELU 两处语义不同；"
              f"线性映射(q/k/v/o, wi/wo)是忠实拷贝；所有 bias 置零")
        print(f"[mid_init] ⚠️ 预期是**加快收敛**而不是抬高上限 ⇒ 判据看周期末端（A1/A2），不看前两轮")
    return srcs


@torch.no_grad()
def port_fidelity(selfblock, t5block, x: torch.Tensor) -> dict:
    """保真度诊断：同一输入下，移植后的 ``SelfBlock`` 与原始 ``T5Block`` 输出的接近程度。

    ``x`` 应当用**真实的瓶颈输入**（stride-64 的均值池化特征）而不是随机噪声 ——
    归一化与激活的差异是与输入分布相关的，用真数据量才有意义。

    Returns:
        ``{"cos": 余弦相似度, "rel": 相对 L2 误差}``（都是逐 token 平均）。
    """
    selfblock.eval()
    t5block.eval()
    # transformers ≥ 4.5x 的 T5Block 需要 cache_position（否则内部取 [-1] 报 TypeError）
    pos = torch.arange(x.shape[1], device=x.device)
    y_ref = t5block(x, cache_position=pos)[0].float()
    y_new = selfblock(x).float()
    cos = torch.nn.functional.cosine_similarity(y_ref.reshape(-1, y_ref.shape[-1]),
                                                y_new.reshape(-1, y_new.shape[-1]), dim=-1)
    rel = (y_ref - y_new).norm(dim=-1) / y_ref.norm(dim=-1).clamp(min=1e-9)
    return {"cos": float(cos.mean()), "rel": float(rel.mean()), "cos_min": float(cos.min())}

