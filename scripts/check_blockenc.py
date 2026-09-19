"""codet5blk（分块编码器）自检 —— **全在 CPU 上跑**，不碰 GPU，避免与正在进行的训练抢卡。

分块编码器的定义性质一共就两条，其余都是围绕它们的实现契约：

    * **块内**：序列长度 = K，attention 真实存在（这是相对 codet5lora 唯一想要的变化）；
    * **块间**：彼此完全独立，且输入/输出向量数严格 1:1（K 进 K 出、不池化、不重叠、不跨样本）。

覆盖 8 条：
  ① 形状契约：输出必须严格是 (B, L, D)，L 与输入逐位一致（下游 codec / heads 依赖）
  ② 1:1 位置数：L 不是 K 的整数倍时也不许多出 / 少掉任何位置
  ③ 覆盖：真实位置的输出不得整块为零（某个块被漏算会立刻暴露）
  ④ 块隔离：改动第 j 块内的 token，其它块的输出必须**逐位不变**（分块的定义性质）
  ⑤ 不跨样本：batch>1 时两条样本互不干扰（块只在样本内切）
  ⑥ 屏蔽：传入含 0 的 attention_mask 时，padding 位置输出必须严格为 0
  ⑦ 退化回归：block=1 必须与 codet5lora 在**真实位置上逐位一致**（同为长度 1 前向）
  ⑧ bf16 autocast：形状不变、无 dtype 冲突

用法::

    OMP_NUM_THREADS=8 python scripts/check_blockenc.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from encoders.codet5 import CodeT5BlockEncoder, CodeT5LoRAEncoder  # noqa: E402

PATH = "checkpoints/codet5-base"
TARGETS = ["v", "o", "wi", "wo"]        # 与 v0.4.1 一致（单变量对照）
K = 128


def banner(text: str) -> None:
    print(f"\n【{text}】")


def new_ids(length: int, batch: int = 1, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(1, 30000, (batch, length), generator=g)


def main() -> None:  # noqa: PLR0915
    print("【0】构建编码器（CPU，加载 CodeT5-base + peft LoRA）")
    enc = CodeT5BlockEncoder(path=PATH, block=K, block_batch=16, ckpt=False,
                             targets=TARGETS).eval()
    assert enc.hidden_size == 768, enc.hidden_size
    print(f"      hidden_size={enc.hidden_size} num_layers={enc.num_layers} n_lora={enc.n_lora / 1e6:.3f}M")

    # ---------------------------------------------------------------- #
    banner("① ② 形状契约与 1:1 位置数（L 取 K 的整倍数与非整倍数）")
    for length in (1, 100, 128, 129, 300, 383, 384, 385, 1000):
        out = enc(new_ids(length))
        assert out.shape == (1, length, 768), f"L={length}: {tuple(out.shape)}"
        print(f"      L={length:>4} -> {tuple(out.shape)} ✓（块数={-(-length // K)}）")

    banner("③ 覆盖：真实位置不得整块为零")
    for length in (129, 300, 1000):
        out = enc(new_ids(length))
        per_pos = out.abs().sum(-1)                       # (1, L)
        n_zero = int((per_pos == 0).sum())
        assert n_zero == 0, f"L={length}: 有 {n_zero} 个真实位置输出全零（块被漏算？）"
    print("      129 / 300 / 1000 三个长度上均无全零位置 ✓")

    banner("④ 块隔离：改第 0 块的 token，其余块必须逐位不变")
    length = 300
    ids = new_ids(length, seed=1)
    base = enc(ids)
    moved = ids.clone()
    moved[0, 0] = 29999                                   # 第 0 块内的一个位置
    after = enc(moved)
    d_block0 = (base[:, :K] - after[:, :K]).abs().max().item()
    d_rest = (base[:, K:] - after[:, K:]).abs().max().item()
    assert d_block0 > 0, "改了第 0 块的 token，第 0 块输出却没变（说明该块没被算）"
    assert d_rest <= 1e-6, f"块间泄漏：块 1..n 的最大变化 {d_rest:.3e} 应为 0"
    print(f"      块 0 最大变化={d_block0:.6f}（应>0），块 1..n 最大变化={d_rest:.3e}（应为 0）✓")

    banner("⑤ 不跨样本：batch=2 的两条样本互不干扰")
    ids2 = new_ids(300, batch=2, seed=2)
    both = enc(ids2)
    solo0 = enc(ids2[:1])
    solo1 = enc(ids2[1:])
    d0 = (both[:1] - solo0).abs().max().item()
    d1 = (both[1:] - solo1).abs().max().item()
    assert d0 <= 1e-6 and d1 <= 1e-6, f"样本间干扰：{d0:.3e} / {d1:.3e}"
    print(f"      单独前向 vs 同批前向的最大差 {max(d0, d1):.3e}（应为 0）✓")

    banner("⑥ 屏蔽：attention_mask 为 0 的位置输出必须严格为 0")
    ids3 = new_ids(300)
    mask = torch.ones(1, 300, dtype=torch.long)
    mask[0, 200:] = 0                                     # 后 100 个位置视为 padding
    out3 = enc(ids3, mask)
    assert out3.shape == (1, 300, 768)
    assert out3[0, 200:].abs().max().item() == 0, "padding 位置未被清零"
    assert out3[0, :200].abs().max().item() > 0, "有效位置被整体清零了"
    print("      后 100 个位置严格为 0，前 200 个不全为 0 ✓")

    banner("⑦ 退化回归：block=1 必须与 codet5lora 逐位一致")
    lora = CodeT5LoRAEncoder(path=PATH, chunk=8, ckpt=False, compile=False,
                             targets=TARGETS).eval()
    blk1 = CodeT5BlockEncoder(path=PATH, block=1, block_batch=8, ckpt=False,
                              targets=TARGETS).eval()
    missing, unexpected = blk1.load_state_dict(lora.state_dict(), strict=False)
    assert not unexpected, f"多余的键：{list(unexpected)[:5]}"
    for length in (1, 5, 100):
        ids4 = new_ids(length, seed=3)
        a = lora(ids4)
        b = blk1(ids4)
        d = (a - b).abs().max().item()
        assert d <= 1e-6, f"L={length}: block=1 与 codet5lora 差了 {d:.3e}"
        print(f"      L={length:>4}：max|diff|={d:.3e} ✓")
    del lora, blk1

    banner("⑧ bf16 autocast：形状 / 屏蔽 / 无 NaN，且无 dtype 冲突")
    with torch.autocast("cpu", dtype=torch.bfloat16):
        for length in (1, 300):
            out5 = enc(new_ids(length))
            assert out5.shape == (1, length, 768), f"autocast 下形状不符：{tuple(out5.shape)}"
            assert out5.dtype.is_floating_point, out5.dtype
            assert torch.isfinite(out5).all(), "autocast 下出现 NaN / Inf"
        ids6 = new_ids(300)
        mask6 = torch.ones(1, 300, dtype=torch.long)
        mask6[0, 200:] = 0
        out6 = enc(ids6, mask6)
        assert out6[0, 200:].abs().max().item() == 0, "autocast 下 padding 位置未被清零"
        assert out6[0, :200].abs().max().item() > 0, "autocast 下有效位置被整体清零"
    print(f"      形状正确、无 NaN、屏蔽仍生效；输出 dtype={out5.dtype}")
    print("      （T5 末层 LayerNorm 在 fp32 里算，所以 autocast 下输出是 fp32 —— 与 codet5lora 一致，不是缺陷）✓")

    banner("⑨ static（静态形状）与默认路径在有效位置上必须逐位一致")
    encs = CodeT5BlockEncoder(path=PATH, block=K, block_batch=16, ckpt=False,
                              static=True, targets=TARGETS).eval()
    _miss, unexp = encs.load_state_dict(enc.state_dict(), strict=False)
    assert not unexp, f"多余的键：{list(unexp)[:5]}"
    for length in (1, 100, 129, 300, 1000):
        ids7 = new_ids(length, seed=4)
        d = (enc(ids7) - encs(ids7)).abs().max().item()
        assert d <= 1e-6, f"L={length}: static 与默认路径差了 {d:.3e}"
        print(f"      L={length:>4}：max|diff|={d:.3e} ✓")
    mask7 = torch.ones(1, 300, dtype=torch.long)
    mask7[0, 200:] = 0
    out7 = encs(new_ids(300), mask7)
    assert out7[0, 200:].abs().max().item() == 0, "static 下 padding 未被清零"
    print("      static 下 padding 位置仍严格为 0 ✓")

    print("\n全部通过 ✓")


if __name__ == "__main__":
    main()
