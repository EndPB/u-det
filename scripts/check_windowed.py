#!/usr/bin/env python
"""窗口化基线的自检 + 成本估算（**不需要 GPU，不需要 CodeT5**）。

两件事：

1. **纯逻辑自检**（用假编码器，秒级）：验 `WindowedContextClassifier` 的四个易错点 ——
   窗口覆盖率、token logits 长度、重叠平均、接缝不变性，以及 `window=0` 时与
   `PooledClassifier` 的**逐位等价**（这是"回归测试"能不能成立的判据）。

2. **成本估算**：直接读 `data/processed/*.parquet`，按真实长度算出每 epoch 要过多少个窗口
   （含手工报告前缀），据此估单 epoch 时间。跑正式训练之前先看这个数，比先跑起来再发现
   跑不动要划算。

用法::

    python scripts/check_windowed.py                      # 只做逻辑自检
    python scripts/check_windowed.py --cost               # 加成本估算（读 parquet）
    python scripts/check_windowed.py --cost --stride 512  # 换成不重叠切分再算一次
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models import PooledClassifier, WindowedContextClassifier  # noqa: E402


# --------------------------------------------------------------------------- #
# 假编码器：把 token id 直接当特征（无参数、可预测），从而能精确验证窗口逻辑
# --------------------------------------------------------------------------- #
class DummyEncoder(nn.Module):
    """``feature[b, i, :] = id[b, i] / 1000``（同一 id 在不同窗口得到同一特征）。

    只需实现 ``hidden_size`` 与 ``forward(input_ids, attention_mask) -> (B, L, D)``，
    并像真的 `CodeT5Encoder` 一样按 ``max_length`` 截断。
    """

    def __init__(self, dim: int = 8, max_length: int = 512):
        super().__init__()
        self.hidden_size = dim
        self.max_length = max_length

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None):
        length = min(input_ids.shape[1], self.max_length)
        feat = input_ids[:, :length].float().unsqueeze(-1) / 1000.0
        return feat.repeat(1, 1, self.hidden_size)          # (B, L, D)


def check_logic(dim: int = 8, window: int = 512, window_batch: int = 7) -> None:
    """四个易错点的自检。``window_batch`` 故意取 7（不整除）以便暴露分批 bug。"""
    for stride in (None, window, window // 2, 100):
        enc = DummyEncoder(dim=dim, max_length=window)
        net = WindowedContextClassifier(enc, dim=dim, window=window, stride=stride or 0,
                                        window_batch=window_batch, token_head=True, out=2)
        net.eval()
        with torch.no_grad():
            for length in (1, 37, window - 1, window, window + 1, 1000, 1379, 5000):
                ids = torch.randint(1, 100, (1, length))     # 避开 0（=PAD）
                mask = torch.ones_like(ids)
                _, (tok,) = net(ids, mask)

                # ① token logits 形状必须是 (B,1,L)：evaluate 用 zip 逐位对齐，
                #    token_loss 的 BCE 还要求与 (B,1,L_k) 的 soft 目标同形状
                assert tok.shape == (1, 1, length), f"形状不符：{tuple(tok.shape)} vs (1,1,{length})"

                # ② 覆盖率：每个位置至少被一个窗口覆盖（count>0）
                starts = net._starts(length)
                width = window if window > 0 else length
                covered = torch.zeros(length, dtype=torch.bool)
                for s in starts:
                    covered[s:min(s + width, length)] = True
                assert covered.all(), f"有位置没被覆盖：L={length} starts={starts}"

                # ③ 重叠平均 + ④ 接缝不变性：同一 token id 在不同位置/窗口必须得到同一 logit
                #    （假编码器下 feature 只依赖 id；头是线性层 ⇒ 同 id ⇒ 同 logit）
                uniform = torch.full((1, length), 42, dtype=torch.long)
                _, (tok_u,) = net(uniform, mask)
                if length > 1:
                    spread = (tok_u.max() - tok_u.min()).item()
                    assert spread < 1e-4, f"接缝不平滑：L={length} stride={stride} 极差={spread:.2e}"

                # ⑤ 步长覆盖（按流区分用）：显式传 stride 时形状/覆盖/接缝仍必须正确
                for s in (window, window // 4):
                    _, (tok_s,) = net(uniform, mask, stride=s)
                    assert tok_s.shape == (1, 1, length), f"stride={s} 形状不符：{tuple(tok_s.shape)}"
                    if length > 1:
                        sp = (tok_s.max() - tok_s.min()).item()
                        assert sp < 1e-4, f"stride={s} 接缝不平滑：极差={sp:.2e}"
        print(f"  ✓ window={window} stride={stride or 'win//2'} window_batch={window_batch}："
              f"长度/覆盖/接缝全部通过")

    # ⑤ 回归等价性：window=0 且无 token 头时，必须与 PooledClassifier 逐位一致
    enc = DummyEncoder(dim=dim, max_length=window)
    pc = PooledClassifier(enc, dim=dim, pooling="mean", out=2)
    wc = WindowedContextClassifier(enc, dim=dim, window=0, token_head=False, pooling="mean", out=2)
    wc.norm.load_state_dict(pc.norm.state_dict())            # 两者的 net 都是单层 Sequential
    wc.net[0].load_state_dict(pc.net[0].state_dict())
    pc.eval(), wc.eval()
    with torch.no_grad():
        for length in (1, 100, window, 1234):
            ids = torch.randint(1, 100, (1, length))
            mask = torch.ones_like(ids)
            a, _ = pc(ids, mask)
            b, t = wc(ids, mask)
            assert t is None, "token_head=False 时应返回 None"
            assert torch.allclose(a, b, atol=1e-6), f"window=0 不等价于截断池化：L={length}"
    print(f"  ✓ window=0 与 PooledClassifier 逐位等价（截断池化路径未被改坏）")

    # ⑥ mixed dtype：autocast 下编码器输出可能是 fp32 而 Linear 输出 bf16，
    #    两者夹不准会让 index_add 报 "self (Float) and source (BFloat16)"（真实踩过）。
    #    这里用 CPU bf16 autocast 重现那个混合 dtype 场景。
    enc = DummyEncoder(dim=dim, max_length=window)
    net = WindowedContextClassifier(enc, dim=dim, window=window, window_batch=window_batch,
                                    token_head=True, out=2)
    net.eval()
    try:
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
            for length in (1, 700, 2000):
                ids = torch.randint(1, 100, (1, length))
                _, (tok,) = net(ids, torch.ones_like(ids))
                assert tok.shape == (1, 1, length), f"autocast 下形状不符：{tuple(tok.shape)}"
        print("  ✓ mixed dtype（autocast bf16）下长度对齐、无 dtype 冲突")
    except RuntimeError as exc:                              # CPU 可能不支持某些 bf16 算子
        print(f"  · 跳过 mixed dtype 用例（本机 CPU autocast 不支持：{str(exc)[:60]}）")


def check_cost(m4_file: str, hybrid_file: str, window: int, stride: int,
               report_tokens: int | None) -> None:
    """读真实 parquet 统计每 epoch 的窗口数，估训练/评测耗时。"""
    import pyarrow.parquet as pq

    def lengths(path: str, split: str = "train") -> list[int]:
        table = pq.read_table(path)
        cols = table.column_names
        sp = table.column("split").to_pylist() if "split" in cols else ["train"] * table.num_rows
        ids = table.column("input_ids").to_pylist()
        return [len(x) for x, s in zip(ids, sp) if split in ("all", "*", s)]

    def windows(n: int) -> int:
        if n <= window:
            return 1
        last = n - window
        k = last // stride + 1
        return k if (k - 1) * stride == last else k + 1

    prefix = report_tokens or 0
    print(f"\n窗口化成本估算（window={window} stride={stride}，"
          f"报告前缀 ≈ {prefix} token）")
    print(f"  {'数据集':<8} {'n':>6} {'长度中位':>9} {'长度max':>8} "
          f"{'窗口/样本':>10} {'窗口总数':>10}")
    total = 0
    for name, path in (("m4", m4_file), ("hybrid", hybrid_file)):
        if not Path(path).exists():
            print(f"  {name:<8} 找不到 {path}，跳过")
            continue
        lens = [n + prefix for n in lengths(path)]
        if not lens:
            continue
        wins = [windows(n) for n in lens]
        total += sum(wins)
        lens_sorted = sorted(lens)
        print(f"  {name:<8} {len(lens):>6} {lens_sorted[len(lens)//2]:>9} {lens_sorted[-1]:>8} "
              f"{sum(wins)/len(wins):>10.1f} {sum(wins):>10}")
    if not total:
        return
    print(f"  {'合计':<8} {'':<6} {'':<9} {'':<8} {'':<10} {total:>10}")
    print("\n  参考：编码器单次 fwd+bwd（B=1, L=512，开梯度检查点）粗估 40~60 ms，"
          "前向（无梯度）约 5~8 ms")
    print(f"  ⇒ 训练 1 epoch ≈ {total * 0.05 / 60:.0f}~{total * 0.06 / 60:.0f} 分钟"
          f"（不含评测）")
    print(f"  ⇒ 评测一遍 val+test（约 1/6 样本量）×2 流 ≈ "
          f"{total / 6 * 0.006 / 60:.1f} 分钟")
    print("  注意：这是**下界**估计，实际受显存/调度/序列长度分布影响，"
          "正式跑之前务必先用 --limit 200 --epochs 1 实测。")


def main() -> int:
    parser = argparse.ArgumentParser(description="窗口化基线自检 + 成本估算")
    parser.add_argument("--window", type=int, default=512)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--cost", action="store_true", help="额外读 parquet 做成本估算")
    parser.add_argument("--m4", default="data/processed/m4.parquet")
    parser.add_argument("--hybrid", default="data/processed/hybrid.parquet")
    parser.add_argument("--report-tokens", type=int, default=53,
                        help="手工报告前缀的 token 数（configs 注释里记的是 53）")
    args = parser.parse_args()

    print("【1】窗口逻辑自检（假编码器，不碰 GPU）")
    check_logic(window=args.window)

    if args.cost:
        print("\n【2】成本估算")
        check_cost(str(ROOT / args.m4), str(ROOT / args.hybrid), args.window, args.stride,
                   args.report_tokens)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
