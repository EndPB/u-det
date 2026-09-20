"""离线分析 ``runs/<tag>/raw_*.pt``：**长度分桶** + **阈值扫描**。

这些 ``.pt`` 由 ``train.py --eval --dump-raw`` 落盘（逐样本的长度/标签/样本概率
/逐行概率/逐 token 概率），所以本脚本**不需要 GPU、也不需要重跑模型**。

两个诊断各自的用途：

1. **长度分桶**：m4 的中位长度只有 293 token，而基线的窗口就是 512 ——
   也就是说在短文档区间，基线一次窗口就装下了整篇，**它在结构上并不吃亏**。
   只有超过 512 之后它才只能看到局部。分桶能回答
   “U-Det 的线性复杂度全长上下文到底在哪个长度区间开始兑现成精度”。
2. **阈值扫描**：仓库内的评测把阈值写死在 0.5。对**行级/片段级/token 级**这类
   不平衡的分割任务，0.5 往往不是最优。这是**不需要重新训练**就能拿到的潜在提升，
   而且扫描本身零成本。

用法::

    python scripts/analyze_raw.py runs/v0.4.4 runs/v0.4.2 runs/base_codet5_tok
    python scripts/analyze_raw.py runs/v0.4.4 --edges 0 512 2048 8192 1e9
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train import chunk_match, prf1  # noqa: E402  ← 复用仓库内的指标定义，避免离线版与在线版算出不同的数


def load(run_dir: Path, name: str, split: str = "test"):
    """读 ``raw_<name>_<split>.pt``；不存在返回 None。"""
    p = run_dir / f"raw_{name}_{split}.pt"
    if not p.exists():
        return None
    return torch.load(p, weights_only=False)


# --------------------------------------------------------------------------- #
def m4_metrics(blob, thr: float) -> dict:
    """样本级：prob > thr 判为 AI。thr=0.5 等价于 argmax（二分类）。"""
    pred = (blob["sample_prob"] > thr).long()
    gold = blob["label"].long()
    d = prf1(pred.tolist(), gold.tolist())
    return {"acc": float((pred == gold).float().mean()), "p": d["p"], "r": d["r"], "f1": d["f1"]}


def hybrid_metrics(blob, thr: float) -> dict:
    """行级 / 片段级 / token 级。行与片段直接复用 train.py 的 prf1 / chunk_match。"""
    pred_lines, gold_lines = [], []
    chunk_tp = chunk_pred = chunk_gold = 0
    token_tp = token_fp = token_fn = 0
    for i in range(blob["n"]):
        gold = blob["line_label"][i].long()
        prob = blob["line_prob"][i].float()
        picked = (prob > thr).long()
        pred_lines.extend(picked.tolist())
        gold_lines.extend(gold.tolist())
        tp, np_, ng = chunk_match(picked.tolist(), gold.tolist())
        chunk_tp += tp; chunk_pred += np_; chunk_gold += ng

        t_prob = blob["token_prob"][i].float()
        t_gold = blob["token_label"][i].long()
        keep = t_gold >= 0                                     # 忽略 IGNORE(-100)
        p_ok = keep & (t_prob > thr)
        g_ok = keep & (t_gold == 1)
        token_tp += int((p_ok & g_ok).sum())
        token_fp += int((p_ok & ~g_ok).sum())
        token_fn += int((~p_ok & g_ok).sum())

    line = prf1(pred_lines, gold_lines)
    out = {"line_p": line["p"], "line_r": line["r"], "line_f1": line["f1"]}
    cp = chunk_tp / max(chunk_pred, 1e-9)
    cr = chunk_tp / max(chunk_gold, 1e-9)
    out.update({"chunk_p": cp, "chunk_r": cr, "chunk_f1": 2 * cp * cr / max(cp + cr, 1e-9)})
    tp = token_tp / max(token_tp + token_fp, 1e-9)
    tr = token_tp / max(token_tp + token_fn, 1e-9)
    out.update({"token_p": tp, "token_r": tr, "token_f1": 2 * tp * tr / max(tp + tr, 1e-9)})
    return out


# --------------------------------------------------------------------------- #
def by_bucket(blob, edges, fn, thr: float) -> None:
    """按 token 长度分桶，逐桶调用 ``fn``。"""
    length = blob["length"].long()
    print(f"    {'区间':>16s} {'n':>6s}   指标")
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (length >= lo) & (length < hi)
        n = int(sel.sum())
        if n == 0:
            print(f"    {f'[{lo:g},{hi:g})':>16s} {0:>6d}   —")
            continue
        sub = dict(blob)
        sub["n"] = n
        sub["length"] = length[sel]
        sub["label"] = blob["label"][sel]
        for k in ("line_label", "line_prob", "token_prob", "token_label"):
            if k in blob:
                sub[k] = [blob[k][i] for i in range(len(length)) if bool(sel[i])]
        sub["sample_prob"] = blob["sample_prob"][sel]
        m = fn(sub, thr)
        txt = "  ".join(f"{k}={v:.4f}" for k, v in m.items())
        print(f"    {f'[{lo:g},{hi:g})':>16s} {n:>6d}   {txt}")


def sweep(blob, fn, thr_grid) -> None:
    """阈值扫描：对每个阈值重算指标，标出使 f1 最大的阈值。"""
    rows = [(t, fn(blob, t)) for t in thr_grid]
    key = "f1" if "f1" in rows[0][1] else "line_f1"
    best_t, best_m = max(rows, key=lambda r: r[1][key])
    for t, m in rows:
        mark = "  ← 最优" if t == best_t else ""
        txt = "  ".join(f"{k}={v:.4f}" for k, v in m.items())
        print(f"    thr={t:.2f}  {txt}{mark}")


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description="离线分析 raw_*.pt：长度分桶 + 阈值扫描")
    ap.add_argument("runs", nargs="+", help="run 目录，如 runs/v0.4.4")
    ap.add_argument("--split", default="test", choices=["test", "val"])
    ap.add_argument("--edges", type=float, nargs="+", default=[0, 512, 2048, 8192, 1e9],
                    help="长度分桶边界（token 数）")
    ap.add_argument("--thr", type=float, nargs="+",
                    default=[0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70])
    ap.add_argument("--thr-opt", type=float, default=0.50, help="分桶报告使用的阈值")
    args = ap.parse_args()

    for spec in args.runs:
        d = Path(spec)
        print(f"\n{'=' * 78}\n=== {spec}（split={args.split}）\n{'=' * 78}")
        blob = load(d, "m4", args.split)
        if blob is not None:
            print(f"\n  【m4】n={blob['n']}  长度中位={int(blob['length'].median())}  "
                  f"max={int(blob['length'].max())}")
            print("  ── 阈值扫描（thr=0.5 等价于仓库内评测的 argmax）")
            sweep(blob, m4_metrics, args.thr)
            print(f"  ── 长度分桶（thr={args.thr_opt}）")
            by_bucket(blob, args.edges, m4_metrics, args.thr_opt)
        else:
            print("  (无 raw_m4_*.pt，跳过 m4)")

        blob = load(d, "hybrid", args.split)
        if blob is not None:
            print(f"\n  【hybrid】n={blob['n']}  长度中位={int(blob['length'].median())}  "
                  f"max={int(blob['length'].max())}")
            print("  ── 阈值扫描（行级/片段级/token 级；仓库内评测写死 0.5）")
            sweep(blob, hybrid_metrics, args.thr)
            print(f"  ── 长度分桶（thr={args.thr_opt}）")
            by_bucket(blob, args.edges, hybrid_metrics, args.thr_opt)
        else:
            print("  (无 raw_hybrid_*.pt，跳过 hybrid)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
