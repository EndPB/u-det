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
# v0.4.9：「额外一路头」的三路消融（用 dump 里分开存的 logits，**不需要重训**）
# --------------------------------------------------------------------------- #
def hybrid_metrics_from_tokens(blob, thr: float, probs_fn) -> dict:
    """从**逐 token 概率**重算行级 / 片段级 / token 级（与 ``evaluate`` 完全同构）。

    ``probs_fn(i) -> (L,) 概率``。行级用 dump 里的 ``line_of_token`` 聚合，
    只统计 ``token_label >= 0``（即非 IGNORE）的位置 —— 与 ``train.evaluate`` 一致。
    """
    pred_lines, gold_lines = [], []
    chunk_tp = chunk_pred = chunk_gold = 0
    token_tp = token_fp = token_fn = 0
    for i in range(blob["n"]):
        prob = probs_fn(i)
        gold = blob["token_label"][i].long()
        lo = blob["line_of_token"][i].long()
        keep = gold >= 0
        p, g, l = prob[keep], gold[keep], lo[keep]
        # token 级
        ok = p > thr
        gi = g == 1
        token_tp += int((ok & gi).sum())
        token_fp += int((ok & ~gi).sum())
        token_fn += int((~ok & gi).sum())
        # 行级：逐行取该行所有 token 概率的均值
        n_line = len(blob["line_label"][i])
        ssum = torch.zeros(n_line, dtype=torch.float64)
        scnt = torch.zeros(n_line, dtype=torch.float64)
        ssum.scatter_add_(0, l, p.double())
        scnt.scatter_add_(0, l, torch.ones_like(p).double())
        picked = [int(ssum[k] / scnt[k].item() > thr) if scnt[k] > 0 else 0
                  for k in range(n_line)]
        gold_l = blob["line_label"][i].long().tolist()
        pred_lines.extend(picked)
        gold_lines.extend(gold_l)
        tp, np_, ng = chunk_match(picked, gold_l)
        chunk_tp += tp
        chunk_pred += np_
        chunk_gold += ng
    line = prf1(pred_lines, gold_lines)
    cp = chunk_tp / max(chunk_pred, 1e-9)
    cr = chunk_tp / max(chunk_gold, 1e-9)
    tp_ = token_tp / max(token_tp + token_fp, 1e-9)
    tr_ = token_tp / max(token_tp + token_fn, 1e-9)
    return {"line_f1": line["f1"], "chunk_f1": 2 * cp * cr / max(cp + cr, 1e-9),
            "token_f1": 2 * tp_ * tr_ / max(tp_ + tr_, 1e-9)}


def head_ablation(blob, thr: float, weights=(0.0, 0.25, 0.5, 0.75, 1.0)) -> None:
    """主干那一级 / 额外一路头 / 两者加权混合 —— 三种读出的对比。

    ★ 这是「只跑一次训练」也能拿到消融的办法：``--dump-raw`` 时把两个头的逐 token
      logits 分开存，离线在 **logit 空间**按权重 ``w`` 混合
      （``w`` = 主干权重，``1−w`` = 额外头权重）再 sigmoid。
      ``w=0.5`` 就是 v0.4.9 的官方口径（两者取均值）。
    """
    tr = blob.get("token_logit_trunk")
    by = blob.get("token_logit_bypass")
    print("\n  ── ★ 额外一路头的三路消融（离线，零 GPU；logit 空间混合 w·trunk + (1−w)·bypass）")

    # 自检：w=0.5 的混合应当复现官方存下来的 token_prob（B8：评测是确定性的）
    def mix(i, w):
        a = tr[i].float()
        b = by[i].float()
        return torch.sigmoid(w * a + (1 - w) * b)

    ref = hybrid_metrics_from_tokens(blob, thr, lambda i: blob["token_prob"][i].float())
    half = hybrid_metrics_from_tokens(blob, thr, lambda i: mix(i, 0.5))
    d = max(abs(ref[k] - half[k]) for k in ref)
    print(f"    自检：w=0.5 离线重算 vs 官方 token_prob 的最大差 = {d:.2e}"
          f"  [{'OK' if d < 2e-3 else '!! 不一致，离线重算逻辑有漂移'}]")

    print(f"    {'主干权重 w':>10}{'额外头权重':>12}{'行级 F1':>10}{'片段 F1':>10}{'token F1':>10}")
    for w in weights:
        m = hybrid_metrics_from_tokens(blob, thr, lambda i, w=w: mix(i, w))
        tag = "   ← 官方口径" if abs(w - 0.5) < 1e-9 else ""
        print(f"    {w:>10.2f}{1 - w:>12.2f}{m['line_f1']:>10.4f}"
              f"{m['chunk_f1']:>10.4f}{m['token_f1']:>10.4f}{tag}")
    print("    （w=1 = 只用主干那一级；w=0 = 只用额外头；" 
          "若 0.5 明显不如 1.0 ⇒ 额外头在**拖后腿**，应把权重调小或直接去掉）")


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
    ap.add_argument("--heads", action="store_true",
                    help="v0.4.9：若 dump 里分开存了主干头/额外头的 logits，做三路消融")
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
            if args.heads and blob.get("token_logit_trunk") and blob.get("token_logit_bypass"):
                head_ablation(blob, args.thr_opt)
            elif args.heads:
                print("  (该 run 的 dump 里没有分开存的两路 logits ⇒ 跳过三路消融；"
                      "只有 v0.4.9 之后的 run 才有)")
        else:
            print("  (无 raw_hybrid_*.pt，跳过 hybrid)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
