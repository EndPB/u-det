#!/usr/bin/env python
"""E17c：新视图（[h⁺;h⁻] ± 逐题中心化）的复核（不新增数据；纯 CPU）。

复核三件事：
  ① 全量 [h⁺;h⁻] 的混淆结构（对照 Δ 全量 0.7626；新视图 cross_val 0.8271）
  ② 易混对二分：qwen05 vs qwen15（Δ 参考 0.7896）、granite2b vs smollm2（Δ 参考 0.8472）
     —— 比较 Δ / [h⁺;h⁻] / 中心化 [h⁺;h⁻] 三视图
  ③ E11 二族复现：qwen15 vs ds13 共同题（Δ 参考 ≈0.88/0.885）—— 新视图是否更高
输出：runs/kernel_e17/verify.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from kernel_e17_framework import cv_disc, load_full, summarize  # noqa: E402

F16 = ROOT / "runs/kernel_e16"
RES = ROOT / "runs/kernel_e17"


def load_family(name):
    z = np.load(F16 / f"h_{name}.npz", allow_pickle=True)
    return (np.asarray(z["h_plus"], dtype="float64"),
            np.asarray(z["h_minus"], dtype="float64"),
            [str(v) for v in z["tasks"]])


def pair_experiment(a, b):
    """返回 {视图: {acc, bal}}；视图 = delta / hp_hm / hp_hm_c（中心化）。"""
    da, db = load_family(a), load_family(b)
    common = sorted(set(da[2]) & set(db[2]))
    datas = []
    for d in (da, db):
        idx = {tt: i for i, tt in enumerate(d[2])}
        sel = [idx[tt] for tt in common]
        datas.append((d[0][sel], d[1][sel]))
    HPs = np.stack([d[0] for d in datas])
    HMs = np.stack([d[1] for d in datas])
    y = np.concatenate([[a] * len(common), [b] * len(common)])
    g = np.concatenate([np.array(common), np.array(common)])
    out = {}
    views = {
        "delta": np.hstack([np.vstack(list(HPs - HMs))]),
        "hp_hm": np.hstack([np.vstack(list(HPs)), np.vstack(list(HMs))]),
        "hp_hm_c": np.hstack([np.vstack(list(HPs - HPs.mean(0, keepdims=True))),
                              np.vstack(list(HMs - HMs.mean(0, keepdims=True)))]),
    }
    for tag, X in views.items():
        m = summarize(cv_disc(X, y, g), y)
        out[tag] = {"acc": m["acc"], "balanced_acc": m["balanced_acc"]}
    return out, len(common)


def main() -> int:
    RES.mkdir(parents=True, exist_ok=True)
    report = {}

    # ① 全量 [h+;h-] 混淆
    Xd, Xhp, Xhm, yf, gf = load_full()
    m = summarize(cv_disc(np.hstack([Xhp, Xhm]), yf, gf), yf)
    report["full_hp_hm"] = m
    print(f"[e17c] 全量 [h+;h-]: acc {m['acc']} / bal {m['balanced_acc']}", flush=True)
    print(f"[e17c]   conf {m['conf']}", flush=True)

    # ② 易混对二分
    for a, b in (("qwen05", "qwen15"), ("granite2b", "smollm2")):
        out, n = pair_experiment(a, b)
        report[f"binary_{a}_vs_{b}"] = {"n_tasks": n, **out}
        print(f"[e17c] 二分 {a} vs {b}（{n} 题）:", flush=True)
        for tag, m2 in out.items():
            print(f"[e17c]   {tag:<8}: acc {m2['acc']:.4f}", flush=True)

    # ③ E11 二族复现（qwen15 vs ds13）
    out, n = pair_experiment("qwen15", "ds13")
    report["e11_repl_qwen15_ds13"] = {"n_tasks": n, **out}
    print(f"[e17c] E11 复现 qwen15 vs ds13（{n} 题）:", flush=True)
    for tag, m2 in out.items():
        print(f"[e17c]   {tag:<8}: acc {m2['acc']:.4f}", flush=True)

    out_p = RES / "verify.json"
    out_p.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n[e17c] 写出 {out_p}")
    print("[e17c] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
