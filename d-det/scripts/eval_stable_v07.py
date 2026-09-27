#!/usr/bin/env python
"""v0.7 稳定版评测套件：二分类 / 四分类 / 归因（零训练、确定性、无调参）。

三件套任务（全部基于已有的 6 族配对数据，不新增任何数据/模型）：
  T1 二分类（样本级）：族对判别（Qwen0.5B↔Qwen1.5B、Granite2B↔SmolLM2、DS1.3B↔Yi1.5B、Qwen1.5B↔DS1.3B）
  T2 四分类（样本级）：四族子集判别（3 组）
  T3 归因：六族判别（全量 5703 / 329 共同题对齐）

每个任务对比 3 种读法（判别器固定为 DiscHead 默认参数，不做任何调参）：
  旧口径： Δ = h⁺ − h⁻（位移）
  新·非转导： [h⁺; h⁻]（双侧联合）
  新·转导： 逐题中心化后的 [h̃⁺; h̃⁻]（需同题兄弟样本；减掉题目项）

协议：按题 GroupKFold=5（同一道题不跨折）；标准化仅用训练折统计。确定性可复现。
输出：runs/v0.7_stable/eval_suite.json
"""
from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import sklearn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from kernel_e17_framework import cv_disc, summarize  # noqa: E402

F16 = ROOT / "runs/kernel_e16"
RES = ROOT / "runs/v0.7_stable"
NAMES = ["qwen05", "qwen15", "ds13", "yi15", "granite2b", "smollm2"]

T1_PAIRS = [
    ("qwen05", "qwen15"),
    ("granite2b", "smollm2"),
    ("ds13", "yi15"),
    ("qwen15", "ds13"),
]
T2_QUADS = [
    ["qwen05", "qwen15", "granite2b", "smollm2"],
    ["qwen05", "ds13", "yi15", "granite2b"],
    ["granite2b", "smollm2", "ds13", "yi15"],
]


def load_fam(name):
    z = np.load(F16 / f"h_{name}.npz", allow_pickle=True)
    return (np.asarray(z["h_plus"], dtype="float64"),
            np.asarray(z["h_minus"], dtype="float64"),
            [str(v) for v in z["tasks"]])


def subset_arrays(names):
    """取各族的共同题；返回 HP/HM (k,T,768), tasks。"""
    fams = [load_fam(n) for n in names]
    common = None
    for _, _, t in fams:
        s = set(t)
        common = s if common is None else (common & s)
    common = sorted(common)
    HP, HM = [], []
    for hp, hm, t in fams:
        idx = {tt: i for i, tt in enumerate(t)}
        sel = [idx[tt] for tt in common]
        HP.append(hp[sel])
        HM.append(hm[sel])
    return np.stack(HP), np.stack(HM), common


def run_task(names, tag, report):
    HP, HM, common = subset_arrays(names)
    y = np.concatenate([[n] * len(common) for n in names])
    g = np.concatenate([np.array(common) for _ in names])
    HPc = HP - HP.mean(axis=0, keepdims=True)
    HMc = HM - HM.mean(axis=0, keepdims=True)
    views = {
        "delta_old": np.hstack([np.vstack(list(HP - HM))]),
        "pair_new": np.hstack([np.vstack(list(HP)), np.vstack(list(HM))]),
        "pair_center_new": np.hstack([np.vstack(list(HPc)), np.vstack(list(HMc))]),
    }
    entry = {"n_families": len(names), "n_tasks": len(common),
             "n_samples": int(len(y)), "chance": round(1.0 / len(names), 4)}
    for v, X in views.items():
        m = summarize(cv_disc(X, y, g), y)
        entry[v] = {"acc": m["acc"], "balanced_acc": m["balanced_acc"]}
    report[tag] = entry
    print(f"[v07] {tag:<26}: 旧 {entry['delta_old']['acc']:.4f} → "
          f"非转导 {entry['pair_new']['acc']:.4f} → 转导 {entry['pair_center_new']['acc']:.4f}",
          flush=True)


def run_full_attribution(report):
    """六族：全量（非转导）与 329 共同题（转导中心化）双口径。"""
    # 全量：各族的全部样本（不取共同题）
    HP_l, HM_l, y_l, g_l = [], [], [], []
    for n in NAMES:
        hp, hm, t = load_fam(n)
        HP_l.append(hp); HM_l.append(hm)
        y_l += [n] * len(t)
        g_l += t
    HPf = np.vstack(HP_l); HMf = np.vstack(HM_l)
    yf = np.array(y_l); gf = np.array(g_l)
    entry = {"n_samples": int(len(yf)), "n_tasks": int(len(set(gf.tolist())))}
    for v, X in (("delta_old", HPf - HMf),
                 ("pair_new", np.hstack([HPf, HMf]))):
        m = summarize(cv_disc(X, yf, gf), yf)
        entry[v] = {"acc": m["acc"], "balanced_acc": m["balanced_acc"]}
    report["attribution_full"] = entry
    print(f"[v07] 归因·全量 {entry['n_samples']} 样本: 旧 {entry['delta_old']['acc']:.4f} → "
          f"新 {entry['pair_new']['acc']:.4f}", flush=True)

    # 329 共同题（转导）
    HP, HM, common = subset_arrays(NAMES)
    y = np.concatenate([[n] * len(common) for n in NAMES])
    g = np.concatenate([np.array(common) for _ in NAMES])
    HPc = HP - HP.mean(axis=0, keepdims=True)
    HMc = HM - HM.mean(axis=0, keepdims=True)
    entry2 = {"n_tasks": len(common), "n_samples": int(len(y))}
    for v, X in (("delta_center_old", np.hstack([np.vstack(list(HPc - HMc))])),
                 ("pair_center_new", np.hstack([np.vstack(list(HPc)), np.vstack(list(HMc))]))):
        m = summarize(cv_disc(X, y, g), y)
        entry2[v] = {"acc": m["acc"], "balanced_acc": m["balanced_acc"]}
    report["attribution_aligned"] = entry2
    print(f"[v07] 归因·对齐 {entry2['n_tasks']} 题: 旧 {entry2['delta_center_old']['acc']:.4f} → "
          f"新 {entry2['pair_center_new']['acc']:.4f}", flush=True)


def main() -> int:
    RES.mkdir(parents=True, exist_ok=True)
    report = {}
    print(f"[v07] python {platform.python_version()} / sklearn {sklearn.__version__}", flush=True)
    for a, b in T1_PAIRS:
        run_task([a, b], f"binary_{a}_vs_{b}", report)
    for quad in T2_QUADS:
        run_task(quad, "four_" + "_".join(quad), report)
    run_full_attribution(report)
    report["meta"] = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "protocol": "按题 GroupKFold=5；DiscHead 默认参数（无调参）；中心化=族组内逐题去均值（不用标签）",
        "views": {"delta_old": "Δ=h+−h−（旧口径）",
                  "pair_new": "[h+;h−]（新·非转导）",
                  "pair_center_new": "[h~+;h~−]（新·转导，需同题兄弟样本）"},
        "budget_note": "全流程零训练（闭式 LDA），满足\"单方法 ≤2ep\"预算纪律",
    }
    out = RES / "eval_suite.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n[v07] 写出 {out}")
    print("[v07] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
