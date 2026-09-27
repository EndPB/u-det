#!/usr/bin/env python
"""E17d：中心化收益 vs 候选集规模 k 的部署边界曲线（不新增数据；纯 CPU）。

问题：转导中心化 [h̃⁺;h̃⁻] 的 +8.3pt（对齐 6 族）需要"同题兄弟样本"。
真实部署时候选集可能只有 k 个家族（k=2..6）。本实验测：
  对给定族子集 S（|S|=k），视图 V0=不中心化 [h⁺;h⁻] vs V1=组内中心化 [h̃⁺;h̃⁻]
  （中心化=减去本题 S 内 k 个样本的均值），5 折按题 GroupKFold，k 类判别。

判读：中心化增益随 k 的衰减曲线；k=2/3 时增益仍显著 → 低配部署可行。
输出：runs/kernel_e17/kwise.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from kernel_e17_framework import cv_disc, summarize  # noqa: E402
from kernel_e17b_views import load_h_aligned  # noqa: E402

RES = ROOT / "runs/kernel_e17"
NAMES = ["qwen05", "qwen15", "ds13", "yi15", "granite2b", "smollm2"]
NAME_IDX = {n: i for i, n in enumerate(NAMES)}

SUBSETS = {
    "k2_qwen05_qwen15": ["qwen05", "qwen15"],
    "k2_granite_smollm": ["granite2b", "smollm2"],
    "k2_ds13_yi15": ["ds13", "yi15"],
    "k3_qwen05_qwen15_ds13": ["qwen05", "qwen15", "ds13"],
    "k3_granite_smollm_qwen05": ["granite2b", "smollm2", "qwen05"],
    "k3_ds13_yi15_granite": ["ds13", "yi15", "granite2b"],
    "k4_qwen_pair_gr_sm": ["qwen05", "qwen15", "granite2b", "smollm2"],
    "k4_qwen05_ds13_yi15_granite": ["qwen05", "ds13", "yi15", "granite2b"],
    "k5_all_but_smollm": ["qwen05", "qwen15", "ds13", "yi15", "granite2b"],
    "k6_all": NAMES,
}


def run_subset(HP_all, HM_all, common, S):
    idxs = [NAME_IDX[s] for s in S]
    out = {}
    for tag, center in (("none", False), ("center", True)):
        HP = HP_all[idxs].copy()
        HM = HM_all[idxs].copy()
        if center:
            HP = HP - HP.mean(axis=0, keepdims=True)
            HM = HM - HM.mean(axis=0, keepdims=True)
        X = np.hstack([np.vstack(list(HP)), np.vstack(list(HM))])
        y = np.concatenate([[s] * len(common) for s in S])
        g = np.concatenate([np.array(common) for _ in S])
        m = summarize(cv_disc(X, y, g), y)
        out[tag] = {"acc": m["acc"], "balanced_acc": m["balanced_acc"]}
    out["delta_acc"] = round(out["center"]["acc"] - out["none"]["acc"], 4)
    out["chance"] = round(1.0 / len(S), 4)
    return out


def main() -> int:
    report = {}
    HP_all, HM_all, common = load_h_aligned()
    print(f"[e17d] 对齐 {HP_all.shape[1]} 题 × 6 族", flush=True)
    for name, S in SUBSETS.items():
        r = run_subset(HP_all, HM_all, common, S)
        report[name] = r
        print(f"[e17d] {name:<28}: 无中心化 {r['none']['acc']:.4f} → "
              f"中心化 {r['center']['acc']:.4f}（Δ {r['delta_acc']:+.4f}，chance {r['chance']}）",
              flush=True)
    out = RES / "kwise.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n[e17d] 写出 {out}")
    print("[e17d] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
