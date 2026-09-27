#!/usr/bin/env python
"""E17b：视图贡献分解（不新增数据；纯 CPU）。

E17 筛查发现：[Δ;h⁺;h⁻] 全量 acc 0.8340 / bal 0.8099（Δ 单用 0.7626/0.7355）。
本脚本做三件事：
  ① 单块视图：h⁺ 单独 / h⁻ 单独 / [h⁺;h⁻]（全量）——检验"单文本归因（免参照对）"是否成立；
  ② 对齐口径的组合与**逐块中心化**：[Δ;h⁺;h⁻] 及中心化版——检验"位置信息"与"任务效应消除"能否叠加；
  ③ 最佳组合上的子空间 QDA。

判读（预注册）：① h⁺ 单用全量 ≥0.78 → 单文本归因（免参照对）成立；
② 组合+中心化若再超 0.834 → 升为新默认视图。
输出：runs/kernel_e17/views.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from models.disc import DiscHead  # noqa: E402
from kernel_e17_framework import (load_full, summarize, cv_disc,  # noqa: E402
                                  NAMES)

F16 = ROOT / "runs/kernel_e16"
RES = ROOT / "runs/kernel_e17"


def load_h_aligned():
    common = None
    mats = []
    for n in NAMES:
        z = np.load(F16 / f"h_{n}.npz", allow_pickle=True)
        hp = np.asarray(z["h_plus"], dtype="float64")
        hm = np.asarray(z["h_minus"], dtype="float64")
        t = [str(v) for v in z["tasks"]]
        common = set(t) if common is None else (common & set(t))
        mats.append((hp, hm, t))
    common = sorted(common)
    HP, HM = [], []
    for hp, hm, t in mats:
        idx = {tt: i for i, tt in enumerate(t)}
        sel = [idx[tt] for tt in common]
        HP.append(hp[sel])
        HM.append(hm[sel])
    return np.stack(HP), np.stack(HM), common


def center_stack(*arrs):
    """每个块沿族轴逐题去均值后按样本堆叠拼接。"""
    out = []
    for A in arrs:                      # A: (k, T, d)
        m = A.mean(axis=0, keepdims=True)
        out.append(np.vstack(list(A - m)))
    return np.hstack(out)


def main() -> int:
    RES.mkdir(parents=True, exist_ok=True)
    report = {}

    # ---------- 全量：新增单块视图 ----------
    Xd, Xhp, Xhm, yf, gf = load_full()
    print(f"[e17b] 全量 {Xd.shape}", flush=True)
    views_full = {"h_plus": Xhp, "h_minus": Xhm, "hp_hm": np.hstack([Xhp, Xhm])}
    for tag, Xv in views_full.items():
        r = DiscHead().cross_val(Xv, yf, gf)
        report[f"full_{tag}"] = {k: round(float(v), 4) for k, v in r.items()
                                 if k in ("acc", "balanced_acc")}
        print(f"[e17b] full {tag:<8}: acc {report[f'full_{tag}']['acc']:.4f} / "
              f"bal {report[f'full_{tag}']['balanced_acc']:.4f}", flush=True)

    # ---------- 对齐：组合与逐块中心化 ----------
    HP, HM, common = load_h_aligned()
    D = HP - HM
    y = np.concatenate([[n] * len(common) for n in NAMES])
    g = np.concatenate([np.array(common) for _ in NAMES])
    combos = {
        "al_delta": np.vstack(list(D)),
        "al_delta_center": center_stack(D),
        "al_hp_hm": np.hstack([np.vstack(list(HP)), np.vstack(list(HM))]),
        "al_hp_hm_center": center_stack(HP, HM),
        "al_all": np.hstack([np.vstack(list(D)), np.vstack(list(HP)), np.vstack(list(HM))]),
        "al_all_center": center_stack(D, HP, HM),
    }
    for tag, Xv in combos.items():
        pred = cv_disc(Xv, y, g)
        report[tag] = summarize(pred, y)
        print(f"[e17b] {tag:<16}: acc {report[tag]['acc']:.4f} / "
              f"bal {report[tag]['balanced_acc']:.4f}", flush=True)

    # ---------- 最佳组合上的子空间 QDA（对齐） ----------
    for tag in ("al_all", "al_all_center"):
        Xv = combos[tag]
        pred = np.empty(len(y), dtype=object)
        for tr, va in GroupKFold(n_splits=5).split(Xv, y, g):
            h = DiscHead().fit(Xv[tr], y[tr])
            Ztr, Zva = h.transform(Xv[tr]), h.transform(Xv[va])
            q = QDA(solver="eigen", shrinkage=0.05).fit(Ztr, y[tr])
            pred[va] = q.predict(Zva)
        pred = pred.astype(str)
        report[f"{tag}_qda"] = summarize(pred, y)
        print(f"[e17b] {tag}_qda: acc {report[f'{tag}_qda']['acc']:.4f} / "
              f"bal {report[f'{tag}_qda']['balanced_acc']:.4f}", flush=True)

    out = RES / "views.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n[e17b] 写出 {out}")
    print("[e17b] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
