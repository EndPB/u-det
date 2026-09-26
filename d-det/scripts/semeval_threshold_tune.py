#!/usr/bin/env python
"""阈值/偏置调优（后处理，零训练）：val OOF 上坐标上升搜 per-class 乘法权重 → test。

用于回答"不平衡宏 F1 的即得空间有多大"（B 的 test 小族召回差是主痛点）。
输出：runs/semeval_ensemble/threshold_tune.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from semeval_ensemble_v2 import lab, lgb_fit, member_block  # noqa: E402


def macro_f1(pred, y) -> float:
    f1s = []
    for c in np.unique(y):
        tp = np.sum((pred == c) & (y == c))
        fp = np.sum((pred == c) & (y != c))
        fn = np.sum((pred != c) & (y == c))
        f1s.append(2 * tp / max(2 * tp + fp + fn, 1))
    return float(np.mean(f1s))


def hier_cv(X, y, n=5, seed=0):
    nc = int(np.max(y)) + 1
    oof = np.zeros((len(y), nc))
    skf = StratifiedKFold(n_splits=n, shuffle=True, random_state=seed)
    for tr, va in skf.split(X, y):
        Xtr, Xva, ytr = X[tr], X[va], y[tr]
        gate = lgb_fit(Xtr, (ytr == 0).astype(int))
        pg = gate.predict_proba(Xva)[:, 1]
        m = ytr > 0
        mclf = lgb_fit(Xtr[m], ytr[m] - 1)
        pm = mclf.predict_proba(Xva)
        oof[va, 0] = pg
        oof[va, 1:] = (1 - pg)[:, None] * pm
    return oof


def hier_fit_predict(X, y, Xt):
    nc = int(np.max(y)) + 1
    gate = lgb_fit(X, (y == 0).astype(int))
    pg = gate.predict_proba(Xt)[:, 1]
    m = y > 0
    mclf = lgb_fit(X[m], y[m] - 1)
    pm = mclf.predict_proba(Xt)
    P = np.zeros((len(Xt), nc))
    P[:, 0] = pg
    P[:, 1:] = (1 - pg)[:, None] * pm
    return P


def coord_ascent(P, y, n_rounds=3):
    """坐标上升：每类一个乘法权重（grid 搜索），最大化 macro-F1。"""
    grid = np.array([0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0])
    nc = P.shape[1]
    w = np.ones(nc)
    best = macro_f1(P.argmax(1), y)
    for _ in range(n_rounds):
        improved = False
        for c in range(1, nc):
            cur = w[c]
            for g in grid:
                w[c] = g
                s = macro_f1((P * w).argmax(1), y)
                if s > best + 1e-9:
                    best, cur, improved = s, g, True
            w[c] = cur
        if not improved:
            break
    return w, best


def main() -> int:
    res = {}
    for task in ("b", "c"):
        try:
            yv, yt = lab(task, "val"), lab(task, "test")
            Xv, names, _ = member_block(task, "val", yv)
            Xt, _, _ = member_block(task, "test", yt)
            oof = hier_cv(Xv, yv)
            Pt = hier_fit_predict(Xv, yv, Xt)
            base_v = macro_f1(oof.argmax(1), yv)
            base_t = macro_f1(Pt.argmax(1), yt)
            w, tuned_v = coord_ascent(oof, yv)
            tuned_t = macro_f1((Pt * w).argmax(1), yt)
            res[task] = {"base_cv_val": round(base_v, 4), "tuned_cv_val": round(tuned_v, 4),
                         "base_test": round(base_t, 4), "tuned_test": round(tuned_t, 4),
                         "weights": [round(float(x), 3) for x in w],
                         "n_features": int(Xv.shape[1])}
            print(f"[tune] {task}: cv_val {base_v:.4f}->{tuned_v:.4f} | "
                  f"test {base_t:.4f}->{tuned_t:.4f} | w={np.round(w, 2)}", flush=True)
        except Exception as e:
            res[task] = {"error": f"{type(e).__name__}: {e}"}
            print(f"[tune] {task} 失败：{e}", flush=True)
    out = ROOT / "runs/semeval_ensemble/threshold_tune.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print("[tune] done ->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
