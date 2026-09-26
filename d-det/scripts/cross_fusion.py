#!/usr/bin/env python
"""离线跨模型融合矩阵：s1_X + s2_Y 组合在跨族池上的 fused AUC（5×4 CV）。

npz 由 probe_scale_verify.py 生成（键 hum/ai4/plus/minus，行=[s1,s2,len]，行对齐）——
同一人类池、同一 pair 文件顺序 ⇒ **任意模型的 s1 可与任意模型的 s2 自由组合**。

用法：
    python scripts/cross_fusion.py --title qwen15 \
        --npz v0.3.0=runs/v0.3.0_fullft/scale_scores.npz \
        --npz v0.4.0=runs/v0.4.0_geom/scale_scores.npz \
        --npz v0.4.1=runs/v0.4.1_covreg/scale_scores.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def fused_auc(y, X, n_repeats=4):
    """5×4 重复分层 CV 的逻辑回归 AUC（与 probe_scale_verify 同口径）。"""
    aucs = []
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_repeats, random_state=0)
    for tr, te in rskf.split(X, y):
        clf = LogisticRegression(max_iter=2000).fit(X[tr], y[tr])
        aucs.append(roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1]))
    return float(np.mean(aucs)), float(np.std(aucs))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", action="append", required=True, metavar="TAG=PATH")
    ap.add_argument("--title", default="pool")
    args = ap.parse_args()

    tags, data = [], {}
    for item in args.npz:
        tag, _, path = item.partition("=")
        d = np.load(str(ROOT / path))
        data[tag] = {"hum": d["hum"], "plus": d["plus"], "minus": d["minus"]}
        tags.append(tag)

    for pool in ("instruct", "base"):
        key = "plus" if pool == "instruct" else "minus"
        print(f"\n===== {args.title} / {pool} vs human：fused AUC（行=s1 来源，列=s2 来源） =====")
        header = "  s1 \\ s2  | " + " | ".join(f"{t:>9s}" for t in tags) + " || s1-only"
        print(header)
        for t1 in tags:
            row = []
            s1 = data[t1]["hum"][:, 0]
            s1p = data[t1][key][:, 0]
            y = np.r_[np.zeros(len(s1)), np.ones(len(s1p))]
            s1_all = np.r_[s1, s1p]
            only = roc_auc_score(y, s1_all)
            for t2 in tags:
                s2p = data[t2][key][:, 1]
                X = np.stack([s1_all, np.r_[data[t2]["hum"][:, 1], s2p]], 1)
                m, _ = fused_auc(y, X)
                row.append(m)
            print(f"  {t1:>9s} | " + " | ".join(f"{v:9.4f}" for v in row) + f" || {only:.4f}")

    print("\n（s2* 参考：probe 输出里 report 行；本脚本只列融合矩阵与 s1-only）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
