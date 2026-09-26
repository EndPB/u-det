#!/usr/bin/env python
"""E6（mini）：多参考对 r* 特征——"多方向"图景的验证。

- 样本：E1 同 198 条；参考对 = Qwen1.5B ∪ DeepSeek-1.3B ∪ Qwen0.5B（本脚本补打 0.5B）
- 检测（人机）：单 r* AUC 对照 + 3 维拼接的 LOO-LR AUC + 拼接 vs s1 / 拼接+s1
- 归因（21 机器族）：NCM（最近族质心）LOO balanced acc，单维 vs 拼接
- 输出：runs/kernel_e6/summary.json（0.5B logp 同存）
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from kernel_e1_rstar import build_samples, score_model  # noqa: E402

OUT = ROOT / "runs/kernel_e6"
Q05 = {"base": ROOT / "checkpoints/qwen2.5-coder-0.5b-base",
       "instruct": ROOT / "checkpoints/qwen2.5-coder-0.5b-instruct"}


def loo_lr_auc(F, y):
    """LOO-CV 的 LR 预测概率 AUC（F 已标准化）。"""
    scores = np.zeros(len(y))
    for i in range(len(y)):
        m = np.ones(len(y), bool)
        m[i] = False
        clf = LogisticRegression(C=1.0, max_iter=500)
        clf.fit(F[m], y[m])
        scores[i] = clf.predict_proba(F[i:i + 1])[0, 1]
    return round(float(roc_auc_score(y, scores)), 4)


def ncm_loo(X, z):
    """最近族质心 LOO balanced acc。"""
    X = np.asarray(X, dtype="float64")
    pred = np.zeros(len(z), dtype=object)
    for i in range(len(z)):
        m = np.ones(len(z), bool)
        m[i] = False
        cents, labs = [], []
        for c in set(z[m].tolist()):
            idx = m & (z == c)
            if idx.sum() < 1:
                continue
            cents.append(X[idx].mean(0))
            labs.append(c)
        d = [np.linalg.norm(X[i] - cent) for cent in cents]
        pred[i] = labs[int(np.argmin(d))]
    z = np.asarray(z)
    accs = [np.mean(pred[z == c] == c) for c in set(z.tolist())]
    return round(float(np.mean(accs)), 4)


def main() -> int:
    smp = build_samples()
    OUT.mkdir(parents=True, exist_ok=True)
    # 0.5B 打分（若缓存缺失）
    cache = OUT / "logp_05.npz"
    if cache.exists():
        d = np.load(cache)
        r05 = d["r"]
    else:
        lp = {}
        for tag, path in Q05.items():
            tj, mn = score_model(path, smp["code"])
            lp[tag] = (tj, mn)
            np.savez_compressed(OUT / f"logp05_{tag}.npz", total=tj, mean=mn)
        r05 = lp["instruct"][0] - lp["base"][0]
        np.savez_compressed(cache, r=r05)
    print(f"[e6] 0.5B r* 就绪（mean={r05.mean():.2f}）", flush=True)

    e1 = np.load(ROOT / "runs/kernel_e1/rstar_total.npz", allow_pickle=True)
    r_qw, y, lab, gen = e1["r"], e1["y"], e1["label"], e1["generator"]
    e5 = np.load(ROOT / "runs/kernel_e5/ds_logp_instruct.npz")
    ds_b = np.load(ROOT / "runs/kernel_e5/ds_logp_base.npz")
    r_ds = e5["total"] - ds_b["total"]

    from scipy.stats import pearsonr
    res = {"n": int(len(y))}
    res["corr_05_vs_qw"] = round(float(pearsonr(r05, r_qw)[0]), 4)
    res["corr_05_vs_ds"] = round(float(pearsonr(r05, r_ds)[0]), 4)
    for tag, r in (("r_qw15", r_qw), ("r_ds13", r_ds), ("r_qw05", r05)):
        res[f"auc_{tag}"] = round(float(roc_auc_score(y, r)), 4)

    # ---- 检测：拼接 LOO-LR ----
    R3 = np.vstack([r_qw, r_ds, r05]).T
    Sc = StandardScaler().fit(R3)
    R3s = Sc.transform(R3)
    res["auc_R3_loo"] = loo_lr_auc(R3s, y)
    s1 = smp["s1"].reshape(-1, 1)
    S1s = StandardScaler().fit(s1).transform(s1)
    res["auc_s1_200"] = round(float(roc_auc_score(y, s1.reshape(-1))), 4)
    res["auc_R3_s1_loo"] = loo_lr_auc(np.hstack([R3s, S1s]), y)

    # ---- 归因：机器族 NCM LOO ----
    m = lab > 0
    z = np.array([str(g) for g in gen])[m]
    Rm = R3s[m]
    for tag, col in (("r_qw15", 0), ("r_ds13", 1), ("r_qw05", 2)):
        Sc1 = StandardScaler().fit(Rm[:, [col]])
        res[f"ncm_{tag}"] = ncm_loo(Sc1.transform(Rm[:, [col]]), z)
    Scm = StandardScaler().fit(Rm)
    res["ncm_R3"] = ncm_loo(Scm.transform(Rm), z)

    (OUT / "summary.json").write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print("[e6] 结果：", json.dumps(res, ensure_ascii=False), flush=True)
    print("[e6] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
