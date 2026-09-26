#!/usr/bin/env python
"""E9（mini）：从"多族位移方向"直接构造 s2 子空间原型（零训练）。

动机（E2/E5/E8）：单方向代理学不到 r*；族位移方向部分共享（llama/yi 沿公共轴 0.84-0.86）部分私有（qwen 垂直）;
→ 用位移方向矩阵的 span 作为 s2 子空间，检验检测/归因/与 r* 对齐。
数据：B val 冻结 h（v1.0）；位移取自 3 个配对族（llama-3.1 / qwen2.5-1.5b / yi-coder-1.5b）。
输出：runs/kernel_e9/subspace.json
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/kernel_e9"


def kind(g: str) -> str:
    gl = g.lower()
    kw = ("instruct", "chat", "-it", "gpt", "v3-0324", "devstral", "thinking")
    return "instruct" if any(k in gl for k in kw) else "base"


def canon(g: str) -> str:
    gl = g.lower()
    for suf in ("-instruct", "-chat", "_instruct", "-it", "-v0.3"):
        if gl.endswith(suf):
            gl = gl[: -len(suf)]
    return gl


def loo_lr_auc(F, y):
    F = StandardScaler().fit_transform(F)
    sc = np.zeros(len(y))
    for i in range(len(y)):
        m = np.ones(len(y), bool)
        m[i] = False
        clf = LogisticRegression(C=1.0, max_iter=1000)
        clf.fit(F[m], y[m])
        sc[i] = clf.predict_proba(F[i:i + 1])[0, 1]
    return round(float(roc_auc_score(y, sc)), 4)


def ncm_loo(X, z):
    X = np.asarray(X, dtype="float64")
    z = np.asarray(z)
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
        d = [np.linalg.norm(X[i] - c) for c in cents]
        pred[i] = labs[int(np.argmin(d))]
    accs = [np.mean(pred[z == c] == c) for c in set(z.tolist())]
    return round(float(np.mean(accs)), 4)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    f = np.load(ROOT / "runs/semeval_zeroshot/feat/b_val.npz", allow_pickle=True)
    H = np.asarray(f["h"], dtype="float64")
    s1 = np.asarray(f["s1"]).reshape(-1)
    s2 = np.asarray(f["s2"]).reshape(-1)
    t = pq.read_table(ROOT / "data/processed/semeval/b_val.parquet",
                      columns=["label", "generator"])
    lab = np.asarray(t.column("label").to_pylist())
    gen = [str(g) for g in t.column("generator").to_pylist()]

    fam_b, fam_i = defaultdict(list), defaultdict(list)
    is_i = np.array([kind(g) == "instruct" for g in gen]) & (lab > 0)
    for i in range(len(H)):
        if lab[i] == 0:
            continue
        (fam_i if is_i[i] else fam_b)[canon(gen[i])].append(i)
    fams = [g for g in fam_b if g in fam_i and len(fam_b[g]) >= 3 and len(fam_i[g]) >= 3]
    D = []
    for g in fams:
        d = H[fam_i[g]].mean(0) - H[fam_b[g]].mean(0)
        D.append(d / (np.linalg.norm(d) + 1e-12))
    D = np.asarray(D)                     # (k, 768)
    U, S, Vt = np.linalg.svd(D, full_matrices=False)
    res = {"n_families": len(fams), "family_list": fams,
           "singular_values": [round(float(x), 4) for x in S]}

    # 投影（顶部 2 与 3 维）
    for r in (2, 3):
        if r > Vt.shape[0]:
            continue
        P = H @ Vt[:r].T                  # (N, r)
        y = (lab > 0).astype(int)
        res[f"auc_detect_r{r}"] = loo_lr_auc(P, y)
        m = lab > 0
        z = np.array([canon(g) for g in gen])[m]
        res[f"ncm_attrib_r{r}"] = ncm_loo(P[m], z)
    # 对照
    res["auc_s1"] = round(float(roc_auc_score((lab > 0).astype(int), s1)), 4)
    res["auc_s2_v1"] = round(float(roc_auc_score((lab > 0).astype(int), s2)), 4)
    # 与 s1 的互补性：每维相关 + 联合检测
    y = (lab > 0).astype(int)
    if Vt.shape[0] >= 2:
        P3 = H @ Vt[:min(3, Vt.shape[0])].T
        res["corr_proj_s1"] = [round(float(np.corrcoef(P3[:, j], s1)[0, 1]), 4)
                               for j in range(P3.shape[1])]
        res["auc_proj3_plus_s1"] = loo_lr_auc(np.hstack([P3, s1.reshape(-1, 1)]), y)
        for j in range(P3.shape[1]):
            res[f"auc_proj{j}"] = round(float(roc_auc_score(y, P3[:, j])), 4)
        if P3.shape[1] >= 3:
            res["auc_proj12_joint"] = loo_lr_auc(P3[:, 1:3], y)
        m = lab > 0
        zz = np.array([canon(g) for g in gen])[m]
        for j in range(P3.shape[1]):
            res[f"ncm_proj{j}"] = ncm_loo(P3[m][:, [j]], zz)
    m = lab > 0
    res["ncm_s2_scalar"] = ncm_loo(s2[m].reshape(-1, 1), np.array([canon(g) for g in gen])[m])
    # 与 r* 对齐（200 子集）
    e1 = ROOT / "runs/kernel_e1/rstar_total.npz"
    if e1.exists():
        d = np.load(e1, allow_pickle=True)
        P3 = H @ Vt[:min(3, Vt.shape[0])].T
        for j in range(P3.shape[1]):
            res[f"corr_proj{j}_rstar"] = round(float(np.corrcoef(P3[d["row"], j], d["r"])[0, 1]), 4)

    (OUT / "subspace.json").write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print(json.dumps(res, ensure_ascii=False, indent=2))
    print("[e9] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
