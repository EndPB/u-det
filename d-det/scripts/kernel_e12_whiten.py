#!/usr/bin/env python
"""E12（mini）：类内白化——"收内+散间"的几何验证（"聚类复活"检验）。

用 E11 缓存 Δ（394×2 族，Qwen1.5B/DS1.3B）：
1) 对照：原始空间 KMeans 纯度（E11 = 0.55）
2) LDA 方向 1D KMeans 纯度（判别投影后的聚类）
3) Σ_W 白化（LedoitWolf）→ KMeans 纯度
4) PCA(100)-白化 → KMeans 纯度
并报：白化前后 类内迹 与 族间距离（马氏比）
输出：runs/kernel_e12/whiten.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/kernel_e12"


def purity(clusters, labels) -> float:
    tot = 0
    for c in set(clusters.tolist()):
        idx = clusters == c
        if idx.sum() == 0:
            continue
        _, cnts = np.unique(labels[idx], return_counts=True)
        tot += cnts.max()
    return float(tot / len(labels))


def km2(X, seed=0) -> np.ndarray:
    return KMeans(n_clusters=2, n_init=20, random_state=seed).fit(X).labels_


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    d = np.load(ROOT / "runs/kernel_e11/feat_cache.npz")
    dq, dd = d["dq"], d["dd"]
    n = len(dq)
    X = np.vstack([dq, dd])
    y = np.array([0] * n + [1] * n)
    res = {"n_pairs": n}

    # ---- 对照 1：原始 ----
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Xz = (X - mu) / sd
    res["kmeans_raw_purity"] = round(purity(km2(Xz), y), 4)

    # ---- 对照 2：LDA 1D ----
    lda = LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto").fit(X, y)
    t1 = lda.transform(X)[:, 0]
    # 1D KMeans（等价位阈值）
    res["kmeans_lda1d_purity"] = round(purity(km2(t1.reshape(-1, 1)), y), 4)

    # ---- 主检验 3：Σ_W 白化 ----
    lw = LedoitWolf().fit(X - X.mean(0))
    Sig = (lw.covariance_ + lw.covariance_.T) / 2
    evals, evecs = np.linalg.eigh(Sig)
    evals = np.clip(evals, 1e-6 * evals.max(), None)
    Winv = evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.T
    Xw = (X - X.mean(0)) @ Winv
    res["kmeans_sigmaW_purity"] = round(purity(km2(Xw), y), 4)

    # ---- 主检验 4：PCA-白化（k=100）----
    pca = PCA(n_components=100, random_state=0).fit(Xz)
    T = pca.transform(Xz)
    T = T / (T.std(0) + 1e-9)
    res["kmeans_pcawhite_purity"] = round(purity(km2(T), y), 4)
    T2 = PCA(n_components=20, random_state=0).fit(Xz)
    T2 = T2.transform(Xz)
    T2 = T2 / (T2.std(0) + 1e-9)
    res["kmeans_pcawhite20_purity"] = round(purity(km2(T2), y), 4)

    # ---- 几何量：白化前后的 类内迹 / 族间距离 ----
    def within_trace(Z, centers):
        return float(np.mean([np.trace(np.cov(Z[y == c] - centers[c], rowvar=False))
                              for c in (0, 1)]))
    m0, m1 = X[y == 0].mean(0), X[y == 1].mean(0)
    res["raw_within_trace"] = round(within_trace(Xz, {0: Xz[y == 0].mean(0),
                                                      1: Xz[y == 1].mean(0)}), 2)
    res["raw_center_dist"] = round(float(np.linalg.norm(
        (Xz[y == 0].mean(0) - Xz[y == 1].mean(0)))), 3)
    res["white_within_trace"] = round(within_trace(Xw, {0: Xw[y == 0].mean(0),
                                                        1: Xw[y == 1].mean(0)}), 2)
    res["white_center_dist"] = round(float(np.linalg.norm(
        Xw[y == 0].mean(0) - Xw[y == 1].mean(0))), 3)

    (OUT / "whiten.json").write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print(json.dumps(res, ensure_ascii=False, indent=2))
    print("[e12] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
