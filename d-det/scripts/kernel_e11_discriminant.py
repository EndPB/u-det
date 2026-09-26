#!/usr/bin/env python
"""E11（mini）：残差判别读出——s2-v2 的第一个正确定义候选验证。

命题（E10 修正版）：族差异藏在"垂直于公共位移方向的残差"里；
正确的读出 = 对配对残差做判别学习（而非对标量差做 margin）。
- 数据：394 同题配对 × 2 族（Qwen1.5B/DS1.3B）× CodeT5 v1.0 特征（本脚本提取并缓存）
- 公共轴 a = 全体 Δ 均值方向；残差 Δ̃ = Δ − (Δ·a)a
- 全部分析用 GroupKFold（同题 q/d 同折——无同题泄漏）：
  1) 对照：全 Δ 上的 LDA(shrinkage)/LR
  2) 主检验：残差 Δ̃ 上的 LDA/LR + KMeans 纯度 + 公共轴投影 1D
输出：runs/kernel_e11/discriminant.json + feat_cache.npz
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import yaml
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from encoders import build_encoder  # noqa: E402
from models import build_model  # noqa: E402
from kernel_e2_loss import encode_codes  # noqa: E402

OUT = ROOT / "runs/kernel_e11"


def purity(clusters, labels) -> float:
    tot = 0
    for c in set(clusters.tolist()):
        idx = clusters == c
        if idx.sum() == 0:
            continue
        _, cnts = np.unique(labels[idx], return_counts=True)
        tot += cnts.max()
    return float(tot / len(labels))


def gkf_acc(X, y, groups, model_fn):
    accs = []
    for tr, va in GroupKFold(n_splits=5).split(X, y, groups):
        m = model_fn()
        m.fit(X[tr], y[tr])
        accs.append(accuracy_score(y[va], m.predict(X[va])))
    return round(float(np.mean(accs)), 4)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cache = OUT / "feat_cache.npz"
    if cache.exists():
        d = np.load(cache)
        dq, dd = d["dq"], d["dd"]
    else:
        tq = pq.read_table(ROOT / "data/processed/pairs_qwen15.parquet",
                           columns=["task_id", "x_plus", "x_minus"])
        td = pq.read_table(ROOT / "data/processed/pairs_ds13.parquet",
                           columns=["task_id", "x_plus", "x_minus"])
        mq = dict(zip(tq.column("task_id").to_pylist(),
                      zip(tq.column("x_plus").to_pylist(), tq.column("x_minus").to_pylist())))
        md = dict(zip(td.column("task_id").to_pylist(),
                      zip(td.column("x_plus").to_pylist(), td.column("x_minus").to_pylist())))
        common = sorted(set(mq) & set(md))
        with open(ROOT / "configs/ddet_v041.yaml", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        enc_cfg = dict(cfg["encoder"])
        enc_cfg["path"] = str(ROOT / enc_cfg["path"])
        enc = build_encoder(**enc_cfg)
        dual = build_model("dual", encoder=enc, dim=enc.hidden_size,
                           pool=cfg["model"].get("pool", "mean"),
                           s2_rank=cfg["model"].get("s2_rank", 1))
        ck = torch.load(str(ROOT / "runs/v0.4.1_covreg/last.pt"), map_location="cpu",
                        weights_only=False)
        dual.load_state_dict(ck["state"], strict=False)
        dual = dual.to(device).eval()
        tok = AutoTokenizer.from_pretrained(enc_cfg["path"])
        H = {}
        for tag, src, key in (("qp", mq, 0), ("qm", mq, 1), ("dp", md, 0), ("dm", md, 1)):
            H[tag] = encode_codes(dual, [src[t][key] for t in common], tok, device).numpy()
            print(f"[e11] {tag} {H[tag].shape}", flush=True)
        dq = H["qp"] - H["qm"]
        dd = H["dp"] - H["dm"]
        np.savez_compressed(cache, dq=dq, dd=dd, tasks=np.array(common, dtype=object))

    n = len(dq)
    X = np.vstack([dq, dd])
    y = np.array([0] * n + [1] * n)
    groups = np.concatenate([np.arange(n), np.arange(n)])  # 同题同组
    res = {"n_pairs": n}

    # 公共轴与残差
    a = X.mean(0)
    a = a / (np.linalg.norm(a) + 1e-9)
    proj = X @ a
    Xr = X - np.outer(proj, a)

    Sc = StandardScaler().fit(X)
    Sca = StandardScaler().fit(Xr)

    # ---- 对照：全 Δ ----
    res["full_LDA_acc"] = gkf_acc(Sc.transform(X), y, groups,
                                  lambda: LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))
    res["full_LR_acc"] = gkf_acc(Sc.transform(X), y, groups,
                                 lambda: LogisticRegression(C=1.0, max_iter=2000))
    # ---- 主检验：残差 ----
    res["resid_LDA_acc"] = gkf_acc(Sca.transform(Xr), y, groups,
                                   lambda: LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))
    res["resid_LR_acc"] = gkf_acc(Sca.transform(Xr), y, groups,
                                  lambda: LogisticRegression(C=1.0, max_iter=2000))
    # 无监督：残差 KMeans 纯度 vs 公共轴 1D acc
    Xrs = Sca.transform(Xr)
    km = KMeans(n_clusters=2, n_init=20, random_state=0).fit(Xrs)
    res["resid_kmeans_purity"] = round(purity(km.labels_, y), 4)
    thr = np.median(proj)
    res["axis_1d_acc"] = round(float(accuracy_score(y, (proj >= thr).astype(int))), 4)

    (OUT / "discriminant.json").write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print(json.dumps(res, ensure_ascii=False, indent=2))
    print("[e11] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
