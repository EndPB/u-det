#!/usr/bin/env python
"""E10（mini）：家族分类 = Δ 空间聚类？（用户命题的直接验证）

数据：pairs_qwen15 ∩ pairs_ds13 同题配对（394 对）× CodeT5 v1.0 冻结特征（本脚本提取）。
检验：
1) Δ=h(x+)−h(x−) 的无监督聚类（2-means / GMM；原始 / 方向归一化）→ 纯度 vs 真族
2) 有监督上限（LR 5-fold）——P1 的 0.98 复现
3) Δs2 单标量对照（v1.0 线性读出）——"单标量与方向的差距"
4) 族内 Δ 方向一致性 vs 跨族（cos）
输出：runs/kernel_e10/cluster.json
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from encoders import build_encoder  # noqa: E402
from models import build_model  # noqa: E402
from kernel_e2_loss import encode_codes  # noqa: E402

OUT = ROOT / "runs/kernel_e10"


def purity(clusters, labels) -> float:
    tot = 0
    for c in set(clusters.tolist()):
        idx = clusters == c
        if idx.sum() == 0:
            continue
        vals, cnts = np.unique(labels[idx], return_counts=True)
        tot += cnts.max()
    return float(tot / len(labels))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- 数据：同题配对对齐 ----
    tq = pq.read_table(ROOT / "data/processed/pairs_qwen15.parquet",
                       columns=["task_id", "x_plus", "x_minus"])
    td = pq.read_table(ROOT / "data/processed/pairs_ds13.parquet",
                       columns=["task_id", "x_plus", "x_minus"])
    mq = {tid: (p, m) for tid, p, m in zip(tq.column("task_id").to_pylist(),
                                           tq.column("x_plus").to_pylist(),
                                           tq.column("x_minus").to_pylist())}
    md = {tid: (p, m) for tid, p, m in zip(td.column("task_id").to_pylist(),
                                           td.column("x_plus").to_pylist(),
                                           td.column("x_minus").to_pylist())}
    common = sorted(set(mq) & set(md))
    print(f"[e10] 同题配对: {len(common)}", flush=True)

    # ---- 特征（CodeT5 v1.0）----
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
    w2 = ck["state"]["w2.weight"].float().numpy().reshape(-1)

    H = {}
    for tag, src, key in (("qp", mq, 0), ("qm", mq, 1), ("dp", md, 0), ("dm", md, 1)):
        codes = [src[t][key] for t in common]
        H[tag] = encode_codes(dual, codes, tok, device).numpy()
        print(f"[e10] {tag} 特征 {H[tag].shape}", flush=True)

    dq = H["qp"] - H["qm"]
    dd = H["dp"] - H["dm"]
    X = np.vstack([dq, dd])
    labs = np.array([0] * len(dq) + [1] * len(dd))
    res = {"n_pairs_per_family": len(common)}

    # ---- 1) 无监督聚类 ----
    Sc = StandardScaler().fit(X)
    Xs = Sc.transform(X)
    km = KMeans(n_clusters=2, n_init=20, random_state=0).fit(Xs)
    res["kmeans_purity_raw"] = round(purity(km.labels_, labs), 4)
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    km2 = KMeans(n_clusters=2, n_init=20, random_state=0).fit(Xn)
    res["kmeans_purity_dir"] = round(purity(km2.labels_, labs), 4)
    gm = GaussianMixture(n_components=2, covariance_type="diag",
                         random_state=0).fit(Xs)
    res["gmm_purity_raw"] = round(purity(gm.predict(Xs), labs), 4)

    # ---- 2) 有监督上限 ----
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    accs = []
    for tr, va in skf.split(Xs, labs):
        clf = LogisticRegression(C=1.0, max_iter=2000)
        clf.fit(Xs[tr], labs[tr])
        accs.append(accuracy_score(labs[va], clf.predict(Xs[va])))
    res["sup_lr_acc5fold"] = round(float(np.mean(accs)), 4)

    # ---- 3) Δs2 单标量对照 ----
    s2 = lambda hh: hh @ w2
    ds2 = np.concatenate([s2(dq), s2(dd)])
    res["delta_s2_auc"] = round(float(roc_auc_score(labs, ds2)), 4)
    thr = np.median(ds2)
    pred1d = (ds2 >= thr).astype(int)
    res["delta_s2_thresh_acc"] = round(float(accuracy_score(labs, pred1d)), 4)

    # ---- 4) 方向一致性 ----
    def mean_cos(D):
        Dn = D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-9)
        idx = np.random.RandomState(0).choice(len(Dn), min(200, len(Dn)), replace=False)
        S = Dn[idx] @ Dn[idx].T
        iu = np.triu_indices(len(idx), k=1)
        return float(np.mean(S[iu]))
    res["within_qwen_cos"] = round(mean_cos(dq), 4)
    res["within_ds_cos"] = round(mean_cos(dd), 4)
    mq_dir = dq.mean(0) / (np.linalg.norm(dq.mean(0)) + 1e-9)
    md_dir = dd.mean(0) / (np.linalg.norm(dd.mean(0)) + 1e-9)
    res["cross_family_cos"] = round(float(mq_dir @ md_dir), 4)

    (OUT / "cluster.json").write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print(json.dumps(res, ensure_ascii=False, indent=2))
    print("[e10] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
