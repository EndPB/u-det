#!/usr/bin/env python
"""P1 探针（补跑）：跨族 Δ 残差与 s2 的家族可分性。

背景：d-det.md 原设计 = s2 承担家族归因（"带符号方向 + 低秩 r=4~8"）；v0.4 的 xfam
损失把跨族 Δ 方向余弦拉到 ~0.998（为检测跨族稳健性服务）。本探针量化：
  1) 族内 plus-vs-minus 的 s2 AUC（sanity，应 ~1）；
  2) 族间 s2/s1 AUC（同题、同 split：qwen15 vs ds13 的 instruct/base 输出）；
  3) Δh 方向余弦（族内 vs 跨族配对）；去公共方向后的残差线性可分性（5 折 CV AUC）。
对比 v0.3.0（无 xfam）与 v0.4.1（xfam / v1.0）→ 量化 xfam 对家族信息的压缩代价。
输出：runs/probe_s2_family/s2family.json
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from encoders import build_encoder  # noqa: E402
from models import build_model  # noqa: E402

OUT = ROOT / "runs/probe_s2_family"


def read_pairs(fname: str) -> dict:
    t = pq.read_table(ROOT / f"data/processed/{fname}")
    d = {}
    for i in range(t.num_rows):
        d[t.column("task_id")[i].as_py()] = {
            "plus": t.column("input_ids_plus")[i].as_py(),
            "minus": t.column("input_ids_minus")[i].as_py(),
            "split": t.column("split")[i].as_py(),
            "family": t.column("family")[i].as_py(),
        }
    return d


@torch.no_grad()
def encode(model, ids_list: list, device: str, batch: int = 16):
    s1s, s2s, hs = [], [], []
    for i in range(0, len(ids_list), batch):
        chunk = ids_list[i:i + batch]
        L = max(len(x) for x in chunk)
        ids = torch.zeros(len(chunk), L, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for r, x in enumerate(chunk):
            ids[r, :len(x)] = torch.tensor(x, dtype=torch.long)
            mask[r, :len(x)] = 1
        ids, mask = ids.to(device), mask.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            s1, s2, h = model.forward_feat(ids, mask)
        s1s.append(s1.float().cpu())
        s2s.append(s2.float().cpu().reshape(-1))
        hs.append(h.float().cpu())
    return (torch.cat(s1s).numpy(), torch.cat(s2s).numpy(), torch.cat(hs).numpy())


def cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-9)


def mean_pairwise_cos(x: np.ndarray) -> float:
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-9
    xn = x / n
    c = xn @ xn.T
    return float((c.sum() - np.trace(c)) / (len(x) * (len(x) - 1)))


def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    p15 = read_pairs("pairs_qwen15.parquet")
    pd_ = read_pairs("pairs_ds13.parquet")
    common = sorted(set(p15) & set(pd_))
    print(f"[join] qwen15={len(p15)} ds13={len(pd_)} common={len(common)} "
          f"splits={dict(Counter(p15[t]['split'] for t in common))}", flush=True)
    N = len(common)
    q_plus = [p15[t]["plus"] for t in common]
    q_minus = [p15[t]["minus"] for t in common]
    d_plus = [pd_[t]["plus"] for t in common]
    d_minus = [pd_[t]["minus"] for t in common]
    print(f"[len] max ids: q+ {max(len(x) for x in q_plus)}, "
          f"q- {max(len(x) for x in q_minus)}", flush=True)

    configs = [
        ("v0.3.0", "configs/ddet_v030.yaml", "runs/v0.3.0_fullft/last.pt"),
        ("v0.4.1(v1.0)", "configs/ddet_v041.yaml", "runs/v0.4.1_covreg/last.pt"),
    ]
    results = {}
    for name, cfg_p, ckpt_p in configs:
        with open(ROOT / cfg_p, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        enc_cfg = dict(cfg["encoder"])
        enc_cfg["path"] = str(ROOT / enc_cfg["path"])
        enc = build_encoder(**enc_cfg)
        dual = build_model("dual", encoder=enc, dim=enc.hidden_size,
                           pool=cfg["model"].get("pool", "mean"),
                           s2_rank=cfg["model"].get("s2_rank", 1))
        ck = torch.load(str(ROOT / ckpt_p), map_location="cpu", weights_only=False)
        st = ck.get("state", ck)
        miss, unexp = dual.load_state_dict(st, strict=False)
        print(f"[{name}] {ckpt_p}（epoch {ck.get('epoch')}）missing={len(miss)} unexpected={len(unexp)}", flush=True)
        dual = dual.to(device).eval()

        s1q_p, s2q_p, hq_p = encode(dual, q_plus, device)
        s1q_m, s2q_m, hq_m = encode(dual, q_minus, device)
        s1d_p, s2d_p, hd_p = encode(dual, d_plus, device)
        s1d_m, s2d_m, hd_m = encode(dual, d_minus, device)

        y_qm = [1] * N + [0] * N
        y_fam = [1] * N + [0] * N
        res = {
            "s2_auc_within_qwen15(plus>minus)": round(float(roc_auc_score(y_qm, np.r_[s2q_p, s2q_m])), 4),
            "s2_auc_within_ds13(plus>minus)": round(float(roc_auc_score(y_qm, np.r_[s2d_p, s2d_m])), 4),
            "s2_auc_family(plus)": round(float(roc_auc_score(y_fam, np.r_[s2q_p, s2d_p])), 4),
            "s2_auc_family(minus)": round(float(roc_auc_score(y_fam, np.r_[s2q_m, s2d_m])), 4),
            "s1_auc_family(plus)": round(float(roc_auc_score(y_fam, np.r_[s1q_p, s1d_p])), 4),
        }

        dh_q, dh_d = hq_p - hq_m, hd_p - hd_m
        res["dh_cos_paired_cross_family"] = round(float(cos(dh_q, dh_d).mean()), 4)
        res["dh_cos_within_qwen15"] = round(mean_pairwise_cos(dh_q), 4)
        res["dh_cos_within_ds13"] = round(mean_pairwise_cos(dh_d), 4)

        # 公共方向 + 残差
        A = np.vstack([dh_q, dh_d])
        u = A.mean(0)
        u = u / (np.linalg.norm(u) + 1e-9)
        r_q = dh_q - np.outer(dh_q @ u, u)
        r_d = dh_d - np.outer(dh_d @ u, u)
        X = np.vstack([r_q, r_d])
        y = np.r_[np.zeros(N), np.ones(N)]
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        cv = cross_val_score(clf, X, y, cv=5, scoring="roc_auc")
        res["residual_family_auc_cv5_kfold"] = round(float(cv.mean()), 4)
        groups = np.r_[np.arange(N), np.arange(N)]          # 同一题的两行同折（防泄漏）
        cv_g = cross_val_score(clf, X, y, cv=GroupKFold(n_splits=5), groups=groups,
                               scoring="roc_auc")
        res["residual_family_auc_cv5_group"] = round(float(cv_g.mean()), 4)
        res["residual_norm_ratio"] = round(float(np.linalg.norm(X, axis=1).mean() /
                                                 np.linalg.norm(A, axis=1).mean()), 4)
        # 纯方向对照：残差单位化 / 全 Δ 单位化（区分"方向信息"与"幅度差"）
        rn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
        cv_rn = cross_val_score(clf, rn, y, cv=GroupKFold(n_splits=5), groups=groups,
                                scoring="roc_auc")
        res["residual_dir_auc_group"] = round(float(cv_rn.mean()), 4)
        A_unit = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
        cv_au = cross_val_score(clf, A_unit, y, cv=GroupKFold(n_splits=5), groups=groups,
                                scoring="roc_auc")
        res["delta_dir_auc_group"] = round(float(cv_au.mean()), 4)

        results[name] = res
        print(f"[{name}] {json.dumps(res, ensure_ascii=False)}", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "s2family.json", "w", encoding="utf-8") as f:
        json.dump({"n_pairs": N, "results": results}, f, ensure_ascii=False, indent=2)
    print(f"[out] {OUT / 's2family.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
