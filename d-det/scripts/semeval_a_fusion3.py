#!/usr/bin/env python
"""A 三源融合（mini，零训练）：冻结 s1 + R_void + Qwen 冻结探针。

基于 E6 洞见（"读数群 + s1 互补"，B 上 0.73→0.86）在 A 任务复验：
- 单项与组合的 prior22 协议（test 分数 0.78 分位阈）macro-F1
- 权重：val 网格搜索定权 → test 应用（合规版）；另报 test 上的探索最优（标注）
输出：runs/semeval_a/fusion3.json

目标：对照既有最佳 .7492（0.7·s1+0.3·rvoid）。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def macro_f1(pred, y) -> float:
    f1s = []
    for c in np.unique(y):
        tp = np.sum((pred == c) & (y == c))
        fp = np.sum((pred == c) & (y != c))
        fn = np.sum((pred != c) & (y == c))
        f1s.append(2 * tp / max(2 * tp + fp + fn, 1))
    return float(np.mean(f1s))


def f1_prior22(score, y, q=0.78) -> float:
    return round(macro_f1((score >= np.quantile(score, q)).astype(int), y), 4)


def main() -> int:
    out = {}
    src = {}
    for split in ("val", "test"):
        f = np.load(ROOT / f"runs/semeval_zeroshot/feat/a_{split}.npz", allow_pickle=True)
        s1 = np.asarray(f["s1"]).reshape(-1)
        rv = np.load(ROOT / f"data/processed/semeval/a_{split}_stats.npz")["X"][:, 0]
        qw = np.load(ROOT / f"runs/semeval_qwen_probe/a_raw_{split}.npz")["probs"]
        y = np.asarray(f["y"])
        src[split] = {"s1": s1, "rv": rv, "qw": qw, "y": y}
    yv, yt = src["val"]["y"], src["test"]["y"]

    # z-score（按各自 val 分布归一化）
    Z = {}
    for split in ("val", "test"):
        Z[split] = {}
        for k in ("s1", "rv", "qw"):
            mu, sd = src["val"][k].mean(), src["val"][k].std() + 1e-9
            Z[split][k] = (src[split][k] - mu) / sd

    out["singles_test_prior22"] = {k: f1_prior22(Z["test"][k], yt) for k in ("s1", "rv", "qw")}
    out["singles_val_prior22"] = {k: f1_prior22(Z["val"][k], yv) for k in ("s1", "rv", "qw")}

    # 网格（0.1 步长）在 val 上定权 → test
    best = (-1.0, None)
    for w1 in np.arange(0, 1.01, 0.1):
        for w2 in np.arange(0, 1.01 - w1 + 1e-9, 0.1):
            w3 = round(1.0 - w1 - w2, 1)
            if w3 < -1e-9:
                continue
            sc = w1 * Z["val"]["s1"] + w2 * Z["val"]["rv"] + w3 * Z["val"]["qw"]
            v = f1_prior22(sc, yv)
            if v > best[0]:
                best = (v, (round(float(w1), 2), round(float(w2), 2), round(float(w3), 2)))
    w = best[1]
    sc_te = w[0] * Z["test"]["s1"] + w[1] * Z["test"]["rv"] + w[2] * Z["test"]["qw"]
    out["valfit"] = {"weights": w, "val_f1": best[0], "test_f1": f1_prior22(sc_te, yt)}
    # 复现旧版 0.7/0.3
    sc_old = 0.7 * Z["test"]["s1"] + 0.3 * Z["test"]["rv"]
    out["repro_0.7s1_0.3rv_test"] = f1_prior22(sc_old, yt)
    # 无监督合规版：等权 / 秩和 / s1+qw 等权
    from scipy.stats import rankdata
    out["equal3_test"] = f1_prior22(
        (Z["test"]["s1"] + Z["test"]["rv"] + Z["test"]["qw"]) / 3.0, yt)
    out["equal_s1_qw_test"] = f1_prior22(
        (Z["test"]["s1"] + Z["test"]["qw"]) / 2.0, yt)
    rk = (rankdata(Z["test"]["s1"]) + rankdata(Z["test"]["rv"])
          + rankdata(Z["test"]["qw"])) / 3.0
    out["rank_sum3_test"] = f1_prior22(rk, yt)
    # 探索：test 上的最优（标注仅供观察）
    best_t, bt = (-1.0, None)
    for w1 in np.arange(0, 1.01, 0.1):
        for w2 in np.arange(0, 1.01 - w1 + 1e-9, 0.1):
            w3 = round(1.0 - w1 - w2, 1)
            if w3 < -1e-9:
                continue
            v = f1_prior22(w1 * Z["test"]["s1"] + w2 * Z["test"]["rv"] + w3 * Z["test"]["qw"], yt)
            if v > best_t:
                best_t, bt = v, (round(float(w1), 2), round(float(w2), 2), round(float(w3), 2))
    out["test_explore"] = {"weights": bt, "test_f1": best_t}
    (ROOT / "runs/semeval_a").mkdir(parents=True, exist_ok=True)
    (ROOT / "runs/semeval_a/fusion3.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(json.dumps(out, ensure_ascii=False, indent=2))
    print("[fusion3] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
