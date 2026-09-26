#!/usr/bin/env python
"""R_void（空行率）复现与单特征基线（对齐 UIT_AMMC 的空白对齐工件假设）。

纯 CPU。检验：
  1) A：train（前 10 万行）与 test 的 Human/AI 空行率均值、Cohen's d、Welch p；
     UT 报告：train 11.7% vs 15.4%（d=0.32）；test 7.25% vs 16.50%（d=0.97）。
  2) 以 train 两类均值中点作阈值 → test macro-F1（单特征基线，对照 UT 的 0.679）。
  3) C：test 四类空行率均值（Human/AI/Hybrid/Adversarial）。
输出：runs/semeval_rvoid/rvoid.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy import stats
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/SemEval-2026-Task13"
OUT = ROOT / "runs/semeval_rvoid"


def rvoid(code: str) -> float:
    lines = code.split("\n")
    if not lines:
        return 0.0
    return sum(1 for ln in lines if not ln.strip()) / len(lines)


def load(path: Path, n: int | None = None):
    t = pq.read_table(path, columns=["code", "label"])
    codes = t.column("code").to_pylist()
    labels = np.asarray(t.column("label").to_pylist())
    total = len(codes)
    if n is not None:
        codes, labels = codes[:n], labels[:n]
    vals = np.array([rvoid(c) for c in codes])
    return vals, labels, total


def desc(a, b):
    d = (a.mean() - b.mean()) / np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    p = float(stats.ttest_ind(a, b, equal_var=False).pvalue)
    return float(d), p


def main() -> int:
    res = {}

    # ---------- A ----------
    v_tr, y_tr, n_tr_raw = load(RAW / "task_a/task_a_training_set_1.parquet", 100_000)
    v_te, y_te, n_te_raw = load(RAW / "task_a/task_a_test_set_sample.parquet")
    h_tr, a_tr = v_tr[y_tr == 0], v_tr[y_tr == 1]
    h_te, a_te = v_te[y_te == 0], v_te[y_te == 1]
    d_tr, p_tr = desc(a_tr, h_tr)
    d_te, p_te = desc(a_te, h_te)
    thr = float((h_tr.mean() + a_tr.mean()) / 2)
    pred = (v_te >= thr).astype(int)
    f1 = float(f1_score(y_te, pred, average="macro"))
    res["a"] = {
        "n_test_raw": n_te_raw, "n_test": len(y_te),
        "train(100k)": {"human": round(float(h_tr.mean()), 4), "ai": round(float(a_tr.mean()), 4),
                        "d": round(d_tr, 3), "p": p_tr},
        "test": {"human": round(float(h_te.mean()), 4), "ai": round(float(a_te.mean()), 4),
                 "d": round(d_te, 3), "p": p_te},
        "rvoid_only_macro_f1(test, thr_from_train)": round(f1, 4), "thr": round(thr, 4),
    }
    print("== A ==", json.dumps(res["a"], ensure_ascii=False))

    # ---------- C ----------
    v_trc, y_trc, _ = load(RAW / "task_c/task_c_training_set_1.parquet", 200_000)
    v_tec, y_tec, n_tec_raw = load(RAW / "task_c/task_c_test_set_sample.parquet")
    names = {0: "Human", 1: "AI", 2: "Hybrid", 3: "Adversarial"}
    means_tr = {names[int(c)]: round(float(v_trc[y_trc == c].mean()), 4) for c in sorted(set(y_trc.tolist()))}
    means_te = {names[int(c)]: round(float(v_tec[y_tec == c].mean()), 4) for c in sorted(set(y_tec.tolist()))}
    res["c"] = {"n_test_raw": n_tec_raw, "train(200k)_mean": means_tr, "test_mean": means_te}
    print("== C ==", json.dumps(res["c"], ensure_ascii=False))

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "rvoid.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"[out] {OUT / 'rvoid.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
