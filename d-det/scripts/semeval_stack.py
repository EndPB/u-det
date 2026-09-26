#!/usr/bin/env python
"""三任务特征堆叠（LightGBM）：手工特征 + 骨干输出（直连头 probs、s1/s2、rank-8 子空间）→ macro-F1。

变体/消融（fail-soft，缺块自动跳过）：
  A：train-fit {s1, stats(含 rvoid), z8, sub_probs}；先验匹配 top-22% 与 val-阈值两版；no-z 消融。
  B：train-fit {z8, sub_probs, z8_ft, probs_ft, s1, s2, stats}；no-zft / no-z 消融；
     val-fit（+直连头 probs）→ test（val 内 5 折 CV 估计）。
  C：同 B（4 类）。
输出：runs/semeval_stack/results.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
FEAT = ROOT / "runs/semeval_zeroshot/feat"
ZDIR = ROOT / "runs/semeval_r/z8"
PDIR = ROOT / "runs/semeval_r"
SDIR = ROOT / "data/processed/semeval"
RES = ROOT / "runs/semeval_stack"

import lightgbm as lgb  # noqa: E402


def labels(task: str, split: str) -> np.ndarray:
    return np.asarray(pq.read_table(SDIR / f"{task}_{split}.parquet",
                                    columns=["label"]).column("label").to_pylist())


def stats(task: str, split: str):
    p = SDIR / f"{task}_{split}_stats.npz"
    return np.load(p)["X"].astype("float32") if p.exists() else None


def base(task: str, split: str):
    p = FEAT / f"{task}_{split}.npz"
    if not p.exists():
        return None, None
    d = np.load(p, allow_pickle=True)
    return (d["s1"].reshape(-1, 1).astype("float32"),
            d["s2"].reshape(len(d["y"]), -1).astype("float32"))


def z8(task: str, split: str, key: str = "z8"):
    p = ZDIR / f"{task}_{split}.npz"
    if not p.exists():
        return None
    d = np.load(p, allow_pickle=True)
    return d[key].astype("float32") if key in d.files else None


def dprobs(task: str, split: str):
    p = PDIR / f"probs_{task}_{split}.npz"
    return np.load(p)["probs"].astype("float32") if p.exists() else None


def build(parts) -> np.ndarray | None:
    blocks = [np.asarray(p, dtype="float32") for p in parts if p is not None]
    return np.hstack(blocks) if blocks else None


def lgb_fit(X, y, seed=0):
    clf = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31,
                             min_child_samples=20, subsample=0.9, colsample_bytree=0.9,
                             class_weight="balanced", random_state=seed, verbose=-1)
    clf.fit(X, y)
    return clf


def f1(y, pred) -> float:
    return round(float(f1_score(y, pred, average="macro")), 4)


def assemble(task: str, split: str, with_z=True, with_zft=True, with_stats=True, with_probs=False):
    s1, s2 = base(task, split)
    parts = [s1, s2]
    if with_stats:
        parts.append(stats(task, split))
    if with_z:
        parts += [z8(task, split), z8(task, split, "probs")]
    if with_zft:
        parts += [z8(task, split, "z8_ft"), z8(task, split, "probs_ft")]
    if with_probs:
        parts.append(dprobs(task, split))
    return build(parts)


def run_a(results):
    ytr, yva, yte = (labels("a", s) for s in ("train", "val", "test"))
    out = {}
    for tag, wz in (("full", True), ("no-z", False)):
        Xtr = assemble("a", "train", with_z=wz, with_zft=False, with_probs=False)
        Xva = assemble("a", "val", with_z=wz, with_zft=False, with_probs=False)
        Xte = assemble("a", "test", with_z=wz, with_zft=False, with_probs=False)
        if Xtr is None:
            continue
        clf = lgb_fit(Xtr, ytr)
        sc_va = clf.predict_proba(Xva)[:, 1]
        sc_te = clf.predict_proba(Xte)[:, 1]
        thr_prior = float(np.quantile(sc_te, 0.78))
        best_t, best_f1 = 0.5, -1.0
        for t in np.arange(0.05, 0.96, 0.01):
            v = f1(yva, (sc_va >= t).astype(int))
            if v > best_f1:
                best_f1, best_t = v, float(t)
        out[tag] = {"prior22_test": f1(yte, (sc_te >= thr_prior).astype(int)),
                    "val_thr": round(best_t, 2), "val_f1": best_f1,
                    "val_thr_test": f1(yte, (sc_te >= best_t).astype(int))}
        print(f"[stack-a] {tag}: {out[tag]}", flush=True)
    results["a"] = out


def run_bc(task, results):
    ytr, yva, yte = (labels(task, s) for s in ("train", "val", "test"))
    out = {}
    for tag, wz, wzft in (("full", True, True), ("no-zft", True, False), ("no-z", False, False)):
        Xtr = assemble(task, "train", with_z=wz, with_zft=wzft)
        Xva = assemble(task, "val", with_z=wz, with_zft=wzft)
        Xte = assemble(task, "test", with_z=wz, with_zft=wzft)
        if Xtr is None:
            continue
        clf = lgb_fit(Xtr, ytr)
        out[tag] = {"val_f1": f1(yva, clf.predict(Xva)),
                    "test_f1": f1(yte, clf.predict(Xte))}
        print(f"[stack-{task}] {tag}: {out[tag]}", flush=True)
    # val-fit（含直连头 probs）
    Xva = assemble(task, "val", with_probs=True)
    Xte = assemble(task, "test", with_probs=True)
    if Xva is not None and dprobs(task, "val") is not None:
        cv_scores = []
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        for tr_i, va_i in skf.split(Xva, yva):
            c = lgb_fit(Xva[tr_i], yva[tr_i])
            cv_scores.append(f1(yva[va_i], c.predict(Xva[va_i])))
        clf = lgb_fit(Xva, yva)
        out["valfit"] = {"cv5_f1": round(float(np.mean(cv_scores)), 4),
                         "test_f1": f1(yte, clf.predict(Xte))}
        print(f"[stack-{task}] valfit: {out['valfit']}", flush=True)
    results[task] = out


def main() -> int:
    results = {}
    print("== A ==", flush=True)
    run_a(results)
    print("== B ==", flush=True)
    run_bc("b", results)
    print("== C ==", flush=True)
    run_bc("c", results)
    RES.mkdir(parents=True, exist_ok=True)
    with open(RES / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[out] {RES / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
