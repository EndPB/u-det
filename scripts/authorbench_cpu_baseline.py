#!/usr/bin/env python
"""Small CPU-only baseline for the local AuthorBench task-aware subset.

The script deliberately avoids model downloads.  It compares a standardized
metadata baseline with a compact character n-gram representation, before and
after task-level centering.  Centered representations are transductive because
they use the eight unlabeled outputs belonging to the same task.
"""
from __future__ import annotations

import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "d-det" / "data" / "h2_authorbench"
OUT = ROOT / "d-det" / "artifacts" / "authorbench_cpu_baseline"
META_COLS = ["char_count", "num_lines", "nloc", "cyclomatic_complexity", "token_size"]
CS = [0.03, 0.1]
SEED = 0


def load_rows():
    rows = []
    with (DATA / "core.jsonl").open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            rows.append(r)
    assert len(rows) == 1912
    tasks = defaultdict(list)
    for i, r in enumerate(rows):
        tasks[r["task_id"]].append(i)
    assert len(tasks) == 239 and all(len(v) == 8 for v in tasks.values())
    assert all(len({rows[i]["task_split"] for i in ix}) == 1 for ix in tasks.values())
    counts = Counter(r["task_split"] for r in rows)
    assert counts == Counter({"train": 1336, "dev": 288, "test": 288})
    return rows, tasks


def standardize(X, train):
    mu = X[train].mean(axis=0)
    sd = np.maximum(X[train].std(axis=0), 1e-2)
    return ((X - mu) / sd).astype(np.float32)


def make_representations(rows, tasks):
    texts = [r["code"] for r in rows]
    vec = HashingVectorizer(
        analyzer="char", ngram_range=(3, 5), n_features=4096,
        alternate_sign=False, norm="l2", lowercase=False, dtype=np.float32,
    )
    X = vec.transform(texts).toarray().astype(np.float32)
    split = np.array([r["task_split"] for r in rows])
    train = split == "train"
    task_mean = {}
    for task, ix in tasks.items():
        task_mean[task] = X[ix].mean(axis=0)
    center = X - np.stack([task_mean[r["task_id"]] for r in rows])
    meta = np.array([[float(r[c]) for c in META_COLS] for r in rows], dtype=np.float32)
    return {
        "meta_std": standardize(meta, train),
        "char_raw": X,
        "char_center": center.astype(np.float32),
        "char_center_std": standardize(center, train),
    }


def eval_one(y, pred, classes):
    cm = confusion_matrix(y, pred, labels=np.arange(len(classes)))
    recall = np.diag(cm) / np.maximum(cm.sum(axis=1), 1)
    return {
        "macro_f1": round(float(f1_score(y, pred, average="macro")), 4),
        "balanced_acc": round(float(balanced_accuracy_score(y, pred)), 4),
        "per_class_recall": {c: round(float(v), 4) for c, v in zip(classes, recall)},
        "confusion": cm.tolist(), "n": int(len(y)),
    }


def fit_eval(X, y, split, classes):
    tr, dv, te = split == "train", split == "dev", split == "test"
    choices = {}
    for C in CS:
        clf = LogisticRegression(C=C, max_iter=1000, solver="lbfgs", random_state=SEED)
        clf.fit(X[tr], y[tr])
        p = clf.predict(X[dv])
        choices[str(C)] = {"dev_macro_f1": round(float(f1_score(y[dv], p, average="macro")), 4)}
    best = max(CS, key=lambda c: (choices[str(c)]["dev_macro_f1"], -c))
    clf = LogisticRegression(C=best, max_iter=1000, solver="lbfgs", random_state=SEED)
    clf.fit(X[tr], y[tr])
    return {"C_selected": best, "C_selection_dev": choices, **eval_one(y[te], clf.predict(X[te]), classes)}


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    rows, tasks = load_rows()
    split = np.array([r["task_split"] for r in rows])
    fams = sorted({r["family"] for r in rows})
    models = sorted({r["model_name"] for r in rows})
    y_f = np.array([fams.index(r["family"]) for r in rows])
    y_m = np.array([models.index(r["model_name"]) for r in rows])
    reps = make_representations(rows, tasks)
    metrics = {
        "protocol": "CPU HashingVectorizer char 3-5gram; no model download",
        "counts": {"rows": len(rows), "tasks": len(tasks), "split_rows": dict(Counter(split))},
        "chance": {"family_6way": 1 / 6, "model_8way": 1 / 8},
        "representations": {
            "meta_std": "five metadata columns, train-row standardization",
            "char_raw": "4096-dim character 3-5gram hashing, l2 normalized",
            "char_center": "char_raw minus mean of eight outputs in the same task (transductive)",
            "char_center_std": "char_center, then train-row per-dimension standardization",
        },
        "family": {}, "model": {},
    }
    predictions = {"split": split, "task_id": np.array([r["task_id"] for r in rows])}
    for name, X in reps.items():
        metrics["family"][name] = fit_eval(X, y_f, split, fams)
        metrics["model"][name] = fit_eval(X, y_m, split, models)
        tr, dv, te = split == "train", split == "dev", split == "test"
        # Save only test predictions to keep the artifact small and inspectable.
        clf_f = LogisticRegression(C=metrics["family"][name]["C_selected"], max_iter=1000, solver="lbfgs", random_state=SEED).fit(X[tr], y_f[tr])
        clf_m = LogisticRegression(C=metrics["model"][name]["C_selected"], max_iter=1000, solver="lbfgs", random_state=SEED).fit(X[tr], y_m[tr])
        predictions[f"family_test_{name}"] = clf_f.predict(X[te]).astype(np.int16)
        predictions[f"model_test_{name}"] = clf_m.predict(X[te]).astype(np.int16)
    metrics["elapsed_seconds"] = round(time.time() - t0, 2)
    metrics["cpu_threads"] = 2
    (OUT / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez_compressed(OUT / "predictions.npz", **predictions)
    (OUT / "config.json").write_text(json.dumps({
        "features": 4096, "char_ngram": [3, 5], "C_grid": CS,
        "standardization": "train rows only", "center_scope": "same task, all eight unlabeled rows",
        "threads": 2, "seed": SEED,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "solver.log").write_text(f"completed in {metrics['elapsed_seconds']} seconds\n", encoding="utf-8")
    (OUT / "README.md").write_text(
        "CPU-only AuthorBench baseline. The char_center and char_center_std representations are transductive; "
        "meta_std and char_raw are strict task-level baselines. No model was downloaded.\n",
        encoding="utf-8",
    )
    print(json.dumps({"out": str(OUT), "elapsed_seconds": metrics["elapsed_seconds"], "family": metrics["family"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
