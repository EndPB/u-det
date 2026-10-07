"""CPU-only generator-heldout TF-IDF pair verification baseline.

This is deliberately a pair verification experiment.  It is not K-way family
attribution, and it never uses test labels to choose a threshold or vocabulary.
Folds with no positive train/dev support are reported as diagnostic-only and
skipped from the admitted aggregate.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "d-det/data/h2_pair_benchmark_v1"
VIEWS = ("raw", "ids_only", "strings_only", "comments_only", "all")


def read(path: Path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def threshold(scores, labels):
    values = np.unique(scores)
    candidates = np.concatenate(([values[0] - 1e-12], (values[:-1] + values[1:]) / 2, [values[-1] + 1e-12]))
    best = max(candidates, key=lambda t: (balanced_accuracy_score(labels, scores >= t), -abs(float(t) - float(np.median(values)))))
    return float(best)


def score(vectorizer, rows, view):
    left = vectorizer.transform([r["left_views"][view] for r in rows])
    right = vectorizer.transform([r["right_views"][view] for r in rows])
    dot = left.multiply(right).sum(axis=1).A1
    norm = np.sqrt(left.multiply(left).sum(axis=1)).A1 * np.sqrt(right.multiply(right).sum(axis=1)).A1
    return np.divide(dot, norm, out=np.zeros_like(dot, dtype=float), where=norm > 0)


def metric(scores, labels, t):
    pred = scores >= t
    return {
        "n": int(labels.size),
        "positive": int(labels.sum()),
        "threshold": float(t),
        "balanced_accuracy": float(balanced_accuracy_score(labels, pred)),
        "f1": float(f1_score(labels, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, scores)) if len(np.unique(labels)) == 2 else None,
        "average_precision": float(average_precision_score(labels, scores)) if len(np.unique(labels)) == 2 else None,
        "mean_positive": float(scores[labels == 1].mean()) if np.any(labels == 1) else None,
        "mean_negative": float(scores[labels == 0].mean()) if np.any(labels == 0) else None,
    }


def task_bootstrap(rows, scores, labels, t, seed=20261007, repeats=500):
    by_task = defaultdict(list)
    for i, row in enumerate(rows):
        by_task[row["task_id"]].append(i)
    tasks = sorted(by_task)
    rng = random.Random(seed)
    values = []
    for _ in range(repeats):
        selected = [tasks[rng.randrange(len(tasks))] for _ in tasks]
        idx = [i for task in selected for i in by_task[task]]
        if len(set(labels[idx])) < 2:
            continue
        values.append(float(balanced_accuracy_score(labels[idx], scores[idx] >= t)))
    return {"tasks": len(tasks), "repeats": repeats, "valid_repeats": len(values),
            "mean": float(np.mean(values)) if values else None,
            "ci95": [float(np.quantile(values, .025)), float(np.quantile(values, .975))] if values else None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "d-det/artifacts/h2_generator_heldout_tfidf_2026-10-07")
    args = parser.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    pairs = {r["pair_id"]: r for r in read(DATA / "pairs.jsonl")}
    fold_rows = read(DATA / "generator_fold_index.jsonl")
    by_fold = defaultdict(list)
    for row in fold_rows:
        if row["pair_id"] in pairs:
            by_fold[row["fold_id"]].append((row, pairs[row["pair_id"]]))
    result = {"schema": "h2_generator_heldout_tfidf_v1", "protocol": "char_wb TF-IDF cosine; vocabulary fit on train endpoints; threshold selected on dev; task-cluster bootstrap on test; no test selection", "views": list(VIEWS), "folds": {}, "aggregate": {}}
    admitted = []
    for fold_id, entries in sorted(by_fold.items()):
        grouped = {role: [] for role in ("train", "dev", "test")}
        for index_row, pair in entries:
            grouped[index_row["role"]].append(pair)
        support = all(grouped[role] and {r["pair_label"] for r in grouped[role]} == {0, 1} for role in grouped)
        fold_result = {"source": entries[0][0]["source"], "family": entries[0][0]["family"], "heldout_generator": entries[0][0]["heldout_generator"], "counts": {k: len(v) for k, v in grouped.items()}, "trainable_positive_support": support, "views": {}}
        if support:
            admitted.append(fold_id)
            for view in VIEWS:
                vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True, max_features=60000)
                vectorizer.fit([r["left_views"][view] for r in grouped["train"]] + [r["right_views"][view] for r in grouped["train"]])
                dev_scores = score(vectorizer, grouped["dev"], view); test_scores = score(vectorizer, grouped["test"], view)
                dev_labels = np.asarray([r["pair_label"] for r in grouped["dev"]], dtype=int); test_labels = np.asarray([r["pair_label"] for r in grouped["test"]], dtype=int)
                t = threshold(dev_scores, dev_labels)
                fold_result["views"][view] = {"vocabulary_size": len(vectorizer.vocabulary_), "dev": metric(dev_scores, dev_labels, t), "test": metric(test_scores, test_labels, t), "test_task_bootstrap": task_bootstrap(grouped["test"], test_scores, test_labels, t)}
        else:
            fold_result["status"] = "diagnostic_only_no_train_dev_positive_support"
        result["folds"][fold_id] = fold_result
        print(f"{fold_id}: {'admitted' if support else 'diagnostic-only'}", flush=True)
    for view in VIEWS:
        rows = []
        for fold_id in admitted:
            rows.append(result["folds"][fold_id]["views"][view]["test"])
        result["aggregate"][view] = {"admitted_folds": admitted, "fold_count": len(rows), "mean_ba": float(np.mean([r["balanced_accuracy"] for r in rows])) if rows else None, "mean_f1": float(np.mean([r["f1"] for r in rows])) if rows else None, "mean_auc": float(np.mean([r["roc_auc"] for r in rows])) if rows else None, "fold_test_metrics": rows}
    result["interpretation"] = ["Pair verification only; not K-way family attribution.", "Google and Mistral two-generator folds are diagnostic-only because holding out one generator leaves no same-family positive in train/dev.", "Aggregate is an unweighted descriptive mean over admitted folds, not a pooled test estimate."]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"admitted_folds": admitted, "aggregate": {k: {x: v for x, v in value.items() if x in ('fold_count','mean_ba','mean_f1','mean_auc')} for k, value in result['aggregate'].items()}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
