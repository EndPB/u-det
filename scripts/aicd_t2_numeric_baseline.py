"""AICD Task 2 compact numeric-label baseline.

The public parquet-derived package exposes integer labels only.  This script
reports numeric-ID performance and never assigns those IDs to model families.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, classification_report, f1_score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "d-det/data/aicd_t2_numeric_balanced_v1"


def read(path):
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows.append((str(row["code"]), int(row["label"])))
    return rows


def metric(y, pred):
    return {"n": int(len(y)), "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
            "report": classification_report(y, pred, output_dict=True, zero_division=0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "d-det/artifacts/aicd_t2_numeric_baseline_2026-10-07")
    ap.add_argument("--max-features", type=int, default=80000)
    args = ap.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    started = time.time()
    train, validation, test = (read(DATA / name) for name in ("train.jsonl", "validation.jsonl", "test.jsonl"))
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=2,
                                 max_features=args.max_features, sublinear_tf=True, dtype=np.float32)
    x_train = vectorizer.fit_transform([x for x, _ in train])
    x_val = vectorizer.transform([x for x, _ in validation])
    x_test = vectorizer.transform([x for x, _ in test])
    y_train = np.asarray([y for _, y in train]); y_val = np.asarray([y for _, y in validation]); y_test = np.asarray([y for _, y in test])
    head = LogisticRegression(class_weight="balanced", C=1.0, max_iter=1000, random_state=20261007)
    head.fit(x_train, y_train)
    pred_train, pred_val, pred_test = head.predict(x_train), head.predict(x_val), head.predict(x_test)
    result = {
        "schema": "aicd_t2_numeric_baseline_v1",
        "data": json.loads((DATA / "summary.json").read_text(encoding="utf-8")),
        "sha256sums": (DATA / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines(),
        "protocol": {
            "label_space": "numeric IDs only; no family names assigned",
            "split": "derived package train/validation/test",
            "features": {"analyzer": "char", "ngram_range": [3, 5], "min_df": 2, "max_features": args.max_features, "sublinear_tf": True},
            "head": "balanced LogisticRegression(C=1.0, max_iter=1000)",
            "selection": "fixed settings; validation is descriptive and test is final",
        },
        "counts": {
            split: {str(int(label)): int(count) for label, count in Counter(values).items()}
            for split, values in (("train", y_train), ("validation", y_val), ("test", y_test))
        },
        "vocabulary_size": len(vectorizer.vocabulary_),
        "train": metric(y_train, pred_train), "validation": metric(y_val, pred_val), "test": metric(y_test, pred_test),
        "chance_balanced_accuracy": 1.0 / len(set(y_train)),
        "runtime_seconds": time.time() - started,
        "interpretation": [
            "Results are numeric-ID closed-set diagnostics only; no integer is named as a model family.",
            "This compact balanced package is not a replacement for the official full T2 protocol.",
            "The public data has no task_id, generator metadata, or official numeric-to-family mapping in the local package.",
        ],
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"counts": {k: len(v) for k, v in (('train', train), ('validation', validation), ('test', test))}, "vocabulary_size": len(vectorizer.vocabulary_), "test": result["test"], "runtime_seconds": result["runtime_seconds"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
