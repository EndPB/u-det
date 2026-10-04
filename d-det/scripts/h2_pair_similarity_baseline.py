"""Small, deterministic CPU baseline for h2_pair_benchmark_v1.

The baseline deliberately measures a similarity score rather than training a
large encoder.  It is a first-round diagnostic for the question: do lexical
views already separate same-family cross-generator pairs from same-task
cross-family hard negatives?  Vectorizers are fit on the training split only
and all reported test bootstrap samples are clustered by task_id.
"""
from __future__ import annotations

import json
import os
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
if not (ROOT / "d-det" / "data" / "h2_pair_benchmark_v1").exists():
    ROOT = Path(__file__).resolve().parents[2]  # 脚本位于 d-det/scripts 时回到仓库根
DATA = ROOT / "d-det/data/h2_pair_benchmark_v1"
OUT = ROOT / "d-det/artifacts/h2_pair_round1/similarity_baseline.json"
VIEWS = ("raw", "ids_only", "strings_only", "comments_only", "all")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def cosine_scores(vectorizer: TfidfVectorizer, rows: list[dict], view: str) -> np.ndarray:
    left = vectorizer.transform([row["left_views"][view] for row in rows])
    right = vectorizer.transform([row["right_views"][view] for row in rows])
    numerator = left.multiply(right).sum(axis=1).A1
    left_norm = np.sqrt(left.multiply(left).sum(axis=1)).A1
    right_norm = np.sqrt(right.multiply(right).sum(axis=1)).A1
    denominator = left_norm * right_norm
    return np.divide(numerator, denominator, out=np.zeros_like(numerator, dtype=float), where=denominator > 0)


def choose_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    """Choose a threshold on dev only; ties prefer the less extreme threshold."""
    values = np.unique(scores)
    if len(values) == 1:
        return float(values[0])
    candidates = np.concatenate(
        ([values[0] - 1e-12], (values[:-1] + values[1:]) / 2.0, [values[-1] + 1e-12])
    )
    best = None
    for threshold in candidates:
        pred = scores >= threshold
        value = balanced_accuracy_score(labels, pred)
        key = (float(value), -abs(float(threshold) - float(np.median(values))))
        if best is None or key > best[0]:
            best = (key, float(threshold))
    assert best is not None
    return best[1]


def metrics(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    pred = scores >= threshold
    return {
        "n": int(len(labels)),
        "positive": int(labels.sum()),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, pred)),
        "f1": float(f1_score(labels, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, scores)) if len(np.unique(labels)) == 2 else None,
        "average_precision": float(average_precision_score(labels, scores)) if len(np.unique(labels)) == 2 else None,
        "mean_score_positive": float(scores[labels == 1].mean()) if np.any(labels == 1) else None,
        "mean_score_negative": float(scores[labels == 0].mean()) if np.any(labels == 0) else None,
    }


def cluster_bootstrap(rows: list[dict], scores: np.ndarray, labels: np.ndarray, threshold: float, seed: int = 20261004, repeats: int = 300) -> dict:
    by_task: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_task[row["task_id"]].append(index)
    task_ids = sorted(by_task)
    rng = random.Random(seed)
    values = []
    for _ in range(repeats):
        chosen = [task_ids[rng.randrange(len(task_ids))] for _ in task_ids]
        indexes = [index for task_id in chosen for index in by_task[task_id]]
        if len({int(labels[index]) for index in indexes}) < 2:
            continue
        values.append(float(balanced_accuracy_score(labels[indexes], scores[indexes] >= threshold)))
    if not values:
        return {"tasks": len(task_ids), "repeats": repeats, "valid_repeats": 0}
    return {
        "tasks": len(task_ids),
        "repeats": repeats,
        "valid_repeats": len(values),
        "balanced_accuracy_mean": float(np.mean(values)),
        "balanced_accuracy_ci95": [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))],
    }


def evaluate_view(source: str, view: str, rows: list[dict], family_rows: list[dict]) -> dict:
    train = [row for row in rows if row["task_split"] == "train"]
    dev = [row for row in rows if row["task_split"] == "dev"]
    test = [row for row in rows if row["task_split"] == "test"]
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True, max_features=60000)
    vectorizer.fit([row["left_views"][view] for row in train] + [row["right_views"][view] for row in train])
    dev_scores = cosine_scores(vectorizer, dev, view)
    test_scores = cosine_scores(vectorizer, test, view)
    dev_labels = np.asarray([row["pair_label"] for row in dev], dtype=int)
    test_labels = np.asarray([row["pair_label"] for row in test], dtype=int)
    threshold = choose_threshold(dev_scores, dev_labels)
    family_test = [row for row in family_rows if row["task_split"] == "test"]
    family_scores = cosine_scores(vectorizer, family_test, view) if family_test else np.array([])
    family_labels = np.asarray([row["pair_label"] for row in family_test], dtype=int)
    return {
        "source": source,
        "view": view,
        "vectorizer": {"analyzer": "char_wb", "ngram_range": [2, 5], "min_df": 2, "max_features": 60000, "vocabulary_size": len(vectorizer.vocabulary_)},
        "tasks": {split: len({row["task_id"] for row in rows if row["task_split"] == split}) for split in ("train", "dev", "test")},
        "dev": metrics(dev_scores, dev_labels, threshold),
        "test": metrics(test_scores, test_labels, threshold),
        "test_task_cluster_bootstrap": cluster_bootstrap(test, test_scores, test_labels, threshold),
        "family_pair_balanced_test": metrics(family_scores, family_labels, threshold) if family_test else None,
    }


def main() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    rows = read_jsonl(DATA / "pairs.jsonl")
    family_rows = read_jsonl(DATA / "family_pair_balanced_pairs.jsonl")
    result = {
        "schema": "h2_pair_similarity_baseline_v1",
        "protocol": "source-isolated char TF-IDF cosine; threshold selected on dev; test bootstrap clustered by task_id",
        "input": "d-det/data/h2_pair_benchmark_v1/pairs.jsonl",
        "rows": len(rows),
        "sources": {},
        "warning": "Diagnostic baseline only. It does not establish a SOTA result or a learned family attribution space.",
    }
    for source in sorted({row["source"] for row in rows}):
        source_rows = [row for row in rows if row["source"] == source]
        source_family_rows = [row for row in family_rows if row["source"] == source]
        result["sources"][source] = {view: evaluate_view(source, view, source_rows, source_family_rows) for view in VIEWS}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
