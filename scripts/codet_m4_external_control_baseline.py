"""CoDET-M4 external control baselines.

The package is an external source-fingerprint control, not same-task H2 data.
This script therefore keeps its two label spaces separate:

* human/AI detection (``target``), and
* closed-set source-model attribution among the five named AI models.

No family name is transferred to Droid, AuthorBench, AICD, or any other
dataset.  Text and upstream structural features are evaluated independently;
language is reported as a subgroup, never used as a label.
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
from sklearn.metrics import balanced_accuracy_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "d-det/data/codet_m4_balanced_control_v1"
STRUCTURE_KEYS = (
    "avgFunctionLength", "avgIdentifierLength", "avgLineLength", "emptyLinesDensity",
    "functionDefinitionDensity", "maxDecisionTokens", "maintainabilityIndex", "whiteSpaceRatio",
)


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def metrics(y_true, y_pred):
    return {
        "n": int(len(y_true)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "labels": sorted({str(x) for x in y_true}),
    }


def subgroup_metrics(y_true, y_pred, groups):
    out = {}
    for group in sorted(set(groups)):
        idx = np.asarray([g == group for g in groups])
        if idx.sum() == 0:
            continue
        out[str(group)] = metrics(np.asarray(y_true)[idx], np.asarray(y_pred)[idx])
    return out


def load_splits():
    return {split: list(read_jsonl(DATA / f"{split}.jsonl")) for split in ("train", "val", "test")}


def vector_text(rows):
    return [str(r.get("code") or "") for r in rows]


def structure_matrix(rows):
    values = []
    missing = 0
    for row in rows:
        feat = row.get("features") or {}
        current = []
        for key in STRUCTURE_KEYS:
            val = feat.get(key)
            if val is None:
                missing += 1
                val = 0.0
            try:
                val = float(val)
            except (TypeError, ValueError):
                missing += 1
                val = 0.0
            if not np.isfinite(val):
                missing += 1
                val = 0.0
            current.append(val)
        values.append(current)
    return np.asarray(values, dtype=np.float32), missing


def run_task(splits, task, feature_kind):
    if task == "detection":
        train = splits["train"]
        val = splits["val"]
        test = splits["test"]
        label = lambda r: str(r["target"])
    elif task == "model_attribution":
        # __none__ has no identified source, while human is a reference class;
        # neither is silently assigned to one of the five named AI models.
        keep = lambda r: str(r.get("model")) not in {"human", "__none__"} and str(r.get("target")) == "ai"
        train = [r for r in splits["train"] if keep(r)]
        val = [r for r in splits["val"] if keep(r)]
        test = [r for r in splits["test"] if keep(r)]
        label = lambda r: str(r["model"])
    else:
        raise ValueError(task)

    if feature_kind == "text":
        vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=2,
                                     max_features=60000, sublinear_tf=True, dtype=np.float32)
        x_train = vectorizer.fit_transform(vector_text(train))
        x_val = vectorizer.transform(vector_text(val))
        x_test = vectorizer.transform(vector_text(test))
        feature_meta = {"kind": "char_tfidf", "vocabulary_size": len(vectorizer.vocabulary_), "ngram_range": [3, 5], "max_features": 60000}
    elif feature_kind == "structure":
        x_train, missing_train = structure_matrix(train)
        x_val, missing_val = structure_matrix(val)
        x_test, missing_test = structure_matrix(test)
        feature_meta = {"kind": "upstream_features", "keys": list(STRUCTURE_KEYS), "missing_values": {"train": missing_train, "val": missing_val, "test": missing_test}}
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_val = scaler.transform(x_val)
        x_test = scaler.transform(x_test)
    else:
        raise ValueError(feature_kind)

    y_train = np.asarray([label(r) for r in train])
    y_val = np.asarray([label(r) for r in val])
    y_test = np.asarray([label(r) for r in test])
    head = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=20261007)
    head.fit(x_train, y_train)
    p_train, p_val, p_test = head.predict(x_train), head.predict(x_val), head.predict(x_test)
    return {
        "task": task, "feature": feature_meta,
        "counts": {split: dict(Counter(label(r) for r in rows)) for split, rows in (("train", train), ("val", val), ("test", test))},
        "train": metrics(y_train, p_train), "val": metrics(y_val, p_val), "test": metrics(y_test, p_test),
        "test_by_language": subgroup_metrics(y_test, p_test, [r.get("language", "") for r in test]),
        "test_report": classification_report(y_test, p_test, output_dict=True, zero_division=0),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "d-det/artifacts/codet_m4_external_control_2026-10-07")
    args = ap.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    started = time.time()
    splits = load_splits()
    result = {
        "schema": "codet_m4_external_control_baseline_v1",
        "data": json.loads((DATA / "summary.json").read_text(encoding="utf-8")),
        "sha256sums": (DATA / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines(),
        "protocol": {
            "fixed_split": "train/val/test supplied by CoDET-M4 derived package",
            "selection": "no test-driven selection; fixed char TF-IDF and balanced logistic settings",
            "label_boundary": "model labels are CoDET-M4 source identifiers only; not merged with other family spaces",
            "human_boundary": "human and __none__ excluded from model attribution; human remains in detection task",
        },
        "results": {},
    }
    for task in ("detection", "model_attribution"):
        for feature in ("text", "structure"):
            key = f"{task}__{feature}"
            result["results"][key] = run_task(splits, task, feature)
            print(key, result["results"][key]["test"], flush=True)
    result["runtime_seconds"] = time.time() - started
    result["interpretation"] = [
        "This is an external control, not same-task H2 evidence and not a cross-dataset family attribution result.",
        "Text performance may include language/source/repository shortcuts; the structural-feature result quantifies a separate shortcut channel.",
        "The package has no task_id/prompt_id and its source model names are not transferred to Droid, AuthorBench, LLM-CodeGen, STACAD, or AICD.",
    ]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: {"test": v["test"], "test_by_language": v["test_by_language"]} for k, v in result["results"].items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
