"""Local CPU controls for the CLOSED H3 frozen-encoder batch; no GPU reruns.

The historical 0.9084 is task-macro AUROC, whereas the GPU result is row
AUROC on different generator masks. Fit the existing lexical control on the
exact GPU folds and report both estimands without modifying the old results.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import numpy as np
import sklearn
from scipy.sparse import hstack
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "d-det/data/h3_stacad_revision_v1"
OUT = ROOT / "d-det/artifacts/h3_matched_fold_closeout_2026-10-11"
GPU_PATH = "d-det/artifacts/h3_gpu_diagnostic_batch_2026-10-11/metrics.json"
SEED = 20261010


def write(path, obj):
    path.write_bytes((json.dumps(obj, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    commit = subprocess.check_output(["git", "rev-parse", "origin/main"], cwd=ROOT, text=True).strip()
    raw = subprocess.check_output(["git", "show", f"{commit}:{GPU_PATH}"], cwd=ROOT)
    gpu = json.loads(raw)
    (OUT / "server_metrics_snapshot.json").write_bytes(raw)
    train_path = DATA / "train_balanced.jsonl"
    dev_path = DATA / "dev_clean.jsonl"
    hashes = {p.name: sha(p) for p in (train_path, dev_path)}
    assert all(hashes[name] == gpu["data_sha256"][name] for name in hashes)
    train = [json.loads(x) for x in train_path.read_text(encoding="utf-8").splitlines() if x]
    dev = [json.loads(x) for x in dev_path.read_text(encoding="utf-8").splitlines() if x]
    assert {r["task_split"] for r in train} == {"train"}
    assert {r["task_split"] for r in dev} == {"dev"}
    assert not ({r["task_id"] for r in train} & {r["task_id"] for r in dev})
    config = {
        "source_results_commit": commit, "source_metrics_sha256": hashlib.sha256(raw).hexdigest(),
        "data_sha256": hashes, "folds": gpu["protocol"]["folds"], "seed": SEED,
        "lexical": {"analyzers": ["char_wb(2,5)", "word(1,2)"], "min_df": 5,
                    "max_features_each": 100000, "lowercase": False, "sublinear_tf": True,
                    "classifier": "LinearSVC(C=1, max_iter=3000)", "fit_scope": "fold train only"},
        "estimands": ["row_auroc", "task_macro_auroc"], "bootstrap_task_repeats": 1000,
        "test_read": False, "gpu_used": False, "historical_gpu_batch_closed": True,
        "script_sha256": sha(Path(__file__)),
    }
    write(OUT / "config.json", config)  # written before fitting
    results = {"folds": {}, "interpretation": [
        "Post hoc local controls, not a new preregistered GPU experiment.",
        "Compare GPU row AUROC only with matched lexical row AUROC; historical .9084 is not matched.",
        "No GPU per-row scores were saved, so paired neural-control CI cannot be reconstructed.",
        "Source-only has three unique runs repeated under two fold names, not six independent runs.",
        "Source unseen_rows=659 counts HUMAN rows, not unseen AI generators.",
        "No end-to-end encoder learning, convergence curves or evaluation variant scores in this batch.",
    ]}
    for fold, heldout in config["folds"].items():
        started = time.time()
        fit = [r for r in train if r["generator"] == "human" or r["generator"] not in heldout]
        eval_rows = [r for r in dev if r["generator"] == "human" or r["generator"] in heldout]
        reference = [r for r in gpu["runs"] if r["job"] == "detection_only" and r["fold"] == fold]
        assert all(r["train_rows"] == len(fit) and r["dev_rows"] == len(eval_rows) for r in reference)
        print(f"FITTING {fold}: train={len(fit)} dev={len(eval_rows)}", flush=True)
        xt, xe = [], []
        for analyzer, ngrams in (("char_wb", (2, 5)), ("word", (1, 2))):
            vectorizer = TfidfVectorizer(analyzer=analyzer, ngram_range=ngrams, min_df=5,
                max_features=100000, lowercase=False, sublinear_tf=True,
                token_pattern=r"[A-Za-z_][A-Za-z0-9_]*")
            xt.append(vectorizer.fit_transform([r["code"] for r in fit]))
            xe.append(vectorizer.transform([r["code"] for r in eval_rows]))
        y = np.asarray([r["label"] for r in eval_rows])
        clf = LinearSVC(C=1, random_state=SEED, max_iter=3000)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            clf.fit(hstack(xt).tocsr(), [r["label"] for r in fit])
        scores = clf.decision_function(hstack(xe).tocsr())
        tasks = np.asarray([r["task_id"] for r in eval_rows])
        unique_tasks = sorted(set(tasks))
        task_auc = np.asarray([roc_auc_score(y[tasks == task], scores[tasks == task]) for task in unique_tasks])
        rng = np.random.default_rng(SEED + 1)
        draws = task_auc[rng.integers(0, len(task_auc), size=(1000, len(task_auc)))].mean(axis=1)
        neural_row = {job: float(np.mean([r["detection"]["auroc"] for r in gpu["runs"]
                     if r["fold"] == fold and r["job"] == job]))
                     for job in ("detection_only", "joint", "joint_invariance")}
        record = {"train_rows": len(fit), "eval_rows": len(eval_rows), "eval_tasks": len(unique_tasks),
                  "row_auroc": float(roc_auc_score(y, scores)), "task_macro_auroc": float(task_auc.mean()),
                  "task_macro_ci95": np.quantile(draws, [.025, .975]).tolist(),
                  "gpu_row_auroc": neural_row, "seconds": time.time() - started,
                  "solver_iterations": int(clf.n_iter_),
                  "convergence_warnings": [str(w.message) for w in caught]}
        results["folds"][fold] = record
        np.savez_compressed(OUT / f"{fold}_lexical_scores.npz", scores=scores, labels=y,
                            row_ids=np.asarray([r["row_id"] for r in eval_rows]), tasks=tasks)
        write(OUT / "metrics.json", results)
        print(json.dumps(record, ensure_ascii=False), flush=True)
    results["aggregate"] = {"lexical_row_auroc_fold_mean": float(np.mean([r["row_auroc"] for r in results["folds"].values()])),
                            "lexical_task_macro_fold_mean": float(np.mean([r["task_macro_auroc"] for r in results["folds"].values()]))}
    results["closure"] = "frozen_codet5_small_three_epoch_head_increment_not_established_closed"
    write(OUT / "metrics.json", results)
    write(OUT / "env.json", {"python": sys.version, "platform": platform.platform(),
                             "sklearn": sklearn.__version__, "numpy": np.__version__})
    (OUT / "commands.txt").write_bytes(b"python scripts/h3_matched_fold_closeout.py\n")
    (OUT / "SHA256SUMS.txt").write_bytes("".join(f"{sha(p)}  {p.name}\n" for p in sorted(OUT.iterdir())
        if p.is_file() and p.name not in {"SHA256SUMS.txt", "run.log.txt"}).encode("utf-8"))


if __name__ == "__main__":
    run()
