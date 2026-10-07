"""Frozen CodeT5-small generator-heldout pair verification.

This is a capacity diagnostic for the H2 pair protocol.  It is deliberately
not a K-way family attribution experiment.  CodeT5-small is frozen; only a
balanced logistic-regression pair head is fitted within each generator-heldout
fold.  The vocabulary is fixed by the local model, the threshold is selected
on dev, and test labels are consumed once for reporting.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "d-det/data/h2_pair_benchmark_v1"
MODEL = ROOT / "d-det/models/codet5-small"
VIEWS = ("raw", "ids_only", "strings_only", "comments_only", "all")


def read(path: Path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def make_tokenizer(path: Path):
    # transformers 5.x changed RobertaTokenizer's constructor from
    # vocab_file/merges_file to vocab/merges.  Passing the old names silently
    # creates a five-token tokenizer, so keep this explicit and auditable.
    from transformers import RobertaTokenizer

    return RobertaTokenizer(
        vocab=str(path / "vocab.json"),
        merges=str(path / "merges.txt"),
        unk_token="<unk>", bos_token="<s>", eos_token="</s>",
        sep_token="</s>", cls_token="<s>", pad_token="<pad>",
        mask_token="<mask>", add_prefix_space=False,
    )


def head_tail(ids: list[int], max_length: int = 512) -> list[int]:
    if len(ids) <= max_length:
        return ids
    if max_length <= 128:
        return ids[:max_length]
    return ids[: max_length - 128] + ids[-128:]


def encode_texts(texts, tokenizer, model, device, batch_size=4, max_length=512):
    """Encode unique texts and return float32 mean-pooled vectors."""
    # Keep the code deterministic and avoid a second tokenizer implementation.
    encoded = [head_tail(tokenizer.encode(t, add_special_tokens=False), max_length) for t in texts]
    dim = int(model.config.d_model)
    result = np.empty((len(encoded), dim), dtype=np.float32)
    model.eval()
    for start in range(0, len(encoded), batch_size):
        batch = encoded[start : start + batch_size]
        length = max((len(x) for x in batch), default=1)
        ids = torch.full((len(batch), length), int(tokenizer.pad_token_id), dtype=torch.long)
        mask = torch.zeros((len(batch), length), dtype=torch.long)
        for i, row in enumerate(batch):
            if row:
                ids[i, : len(row)] = torch.tensor(row, dtype=torch.long)
                mask[i, : len(row)] = 1
        ids, mask = ids.to(device), mask.to(device)
        with torch.inference_mode():
            if device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    hidden = model(input_ids=ids, attention_mask=mask, return_dict=True).last_hidden_state
            else:
                hidden = model(input_ids=ids, attention_mask=mask, return_dict=True).last_hidden_state
            pooled = (hidden.float() * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1).float()
        result[start : start + len(batch)] = pooled.cpu().numpy()
    return result


def metric(scores, labels, threshold):
    pred = scores >= threshold
    return {
        "n": int(labels.size), "positive": int(labels.sum()), "threshold": float(threshold),
        "balanced_accuracy": float(balanced_accuracy_score(labels, pred)),
        "f1": float(f1_score(labels, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, scores)) if len(np.unique(labels)) == 2 else None,
        "average_precision": float(average_precision_score(labels, scores)) if len(np.unique(labels)) == 2 else None,
    }


def choose_threshold(scores, labels):
    values = np.unique(scores)
    candidates = np.concatenate(([values[0] - 1e-12], (values[:-1] + values[1:]) / 2, [values[-1] + 1e-12]))
    return float(max(candidates, key=lambda t: (balanced_accuracy_score(labels, scores >= t), -abs(float(t) - float(np.median(values))))))


def task_bootstrap(rows, scores, labels, threshold, repeats=500, seed=20261007):
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
        values.append(float(balanced_accuracy_score(labels[idx], scores[idx] >= threshold)))
    return {"tasks": len(tasks), "repeats": repeats, "valid_repeats": len(values),
            "mean": float(np.mean(values)) if values else None,
            "ci95": [float(np.quantile(values, .025)), float(np.quantile(values, .975))] if values else None}


def text_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "d-det/artifacts/h2_generator_heldout_codet5_small_2026-10-07")
    ap.add_argument("--views", nargs="+", choices=VIEWS, default=list(VIEWS))
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--diagnostic-folds", action="store_true", help="also run folds without train/dev positive support")
    args = ap.parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    torch.manual_seed(20261007)

    pairs = {r["pair_id"]: r for r in read(DATA / "pairs.jsonl")}
    fold_rows = read(DATA / "generator_fold_index.jsonl")
    by_fold = defaultdict(list)
    for index_row in fold_rows:
        if index_row["pair_id"] in pairs:
            by_fold[index_row["fold_id"]].append((index_row, pairs[index_row["pair_id"]]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = make_tokenizer(MODEL)
    from transformers import T5EncoderModel
    model = T5EncoderModel.from_pretrained(MODEL, local_files_only=True, dtype=(torch.float16 if device.type == "cuda" else torch.float32))
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    result = {
        "schema": "h2_generator_heldout_codet5_small_v1",
        "protocol": {
            "task": "same-family pair verification under generator-heldout split",
            "encoder": "Salesforce/codet5-small encoder only, frozen",
            "model_path": str(MODEL), "tokenizer": "RobertaTokenizer(vocab=..., merges=...)",
            "tokenizer_vocab_size": len(tokenizer), "hidden_size": int(model.config.d_model),
            "max_length": args.max_length, "long_sequence": "first 384 tokens + last 128 tokens",
            "special_tokens": False, "pooling": "attention-mask mean pooling",
            "pair_head": "LogisticRegression(class_weight=balanced, max_iter=500) on [z_left; z_right]",
            "threshold": "balanced-accuracy maximizing threshold on dev only",
            "test": "evaluated once after model and threshold selection",
            "device": str(device), "dtype": "float16" if device.type == "cuda" else "float32",
        },
        "folds": {}, "aggregate": {}, "diagnostic_folds": [], "runtime_seconds": None,
    }
    started = time.time()
    admitted = []
    for fold_id, entries in sorted(by_fold.items()):
        grouped = {role: [] for role in ("train", "dev", "test")}
        for index_row, pair in entries:
            grouped[index_row["role"]].append(pair)
        support = all(grouped[role] and {r["pair_label"] for r in grouped[role]} == {0, 1} for role in grouped)
        fold_result = {
            "source": entries[0][0]["source"], "family": entries[0][0]["family"],
            "heldout_generator": entries[0][0]["heldout_generator"],
            "counts": {k: len(v) for k, v in grouped.items()},
            "trainable_positive_support": support, "views": {},
        }
        if not support and not args.diagnostic_folds:
            fold_result["status"] = "diagnostic_only_no_train_dev_positive_support"
            result["diagnostic_folds"].append(fold_id)
            result["folds"][fold_id] = fold_result
            print(f"{fold_id}: diagnostic-only", flush=True)
            continue
        if support:
            admitted.append(fold_id)
        for view in args.views:
            texts = []
            for role in ("train", "dev", "test"):
                for row in grouped[role]:
                    texts.extend((row["left_views"][view], row["right_views"][view]))
            unique = list(dict.fromkeys(texts))
            embeddings = encode_texts(unique, tokenizer, model, device, args.batch_size, args.max_length)
            lookup = {text_key(text): embeddings[i] for i, text in enumerate(unique)}

            def make_xy(rows):
                left = np.stack([lookup[text_key(row["left_views"][view])] for row in rows])
                right = np.stack([lookup[text_key(row["right_views"][view])] for row in rows])
                return np.concatenate([left, right], axis=1), np.asarray([row["pair_label"] for row in rows], dtype=int)

            x_train, y_train = make_xy(grouped["train"])
            x_dev, y_dev = make_xy(grouped["dev"])
            x_test, y_test = make_xy(grouped["test"])
            head = LogisticRegression(class_weight="balanced", max_iter=500, random_state=20261007)
            head.fit(x_train, y_train)
            dev_scores = head.predict_proba(x_dev)[:, 1]
            test_scores = head.predict_proba(x_test)[:, 1]
            threshold = choose_threshold(dev_scores, y_dev)
            fold_result["views"][view] = {
                "unique_texts": len(unique), "train": metric(head.predict_proba(x_train)[:, 1], y_train, threshold),
                "dev": metric(dev_scores, y_dev, threshold), "test": metric(test_scores, y_test, threshold),
                "test_task_bootstrap": task_bootstrap(grouped["test"], test_scores, y_test, threshold),
            }
            print(f"{fold_id} {view}: ntext={len(unique)} test_ba={fold_result['views'][view]['test']['balanced_accuracy']:.4f}", flush=True)
        result["folds"][fold_id] = fold_result
    for view in args.views:
        rows = [result["folds"][fold]["views"][view]["test"] for fold in admitted]
        result["aggregate"][view] = {
            "admitted_folds": admitted, "fold_count": len(rows),
            "mean_ba": float(np.mean([r["balanced_accuracy"] for r in rows])) if rows else None,
            "mean_f1": float(np.mean([r["f1"] for r in rows])) if rows else None,
            "mean_auc": float(np.mean([r["roc_auc"] for r in rows])) if rows else None,
            "fold_test_metrics": rows,
        }
    result["runtime_seconds"] = time.time() - started
    result["interpretation"] = [
        "Pair verification only; this does not establish K-way family attribution.",
        "Google and Mistral folds are diagnostic-only unless --diagnostic-folds is explicitly supplied, because holding out one of two generators removes same-family positive support from train/dev.",
        "The unweighted aggregate is descriptive across admitted folds; it is not a pooled test estimate and is not directly comparable to CodeT5-base or fine-tuned d-det results.",
    ]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"device": str(device), "admitted_folds": admitted, "aggregate": {k: {x: v for x, v in val.items() if x in ("fold_count", "mean_ba", "mean_f1", "mean_auc")} for k, val in result["aggregate"].items()}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
