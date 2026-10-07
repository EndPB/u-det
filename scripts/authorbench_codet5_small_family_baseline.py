"""Frozen CodeT5-small K-way family attribution on task-heldout AuthorBench."""
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
from sklearn.metrics import balanced_accuracy_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "d-det/data/h2_authorbench"
MODEL = ROOT / "d-det/models/codet5-small"


def read(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def make_tokenizer(path):
    from transformers import RobertaTokenizer
    return RobertaTokenizer(
        vocab=str(path / "vocab.json"), merges=str(path / "merges.txt"),
        unk_token="<unk>", bos_token="<s>", eos_token="</s>",
        sep_token="</s>", cls_token="<s>", pad_token="<pad>",
        mask_token="<mask>", add_prefix_space=False,
    )


def encode_ids(ids, max_length=512):
    if len(ids) <= max_length:
        return ids
    return ids[: max_length - 128] + ids[-128:]


def encode_texts(texts, tokenizer, model, device, batch_size):
    encoded = [encode_ids(tokenizer.encode(t, add_special_tokens=False)) for t in texts]
    dim = int(model.config.d_model)
    out = np.empty((len(encoded), dim), dtype=np.float32)
    for start in range(0, len(encoded), batch_size):
        batch = encoded[start : start + batch_size]
        length = max((len(x) for x in batch), default=1)
        ids = torch.full((len(batch), length), tokenizer.pad_token_id, dtype=torch.long)
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
        out[start : start + len(batch)] = pooled.cpu().numpy()
    return out


def task_bootstrap(rows, y, pred, repeats=500, seed=20261007):
    by_task = defaultdict(list)
    for i, row in enumerate(rows):
        by_task[row["task_id"]].append(i)
    tasks = sorted(by_task)
    rng = random.Random(seed)
    values = []
    y = np.asarray(y); pred = np.asarray(pred)
    for _ in range(repeats):
        chosen = [tasks[rng.randrange(len(tasks))] for _ in tasks]
        idx = [i for task in chosen for i in by_task[task]]
        values.append(float(balanced_accuracy_score(y[idx], pred[idx])))
    return {"tasks": len(tasks), "repeats": repeats,
            "mean": float(np.mean(values)),
            "ci95": [float(np.quantile(values, .025)), float(np.quantile(values, .975))]}


def metric(y, pred):
    return {"n": int(len(y)), "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
            "report": classification_report(y, pred, output_dict=True, zero_division=0)}


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DATA / "core.jsonl")
    ap.add_argument("--out", type=Path, default=ROOT / "d-det/artifacts/authorbench_codet5_small_family_2026-10-07")
    ap.add_argument("--batch-size", type=int, default=4)
    args = ap.parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    started = time.time()
    rows = read(args.data)
    groups = {split: [r for r in rows if r.get("task_split") == split] for split in ("train", "dev", "test")}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = make_tokenizer(MODEL)
    from transformers import T5EncoderModel
    model = T5EncoderModel.from_pretrained(MODEL, local_files_only=True, dtype=(torch.float16 if device.type == "cuda" else torch.float32))
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    texts = [r["code"] for split in ("train", "dev", "test") for r in groups[split]]
    unique = list(dict.fromkeys(texts))
    emb = encode_texts(unique, tokenizer, model, device, args.batch_size)
    lookup = {hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest(): emb[i] for i, text in enumerate(unique)}

    def xy(split):
        part = groups[split]
        x = np.stack([lookup[hashlib.sha256(r["code"].encode("utf-8", errors="replace")).hexdigest()] for r in part])
        y = np.asarray([r["family"] for r in part])
        return x, y

    x_train, y_train = xy("train"); x_dev, y_dev = xy("dev"); x_test, y_test = xy("test")
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train); x_dev = scaler.transform(x_dev); x_test = scaler.transform(x_test)
    selection = {}
    best_c, best_dev = None, -1.0
    for c in (0.03, 0.1, 0.3):
        head = LogisticRegression(C=c, class_weight="balanced", max_iter=1000, random_state=20261007)
        head.fit(x_train, y_train)
        pred = head.predict(x_dev)
        score = f1_score(y_dev, pred, average="macro", zero_division=0)
        selection[str(c)] = metric(y_dev, pred)
        if score > best_dev:
            best_c, best_dev = c, score
    head = LogisticRegression(C=best_c, class_weight="balanced", max_iter=1000, random_state=20261007)
    head.fit(x_train, y_train)
    pred_train, pred_dev, pred_test = head.predict(x_train), head.predict(x_dev), head.predict(x_test)
    result = {
        "schema": "authorbench_codet5_small_family_v1",
        "protocol": {
            "data": str(args.data), "task": "six-way family attribution with supplied task-heldout split",
            "encoder": "Salesforce/codet5-small encoder only, frozen",
            "tokenizer_vocab_size": len(tokenizer), "max_length": 512,
            "truncation": "first 384 + last 128", "special_tokens": False,
            "pooling": "attention-mask mean pooling", "standardization": "train rows only",
            "head": "balanced LogisticRegression; C selected by dev macro-F1 only",
            "device": str(device), "dtype": "float16" if device.type == "cuda" else "float32",
        },
        "counts": {s: len(groups[s]) for s in groups},
        "classes": sorted(set(y_train)), "unique_texts": len(unique),
        "selection": selection, "best_C": best_c,
        "train": metric(y_train, pred_train), "dev": metric(y_dev, pred_dev),
        "test": metric(y_test, pred_test),
        "test_task_bootstrap": task_bootstrap(groups["test"], y_test, pred_test),
        "model_sha256": {name: sha256(MODEL / name) for name in ("config.json", "vocab.json", "merges.txt", "pytorch_model.bin")},
        "runtime_seconds": time.time() - started,
        "interpretation": [
            "This is task-heldout K-way family attribution, not generator-heldout evidence for every family.",
            "The source has one generator for most families and three for OpenAI in the broader AuthorBench package; no post-training causal role is inferred.",
            "The result is compared only with the same AuthorBench split and metric protocol, not with Droid or H2 pair verification aggregates.",
        ],
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("counts", "best_C", "dev", "test", "test_task_bootstrap", "runtime_seconds")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
