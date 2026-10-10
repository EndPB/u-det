#!/usr/bin/env python3
"""One-shot GPU diagnostic batch for the revised STACAD H3 package.

This batch is deliberately separate from the formal H3 gate.  It answers a
practical question the gate cannot answer: whether a shared CodeT5
representation can add signal after the v1 data revision.  It never reads
test payloads, never downloads weights, and never changes a configuration
after the first job starts.

The encoder is run on CUDA once and its frozen representations are cached in
memory.  Four small heads then run on CUDA for two deterministic
generator-held-out folds and three seeds.  ``joint_invariance`` adds a
parent/variant logit consistency loss using train-only variants.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "d-det/data/h3_stacad_revision_v1"
MODEL = ROOT / "d-det/models/codet5-small"
SEEDS = (20261011, 20261012, 20261013)
JOBS = ("detection_only", "source_only", "joint", "joint_invariance")
MAX_SOURCE_CHARS = 24000


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def deterministic_limit(rows: list[dict], limit: int | None) -> list[dict]:
    if not limit or len(rows) <= limit:
        return rows
    # Evenly spaced selection is deterministic and keeps the original
    # task-balanced ordering of the v1 materialization.
    indices = np.linspace(0, len(rows) - 1, num=limit, dtype=int)
    return [rows[int(i)] for i in indices]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def head_tail(ids: list[int], max_length: int = 512) -> list[int]:
    if len(ids) <= max_length:
        return ids
    return ids[: max_length - 128] + ids[-128:]


def make_tokenizer():
    from transformers import RobertaTokenizer

    return RobertaTokenizer(
        vocab=str(MODEL / "vocab.json"), merges=str(MODEL / "merges.txt"),
        unk_token="<unk>", bos_token="<s>", eos_token="</s>",
        sep_token="</s>", cls_token="<s>", pad_token="<pad>",
        mask_token="<mask>", add_prefix_space=False,
    )


def encode_unique(rows: list[dict], tokenizer, encoder, device, batch_size: int,
                  max_length: int) -> tuple[dict[str, np.ndarray], dict]:
    texts = {}
    for row in rows:
        texts[row["code_sha256"]] = row["code"]
    keys = sorted(texts)
    vectors = np.empty((len(keys), int(encoder.config.d_model)), dtype=np.float32)
    encoder.eval()
    lengths = []
    started = time.time()
    for start in range(0, len(keys), batch_size):
        batch_keys = keys[start : start + batch_size]
        # Avoid spending minutes tokenizing pathological multi-hundred-kilobyte
        # files before the 512-token head/tail cap can take effect.
        compact = []
        for key in batch_keys:
            text = texts[key]
            if len(text) > MAX_SOURCE_CHARS:
                text = text[: MAX_SOURCE_CHARS - 6000] + "\n/* ... source_chars_capped ... */\n" + text[-5900:]
            compact.append(text)
        encoded = [head_tail(tokenizer.encode(text, add_special_tokens=False), max_length)
                   for text in compact]
        lengths.extend(len(x) for x in encoded)
        width = max((len(x) for x in encoded), default=1)
        ids = torch.full((len(encoded), width), int(tokenizer.pad_token_id), dtype=torch.long, device=device)
        mask = torch.zeros_like(ids)
        for i, tokens in enumerate(encoded):
            if tokens:
                ids[i, : len(tokens)] = torch.tensor(tokens, dtype=torch.long, device=device)
                mask[i, : len(tokens)] = 1
        with torch.inference_mode():
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                hidden = encoder(input_ids=ids, attention_mask=mask, return_dict=True).last_hidden_state
            pooled = (hidden.float() * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1).float()
        vectors[start : start + len(batch_keys)] = pooled.cpu().numpy()
        if start == 0 or start + len(batch_keys) == len(keys):
            print(f"ENCODE {start + len(batch_keys)}/{len(keys)}", flush=True)
    return {key: vectors[i] for i, key in enumerate(keys)}, {
        "unique_codes": len(keys), "median_tokens": int(np.median(lengths)) if lengths else 0,
        "max_tokens": max(lengths, default=0), "seconds": time.time() - started,
    }


class SharedHeads(nn.Module):
    def __init__(self, dim: int, source_classes: int):
        super().__init__()
        hidden = min(512, max(128, dim // 2))
        self.body = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(0.10))
        self.det = nn.Linear(hidden, 1)
        self.source = nn.Linear(hidden, source_classes)

    def forward(self, x):
        z = self.body(x)
        return self.det(z).squeeze(-1), self.source(z), z


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def metric(scores: np.ndarray, labels: np.ndarray) -> dict:
    pred = scores >= 0.5
    return {
        "n": int(labels.size), "positive": int(labels.sum()),
        "balanced_accuracy": float(balanced_accuracy_score(labels, pred)),
        "auroc": float(roc_auc_score(labels, scores)) if len(np.unique(labels)) == 2 else None,
    }


def source_metric(logits: np.ndarray, labels: np.ndarray, known: set[int]) -> dict:
    keep = np.asarray([int(y) in known for y in labels], dtype=bool)
    if not keep.any():
        return {"status": "no_seen_source_in_eval", "n": 0}
    pred = logits[keep].argmax(1)
    return {"status": "seen_source_only", "n": int(keep.sum()),
            "accuracy": float((pred == labels[keep]).mean()),
            "unseen_rows": int((~keep).sum())}


def tensor_rows(rows: list[dict], vectors: dict[str, np.ndarray], device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[dict]]:
    x = torch.tensor(np.stack([vectors[r["code_sha256"]] for r in rows]), dtype=torch.float32, device=device)
    y = torch.tensor([int(r["label"]) for r in rows], dtype=torch.float32, device=device)
    return x, y, rows


def run_head(job: str, seed: int, fold_id: str, train_rows: list[dict], dev_rows: list[dict],
             vectors: dict[str, np.ndarray], source_to_id: dict[str, int], variants: list[dict], device,
             epochs: int, batch_size: int, lr: float) -> dict:
    set_seed(seed)
    dim = len(next(iter(vectors.values())))
    model = SharedHeads(dim, len(source_to_id)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    train_x, train_y, _ = tensor_rows(train_rows, vectors, device)
    train_source = torch.tensor([source_to_id.get(r["generator"], -1) for r in train_rows], dtype=torch.long, device=device)
    train_source_mask = (train_source >= 0) & (train_y > 0)
    variant_pairs = []
    if job == "joint_invariance":
        by_parent = defaultdict(list)
        for row in variants:
            if row.get("parent_row_id"):
                by_parent[row["parent_row_id"]].append(row)
        for row in train_rows:
            choices = by_parent.get(row["row_id"], [])
            if choices:
                variant_pairs.append((row, choices[0]))
    order = np.arange(len(train_rows))
    for epoch in range(epochs):
        np.random.shuffle(order)
        model.train()
        for start in range(0, len(order), batch_size):
            idx = torch.as_tensor(order[start : start + batch_size], dtype=torch.long, device=device)
            det_logits, src_logits, _ = model(train_x[idx])
            loss = F.binary_cross_entropy_with_logits(det_logits, train_y[idx])
            if job in {"source_only", "joint", "joint_invariance"}:
                mask = train_source_mask[idx]
                if bool(mask.any()):
                    source_loss = F.cross_entropy(src_logits[mask], train_source[idx][mask])
                    loss = source_loss if job == "source_only" else loss + 0.20 * source_loss
            if job == "joint_invariance" and variant_pairs:
                parents = variant_pairs[start % len(variant_pairs): (start % len(variant_pairs)) + min(batch_size, len(variant_pairs))]
                px = torch.tensor(np.stack([vectors[p[0]["code_sha256"]] for p in parents]), dtype=torch.float32, device=device)
                vx = torch.tensor(np.stack([vectors[p[1]["code_sha256"]] for p in parents]), dtype=torch.float32, device=device)
                plogit = model(px)[0]; vlogit = model(vx)[0]
                loss = loss + 0.10 * F.mse_loss(torch.sigmoid(plogit), torch.sigmoid(vlogit))
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()

    model.eval()
    with torch.inference_mode():
        dev_x, dev_y, _ = tensor_rows(dev_rows, vectors, device)
        det, src, _ = model(dev_x)
        det_scores = torch.sigmoid(det).cpu().numpy()
        src_logits = src.cpu().numpy()
    labels = dev_y.cpu().numpy().astype(int)
    result = {"job": job, "seed": seed, "fold": fold_id, "train_rows": len(train_rows),
              "dev_rows": len(dev_rows),
              "detection": ({"status": "not_trained_source_only"} if job == "source_only" else metric(det_scores, labels))}
    if job in {"source_only", "joint", "joint_invariance"}:
        src_labels = np.asarray([source_to_id.get(r["generator"], -1) for r in dev_rows])
        result["source"] = source_metric(src_logits, src_labels, set(source_to_id.values()) & set(source_to_id.get(r["generator"], -1) for r in train_rows))
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "d-det/artifacts/h3_gpu_diagnostic_batch_2026-10-11")
    ap.add_argument("--jobs", nargs="+", choices=JOBS, default=list(JOBS))
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--encode-batch-size", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--limit-train", type=int, default=None, help="local smoke cap; omitted for the full batch")
    ap.add_argument("--limit-dev", type=int, default=None, help="local smoke cap; omitted for the full batch")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this batch; no CPU fallback is allowed")
    device = torch.device("cuda")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("MKL_NUM_THREADS", "4")
    args.out.mkdir(parents=True, exist_ok=True)
    train = deterministic_limit(read_jsonl(DATA / "train_balanced.jsonl"), args.limit_train)
    dev = deterministic_limit(read_jsonl(DATA / "dev_clean.jsonl"), args.limit_dev)
    variants = read_jsonl(DATA / "train_variants.jsonl")
    if args.limit_train:
        parent_ids = {r["row_id"] for r in train}
        variants = [r for r in variants if r.get("parent_row_id") in parent_ids]
    # Only v1 train/dev files are read.  The test split is not opened.
    all_rows = train + dev + variants
    tokenizer = make_tokenizer()
    from transformers import T5EncoderModel
    encoder = T5EncoderModel.from_pretrained(str(MODEL), local_files_only=True, torch_dtype=torch.float16).to(device).eval()
    for p in encoder.parameters(): p.requires_grad_(False)
    vectors, encode_info = encode_unique(all_rows, tokenizer, encoder, device, args.encode_batch_size, args.max_length)
    source_names = sorted({r["generator"] for r in train + dev if r["label"] == 1})
    source_to_id = {name: i for i, name in enumerate(source_names)}
    generators = source_names
    mid = max(1, len(generators) // 2)
    folds = {"fold_0": generators[:mid], "fold_1": generators[mid:]}
    result = {
        "schema": "h3_gpu_diagnostic_batch_v1", "formal_status": "diagnostic_only_formal_gate_revise_data",
        "data_sha256": {p.name: sha256_file(p) for p in (DATA / "train_balanced.jsonl", DATA / "dev_clean.jsonl", DATA / "train_variants.jsonl")},
        "model_path": str(MODEL), "model_sha256": sha256_file(MODEL / "pytorch_model.bin"),
        "device": torch.cuda.get_device_name(0), "cuda": torch.version.cuda, "torch": torch.__version__,
        "protocol": {"test_read": False, "generation": False, "weights_downloaded": False,
                     "max_length": args.max_length, "max_source_chars": MAX_SOURCE_CHARS,
                     "limit_train": args.limit_train, "limit_dev": args.limit_dev,
                     "encoder": "frozen CodeT5-small mean pooling",
                     "folds": folds, "seeds": list(SEEDS), "jobs": args.jobs, "epochs": args.epochs},
        "encode": encode_info, "runs": [],
    }
    for fold_id, heldout in folds.items():
        heldout_set = set(heldout)
        fold_train = [r for r in train if r["generator"] == "human" or r["generator"] not in heldout_set]
        fold_dev = [r for r in dev if r["generator"] == "human" or r["generator"] in heldout_set]
        for job in args.jobs:
            # Source classification is a task-heldout readout over all observed
            # generators.  A generator-heldout source class has no training
            # support by definition, so it is reported as unsupported rather
            # than disguised as a zero score.
            job_train = train if job == "source_only" else fold_train
            job_dev = dev if job == "source_only" else fold_dev
            for seed in SEEDS:
                print(f"RUN {job} {fold_id} seed={seed} train={len(job_train)} dev={len(job_dev)}", flush=True)
                result["runs"].append(run_head(job, seed, fold_id, job_train, job_dev, vectors, source_to_id, variants, device, args.epochs, args.batch_size, 2e-3))
    result["aggregate"] = {}
    for job in args.jobs:
        rows = [r for r in result["runs"] if r["job"] == job]
        result["aggregate"][job] = {"runs": len(rows)}
        detection_rows = [r["detection"] for r in rows if r["detection"].get("auroc") is not None]
        if detection_rows:
            result["aggregate"][job].update({
                "detection_auroc_mean": float(np.mean([r["auroc"] for r in detection_rows])),
                "detection_balanced_accuracy_mean": float(np.mean([r["balanced_accuracy"] for r in detection_rows]))})
        source_rows = [r["source"] for r in rows if r.get("source", {}).get("accuracy") is not None]
        if source_rows:
            result["aggregate"][job]["source_seen_accuracy_mean"] = float(np.mean([r["accuracy"] for r in source_rows]))
    (args.out / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.out / "run_manifest.json").write_text(json.dumps({"schema": result["schema"], "data_sha256": result["data_sha256"], "model_sha256": result["model_sha256"], "jobs": args.jobs, "folds": folds, "seeds": list(SEEDS)}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["aggregate"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
