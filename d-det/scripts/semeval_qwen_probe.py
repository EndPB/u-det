#!/usr/bin/env python
"""Qwen 冻结特征探针（A 任务主口径）：原始底座隐藏态 mean-pool → LR。

与 CodeT5 冻结探针（s1 .6614 / AUC .8213）同口径对比；评测含：
  - prior22：test 分数 0.78 分位为阈值（先验匹配协议，机器占比 22%）
  - val_thr：val 上搜最优阈值 → test 应用
用法：python scripts/semeval_qwen_probe.py --task a [--weights none|runs/*/best.pt] --tag raw
输出：runs/semeval_qwen_probe/{task}_{tag}_{split}.npz（probs）+ probe.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from encoders import build_encoder  # noqa: E402
from semeval_finetune import SevDataset, pad_collate  # noqa: E402

OUT = ROOT / "runs/semeval_qwen_probe"


def macro_f1(pred, y) -> float:
    f1s = []
    for c in np.unique(y):
        tp = np.sum((pred == c) & (y == c))
        fp = np.sum((pred == c) & (y != c))
        fn = np.sum((pred != c) & (y == c))
        f1s.append(2 * tp / max(2 * tp + fp + fn, 1))
    return float(np.mean(f1s))


@torch.no_grad()
def extract(enc, data_dir: Path, task: str, split: str, device: str):
    ds = SevDataset(str(data_dir / f"{task}_{split}.parquet"))
    dl = DataLoader(ds, batch_size=8, collate_fn=pad_collate, num_workers=0)
    F, Y = [], []
    for b in dl:
        ids = b["input_ids"].to(device)
        mask = b["attention_mask"].to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            h = enc(ids, mask)
        m = mask.unsqueeze(-1).to(h.dtype)
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)
        F.append(pooled.float().cpu())
        Y.extend(b["labels"].tolist())
    return torch.cat(F).numpy(), np.array(Y)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="a")
    ap.add_argument("--config", default="configs/ddet_qwen.yaml")
    ap.add_argument("--data-dir", default="data/processed/semeval_qwen")
    ap.add_argument("--weights", default="none", help="none=原始底座；或某 run 的 best.pt")
    ap.add_argument("--tag", default="raw")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    with open(ROOT / args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    enc_cfg = dict(cfg["encoder"])
    enc_cfg["path"] = str(ROOT / enc_cfg["path"])
    enc = build_encoder(**enc_cfg)
    if str(args.weights).lower() not in ("none", "no", ""):
        st = torch.load(str(ROOT / args.weights), map_location="cpu", weights_only=False)
        sub = {k.replace("dual.encoder.", ""): v for k, v in st["state"].items()
               if k.startswith("dual.encoder.")}
        miss = enc.load_state_dict(sub, strict=False)
        print(f"[probe] 编码器权重 ← {args.weights}（载入 {len(sub)} 键，missing={len(miss.missing_keys)}）",
              flush=True)
    enc.requires_grad_(False)
    enc = enc.to(device).eval()
    tok = AutoTokenizer.from_pretrained(enc_cfg["path"])  # noqa: F841

    data_dir = ROOT / args.data_dir
    Xtr, ytr = extract(enc, data_dir, args.task, "train", device)
    Xva, yva = extract(enc, data_dir, args.task, "val", device)
    Xte, yte = extract(enc, data_dir, args.task, "test", device)
    print(f"[probe] 特征 d={Xtr.shape[1]}（train {len(ytr)} / val {len(yva)} / test {len(yte)}）",
          flush=True)

    clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")
    clf.fit(Xtr, ytr)
    sc_va = clf.predict_proba(Xva)[:, 1]
    sc_te = clf.predict_proba(Xte)[:, 1]

    thr_prior = float(np.quantile(sc_te, 0.78))
    best_t, best_v = 0.5, -1.0
    for t in np.arange(0.05, 0.96, 0.01):
        v = macro_f1((sc_va >= t).astype(int), yva)
        if v > best_v:
            best_v, best_t = v, float(t)
    res = {
        "task": args.task, "tag": args.tag, "weights": args.weights,
        "n_features": int(Xtr.shape[1]),
        "val_f1@0.5": round(macro_f1((sc_va >= 0.5).astype(int), yva), 4),
        "val_thr": round(best_t, 2), "val_f1@thr": round(best_v, 4),
        "test_f1@0.5": round(macro_f1((sc_te >= 0.5).astype(int), yte), 4),
        "test_f1@prior22": round(macro_f1((sc_te >= thr_prior).astype(int), yte), 4),
        "test_f1@val_thr": round(macro_f1((sc_te >= best_t).astype(int), yte), 4),
        "test_auc": round(float(roc_auc_score(yte, sc_te)), 4),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    for split, sc, yy in (("val", sc_va, yva), ("test", sc_te, yte)):
        np.savez_compressed(OUT / f"{args.task}_{args.tag}_{split}.npz",
                            probs=sc.astype("float32"), y=yy)
    jf = OUT / "probe.json"
    old = json.loads(jf.read_text()) if jf.exists() else {}
    old[f"{args.task}_{args.tag}"] = res
    jf.write_text(json.dumps(old, ensure_ascii=False, indent=2))
    print(f"[probe] {args.tag}：{res}", flush=True)
    print("[probe] done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
