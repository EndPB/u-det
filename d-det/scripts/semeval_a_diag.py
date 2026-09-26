#!/usr/bin/env python
"""A 臂塌陷诊断：AUC / 翻转假设 / R_void 融合（val 拟合 → test 评测）。

背景：A 臂 test macro-F1 0.3955（@0.5 0.4463）vs 单特征 R_void=0.6688。
本脚本回答：
  1) 模型在 test 上是"没信号"（AUC≈0.5）还是"信号反向"（AUC<0.5，Yuvan 翻转现象）；
  2) [logit(p), R_void] 线性融合（val 拟合、test 评测）能否补救（对照 UIT_AMMC 的融合思路）；
  3) 顺带核对 last / best 检查点差异。

输出：runs/semeval_a/diag.json + diag.npz（probs/R_void 缓存，便于复算）
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from encoders import build_encoder  # noqa: E402
from models import build_model  # noqa: E402
from semeval_finetune import SevDataset, TaskModel, collect_preds, pad_collate  # noqa: E402

RAW = ROOT / "data/raw/SemEval-2026-Task13"
OUT = ROOT / "runs/semeval_a"


def rvoid(code: str) -> float:
    lines = code.split("\n")
    return sum(1 for ln in lines if not ln.strip()) / max(len(lines), 1)


def replay_val_idx() -> list[int]:
    """复现 prepare 的 val 抽样（seed=0；先重放 train 抽取以对齐 rng 状态）。"""
    lab_tr = np.asarray(pq.read_table(RAW / "task_a/task_a_training_set_1.parquet",
                                      columns=["label"]).column("label").to_pylist())
    lab_va = np.asarray(pq.read_table(RAW / "task_a/task_a_validation_set.parquet",
                                      columns=["label"]).column("label").to_pylist())
    rng = random.Random(0)
    for lab, q in ((0, 16000), (1, 16000)):            # 重放 train 抽取
        pool = np.where(lab_tr == lab)[0].tolist()
        rng.sample(pool, min(q, len(pool)))
    idx: list[int] = []
    for lab, q in ((0, 6000), (1, 6000)):              # 记录 val 抽取
        pool = np.where(lab_va == lab)[0].tolist()
        idx += rng.sample(pool, min(q, len(pool)))
    return sorted(idx)


def rvoid_for_rows(path: Path, idx: list[int], tok) -> np.ndarray:
    """按 encode_rows 的规则对齐：升序遍历、token 化后 <8 的丢弃。"""
    codes = pq.read_table(path, columns=["code"]).column("code").to_pylist()
    vals = []
    for i in idx:
        c = codes[i]
        if len(tok(c, add_special_tokens=False)["input_ids"]) < 8:
            continue
        vals.append(rvoid(c))
    return np.asarray(vals)


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def best_thr(y, p, flip=False):
    best = (0.5, -1.0)
    for t in np.arange(0.05, 0.96, 0.01):
        pred = (p >= t).astype(int)
        if flip:
            pred = 1 - pred
        f1 = f1_score(y, pred, average="macro")
        if f1 > best[1]:
            best = (float(t), float(f1))
    return best


def fit_lr(X, y, X_te):
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    clf.fit(X, y)
    return clf.predict(X_te)


def main() -> int:
    import yaml

    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp, amp_dtype = device == "cuda", "cuda"
    with open(ROOT / "configs/ddet_v041.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    enc_cfg = dict(cfg["encoder"])
    enc_cfg["path"] = str(ROOT / enc_cfg["path"])
    enc = build_encoder(**enc_cfg)
    dual = build_model("dual", encoder=enc, dim=enc.hidden_size,
                       pool=cfg["model"].get("pool", "mean"),
                       s2_rank=cfg["model"].get("s2_rank", 1))
    model = TaskModel(dual, "a", 2).to(device)
    tok = AutoTokenizer.from_pretrained(enc_cfg["path"])

    # ---------- R_void（对齐到我们的评测子集） ----------
    te_path = ROOT / "data/processed/semeval/a_test.parquet"
    va_path = ROOT / "data/processed/semeval/a_val.parquet"
    y_te = np.asarray(pq.read_table(te_path, columns=["label"]).column("label").to_pylist())
    y_va = np.asarray(pq.read_table(va_path, columns=["label"]).column("label").to_pylist())
    rv_te = rvoid_for_rows(RAW / "task_a/task_a_test_set_sample.parquet",
                           list(range(1000)), tok)
    rv_va = rvoid_for_rows(RAW / "task_a/task_a_validation_set.parquet",
                           replay_val_idx(), tok)
    assert len(rv_te) == len(y_te) == 1000, (len(rv_te), len(y_te))
    assert len(rv_va) == len(y_va), (len(rv_va), len(y_va))
    print(f"[rvoid] test n={len(rv_te)} val n={len(rv_va)}", flush=True)

    loader_va = DataLoader(SevDataset(str(va_path)), batch_size=8,
                           collate_fn=pad_collate, num_workers=0)
    loader_te = DataLoader(SevDataset(str(te_path)), batch_size=8,
                           collate_fn=pad_collate, num_workers=0)

    diag = {"rvoid_test_auc": round(float(roc_auc_score(y_te, rv_te)), 4)}
    for name in ("last", "best"):
        ck = torch.load(str(OUT / f"{name}.pt"), map_location="cpu", weights_only=False)
        miss, unexp = model.load_state_dict(ck["state"], strict=False)
        print(f"[ckpt] {name} (epoch {ck.get('epoch')}) missing={len(miss)} unexpected={len(unexp)}", flush=True)
        model.eval()
        p_te, _, _ = collect_preds(model, loader_te, device, amp, amp_dtype)
        p_va, _, _ = collect_preds(model, loader_va, device, amp, amp_dtype)
        lg_te, lg_va = logit(p_te), logit(p_va)

        f_at05 = float(f1_score(y_te, (p_te >= 0.5).astype(int), average="macro"))
        f_flip05 = float(f1_score(y_te, 1 - (p_te >= 0.5).astype(int), average="macro"))
        thr_star, f_star = best_thr(y_te, p_te)
        thr_flip, f_flip = best_thr(y_te, p_te, flip=True)
        entry = {
            "auc_test": round(float(roc_auc_score(y_te, p_te)), 4),
            "f1@0.5": round(f_at05, 4), "f1_flip@0.5": round(f_flip05, 4),
            "f1_best_thr": round(f_star, 4), "best_thr": round(thr_star, 2),
            "f1_flip_best_thr": round(f_flip, 4), "flip_best_thr": round(thr_flip, 2),
        }
        # 融合（val 拟合 → test 评测）
        entry["lr_rvoid(val)"] = round(float(f1_score(
            y_te, fit_lr(rv_va[:, None], y_va, rv_te[:, None]), average="macro")), 4)
        entry["lr_logit(val)"] = round(float(f1_score(
            y_te, fit_lr(lg_va[:, None], y_va, lg_te[:, None]), average="macro")), 4)
        entry["lr_logit+rvoid(val)"] = round(float(f1_score(
            y_te, fit_lr(np.stack([lg_va, rv_va], 1), y_va,
                         np.stack([lg_te, rv_te], 1)), average="macro")), 4)
        diag[name] = entry
        print(f"[{name}] {json.dumps(entry, ensure_ascii=False)}", flush=True)

        if name == "last":
            np.savez_compressed(OUT / "diag.npz", p_te=p_te, p_va=p_va,
                                rv_te=rv_te, rv_va=rv_va, y_te=y_te, y_va=y_va)

    # ---------- 冻结 s1（v1.0 零样本）与 R_void 的融合（特征取自探针缓存） ----------
    zva_p = ROOT / "runs/semeval_zeroshot/feat/a_val.npz"
    zte_p = ROOT / "runs/semeval_zeroshot/feat/a_test.npz"
    if zva_p.exists() and zte_p.exists():
        zva = np.load(zva_p, allow_pickle=True)
        zte = np.load(zte_p, allow_pickle=True)
        s1_va, s1_te = zva["s1"].reshape(-1), zte["s1"].reshape(-1)
        assert np.array_equal(zva["y"], y_va) and np.array_equal(zte["y"], y_te)
        frozen = {
            "auc_s1_frozen(test)": round(float(roc_auc_score(y_te, s1_te)), 4),
            "lr_s1(val)": round(float(f1_score(y_te, fit_lr(
                s1_va[:, None], y_va, s1_te[:, None]), average="macro")), 4),
            "lr_s1+rvoid(val)": round(float(f1_score(y_te, fit_lr(
                np.stack([s1_va, rv_va], 1), y_va,
                np.stack([s1_te, rv_te], 1)), average="macro")), 4),
        }
        diag["frozen_fusion"] = frozen
        print(f"[frozen] {json.dumps(frozen, ensure_ascii=False)}", flush=True)

    with open(OUT / "diag.json", "w", encoding="utf-8") as f:
        json.dump(diag, f, ensure_ascii=False, indent=2)
    print(f"[out] {OUT / 'diag.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
