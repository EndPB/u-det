#!/usr/bin/env python
"""SemEval 对比臂：冻结 v1.0 特征的零样本 / 线性探针（不做任何骨干微调）。

协议（全部在官方 train/val 内部拟合，test 只评测一次）：
  A 二分类：
    s1@0.5            —— v1.0 的 s1 头直接打分的零样本
    s1(thr*)          —— 阈值在 val 上校准后用于 test
    [s1,s2]-LR(val)   —— 双轴融合线性头（在 val 上拟合，对应冻结协议的 fused）
    h-LR(train)       —— 512/768 维池化特征的线性探针（冻结特征臂）
    [h,s1,s2]-LR(train)
  B/C 多分类：
    h-LR(train)、[h,s1,s2]-LR(train)
输出：runs/semeval_zeroshot/{task}.json + 特征缓存 feat/{task}_{split}.npz

用法：
    python scripts/semeval_zeroshot.py            # 全部三个任务
    python scripts/semeval_zeroshot.py --tasks a  # 单任务
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from encoders import build_encoder  # noqa: E402
from models import build_model  # noqa: E402
from semeval_finetune import SevDataset, pad_collate  # noqa: E402

OUT = ROOT / "runs/semeval_zeroshot"


@torch.no_grad()
def extract(dual, ds, device, batch: int = 8):
    loader = DataLoader(ds, batch_size=batch, collate_fn=pad_collate, num_workers=0)
    H, S1, S2, Y = [], [], [], []
    for b in tqdm(loader, desc="extract", leave=False):
        ids = b["input_ids"].to(device)
        mask = b["attention_mask"].to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            s1, s2, h = dual.forward_feat(ids, mask)
        H.append(h.float().cpu().numpy())
        S1.append(s1.float().cpu().numpy().reshape(-1))
        S2.append(s2.float().cpu().numpy().reshape(len(ids), -1))
        Y.extend(b["labels"].tolist())
    return (np.concatenate(H), np.concatenate(S1), np.concatenate(S2), np.array(Y))


def load_split(task: str, split: str, device, dual, batch: int, use_cache: bool = True):
    cache = OUT / f"feat/{task}_{split}.npz"
    if use_cache and cache.exists():
        d = np.load(cache, allow_pickle=True)
        return d["h"], d["s1"], d["s2"], d["y"], list(d["lang"])
    ds = SevDataset(str(ROOT / f"data/processed/semeval/{task}_{split}.parquet"))
    h, s1, s2, y = extract(dual, ds, device, batch)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, h=h, s1=s1, s2=s2, y=y, lang=np.array(ds.langs, dtype=object))
    return h, s1, s2, y, ds.langs


def cls_report(task: str, y, pred, p_pos=None) -> dict:
    rep = {"macro_f1": round(float(f1_score(y, pred, average="macro")), 4),
           "acc": round(float((np.asarray(pred) == y).mean()), 4),
           "n": int(len(y))}
    if p_pos is not None and task == "a":
        rep["auc"] = round(float(roc_auc_score(y, p_pos)), 4)
    per = {}
    for c in sorted(set(y.tolist())):
        tp = int(((pred == c) & (y == c)).sum())
        fp = int(((pred == c) & (y != c)).sum())
        fn = int(((pred != c) & (y == c)).sum())
        f1 = 2 * tp / max(2 * tp + fp + fn, 1e-9)
        per[str(c)] = {"f1": round(f1, 4), "n": int((y == c).sum())}
    rep["per_class"] = per
    return rep


def by_lang(task, y, pred, langs, p_pos=None) -> dict:
    out = {}
    langs = np.array(langs, dtype=object)
    for lg in sorted(set(langs.tolist())):
        m = langs == lg
        if m.sum() == 0:
            continue
        r = cls_report(task, y[m], np.asarray(pred)[m],
                       None if p_pos is None else np.asarray(p_pos)[m])
        out[str(lg)] = r
    return out


def lr_fit_predict(X_tr, y_tr, X_te, multi: bool):
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=1500, C=1.0))
    clf.fit(X_tr, y_tr)
    if multi:
        return clf.predict(X_te)
    return clf.predict(X_te), clf.predict_proba(X_te)[:, 1]


def tune_thr(probs, y) -> float:
    best_t, best_f1 = 0.5, -1.0
    for t in np.arange(0.05, 0.96, 0.01):
        f1 = f1_score(y, (probs >= t).astype(int), average="macro")
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t


def run_task_a(dual, device, batch: int) -> dict:
    ftr = load_split("a", "train", device, dual, batch)
    fva = load_split("a", "val", device, dual, batch)
    fte = load_split("a", "test", device, dual, batch)
    h_tr, s1_tr, s2_tr, y_tr, _ = ftr
    h_va, s1_va, s2_va, y_va, _ = fva
    h_te, s1_te, s2_te, y_te, lg_te = fte

    res = {}

    # 1) 原始 s1@0.5（零样本，不打磨）
    pred = (s1_te >= 0.0).astype(int)  # s1 是 logit，0 即概率 0.5
    res["s1@logit0"] = cls_report("a", y_te, pred)

    # 2) s1 + val 阈值校准（概率化）
    p_va = 1.0 / (1.0 + np.exp(-s1_va))
    p_te = 1.0 / (1.0 + np.exp(-s1_te))
    thr = tune_thr(p_va, y_va)
    pred = (p_te >= thr).astype(int)
    r = cls_report("a", y_te, pred, p_pos=p_te)
    r["thr"] = round(thr, 2)
    r["by_language"] = by_lang("a", y_te, pred, lg_te, p_pos=p_te)
    res["s1(thr*)"] = r

    # 3) [s1,s2] LR 在 val 上拟合（冻结双轴融合）
    X_va = np.stack([s1_va, s2_va[:, 0]], 1)
    X_te = np.stack([s1_te, s2_te[:, 0]], 1)
    pred, p = lr_fit_predict(X_va, y_va, X_te, multi=False)
    r = cls_report("a", y_te, pred, p_pos=p)
    r["by_language"] = by_lang("a", y_te, pred, lg_te, p_pos=p)
    res["[s1,s2]-LR(val)"] = r

    # 4) h-LR 在 train 上拟合（冻结特征臂）
    pred, p = lr_fit_predict(h_tr, y_tr, h_te, multi=False)
    r = cls_report("a", y_te, pred, p_pos=p)
    r["by_language"] = by_lang("a", y_te, pred, lg_te, p_pos=p)
    res["h-LR(train)"] = r

    # 5) [h,s1,s2] LR
    X_tr = np.concatenate([h_tr, s1_tr[:, None], s2_tr], 1)
    X_te = np.concatenate([h_te, s1_te[:, None], s2_te], 1)
    pred, p = lr_fit_predict(X_tr, y_tr, X_te, multi=False)
    r = cls_report("a", y_te, pred, p_pos=p)
    r["by_language"] = by_lang("a", y_te, pred, lg_te, p_pos=p)
    res["[h,s1,s2]-LR(train)"] = r
    return res


def run_task_mc(task: str, dual, device, batch: int) -> dict:
    ftr = load_split(task, "train", device, dual, batch)
    fte = load_split(task, "test", device, dual, batch)
    h_tr, s1_tr, s2_tr, y_tr, _ = ftr
    h_te, s1_te, s2_te, y_te, lg_te = fte

    res = {}
    pred = lr_fit_predict(h_tr, y_tr, h_te, multi=True)
    r = cls_report(task, y_te, pred)
    r["by_language"] = by_lang(task, y_te, pred, lg_te)
    res["h-LR(train)"] = r

    X_tr = np.concatenate([h_tr, s1_tr[:, None], s2_tr], 1)
    X_te = np.concatenate([h_te, s1_te[:, None], s2_te], 1)
    pred = lr_fit_predict(X_tr, y_tr, X_te, multi=True)
    r = cls_report(task, y_te, pred)
    r["by_language"] = by_lang(task, y_te, pred, lg_te)
    res["[h,s1,s2]-LR(train)"] = r
    return res


def main() -> int:
    import yaml

    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="a,b,c")
    ap.add_argument("--init", default="runs/v0.4.1_covreg/last.pt")
    ap.add_argument("--config", default="configs/ddet_v041.yaml")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    with open(ROOT / args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    enc_cfg = dict(cfg["encoder"])
    enc_cfg["path"] = str(ROOT / enc_cfg["path"])
    enc = build_encoder(**enc_cfg)
    dual = build_model("dual", encoder=enc, dim=enc.hidden_size,
                       pool=cfg["model"].get("pool", "mean"),
                       s2_rank=cfg["model"].get("s2_rank", 1))
    ck = torch.load(str(ROOT / args.init), map_location="cpu", weights_only=False)
    missing, unexpected = dual.load_state_dict(ck.get("state", {}), strict=False)
    print(f"[init] {args.init}（epoch {ck.get('epoch')}）missing={len(missing)} unexpected={len(unexpected)}")
    dual = dual.to(device).eval()

    tok = AutoTokenizer.from_pretrained(enc_cfg["path"])  # noqa: F841（SevDataset 已含 ids）

    results = {}
    for task in [t.strip() for t in args.tasks.split(",") if t.strip()]:
        print(f"\n===== task {task}（冻结探针）=====")
        if task == "a":
            results["a"] = run_task_a(dual, device, args.batch)
        else:
            results[task] = run_task_mc(task, dual, device, args.batch)
        for k, v in results[task].items():
            print(f"  {k}: macro_f1={v['macro_f1']:.4f} acc={v['acc']:.4f}"
                  + (f" auc={v['auc']:.4f}" if "auc" in v else ""))

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "zeroshot.json", "w", encoding="utf-8") as f:
        json.dump({"init": args.init, "results": results}, f,
                  ensure_ascii=False, indent=2)
    print(f"\n[out] {OUT / 'zeroshot.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
