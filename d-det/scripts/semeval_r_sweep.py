#!/usr/bin/env python
"""SemEval-B 低秩 s2 秩扫描（冻结 v1.0 特征）——确定最佳 r，供 A/C 复用（用户指令 2026-09-24）。

支线 1（CE-bottleneck，主）：z = A_r(h)（768→r）→ Linear(r→11)，家族 CE 训练（冻结 h）。
    扫描 r ∈ {1,2,4,8,11,16,32} + 直连参考（768→11）。→ 归因读出需要多少维
支线 2（hinge-pairs，设计原案多模型版）：用 B 训练集内四族同模型 base↔instruct 对照
    （Qwen2.5-Coder-1.5B、Yi-Coder-1.5B、Llama-3.1-8B、Granite-3.3-8b）训练 w2_r
    （relu(m−‖z₊−z₋‖₂) 铰链；row0= v1.0 w2，其余 N(0,1e-3) 防死锁）→ LR(z) 做 B 归因。
输出：runs/semeval_r/{sweep.json, ce_r*.pt, hinge_r*.pt, pairs_h.npz}
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from encoders import build_encoder  # noqa: E402
from models import build_model  # noqa: E402
from semeval_finetune import SevDataset, pad_collate  # noqa: E402

FEAT = ROOT / "runs/semeval_zeroshot/feat"
OUT = ROOT / "runs/semeval_r"
RAW = ROOT / "data/raw/SemEval-2026-Task13/task_b/task_b_training_set.parquet"

PAIR_MODELS = {   # family -> (instruct 侧型号, base 侧型号)
    "qwen": ("Qwen/Qwen2.5-Coder-1.5B-Instruct", "Qwen/Qwen2.5-Coder-1.5B"),
    "yi": ("01-ai/Yi-Coder-1.5B-Chat", "01-ai/Yi-Coder-1.5B"),
    "llama": ("meta-llama/Llama-3.1-8B-Instruct", "meta-llama/Llama-3.1-8B"),
    "granite": ("ibm-granite/granite-3.3-8b-instruct", "ibm-granite/granite-3.3-8b-base"),
}
RS_CE = (1, 2, 4, 8, 11, 16, 32)
RS_HG = (1, 2, 4, 8, 16, 32)


@torch.no_grad()
def encode_ids(model, ids_list, device, batch=8):
    hs = []
    for i in range(0, len(ids_list), batch):
        chunk = ids_list[i:i + batch]
        L = max(len(x) for x in chunk)
        ids = torch.zeros(len(chunk), L, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for r, x in enumerate(chunk):
            ids[r, :len(x)] = torch.tensor(x, dtype=torch.long)
            mask[r, :len(x)] = 1
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            _, _, h = model.forward_feat(ids.to(device), mask.to(device))
        hs.append(h.float().cpu())
    return torch.cat(hs).numpy()


def load_split(task: str, split: str, model, device, tok):
    cache = FEAT / f"{task}_{split}.npz"
    if cache.exists():
        d = np.load(cache, allow_pickle=True)
        return d["h"].astype("float32"), d["y"], list(d["lang"])
    ds = SevDataset(str(ROOT / f"data/processed/semeval/{task}_{split}.parquet"))
    dl = DataLoader(ds, batch_size=8, collate_fn=pad_collate, num_workers=0)
    H, Y = [], []
    for b in dl:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            _, _, h = model.forward_feat(b["input_ids"].to(device), b["attention_mask"].to(device))
        H.append(h.float().cpu())
        Y.extend(b["labels"].tolist())
    h = torch.cat(H).numpy().astype("float32")
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, h=h, s1=np.zeros(len(h)), s2=np.zeros((len(h), 1)),
                        y=np.array(Y), lang=np.array(ds.langs, dtype=object))  # s1/s2 占位
    return h, np.array(Y), ds.langs


def extract_pairs(model, device, tok, cap=1200):
    cache = OUT / "pairs_h.npz"
    if cache.exists():
        d = np.load(cache, allow_pickle=True)
        return {k: d[k] for k in d.files}
    gens = [g for pair in PAIR_MODELS.values() for g in pair]
    want = {g: [] for g in gens}
    pf = pq.ParquetFile(RAW)
    for batch in pf.iter_batches(batch_size=8192, columns=["code", "generator"]):
        codes = batch.column("code").to_pylist()
        gs = batch.column("generator").to_pylist()
        for c, g in zip(codes, gs):
            if g in want and len(want[g]) < cap:
                want[g].append(c)
        if all(len(v) >= cap for v in want.values()):
            break
    hs = {}
    for g in gens:
        ids_list = []
        for c in want[g]:
            ids = tok(c, add_special_tokens=False)["input_ids"]
            if len(ids) > 1024:
                ids = ids[:768] + ids[-256:]
            ids_list.append(ids)
        hs[g] = encode_ids(model, ids_list, device)
        print(f"[pairs] {g}: {len(ids_list)} rows", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **hs)
    return hs


def train_ce_bottleneck(H_tr, y_tr, H_va, y_va, H_te, y_te, r, epochs=80, seed=0):
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Xtr = torch.tensor(H_tr, device=dev)
    ytr = torch.tensor(y_tr, device=dev)
    Xva = torch.tensor(H_va, device=dev)
    Xte = torch.tensor(H_te, device=dev)
    A = nn.Linear(H_tr.shape[1], r, bias=False).to(dev)
    cls = nn.Linear(r, 11).to(dev)
    opt = torch.optim.Adam(list(A.parameters()) + list(cls.parameters()), lr=0.01, weight_decay=1e-5)
    n = len(H_tr)
    best = (-1, None, None)
    for ep in range(epochs):
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, 1024):
            idx = perm[i:i + 1024]
            loss = nn.functional.cross_entropy(cls(A(Xtr[idx])), ytr[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        if (ep + 1) % 5 == 0 or ep == epochs - 1:
            with torch.no_grad():
                f1v = f1_score(y_va, cls(A(Xva)).argmax(1).cpu().numpy(), average="macro")
            if f1v > best[0]:
                best = (f1v, {k: v.clone() for k, v in A.state_dict().items()},
                        {k: v.clone() for k, v in cls.state_dict().items()})
    with torch.no_grad():
        A.load_state_dict(best[1]); cls.load_state_dict(best[2])
        f1t = f1_score(y_te, cls(A(Xte)).argmax(1).cpu().numpy(), average="macro")
        accv = float((cls(A(Xva)).argmax(1).cpu().numpy() == y_va).mean())
    return {"val_f1": round(float(best[0]), 4), "test_f1": round(float(f1t), 4),
            "val_acc": round(accv, 4)}, A.weight.detach().cpu().numpy()


def train_hinge(H_list_p, H_list_m, r, w2_init, m_mult=1.2, steps=1500, seed=0, batch=256):
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    fams = [(torch.tensor(hp.astype("float32"), device=dev),
             torch.tensor(hm.astype("float32"), device=dev))
            for hp, hm in zip(H_list_p, H_list_m)]
    dim = fams[0][0].shape[1]
    A = nn.Linear(dim, r, bias=False).to(dev)
    with torch.no_grad():
        A.weight.zero_()
        A.weight[0].copy_(torch.tensor(w2_init, device=dev))
        if r > 1:
            A.weight[1:].normal_(0, 1e-3)

    def pair_norms(limit=None):
        ds = []
        for hp_t, hm_t in fams:
            k = min(len(hp_t), len(hm_t)) if limit is None else min(len(hp_t), len(hm_t), limit)
            ds.append((A(hp_t[:k]) - A(hm_t[:k])).norm(dim=1))
        return torch.cat(ds)

    with torch.no_grad():
        d0 = pair_norms(limit=500)
        m = float(max(0.2, m_mult * d0.median().item()))
    opt = torch.optim.Adam(A.parameters(), lr=1e-3)
    for step in range(steps):
        fi = torch.randint(0, len(fams), (1,), device=dev).item()
        hp_t, hm_t = fams[fi]
        ip = torch.randint(0, len(hp_t), (batch,), device=dev)
        im = torch.randint(0, len(hm_t), (batch,), device=dev)
        d = (A(hp_t[ip]) - A(hm_t[im])).norm(dim=1)
        loss = torch.relu(m - d).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        d = pair_norms()
        sat = float((d >= m).float().mean())
    return A.weight.detach().cpu().numpy(), {"margin": round(m, 4), "satisfy_rate": round(sat, 4),
                                             "d_median": round(float(d.median()), 4)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=1200)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(ROOT / "configs/ddet_v041.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    enc_cfg = dict(cfg["encoder"])
    enc_cfg["path"] = str(ROOT / enc_cfg["path"])
    enc = build_encoder(**enc_cfg)
    dual = build_model("dual", encoder=enc, dim=enc.hidden_size,
                       pool=cfg["model"].get("pool", "mean"),
                       s2_rank=cfg["model"].get("s2_rank", 1))
    ck = torch.load(str(ROOT / "runs/v0.4.1_covreg/last.pt"), map_location="cpu", weights_only=False)
    dual.load_state_dict(ck["state"], strict=False)
    dual = dual.to(device).eval()
    for p in dual.parameters():
        p.requires_grad_(False)
    tok = AutoTokenizer.from_pretrained(enc_cfg["path"])
    w2_init = dual.w2.weight.detach().cpu().numpy().reshape(-1)

    # ---------- 特征 ----------
    h_tr, y_tr, _ = load_split("b", "train", dual, device, tok)
    h_va, y_va, _ = load_split("b", "val", dual, device, tok)
    h_te, y_te, _ = load_split("b", "test", dual, device, tok)
    mu, sd = h_tr.mean(0), h_tr.std(0) + 1e-6
    Htr, Hva, Hte = [(x - mu) / sd for x in (h_tr, h_va, h_te)]   # CE 支线用标准化空间
    print(f"[feat] train={Htr.shape} val={Hva.shape} test={Hte.shape}", flush=True)

    res = {"ce": {}, "hinge": {}, "n_pair_cap": args.cap}
    # ---------- 支线1：CE-bottleneck ----------
    for r in RS_CE:
        m_, W = train_ce_bottleneck(Htr, y_tr, Hva, y_va, Hte, y_te, r)
        res["ce"][str(r)] = m_
        OUT.mkdir(parents=True, exist_ok=True)
        np.save(OUT / f"ce_r{r}.npy", W)
        print(f"[CE] r={r}: {m_}", flush=True)
    # 直连参考（768→11）
    m_, _ = train_ce_bottleneck(Htr, y_tr, Hva, y_va, Hte, y_te, 768)
    res["ce"]["plain768"] = m_
    print(f"[CE] plain768: {m_}", flush=True)

    # ---------- 支线2：hinge-pairs（raw h 空间） ----------
    hs = extract_pairs(dual, device, tok, cap=args.cap)
    Hp_l, Hm_l = [], []
    for fam, (g_ins, g_base) in PAIR_MODELS.items():
        Hp_l.append(hs[g_ins]); Hm_l.append(hs[g_base])
    n_pairs = min(len(x) for x in Hp_l)
    res["hinge_n_per_family"] = n_pairs
    print(f"[hinge] per-family pairs={n_pairs}", flush=True)
    for r in RS_HG:
        W, info = train_hinge(Hp_l, Hm_l, r, w2_init)
        z_tr = Htr @ W.T
        z_va = Hva @ W.T
        z_te = Hte @ W.T
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        clf.fit(z_tr, y_tr)
        f1v = f1_score(y_va, clf.predict(z_va), average="macro")
        f1t = f1_score(y_te, clf.predict(z_te), average="macro")
        res["hinge"][str(r)] = {**info, "val_f1": round(float(f1v), 4), "test_f1": round(float(f1t), 4)}
        np.save(OUT / f"hinge_r{r}.npy", W)
        print(f"[hinge] r={r}: {res['hinge'][str(r)]}", flush=True)

    with open(OUT / "sweep.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"[out] {OUT / 'sweep.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
