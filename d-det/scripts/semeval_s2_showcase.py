#!/usr/bin/env python
"""s2 家族能力展示与迭代（r=8 固定）+ A/C 子空间（用户指令 2026-09-25）。

Part A 过拟合缺口（回答"r 变大为何下降"）：r∈{8,11,16,32,768}×wd 网格的 train/val/test F1。
Part B r=8 冻结 B 子空间迭代：5 配置（ep/wd/类权/特征drop）选 val 最优 → 3 种子集成；
        输出 z8 + 子空间 probs（供堆叠）+ 独立评测（per-class）。
Part C B 微调编码器特征上的 r=8 子空间（同配置）；输出 z8_ft + probs_ft。
Part D A/C 的 r=8 子空间（2/4 类 CE）；输出各自 z8 + 评测。
输出：runs/semeval_r/showcase.json、runs/semeval_r/z8/{task}_{split}.npz
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from encoders import build_encoder  # noqa: E402
from models import build_model  # noqa: E402
from semeval_finetune import SevDataset, pad_collate  # noqa: E402

FEAT = ROOT / "runs/semeval_zeroshot/feat"
OUT = ROOT / "runs/semeval_r"
ZDIR = OUT / "z8"
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def fit_ce(Htr, ytr, Hva, yva, Hte, yte, r, epochs, wd, cw=False, drop=0.0, seed=0):
    torch.manual_seed(seed)
    n_cls = int(max(ytr.max(), yva.max(), yte.max())) + 1
    X = torch.tensor(Htr, device=DEV); y = torch.tensor(ytr, device=DEV)
    Xva = torch.tensor(Hva, device=DEV); Xte = torch.tensor(Hte, device=DEV)
    A = nn.Linear(Htr.shape[1], r, bias=False).to(DEV)
    C = nn.Linear(r, n_cls).to(DEV)
    opt = torch.optim.Adam(list(A.parameters()) + list(C.parameters()), lr=0.01, weight_decay=wd)
    w = None
    if cw:
        cnt = np.bincount(ytr, minlength=n_cls).astype(float)
        ww = 1.0 / np.sqrt(np.maximum(cnt, 1)); ww /= ww.mean()
        w = torch.tensor(ww, dtype=torch.float32, device=DEV)
    best = (-1.0, None)
    for ep in range(epochs):
        perm = torch.randperm(len(X), device=DEV)
        for i in range(0, len(X), 1024):
            idx = perm[i:i + 1024]
            hb = X[idx]
            if drop > 0:
                hb = hb * (torch.rand_like(hb) > drop)
            loss = torch.nn.functional.cross_entropy(C(A(hb)), y[idx], weight=w)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if (ep + 1) % 10 == 0 or ep == epochs - 1:
            with torch.no_grad():
                f1v = f1_score(yva, C(A(Xva)).argmax(1).cpu().numpy(), average="macro")
            if f1v > best[0]:
                best = (f1v, (A.weight.detach().clone(), C.weight.detach().clone(),
                               C.bias.detach().clone()))
    Aw, Cw, Cb = best[1]
    with torch.no_grad():
        A.weight.copy_(Aw); C.weight.copy_(Cw); C.bias.copy_(Cb)
        tr_f1 = f1_score(ytr, C(A(X)).argmax(1).cpu().numpy(), average="macro")
        va_p = torch.softmax(C(A(Xva)), 1).cpu().numpy()
        te_p = torch.softmax(C(A(Xte)), 1).cpu().numpy()
        z_tr, z_va, z_te = A(X).cpu().numpy(), A(Xva).cpu().numpy(), A(Xte).cpu().numpy()
    return {"tr_f1": float(tr_f1), "val_f1": float(best[0]),
            "test_f1": float(f1_score(yte, te_p.argmax(1), average="macro")),
            "val_p": va_p, "te_p": te_p, "z": (z_tr, z_va, z_te), "A": Aw.cpu().numpy()}


def std_split(h_tr, *others):
    mu, sd = h_tr.mean(0), h_tr.std(0) + 1e-6
    return [((x - mu) / sd).astype("float32") for x in (h_tr, *others)]


def load_split(task, split):
    d = np.load(FEAT / f"{task}_{split}.npz", allow_pickle=True)
    return d["h"].astype("float32"), d["y"]


@torch.no_grad()
def extract_ft(model_dual, task, split):
    ds = SevDataset(str(ROOT / f"data/processed/semeval/{task}_{split}.parquet"))
    dl = DataLoader(ds, batch_size=8, collate_fn=pad_collate, num_workers=0)
    H = []
    for b in tqdm(dl, desc=f"ftfeat {task}_{split}", unit="it"):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEV == "cuda"):
            _, _, h = model_dual.forward_feat(b["input_ids"].to(DEV), b["attention_mask"].to(DEV))
        H.append(h.float().cpu())
    return torch.cat(H).numpy().astype("float32")


def save_z(task, split, z, p):
    ZDIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(ZDIR / f"{task}_{split}.npz", z8=z, probs=p)


def main() -> int:
    res = {}
    ZDIR.mkdir(parents=True, exist_ok=True)

    # ---------- 冻结 B 特征 ----------
    hb_tr, yb_tr = load_split("b", "train")
    hb_va, yb_va = load_split("b", "val")
    hb_te, yb_te = load_split("b", "test")
    Hb_tr, Hb_va, Hb_te = std_split(hb_tr, hb_va, hb_te)

    # Part A：过拟合缺口
    gap = []
    for r, wd in [(8, 1e-5), (11, 1e-5), (16, 1e-5), (32, 1e-5), (768, 1e-5),
                  (16, 1e-3), (32, 1e-3), (768, 1e-3)]:
        m = fit_ce(Hb_tr, yb_tr, Hb_va, yb_va, Hb_te, yb_te, r, 80, wd)
        row = {"r": r, "wd": wd, "train": round(m["tr_f1"], 4),
               "val": round(m["val_f1"], 4), "test": round(m["test_f1"], 4),
               "gap": round(m["tr_f1"] - m["val_f1"], 4)}
        gap.append(row)
        print(f"[A] {row}", flush=True)
    res["gap"] = gap

    # Part B：r=8 网格 + 集成
    cfgs = [(80, 1e-5, False, 0.0), (200, 1e-5, False, 0.0), (200, 1e-3, False, 0.0),
            (200, 1e-3, True, 0.0), (200, 3e-3, True, 0.1)]
    grid, best = [], None
    for (ep, wd, cw, dr) in cfgs:
        m = fit_ce(Hb_tr, yb_tr, Hb_va, yb_va, Hb_te, yb_te, 8, ep, wd, cw, dr)
        row = {"ep": ep, "wd": wd, "cw": cw, "drop": dr, "train": round(m["tr_f1"], 4),
               "val": round(m["val_f1"], 4), "test": round(m["test_f1"], 4)}
        grid.append(row)
        print(f"[B] {row}", flush=True)
        if best is None or m["val_f1"] > best[0]["val_f1"]:
            best = (m, (ep, wd, cw, dr))
    ep, wd, cw, dr = best[1]
    ms = [fit_ce(Hb_tr, yb_tr, Hb_va, yb_va, Hb_te, yb_te, 8, ep, wd, cw, dr, seed=s)
          for s in (0, 1, 2)]
    z_tr = np.mean([m["z"][0] for m in ms], 0)
    z_va = np.mean([m["z"][1] for m in ms], 0)
    z_te = np.mean([m["z"][2] for m in ms], 0)
    p_va = np.mean([m["val_p"] for m in ms], 0)
    p_te = np.mean([m["te_p"] for m in ms], 0)
    per_class = f1_score(yb_va, p_va.argmax(1), average=None).round(4).tolist()
    res["r8_best_cfg"] = {"ep": ep, "wd": wd, "cw": cw, "drop": dr}
    res["r8_grid"] = grid
    res["r8_ensemble"] = {"val_f1": round(float(f1_score(yb_va, p_va.argmax(1), average="macro")), 4),
                          "test_f1": round(float(f1_score(yb_te, p_te.argmax(1), average="macro")), 4),
                          "val_per_class": per_class}
    print(f"[B] ensemble: {res['r8_ensemble']}", flush=True)
    for split, z, p in (("train", z_tr, None), ("val", z_va, p_va), ("test", z_te, p_te)):
        if p is None:
            p = np.zeros((len(z), 11), dtype="float32")
        save_z("b", split, z.astype("float32"), p.astype("float32"))

    # Part C：B 微调编码器特征上的 r=8 子空间
    enc2 = build_encoder(**enc_cfg_holder)
    dual2 = build_model("dual", encoder=enc2, dim=enc2.hidden_size,
                        pool=_cfg["model"].get("pool", "mean"),
                        s2_rank=_cfg["model"].get("s2_rank", 1))
    st = torch.load(str(ROOT / "runs/semeval_b/last.pt"), map_location="cpu", weights_only=False)
    dual2.load_state_dict(st["state"], strict=False)
    dual2 = dual2.to(DEV).eval()
    ft = {}
    for split, y in (("train", yb_tr), ("val", yb_va), ("test", yb_te)):
        ft[split] = extract_ft(dual2, "b", split)
    Ftr, Fva, Fte = std_split(ft["train"], ft["val"], ft["test"])
    mft = fit_ce(Ftr, yb_tr, Fva, yb_va, Fte, yb_te, 8, ep, wd, cw, dr)
    res["r8_ft"] = {"train": round(mft["tr_f1"], 4), "val": round(mft["val_f1"], 4),
                    "test": round(mft["test_f1"], 4)}
    print(f"[C] ft-subspace: {res['r8_ft']}", flush=True)
    for split, z, p in (("train", mft["z"][0], None), ("val", mft["z"][1], mft["val_p"]),
                        ("test", mft["z"][2], mft["te_p"])):
        if p is None:
            p = np.zeros((len(z), 11), dtype="float32")
        d = np.load(ZDIR / f"b_{split}.npz", allow_pickle=True)
        np.savez_compressed(ZDIR / f"b_{split}.npz", z8=d["z8"], probs=d["probs"],
                            z8_ft=z.astype("float32"), probs_ft=p.astype("float32"))

    # Part D：A/C 的 r=8 子空间
    for task in ("a", "c"):
        ha_tr, ya_tr = load_split(task, "train")
        ha_va, ya_va = load_split(task, "val")
        ha_te, ya_te = load_split(task, "test")
        Ttr, Tva, Tte = std_split(ha_tr, ha_va, ha_te)
        n_cls = int(max(ya_tr.max(), ya_va.max(), ya_te.max())) + 1
        m = fit_ce(Ttr, ya_tr, Tva, ya_va, Tte, ya_te, 8, 200, 1e-3)
        res[f"r8_{task}"] = {"train": round(m["tr_f1"], 4), "val": round(m["val_f1"], 4),
                             "test": round(m["test_f1"], 4)}
        print(f"[D] {task}: {res[f'r8_{task}']}", flush=True)
        for split, z, p in (("train", m["z"][0], None), ("val", m["z"][1], m["val_p"]),
                            ("test", m["z"][2], m["te_p"])):
            if p is None:
                p = np.zeros((len(z), n_cls), dtype="float32")
            save_z(task, split, z.astype("float32"), p.astype("float32"))

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "showcase.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"[out] {OUT / 'showcase.json'}")
    return 0


if __name__ == "__main__":
    with open(ROOT / "configs/ddet_v041.yaml", encoding="utf-8") as f:
        _cfg = yaml.safe_load(f)
    enc_cfg_holder = dict(_cfg["encoder"])
    enc_cfg_holder["path"] = str(ROOT / enc_cfg_holder["path"])
    raise SystemExit(main())
