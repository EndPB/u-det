#!/usr/bin/env python
"""E2（mini）：s2 读出损失对照——hinge vs Bradley-Terry logistic，以真实 r* 作决胜。

- 数据：pairs（x_plus/x_minus）；特征：CodeT5 冻结 v1.0（v0.4.1_covreg 编码器）mean-pool h
- 读出：D→1 线性 × {hinge, BT} × 3 seeds；全批训练（CPU 秒级）
- 评测：dir_acc / Δs2；★决胜：在 E1 的 198 条 B 样本上 corr(w·h, r*_total)，按 family 子集对照
- 输出：runs/kernel_e2/summary.json（含特征缓存 pair_feat.npz）
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from encoders import build_encoder  # noqa: E402
from models import build_model  # noqa: E402
from semeval_finetune import pad_collate  # noqa: E402

OUT = ROOT / "runs/kernel_e2"
PAIRS = ROOT / "data/processed/pairs.parquet"
QW15 = ROOT / "data/processed/pairs_qwen15.parquet"


class PairSeq(torch.utils.data.Dataset):
    def __init__(self, codes, tok):
        self.codes = codes
        self.tok = tok

    def __len__(self):
        return len(self.codes)

    def __getitem__(self, i):
        ids = self.tok(self.codes[i], add_special_tokens=False)["input_ids"][:1024]
        return {"input_ids": ids, "label": 0, "language": ""}


@torch.no_grad()
def encode_codes(dual, codes, tok, device):
    dl = DataLoader(PairSeq(codes, tok), batch_size=8, collate_fn=pad_collate, num_workers=0)
    H = []
    for b in dl:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            _, _, h = dual.forward_feat(b["input_ids"].to(device),
                                        b["attention_mask"].to(device))
        H.append(h.float().cpu())
    return torch.cat(H)


def fit_readout(Hp, Hm, loss: str, seed: int, steps: int = 400, lr: float = 1e-2):
    torch.manual_seed(seed)
    W = nn.Linear(Hp.shape[1], 1)
    nn.init.normal_(W.weight, 0.0, 0.01)
    nn.init.zeros_(W.bias)
    opt = torch.optim.AdamW(W.parameters(), lr=lr)
    for _ in range(steps):
        sp = W(Hp).squeeze(1)
        sm = W(Hm).squeeze(1)
        if loss == "hinge":
            L = F.relu(1.0 - sp + sm).mean()
        else:  # Bradley-Terry logistic（log-ratio 的 MLE）
            L = F.binary_cross_entropy_with_logits(sp - sm, torch.ones_like(sp))
        opt.zero_grad()
        L.backward()
        opt.step()
    return W


def eval_pair(W, Hp, Hm):
    with torch.no_grad():
        sp = W(Hp).squeeze(1)
        sm = W(Hm).squeeze(1)
        d = (sp - sm).numpy()
    return {"dir_acc": round(float((d > 0).mean()), 4), "delta_mean": round(float(d.mean()), 3),
            "delta_std": round(float(d.std()), 3)}


def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUT.mkdir(parents=True, exist_ok=True)
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
    tok = AutoTokenizer.from_pretrained(enc_cfg["path"])

    summary = {}
    for name, path in (("all", PAIRS), ("qwen15", QW15)):
        if not path.exists():
            continue
        t = pq.read_table(path)
        xp = t.column("x_plus").to_pylist()
        xm = t.column("x_minus").to_pylist()
        fam = t.column("family").to_pylist() if "family" in t.schema.names else [""] * len(xp)
        sp_arr = t.column("split").to_pylist() if "split" in t.schema.names else ["train"] * len(xp)
        keep = [i for i, s in enumerate(sp_arr) if s == "train"]
        if len(keep) > 600:
            rng = np.random.RandomState(0)
            keep = sorted(rng.choice(keep, 600, replace=False).tolist())
        xp = [xp[i] for i in keep]
        xm = [xm[i] for i in keep]
        fams = Counter(fam[i] for i in keep)
        print(f"[e2] {name}: n={len(xp)} families={dict(fams)}", flush=True)
        Hp = encode_codes(dual, xp, tok, device)
        Hm = encode_codes(dual, xm, tok, device)
        np.savez_compressed(OUT / f"{name}_feat.npz", hp=Hp.numpy(), hm=Hm.numpy())
        res = {"n": len(xp), "families": {str(k): int(v) for k, v in fams.items()}}
        # E1 靶子
        e1p = ROOT / "runs/kernel_e1/rstar_total.npz"
        target = None
        if e1p.exists():
            d = np.load(e1p, allow_pickle=True)
            featb = np.load(ROOT / "runs/semeval_zeroshot/feat/b_val.npz", allow_pickle=True)
            Hb = torch.from_numpy(np.asarray(featb["h"])[d["row"]].astype("float32"))
            target = {"r": d["r"], "y": d["y"]}
        for loss in ("hinge", "bt"):
            accs, deltas, cors, cors_s = [], [], [], []
            w_cos = []
            Ws = []
            for seed in range(3):
                W = fit_readout(Hp, Hm, loss, seed)
                ev = eval_pair(W, Hp, Hm)
                accs.append(ev["dir_acc"])
                deltas.append(ev["delta_mean"])
                Ws.append(W)
                if target is not None:
                    with torch.no_grad():
                        s_200 = W(Hb).squeeze(1).numpy()
                    cors.append(float(pearsonr(s_200, target["r"])[0]))
                    cors_s.append(float(spearmanr(s_200, target["r"])[0]))
            res[loss] = {"dir_acc": round(float(np.mean(accs)), 4),
                         "delta_mean": round(float(np.mean(deltas)), 3),
                         "corr_rstar_pearson": round(float(np.mean(cors)), 4) if cors else None,
                         "corr_rstar_spearman": round(float(np.mean(cors_s)), 4) if cors_s else None}
            print(f"[e2] {name}/{loss}: {res[loss]}", flush=True)
        summary[name] = res
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print("[e2] done ->", OUT / "summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
