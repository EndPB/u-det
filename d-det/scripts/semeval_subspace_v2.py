#!/usr/bin/env python
"""s2 低秩子空间 v2（B 任务，冻结特征）：r=8 最优配置 + center-loss + 多种子 + 跨族留出。

变体：
  base8   : CE, ep200, wd=1e-5（v1 最优，复跑头）
  center  : CE + center-loss λ（类内紧致，增强"家族指纹"可读性）
  3seed   : base8/center 各 3 种子 → z 与 probs 平均
  跨族留出：每族轮流出局训练，测得对"未见家族"的鲁棒性
输出：runs/semeval_r/z8v2/b_{split}.npz（z8, probs, probs_center）; subspace_v2.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

OUT = ROOT / "runs/semeval_r/z8v2"
FEAT = ROOT / "runs/semeval_zeroshot/feat"


def load_feat(split: str):
    d = np.load(FEAT / f"b_{split}.npz", allow_pickle=True)
    return d["h"].astype("float32"), d["y"].astype("int64")


def fit_readout(H: np.ndarray, y: np.ndarray, r: int = 8, epochs: int = 200,
                wd: float = 1e-5, lam_center: float = 0.0, seed: int = 0):
    torch.manual_seed(seed)
    D = H.shape[1]
    W = nn.Linear(D, r, bias=True)
    head = nn.Linear(r, int(y.max()) + 1)
    opt = torch.optim.AdamW(list(W.parameters()) + list(head.parameters()),
                            lr=2e-3, weight_decay=wd)
    t = torch.from_numpy(H)
    yt = torch.from_numpy(y)
    n = len(yt)
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, 512):
            idx = perm[i:i + 512]
            z = W(t[idx])
            loss = F.cross_entropy(head(z), yt[idx])
            if lam_center > 0:
                mu = torch.zeros(int(y.max()) + 1, z.shape[1], dtype=z.dtype)
                cnt = torch.zeros(int(y.max()) + 1, dtype=z.dtype)
                mu.index_add_(0, yt[idx], z.detach())
                cnt.index_add_(0, yt[idx], torch.ones(len(idx), dtype=z.dtype))
                mu = mu / cnt.clamp(min=1).unsqueeze(-1)
                loss = loss + lam_center * torch.mean(
                    ((z - mu[yt[idx]]).detach() * (z - mu[yt[idx]])).sum(-1).clamp(max=50))
            opt.zero_grad()
            loss.backward()
            opt.step()
    W.eval()
    return W, head


def main() -> int:
    res = {}
    Htr, ytr = load_feat("train")
    Hva, yva = load_feat("val")
    Hte, yte = load_feat("test")
    mu, sd = Htr.mean(0), Htr.std(0) + 1e-6
    Htr = (Htr - mu) / sd
    Hva = (Hva - mu) / sd
    Hte = (Hte - mu) / sd

    def macro_f1(pred, y):
        f1s = []
        for c in np.unique(y):
            tp = np.sum((pred == c) & (y == c))
            fp = np.sum((pred == c) & (y != c))
            fn = np.sum((pred != c) & (y == c))
            f1s.append(2 * tp / max(2 * tp + fp + fn, 1))
        return float(np.mean(f1s))

    # ★ 训练需要梯度：用 enable_grad 覆盖外层上下文（曾是 no_grad 导致 backward 报错）
    with torch.enable_grad():
        for tag, lam in (("base8", 0.0), ("center", 0.03)):
            zs_va, ps_va, zs_te, ps_te = [], [], [], []
            for seed in range(3):
                W, head = fit_readout(Htr, ytr, r=8, lam_center=lam, seed=seed)
                with torch.no_grad():
                    zva = W(torch.from_numpy(Hva)).numpy()
                    zte = W(torch.from_numpy(Hte)).numpy()
                    pva = torch.softmax(head(torch.from_numpy(zva)), -1).numpy()
                    pte = torch.softmax(head(torch.from_numpy(zte)), -1).numpy()
                zs_va.append(zva); ps_va.append(pva); zs_te.append(zte); ps_te.append(pte)
                print(f"[sub2] {tag} seed{seed}: val F1={macro_f1(pva.argmax(1), yva):.4f}", flush=True)
            zva, pva = np.mean(zs_va, 0), np.mean(ps_va, 0)
            zte, pte = np.mean(zs_te, 0), np.mean(ps_te, 0)
            OUT.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(OUT / f"b_val_{tag}.npz", z8=zva, probs=pva, y=yva)
            np.savez_compressed(OUT / f"b_test_{tag}.npz", z8=zte, probs=pte, y=yte)
            res[tag] = {"val_f1": macro_f1(pva.argmax(1), yva),
                        "test_f1": macro_f1(pte.argmax(1), yte)}
            print(f"[sub2] {tag} 3seed 平均: val={res[tag]['val_f1']:.4f} "
                  f"test={res[tag]['test_f1']:.4f}", flush=True)

        # ---- 跨族留出：每次丢掉一族（连同其训练样本），在其余类上训练，看退化 ----
        holdout = {}
        classes = sorted(set(ytr.tolist()))
        for c in classes:
            m_tr = ytr != c
            m_va = yva != c
            if m_va.sum() < 50:
                continue
            W, head = fit_readout(Htr[m_tr], ytr[m_tr], r=8, epochs=100)
            with torch.no_grad():
                pva = torch.softmax(head(W(torch.from_numpy(Hva[m_va]))), -1).numpy()
            holdout[int(c)] = macro_f1(pva.argmax(1), yva[m_va])
        res["holdout"] = holdout
        print(f"[sub2] 跨族留出 macro-F1 均值={np.mean(list(holdout.values())):.4f}  "
              f"最差族={min(holdout, key=holdout.get)}", flush=True)
    with open(ROOT / "runs/semeval_r/subspace_v2.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print("[sub2] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
