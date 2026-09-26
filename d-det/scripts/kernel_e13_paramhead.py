#!/usr/bin/env python
"""E13（mini）：端到端判别头（参数化 Fisher 投影）——跨题泛化检验。

动机（E11/E12）：闭式 LDA 跨题 0.88、1D 分割 in-sample 0.999；E13 检验"可学习的
Fisher 投影"能否 (a) 达到/超过 LDA 的跨题水平、(b) 给出可扩到多族的训练形态。
- 数据：E11 缓存 Δ（394×2 族）；5 折 GroupKFold（按题，同题 q/d 同折）
- 每折：训练 W（768→r，可微 Fisher 损失 log(intra)−log(inter)）→ 测试折最近中心分类
- 对照：闭式 LDA（0.88，E11）；r ∈ {1,2,8}，3 seeds
输出：runs/kernel_e13/param_head.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/kernel_e13"


def train_proj(Xtr, ytr, r=2, steps=800, lr=1e-2, seed=0):
    torch.manual_seed(seed)
    D = Xtr.shape[1]
    W = nn.Linear(D, r, bias=False)
    nn.init.normal_(W.weight, 0.0, 0.02)
    opt = torch.optim.Adam(W.parameters(), lr=lr)
    X = torch.tensor(Xtr, dtype=torch.float32)
    y = torch.tensor(ytr, dtype=torch.long)
    for _ in range(steps):
        z = W(X)
        mu0 = z[y == 0].mean(0)
        mu1 = z[y == 1].mean(0)
        zc = torch.where((y == 0).unsqueeze(1), mu0.unsqueeze(0), mu1.unsqueeze(0))
        intra = ((z - zc) ** 2).sum(1).mean()
        inter = ((mu0 - mu1) ** 2).sum()
        loss = torch.log(intra + 1e-6) - torch.log(inter + 1e-6)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        ztr = W(X).numpy()
    return W, ztr


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    d = np.load(ROOT / "runs/kernel_e11/feat_cache.npz")
    dq, dd = d["dq"], d["dd"]
    n = len(dq)
    X = np.vstack([dq, dd])
    y = np.array([0] * n + [1] * n)
    groups = np.concatenate([np.arange(n), np.arange(n)])
    res = {"n_pairs": n}
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Xz = (X - mu) / sd

    for r in (1, 2, 8):
        accs = []
        for seed in range(3):
            acc_folds = []
            for tr, va in GroupKFold(n_splits=5).split(Xz, y, groups):
                W, ztr = train_proj(Xz[tr], y[tr], r=r, seed=seed)
                with torch.no_grad():
                    zva = W(torch.tensor(Xz[va], dtype=torch.float32)).numpy()
                # 最近中心分类（用训练折投影中心）
                c0 = ztr[y[tr] == 0].mean(0)
                c1 = ztr[y[tr] == 1].mean(0)
                d0 = ((zva - c0) ** 2).sum(1)
                d1 = ((zva - c1) ** 2).sum(1)
                pred = (d1 < d0).astype(int)
                acc_folds.append(float((pred == y[va]).mean()))
            accs.append(np.mean(acc_folds))
        res[f"r{r}_gkf_acc"] = round(float(np.mean(accs)), 4)
        res[f"r{r}_std"] = round(float(np.std(accs)), 4)
        print(f"[e13] r={r}: GroupKFold acc = {res[f'r{r}_gkf_acc']} "
              f"(±{res[f'r{r}_std']})", flush=True)

    # in-sample 几何（对照 E12：判别空间的 收内/散间 效果）
    W, ztr = train_proj(Xz, y, r=2, seed=0)
    c0, c1 = ztr[y == 0].mean(0), ztr[y == 1].mean(0)
    intra = float(np.mean([((ztr[y == c] - [c0, c1][c]) ** 2).sum(1).mean() for c in (0, 1)]))
    res["insample_intra2d"] = round(intra, 3)
    res["insample_inter2d"] = round(float(((c0 - c1) ** 2).sum()), 3)
    res["insample_ratio"] = round(res["insample_inter2d"] / (intra + 1e-9), 3)
    res["ref_lda_gkf_acc_e11"] = 0.8846

    (OUT / "param_head.json").write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print(json.dumps(res, ensure_ascii=False, indent=2))
    print("[e13] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
