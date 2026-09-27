#!/usr/bin/env python
"""K2-full-lite：DCAN 式训练版"残差分离"归因头（冻结特征上；GPU 秒级）。

背景与判读（预注册，出结果前不得修改）：
  外部方案 K2（DCAN：h_com 减公共成分 + 同题一致性）在我们数据上的最小可信版。
  对照口径（同折、同数据、脚本内重算闭式基线）：
    · 全量 5703：闭式 DiscHead raw；
    · 对齐 329 题（1974 样本）：闭式 raw / 闭式逐题中心化（E15-K2-lite=0.7741）。
  通过线：任一臂在对应口径超过闭式最优 **+1.0pt**（对齐 >0.7841 或 全量 >0.7726）
  → K2-full 成立；否则记录负结果（第 4 例"小样本学习 < 闭式"风险）。

模型：h_com = MLP(u)（512 hidden, GELU）；h_spec = u − h_com；logits = Lin(h_spec)。
损失：CE + λ_rc · (1 − cos(h_com(pair_i), h_com(pair_j)))，(i,j) 为同题异族的"对的对"。
输入臂：delta = h+ − h−；cat = [h+; h−]。λ_rc ∈ {0, 0.2}（delta）+ {0.2}（cat）。
协议：按题 GroupKFold=5（与 DiscHead 完全同折）；标准化仅用训练折统计。

运行：
  OMP_NUM_THREADS=8 /root/miniconda3/envs/udet/bin/python scripts/kernel_e16_dcan.py --smoke
  OMP_NUM_THREADS=8 /root/miniconda3/envs/udet/bin/python scripts/kernel_e16_dcan.py
输出：runs/kernel_e16/dcan.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.disc import DiscHead  # noqa: E402

OUT = ROOT / "runs/kernel_e16"
NAMES = ["qwen05", "qwen15", "ds13", "yi15", "granite2b", "smollm2"]
CONF_PAIRS = [("granite2b", "smollm2"), ("qwen05", "qwen15")]
ARMS = [("delta", 0.0), ("delta", 0.2)]


# --------------------------------------------------------------------------- #
def load_inputs(arm: str, common_only: bool = False):
    """返回 (U, y, g)；common_only 时仅保留 6 族共同题。"""
    data = {}
    for n in NAMES:
        z = np.load(OUT / f"h_{n}.npz", allow_pickle=True)
        data[n] = (np.asarray(z["h_plus"], dtype="float32"),
                   np.asarray(z["h_minus"], dtype="float32"),
                   [str(v) for v in z["tasks"]])
    if common_only:
        common = None
        for n in NAMES:
            t = set(data[n][2])
            common = t if common is None else (common & t)
        common = sorted(common)
    X, y, g = [], [], []
    for ci, n in enumerate(NAMES):
        hp, hm, t = data[n]
        if common_only:
            idx = {tt: i for i, tt in enumerate(t)}
            sel = [idx[tt] for tt in common]
            hp, hm, t = hp[sel], hm[sel], [t[i] for i in sel]
        u = hp - hm if arm == "delta" else np.concatenate([hp, hm], axis=1)
        X.append(u)
        y += [ci] * len(u)
        g += t
    return np.vstack(X), np.array(y), np.array(g)


def build_rc_pool(g, y, cap=60000, seed=0):
    rng = np.random.default_rng(seed)
    by_task: dict = {}
    for i, t in enumerate(g.tolist()):
        by_task.setdefault(t, []).append(i)
    pairs = []
    for t, idx in by_task.items():
        if len(idx) < 2:
            continue
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                if y[idx[a]] != y[idx[b]]:
                    pairs.append((idx[a], idx[b]))
    if len(pairs) > cap:
        sel = rng.choice(len(pairs), size=cap, replace=False)
        pairs = [pairs[k] for k in sel]
    return np.array(pairs, dtype=np.int64)


class DCANHead(nn.Module):
    def __init__(self, d: int, n_cls: int = 6, hidden: int = 512):
        super().__init__()
        self.com = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, d))
        self.cls = nn.Linear(d, n_cls)

    def forward(self, u):
        hc = self.com(u)
        return self.cls(u - hc), hc


def train_head(Utr, ytr, gtr, lam, epochs, seed=0, batch=256, lr=1e-3, device="cuda"):
    torch.manual_seed(seed)
    model = DCANHead(Utr.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    Ut = torch.tensor(Utr, device=device)
    yt = torch.tensor(ytr, device=device)
    rc_np = build_rc_pool(gtr, ytr)
    rc_t = torch.tensor(rc_np, device=device) if len(rc_np) else None
    n = len(Ut)
    gen = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        perm = torch.randperm(n, generator=gen)
        for s in range(0, n, batch):
            idx = perm[s:s + batch].to(device)
            logits, _ = model(Ut[idx])
            loss = F.cross_entropy(logits, yt[idx])
            if lam > 0 and rc_t is not None:
                k = min(512, len(rc_t))
                sel = torch.randint(0, len(rc_t), (k,), generator=gen)
                pa, pb = rc_t[sel, 0], rc_t[sel, 1]
                _, hca = model(Ut[pa])
                _, hcb = model(Ut[pb])
                loss = loss + lam * (1 - F.cosine_similarity(hca, hcb, dim=1)).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    return model


@torch.no_grad()
def predict(model, U, device="cuda", batch=2048):
    model.eval()
    out = []
    for s in range(0, len(U), batch):
        ub = torch.tensor(U[s:s + batch], device=device)
        logits, _ = model(ub)
        out.append(logits.argmax(1).cpu().numpy())
    return np.concatenate(out)


def summarize(pred, y):
    rec = {str(c): round(float(((pred == c) & (y == c)).sum()) / max(1, int((y == c).sum())), 3)
           for c in np.unique(y)}
    conf = {}
    for a, b in CONF_PAIRS:
        if (y == a).any() and (y == b).any():
            conf[f"{a}->{b}"] = int(((y == a) & (pred == b)).sum())
            conf[f"{b}->{a}"] = int(((y == b) & (pred == a)).sum())
    return {"acc": round(float(accuracy_score(y, pred)), 4),
            "balanced_acc": round(float(balanced_accuracy_score(y, pred)), 4),
            "recall": rec, "conf": conf}


def run_protocol(X, y, g, arms, epochs, max_folds, device, tag, lr=1e-3):
    preds = {f"{arm}_rc{lam}": np.empty(len(y), dtype=object) for arm, lam in arms}
    fold_accs = {k: [] for k in preds}
    visited = np.zeros(len(y), dtype=bool)
    splits = list(GroupKFold(n_splits=5).split(X, y, g))
    if max_folds:
        splits = splits[:max_folds]
    for fi, (tr, va) in enumerate(splits):
        visited[va] = True
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        Utr, Uva = (X[tr] - mu) / sd, (X[va] - mu) / sd
        for arm, lam in arms:
            m = train_head(Utr, y[tr], g[tr], lam, epochs, seed=fi, device=device, lr=lr)
            p = predict(m, Uva, device=device)
            preds[f"{arm}_rc{lam}"][va] = [NAMES[c] for c in p]
            fold_accs[f"{arm}_rc{lam}"].append(
                float(accuracy_score([NAMES[c] for c in y[va]], [NAMES[c] for c in p])))
        print(f"  [{tag}] fold {fi} done", flush=True)
    res = {}
    y_str = np.array([NAMES[c] for c in y])
    for k, p in preds.items():
        p = np.array(p, dtype=str)
        res[k] = summarize(p[visited], y_str[visited])
        res[k]["fold_accs"] = [round(v, 4) for v in fold_accs[k]]
        res[k]["n"] = int(visited.sum())
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--max-folds", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--only-aligned", action="store_true")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[e16] device={device}", flush=True)
    report = {"baselines": {}, "full": {}, "aligned": {}}
    arms = ARMS

    # ---- 全量口径 ----
    X, y, g = load_inputs("delta")
    raw = DiscHead().cross_val(X, y, g)
    report["baselines"]["full_raw_disc"] = {k: round(float(v), 4) for k, v in raw.items()}
    print(f"[e16] 全量闭式基线 raw: {report['baselines']['full_raw_disc']}", flush=True)
    epochs = 3 if args.smoke else args.epochs
    max_folds = 1 if args.smoke else args.max_folds
    use_arms = [("delta", 0.2)] if args.smoke else arms
    if not (args.smoke or args.only_aligned):
        print(f"[e16] 全量 {X.shape}：训练 {len(use_arms)} 臂 × 5 折 ...", flush=True)
        report["full"] = run_protocol(X, y, g, use_arms, epochs, max_folds, device, "full", lr=args.lr)

    # ---- 对齐口径（329 共同题）----
    Xa, ya, ga = load_inputs("delta", common_only=True)
    raw_a = DiscHead().cross_val(Xa, ya, ga)
    report["baselines"]["aligned_raw_disc"] = {k: round(float(v), 4) for k, v in raw_a.items()}
    # 中心化闭式（脚本内重算，与 E15 一致）
    k = len(NAMES)
    T = len(Xa) // k
    M = np.stack([Xa[i * T:(i + 1) * T] for i in range(k)])
    Mc = M - M.mean(axis=0, keepdims=True)
    Xac = np.vstack(list(Mc))
    cen = DiscHead().cross_val(Xac, ya, ga)
    report["baselines"]["aligned_centered_disc"] = {k2: round(float(v), 4) for k2, v in cen.items()}
    print(f"[e16] 对齐闭式基线：raw {report['baselines']['aligned_raw_disc']} ", flush=True)
    print(f"                中心化 {report['baselines']['aligned_centered_disc']}", flush=True)
    sal_arms = [("delta", 0.2)] if args.smoke else arms
    print(f"[e16] 对齐 {Xa.shape}：训练 {len(sal_arms)} 臂 × {max_folds or 5} 折 ...", flush=True)
    report["aligned"] = run_protocol(Xa, ya, ga, sal_arms, epochs, max_folds, device, "aligned", lr=args.lr)

    if args.smoke:
        fname = "dcan_smoke.json"
    elif args.only_aligned:
        fname = f"dcan_aligned_ep{args.epochs}_lr{args.lr}.json"
    else:
        fname = "dcan.json"
    out = OUT / fname
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print("\n===== 结果摘要 =====")
    for scope in ("full", "aligned"):
        for kk, vv in report.get(scope, {}).items():
            print(f"  [{scope}] {kk:>14}: acc {vv['acc']:.4f} / bal {vv['balanced_acc']:.4f} / conf {vv['conf']}")
    print(f"\n[e16] 写出 {out}")
    print("[e16] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
