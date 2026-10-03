#!/usr/bin/env python
"""Round3 诊断：正确 eval 后性能下降的归因（BN 运行统计 vs eval 批统计 vs dropout）。

在 late_fusion(seed0) 上训练同一模型（与 round3 套件一致），随后用四种模式评估 test：
  A 标准修正口径          : model.eval()（BN 用 running stats；dropout off）
  D 仅 BN 用批统计        : eval() 后手动把 BatchNorm1d 置回 train()（dropout off；单大批 BN 批统计）
  B 原缺陷口径（复刻近似）: 全程 train()（BN 批统计 + dropout 开启）
  C BN 重校准后标准口径   : 在 train 数据上前向若干批（momentum=None）重估 running stats，再 eval()
输出：artifacts/acl_dcan_round3_audit/bn_diag.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import dcan_four_models as r1  # noqa: E402
from dcan_round3_mlp import ARM_CFG, build, get_state, set_state, f1  # noqa: E402

OUT = ROOT / "artifacts" / "acl_dcan_round3_audit"


def main():
    rows = r1.load_rows()
    emb = r1.encode_semantics(rows)
    st_raw = np.array([r1.struct_features(r) for r in rows], dtype=np.float32)
    split = [r["task_split"] for r in rows]
    fam = [r["family"] for r in rows]
    task = [r["task_id"] for r in rows]
    tmap = {t: i for i, t in enumerate(dict.fromkeys(task))}
    task_int = np.array([tmap[t] for t in task])
    tr_mask = np.array([s == "train" for s in split])
    mu, sd = st_raw[tr_mask].mean(0), st_raw[tr_mask].std(0) + 1e-6
    st = (st_raw - mu) / sd
    fam_idx = {f: i for i, f in enumerate(r1.FAMILIES)}
    y = np.array([fam_idx[f] for f in fam])
    tr = np.array([i for i, s in enumerate(split) if s == "train"])
    dv = np.array([i for i, s in enumerate(split) if s == "dev"])
    te = np.array([i for i, s in enumerate(split) if s == "test"])
    dev_ = "cuda"
    Xe = torch.tensor(emb, dtype=torch.float32, device=dev_)
    Xs = torch.tensor(st, dtype=torch.float32, device=dev_)
    Y = torch.tensor(y, dtype=torch.long, device=dev_)
    T = np.asarray(task_int)

    cfg = ARM_CFG["late_fusion"]
    seed = 0
    torch.manual_seed(seed); np.random.seed(seed)
    rng = np.random.default_rng(seed)
    mods = build(seed, st.shape[1], emb.shape[1])
    for m in mods.values():
        m.to(dev_)
    opt = torch.optim.AdamW([p for m in mods.values() for p in m.parameters()], lr=1e-3, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()
    best = {"dev": -1, "state": None, "epoch": 0}
    t0 = time.time()
    for ep in range(1, 41):
        for m in mods.values():
            m.train()
        perm = rng.permutation(tr)
        for bidx in r1.make_task_batches(perm, T, rng):
            z_s = mods["ms"](Xe[bidx])
            z_f = mods["mf"](Xs[bidx])
            yb = Y[bidx]
            tb = torch.tensor(T[bidx], dtype=torch.long, device=dev_)
            loss = ce(mods["hs"](z_s), yb) + ce(mods["hf"](z_f), yb) + 0.3 * r1.supcon(z_s, tb, yb)
            opt.zero_grad(); loss.backward(); opt.step()
        for m in mods.values():
            m.eval()
        with torch.inference_mode():
            ps = F.softmax(mods["hs"](mods["ms"](Xe)), 1).cpu().numpy()
            pf = F.softmax(mods["hf"](mods["mf"](Xs)), 1).cpu().numpy()
        devf1 = f1(y[dv], ((ps[dv] + pf[dv]) / 2).argmax(1))
        if devf1 > best["dev"]:
            best = {"dev": devf1, "epoch": ep, "state": get_state(mods)}
        if ep - best["epoch"] >= 10:
            break
    set_state(mods, best["state"])

    bn_modules = [m.net[1] for m in (mods["ms"], mods["mf"])]
    out = {"arm": "late_fusion", "seed": seed, "best_dev_f1": best["dev"], "best_epoch": best["epoch"],
           "train_sec": round(time.time() - t0, 1)}

    def probs(mode):
        if mode == "A":
            for m in mods.values():
                m.eval()
        elif mode == "D":
            for m in mods.values():
                m.eval()
            for b in bn_modules:
                b.train()
        elif mode == "B":
            for m in mods.values():
                m.train()
        with torch.inference_mode():
            ps = F.softmax(mods["hs"](mods["ms"](Xe[te])), 1).cpu().numpy()
            pf = F.softmax(mods["hf"](mods["mf"](Xs[te])), 1).cpu().numpy()
        return (ps + pf) / 2

    torch.manual_seed(123)
    for mode in ("A", "D", "B"):
        p = probs(mode)
        out[f"test_f1_{mode}"] = f1(y[te], p.argmax(1))

    # C: BN 重校准（train 数据，momentum=None）+ 标准 eval
    for b in bn_modules:
        b.train()
        b.momentum = None
    with torch.inference_mode():
        order = np.random.default_rng(1).permutation(len(tr))
        for i in range(0, len(order), 256):
            idx = tr[order[i:i + 256]]
            _ = mods["ms"](Xe[idx])
            _ = mods["mf"](Xs[idx])
    p = probs("A")
    out["test_f1_C_bn_recalibrated_eval"] = f1(y[te], p.argmax(1))
    out["notes"] = {
        "A": "标准修正口径（running stats, dropout off）",
        "D": "仅把 BN 切成批统计（dropout off，单大批）",
        "B": "复刻原缺陷近似（train 模式：BN 批统计 + dropout on）",
        "C": "BN 在 train 上重校准（momentum=None）后按 A 评估",
        "interpretation": "A vs D 隔离 BN 统计口径；D vs B 隔离 dropout；C 检验 BN 重校准可否恢复"}
    (OUT / "bn_diag.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print("[bndiag]", json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
