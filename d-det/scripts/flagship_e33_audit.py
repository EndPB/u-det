#!/usr/bin/env python
"""E33 暖启动实现审计（规范 §3 规则③的检查流程；诊断用途，非新实验结论）。

检查项：
  1. W_0/b_0 转置与类别顺序：b0.classes_ == [0,1,2,3]；W_0·ã+b_0 vs b0.decision_function；
  2. 暖启动初始评估：D0 在训练前的 val BA_F/BA_G（量化 u=ã+V·GELU(Uã) 扰动代价）；
  3. 冻结 U,V（仅训 W_F）同 budget 变体：检验"暖启动协议本身"是否可达 B0；
  4. 完整 D0 训练后的 U/V/W_F 漂移量：定位退化来源（表示漂移 vs 头漂移）。

输出：artifacts/flagship_e33/warmstart_audit.json（不改动 E33 原始产物）。
"""
from __future__ import annotations

import copy
import json
import math
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from flagship_round1 import SEEN_LABS  # noqa: E402
from flagship_e28_probe import (load_train_val, ids_md5, fit_scaler,  # noqa: E402
                                apply_scaler, default_args)
from flagship_e33 import (MLP2, wl_match, WL, FAM4, C_FIXED, TOL, MAX_ITER,  # noqa: E402
                          BATCH, STEPS_SEED, LR, WD)

R28_FEATS = ROOT / "runs/flagship_e28/features_train_val.npz"
OUT = ROOT / "artifacts/flagship_e33"


def fam_bal(pred, y, fams):
    rec = []
    for k in range(len(fams)):
        m = y == k
        rec.append(float((pred[m] == k).mean()))
    return float(np.mean(rec)), rec


def main() -> int:
    args = default_args(smoke=False)
    corpus = load_train_val(args)
    docs_all = corpus["train"] + corpus["val"]
    y_ai = np.array([d.y_ai for d in docs_all], bool)
    fam = np.array([SEEN_LABS.index(d.fam) if d.fam in SEEN_LABS else -1
                    for d in docs_all], np.int16)
    gen = np.array([d.gen for d in docs_all])
    sp = np.array([d.split for d in docs_all])
    fz = dict(np.load(R28_FEATS, allow_pickle=True))
    m_tr = np.array([ids_md5(d.ids) for d in corpus["train"]])
    m_va = np.array([ids_md5(d.ids) for d in corpus["val"]])
    assert np.array_equal(m_tr, fz["md5_train"]) and np.array_equal(m_va, fz["md5_val"])
    raw = np.vstack([fz["raw_train"], fz["raw_val"]])

    sel_ai = np.array([i for i in np.where(y_ai)[0] if wl_match(str(gen[i]))])
    tr_ai = np.array([i for i in sel_ai if sp[i] == "train"])
    va_ai = np.array([i for i in sel_ai if sp[i] == "val"])
    m_r, s_r = fit_scaler(raw[tr_ai])
    A_tr = apply_scaler(raw[tr_ai], m_r, s_r)
    A_va = apply_scaler(raw[va_ai], m_r, s_r)
    y4_tr = np.array([FAM4.index(SEEN_LABS[int(fam[i])]) for i in tr_ai])
    y4_va = np.array([FAM4.index(SEEN_LABS[int(fam[i])]) for i in va_ai])

    gen_counts = defaultdict(int)
    for i in tr_ai:
        gen_counts[str(gen[i])] += 1
    fam_gens = {f: sorted({str(gen[i]) for i in tr_ai
                           if SEEN_LABS[int(fam[i])] == f}) for f in FAM4}
    N = len(tr_ai); K = len(FAM4)
    w_sample = np.array([N / (K * len(fam_gens[SEEN_LABS[int(fam[i])]])
                              * gen_counts[str(gen[i])]) for i in tr_ai])
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter("always")
        b0 = LogisticRegression(max_iter=MAX_ITER, tol=TOL, C=C_FIXED).fit(
            A_tr, y4_tr, sample_weight=w_sample)

    audit = {}
    # 1) 转置/类别序
    W0 = np.asarray(b0.coef_); b0v = np.asarray(b0.intercept_)
    df = b0.decision_function(A_va)
    manual = A_va @ W0.T + b0v
    audit["check1_mapping"] = {
        "classes_": [int(c) for c in b0.classes_],
        "coef_shape": list(W0.shape),
        "logits_maxdiff_manual_vs_sklearn": float(np.abs(manual - df).max()),
    }

    # 2) D0 初始评估（同 seed 序列复刻）
    torch.manual_seed(0)
    _dummy = torch.nn.Linear(768, 4)
    torch.nn.init.xavier_uniform_(_dummy.weight)
    torch.nn.init.zeros_(_dummy.bias)
    _mC1 = MLP2()
    mD0 = MLP2()
    with torch.no_grad():
        mD0.WF.weight.copy_(torch.as_tensor(W0, dtype=torch.float32))
        mD0.WF.bias.copy_(torch.as_tensor(b0v, dtype=torch.float32))
    with torch.no_grad():
        u_init, logits_init = mD0(torch.as_tensor(A_va))
        audit["check2_init_eval"] = {
            "logits_vs_b0_maxdiff": float(np.abs(logits_init.numpy() - df).max()),
            "init_BA_F": round(fam_bal(logits_init.argmax(1).numpy(), y4_va, FAM4)[0], 4),
            "u_vs_a_rel_fro": float(
                (torch.norm(u_init - torch.as_tensor(A_va)) /
                 torch.norm(torch.as_tensor(A_va))).item()),
        }

    # 批调度（与 E31-E33 相同）
    fam_idx = {f: np.where(y4_tr == k)[0] for k, f in enumerate(FAM4)}
    gen_of = {}
    for k, f in enumerate(FAM4):
        gen_of[f] = {}
        for g in fam_gens[f]:
            gen_of[f][g] = np.array([p for p in fam_idx[f]
                                     if str(gen[tr_ai[p]]) == g])
    steps = math.ceil(N / BATCH)
    rng = np.random.RandomState(STEPS_SEED)
    schedule = []
    for step in range(steps * 2):
        batch = []
        for f in FAM4:
            while True:
                draws = []
                for _ in range(BATCH // K):
                    g = fam_gens[f][rng.randint(len(fam_gens[f]))]
                    draws.append(int(gen_of[f][g][rng.randint(len(gen_of[f][g]))]))
                if len(set(draws)) >= 2:
                    break
            batch += draws
        schedule.append(np.array(batch))

    A_tr_t = torch.as_tensor(A_tr); y4_tr_t = torch.as_tensor(y4_tr)

    def train_variant(freeze_uv: bool, tag: str):
        model = MLP2()
        model.load_state_dict(copy.deepcopy(mD0.state_dict()))
        if freeze_uv:
            for pname, p in model.named_parameters():
                if not pname.startswith("WF"):
                    p.requires_grad = False
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                lr=LR, weight_decay=WD)
        uv0 = {n: p.detach().clone() for n, p in model.named_parameters()
               if not n.startswith("WF")}
        wf0 = model.WF.weight.detach().clone()
        traj = []
        for ep in range(2):
            for st in range(steps):
                bidx = schedule[ep * steps + st]
                _, logits = model(A_tr_t[bidx])
                loss = F.cross_entropy(logits, y4_tr_t[bidx])
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step()
                if st in (0, steps - 1):
                    traj.append({"epoch": ep, "step": st,
                                 "ce": round(float(loss.detach()), 4)})
        with torch.no_grad():
            _, logits = model(torch.as_tensor(A_va))
        ba_f, rec = fam_bal(logits.argmax(1).numpy(), y4_va, FAM4)
        drift = {n: float(torch.norm(p.detach() - uv0[n]).item())
                 for n, p in model.named_parameters() if n in uv0}
        drift["WF.weight_vs_W0"] = float(
            torch.norm(model.WF.weight.detach() - wf0).item())
        return {"BA_F": round(ba_f, 4), "recall": [round(x, 4) for x in rec],
                "traj": traj, "drift": drift, "frozen_uv": freeze_uv}

    audit["check3_frozen_uv_variant"] = train_variant(True, "frozen_uv")
    audit["check4_full_d0_variant"] = train_variant(False, "full")
    audit["note"] = ("诊断用途；冻结 U,V 变体用于隔离表示漂移与头训练；"
                     "两者均与 E33 同批序/同预算/同优化器。")
    (OUT / "warmstart_audit.json").write_text(json.dumps(audit, ensure_ascii=False,
                                                         indent=1))
    print(json.dumps(audit, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
