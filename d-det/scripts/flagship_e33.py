#!/usr/bin/env python
"""E33：暖启动下的来源几何增量（D0/D1/D2；只修复读出优化起点）。

依据 `docx/d-det_E33暖启动来源几何增量指导_2026-09-30.md`：
  · 沿用 E31/E32 五模型子集/原 train-val/E28 raw/批序/batch=128/2 epoch/τ=.1/λ_C=.1；
  · 网络保持 E32：u=a+V·GELU(Ua+b1)+b2（无 L2 归一化），ℓ=W_F u+b_F；
  · D0 = 暖启动 CE（W_F=W_0, b_F=b_0 取 B0 收敛参数；U,V 按 E32 初始化）→ L_F；
    D1 = 与 D0 完全同初始状态 + 同批序 → L_F+0.1·L_C；
    D2 = E32 随机初始化（复刻 C1 序列）→ L_F+0.1·L_C（复现参照）；
  · SupCon：cosine(u)、同家族正对、τ=.1 只除一次；记录同/跨 gen 正对比例；
  · 暖启动一致性检查：D0 初始 logits vs B0 decision_function 的 max|Δ|（转置/类别序/标准化）。

预注册判读（§3；close:= ≥B0−.02）：
  ①D0≈B0 且 D1−D0≥1pt → 可达基线附近的来源对比初步增量；
  ②D0≈B0 且 D1−D0<1pt → SupCon 无实用增量，不再调 λ/τ/几何；
  ③D0≪B0 → 先查实现（标准化/类别序/W_0 转置/logits）；
  ④D1<D0 且 Mistral 召回↑ → 家族间再分配（不按总增益报告）。

产物：artifacts/flagship_e33/{config.json, manifest.json, metrics.json, predictions.npz,
train.log}（report.md 由报告复制）；smoke → artifacts/flagship_e33_smoke；默认拒绝覆盖。
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import subprocess
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from flagship_round1 import SEEN_LABS  # noqa: E402
from flagship_e28_probe import (load_train_val, ids_md5, fit_scaler,  # noqa: E402
                                apply_scaler, default_args)

R28_FEATS = ROOT / "runs/flagship_e28/features_train_val.npz"
ENC_PATH = ROOT / "runs/v0.4.1_covreg/last.pt"
WL = ["GPT-4o", "DeepSeek-V3-0324", "Mistral-7B-Instruct-v0.3",
      "Devstral-Small-2505", "gemma-3-27b-it"]
FAM4 = ["OpenAI", "DeepSeek", "Mistral", "Google"]
C_FIXED = 0.1
TOL = 1e-6
MAX_ITER = 10000
BATCH = 128
STEPS_SEED = 0
LR = 1e-3
WD = 1e-4
TAU = 0.1
LAM_C = 0.1
EPOCHS = 2
STD_FLOOR = 1e-2
CLOSE = 0.02
E32_C2_BA_F = 0.5255      # E32 复现参照
E32_C2_BA_G = 0.5264


def wl_match(g):
    return any(p in g for p in WL)


class MLP2(torch.nn.Module):
    """u=ã+V·GELU(U·ã+b1)+b2；ℓ=W_F u+b_F（无 L2 归一化）。"""

    def __init__(self, d=768, k=4):
        super().__init__()
        self.U = torch.nn.Linear(d, d)
        self.V = torch.nn.Linear(d, d)
        self.WF = torch.nn.Linear(d, k)
        torch.nn.init.xavier_uniform_(self.U.weight)
        torch.nn.init.zeros_(self.U.bias)
        torch.nn.init.normal_(self.V.weight, std=1e-3)
        torch.nn.init.zeros_(self.V.bias)
        torch.nn.init.xavier_uniform_(self.WF.weight)
        torch.nn.init.zeros_(self.WF.bias)

    def forward(self, a):
        u = a + self.V(F.gelu(self.U(a)))
        return u, self.WF(u)


def supcon_cos(u, fam):
    un = u / u.norm(dim=1, keepdim=True).clamp(min=1e-6)
    sim = un @ un.t() / TAU
    b = u.shape[0]
    self_mask = torch.eye(b, dtype=torch.bool, device=u.device)
    sim = sim.masked_fill(self_mask, -1e9)
    logz = torch.logsumexp(sim, dim=1)
    same = (fam[:, None] == fam[None, :]) & ~self_mask
    cnt = same.sum(1)
    lp = ((sim - logz[:, None]) * same).sum(1) / cnt.clamp(min=1)
    valid = cnt > 0
    return -lp[valid].mean(), int((~valid).sum())


def fam_eval(pred, y, fams, gen_list):
    rec_f, rec_g = {}, {}
    for k, f in enumerate(fams):
        m = y == k
        rec_f[f] = round(float((pred[m] == k).mean()), 4)
        gv = defaultdict(list)
        for j in np.where(m)[0]:
            gv[gen_list[j]].append(float(pred[j] == k))
        rec_g[f] = {g: round(float(np.mean(v)), 4) for g, v in gv.items()}
    BA_F = float(np.mean([rec_f[f] for f in fams]))
    BA_G = float(np.mean([np.mean(list(rec_g[f].values())) for f in fams]))
    return {"BA_F": round(BA_F, 4), "BA_G": round(BA_G, 4),
            "per_family_recall": rec_f, "per_generator_recall": rec_g}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    args = default_args(smoke=a.smoke)
    OUT = ROOT / "artifacts" / ("flagship_e33_smoke" if a.smoke else "flagship_e33")
    if OUT.exists() and any(OUT.iterdir()) and not a.force:
        print(f"[e33] 已存在产物，默认拒绝覆盖：{OUT}（--force 可覆盖）", flush=True)
        return 2
    OUT.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    log_lines: list[str] = []

    def say(msg):
        print(msg, flush=True)
        log_lines.append(msg)

    commit = subprocess.run(["git", "-C", str(ROOT.parent), "rev-parse", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    say(f"[e33] commit {commit} | smoke={a.smoke}")

    # ---- 数据（与 E31/E32 完全一致；只读 b_train/b_val） ----
    corpus = load_train_val(args)
    docs_all = corpus["train"] + corpus["val"]
    y_ai = np.array([d.y_ai for d in docs_all], bool)
    fam = np.array([SEEN_LABS.index(d.fam) if d.fam in SEEN_LABS else -1
                    for d in docs_all], np.int16)
    gen = np.array([d.gen for d in docs_all])
    rows = np.array([d.row for d in docs_all], np.int64)
    sp = np.array([d.split for d in docs_all])

    feats_sha = hashlib.sha256(R28_FEATS.read_bytes()).hexdigest() if not a.smoke else None
    enc_sha = hashlib.sha256(ENC_PATH.read_bytes()).hexdigest()
    if a.smoke:
        import yaml
        from encoders import build_encoder
        from models import build_model
        from flagship_round1 import SetPool
        from flagship_e28_probe import export_pass
        ms = torch.load(ROOT / "runs/flagship_e27/model_state.pt", map_location="cpu",
                        weights_only=False)
        with open(ROOT / "configs/ddet_v041.yaml", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        ec = dict(cfg["encoder"]); ec["path"] = str(ROOT / ec["path"])
        dual = build_model("dual", encoder=build_encoder(**ec), dim=768,
                           pool=cfg["model"].get("pool", "mean"),
                           s2_rank=cfg["model"].get("s2_rank", 1))
        dual.load_state_dict(torch.load(ENC_PATH, map_location="cpu",
                                        weights_only=False)["state"], strict=False)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        enc = dual.encoder.to(dev).eval()
        head = SetPool(768, 512).to(dev)
        head.load_state_dict(torch.load(ROOT / "runs/flagship_r1/head.pt",
                                        map_location=dev)["head"])
        head.eval()
        wq = torch.as_tensor(ms["wq"], device=dev)
        bq = torch.as_tensor(ms["bq"], device=dev)
        m_tT = torch.as_tensor(ms["m_t"], dtype=torch.float32, device=dev)
        s_tT = torch.as_tensor(ms["s_t"], dtype=torch.float32, device=dev)
        r1, _, _ = export_pass(enc, head, corpus["train"], args.batch, dev, wq, bq,
                               m_tT, s_tT)
        r2, _, _ = export_pass(enc, head, corpus["val"], args.batch, dev, wq, bq,
                               m_tT, s_tT)
        raw = np.vstack([r1, r2])
    else:
        fz = dict(np.load(R28_FEATS, allow_pickle=True))
        m_tr = np.array([ids_md5(d.ids) for d in corpus["train"]])
        m_va = np.array([ids_md5(d.ids) for d in corpus["val"]])
        assert np.array_equal(m_tr, fz["md5_train"]) and \
            np.array_equal(m_va, fz["md5_val"]), "清单与 E28 特征不符"
        assert len(corpus["train"]) == 17600 and len(corpus["val"]) == 5372
        raw = np.vstack([fz["raw_train"], fz["raw_val"]])
        say(f"[e33] 特征 r0={R28_FEATS.name}（sha256 {feats_sha[:16]}…）| 编码器 "
            f"{enc_sha[:16]}…")

    sel_ai = np.array([i for i in np.where(y_ai)[0] if wl_match(str(gen[i]))])
    tr_ai = np.array([i for i in sel_ai if sp[i] == "train"])
    va_ai = np.array([i for i in sel_ai if sp[i] == "val"])
    hum_tr = np.where((~y_ai) & (sp == "train"))[0]
    hum_va = np.where((~y_ai) & (sp == "val"))[0]
    if not a.smoke:
        assert len(tr_ai) == 4013 and len(va_ai) == 1654, \
            f"子集计数不符：{len(tr_ai)}/{len(va_ai)}"
    say(f"[e33] 子集：AI train {len(tr_ai)} / val {len(va_ai)}；Human "
        f"{len(hum_tr)}/{len(hum_va)}")

    m_r, s_r = fit_scaler(raw[tr_ai])
    A_tr = apply_scaler(raw[tr_ai], m_r, s_r)
    A_va = apply_scaler(raw[va_ai], m_r, s_r)
    A_hum_tr = apply_scaler(raw[hum_tr], m_r, s_r)
    A_hum_va = apply_scaler(raw[hum_va], m_r, s_r)
    y4_tr = np.array([FAM4.index(SEEN_LABS[int(fam[i])]) for i in tr_ai])
    y4_va = np.array([FAM4.index(SEEN_LABS[int(fam[i])]) for i in va_ai])
    gen_va = [str(gen[i]) for i in va_ai]

    # ---- 检测头（冻结，同 E31/E32） ----
    det_lr = LogisticRegression(max_iter=MAX_ITER, tol=TOL, C=C_FIXED).fit(
        np.vstack([A_tr, A_hum_tr]), np.concatenate([np.ones(len(A_tr)),
                                                     np.zeros(len(A_hum_tr))]))
    det_va = det_lr.predict_proba(np.vstack([A_va, A_hum_va]))[:, 1]
    det_y = np.concatenate([np.ones(len(A_va)), np.zeros(len(A_hum_va))])
    det_auroc = float(roc_auc_score(det_y, det_va))
    say(f"[e33] 检测头（冻结）：val AUROC {det_auroc:.4f}")

    # ---- B0（同 E31/E32 权重与收敛参照） ----
    N = len(tr_ai); K = len(FAM4)
    gen_counts = defaultdict(int)
    for i in tr_ai:
        gen_counts[str(gen[i])] += 1
    fam_gens = {f: sorted({str(gen[i]) for i in tr_ai
                           if SEEN_LABS[int(fam[i])] == f}) for f in FAM4}
    w_sample = np.array([N / (K * len(fam_gens[SEEN_LABS[int(fam[i])]])
                              * gen_counts[str(gen[i])]) for i in tr_ai])
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter("always")
        b0 = LogisticRegression(max_iter=MAX_ITER, tol=TOL, C=C_FIXED).fit(
            A_tr, y4_tr, sample_weight=w_sample)
    b0_conv = not any(issubclass(x.category, ConvergenceWarning) for x in wl)
    b0_val = fam_eval(b0.predict(A_va), y4_va, FAM4, gen_va)
    b0_val["converged"] = b0_conv
    b0_val["n_iter"] = int(np.max(b0.n_iter_))
    say(f"[e33] B0：BA_F {b0_val['BA_F']} BA_G {b0_val['BA_G']}（收敛 {b0_conv}）")

    # ---- 批调度（复制 E31/E32 生成代码：同 seed、同序 → 同批序） ----
    fam_idx = {f: np.where(y4_tr == k)[0] for k, f in enumerate(FAM4)}
    gen_of = {}
    for k, f in enumerate(FAM4):
        gen_of[f] = {}
        for g in fam_gens[f]:
            idxs = np.array([p for p in fam_idx[f] if str(gen[tr_ai[p]]) == g])
            gen_of[f][g] = idxs
    steps = math.ceil(N / BATCH)
    rng = np.random.RandomState(STEPS_SEED)
    schedule = []
    pair_stats = {"same_gen": 0, "cross_gen": 0, "batches": 0, "redraw": 0, "empty_P": 0}
    for step in range(steps * EPOCHS):
        batch = []
        for f in FAM4:
            while True:
                draws = []
                for _ in range(BATCH // K):
                    g = fam_gens[f][rng.randint(len(fam_gens[f]))]
                    draws.append(int(gen_of[f][g][rng.randint(len(gen_of[f][g]))]))
                if len(set(draws)) >= 2:
                    break
                pair_stats["redraw"] += 1
            batch += draws
        bidx = np.array(batch)
        fams_b = y4_tr[bidx]
        gens_b = [str(gen[tr_ai[p]]) for p in bidx]
        for i in range(len(bidx)):
            for j in range(i + 1, len(bidx)):
                if fams_b[i] == fams_b[j]:
                    if gens_b[i] == gens_b[j]:
                        pair_stats["same_gen"] += 1
                    else:
                        pair_stats["cross_gen"] += 1
        pair_stats["batches"] += 1
        schedule.append(bidx)
    say(f"[e33] 批调度（同 E31/E32）：{steps} step/epoch×{EPOCHS}；正对 同gen "
        f"{pair_stats['same_gen']} / 跨gen {pair_stats['cross_gen']}")

    # ---- 初始化（复刻 E32 的 seed 序列） ----
    torch.manual_seed(0)
    _dummy = torch.nn.Linear(768, 4)                 # 复刻 E32 c0 的 RNG 消耗
    torch.nn.init.xavier_uniform_(_dummy.weight)
    torch.nn.init.zeros_(_dummy.bias)
    mC1 = MLP2()                                     # == E32 的 C1 精确初始化
    mD0 = MLP2()                                     # D0/D1 的 U/V（E32 协议）
    with torch.no_grad():                            # 暖启动：W_F=W_0, b_F=b_0
        mD0.WF.weight.copy_(torch.as_tensor(b0.coef_, dtype=torch.float32))
        mD0.WF.bias.copy_(torch.as_tensor(b0.intercept_, dtype=torch.float32))
    mD1 = MLP2()
    mD1.load_state_dict(copy.deepcopy(mD0.state_dict()))
    mD2 = MLP2()
    mD2.load_state_dict(copy.deepcopy(mC1.state_dict()))

    # 暖启动一致性检查（D0 初始 logits vs B0 decision_function）
    with torch.no_grad():
        _, logits_init = mD0(torch.as_tensor(A_va))
        b0_df = b0.decision_function(A_va)
    warm_diff = float(np.abs(logits_init.numpy() - b0_df).max())
    say(f"[e33] 暖启动一致性：初始 logits vs B0 max|Δ| = {warm_diff:.2e}")

    arms = {
        "D0": (mD0, torch.optim.AdamW(mD0.parameters(), lr=LR, weight_decay=WD), False),
        "D1": (mD1, torch.optim.AdamW(mD1.parameters(), lr=LR, weight_decay=WD), True),
        "D2": (mD2, torch.optim.AdamW(mD2.parameters(), lr=LR, weight_decay=WD), True),
    }
    A_tr_t = torch.as_tensor(A_tr)
    y4_tr_t = torch.as_tensor(y4_tr)
    stats = {k: [] for k in arms}
    for ep in range(EPOCHS):
        for step in range(steps):
            bidx = schedule[ep * steps + step]
            xb = A_tr_t[bidx]; yb = y4_tr_t[bidx]
            for tag, (model, opt, use_c) in arms.items():
                u, logits = model(xb)
                L_F = F.cross_entropy(logits, yb)
                L_C = None
                if use_c:
                    L_C, n_empty = supcon_cos(u, yb)
                    pair_stats["empty_P"] += n_empty
                    loss = L_F + LAM_C * L_C
                else:
                    loss = L_F
                opt.zero_grad(); loss.backward()
                gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
                opt.step()
                if step in (0, steps - 1):
                    stats[tag].append({"epoch": ep, "step": step,
                                       "ce": round(float(L_F.detach()), 4),
                                       "lc": round(float(L_C.detach()), 4)
                                       if L_C is not None else None,
                                       "grad_norm": round(gnorm, 3)})
        say(f"[e33] epoch {ep} 末首/末批：" + " | ".join(
            f"{t} {[s for s in stats[t] if s['epoch'] == ep]}" for t in arms))

    # ---- 评估 ----
    res = {}
    preds = {"B0": b0.predict(A_va).astype(np.int8)}
    for tag, (model, _, _) in arms.items():
        with torch.no_grad():
            _, logits = model(torch.as_tensor(A_va))
        pred = logits.argmax(1).numpy()
        res[tag] = fam_eval(pred, y4_va, FAM4, gen_va)
        preds[tag] = pred.astype(np.int8)
    for tag in ("B0", "D0", "D1", "D2"):
        r = b0_val if tag == "B0" else res[tag]
        say(f"[e33] val·{tag}：BA_F {r['BA_F']} BA_G {r['BA_G']}")

    d01 = res["D1"]["BA_F"] - res["D0"]["BA_F"]
    d2_rep_f = res["D2"]["BA_F"] - E32_C2_BA_F
    d2_rep_g = res["D2"]["BA_G"] - E32_C2_BA_G
    param_diff = {p: float((dict(mD0.named_parameters())[p] -
                            dict(mD1.named_parameters())[p]).abs().max())
                  for p in ("U.weight", "V.weight", "WF.weight")}
    mistral_d0 = res["D0"]["per_family_recall"]["Mistral"]
    mistral_d1 = res["D1"]["per_family_recall"]["Mistral"]

    # ---- 预注册判读（§3） ----
    close = lambda x: x >= b0_val["BA_F"] - CLOSE
    flags = {
        "rule1_warmstart_contrast_candidate": bool(close(res["D0"]["BA_F"]) and d01 >= 0.01),
        "rule2_no_practical_increment": bool(close(res["D0"]["BA_F"]) and d01 < 0.01),
        "rule3_warmstart_below_b0": bool(not close(res["D0"]["BA_F"])),
        "rule4_redistribution_mistral": bool(d01 < 0 and mistral_d1 > mistral_d0),
    }
    if flags["rule3_warmstart_below_b0"]:
        verdict = "③暖启动实现疑似问题：先查标准化/类别序/W_0 转置/logits（停止解释）"
    elif flags["rule1_warmstart_contrast_candidate"]:
        verdict = "①可达强基线附近，来源对比具有初步增量（可迁移跨 generator 实验）"
    elif flags["rule2_no_practical_increment"]:
        verdict = "②当前 SupCon 无实用增量（不再调 λ_C/τ/几何）"
    else:
        verdict = "未命中①②③（记录全部数字，另行分析）"
    if flags["rule4_redistribution_mistral"]:
        verdict += "；另注④：D1<D0 且 Mistral 召回上升——家族间再分配证据"
    say(f"[e33] 判读：Δ(D1−D0)={d01:+.4f}；Δ(D2−E32C2)={d2_rep_f:+.4f}/"
        f"{d2_rep_g:+.4f}；flags={flags} → {verdict}")

    # ---- 产物 ----
    np.savez_compressed(
        OUT / "predictions.npz",
        va_rows=rows[va_ai], va_split=sp[va_ai], va_family=y4_va.astype(np.int8),
        va_generator=np.array(gen_va), families_=np.array(FAM4),
        b0_pred=preds["B0"], d0_pred=preds["D0"], d1_pred=preds["D1"],
        d2_pred=preds["D2"],
        det_prob_val=det_va.astype(np.float32), det_y=det_y.astype(np.int8),
        va_rows_human=rows[hum_va])
    metrics = {
        "commit": commit, "smoke": a.smoke, "test_accessed": False,
        "B0": b0_val, "D0": res["D0"], "D1": res["D1"], "D2": res["D2"],
        "deltas": {"D1_minus_D0_BA_F": round(d01, 4),
                   "D1_minus_D0_BA_G": round(res["D1"]["BA_G"] - res["D0"]["BA_G"], 4),
                   "D2_vs_E32C2_BA_F": round(d2_rep_f, 4),
                   "D2_vs_E32C2_BA_G": round(d2_rep_g, 4),
                   "D0_minus_B0_BA_F": round(res["D0"]["BA_F"] - b0_val["BA_F"], 4)},
        "flags": flags, "verdict": verdict,
        "warmstart_logits_maxdiff": warm_diff,
        "det_auroc_frozen": round(det_auroc, 4),
        "batch_stats": {"first_last": stats, "pairs": pair_stats},
        "d0_d1_param_diff": param_diff,
        "mistral_recall": {"D0": mistral_d0, "D1": mistral_d1},
        "note": "D0/D1 共享初始状态；D2 复现 E32 随机初始化臂（仅参照）；"
                "C0w（E32）无来源对比项，不作为 H2 结果",
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=1))
    manifest = {
        "commit": commit, "whitelist": WL, "families": FAM4,
        "family_generators": fam_gens,
        "counts": {"ai_train": int(N), "ai_val": int(len(va_ai)),
                   "human_train": int(len(hum_tr)), "human_val": int(len(hum_va)),
                   "generator_train_counts": dict(gen_counts)},
        "features_sha256": feats_sha, "encoder_sha256": enc_sha,
        "pair_ratios": {"same_gen": pair_stats["same_gen"],
                        "cross_gen": pair_stats["cross_gen"],
                        "note": "跨 gen 正对仅来自 Mistral；D1 若收益只能称强模型集合上的"
                                "来源聚合候选"},
        "schedule": "与 E31/E32 相同（同 seed/同采样序）",
        "test_accessed": False, "unseen_accessed": False, "smoke": a.smoke,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    (OUT / "config.json").write_text(json.dumps({
        "commit": commit, "C_fixed": C_FIXED, "batch": BATCH, "lr": LR, "wd": WD,
        "tau": TAU, "lambda_C": LAM_C, "epochs": EPOCHS, "seed": STEPS_SEED,
        "arms": {"D0": "暖启动 CE（W_F=W_0,b_F=b_0；U,V 按 E32 初始化）",
                 "D1": "D0 同初始状态 + 0.1·SupCon",
                 "D2": "E32 随机初始化复现参照 + 0.1·SupCon"},
        "verdict_rule": "§3：close:= ≥B0−.02；①D0≈B0∧Δ(D1−D0)≥.01；②D0≈B0∧Δ<.01；"
                        "③D0≪B0（先查实现）；④D1<D0∧Mistral 召回↑=再分配",
    }, ensure_ascii=False, indent=1))
    (OUT / "train.log").write_text("\n".join(log_lines) + "\n")
    say(f"[e33] 完成 {time.time()-t_start:.1f}s；产物 {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
