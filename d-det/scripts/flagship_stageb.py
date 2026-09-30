#!/usr/bin/env python
"""阶段 B：固定主轴的来源残差（E0/E1/E2）——ACL 主线下一步唯一实验。

依据 `docx/d-det_ACL主线与阶段路径_2026-09-30.md` §6 阶段 B：
  · 固定主模型主轴 ℓ_0(a)=W_0·a+b_0（B0 收敛参数，冻结，不参与梯度）；
  · 门控来源残差：r(a)=V·GELU(Ua+b1)+b2；ℓ_F(a)=ℓ_0(a)+γ·W_R·r(a)；
    γ=0.1·tanh(η)，η 初值 0 ⇒ 残差零贡献起步；
  · E0=残差 CE；E1=E0 同初始状态+0.1·SupCon（cosine 于 r）；E2=冻结 γ=0 实现审计；
  · 冒烟发现字面门控（η=0 ⇒ γ≡0）使分支 CE 梯度恒为 0（∂ℓ_F/∂(U,V,W_R)=γ·…=0），
    η 仅以噪声级漂移——字面臂无法检验 H1。故增设门控修正臂（预注册）：
    E0g/E1g = 输出零初始化（W_R=0）+ η=0.5 起步（γ≈0.046）：初始贡献逐位为零
    （‖γW_Rr‖=0）且梯度全活；其余完全不变。若字面臂未点火（|γ|<5e-3 或参与度<1%），
    阶段 B 判读以 E1g−E0g 为准，E0/E1 作为审计记录。
  · 数据/协议沿用 E31–E33（五模型控制集、E28 raw、同批序、批 128、AdamW 1e-3/wd 1e-4/
    clip 1、2 epoch、检测头冻结）；不读 test/unseen。
  · 预注册判读：Δ=E1−E0（或修正臂 E1g−E0g）的 BA_F ≥1pt → 阶段 C 触发候选；
    <1pt → 无实用增量（不调 τ、不加几何变体）。附报 E0/E1 vs B0、BA_G、γ 终值、
    残差参与度、参数漂移、首末批 CE/L_C/梯度。

同时输出阶段 A 审计清单（stage_a_audit.json）：raw 768、B0 复现、访问边界、去重、编码器
重叠、24 generators 的 family/base-instruct/规模标签表（按名称标注）。

产物：artifacts/flagship_stageb/{config.json, manifest.json, metrics.json, predictions.npz,
train.log, stage_a_audit.json}（report.md 由报告复制）；smoke → artifacts/flagship_stageb_smoke；
默认拒绝覆盖。
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
from sklearn.metrics import roc_auc_score

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
GAMMA_SCALE = 0.1
DELTA_THRESH = 0.01

# 阶段 A：generator 角色/规模标签（按名称人工标注；数据集拼写"Codder"原样保留）
GEN_LABELS = {
    "GPT-4o": ("closed", "≥100B"),
    "deepseek-ai/DeepSeek-V3-0324": ("instruct", "671B-MoE"),
    "deepseek-ai/DeepSeek-R1": ("instruct-reasoning", "671B-MoE"),
    "deepseek-ai/deepseek-coder-1.3b-base": ("base", "1.3B"),
    "Qwen/Qwen2.5-Coder-1.5B": ("base", "1.5B"),
    "Qwen/Qwen2.5-Coder-1.5B-Instruct": ("instruct", "1.5B"),
    "Qwen/Qwen2.5-Codder-14B-Instruct": ("instruct", "14B（名称拼写 Codder 为数据集原样）"),
    "Qwen/Qwen2.5-72B-Instruct": ("instruct", "72B"),
    "Qwen/QwQ-32B": ("instruct-reasoning", "32B"),
    "meta-llama/Llama-3.1-8B": ("base", "8B"),
    "meta-llama/Llama-3.1-8B-Instruct": ("instruct", "8B"),
    "meta-llama/Llama-3.2-1B": ("base", "1B"),
    "meta-llama/Llama-3.2-3B": ("base", "3B"),
    "meta-llama/Llama-3.2-11B-Vision-Instruct": ("instruct", "11B"),
    "mistralai/Mistral-7B-Instruct-v0.3": ("instruct", "7B"),
    "mistralai/Devstral-Small-2505": ("instruct-agent", "24B"),
    "ibm-granite/granite-3.2-2b-instruct": ("instruct", "2B"),
    "ibm-granite/granite-3.3-8b-instruct": ("instruct", "8B"),
    "ibm-granite/granite-3.3-8b-base": ("base", "8B"),
    "01-ai/Yi-Coder-1.5B": ("base", "1.5B"),
    "01-ai/Yi-Coder-1.5B-Chat": ("chat", "1.5B"),
    "gemma-3-27b-it": ("instruct", "27B"),
    "gemma-3n-e4b-it": ("instruct", "~4B"),
    "google/codegemma-2b": ("base", "2B"),
}


def wl_match(g):
    return any(p in g for p in WL)


class GatedResidual(torch.nn.Module):
    """ℓ_F(a)=ℓ_0(a)+γ·W_R·r(a)；ℓ_0 冻结；r=V·GELU(Ua+b1)+b2；γ=0.1·tanh(η)。

    wr_zero=True（门控修正）：W_R 零初始化（初始贡献逐位为零但梯度全活）；
    eta0：η 初值（字面 0；修正 0.5）。
    """

    def __init__(self, W0, b0, d=768, k=4, wr_zero=False, eta0=0.0):
        super().__init__()
        self.register_buffer("W0", torch.as_tensor(W0, dtype=torch.float32))
        self.register_buffer("b0", torch.as_tensor(b0, dtype=torch.float32))
        self.U = torch.nn.Linear(d, d)
        self.V = torch.nn.Linear(d, d)
        self.WR = torch.nn.Linear(d, k)
        torch.nn.init.xavier_uniform_(self.U.weight)
        torch.nn.init.zeros_(self.U.bias)
        torch.nn.init.normal_(self.V.weight, std=1e-3)
        torch.nn.init.zeros_(self.V.bias)
        if wr_zero:
            torch.nn.init.zeros_(self.WR.weight)
        else:
            torch.nn.init.xavier_uniform_(self.WR.weight)
        torch.nn.init.zeros_(self.WR.bias)
        self.eta = torch.nn.Parameter(torch.full((1,), float(eta0)))

    def forward(self, a):
        l0 = a @ self.W0.t() + self.b0
        r = self.V(F.gelu(self.U(a)))
        gamma = GAMMA_SCALE * torch.tanh(self.eta)
        return r, l0 + gamma * self.WR(r), gamma


def supcon_cos(r, fam):
    rn = r / r.norm(dim=1, keepdim=True).clamp(min=1e-6)
    sim = rn @ rn.t() / TAU
    b = r.shape[0]
    self_mask = torch.eye(b, dtype=torch.bool, device=r.device)
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
    OUT = ROOT / "artifacts" / ("flagship_stageb_smoke" if a.smoke else "flagship_stageb")
    if OUT.exists() and any(OUT.iterdir()) and not a.force:
        print(f"[stageb] 已存在产物，默认拒绝覆盖：{OUT}（--force 可覆盖）", flush=True)
        return 2
    OUT.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    log_lines: list[str] = []

    def say(msg):
        print(msg, flush=True)
        log_lines.append(msg)

    commit = subprocess.run(["git", "-C", str(ROOT.parent), "rev-parse", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    say(f"[stageb] commit {commit} | smoke={a.smoke}")

    # ---- 数据（沿用 E31–E33；只读 b_train/b_val） ----
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
        say(f"[stageb] 特征 r0={R28_FEATS.name}（sha256 {feats_sha[:16]}…）| 编码器 "
            f"{enc_sha[:16]}…")

    sel_ai = np.array([i for i in np.where(y_ai)[0] if wl_match(str(gen[i]))])
    tr_ai = np.array([i for i in sel_ai if sp[i] == "train"])
    va_ai = np.array([i for i in sel_ai if sp[i] == "val"])
    hum_tr = np.where((~y_ai) & (sp == "train"))[0]
    hum_va = np.where((~y_ai) & (sp == "val"))[0]
    if not a.smoke:
        assert len(tr_ai) == 4013 and len(va_ai) == 1654, "子集计数不符"
    say(f"[stageb] 子集：AI train {len(tr_ai)} / val {len(va_ai)}")

    m_r, s_r = fit_scaler(raw[tr_ai])
    A_tr = apply_scaler(raw[tr_ai], m_r, s_r)
    A_va = apply_scaler(raw[va_ai], m_r, s_r)
    A_hum_tr = apply_scaler(raw[hum_tr], m_r, s_r)
    A_hum_va = apply_scaler(raw[hum_va], m_r, s_r)
    y4_tr = np.array([FAM4.index(SEEN_LABS[int(fam[i])]) for i in tr_ai])
    y4_va = np.array([FAM4.index(SEEN_LABS[int(fam[i])]) for i in va_ai])
    gen_va = [str(gen[i]) for i in va_ai]

    # ---- 检测头（冻结，同 E31–E33） ----
    det_lr = LogisticRegression(max_iter=MAX_ITER, tol=TOL, C=C_FIXED).fit(
        np.vstack([A_tr, A_hum_tr]), np.concatenate([np.ones(len(A_tr)),
                                                     np.zeros(len(A_hum_tr))]))
    det_va = det_lr.predict_proba(np.vstack([A_va, A_hum_va]))[:, 1]
    det_y = np.concatenate([np.ones(len(A_va)), np.zeros(len(A_hum_va))])
    det_auroc = float(roc_auc_score(det_y, det_va))
    say(f"[stageb] 检测头（冻结）：val AUROC {det_auroc:.4f}")

    # ---- B0 主轴（学 E31–E33 的权重/收敛） ----
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
    say(f"[stageb] B0 主轴：BA_F {b0_val['BA_F']} BA_G {b0_val['BA_G']}")

    # ---- 批调度（与 E31–E33 相同） ----
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
    say(f"[stageb] 批调度（同 E31–E33）：{steps} step/epoch×{EPOCHS}；正对 同gen "
        f"{pair_stats['same_gen']} / 跨gen {pair_stats['cross_gen']}")

    # ---- 模型（同臂共享初始状态；E2 同初始状态、η 冻结） ----
    W0, b0v = np.asarray(b0.coef_), np.asarray(b0.intercept_)
    torch.manual_seed(0)
    mE0 = GatedResidual(W0, b0v)
    mE1 = GatedResidual(W0, b0v)
    mE1.load_state_dict(copy.deepcopy(mE0.state_dict()))
    mE2 = GatedResidual(W0, b0v)
    mE2.load_state_dict(copy.deepcopy(mE0.state_dict()))
    mE2.eta.requires_grad_(False)                     # 冻结 γ=0 实现审计
    mE0g = GatedResidual(W0, b0v, wr_zero=True, eta0=0.5)
    mE1g = GatedResidual(W0, b0v, wr_zero=True, eta0=0.5)
    mE1g.load_state_dict(copy.deepcopy(mE0g.state_dict()))

    arms = {
        "E0": (mE0, torch.optim.AdamW(mE0.parameters(), lr=LR, weight_decay=WD), False),
        "E1": (mE1, torch.optim.AdamW(mE1.parameters(), lr=LR, weight_decay=WD), True),
        "E2": (mE2, torch.optim.AdamW([p for p in mE2.parameters()
                                       if p.requires_grad], lr=LR, weight_decay=WD),
               False),
        "E0g": (mE0g, torch.optim.AdamW(mE0g.parameters(), lr=LR, weight_decay=WD),
                False),
        "E1g": (mE1g, torch.optim.AdamW(mE1g.parameters(), lr=LR, weight_decay=WD),
                True),
    }
    init_state = {t: copy.deepcopy(dict(m.named_parameters()))
                  for t, (m, _, _) in arms.items()}
    A_tr_t = torch.as_tensor(A_tr)
    y4_tr_t = torch.as_tensor(y4_tr)
    stats = {k: [] for k in arms}
    gamma_track = {k: [] for k in arms}
    for ep in range(EPOCHS):
        for step in range(steps):
            bidx = schedule[ep * steps + step]
            xb = A_tr_t[bidx]; yb = y4_tr_t[bidx]
            for tag, (model, opt, use_c) in arms.items():
                r_, logits, gamma = model(xb)
                L_F = F.cross_entropy(logits, yb)
                L_C = None
                if use_c:
                    L_C, n_empty = supcon_cos(r_, yb)
                    pair_stats["empty_P"] += n_empty
                    loss = L_F + LAM_C * L_C
                else:
                    loss = L_F
                opt.zero_grad(); loss.backward()
                gnorm = float(torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0))
                opt.step()
                if step in (0, steps - 1):
                    stats[tag].append({"epoch": ep, "step": step,
                                       "ce": round(float(L_F.detach()), 4),
                                       "lc": round(float(L_C.detach()), 4)
                                       if L_C is not None else None,
                                       "grad_norm": round(gnorm, 3),
                                       "gamma": round(float(gamma.detach()), 5)})
        for tag, (model, _, _) in arms.items():
            gamma_track[tag].append(round(float(
                GAMMA_SCALE * torch.tanh(model.eta).detach()), 5))
        say(f"[stageb] epoch {ep} 末首/末批：" + " | ".join(
            f"{t} {[s for s in stats[t] if s['epoch'] == ep]}" for t in arms))

    # ---- 评估 ----
    res = {}
    preds = {"B0": b0.predict(A_va).astype(np.int8)}
    engage = {}
    for tag, (model, _, _) in arms.items():
        with torch.no_grad():
            r_, logits, gamma = model(torch.as_tensor(A_va))
            l0 = torch.as_tensor(A_va) @ model.W0.t() + model.b0
            corr = (gamma * model.WR(r_)).norm(dim=1).mean()
            base = l0.norm(dim=1).mean().clamp(min=1e-9)
            ratio = float(corr / base)
        pred = logits.argmax(1).numpy()
        res[tag] = fam_eval(pred, y4_va, FAM4, gen_va)
        preds[tag] = pred.astype(np.int8)
        engage[tag] = {"gamma_final": round(float(gamma), 5),
                       "correction_norm_ratio": round(ratio, 5)}
        say(f"[stageb] val·{tag}：BA_F {res[tag]['BA_F']} BA_G {res[tag]['BA_G']}"
            f" | γ={float(gamma):.5f} 修正比={ratio:.4f}")

    # 参数漂移（诊断死门/点火）
    drifts = {}
    for tag, (model, _, _) in arms.items():
        with torch.no_grad():
            drifts[tag] = {p: round(float((t.detach()
                                           - init_state[tag][p]).norm()), 5)
                           for p, t in model.named_parameters()}
    say("[stageb] 参数漂移 L2：" + " | ".join(
        f"{t} U={drifts[t]['U.weight']} V={drifts[t]['V.weight']} "
        f"WR={drifts[t]['WR.weight']} η={drifts[t]['eta']}" for t in arms))

    # E2 审计：ℓF≡ℓ0 ⇒ 与 B0 逐位一致
    with torch.no_grad():
        _, logits_e2, _ = mE2(torch.as_tensor(A_va))
        b0_df = b0.decision_function(A_va)
    e2_audit = {"logits_maxdiff_vs_B0": float(np.abs(logits_e2.numpy() - b0_df).max()),
                "BA_F_equal": bool(res["E2"]["BA_F"] == b0_val["BA_F"]),
                "gamma_frozen_zero": bool(float(GAMMA_SCALE * torch.tanh(mE2.eta)) == 0.0)}
    say(f"[stageb] E2 审计：logits vs B0 max|Δ|={e2_audit['logits_maxdiff_vs_B0']:.2e}；"
        f"BA_F 相等={e2_audit['BA_F_equal']}")

    d01 = res["E1"]["BA_F"] - res["E0"]["BA_F"]
    d01g = res["E1g"]["BA_F"] - res["E0g"]["BA_F"]
    with torch.no_grad():
        param_diff = {p: round(float((dict(mE0.named_parameters())[p]
                                      - dict(mE1.named_parameters())[p]).abs().max()), 6)
                      for p in ("U.weight", "V.weight", "WR.weight")}
    # ---- 预注册判读（点火检测 → 选基准臂） ----
    lit_engaged = bool(
        max(abs(engage[t]["gamma_final"]) for t in ("E0", "E1")) >= 5e-3 or
        max(engage[t]["correction_norm_ratio"] for t in ("E0", "E1")) >= 0.01)
    primary = "E1-E0" if lit_engaged else "E1g-E0g"
    d_primary = d01 if lit_engaged else d01g
    trigger = bool(d_primary >= DELTA_THRESH)
    verdict = (f"判读基准={primary}（字面臂{'已' if lit_engaged else '未'}点火）→ " +
               ("阶段 C 触发候选（≥1pt；单种子仅作候选证据）" if trigger else
                "无实用增量（不调 τ、不加几何变体；阶段 C 不启动）"))
    say(f"[stageb] 判读：Δ01={d01:+.4f}；Δ01g={d01g:+.4f}；"
        f"E0−B0={res['E0']['BA_F']-b0_val['BA_F']:+.4f}；"
        f"E0g−B0={res['E0g']['BA_F']-b0_val['BA_F']:+.4f} → {verdict}")

    # ---- 阶段 A 审计清单（generator 表覆盖全部已见家族 AI 样本，而非五模型子集） ----
    all_ai = np.where(y_ai)[0]
    gen_n_all = defaultdict(int)
    fam_of_gen = {}
    for i in all_ai:
        g = str(gen[i])
        gen_n_all[g] += 1
        fam_of_gen[g] = SEEN_LABS[int(fam[i])]
    gen_table = {g: {"family": fam_of_gen.get(g, "?"),
                     "role": GEN_LABELS.get(g, ("unknown", "unknown"))[0],
                     "size": GEN_LABELS.get(g, ("unknown", "unknown"))[1],
                     "n_train_val": gen_n_all[g]} for g in sorted(gen_n_all)}
    stage_a = {
        "raw_dim": 768,
        "B0_reproducibility": {"E31": 0.5879, "E32": 0.5879, "E33": 0.5879,
                               "this_run": b0_val["BA_F"],
                               "note": "五模型子集上跨轮完全一致（收敛逻辑回归）"},
        "access_boundary": "E28–E33 均只读 b_train/b_val；test/unseen 未触碰"
                           "（见各轮 manifest test_accessed=false）",
        "dedup": "E31：子集 ids 精确重复 0/0（剔除清单空）；E26 近重复记录在案"
                 "（Mistral-7B 同 gen 1 对）",
        "encoder_overlap": "E31 §4：v0.4.1 训练数据=m4/hybrid/pairs/pairs_qwen15；Task B "
                           "抽样 4000 条解码 0 命中；使用 v0.4.1 权重（适配件未用）",
        "generator_table": gen_table,
        "generator_table_note": "role/size 按模型名人工标注（closed=闭源）；"
                                "Qwen2.5-Codder-14B-Instruct 为数据集内拼写",
        "stage_a_exit": "可读性存在（B0 .5879）；可迁移性与混杂边界已在 E30/E31 记录",
    }
    (OUT / "stage_a_audit.json").write_text(json.dumps(stage_a, ensure_ascii=False,
                                                       indent=1))

    # ---- 产物 ----
    np.savez_compressed(
        OUT / "predictions.npz",
        va_rows=rows[va_ai], va_split=sp[va_ai], va_family=y4_va.astype(np.int8),
        va_generator=np.array(gen_va), families_=np.array(FAM4),
        b0_pred=preds["B0"], e0_pred=preds["E0"], e1_pred=preds["E1"],
        e2_pred=preds["E2"], e0g_pred=preds["E0g"], e1g_pred=preds["E1g"],
        det_prob_val=det_va.astype(np.float32), det_y=det_y.astype(np.int8),
        va_rows_human=rows[hum_va])
    metrics = {
        "commit": commit, "smoke": a.smoke, "test_accessed": False,
        "B0": b0_val, "E0": res["E0"], "E1": res["E1"], "E2": res["E2"],
        "E0g": res["E0g"], "E1g": res["E1g"],
        "deltas": {"E1_minus_E0_BA_F": round(d01, 4),
                   "E1g_minus_E0g_BA_F": round(d01g, 4),
                   "E1_minus_E0_BA_G": round(res["E1"]["BA_G"] - res["E0"]["BA_G"], 4),
                   "E1g_minus_E0g_BA_G": round(res["E1g"]["BA_G"] - res["E0g"]["BA_G"], 4),
                   "E0_minus_B0": round(res["E0"]["BA_F"] - b0_val["BA_F"], 4),
                   "E1_minus_B0": round(res["E1"]["BA_F"] - b0_val["BA_F"], 4),
                   "E0g_minus_B0": round(res["E0g"]["BA_F"] - b0_val["BA_F"], 4),
                   "E1g_minus_B0": round(res["E1g"]["BA_F"] - b0_val["BA_F"], 4)},
        "literal_gate_engaged": lit_engaged, "primary_comparison": primary,
        "trigger_stage_c": trigger, "verdict": verdict,
        "gamma_track": gamma_track, "engagement": engage, "param_drifts": drifts,
        "e2_audit": e2_audit, "det_auroc_frozen": round(det_auroc, 4),
        "batch_stats": {"first_last": stats, "pairs": pair_stats},
        "e0_e1_param_diff": param_diff,
        "note": "主轴 ℓ_0 全程冻结；γ=0.1·tanh(η)。字面臂 η0=0；门控修正臂 E0g/E1g："
                "W_R 零初始化+η0=0.5（初始贡献逐位为零且梯度全活）。E1 系 SupCon 作用于"
                "残差 r（τ=.1, λ_C=.1）。若字面臂未点火，阶段 B 判读以 E1g−E0g 为准。",
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=1))
    manifest = {
        "commit": commit, "whitelist": WL, "families": FAM4,
        "family_generators": fam_gens, "counts": {
            "ai_train": int(N), "ai_val": int(len(va_ai)),
            "human_train": int(len(hum_tr)), "human_val": int(len(hum_va))},
        "features_sha256": feats_sha, "encoder_sha256": enc_sha,
        "schedule": "与 E31–E33 相同（同 seed/同采样序）",
        "pair_ratios": {"same_gen": pair_stats["same_gen"],
                        "cross_gen": pair_stats["cross_gen"],
                        "note": "跨 gen 正对仅来自 Mistral；阶段 B 主损失为同家族正对"},
        "stage": "B（固定主轴来源残差）",
        "test_accessed": False, "unseen_accessed": False, "smoke": a.smoke,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    (OUT / "config.json").write_text(json.dumps({
        "commit": commit, "C_fixed": C_FIXED, "batch": BATCH, "lr": LR, "wd": WD,
        "tau": TAU, "lambda_C": LAM_C, "epochs": EPOCHS, "seed": STEPS_SEED,
        "gamma": "0.1*tanh(eta)",
        "model": "ℓ_F(a)=ℓ_0(a)+γ·W_R·r(a)；ℓ_0=W_0 a+b_0 冻结；"
                 "r=V·GELU(Ua+b1)+b2（U Xavier、V std=1e-3、b=0）",
        "arms": {"E0": "残差 CE（字面：η0=0，W_R Xavier）",
                 "E1": "残差 CE+0.1·SupCon(r)（字面）",
                 "E2": "冻结 γ=0 实现审计",
                 "E0g": "门控修正：W_R 零初始化+η0=0.5（初始贡献逐位为零、梯度全活）",
                 "E1g": "门控修正+0.1·SupCon(r)"},
        "verdict_rule": "点火检测：若字面臂 |γ|<5e-3 且参与度<1%（死门），基准改为 "
                        f"E1g−E0g；Δ基准（BA_F）≥{DELTA_THRESH} → 阶段 C 触发候选；"
                        "否则无实用增量",
    }, ensure_ascii=False, indent=1))
    (OUT / "train.log").write_text("\n".join(log_lines) + "\n")
    say(f"[stageb] 完成 {time.time()-t_start:.1f}s；产物 {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
