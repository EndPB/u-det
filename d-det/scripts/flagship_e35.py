#!/usr/bin/env python
"""E35：论文级 H2 直接检验——E30 全 generator 关系集上的固定主轴来源残差（F0/F1）。

依据 `docx/d-det_E35论文级H2直接检验指导_2026-09-30.md`（基线 25824ca）：
  · 数据/折：沿用 E30 七家族（排除单 generator 的 OpenAI）/23 generators；每家族
    RandomState(1000+fi) 打乱 generator：2→1/1、3→1/2、5→2/3；fold0 val=A/train=B、
    fold1 互补；两侧均有 generator 且完全不相交（断言）。非冒烟与 E30 manifest
    fold_generators / metrics 计数比对一致。
  · 主模型：固定 B0 主轴 ℓ0(z)=W0z+b0（加权 LR，w=N/(K·|G_f|·N_g)，C=.1，逐折拟合后
    **全程冻结**）；来源残差 r=V·GELU(Uz+b1)+b2；ℓ=ℓ0+γ·W_R·r；**阶段 B 修正门控
    初始化**（W_R=0、η0=0.5、γ=0.1·tanh(η)、U Xavier、V std=1e-3、b=0）。
  · 臂：F0=L_F（残差 CE）；F1=L_F+0.1·L_cross（τ=.1 只除一次；正对=同家族不同
    generator；无正对 anchor 跳过并报有效比例；分母含全批；不混入同 generator 正对）。
  · 协议：批 7 族×18；E31 式族→generator→样本均匀采样；2 epoch；AdamW 1e-3/wd 1e-4/
    clip 1；F0/F1 共享初始化与批序；标准化仅训练折（std≥1e-2 下限）；F0/F1 只差 L_cross。
  · 指标：BA_F/BA_G（机会 1/7）、逐折差值、pooled generator-held-out、逐家族/逐
    generator 召回、role/size 附报、检测 AUROC（E30 协议）；随机切分（E30 同本控制）
    仅作迁移损失参照，不作为 H2 门槛。
  · 预注册出口（config.json exit_rule）：折方向一致性 + pooled 阈值 + anchor guard。
  · 不读 test/unseen；不调门控/τ/λ_C/lr/轮数。

产物：artifacts/flagship_e35/{config,manifest,metrics,predictions.npz,solver.log}
（report.md 由报告复制）；smoke → artifacts/flagship_e35_smoke；默认拒绝覆盖。
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
from sklearn.metrics import f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from flagship_round1 import SEEN_LABS  # noqa: E402
from flagship_e28_probe import (load_train_val, ids_md5, fit_scaler,  # noqa: E402
                                apply_scaler, default_args)

R28_FEATS = ROOT / "runs/flagship_e28/features_train_val.npz"
E30_MANIFEST = ROOT / "artifacts/flagship_e30/manifest.json"
E30_METRICS = ROOT / "artifacts/flagship_e30/metrics.json"
C_FIXED = 0.1
STD_FLOOR = 1e-2
MAX_ITER = 10000
TOL = 1e-6
PER_FAM = 18
LR = 1e-3
WD = 1e-4
TAU = 0.1
LAM_C = 0.1
EPOCHS = 2
GAMMA_SCALE = 0.1
CHANCE = 1.0 / 7.0
DELTA_THRESH = 0.01
ANCHOR_MIN = 0.15
NEAR_CHANCE = 0.20
FAM7 = [f for f in SEEN_LABS if f != "OpenAI"]
LUT7 = {int(SEEN_LABS.index(f)): k for k, f in enumerate(FAM7)}

# role/规模标签（与阶段 B 相同；按模型名人工标注；数据集拼写"Codder"原样保留）
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


class GatedResidual(torch.nn.Module):
    """ℓ=ℓ0+γ·W_R·r；ℓ0 冻结；r=V·GELU(Uz+b1)+b2；γ=0.1·tanh(η)（阶段 B 修正门控）。"""

    def __init__(self, W0, b0, d=768, k=7, wr_zero=True, eta0=0.5):
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

    def forward(self, z):
        l0 = z @ self.W0.t() + self.b0
        r = self.V(F.gelu(self.U(z)))
        gamma = GAMMA_SCALE * torch.tanh(self.eta)
        return r, l0 + gamma * self.WR(r), gamma


def cross_pos(fb, gb):
    """同家族、不同 generator、非自身的正对掩码与计数（批内）。"""
    b = fb.shape[0]
    self_mask = torch.eye(b, dtype=torch.bool, device=fb.device)
    pos = (fb[:, None] == fb[None, :]) & (gb[:, None] != gb[None, :]) & ~self_mask
    return pos, pos.sum(1)


def supcon_cross_from(r, pos, cnt):
    """L_{C,cross}：cos/τ 只除一次；分母含全批；仅有效 anchor 参与均值。"""
    rn = r / r.norm(dim=1, keepdim=True).clamp(min=1e-6)
    sim = rn @ rn.t() / TAU
    sim = sim.masked_fill(torch.eye(r.shape[0], dtype=torch.bool,
                                    device=r.device), -1e9)
    logz = torch.logsumexp(sim, dim=1)
    valid = cnt > 0
    if not bool(valid.any()):
        return None, 0
    lp = ((sim - logz[:, None]) * pos).sum(1) / cnt.clamp(min=1)
    return -lp[valid].mean(), int(valid.sum())


def fam_eval(pred, y, fams_used, gen_list):
    rec_f, rec_g = {}, {}
    for k, f in enumerate(fams_used):
        m = y == k
        rec_f[f] = round(float((pred[m] == k).mean()), 4)
        gv = defaultdict(list)
        for j in np.where(m)[0]:
            gv[gen_list[j]].append(float(pred[j] == k))
        rec_g[f] = {g: round(float(np.mean(v)), 4) for g, v in gv.items()}
    BA_F = float(np.mean([rec_f[f] for f in fams_used]))
    BA_G = float(np.mean([np.mean(list(rec_g[f].values())) for f in fams_used]))
    return {"BA_F": round(BA_F, 4), "BA_G": round(BA_G, 4),
            "per_family_recall": rec_f, "per_generator_recall": rec_g}


def role_breakdown(rec_g):
    by = defaultdict(list)
    for f, gd in rec_g.items():
        for g, r in gd.items():
            by[GEN_LABELS.get(g, ("unknown", ""))[0]].append(r)
    return {k: round(float(np.mean(v)), 4) for k, v in sorted(by.items())}


def fit_lr(X, y, w=None, tag="", say=print):
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter("always")
        clf = LogisticRegression(max_iter=MAX_ITER, tol=TOL, C=C_FIXED).fit(
            X, y, sample_weight=w)
    conv = not any(issubclass(x.category, ConvergenceWarning) for x in wl)
    say(f"[e35] {tag}: 收敛={conv} n_iter={int(np.max(clf.n_iter_))}")
    return clf, conv


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    args = default_args(smoke=a.smoke)
    OUT = ROOT / "artifacts" / ("flagship_e35_smoke" if a.smoke else "flagship_e35")
    if OUT.exists() and any(OUT.iterdir()) and not a.force:
        print(f"[e35] 已存在产物，默认拒绝覆盖：{OUT}（--force 可覆盖）", flush=True)
        return 2
    OUT.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    solver_log: list[str] = []

    def say(msg):
        print(msg, flush=True)
        solver_log.append(msg)

    commit = subprocess.run(["git", "-C", str(ROOT.parent), "rev-parse", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    say(f"[e35] commit {commit} | smoke={a.smoke} | device={device}")

    # ---- 语料（只读 b_train/b_val）与特征 ----
    corpus = load_train_val(args)
    docs_all = corpus["train"] + corpus["val"]
    y_ai = np.array([d.y_ai for d in docs_all], bool)
    fam = np.array([SEEN_LABS.index(d.fam) if d.fam in SEEN_LABS else -1
                    for d in docs_all], np.int16)
    gen = np.array([d.gen for d in docs_all])
    lang = np.array([d.lang for d in docs_all])
    ntok = np.array([len(d.ids) for d in docs_all], np.int32)
    sp = np.array([d.split for d in docs_all])
    rows = np.array([d.row for d in docs_all], np.int64)

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
        dual.load_state_dict(torch.load(ROOT / "runs/v0.4.1_covreg/last.pt",
                                        map_location="cpu",
                                        weights_only=False)["state"], strict=False)
        enc = dual.encoder.to(device).eval()
        head = SetPool(768, 512).to(device)
        head.load_state_dict(torch.load(ROOT / "runs/flagship_r1/head.pt",
                                        map_location=device)["head"])
        head.eval()
        wq = torch.as_tensor(ms["wq"], device=device)
        bq = torch.as_tensor(ms["bq"], device=device)
        m_tT = torch.as_tensor(ms["m_t"], dtype=torch.float32, device=device)
        s_tT = torch.as_tensor(ms["s_t"], dtype=torch.float32, device=device)
        raw_tr, _, _ = export_pass(enc, head, corpus["train"], args.batch, device,
                                   wq, bq, m_tT, s_tT)
        raw_va, _, _ = export_pass(enc, head, corpus["val"], args.batch, device,
                                   wq, bq, m_tT, s_tT)
        raw = np.vstack([raw_tr, raw_va])
        feats_sha = None
        say(f"[e35] smoke：自建特征 AI={int(y_ai.sum())} / Human={int((~y_ai).sum())}")
    else:
        feats_sha = hashlib.sha256(R28_FEATS.read_bytes()).hexdigest()
        fz = dict(np.load(R28_FEATS, allow_pickle=True))
        m_tr = np.array([ids_md5(d.ids) for d in corpus["train"]])
        m_va = np.array([ids_md5(d.ids) for d in corpus["val"]])
        assert np.array_equal(m_tr, fz["md5_train"]) and \
            np.array_equal(m_va, fz["md5_val"]), "清单与 E28 特征不符"
        assert len(corpus["train"]) == 17600 and len(corpus["val"]) == 5372
        raw = np.vstack([fz["raw_train"], fz["raw_val"]])
        say(f"[e35] 特征 r0={R28_FEATS.name}（sha256 {feats_sha[:16]}…）")

    ai7_idx = np.array([i for i in np.where(y_ai)[0] if int(fam[i]) in LUT7])
    hum_idx = np.where(~y_ai)[0]
    assert raw.shape[1] == 768

    # ---- 家族 generator 分配（与 E30 完全一致） ----
    gen_split = {}
    for f in FAM7:
        fi = SEEN_LABS.index(f)
        gens = sorted({str(gen[i]) for i in ai7_idx if fam[i] == fi})
        rs = np.random.RandomState(1000 + fi)
        rs.shuffle(gens)
        m = len(gens)
        if a.smoke and m < 2:
            say(f"[e35] smoke：家族 {f} 仅 {m} 个 generator，跳过（全量不会发生）")
            continue
        nA = 1 if m in (2, 3) else (2 if m == 5 else max(1, m // 2))
        A, B = gens[:nA], gens[nA:]
        assert len(A) >= 1 and len(B) >= 1, f"{f} 分配后空侧"
        gen_split[f] = {"A": A, "B": B}
    fams_used = list(gen_split.keys())
    if not a.smoke:
        assert len(fams_used) == 7
        e30m = json.loads(E30_MANIFEST.read_text())
        for fold in (0, 1):
            for f in fams_used:
                ours = {k: sorted(gen_split[f]["A" if (fold == 0) == (k == "val") else "B"])
                        for k in ("val", "train")}
                e30f = e30m["fold_generators"][f"fold{fold}"][f]
                assert ours["val"] == sorted(e30f["val"]) and \
                    ours["train"] == sorted(e30f["train"]), \
                    f"fold{fold} {f} generator 清单与 E30 不一致"
        say("[e35] 折 generator 清单与 E30 manifest 比对一致 ✓")
    say(f"[e35] 七家族 AI {len(ai7_idx)}；generators {len(set(gen[ai7_idx].tolist()))}；"
        f"Human {len(hum_idx)}")

    # ---- Human 二折（E30 协议：RandomState(1)，两折共用） ----
    ph = np.random.RandomState(1).permutation(len(hum_idx))
    half = len(hum_idx) // 2
    hum_fold_val = {0: ph[:half], 1: ph[half:]}
    hum_fold_tr = {0: ph[half:], 1: ph[:half]}

    def fold_masks(fold):
        sel = {f: set(gen_split[f]["A" if fold == 0 else "B"]) for f in fams_used}
        is_val = np.array([str(gen[i]) in sel[SEEN_LABS[int(fam[i])]]
                           for i in ai7_idx])
        return ~is_val, is_val

    # ---- 逐折训练与评估 ----
    e30m_metrics = json.loads(E30_METRICS.read_text()) if not a.smoke else None
    fold_out = {}
    preds_out = {}
    pool_rec = {k: {"pred": [], "y": [], "gen": [], "fam": []} for k in
                ("main", "F0", "F1")}
    for fold in (0, 1):
        tr_m, va_m = fold_masks(fold)
        atr = ai7_idx[tr_m]; ava = ai7_idx[va_m]
        if not a.smoke:
            e30r = e30m_metrics["arms"]["R0_mraw"]["gen"][f"fold{fold}"]
            assert len(atr) == e30r["n_ai_train"] and len(ava) == e30r["n_ai_val"], \
                "折样本数与 E30 不一致"
        y7_tr = np.array([LUT7[int(fam[i])] for i in atr], np.int64)
        y7_va = np.array([LUT7[int(fam[i])] for i in ava], np.int64)
        gen_tr = gen[atr]; gen_va = gen[ava]
        fam_tr_name = np.array([SEEN_LABS[int(fam[i])] for i in atr])
        say(f"[e35] fold{fold}: AI train {len(atr)} / val {len(ava)}")

        m_f, s_f = fit_scaler(raw[atr])
        Z_tr = apply_scaler(raw[atr], m_f, s_f)
        Z_va = apply_scaler(raw[ava], m_f, s_f)

        # 主分类器：加权重加权 B0 协议（逐折拟合后冻结）
        fam_gens = {f: sorted({str(gen_tr[i]) for i in range(len(gen_tr))
                               if fam_tr_name[i] == f}) for f in fams_used}
        gc = {g: int((gen_tr == g).sum()) for f in fams_used for g in fam_gens[f]}
        w_sample = np.array([len(atr) / (len(fams_used) * len(fam_gens[fn]) * gc[g])
                             for fn, g in zip(fam_tr_name, gen_tr)])
        b0, b0_conv = fit_lr(Z_tr, y7_tr, w_sample, f"fold{fold} B0 主轴（加权）", say)
        main_va = fam_eval(b0.predict(Z_va), y7_va, fams_used, list(gen_va))
        b0u, _ = fit_lr(Z_tr, y7_tr, None, f"fold{fold} 参照（未加权，对齐 E30 R0）", say)
        ref_va = fam_eval(b0u.predict(Z_va), y7_va, fams_used, list(gen_va))
        say(f"[e35] fold{fold} 主轴：BA_F {main_va['BA_F']} / BA_G {main_va['BA_G']}"
            f"（参照未加权 {ref_va['BA_F']}/{ref_va['BA_G']}）")

        # 检测（E30 协议：全训练侧标准化 + C=.1）
        htr = hum_idx[hum_fold_tr[fold]]; hva = hum_idx[hum_fold_val[fold]]
        tr_all = np.concatenate([atr, htr]); va_all = np.concatenate([ava, hva])
        m_d, s_d = fit_scaler(raw[tr_all])
        Zd_tr = apply_scaler(raw[tr_all], m_d, s_d)
        Zd_va = apply_scaler(raw[va_all], m_d, s_d)
        clf_d, _ = fit_lr(Zd_tr, y_ai[tr_all], None, f"fold{fold} 检测头", say)
        p_d = clf_d.predict_proba(Zd_va)[:, 1]
        det_auroc = float(roc_auc_score(y_ai[va_all], p_d))
        det_f1 = float(f1_score(y_ai[va_all], p_d > 0.5, average="macro"))

        # 批调度（E31 式：族→generator→样本均匀；F0/F1 共用）
        gen_of = {f: {} for f in fams_used}
        for f in fams_used:
            for g in fam_gens[f]:
                gen_of[f][g] = np.where(gen_tr == g)[0]
        batch_size = PER_FAM * len(fams_used)
        steps = math.ceil(len(atr) / batch_size)
        rng = np.random.RandomState(0)
        schedule, redraw = [], 0
        for _ in range(steps * EPOCHS):
            batch = []
            for f in fams_used:
                while True:
                    draws = []
                    for _ in range(PER_FAM):
                        g = fam_gens[f][rng.randint(len(fam_gens[f]))]
                        draws.append(int(gen_of[f][g][rng.randint(len(gen_of[f][g]))]))
                    if len(set(draws)) >= 2 or len(fam_gens[f]) < 2:
                        break
                    redraw += 1
                batch += draws
            schedule.append(np.array(batch))
        say(f"[e35] fold{fold} 调度：{steps} step/epoch×{EPOCHS}（批 {batch_size}，"
            f"重抽 {redraw}）")

        # 模型：F0/F1 共享初始状态（阶段 B 修正门控）
        W0, b0v = np.asarray(b0.coef_), np.asarray(b0.intercept_)
        torch.manual_seed(0)
        mF0 = GatedResidual(W0, b0v)
        mF1 = GatedResidual(W0, b0v)
        mF1.load_state_dict(copy.deepcopy(mF0.state_dict()))
        Z_va_t = torch.as_tensor(Z_va)
        with torch.no_grad():
            _, logits_init, _ = mF0(Z_va_t)
            l0_val = Z_va_t @ mF0.W0.t() + mF0.b0
            init_maxdiff = float((logits_init - l0_val).abs().max())
        assert init_maxdiff == 0.0, "W_R=0 初始 ℓ≠ℓ0"
        init_audit = {"logits_maxdiff_vs_main_axis": init_maxdiff,
                      "note": "W_R 零初始化 ⇒ 训练前 ℓ≡ℓ0（逐位）"}
        say(f"[e35] fold{fold} 初始化审计：ℓ 与主轴 max|Δ|={init_maxdiff:.2e} ✓")

        arms = {"F0": (mF0, torch.optim.AdamW(mF0.parameters(), lr=LR, weight_decay=WD)),
                "F1": (mF1, torch.optim.AdamW(mF1.parameters(), lr=LR, weight_decay=WD))}
        Z_tr_t = torch.as_tensor(Z_tr)
        y7_tr_t = torch.as_tensor(y7_tr)
        fam7_tr_t = torch.as_tensor(y7_tr)
        gcode = {g: c for c, g in enumerate(sorted(set(gen_tr.tolist())))}
        gint_tr = np.array([gcode[str(g)] for g in gen_tr], np.int64)
        gint_tr_t = torch.as_tensor(gint_tr)

        stats = {k: [] for k in arms}
        gamma_track = {k: [] for k in arms}
        anchor = {"total": 0, "valid": 0,
                  "per_family": {f: {"total": 0, "valid": 0} for f in fams_used}}
        for ep in range(EPOCHS):
            for step in range(steps):
                bidx = schedule[ep * steps + step]
                xb = Z_tr_t[bidx]; yb = y7_tr_t[bidx]
                fb = fam7_tr_t[bidx]; gb = gint_tr_t[bidx]
                with torch.no_grad():
                    pos, cnt = cross_pos(fb, gb)
                    cnt_np = cnt.numpy()
                    valid_np = cnt_np > 0
                    anchor["total"] += len(bidx)
                    anchor["valid"] += int(valid_np.sum())
                    for k, f in enumerate(fams_used):
                        msk = y7_tr[bidx] == k
                        anchor["per_family"][f]["total"] += int(msk.sum())
                        anchor["per_family"][f]["valid"] += int((msk & valid_np).sum())
                for tag, (model, opt) in arms.items():
                    r_, logits, gamma = model(xb)
                    L_F = F.cross_entropy(logits, yb)
                    L_C = None
                    if tag == "F1":
                        L_C, _ = supcon_cross_from(r_, pos, cnt)
                    loss = L_F if L_C is None else L_F + LAM_C * L_C
                    opt.zero_grad(); loss.backward()
                    gnorm = float(torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad], 1.0))
                    opt.step()
                    if step in (0, steps - 1):
                        stats[tag].append({
                            "epoch": ep, "step": step,
                            "ce": round(float(L_F.detach()), 4),
                            "lc": None if L_C is None else round(float(L_C.detach()), 4),
                            "grad_norm": round(gnorm, 3),
                            "gamma": round(float(gamma.detach()), 5)})
            for tag, (model, _) in arms.items():
                gamma_track[tag].append(round(float(
                    GAMMA_SCALE * torch.tanh(model.eta).detach()), 5))
            say(f"[e35] fold{fold} epoch{ep} 末：" + " | ".join(
                f"{t} {[s for s in stats[t] if s['epoch'] == ep]}" for t in arms))

        # 评估（val 侧）
        res = {"main": main_va}
        pred_store = {"main": b0.predict(Z_va).astype(np.int8)}
        engage = {}
        with torch.no_grad():
            for tag, (model, _) in arms.items():
                r_, logits, gamma = model(Z_va_t)
                corr = (gamma * model.WR(r_)).norm(dim=1).mean()
                base = l0_val.norm(dim=1).mean().clamp(min=1e-9)
                engage[tag] = {"gamma_final": round(float(gamma), 5),
                               "correction_norm_ratio": round(float(corr / base), 5)}
                pred = logits.argmax(1).numpy()
                res[tag] = fam_eval(pred, y7_va, fams_used, list(gen_va))
                pred_store[tag] = pred.astype(np.int8)
        drifts = {}
        for tag, (model, _) in arms.items():
            with torch.no_grad():
                drifts[tag] = {p: round(float(t.detach().norm()), 4)
                               for p, t in model.named_parameters()
                               if p in ("U.weight", "V.weight", "WR.weight")}
        with torch.no_grad():
            diff_dict = {p: round(float((dict(mF0.named_parameters())[p]
                                         - dict(mF1.named_parameters())[p]).abs().max()), 6)
                         for p in ("U.weight", "V.weight", "WR.weight")}

        dF = round(res["F1"]["BA_F"] - res["F0"]["BA_F"], 4)
        dG = round(res["F1"]["BA_G"] - res["F0"]["BA_G"], 4)
        anchor["overall_ratio"] = round(anchor["valid"] / max(anchor["total"], 1), 4)
        for f in fams_used:
            td = anchor["per_family"][f]["total"]
            anchor["per_family"][f]["ratio"] = round(
                anchor["per_family"][f]["valid"] / max(td, 1), 4)
        say(f"[e35] fold{fold} 结果：F0 {res['F0']['BA_F']}/{res['F0']['BA_G']}，"
            f"F1 {res['F1']['BA_F']}/{res['F1']['BA_G']}，Δ={dF:+.4f}/{dG:+.4f}；"
            f"anchor {anchor['valid']}/{anchor['total']}="
            f"{anchor['overall_ratio']:.3f}；det {det_auroc:.4f}")

        # 随机切分参照（E30 同本控制；仅主轴，拟合后评估）
        atr_l, ava_l = [], []
        for f in fams_used:
            fi = SEEN_LABS.index(f)
            fidx = np.array([i for i in ai7_idx if fam[i] == fi])
            n_va = int(va_m[np.isin(ai7_idx, fidx)].sum())
            rs = np.random.RandomState(200 + fold * 10 + fi)
            perm = rs.permutation(len(fidx))
            ava_l += fidx[perm[:n_va]].tolist()
            atr_l += fidx[perm[n_va:]].tolist()
        atr_r = np.array(sorted(atr_l)); ava_r = np.array(sorted(ava_l))
        y7_rtr = np.array([LUT7[int(fam[i])] for i in atr_r], np.int64)
        y7_rva = np.array([LUT7[int(fam[i])] for i in ava_r], np.int64)
        fn_r = np.array([SEEN_LABS[int(fam[i])] for i in atr_r])
        g_r = gen[atr_r]
        fg_r = {f: sorted({str(g_r[i]) for i in range(len(g_r))
                           if fn_r[i] == f}) for f in fams_used}
        gc_r = {g: int((g_r == g).sum()) for f in fams_used for g in fg_r[f]}
        w_r = np.array([len(atr_r) / (len(fams_used) * len(fg_r[fn]) * gc_r[g])
                        for fn, g in zip(fn_r, g_r)])
        m_r, s_r = fit_scaler(raw[atr_r])
        b0r, _ = fit_lr(apply_scaler(raw[atr_r], m_r, s_r), y7_rtr, w_r,
                        f"fold{fold} 随机参照主轴", say)
        ref_rand = fam_eval(b0r.predict(apply_scaler(raw[ava_r], m_r, s_r)),
                            y7_rva, fams_used, list(gen[ava_r]))
        say(f"[e35] fold{fold} 随机参照：BA_F {ref_rand['BA_F']} / BA_G "
            f"{ref_rand['BA_G']}（对照 gen {main_va['BA_F']}/{main_va['BA_G']}）")

        fold_out[f"fold{fold}"] = {
            "n_ai_train": len(atr), "n_ai_val": len(ava),
            "fold_generators": {f: {"val": sorted(gen_split[f]["A" if fold == 0 else "B"]),
                                    "train": sorted(gen_split[f]["B" if fold == 0 else "A"])}
                                for f in fams_used},
            "main_axis": main_va, "main_axis_unweighted_ref": ref_va,
            "main_b0_converged": b0_conv,
            "F0": res["F0"], "F1": res["F1"],
            "deltas": {"F1_minus_F0_BA_F": dF, "F1_minus_F0_BA_G": dG},
            "det_auroc": round(det_auroc, 4), "det_f1@.5": round(det_f1, 4),
            "anchor": anchor, "engagement": engage, "param_drifts": drifts,
            "F0_F1_param_diff": diff_dict, "init_audit": init_audit,
            "batch_stats": {"first_last": stats, "pairs_redraw": redraw},
            "gamma_track": gamma_track,
            "random_ref": {"BA_F": ref_rand["BA_F"], "BA_G": ref_rand["BA_G"],
                           "note": "同本随机控制（E30 协议）；仅主轴、不训练残差；参照用"},
        }
        # 逐样本预测保存（val 侧，含 doc/role/size/lang/len/classes）
        preds_out[f"f{fold}_rows"] = rows[ava]
        preds_out[f"f{fold}_split"] = sp[ava]
        preds_out[f"f{fold}_family"] = np.array([LUT7[int(fam[i])] for i in ava], np.int8)
        preds_out[f"f{fold}_generator"] = gen_va
        preds_out[f"f{fold}_role"] = np.array(
            [GEN_LABELS.get(str(g), ("unknown", ""))[0] for g in gen_va])
        preds_out[f"f{fold}_size"] = np.array(
            [GEN_LABELS.get(str(g), ("unknown", ""))[1] for g in gen_va])
        preds_out[f"f{fold}_language"] = lang[ava]
        preds_out[f"f{fold}_length"] = ntok[ava]
        preds_out[f"f{fold}_classes"] = np.array(fams_used)
        for tag in ("main", "F0", "F1"):
            preds_out[f"f{fold}_pred_{tag}"] = pred_store[tag]
            pool_rec[tag]["pred"].append(pred_store[tag])
            pool_rec[tag]["y"].append(y7_va)
            pool_rec[tag]["gen"].append(gen_va)
            pool_rec[tag]["fam"].append(np.array([LUT7[int(fam[i])] for i in ava]))

    # ---- pool（两折 val 互补：每个 generator 恰被留出一次） ----
    pooled = {}
    for tag in ("main", "F0", "F1"):
        pred = np.concatenate(pool_rec[tag]["pred"])
        y = np.concatenate(pool_rec[tag]["y"])
        g = np.concatenate(pool_rec[tag]["gen"])
        pooled[tag] = fam_eval(pred, y, fams_used, list(g))
        pooled[tag]["role_recall"] = role_breakdown(pooled[tag]["per_generator_recall"])
    pooled["deltas"] = {
        "F1_minus_F0_BA_F": round(pooled["F1"]["BA_F"] - pooled["F0"]["BA_F"], 4),
        "F1_minus_F0_BA_G": round(pooled["F1"]["BA_G"] - pooled["F0"]["BA_G"], 4),
    }
    say(f"[e35] pooled：主轴 {pooled['main']['BA_F']}/{pooled['main']['BA_G']}；"
        f"F0 {pooled['F0']['BA_F']}/{pooled['F0']['BA_G']}；"
        f"F1 {pooled['F1']['BA_F']}/{pooled['F1']['BA_G']}；"
        f"Δ {pooled['deltas']['F1_minus_F0_BA_F']:+.4f}/"
        f"{pooled['deltas']['F1_minus_F0_BA_G']:+.4f}")

    # ---- 预注册出口 ----
    dF0 = fold_out["fold0"]["deltas"]["F1_minus_F0_BA_F"]
    dG0 = fold_out["fold0"]["deltas"]["F1_minus_F0_BA_G"]
    dF1 = fold_out["fold1"]["deltas"]["F1_minus_F0_BA_F"]
    dG1 = fold_out["fold1"]["deltas"]["F1_minus_F0_BA_G"]
    pdF = pooled["deltas"]["F1_minus_F0_BA_F"]
    pdG = pooled["deltas"]["F1_minus_F0_BA_G"]
    fold_positive = bool(dF0 > 0 and dG0 > 0 and dF1 > 0 and dG1 > 0)
    anchor_ok = bool(fold_out["fold0"]["anchor"]["overall_ratio"] >= ANCHOR_MIN and
                     fold_out["fold1"]["anchor"]["overall_ratio"] >= ANCHOR_MIN)
    pass_strict = bool(fold_positive and pdF >= DELTA_THRESH and pdG >= DELTA_THRESH)
    pass_primary = bool(fold_positive and pdF >= DELTA_THRESH)
    near_chance = bool(pooled["F0"]["BA_F"] < NEAR_CHANCE and
                       pooled["F1"]["BA_F"] < NEAR_CHANCE)
    if not anchor_ok:
        outcome = "anchor_guard：有效 anchor 比例过低 ⇒ 未充分检验（不得写成阴性）"
    elif pass_strict:
        outcome = "1：H2 初步支持（两折方向一致且 pooled ΔBA_F、ΔBA_G 均 ≥1pt）→ 可进入一次 DMHM 式局部几何消融"
    elif pass_primary:
        outcome = "1-边界：两折方向一致且 pooled ΔBA_F≥1pt，但 pooled ΔBA_G<1pt ⇒ 单列待指导端裁定"
    elif fold_positive and (pdF > 0 or pdG > 0):
        outcome = "2：方向一致但 pooled 增量 <1pt ⇒ H2 在当前 768 维表示与跨 generator 正对定义下不支持；停止几何损失路线"
    else:
        outcome = "2：方向不一致或增量为非正 ⇒ H2 不支持；停止几何损失路线"
    exit_info = {
        "rule": {"fold_positive": "F1−F0 的 BA_F 与 BA_G 在两折均 >0",
                 "pooled_bar": f"pooled ΔBA_F ≥ {DELTA_THRESH}（主判据）且 "
                               f"pooled ΔBA_G ≥ {DELTA_THRESH}（标准出口并列条件）",
                 "anchor_guard": f"任一折整体 anchor 比例 < {ANCHOR_MIN} ⇒ 未充分检验",
                 "near_chance": f"pooled BA_F(F0) 与 BA_F(F1) 均 < {NEAR_CHANCE} ⇒ 数据不足出口"},
        "fold_deltas": {"fold0": {"BA_F": dF0, "BA_G": dG0},
                        "fold1": {"BA_F": dF1, "BA_G": dG1}},
        "pooled_deltas": {"BA_F": pdF, "BA_G": pdG},
        "fold_positive": fold_positive, "anchor_ok": anchor_ok,
        "pass_strict": pass_strict, "pass_primary": pass_primary,
        "near_chance": near_chance, "outcome": outcome}
    say(f"[e35] 出口：fold_positive={fold_positive} anchor_ok={anchor_ok} "
        f"pass_strict={pass_strict} pass_primary={pass_primary} → {outcome}")

    # ---- 产物 ----
    key_lines = "\n".join(f"{d.split}:{d.row}:{ids_md5(d.ids)}" for d in docs_all)
    metrics = {
        "commit": commit, "smoke": a.smoke, "test_accessed": False,
        "chance": round(CHANCE, 4), "families": fams_used,
        "excluded": ["OpenAI（单 generator，不进入归因折）"],
        "folds": fold_out, "pooled": pooled, "exit": exit_info,
        "protocol": "E30 折 + 冻结 B0 主轴（加权 LR）+ 阶段 B 修正门控残差；"
                    "F0=CE，F1=CE+0.1·L_cross；τ=.1；批 7×18；2ep；AdamW 1e-3/wd 1e-4",
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=1))
    preds_out["pooled_classes"] = np.array(fams_used)
    np.savez_compressed(OUT / "predictions.npz", **preds_out)
    manifest = {
        "commit": commit, "smoke": a.smoke,
        "fold_protocol": "E30 同源：RandomState(1000+fi)；2→1/1、3→1/2、5→2/3；"
                         "fold0 val=A/train=B，fold1 互补",
        "fold_generators": {f"fold{fold}": fold_out[f"fold{fold}"]["fold_generators"]
                            for fold in (0, 1)},
        "features_sha256": feats_sha,
        "input_manifest_sha256": hashlib.sha256(key_lines.encode()).hexdigest(),
        "counts": {"ai7": int(len(ai7_idx)), "human": int(len(hum_idx))},
        "generator_labels": {g: {"role": GEN_LABELS[g][0], "size": GEN_LABELS[g][1]}
                             for g in GEN_LABELS},
        "test_accessed": False, "unseen_accessed": False,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    (OUT / "config.json").write_text(json.dumps({
        "commit": commit, "C_fixed": C_FIXED, "lr": LR, "wd": WD, "tau": TAU,
        "lambda_C": LAM_C, "epochs": EPOCHS, "batch": "7 族×18", "seed": 0,
        "model": "ℓ=ℓ0+γ·W_R·r；ℓ0=加权 LR（w=N/(K|G_f|N_g)）逐折拟合后冻结；"
                 "r=V·GELU(Uz+b1)+b2；修正门控 W_R=0、η0=0.5、γ=0.1·tanh(η)",
        "arms": {"F0": "L_F（残差 CE）", "F1": "L_F+0.1·L_cross（同家族不同 generator 正对）"},
        "exit_rule": exit_info["rule"], "anchor_min": ANCHOR_MIN,
        "het_note": "若 F1 只改善单角色/单家族，只报异质性（不写成总体通过）",
    }, ensure_ascii=False, indent=1))
    (OUT / "solver.log").write_text("\n".join(solver_log) + "\n")
    say(f"[e35] 完成 {time.time()-t_start:.1f}s；产物 {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
