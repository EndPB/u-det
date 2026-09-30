#!/usr/bin/env python
"""E32：表示瓶颈审计（C0/C1/C2；只解释 E31 负结果，不加新流形损失）。

依据 `docx/d-det_E32表示瓶颈审计与来源几何指导_2026-09-30.md`：
  · 沿用 E31 五模型白名单/原 train/val/seed/batch/采样顺序/2 epoch/检测头/B0；
  · C0 = ã→Wã+b + 家族 CE（神经线性基线；随机初始化为主，B0 权重移植微调为诊断 C0w）；
  · C1 = u→Wu+b + CE（残差 MLP，**不归一化**，保留径向信息）；
    u = ã + V·GELU(U·ã+b1)+b2（U Xavier、V std=1e-3、bias 0；W_F Xavier）；
  · C2 = C1 同初始化、同批序 + 0.1·L_C（SupCon 以 u 的 cosine 计算，τ=.1 只除一次；
    分类输入仍为原始 u，不用归一化结果）；
  · 预注册判读（按序）：①C0≈B0 且 C1 不≈ → 残差 MLP 训练/容量瓶颈；②C1≈B0 且 C2−C1<1pt →
    负结果主要来自 L2 归一化；③C2−C1≥1pt 且检测降 ≤0.5pt → 对比项初候选；④C1,C2 均 ≤C0−2pt →
    优化负结果（先改预算/分类头）。close(x,y):=x≥y−0.02。

产物：artifacts/flagship_e32/{config.json, manifest.json, metrics.json, predictions.npz,
train.log}（report.md 由报告复制）；smoke → artifacts/flagship_e32_smoke；默认拒绝覆盖。
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


def wl_match(g):
    return any(p in g for p in WL)


class MLP2(torch.nn.Module):
    """u=ã+V·GELU(U·ã+b1)+b2；ℓ=W_F u+b_F（**不归一化**）。"""

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
    """L_C（以 u 的 cosine；τ 只除一次；P=同族；B'=P 非空的 anchors）。"""
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
    OUT = ROOT / "artifacts" / ("flagship_e32_smoke" if a.smoke else "flagship_e32")
    if OUT.exists() and any(OUT.iterdir()) and not a.force:
        print(f"[e32] 已存在产物，默认拒绝覆盖：{OUT}（--force 可覆盖）", flush=True)
        return 2
    OUT.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    log_lines: list[str] = []

    def say(msg):
        print(msg, flush=True)
        log_lines.append(msg)

    commit = subprocess.run(["git", "-C", str(ROOT.parent), "rev-parse", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    say(f"[e32] commit {commit} | smoke={a.smoke}")

    # ---- 数据（与 E31 完全一致；只读 b_train/b_val） ----
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
        say(f"[e32] 特征 r0={R28_FEATS.name}（sha256 {feats_sha[:16]}…）| 编码器 "
            f"{enc_sha[:16]}…")

    sel_ai = np.array([i for i in np.where(y_ai)[0] if wl_match(str(gen[i]))])
    tr_ai = np.array([i for i in sel_ai if sp[i] == "train"])
    va_ai = np.array([i for i in sel_ai if sp[i] == "val"])
    hum_tr = np.where((~y_ai) & (sp == "train"))[0]
    hum_va = np.where((~y_ai) & (sp == "val"))[0]
    if not a.smoke:
        assert len(tr_ai) == 4013 and len(va_ai) == 1654, \
            f"子集计数不符：{len(tr_ai)}/{len(va_ai)}"
    say(f"[e32] 子集：AI train {len(tr_ai)} / val {len(va_ai)}；Human "
        f"{len(hum_tr)}/{len(hum_va)}")

    m_r, s_r = fit_scaler(raw[tr_ai])
    A_tr = apply_scaler(raw[tr_ai], m_r, s_r)
    A_va = apply_scaler(raw[va_ai], m_r, s_r)
    A_hum_tr = apply_scaler(raw[hum_tr], m_r, s_r)
    A_hum_va = apply_scaler(raw[hum_va], m_r, s_r)
    y4_tr = np.array([FAM4.index(SEEN_LABS[int(fam[i])]) for i in tr_ai])
    y4_va = np.array([FAM4.index(SEEN_LABS[int(fam[i])]) for i in va_ai])
    gen_va = [str(gen[i]) for i in va_ai]

    # ---- 检测头（冻结，同 E31） ----
    det_lr = LogisticRegression(max_iter=MAX_ITER, tol=TOL, C=C_FIXED).fit(
        np.vstack([A_tr, A_hum_tr]), np.concatenate([np.ones(len(A_tr)),
                                                     np.zeros(len(A_hum_tr))]))
    det_va = det_lr.predict_proba(np.vstack([A_va, A_hum_va]))[:, 1]
    det_y = np.concatenate([np.ones(len(A_va)), np.zeros(len(A_hum_va))])
    det_auroc = float(roc_auc_score(det_y, det_va))
    say(f"[e32] 检测头（冻结）：val AUROC {det_auroc:.4f}")

    # ---- B0（sklearn 参照，不反传；同 E31 权重） ----
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
    say(f"[e32] B0：BA_F {b0_val['BA_F']} BA_G {b0_val['BA_G']}（收敛 {b0_conv}）")

    # ---- 批调度（复制 E31 生成代码：同 seed、同序 → 同批序） ----
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
    say(f"[e32] 批调度（同 E31）：{steps} step/epoch×{EPOCHS}；正对 同gen "
        f"{pair_stats['same_gen']} / 跨gen {pair_stats['cross_gen']}")

    # ---- 模型与训练（C0/C1/C2 同批序；C1/C2 同 init） ----
    torch.manual_seed(0)
    c0 = torch.nn.Linear(768, 4)
    torch.nn.init.xavier_uniform_(c0.weight)
    torch.nn.init.zeros_(c0.bias)
    m1 = MLP2()
    m2 = MLP2()
    m2.load_state_dict(copy.deepcopy(m1.state_dict()))
    c0w = torch.nn.Linear(768, 4)      # 诊断：B0 权重移植
    c0w.weight.data = torch.as_tensor(b0.coef_, dtype=torch.float32)
    c0w.bias.data = torch.as_tensor(b0.intercept_, dtype=torch.float32)

    arms = {
        "C0": (c0, torch.optim.AdamW(c0.parameters(), lr=LR, weight_decay=WD), "linear",
               False),
        "C1": (m1, torch.optim.AdamW(m1.parameters(), lr=LR, weight_decay=WD), "mlp",
               False),
        "C2": (m2, torch.optim.AdamW(m2.parameters(), lr=LR, weight_decay=WD), "mlp",
               True),
        "C0w": (c0w, torch.optim.AdamW(c0w.parameters(), lr=LR, weight_decay=WD),
                "linear", False),
    }
    A_tr_t = torch.as_tensor(A_tr)
    y4_tr_t = torch.as_tensor(y4_tr)
    stats = {k: [] for k in arms}
    for ep in range(EPOCHS):
        for step in range(steps):
            bidx = schedule[ep * steps + step]
            xb = A_tr_t[bidx]; yb = y4_tr_t[bidx]
            for tag, (model, opt, kind, use_c) in arms.items():
                if kind == "linear":
                    logits = model(xb)
                else:
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
        say(f"[e32] epoch {ep} 末首/末批：" + " | ".join(
            f"{t} {[s for s in stats[t] if s['epoch'] == ep]}" for t in arms))

    # ---- 评估 ----
    res = {}
    preds = {"B0": b0.predict(A_va).astype(np.int8)}
    for tag, (model, _, kind, _) in arms.items():
        with torch.no_grad():
            if kind == "linear":
                logits = model(torch.as_tensor(A_va))
            else:
                _, logits = model(torch.as_tensor(A_va))
        pred = logits.argmax(1).numpy()
        res[tag] = fam_eval(pred, y4_va, FAM4, gen_va)
        preds[tag] = pred.astype(np.int8)
    for tag in ("B0", "C0", "C1", "C2", "C0w"):
        r = b0_val if tag == "B0" else res[tag]
        say(f"[e32] val·{tag}：BA_F {r['BA_F']} BA_G {r['BA_G']}")

    c1c2_diff = {p: float((dict(m1.named_parameters())[p] -
                           dict(m2.named_parameters())[p]).abs().max())
                 for p in ("U.weight", "V.weight", "WF.weight")}

    # ---- 预注册判读（§3 顺序） ----
    close = lambda x, y: x >= y - CLOSE
    r1 = close(res["C0"]["BA_F"], b0_val["BA_F"]) and \
        not close(res["C1"]["BA_F"], b0_val["BA_F"])
    r2 = close(res["C1"]["BA_F"], b0_val["BA_F"]) and \
        (res["C2"]["BA_F"] - res["C1"]["BA_F"]) < 0.01
    det_drop = 0.0     # 检测头冻结（同 E31）；如实记录该条件退化
    r3 = (res["C2"]["BA_F"] - res["C1"]["BA_F"]) >= 0.01 and det_drop <= 0.005
    r4 = (res["C1"]["BA_F"] <= res["C0"]["BA_F"] - CLOSE) and \
        (res["C2"]["BA_F"] <= res["C0"]["BA_F"] - CLOSE)
    flags = {"rule1_mlp_bottleneck": bool(r1), "rule2_norm_bottleneck": bool(r2),
             "rule3_contrast_candidate": bool(r3), "rule4_optimization_negative": bool(r4)}
    order = ["rule1_mlp_bottleneck", "rule2_norm_bottleneck", "rule3_contrast_candidate",
             "rule4_optimization_negative"]
    fired = [k for k in order if flags[k]]
    verdict = (f"判定 {fired[0]}" if fired else "四条规则均未命中（记录全部数字，另行分析）")
    say(f"[e32] 判读：Δ(C2−C1)={res['C2']['BA_F']-res['C1']['BA_F']:+.4f}；"
        f"Δ(C1−B0)={res['C1']['BA_F']-b0_val['BA_F']:+.4f}；"
        f"Δ(C0−B0)={res['C0']['BA_F']-b0_val['BA_F']:+.4f}；flags={flags} → {verdict}")

    # ---- 产物 ----
    np.savez_compressed(
        OUT / "predictions.npz",
        va_rows=rows[va_ai], va_split=sp[va_ai], va_family=y4_va.astype(np.int8),
        va_generator=np.array(gen_va), families_=np.array(FAM4),
        b0_pred=preds["B0"], c0_pred=preds["C0"], c1_pred=preds["C1"],
        c2_pred=preds["C2"], c0w_pred=preds["C0w"],
        det_prob_val=det_va.astype(np.float32), det_y=det_y.astype(np.int8),
        va_rows_human=rows[hum_va])
    metrics = {
        "commit": commit, "smoke": a.smoke, "test_accessed": False,
        "B0": b0_val, "C0": res["C0"], "C1": res["C1"], "C2": res["C2"],
        "C0w_diagnostic": res["C0w"],
        "deltas": {"C2_minus_C1": round(res["C2"]["BA_F"] - res["C1"]["BA_F"], 4),
                   "C1_minus_B0": round(res["C1"]["BA_F"] - b0_val["BA_F"], 4),
                   "C0_minus_B0": round(res["C0"]["BA_F"] - b0_val["BA_F"], 4)},
        "flags": flags, "verdict": verdict,
        "det_auroc_frozen": round(det_auroc, 4), "det_drop": det_drop,
        "batch_stats": {"first_last": stats, "pairs": pair_stats},
        "c1_c2_param_diff": c1c2_diff,
        "note": "C0w=B0 权重移植后同预算微调（诊断，不参与主比较）",
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
                        "note": "跨 gen 正对仅来自 Mistral"},
        "schedule": "与 E31 相同（同 seed/同采样序）",
        "test_accessed": False, "unseen_accessed": False, "smoke": a.smoke,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    (OUT / "config.json").write_text(json.dumps({
        "commit": commit, "C_fixed": C_FIXED, "batch": BATCH, "lr": LR, "wd": WD,
        "tau": TAU, "lambda_C": LAM_C, "epochs": EPOCHS, "seed": STEPS_SEED,
        "arms": {"C0": "ã→Wã+b CE（Xavier 随机 init）",
                 "C1": "u→Wu+b CE（不归一化）",
                 "C2": "C1+0.1·SupCon（cosine 仅用于损失）",
                 "C0w": "B0 权重移植微调（诊断）"},
        "verdict_rule": "§3 顺序：①C0≈B0∧C1不≈；②C1≈B0∧Δ(C2−C1)<.01；"
                        "③Δ(C2−C1)≥.01∧检测降≤.005；④C1,C2≤C0−.02；close:=≥y−.02",
    }, ensure_ascii=False, indent=1))
    (OUT / "train.log").write_text("\n".join(log_lines) + "\n")
    say(f"[e32] 完成 {time.time()-t_start:.1f}s；产物 {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
