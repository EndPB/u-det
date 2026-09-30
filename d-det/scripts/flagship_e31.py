#!/usr/bin/env python
"""E31：H2 最小对比实验（强模型子集 B0/B1/B2；同容量、同初始化、同批序，只加来源对比）。

依据 `docx/d-det_E31核心假设修订与最小对比实验_2026-09-30.md` §4–§5：
  · 数据 = E26 固化强模型子集（5 generators / 4 家族：GPT-4o、DeepSeek-V3-0324、
    Mistral-7B-Instruct-v0.3、Devstral-Small-2505、gemma-3-27b-it；AI 4013/1654）+ Human 供检测；
    只复用 E28 raw 缓存、保留原 split；
  · B0 = 标准化 raw 逻辑回归（C=.1, tol 1e-6, max_iter 10000；sample_weight=N/(K|G_f|N_g)）；
    B1 = 残差 MLP + 家族 CE；B2 = 同容量同初始化同批序 + 0.1·来源对比 SupCon（τ=.1）；
    u=ã+V·GELU(U·ã+b1)+b2，h=u/max(‖u‖,1e-6)；U Xavier、V std=1e-3、bias=0；
  · 批 128=4 族×32（族→generator→样本 均匀；每族 ≥2 distinct）；steps=ceil(N/128)；2 epoch
    同推进、同 seed、同批序；AdamW lr=1e-3 wd=1e-4 clip=1；取 epoch 2（不按 val 挑）；
  · 检测头 = ã 上二项逻辑回归（AI 子集+Human 训练集；C=.1）冻结，B1/B2 不改；
  · 指标：val AI 子集 BA_F/BA_G/逐族逐 gen 召回/语言长度桶 + det AUROC；首/末批 CE、L_C、
    梯度范数；正对（同/跨 generator）比例；预注册判读：B2>B1 且 B2>B0 且 Δ≥1pt → H2 候选。

产物：artifacts/flagship_e31/{config.json, manifest.json, metrics.json, predictions.npz,
train.log}（report.md 由报告复制）；smoke → artifacts/flagship_e31_smoke；默认拒绝覆盖。
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
      "Devstral-Small-2505", "gemma-3-27b-it"]          # 五模型白名单（子串匹配）
FAM4 = ["OpenAI", "DeepSeek", "Mistral", "Google"]      # 四家族（SEEN_LABS 顺序子集）
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
LEN_EDGES = [(8, 128), (129, 256), (257, 512), (513, 1024), (1025, 2048)]


def wl_match(g):
    return any(p in g for p in WL)


class GeoMLP(torch.nn.Module):
    """u=â+V·GELU(U·â+b1)+b2；h=u/max(‖u‖,1e-6)；ℓ=W_F h+b_F。"""

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
        h = u / u.norm(dim=1, keepdim=True).clamp(min=1e-6)
        return h, self.WF(h)


def supcon(h, fam):
    """L_C（式 6）：A(i)=B\\{i}；P(i)=同族；温度只除一次。"""
    sim = h @ h.t() / TAU
    b = h.shape[0]
    self_mask = torch.eye(b, dtype=torch.bool, device=h.device)
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
    OUT = ROOT / "artifacts" / ("flagship_e31_smoke" if a.smoke else "flagship_e31")
    if OUT.exists() and any(OUT.iterdir()) and not a.force:
        print(f"[e31] 已存在产物，默认拒绝覆盖：{OUT}（--force 可覆盖）", flush=True)
        return 2
    OUT.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    log_lines: list[str] = []

    def say(msg):
        print(msg, flush=True)
        log_lines.append(msg)

    commit = subprocess.run(["git", "-C", str(ROOT.parent), "rev-parse", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    say(f"[e31] commit {commit} | smoke={a.smoke}")

    # ---- 数据（只读 b_train/b_val；E28 缓存） ----
    corpus = load_train_val(args)
    docs_all = corpus["train"] + corpus["val"]
    y_ai = np.array([d.y_ai for d in docs_all], bool)
    fam = np.array([SEEN_LABS.index(d.fam) if d.fam in SEEN_LABS else -1
                    for d in docs_all], np.int16)
    gen = np.array([d.gen for d in docs_all])
    lang = np.array([d.lang for d in docs_all])
    ntok = np.array([len(d.ids) for d in docs_all], np.int32)
    rows = np.array([d.row for d in docs_all], np.int64)
    sp = np.array([d.split for d in docs_all])

    feats_sha = hashlib.sha256(R28_FEATS.read_bytes()).hexdigest() if not a.smoke else None
    enc_sha = hashlib.sha256(ENC_PATH.read_bytes()).hexdigest()
    if a.smoke:
        # 冒烟：小样本自建 raw 前向
        import torch as _t
        import yaml
        from encoders import build_encoder
        from models import build_model
        from flagship_round1 import SetPool
        ms = _t.load(ROOT / "runs/flagship_e27/model_state.pt", map_location="cpu",
                     weights_only=False)
        with open(ROOT / "configs/ddet_v041.yaml", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        ec = dict(cfg["encoder"]); ec["path"] = str(ROOT / ec["path"])
        dual = build_model("dual", encoder=build_encoder(**ec), dim=768,
                           pool=cfg["model"].get("pool", "mean"),
                           s2_rank=cfg["model"].get("s2_rank", 1))
        dual.load_state_dict(_t.load(ENC_PATH, map_location="cpu",
                                     weights_only=False)["state"], strict=False)
        dev = "cuda" if _t.cuda.is_available() else "cpu"
        enc = dual.encoder.to(dev).eval()
        head = SetPool(768, 512).to(dev)
        head.load_state_dict(_t.load(ROOT / "runs/flagship_r1/head.pt",
                                     map_location=dev)["head"])
        head.eval()
        from flagship_e28_probe import export_pass
        wq = _t.as_tensor(ms["wq"], device=dev)
        bq = _t.as_tensor(ms["bq"], device=dev)
        m_tT = _t.as_tensor(ms["m_t"], dtype=_t.float32, device=dev)
        s_tT = _t.as_tensor(ms["s_t"], dtype=_t.float32, device=dev)
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
        say(f"[e31] 特征 r0=runs/flagship_e28/features_train_val.npz（sha256 "
            f"{feats_sha[:16]}…）| 编码器 sha256 {enc_sha[:16]}…")

    # ---- 子集 ----
    sel_ai = np.array([i for i in np.where(y_ai)[0] if wl_match(str(gen[i]))])
    tr_ai = np.array([i for i in sel_ai if sp[i] == "train"])
    va_ai = np.array([i for i in sel_ai if sp[i] == "val"])
    hum_tr = np.where((~y_ai) & (sp == "train"))[0]
    hum_va = np.where((~y_ai) & (sp == "val"))[0]
    if not a.smoke:
        assert len(tr_ai) == 4013 and len(va_ai) == 1654, \
            f"子集计数不符：{len(tr_ai)}/{len(va_ai)}"
        say(f"[e31] 子集：AI train {len(tr_ai)} / val {len(va_ai)}（五模型白名单一致）")
    else:
        say(f"[e31] smoke 子集：AI train {len(tr_ai)} / val {len(va_ai)}")
    gen_counts = defaultdict(int)
    for i in tr_ai:
        gen_counts[str(gen[i])] += 1
    say(f"[e31] 逐 generator 训练数：{json.dumps(dict(gen_counts), ensure_ascii=False)}")

    # ---- 去重审计（ids 精确；沿用"记录不删除"规则） ----
    hs_tr = defaultdict(list)
    for i in tr_ai:
        hs_tr[ids_md5(docs_all[i].ids)].append(int(i))
    dup_tr = {h: v for h, v in hs_tr.items() if len(v) > 1}
    hs_va = {}
    for i in va_ai:
        hs_va[ids_md5(docs_all[i].ids)] = int(i)
    cross = sum(1 for h in hs_tr if h in hs_va)
    say(f"[e31] 去重（子集 ids 精确）：train 内重复组 {len(dup_tr)}、train↔val {cross}"
        f"（剔除清单为空——不删除样本；近重复沿用 E26 记录：Mistral-7B 同 gen 1 对）")

    # ---- 标准化（AI 子集 train）+ 检测头 ----
    m_r, s_r = fit_scaler(raw[tr_ai])
    A_tr = apply_scaler(raw[tr_ai], m_r, s_r)
    A_va = apply_scaler(raw[va_ai], m_r, s_r)
    A_hum_tr = apply_scaler(raw[hum_tr], m_r, s_r)
    A_hum_va = apply_scaler(raw[hum_va], m_r, s_r)
    y4_tr = np.array([FAM4.index(SEEN_LABS[int(fam[i])]) for i in tr_ai])
    y4_va = np.array([FAM4.index(SEEN_LABS[int(fam[i])]) for i in va_ai])
    gen_va = [str(gen[i]) for i in va_ai]

    det_lr = LogisticRegression(max_iter=MAX_ITER, tol=TOL, C=C_FIXED).fit(
        np.vstack([A_tr, A_hum_tr]), np.concatenate([np.ones(len(A_tr)),
                                                     np.zeros(len(A_hum_tr))]))
    det_va = det_lr.predict_proba(np.vstack([A_va, A_hum_va]))[:, 1]
    det_y = np.concatenate([np.ones(len(A_va)), np.zeros(len(A_hum_va))])
    det_auroc = float(roc_auc_score(det_y, det_va))
    say(f"[e31] 检测头（冻结）：val AUROC {det_auroc:.4f}（n={len(det_y)}）")

    # ---- B0 ----
    N = len(tr_ai); K = len(FAM4)
    # |G_f| 与 N_g 权重：w_i = N/(K|G_f|N_g)
    fam_gens = {f: sorted({str(gen[i]) for i in tr_ai
                           if SEEN_LABS[int(fam[i])] == f}) for f in FAM4}
    w_sample = np.array([N / (K * len(fam_gens[SEEN_LABS[int(fam[i])]])
                              * gen_counts[str(gen[i])]) for i in tr_ai])
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter("always")
        b0 = LogisticRegression(max_iter=MAX_ITER, tol=TOL, C=C_FIXED).fit(
            A_tr, y4_tr, sample_weight=w_sample)
    b0_conv = not any(issubclass(x.category, ConvergenceWarning) for x in wl)
    b0_pred = b0.predict(A_va)
    b0_val = fam_eval(b0_pred, y4_va, FAM4, gen_va)
    b0_val["converged"] = b0_conv
    b0_val["n_iter"] = int(np.max(b0.n_iter_))
    say(f"[e31] B0：BA_F {b0_val['BA_F']} BA_G {b0_val['BA_G']}（收敛 {b0_conv}，"
        f"n_iter {b0_val['n_iter']}）")

    # ---- 批调度（一次生成，两臂共用） ----
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
        # 正对统计
        fams_b = y4_tr[bidx]; gens_b = [str(gen[tr_ai[p]]) for p in bidx]
        for i in range(len(bidx)):
            for j in range(i + 1, len(bidx)):
                if fams_b[i] == fams_b[j]:
                    if gens_b[i] == gens_b[j]:
                        pair_stats["same_gen"] += 1
                    else:
                        pair_stats["cross_gen"] += 1
        pair_stats["batches"] += 1
        schedule.append(bidx)
    say(f"[e31] 批调度：{steps} step/epoch×{EPOCHS}；正对 同gen {pair_stats['same_gen']}"
        f" / 跨gen {pair_stats['cross_gen']}（重抽 {pair_stats['redraw']}）")

    # ---- B1/B2 训练（同 init、同批序；2 epoch 同推进；取 epoch2） ----
    torch.manual_seed(0)
    m1 = GeoMLP()
    m2 = GeoMLP()
    m2.load_state_dict(copy.deepcopy(m1.state_dict()))
    opt1 = torch.optim.AdamW(m1.parameters(), lr=LR, weight_decay=WD)
    opt2 = torch.optim.AdamW(m2.parameters(), lr=LR, weight_decay=WD)
    A_tr_t = torch.as_tensor(A_tr)
    y4_tr_t = torch.as_tensor(y4_tr)
    stats = {"B1": [], "B2": []}
    for ep in range(EPOCHS):
        for step in range(steps):
            bidx = schedule[ep * steps + step]
            xb = A_tr_t[bidx]; yb = y4_tr_t[bidx]
            for tag, model, opt, use_c in (("B1", m1, opt1, False),
                                           ("B2", m2, opt2, True)):
                h, logits = model(xb)
                L_F = F.cross_entropy(logits, yb)
                if use_c:
                    L_C, n_empty = supcon(h, yb)
                    pair_stats["empty_P"] += n_empty
                    loss = L_F + LAM_C * L_C
                else:
                    with torch.no_grad():
                        L_C, _ = supcon(h, yb)
                    loss = L_F
                opt.zero_grad(); loss.backward()
                gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
                opt.step()
                if step in (0, steps - 1):
                    stats[tag].append({"epoch": ep, "step": step,
                                       "ce": round(float(L_F.detach()), 4),
                                       "lc": round(float(L_C.detach()), 4),
                                       "grad_norm": round(gnorm, 3)})
        say(f"[e31] epoch {ep} 末：B1 首/末批 " +
            str([s for s in stats["B1"] if s["epoch"] == ep]) +
            " | B2 " + str([s for s in stats["B2"] if s["epoch"] == ep]))

    mlp_diff = {p: float((dict(m1.named_parameters())[p] -
                          dict(m2.named_parameters())[p]).abs().max())
                for p in ("U.weight", "V.weight", "WF.weight")}
    say(f"[e31] B1↔B2 参数差（max|Δ|）：{mlp_diff}")

    # ---- 评估（epoch2 = 最终模型） ----
    res = {}
    preds = {"b0": b0_pred}
    for tag, model in (("B1", m1), ("B2", m2)):
        with torch.no_grad():
            h, logits = model(torch.as_tensor(A_va))
            pred = logits.argmax(1).numpy()
        res[tag] = fam_eval(pred, y4_va, FAM4, gen_va)
        preds[tag] = pred.astype(np.int8)
    say(f"[e31] val：B0 BA_F {b0_val['BA_F']}/BA_G {b0_val['BA_G']} | "
        f"B1 {res['B1']['BA_F']}/{res['B1']['BA_G']} | "
        f"B2 {res['B2']['BA_F']}/{res['B2']['BA_G']}")

    # 语言/长度桶（BA_F 粗分）
    buckets = {}
    lg = np.array([str(lang[i]) for i in va_ai])
    lt = np.array([int(ntok[i]) for i in va_ai])
    for label, mask in ([(f"lang:{l}", lg == l)
                         for l in sorted(set(lg.tolist())) if (lg == l).sum() >= 30]
                        + [(f"len:{lo}-{hi}", (lt >= lo) & (lt <= hi))
                           for lo, hi in LEN_EDGES]):
        if mask.sum() < 20 or len(set(y4_va[mask].tolist())) < 2:
            continue
        row = {"n": int(mask.sum())}
        for tag, p in (("B0", preds["b0"]), ("B1", preds["B1"]), ("B2", preds["B2"])):
            row[tag] = round(float(balanced_accuracy_score(y4_va[mask], p[mask])), 4)
        buckets[label] = row

    # ---- 预注册判读 ----
    d21 = res["B2"]["BA_F"] - res["B1"]["BA_F"]
    d20 = res["B2"]["BA_F"] - b0_val["BA_F"]
    d10 = res["B1"]["BA_F"] - b0_val["BA_F"]
    if d21 >= 0.01 and d20 > 0:
        verdict = "H2 初步候选（B2>B1≥1pt 且 >B0）"
    elif d10 > 0 and d20 > 0 and d21 < 0.01:
        verdict = "收益来自非线性容量（B1/B2>B0 但 B2 无增量）"
    else:
        verdict = "负结果（B2 无增益；不增加多个损失解释失败）"
    say(f"[e31] 判读：Δ(B2−B1)={d21:+.4f} Δ(B2−B0)={d20:+.4f} Δ(B1−B0)={d10:+.4f}"
        f" → {verdict}")

    # ---- 产物 ----
    np.savez_compressed(
        OUT / "predictions.npz",
        va_rows=rows[va_ai], va_split=sp[va_ai], va_family=y4_va.astype(np.int8),
        va_generator=np.array(gen_va), va_language=lg, va_length=lt,
        families_=np.array(FAM4), b0_pred=preds["b0"].astype(np.int8),
        b1_pred=preds["B1"], b2_pred=preds["B2"],
        det_prob_val=det_va.astype(np.float32), det_y=det_y.astype(np.int8),
        va_rows_human=rows[hum_va])
    metrics = {
        "commit": commit, "smoke": a.smoke, "test_accessed": False,
        "chance": 0.25, "C_fixed": C_FIXED, "N_ai_train": N, "steps_per_epoch": steps,
        "B0": b0_val, "B1": res["B1"], "B2": res["B2"],
        "B2_minus_B1": round(d21, 4), "B2_minus_B0": round(d20, 4),
        "B1_minus_B0": round(d10, 4), "verdict": verdict,
        "det_auroc_frozen": round(det_auroc, 4),
        "batch_stats": {"first_last": stats, "pairs": pair_stats},
        "mlp_B1_B2_diff": mlp_diff, "buckets": buckets,
        "dedup": {"dup_train_groups": len(dup_tr), "train_val_cross": cross,
                  "exclusion_list": [], "note": "不删除样本；近重复沿用 E26（子集相关："
                                                "Mistral-7B 同 gen 1 对）"},
        "encoder_overlap_check": {
            "config_level": "v0.4.1 训练数据仅 m4/hybrid/pairs/pairs_qwen15（无 SemEval）",
            "sample_level": "Task B train/val 抽样 4000 条解码 vs 内部 31978 码："
                            "精确归一化 0、标识符集合 0 命中",
            "weights_used": "v0.4.1（SemEval 适配微调检查点未被使用）"},
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=1))
    manifest = {
        "commit": commit, "whitelist": WL,
        "families": FAM4, "family_generators": fam_gens,
        "counts": {"ai_train": int(N), "ai_val": int(len(va_ai)),
                   "human_train": int(len(hum_tr)), "human_val": int(len(hum_va)),
                   "generator_train_counts": dict(gen_counts)},
        "features_sha256": feats_sha, "encoder_sha256": enc_sha,
        "e24_e27_heads": "未加载（每臂重新初始化）",
        "dedup_exclusion_list": [],
        "pair_ratios": {"same_gen": pair_stats["same_gen"],
                        "cross_gen": pair_stats["cross_gen"],
                        "note": "跨 gen 正对仅来自 Mistral（两 generator）"},
        "test_accessed": False, "unseen_accessed": False,
        "smoke": a.smoke,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    (OUT / "config.json").write_text(json.dumps({
        "commit": commit, "C_fixed": C_FIXED, "batch": BATCH, "lr": LR, "wd": WD,
        "tau": TAU, "lambda_C": LAM_C, "epochs": EPOCHS, "seed": STEPS_SEED,
        "clip": 1, "std_floor": STD_FLOOR,
        "sampling": "族均匀→generator 均匀→样本；每族≥2 distinct",
        "mlp": "u=ã+V·GELU(Uã+b1)+b2; h=u/max(||u||,1e-6); U Xavier, V std=1e-3, "
               "bias 0; W_F Xavier",
        "verdict_rule": "B2>B1(≥1pt) 且 B2>B0 → H2 候选；B1/B2>B0 无增量 → 容量；"
                        "否则负结果（预声明）",
    }, ensure_ascii=False, indent=1))
    (OUT / "train.log").write_text("\n".join(log_lines) + "\n")
    say(f"[e31] 完成 {time.time()-t_start:.1f}s；产物 {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
