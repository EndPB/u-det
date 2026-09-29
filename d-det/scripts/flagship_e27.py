#!/usr/bin/env python
"""E27：修复显著位置通道（768 维 s_top、非零初始化）+ 公平基线（单实验）。

依据 `docx/d-det_E27_云端继续指导文档_2026-09-29.md`（基线提交 9d69b4d）：
  · 唯一新增训练臂 μ+top：z=[μ;s_top]∈R^1536，s_top 为 top-k 显著位置 token 均值（768 维，
    替换旧标量实现）；w_q ~ N(0, 1/768)、b_q=0；通道与头均非零初始化；
  · 对照臂 μ-only（R^768）；相同编码器/批次/epoch/学习率/初始化/标准化；
  · L = BCE(s_D,y) + 1.0·CE(a_F, y_F|AI)（λ_F=1.0=E24 同值）；无 SupCon/能量/L_task；
  · epoch 选择：val 家族 bal → det AUROC → 较早（保存每 epoch 快照，按选中 epoch 回滚）；
  · 离线臂（B0 rm / B2 z[:1536] / B3 rF / B4 能量标量）用 C* 网格 {0.01..10}（val 家族
    bal，tie→det AUROC→小 C）；
  · test_seen/unseen 在 epoch 与 C* 冻结后**只读一次**；两项置换检查并记录。

实现注记（写入 e27_config.json / 报告）：
  1) token 基 = E24 SetPool 的 ht（LN→W1→W2 残差，W1/W2 冻结在 E24 训练值）——依据
     "冻结编码器以及 E24 的 W1、W2" 与 §1.1 mu-only=.4213（E26 网格 ht-μ 口径）；
  2) s_top = Σ_{i∈T_k} softmax(q|T_k)_i · ht_i：文档字面 1/k 均匀权重对 w_q 不可导
     （∂s_top/∂w_q≡0，会直接触发 §6 的实现失败判据）；按 §4.1 已定义的 α 使用 top-k
     支撑集内 softmax 权重（q 在标准化 token 上分辨力为 O(1)，非近均匀）。

输出：runs/flagship_e27/{e27_config.json, train.log, metrics.json, s_top_stats.json,
permutation_checks.json, model_state.pt, commit.txt, README.md}
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from encoders import build_encoder  # noqa: E402
from models import build_model  # noqa: E402
from flagship_round1 import (Doc, SetPool, SEEN_LABS, _batches,  # noqa: E402
                             collate, residualize)
from flagship_e25_stage0 import load_corpus_tagged, ids_md5  # noqa: E402

OUT = ROOT / "runs/flagship_e27"
R1 = ROOT / "runs/flagship_r1"
R25 = ROOT / "runs/flagship_e25"
R26 = ROOT / "runs/flagship_e26"

STD_FLOOR = 1e-2          # 逐维标准化标准差下限（规范 §3）
TE = 1537                 # 能量温度（规范 §2）
LAM_F = 1.0               # 家族 CE 权重（E24 同值）
C_GRID = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]


def default_args(smoke: bool = False) -> Namespace:
    if smoke:
        return Namespace(smoke=True, max_len=2048, train_human=600, train_fam=120,
                         val_human=300, val_fam=80, test_human=200, test_fam=60,
                         unseen_cap=120, enc_bs=16, batch=32, epochs=2, lr=1e-3,
                         weight_decay=1e-4, d_m=512)
    return Namespace(smoke=False, max_len=2048, train_human=8000, train_fam=1200,
                     val_human=1500, val_fam=500, test_human=500, test_fam=200,
                     unseen_cap=600, enc_bs=32, batch=64, epochs=2, lr=1e-3,
                     weight_decay=1e-4, d_m=512)


# --------------------------------------------------------------------------- #
# 前向基础件
# --------------------------------------------------------------------------- #
@torch.no_grad()
def to_tokens(enc, head, batch, device):
    """编码器 + E24 冻结 W1/W2 → ht (B,L,768) 与 mask。"""
    ids, mask = collate(batch, device)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
        h = enc(ids, mask)
    hf = h.float()
    u = head.ln(hf)
    ht = (head.W2(F.gelu(head.W1(u))) + u) * mask.unsqueeze(-1)
    return ht, mask


def masked_mean(ht, mask):
    return ht.sum(1) / mask.sum(1, keepdim=True).clamp(min=1)


def stop_forward(ht, mask, wq, bq, m_t, s_t):
    """E27 显著位置通道：top-k 支撑内 softmax 加权的 token 均值（768 维）。

    q_i = w_q·((ht_i-m_t)/s_t) + b_q；k = clamp(ceil(0.1·n_valid), min 8, max n_valid)。
    """
    hn = (ht - m_t) / s_t
    q = (hn @ wq + bq).squeeze(-1)
    q = q.masked_fill(mask == 0, -1e9)
    n = mask.sum(1)
    k = torch.clamp(torch.ceil(0.1 * n).long(), min=8)
    k = torch.minimum(k, n.long()).clamp(min=1)
    kmax = int(k.max())
    topq, topi = torch.topk(q, kmax, dim=1)                       # (B,kmax)
    keep = (torch.arange(kmax, device=q.device)[None, :] < k[:, None])
    a = torch.softmax(topq.masked_fill(~keep, -1e9), dim=1) * keep
    a = a / a.sum(1, keepdim=True).clamp(min=1e-9)
    ht_sel = torch.gather(ht, 1, topi[:, :, None].expand(-1, -1, ht.shape[2]))
    return (a[:, :, None] * ht_sel).sum(1)                        # (B,768)


def z_from_mu_stop(mu, stp, m_mu, s_mu, m_st, s_st):
    return torch.cat([(mu - m_mu) / s_mu, (stp - m_st) / s_st], dim=1)


# --------------------------------------------------------------------------- #
# 评估
# --------------------------------------------------------------------------- #
def det_metrics(sD, docs, thr=None):
    y = np.array([d.y_ai for d in docs])
    sig = 1 / (1 + np.exp(-np.clip(sD, -50, 50)))
    out = {"auroc": round(float(roc_auc_score(y, sD)), 4),
           "f1@.5": round(float(f1_score(y, (sig > 0.5).astype(int),
                                         average="macro")), 4), "n": int(len(y))}
    if thr is not None:
        out["thr"] = round(float(thr), 4)
        out["f1@thr"] = round(float(f1_score(y, (sig > thr).astype(int),
                                             average="macro")), 4)
    return out, sig


def best_f1_thr(sig, docs):
    y = np.array([d.y_ai for d in docs])
    best = (0.0, 0.5)
    for t in np.unique(np.round(sig, 3)):
        f1 = f1_score(y, (sig > t).astype(int), average="macro")
        if f1 > best[0]:
            best = (float(f1), float(t))
    return best[1], best[0]


def fam_metrics(p, docs):
    idx = [i for i, d in enumerate(docs) if d.y_ai == 1 and d.fam in SEEN_LABS]
    if not idx:
        return {"acc": None, "bal": None, "n": 0, "recall": {}}
    yt = np.array([SEEN_LABS.index(docs[i].fam) for i in idx])
    pred = p[idx].argmax(1)
    rec = {}
    for c, lab in enumerate(SEEN_LABS):
        m = yt == c
        if int(m.sum()) > 0:
            rec[lab] = round(float((pred[m] == c).mean()), 4)
    return {"acc": round(float((pred == yt).mean()), 4),
            "bal": round(float(balanced_accuracy_score(yt, pred)), 4),
            "n": int(len(yt)), "recall": rec}


def rej_metrics(seen_p, unseen_p):
    lab = np.array([0] * len(seen_p) + [1] * len(unseen_p))
    scr = np.concatenate([seen_p, unseen_p])
    return {"auc_negmaxp": round(float(roc_auc_score(lab, -scr)), 4),
            "auc_maxp": round(float(roc_auc_score(lab, scr)), 4),
            "n_seen": int(len(seen_p)), "n_unseen": int(len(unseen_p))}


def unseen_det(sD_unseen, sD_va_all, val_docs):
    hum = [i for i, d in enumerate(val_docs) if d.y_ai == 0]
    s = np.concatenate([sD_unseen, sD_va_all[hum]])
    y = np.array([1] * len(sD_unseen) + [0] * len(hum))
    return {"auroc": round(float(roc_auc_score(y, s)), 4),
            "n_unseen": int(len(sD_unseen)), "n_human": int(len(hum))}


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = default_args(smoke=ap.parse_args().smoke)

    OUT.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log: list[str] = []

    def say(msg: str):
        print(msg, flush=True)
        log.append(msg)

    commit = subprocess.run(["git", "-C", str(ROOT.parent), "rev-parse", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    say(f"[e27] commit {commit} | smoke={args.smoke} | device={device}")

    # ---- 编码器 + E24 头（ht 与 stats） ----
    with open(ROOT / "configs/ddet_v041.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    enc_cfg = dict(cfg["encoder"])
    enc_cfg["path"] = str(ROOT / enc_cfg["path"])
    enc0 = build_encoder(**enc_cfg)
    dual = build_model("dual", encoder=enc0, dim=enc0.hidden_size,
                       pool=cfg["model"].get("pool", "mean"),
                       s2_rank=cfg["model"].get("s2_rank", 1))
    ck = torch.load(str(ROOT / "runs/v0.4.1_covreg/last.pt"), map_location="cpu",
                    weights_only=False)
    dual.load_state_dict(ck["state"], strict=False)
    enc = dual.encoder.to(device).eval()
    for p in enc.parameters():
        p.requires_grad = False
    head = SetPool(768, args.d_m).to(device)
    head.load_state_dict(torch.load(R1 / "head.pt", map_location=device)["head"])
    head.eval()
    for p in head.parameters():
        p.requires_grad = False          # E24 W1/W2/LN 冻结（wq 不用，重建）
    st = dict(np.load(R1 / "stats.npz", allow_pickle=True))
    bD_e24 = float(torch.load(R1 / "head.pt", map_location="cpu")["bD"])

    # ---- 语料 ----
    corpus = load_corpus_tagged(args)
    say("[e27] 语料：" + " | ".join(f"{k}:{len(v)}" for k, v in corpus.items()))
    counts = {}
    for k, v in corpus.items():
        fam_cnt = {}
        for d in v:
            fam_cnt[d.fam] = fam_cnt.get(d.fam, 0) + 1
        counts[k] = {"n": len(v), "family": fam_cnt,
                     "n_generators": len({d.gen for d in v})}

    # ---- 初始化（非零） ----
    torch.manual_seed(0)
    wq = nn.Parameter(torch.randn(768, 1, device=device) * math.sqrt(1 / 768))
    bq = nn.Parameter(torch.zeros(1, device=device))
    init_wq_norm = float(wq.detach().norm())

    def make_head(n_in):
        wD = nn.Parameter(torch.randn(1, n_in, device=device) * 0.01)
        bD = nn.Parameter(torch.zeros(1, device=device))
        wF = nn.Parameter(torch.randn(len(SEEN_LABS), n_in, device=device) * 0.01)
        bF = nn.Parameter(torch.zeros(len(SEEN_LABS), device=device))
        return {"wD": wD, "bD": bD, "wF": wF, "bF": bF}

    armM = make_head(768)
    armT = make_head(1536)
    say(f"[e27] init：‖w_q‖={init_wq_norm:.4f}（>0 ✓）b_q=0；"
        f"头 ‖w_D‖ M={float(armM['wD'].detach().norm()):.4f} / "
        f"T={float(armT['wD'].detach().norm()):.4f}")
    assert init_wq_norm > 0 and float(armM["wD"].detach().norm()) > 0

    # ---- 统计量 pass A：token 统计 + μ 统计 ----
    t0 = time.time()
    tok_sum = np.zeros(768, np.float64); tok_sq = np.zeros(768, np.float64)
    n_tok = 0
    mu_sum = np.zeros(768, np.float64); mu_sq = np.zeros(768, np.float64)
    n_s = 0
    for batch in _batches(corpus["train"], args.batch, False):
        ht, mask = to_tokens(enc, head, batch, device)
        valid = mask.bool()
        flat = ht[valid]
        tok_sum += flat.sum(0).double().cpu().numpy()
        tok_sq += (flat.double() ** 2).sum(0).cpu().numpy()
        n_tok += int(flat.shape[0])
        mu = masked_mean(ht, mask)
        mu_sum += mu.double().sum(0).cpu().numpy()
        mu_sq += (mu.double() ** 2).sum(0).cpu().numpy()
        n_s += len(batch)
    m_t = tok_sum / n_tok
    s_t = np.maximum(np.sqrt(np.clip(tok_sq / n_tok - m_t ** 2, 0, None)), STD_FLOOR)
    m_mu = mu_sum / n_s
    s_mu = np.maximum(np.sqrt(np.clip(mu_sq / n_s - m_mu ** 2, 0, None)), STD_FLOOR)
    say(f"[e27] pass A（token+μ 统计）{time.time()-t0:.1f}s | tokens {n_tok}")

    m_tT = torch.as_tensor(m_t, dtype=torch.float32, device=device)
    s_tT = torch.as_tensor(s_t, dtype=torch.float32, device=device)
    m_muT = torch.as_tensor(m_mu, dtype=torch.float32, device=device)
    s_muT = torch.as_tensor(s_mu, dtype=torch.float32, device=device)

    # ---- 统计量 pass B：s_top（初始 w_q）统计 ----
    t0 = time.time()
    st_sum = np.zeros(768, np.float64); st_sq = np.zeros(768, np.float64)
    cos_init = 0.0
    for batch in _batches(corpus["train"], args.batch, False):
        ht, mask = to_tokens(enc, head, batch, device)
        with torch.no_grad():
            stp = stop_forward(ht, mask, wq, bq, m_tT, s_tT)
            mu = masked_mean(ht, mask)
        st_sum += stp.double().sum(0).cpu().numpy()
        st_sq += (stp.double() ** 2).sum(0).cpu().numpy()
        cos_init += float(F.cosine_similarity(stp, mu, dim=1).sum().cpu())
    m_st = st_sum / n_s
    s_st = np.maximum(np.sqrt(np.clip(st_sq / n_s - m_st ** 2, 0, None)), STD_FLOOR)
    cos_init /= n_s
    init_dim_var = np.clip(st_sq / n_s - m_st ** 2, 0, None)
    var_min_init = float(init_dim_var.min())
    say(f"[e27] pass B（s_top@init 统计）{time.time()-t0:.1f}s | 逐维方差最小 "
        f"{var_min_init:.3e} | cos(s_top,μ)={cos_init:.4f}")
    m_stT = torch.as_tensor(m_st, dtype=torch.float32, device=device)
    s_stT = torch.as_tensor(s_st, dtype=torch.float32, device=device)

    std_md5 = hashlib.md5(np.concatenate(
        [m_t, s_t, m_mu, s_mu, m_st, s_st]).tobytes()).hexdigest()[:12]

    # ---- 双臂训练 ----
    optM = torch.optim.AdamW(list(armM.values()), lr=args.lr,
                             weight_decay=args.weight_decay)
    optT = torch.optim.AdamW([wq, bq] + list(armT.values()), lr=args.lr,
                             weight_decay=args.weight_decay)

    def arm_forward(zi, arm, y, ai_idx, fam_t):
        sD = F.linear(zi, arm["wD"], arm["bD"]).squeeze(-1)
        aF = F.linear(zi, arm["wF"], arm["bF"])
        L_D = F.binary_cross_entropy_with_logits(sD, y)
        if ai_idx:
            L_F = F.cross_entropy(aF[ai_idx], fam_t)
            with torch.no_grad():
                ent = float((-(F.softmax(aF[ai_idx], 1)
                              * F.log_softmax(aF[ai_idx], 1)).sum(1)).mean())
        else:
            L_F = torch.zeros((), device=zi.device)
            ent = None
        return L_D + LAM_F * L_F, L_D, L_F, ent

    def eval_arms(docs):
        n = len(docs)
        sD = {k: np.empty(n, np.float32) for k in ("M", "T")}
        fam = {k: np.empty((n, len(SEEN_LABS)), np.float32) for k in ("M", "T")}
        mu_arr = np.empty((n, 768), np.float32)
        st_arr = np.empty((n, 768), np.float32)
        order = np.argsort([len(d.ids) for d in docs], kind="stable")
        chunks = [order[i:i + args.batch] for i in range(0, n, args.batch)]
        for ch in chunks:
            batch = [docs[i] for i in ch]
            ht, mask = to_tokens(enc, head, batch, device)
            mu = masked_mean(ht, mask)
            zmu = (mu - m_muT) / s_muT
            with torch.no_grad():
                stp = stop_forward(ht, mask, wq, bq, m_tT, s_tT)
            zt = z_from_mu_stop(mu, stp, m_muT, s_muT, m_stT, s_stT)
            with torch.no_grad():
                for k, arm, zi in (("M", armM, zmu), ("T", armT, zt)):
                    sD[k][ch] = F.linear(zi, arm["wD"], arm["bD"]).squeeze(-1).cpu().numpy()
                    fam[k][ch] = F.softmax(F.linear(zi, arm["wF"], arm["bF"]), 1).cpu().numpy()
            mu_arr[ch] = mu.cpu().numpy()
            st_arr[ch] = stp.cpu().numpy()
        return sD, fam, mu_arr, st_arr

    def snapshot():
        return {"M": {k: v.detach().clone().cpu() for k, v in armM.items()},
                "T": {k: v.detach().clone().cpu() for k, v in armT.items()},
                "wq": wq.detach().clone().cpu(), "bq": bq.detach().clone().cpu()}

    hist = {"M": [], "T": []}
    snaps = []
    s_top_hist = []
    for ep in range(args.epochs):
        t0 = time.time()
        nbatch = 0
        loss_acc = {"M": 0.0, "T": 0.0, "Dm": 0.0, "Dt": 0.0, "Fm": 0.0, "Ft": 0.0}
        gq_sum, gq_max, gbq_sum = 0.0, 0.0, 0.0
        ents = []
        for batch in _batches(corpus["train"], args.batch, True, seed=ep):
            ht, mask = to_tokens(enc, head, batch, device)
            mu = masked_mean(ht, mask)
            zmu = (mu - m_muT) / s_muT
            stp = stop_forward(ht, mask, wq, bq, m_tT, s_tT)
            zt = z_from_mu_stop(mu, stp, m_muT, s_muT, m_stT, s_stT)
            y = torch.as_tensor([d.y_ai for d in batch], dtype=torch.float32,
                                device=device)
            ai_idx = [i for i, d in enumerate(batch) if d.y_ai == 1]
            fam_t = (torch.as_tensor([SEEN_LABS.index(batch[i].fam) for i in ai_idx],
                                     device=device) if ai_idx
                     else torch.zeros(0, dtype=torch.long, device=device))
            lossM, L_Dm, L_Fm, ent_m = arm_forward(zmu, armM, y, ai_idx, fam_t)
            lossT, L_Dt, L_Ft, ent_t = arm_forward(zt, armT, y, ai_idx, fam_t)
            optM.zero_grad(); optT.zero_grad()
            lossM.backward(); lossT.backward()
            gq = float(wq.grad.norm()); gbq = float(bq.grad.abs().max())
            optM.step(); optT.step()
            nbatch += 1
            loss_acc["M"] += float(lossM.detach()); loss_acc["T"] += float(lossT.detach())
            loss_acc["Dm"] += float(L_Dm.detach()); loss_acc["Dt"] += float(L_Dt.detach())
            loss_acc["Fm"] += float(L_Fm.detach()); loss_acc["Ft"] += float(L_Ft.detach())
            gq_sum += gq; gq_max = max(gq_max, gq); gbq_sum += gbq
            if ent_m is not None:
                ents.append((ent_m, ent_t))
        nb = max(nbatch, 1)
        entm = float(np.mean([e[0] for e in ents])) if ents else float("nan")
        entt = float(np.mean([e[1] for e in ents])) if ents else float("nan")
        say(f"[e27] epoch {ep}：loss M {loss_acc['M']/nb:.4f}（D {loss_acc['Dm']/nb:.4f}"
            f"/F {loss_acc['Fm']/nb:.4f}）T {loss_acc['T']/nb:.4f}（D {loss_acc['Dt']/nb:.4f}"
            f"/F {loss_acc['Ft']/nb:.4f}）| ‖∇w_q‖ 均 {gq_sum/nb:.2e} 最大 {gq_max:.2e}"
            f" | ‖∇b_q‖ 均 {gbq_sum/nb:.2e} | ‖w_q‖ {float(wq.detach().norm()):.4f}"
            f" | 熵 M {entm:.3f} T {entt:.3f} | {time.time()-t0:.0f}s")

        sD, fam, mu_arr, st_arr = eval_arms(corpus["val"])
        for k in ("M", "T"):
            dm, _ = det_metrics(sD[k], corpus["val"])
            fm = fam_metrics(fam[k], corpus["val"])
            hist[k].append({"epoch": ep, "det": dm, "fam": fm, "loss": loss_acc[k] / nb})
            say(f"[e27] epoch {ep} val·{k}：det {dm['auroc']} | fam acc/bal "
                f"{fm['acc']}/{fm['bal']}")
        cos_v = float(F.cosine_similarity(
            torch.as_tensor(st_arr), torch.as_tensor(mu_arr), dim=1).mean())
        dim_var = st_arr.var(0)
        s_top_hist.append({
            "epoch": ep, "wq_norm": float(wq.detach().norm()),
            "grad_wq_mean": gq_sum / nb, "grad_wq_max": gq_max,
            "grad_bq_mean": gbq_sum / nb,
            "std_dims_mean": float(st_arr.std(0).mean()),
            "var_min_dim": float(dim_var.min()), "cos_mu_mean": cos_v,
            "dim_mean": [round(float(x), 5) for x in st_arr.mean(0)],
            "dim_std": [round(float(x), 5) for x in st_arr.std(0)]})
        say(f"[e27] epoch {ep} s_top：cos(μ)={cos_v:.4f} | 逐维方差最小 "
            f"{float(dim_var.min()):.3e}（失败阈 1e-6）")
        snaps.append(snapshot())

    # ---- epoch 选择（家族 bal → det AUROC → 较早）并按选中 epoch 回滚 ----
    sel = {}
    for k in ("M", "T"):
        best = max(hist[k], key=lambda h: (h["fam"]["bal"], h["det"]["auroc"],
                                           -h["epoch"]))
        sel[k] = best
        say(f"[e27] 选定 {k}：epoch {best['epoch']}（val bal {best['fam']['bal']}、"
            f"det {best['det']['auroc']}）")
    epM, epT = sel["M"]["epoch"], sel["T"]["epoch"]
    for k, v in snaps[epM]["M"].items():
        armM[k].data.copy_(v.to(device))
    for k, v in snaps[epT]["T"].items():
        armT[k].data.copy_(v.to(device))
    wq.data.copy_(snaps[epT]["wq"].to(device))
    bq.data.copy_(snaps[epT]["bq"].to(device))
    say(f"[e27] 回滚完成：M@{epM} / T@{epT}（w_q 随 T）")

    # ---- 离线特征臂（按 ids_md5 与 features.npz 对齐；全量等价、冒烟可裁） ----
    feats = dict(np.load(R25 / "features.npz", allow_pickle=True))
    meta25 = json.loads((R25 / "meta.json").read_text())

    def rows_for(split, docs):
        m = {h: i for i, h in enumerate(meta25[split]["ids_md5"])}
        return np.array([m[ids_md5(d.ids)] for d in docs])

    docs_tr, docs_va = corpus["train"], corpus["val"]
    docs_te, docs_un = corpus["test_seen"], corpus["unseen"]
    r_tr, r_va = rows_for("train", docs_tr), rows_for("val", docs_va)
    r_te, r_un = rows_for("test_seen", docs_te), rows_for("unseen", docs_un)
    rm_tr, rm_va = feats["rm_train"][r_tr], feats["rm_val"][r_va]
    rm_te, rm_un = feats["rm_test_seen"][r_te], feats["rm_unseen"][r_un]
    z_tr, z_va = feats["z_train"][r_tr], feats["z_val"][r_va]
    z_te, z_un = feats["z_test_seen"][r_te], feats["z_unseen"][r_un]
    rF_tr, rF_va = residualize(z_tr, st), residualize(z_va, st)
    rF_te, rF_un = residualize(z_te, st), residualize(z_un, st)

    def energy_b4(z):
        mu_c, v_c, pi = st["mu"], st["v"], st["pi"]
        G = 0.5 * (((z[:, None, :] - mu_c[None]) ** 2) / v_c[None]
                   + np.log(v_c)[None]).sum(-1)
        pif = pi[1:] / pi[1:].sum()
        g_ai = -np.log(np.exp(np.log(pif)[None] - G[:, 1:] / TE).sum(1))
        return G[:, 0] / TE - g_ai + np.log(pi[1:].sum() / pi[0]) + bD_e24

    y_tr_all = np.array([d.y_ai for d in docs_tr])
    y_va_all = np.array([d.y_ai for d in docs_va])
    ai_tr = np.array([d.y_ai == 1 and d.fam in SEEN_LABS for d in docs_tr])
    fam_tr = np.array([SEEN_LABS.index(d.fam) for d in docs_tr if
                       (d.y_ai == 1 and d.fam in SEEN_LABS)])
    ai_va = np.array([d.y_ai == 1 and d.fam in SEEN_LABS for d in docs_va])
    fam_va = np.array([SEEN_LABS.index(d.fam) for d in docs_va if
                       (d.y_ai == 1 and d.fam in SEEN_LABS)])

    reps = {"B0_raw_mean": (rm_tr, rm_va, rm_te, rm_un),
            "B2_mean_logv": (z_tr[:, :1536], z_va[:, :1536],
                             z_te[:, :1536], z_un[:, :1536]),
            "B3_rF": (rF_tr, rF_va, rF_te, rF_un)}
    offline = {}
    for name, (Xtr, Xva, Xte, Xun) in reps.items():
        scf = StandardScaler().fit(Xtr[ai_tr])
        rows = []
        for C in C_GRID:
            lr = LogisticRegression(max_iter=2000, C=C).fit(
                scf.transform(Xtr[ai_tr]), fam_tr)
            rows.append((C, float(balanced_accuracy_score(
                fam_va, lr.predict(scf.transform(Xva[ai_va]))))))
        best = max(b for _, b in rows)
        cand = [C for C, b in rows if abs(b - best) < 1e-12]
        if len(cand) == 1:
            C_star = cand[0]
        else:
            scd0 = StandardScaler().fit(Xtr)
            best_au, C_star = -1.0, min(cand)
            for C in cand:
                dlr = LogisticRegression(max_iter=2000, C=C).fit(
                    scd0.transform(Xtr), y_tr_all)
                au = roc_auc_score(y_va_all,
                                   dlr.predict_proba(scd0.transform(Xva))[:, 1])
                if au > best_au + 1e-12:
                    best_au, C_star = float(au), C
        say(f"[e27] {name}: C*={C_star}（网格 {[(c, round(b, 4)) for c, b in rows]}）")
        lrf = LogisticRegression(max_iter=2000, C=C_star).fit(
            scf.transform(Xtr[ai_tr]), fam_tr)
        scd = StandardScaler().fit(Xtr)
        lrd = LogisticRegression(max_iter=2000, C=C_star).fit(scd.transform(Xtr), y_tr_all)
        pv = np.zeros((len(docs_va), len(SEEN_LABS)), np.float32)
        pv[ai_va] = lrf.predict_proba(scf.transform(Xva[ai_va]))
        sD_v = lrd.decision_function(scd.transform(Xva))
        dm_v, sig_v = det_metrics(sD_v, docs_va)
        thr, f1v = best_f1_thr(sig_v, docs_va)
        offline[name] = {"C_star": C_star,
                         "val": {"det": dm_v, "fam": fam_metrics(pv, docs_va)},
                         "thr": thr, "f1@thr_val": round(f1v, 4),
                         "_lrf": lrf, "_scf": scf, "_lrd": lrd, "_scd": scd}
        say(f"[e27] {name}: val det {dm_v['auroc']} / fam bal "
            f"{offline[name]['val']['fam']['bal']}")

    sD_b4_va = energy_b4(z_va)
    dm4, sig4 = det_metrics(sD_b4_va, docs_va)
    thr4, f14 = best_f1_thr(sig4, docs_va)
    offline["B4_energy"] = {"C_star": None, "val": {"det": dm4}, "thr": thr4,
                            "f1@thr_val": round(f14, 4)}
    say(f"[e27] B4_energy：val det {dm4['auroc']}")

    # ---- 冻结声明 + test/unseen 一次性评估 ----
    say(f"[e27] FREEZE：epoch M={epM} / T={epT}；C* 已定；现在只读一次 "
        f"test_seen/unseen。")

    sD_va, fam_va_p, mu_va, stp_va = eval_arms(corpus["val"])
    sD_te, fam_te_p, _, _ = eval_arms(corpus["test_seen"])
    sD_un, fam_un_p, _, _ = eval_arms(corpus["unseen"])

    arms_test = {}
    for k in ("M", "T"):
        sig_va = 1 / (1 + np.exp(-np.clip(sD_va[k], -50, 50)))
        thr_k, f1t_k = best_f1_thr(sig_va, corpus["val"])
        dm_te, _ = det_metrics(sD_te[k], corpus["test_seen"], thr=thr_k)
        fm_te = fam_metrics(fam_te_p[k], corpus["test_seen"])
        un_det = unseen_det(sD_un[k], sD_va[k], corpus["val"])
        seen_p = [fam_va_p[k][i].max() for i, d in enumerate(corpus["val"])
                  if d.y_ai == 1]
        un_p = [fam_un_p[k][i].max() for i in range(len(corpus["unseen"]))]
        arms_test[k] = {"val_thr": thr_k, "test_det": dm_te, "test_fam": fm_te,
                        "unseen_det": un_det,
                        "unseen_rej": rej_metrics(seen_p, un_p)}
        say(f"[e27] {k}：te det {dm_te['auroc']} | te fam bal {fm_te['bal']} | "
            f"unseen det {un_det['auroc']} | rej −maxp "
            f"{arms_test[k]['unseen_rej']['auc_negmaxp']}")

    for name, (Xtr, Xva, Xte, Xun) in reps.items():
        o = offline[name]
        sD_te_o = o["_lrd"].decision_function(o["_scd"].transform(Xte))
        dm_te, _ = det_metrics(sD_te_o, corpus["test_seen"], thr=o["thr"])
        ai_te = np.array([d.y_ai == 1 and d.fam in SEEN_LABS
                          for d in corpus["test_seen"]])
        pte = np.zeros((len(corpus["test_seen"]), len(SEEN_LABS)), np.float32)
        pte[ai_te] = o["_lrf"].predict_proba(o["_scf"].transform(Xte[ai_te]))
        fm_te = fam_metrics(pte, corpus["test_seen"])
        sD_un_o = o["_lrd"].decision_function(o["_scd"].transform(Xun))
        sD_va_o = o["_lrd"].decision_function(o["_scd"].transform(Xva))
        un_det = unseen_det(sD_un_o, sD_va_o, corpus["val"])
        pv_full = np.zeros((len(docs_va), len(SEEN_LABS)), np.float32)
        pv_full[ai_va] = o["_lrf"].predict_proba(o["_scf"].transform(Xva[ai_va]))
        seen_p = [pv_full[i].max() for i, d in enumerate(corpus["val"]) if d.y_ai == 1]
        pun = o["_lrf"].predict_proba(o["_scf"].transform(Xun))
        un_p = [pun[i].max() for i in range(len(corpus["unseen"]))]
        o["test_det"] = dm_te; o["test_fam"] = fm_te; o["unseen_det"] = un_det
        o["unseen_rej"] = rej_metrics(seen_p, un_p)
        say(f"[e27] {name}：te det {dm_te['auroc']} | te fam bal {fm_te['bal']} | "
            f"unseen det {un_det['auroc']} | rej {o['unseen_rej']['auc_negmaxp']}")

    sD_b4_te = energy_b4(z_te)
    dm4_te, _ = det_metrics(sD_b4_te, corpus["test_seen"], thr=thr4)
    un_det4 = unseen_det(energy_b4(z_un), sD_b4_va, corpus["val"])
    offline["B4_energy"]["test_det"] = dm4_te
    offline["B4_energy"]["unseen_det"] = un_det4
    say(f"[e27] B4_energy：te det {dm4_te['auroc']} | unseen det {un_det4['auroc']}")

    # ---- 置换检查 ----
    t0 = time.time()
    sub = corpus["val"][:64]
    ht0, mask0 = to_tokens(enc, head, sub, device)
    mu0 = masked_mean(ht0, mask0)
    with torch.no_grad():
        st0 = stop_forward(ht0, mask0, wq, bq, m_tT, s_tT)
        z0 = z_from_mu_stop(mu0, st0, m_muT, s_muT, m_stT, s_stT)
    ht1 = ht0.clone()
    rng = np.random.RandomState(0)
    for i in range(len(sub)):
        n_i = int(mask0[i].sum())
        p = torch.as_tensor(rng.permutation(n_i), device=device)
        ht1[i, :n_i] = ht0[i, :n_i][p]
    mu1 = masked_mean(ht1, mask0)
    with torch.no_grad():
        st1 = stop_forward(ht1, mask0, wq, bq, m_tT, s_tT)
        z1 = z_from_mu_stop(mu1, st1, m_muT, s_muT, m_stT, s_stT)
    diff_a = float((z0 - z1).abs().max())
    sub2 = []
    for d in sub:
        nd = Doc(d.ids.copy(), d.y_ai, d.fam, d.lang)
        nd.split, nd.gen, nd.row = d.split, d.gen, d.row
        p = rng.permutation(len(nd.ids))
        nd.ids = nd.ids[p]
        sub2.append(nd)
    ht2, mask2 = to_tokens(enc, head, sub2, device)
    mu2 = masked_mean(ht2, mask2)
    with torch.no_grad():
        st2 = stop_forward(ht2, mask2, wq, bq, m_tT, s_tT)
        z2 = z_from_mu_stop(mu2, st2, m_muT, s_muT, m_stT, s_stT)
    diff_b = float((z0 - z2).abs().max())
    perm = {"fixedH_rowperm_maxdiff": diff_a, "rawtoken_reorder_maxdiff": diff_b,
            "n_samples": len(sub), "seconds": round(time.time() - t0, 1)}
    say(f"[e27] 置换：固定 H 行 {diff_a:.3e} | 原始 token 重排 {diff_b:.4f}")

    # ---- 产物 ----
    dup = json.loads((R26 / "e26_dedup.json").read_text())

    def slim(o):
        return {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
                for k, v in o.items()}

    metrics = {
        "commit": commit, "smoke": args.smoke, "seed": 0, "TE": TE,
        "std_md5": std_md5, "std_floor": STD_FLOOR, "lam_F": LAM_F,
        "counts": counts,
        "selection": {k: {"epoch": sel[k]["epoch"], "val_fam_bal": sel[k]["fam"]["bal"],
                          "val_det": sel[k]["det"]["auroc"]} for k in ("M", "T")},
        "epoch_history": {k: [{"epoch": h["epoch"], "loss": h["loss"],
                               "det": h["det"], "fam": h["fam"]} for h in hist[k]]
                          for k in ("M", "T")},
        "offline": slim(offline),
        "arms_test": arms_test,
        "arm_val": {k: {"det": hist[k][sel[k]["epoch"]]["det"],
                        "fam": hist[k][sel[k]["epoch"]]["fam"]} for k in ("M", "T")},
        "permutation": perm,
        "data_quality": {
            "exact_dup_groups": dup["exact_norm_dup_group_count"],
            "near_dup_pairs": dup["near_dup_pair_counts"],
            "handling": "记录、不删除任何样本（规范 §1.6）"},
        "test_read_once": True,
        "notes": {"token_basis": "E24 SetPool ht（W1/W2 冻结）",
                  "stop_weights": "top-k 支撑内 softmax（α）加权；字面 1/k 均匀对 w_q 不可导",
                  "grad_bq_note": "b_q 梯度≈0 为 softmax 平移不变性（设计性质，非实现失败）；"
                                   "w_q 梯度非零为 §6.3 主检查",
                  "cross_round": "E24/E25 已读过 test 的历史读数保留在各自报告；E27 内部只读一次"},
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=1))
    (OUT / "s_top_stats.json").write_text(json.dumps(
        {"init": {"wq_norm": init_wq_norm, "cos_mu": cos_init,
                  "var_min_dim": var_min_init,
                  "dim_mean": [round(float(x), 5) for x in m_st],
                  "dim_std": [round(float(x), 5) for x in s_st]},
         "epochs": s_top_hist}, ensure_ascii=False))
    (OUT / "permutation_checks.json").write_text(json.dumps(perm, ensure_ascii=False))
    torch.save({"wq": wq.detach().cpu(), "bq": bq.detach().cpu(),
                "armM": {k: v.detach().cpu() for k, v in armM.items()},
                "armT": {k: v.detach().cpu() for k, v in armT.items()},
                "m_t": m_t, "s_t": s_t, "m_mu": m_mu, "s_mu": s_mu,
                "m_st": m_st, "s_st": s_st,
                "selected": {k: sel[k]["epoch"] for k in sel},
                "C_star": {name: offline[name]["C_star"] for name in offline}},
               OUT / "model_state.pt")
    (OUT / "commit.txt").write_text(commit + "\n")
    cfg_out = {"args": vars(args), "C_GRID": C_GRID, "STD_FLOOR": STD_FLOOR,
               "TE": TE, "LAM_F": LAM_F,
               "token_basis": "E24 SetPool ht（W1/W2 冻结于 E24 训练值）",
               "s_top": "top-k(ceil(0.1n),min8) 支撑内 softmax 加权 token 均值（768 维）",
               "interpretation_notes": [
                   "字面 1/k 均匀权重对 w_q 不可导（∂s_top/∂w_q≡0 → §6 实现失败）；"
                   "按 §4.1 定义的 α 采用 top-k 内 softmax 权重",
                   "b_q 在 softmax 权重下平移不变 → ‖∇b_q‖≈0 属设计性质；§6.3 以 w_q 为准",
                   "标准化：token/μ/s_top(init) 三组训练折统计，std 下限 1e-2，全程固定"],
               "commit": commit, "std_md5": std_md5}
    (OUT / "e27_config.json").write_text(json.dumps(cfg_out, ensure_ascii=False, indent=1))
    (OUT / "README.md").write_text(
        "# runs/flagship_e27\n\nE27 修复显著位置通道（768 维 s_top、非零初始化）+ 公平基线。\n"
        f"- commit {commit}｜std_md5 {std_md5}｜TE {TE}\n"
        "- 文件：e27_config.json / metrics.json / s_top_stats.json / permutation_checks.json /\n"
        "  model_state.pt / train.log / commit.txt\n"
        "- 复现：`OMP_NUM_THREADS=8 python scripts/flagship_e27.py`（--smoke 快速）\n")
    say(f"[e27] 完成：{time.time()-t_start:.1f}s；产物 {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
