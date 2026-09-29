#!/usr/bin/env python
"""E24：旗舰轮（第一版）——来源流形 + 条件残差 + 对比分离 的单实验实现。

依据：`docx/d-det_core_method_spec_2026-09-29.md`（第一轮范围）。
本实验（一次训练、协议先定）：

  · 数据：SemEval-B 预分词 ids。已见家族 8 个（OpenAI/DeepSeek/Qwen/Meta/Mistral/
    IBM/01-ai/Google）+ Human；**held-out 家族 = Microsoft(Phi)/BigCode**（训练完全
    剔除，仅最终泛化/拒识测试）；b_test = 已见家族**未见新模型**（GPT-4o-mini 等）。
  · 表示：冻结 v1.0 d-det 编码器（codet5blk，全参微调底座）→
    h̃ = W2·GELU(W1·LN(h)) + LN(h)；z = [μ; log(v+ε); s_top] ∈ R^1537；
    s_top = top-K 的 w_q 分数均值，K = min(L, max(8, ⌈0.1L⌉))。
  · 流形：逐类对角收缩协方差（λ=0.9）能量 G_c；
    s_D = G_H − G_AI + log(πAI/πH) + b_D（b_D 可训练，起点 0）；
    δ = μ_AI − μ_H；r_F = z − μ_H − α(z)·δ；家族能量在 r_F 上（对照：raw z）。
  · 损失：L = 1·L_D + 1·L_F + 0.1·SupCon^D + 0.1·SupCon^F（batch 内正例优先跨语言；
    τ=0.1）；L_task-hard 因数据无 task_id **关闭**（协议记录）。
  · 协议：统计量仅训练折（初始化 + 每 epoch 末重算；验证/测试用冻结值）；2 epoch、
    冻结骨干；同数据对照 M0（raw-mean 逻辑回归/判别器探针）、M1（μ 段能量）、
    raw-z vs 残差家族能量；token 置换不变性检查。

运行：
  OMP_NUM_THREADS=8 python scripts/flagship_round1.py --smoke   # 冒烟
  OMP_NUM_THREADS=8 python scripts/flagship_round1.py           # 全量（~40min）
输出：runs/flagship_r1/{results.json, head.pt, stats.npz}
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from encoders import build_encoder  # noqa: E402
from models import build_model  # noqa: E402
from models.disc import DiscHead  # noqa: E402

BIG = ROOT / "data/processed/semeval_big"
OUT = ROOT / "runs/flagship_r1"

SEEN_LABS = ["OpenAI", "DeepSeek", "Qwen", "Meta", "Mistral", "IBM", "01-ai", "Google"]
UNSEEN_LABS = ["Microsoft", "BigCode"]
ALL_LABS = ["Human"] + SEEN_LABS
EPS = 1e-6
LAMBDA_SHRINK = 0.9          # 对角收缩（第一轮预固定）
TAU_D, TAU_F = 0.1, 0.1
LAM = {"D": 1.0, "F": 1.0, "CD": 0.1, "CF": 0.1}


def lab_of(gen: str) -> str:
    g = str(gen)
    if g == "Human":
        return "Human"
    for pat, lab in (("GPT", "OpenAI"), ("deepseek", "DeepSeek"), ("Qwen/", "Qwen"),
                     ("meta-llama/", "Meta"), ("mistralai/", "Mistral"),
                     ("ibm-granite/", "IBM"), ("01-ai/", "01-ai"),
                     ("gemma", "Google"), ("microsoft/", "Microsoft"),
                     ("bigcode/", "BigCode"), ("Phi", "Microsoft")):
        if pat.lower() in g.lower():
            return lab
    return "UNKNOWN"


@dataclass
class Doc:
    ids: np.ndarray
    y_ai: int
    fam: str
    lang: str


# --------------------------------------------------------------------------- #
# 数据
# --------------------------------------------------------------------------- #
def load_corpus(args) -> dict:
    rng = np.random.RandomState(0)

    def subset(path, human_cap, fam_cap, unseen_cap):
        rows = pq.read_table(path, columns=["ids", "generator", "language"]).to_pylist()
        human, seen, unseen = [], [], {}
        by_fam = {}
        for r in rows:
            lab = lab_of(r["generator"])
            if lab == "UNKNOWN":
                continue
            ids = np.asarray(r["ids"][:args.max_len], dtype="int64")
            if len(ids) == 0:
                continue
            d = Doc(ids, int(lab != "Human"), lab, str(r["language"]))
            if lab == "Human":
                human.append(d)
            elif lab in SEEN_LABS:
                by_fam.setdefault(lab, []).append(d)
            elif lab in UNSEEN_LABS:
                unseen.setdefault(lab, []).append(d)
        rng.shuffle(human)
        out = {"human": human[:human_cap], "seen": [], "unseen": []}
        for f in SEEN_LABS:
            lst = by_fam.get(f, [])
            rng.shuffle(lst)
            out["seen"].extend(lst[:fam_cap])
        for f, lst in unseen.items():                     # 未见家族：按族各取上限
            rng.shuffle(lst)
            out["unseen"].extend(lst[:unseen_cap])
        return out

    tr = subset(BIG / "b_train.parquet", args.train_human, args.train_fam, args.unseen_cap)
    va = subset(BIG / "b_val.parquet", args.val_human, args.val_fam, args.unseen_cap)
    te = subset(BIG / "b_test.parquet", args.test_human, args.test_fam, args.unseen_cap)
    corpus = {"train": tr["human"] + tr["seen"],
              "val": va["human"] + va["seen"],
              "test_seen": te["human"] + te["seen"],
              "unseen": tr["unseen"] + va["unseen"] + te["unseen"]}
    meta = {k: {f: sum(1 for d in v if d.fam == f)
                for f in ["Human"] + SEEN_LABS + UNSEEN_LABS}
            for k, v in corpus.items()}
    return {"docs": corpus, "meta": meta}


def _batches(docs, bs, shuffle, seed=0):
    idx = np.argsort([len(d.ids) for d in docs], kind="stable")
    chunks = [idx[i:i + bs] for i in range(0, len(idx), bs)]
    if shuffle:
        np.random.RandomState(seed).shuffle(chunks)
    for ch in chunks:
        yield [docs[i] for i in ch]


def collate(docs, device):
    L = max(len(d.ids) for d in docs)
    ids = torch.zeros(len(docs), L, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for j, d in enumerate(docs):
        ids[j, :len(d.ids)] = torch.from_numpy(d.ids)
        mask[j, :len(d.ids)] = 1
    return ids.to(device), mask.to(device)


# --------------------------------------------------------------------------- #
# 模型头
# --------------------------------------------------------------------------- #
class SetPool(nn.Module):
    """h (B,L,768) → z (B,1537)：LN→MLP 残差 → [μ; log v; s_top]。"""

    def __init__(self, dim=768, d_m=512):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.W1 = nn.Linear(dim, d_m)
        self.W2 = nn.Linear(d_m, dim)
        self.wq = nn.Linear(dim, 1)
        nn.init.zeros_(self.wq.weight)
        nn.init.zeros_(self.wq.bias)

    def forward(self, h, mask):
        u = self.ln(h.float())
        ht = (self.W2(F.gelu(self.W1(u))) + u) * mask.unsqueeze(-1)
        n = mask.sum(1, keepdim=True).clamp(min=1)
        mu = ht.sum(1) / n
        dev = (ht - mu[:, None]) ** 2 * mask.unsqueeze(-1)
        v = dev.sum(1) / n
        q = self.wq(ht).squeeze(-1).masked_fill(mask == 0, -1e9)
        L = q.shape[1]
        k_i = torch.clamp(torch.ceil(0.1 * mask.sum(1)).long(), min=8)
        k_i = torch.minimum(k_i, mask.sum(1).long()).clamp(min=1)
        vals, _ = torch.sort(q, dim=1, descending=True)
        cols = torch.arange(L, device=q.device)[None, :]
        keep = (cols < k_i[:, None]).to(vals.dtype)
        s_top = (vals * keep).sum(1) / k_i.clamp(min=1)
        return torch.cat([mu, torch.log(v + EPS), s_top.unsqueeze(-1)], dim=1)


def energy(z, mu_c, v_c):
    r = z[:, None, :] - mu_c[None]
    return 0.5 * ((r * r) / v_c[None] + torch.log(v_c)[None]).sum(-1)


def supcon(emb, labels, langs, tau, prefer_diff_lang=True):
    e = emb / emb.norm(dim=1, keepdim=True).clamp(min=1e-9)
    sim = e @ e.t() / tau
    n = len(labels)
    same = torch.eq(labels[:, None], labels[None, :]) & ~torch.eye(
        n, dtype=torch.bool, device=emb.device)
    if prefer_diff_lang:
        diff = same & torch.ne(langs[:, None], langs[None, :])
        use = torch.where(diff.sum(1, keepdim=True) > 0, diff, same)
    else:
        use = same
    cntp = use.sum(1)
    valid = cntp > 0
    if int(valid.sum()) == 0:
        return torch.zeros((), device=emb.device)
    sim = sim - sim.max(1, keepdim=True).values.detach()
    logz = torch.logsumexp(sim.masked_fill(
        torch.eye(n, dtype=torch.bool, device=emb.device), -1e9), dim=1)
    lp = ((sim - logz[:, None]) * use).sum(1) / cntp.clamp(min=1)
    return -lp[valid].mean()


# --------------------------------------------------------------------------- #
# 统计（训练折内；numpy）
# --------------------------------------------------------------------------- #
def z_pass(enc, head, docs, bs, device, keep_raw_mean=False):
    """按长度分桶加速，但输出**回填到输入 docs 的原序**。"""
    order = np.argsort([len(d.ids) for d in docs], kind="stable")
    chunks = [order[i:i + bs] for i in range(0, len(order), bs)]
    z_out = np.empty((len(docs), 2 * 768 + 1), dtype="float32")
    rm_out = np.empty((len(docs), 768), dtype="float32") if keep_raw_mean else None
    with torch.no_grad():
        for ch in chunks:
            batch = [docs[i] for i in ch]
            ids, mask = collate(batch, device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                h = enc(ids, mask)
            z_out[ch] = head(h, mask).float().cpu().numpy()
            if keep_raw_mean:
                for j, orig in enumerate(ch):
                    rm_out[orig] = h[j][mask[j].bool()].float().mean(0).cpu().numpy()
    return z_out, rm_out


def accumulate_class_stats(zs, docs):
    n_cls = len(ALL_LABS)
    s1 = np.zeros((n_cls, zs.shape[1])); s2 = np.zeros_like(s1); cnt = np.zeros(n_cls)
    f1 = np.zeros((len(SEEN_LABS), zs.shape[1])); f2 = np.zeros_like(f1)
    fcnt = np.zeros(len(SEEN_LABS))
    ci = {c: i for i, c in enumerate(ALL_LABS)}
    for z, d in zip(zs, docs):
        k = ci[d.fam]
        s1[k] += z; s2[k] += z * z; cnt[k] += 1
        if d.y_ai:
            f = SEEN_LABS.index(d.fam)
            f1[f] += z; f2[f] += z * z; fcnt[f] += 1
    return s1, s2, cnt, f1, f2, fcnt


def class_moments(s1, s2, cnt):
    mu = s1 / np.maximum(cnt[:, None], 1)
    v = np.clip(s2 / np.maximum(cnt[:, None], 1) - mu * mu, 0, None)
    return mu, v


def shrink(v, n):
    v_pool = (v * n[:, None]).sum(0) / max(n.sum(), 1)
    return (1 - LAMBDA_SHRINK) * v + LAMBDA_SHRINK * v_pool + EPS


def make_stats(zstats, mudim=False):
    s1, s2, cnt, f1_, f2_, fcnt = zstats
    if mudim:
        s1, s2 = s1[:, :768], s2[:, :768]
    mu, v = class_moments(s1, s2, cnt)
    v_shr = shrink(v, cnt)
    pi = cnt / max(cnt.sum(), 1)
    mu_ai = (pi[1:, None] * mu[1:]).sum(0) / max(pi[1:].sum(), 1e-9)
    fam_mu, fam_v = class_moments(f1_, f2_, fcnt)
    f32 = lambda a: np.asarray(a, dtype=np.float32)  # noqa: E731
    return {"mu": f32(mu), "v": f32(v_shr), "pi": f32(pi), "delta": f32(mu_ai - mu[0]),
            "mu_H": f32(mu[0]), "vH": f32(v_shr[0]),
            "fam_mu": f32(fam_mu), "fam_v_raw": f32(shrink(fam_v, fcnt)),
            "fam_n": fcnt}


def residualize(zs, st):
    delta, muH, vH = st["delta"], st["mu_H"], st["vH"]
    denom = float((delta * delta / vH).sum()) + EPS
    alpha = ((zs - muH) / vH) @ delta / denom
    return zs - muH[None] - alpha[:, None] * delta[None]


def residual_fam_stats(rF, docs):
    f1 = np.zeros((len(SEEN_LABS), rF.shape[1])); f2 = np.zeros_like(f1)
    fcnt = np.zeros(len(SEEN_LABS))
    for r, d in zip(rF, docs):
        if not d.y_ai:
            continue
        f = SEEN_LABS.index(d.fam)
        f1[f] += r; f2[f] += r * r; fcnt[f] += 1
    mu, v = class_moments(f1, f2, fcnt)
    return {"mu": mu.astype(np.float32), "v": shrink(v, fcnt).astype(np.float32),
            "n": fcnt}


# --------------------------------------------------------------------------- #
# numpy 评分
# --------------------------------------------------------------------------- #
def energy_np(z, mu, v):
    r = z[:, None, :] - mu[None]
    return 0.5 * ((r * r) / v[None] + np.log(v)[None]).sum(-1)


def logsumexp_np(a):
    m = a.max(1, keepdims=True)
    return (m + np.log(np.exp(a - m).sum(1, keepdims=True)))[:, 0]


def s_d_scalar(z, st, bD=0.0):
    G = energy_np(z, st["mu"], st["v"]) / z.shape[1]
    pi = st["pi"]
    pi_f = pi[1:] / pi[1:].sum()
    lse = logsumexp_np(np.log(pi_f)[None] - G[:, 1:])
    return G[:, 0] + lse + np.log(pi[1:].sum() / pi[0]) + bD


def fam_post(z, mu, v, pi):
    G = energy_np(z, mu, v)
    if G.shape[1] == len(pi):                  # 传入含 Human 的全类统计
        G = G[:, 1:]
    pi_f = pi[1:] / pi[1:].sum()
    logits = -G / z.shape[1] + np.log(pi_f)[None]
    m = logits.max(1, keepdims=True)
    e = np.exp(logits - m)
    return e / e.sum(1, keepdims=True)


def fam_metrics(p, docs):
    idx = [i for i, d in enumerate(docs) if d.y_ai == 1 and d.fam in SEEN_LABS]
    if not idx:
        return {"acc": None, "n": 0}
    yt = np.array([SEEN_LABS.index(docs[i].fam) for i in idx])
    pred = p[idx].argmax(1)
    return {"acc": round(float((pred == yt).mean()), 4),
            "balanced_acc": round(float(balanced_accuracy_score(yt, pred)), 4),
            "n": int(len(yt))}


def det_metrics(sD, docs):
    y = np.array([d.y_ai for d in docs])
    if len(set(y.tolist())) < 2:
        return None
    sig = 1 / (1 + np.exp(-np.clip(sD, -50, 50)))
    return {"auroc": round(float(roc_auc_score(y, sD)), 4),
            "macro_f1": round(float(f1_score(y, (sig > 0.5).astype(int),
                                             average="macro")), 4),
            "n": int(len(y))}


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--enc-bs", type=int, default=32)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--train-human", type=int, default=8000)
    ap.add_argument("--train-fam", type=int, default=1200)
    ap.add_argument("--val-human", type=int, default=1500)
    ap.add_argument("--val-fam", type=int, default=500)
    ap.add_argument("--test-human", type=int, default=500)
    ap.add_argument("--test-fam", type=int, default=200)
    ap.add_argument("--unseen-cap", type=int, default=600)
    ap.add_argument("--d-m", type=int, default=512)
    args = ap.parse_args()
    if args.smoke:
        args.train_human, args.train_fam = 128, 24
        args.val_human, args.val_fam = 48, 12
        args.test_human, args.test_fam = 24, 10
        args.unseen_cap = 16
        args.enc_bs, args.batch = 16, 32
        args.max_len = 768

    OUT.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(ROOT / "configs/ddet_v041.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    enc_cfg = dict(cfg["encoder"])
    enc_cfg["path"] = str(ROOT / enc_cfg["path"])
    enc = build_encoder(**enc_cfg)
    dual = build_model("dual", encoder=enc, dim=enc.hidden_size,
                       pool=cfg["model"].get("pool", "mean"),
                       s2_rank=cfg["model"].get("s2_rank", 1))
    ck = torch.load(str(ROOT / "runs/v0.4.1_covreg/last.pt"), map_location="cpu",
                    weights_only=False)
    dual.load_state_dict(ck["state"], strict=False)
    enc = dual.encoder.to(device).eval()
    for p in enc.parameters():
        p.requires_grad = False
    print(f"[r1] 冻结编码器就绪（v1.0 codet5blk） device={device}", flush=True)

    pack = load_corpus(args)
    corpus, meta = pack["docs"], pack["meta"]
    print("[r1] 语料：" + " | ".join(f"{k}:{len(v)}" for k, v in corpus.items()),
          flush=True)
    print(f"[r1] 训练分布：{json.dumps(meta['train'], ensure_ascii=False)}", flush=True)

    head = SetPool(768, args.d_m).to(device)
    bD = nn.Parameter(torch.zeros(1, device=device))
    opt = torch.optim.AdamW(list(head.parameters()) + [bD], lr=args.lr, weight_decay=1e-4)

    def full_stats(docs_train, keep_raw=False):
        t0 = time.time()
        zs, rms = z_pass(enc, head, docs_train, args.enc_bs, device, keep_raw)
        zstats = accumulate_class_stats(zs, docs_train)
        st = make_stats(zstats)
        st_mu = make_stats(zstats, mudim=True)
        rF = residualize(zs, st)
        fs = residual_fam_stats(rF, docs_train)
        st["fam_res_mu"], st["fam_res_v"] = fs["mu"], fs["v"]
        print(f"[r1] 统计轮：{len(docs_train)} 条 / {time.time()-t0:.1f}s", flush=True)
        return st, st_mu, rms

    st, st_mu, raw_means_train = full_stats(corpus["train"], keep_raw=True)

    def tensor_pack(st_, device):
        T = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32, device=device)
        return {"mu_c": T(st_["mu"]), "v_c": T(st_["v"]), "pi": T(st_["pi"]),
                "delta": T(st_["delta"]), "muH": T(st_["mu_H"]), "vH": T(st_["vH"]),
                "mu_f": T(st_["fam_res_mu"]), "v_f": T(st_["fam_res_v"]),
                "logpi_f": torch.log(T(st_["pi"])[1:]),
                "log_ratio": float(np.log(st_["pi"][1:].sum() / st_["pi"][0]))}

    def quick_val(st_, st_mu_):
        zv, _ = z_pass(enc, head, corpus["val"], args.enc_bs, device, False)
        rFv = residualize(zv, st_)
        pi = st_["pi"]
        det = det_metrics(s_d_scalar(zv, st_, float(bD.detach().cpu())), corpus["val"])
        fam_res = fam_metrics(fam_post(rFv, st_["fam_res_mu"], st_["fam_res_v"], pi),
                              corpus["val"])
        fam_raw = fam_metrics(fam_post(zv, st_["fam_mu"], st_["fam_v_raw"], pi),
                              corpus["val"])
        return {"det": det, "fam_res": fam_res, "fam_raw": fam_raw}

    log = []
    for ep in range(args.epochs):
        t0 = time.time()
        tp = tensor_pack(st, device)
        tot = 0.0; nb = 0; acc = {"D": 0.0, "F": 0.0, "CD": 0.0, "CF": 0.0}
        for batch in _batches(corpus["train"], args.batch, True, seed=ep):
            ids, mask = collate(batch, device)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16,
                                                 enabled=device == "cuda"):
                h = enc(ids, mask)
            z = head(h, mask)
            dim = z.shape[1]
            G = energy(z, tp["mu_c"], tp["v_c"]) / dim
            g_ai = -torch.logsumexp(tp["logpi_f"] - G[:, 1:], dim=1)
            sD = G[:, 0] - g_ai + tp["log_ratio"] + bD
            y = torch.as_tensor([d.y_ai for d in batch], dtype=torch.float32,
                                device=device)
            L_D = F.binary_cross_entropy_with_logits(sD, y)
            alpha = ((z - tp["muH"]) / tp["vH"]) @ tp["delta"] / (
                (tp["delta"] ** 2 / tp["vH"]).sum() + EPS)
            rF = z - tp["muH"] - alpha[:, None] * tp["delta"]
            log_pf = -energy(rF, tp["mu_f"], tp["v_f"]) / dim + tp["logpi_f"][None]
            log_pf = log_pf - torch.logsumexp(log_pf, dim=1, keepdim=True)
            ai_idx = [i for i, d in enumerate(batch) if d.y_ai == 1]
            if ai_idx:
                fam_t = torch.as_tensor([SEEN_LABS.index(batch[i].fam) for i in ai_idx],
                                        device=device)
                L_F = F.nll_loss(log_pf[ai_idx], fam_t)
            else:
                L_F = torch.zeros((), device=device)
            y_lab = torch.as_tensor([d.y_ai for d in batch], device=device)
            lang_l = torch.as_tensor(
                [hash(d.lang) % 64 for d in batch], device=device)
            L_CD = supcon(z, y_lab, lang_l, TAU_D)
            if ai_idx:
                fam_l = torch.as_tensor([SEEN_LABS.index(batch[i].fam) for i in ai_idx],
                                        device=device)
                lang_ai = torch.as_tensor([hash(batch[i].lang) % 64 for i in ai_idx],
                                          device=device)
                L_CF = supcon(rF[ai_idx], fam_l, lang_ai, TAU_F)
            else:
                L_CF = torch.zeros((), device=device)
            loss = LAM["D"] * L_D + LAM["F"] * L_F + LAM["CD"] * L_CD + LAM["CF"] * L_CF
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.detach()); nb += 1
            acc["D"] += float(L_D.detach()); acc["F"] += float(L_F.detach())
            acc["CD"] += float(L_CD.detach()); acc["CF"] += float(L_CF.detach())
        msg = (f"[r1] epoch {ep}: loss {tot/max(nb,1):.4f} "
               f"(D {acc['D']/nb:.4f} F {acc['F']/nb:.4f} "
               f"CD {acc['CD']/nb:.4f} CF {acc['CF']/nb:.4f}) / {time.time()-t0:.0f}s")
        print(msg, flush=True)
        log.append(msg)
        st, st_mu, _ = full_stats(corpus["train"])
        qv = quick_val(st, st_mu)
        msg2 = (f"[r1] epoch {ep} 验证：det {qv['det']} | 家族残差 {qv['fam_res']} "
                f"| 家族raw {qv['fam_raw']}")
        print(msg2, flush=True)
        log.append(msg2)

    # ---------- 最终评估 ----------
    def evaluate():
        out = {}
        zs = {}
        for name in ("val", "test_seen", "unseen"):
            z, rm = z_pass(enc, head, corpus[name], args.enc_bs, device, True)
            zs[name] = (z, rm)
        bDv = float(bD.detach().cpu())
        for name in ("val", "test_seen"):
            z, rm = zs[name]
            rF = residualize(z, st)
            r = {"detection_sD": det_metrics(s_d_scalar(z, st, bDv), corpus[name]),
                 "detection_m1": det_metrics(s_d_scalar(z[:, :768], st_mu, 0.0),
                                             corpus[name]),
                 "family_residual": fam_metrics(
                     fam_post(rF, st["fam_res_mu"], st["fam_res_v"], st["pi"]),
                     corpus[name]),
                 "family_rawz": fam_metrics(
                     fam_post(z, st["fam_mu"], st["fam_v_raw"], st["pi"]), corpus[name]),
                 "family_m1": fam_metrics(
                     fam_post(z[:, :768], st_mu["mu"], st_mu["v"], st_mu["pi"]),
                     corpus[name])}
            out[name] = r
            # s_D 与"预测家族能量"的相关
            ai_i = [i for i, d in enumerate(corpus[name]) if d.y_ai == 1]
            if ai_i:
                G = energy_np(rF[ai_i], st["fam_res_mu"], st["fam_res_v"])
                Gpred = G[np.arange(len(ai_i)), G.argmin(1)]
                sD = s_d_scalar(z, st, bDv)[ai_i]
                out[name]["corr_sD_Gpred"] = round(float(np.corrcoef(sD, Gpred)[0, 1]), 4)
        # 未见家族：检测（val 人类 vs unseen AI）+ 拒识
        zv, _ = zs["val"]; zu, _ = zs["unseen"]
        zh = np.array([d.y_ai for d in corpus["val"]]) == 0
        sD_all = np.concatenate([s_d_scalar(zv[zh], st, bDv), s_d_scalar(zu, st, bDv)])
        y_all = np.concatenate([np.zeros(int(zh.sum())), np.ones(len(zu))])
        det_unseen = {"auroc": round(float(roc_auc_score(y_all, sD_all)), 4),
                      "n": int(len(y_all))}
        zu_all = np.concatenate([zv[zh], zu])
        out["unseen"] = {"detection_sD_vs_human": det_unseen,
                         "detection_m1_vs_human": {
                             "auroc": round(float(roc_auc_score(
                                 y_all, np.concatenate(
                                     [s_d_scalar(zv[zh][:, :768], st_mu, 0.0),
                                      s_d_scalar(zu[:, :768], st_mu, 0.0)]))), 4)}}
        pv = fam_post(residualize(zv, st), st["fam_res_mu"], st["fam_res_v"], st["pi"])
        pu = fam_post(residualize(zu, st), st["fam_res_mu"], st["fam_res_v"], st["pi"])
        seen_p = [pv[i].max() for i, d in enumerate(corpus["val"]) if d.y_ai]
        unseen_p = [pu[i].max() for i in range(len(corpus["unseen"]))]
        lab = np.array([0] * len(seen_p) + [1] * len(unseen_p))
        scr = np.array(seen_p + unseen_p)
        out["unseen"]["rejection_auc_maxp"] = round(float(roc_auc_score(lab, scr)), 4)
        out["unseen"]["mean_maxp_seen"] = round(float(np.mean(seen_p)), 4)
        out["unseen"]["mean_maxp_unseen"] = round(float(np.mean(unseen_p)), 4)
        return out

    results = {"meta": meta, "args": vars(args), "log": log}
    results["metrics"] = evaluate()
    for name, r in results["metrics"].items():
        print(f"[r1] 评估 {name}: {json.dumps(r, ensure_ascii=False, default=str)}",
              flush=True)

    # ---------- M0 探针（raw mean） ----------
    if raw_means_train is not None:
        ai_tr = [i for i, d in enumerate(corpus["train"]) if d.y_ai == 1]
        y_tr = np.array([d.y_ai for d in corpus["train"]])
        scaler = StandardScaler().fit(raw_means_train)
        lr = LogisticRegression(max_iter=3000, C=1.0).fit(
            scaler.transform(raw_means_train), y_tr)
        dh = DiscHead().fit(raw_means_train[ai_tr],
                            [corpus["train"][i].fam for i in ai_tr])
        m0 = {}
        # 重新取 raw means（评估循环内未保留）——直接再算（轻量）
        for name in ("val", "test_seen"):
            _, rm = z_pass(enc, head, corpus[name], args.enc_bs, device, True)
            y = np.array([d.y_ai for d in corpus[name]])
            prob = lr.predict_proba(scaler.transform(rm))[:, 1]
            ai_i = [i for i, d in enumerate(corpus[name])
                    if d.y_ai == 1 and d.fam in SEEN_LABS]
            p = dh.predict(rm[ai_i])
            yt = np.array([corpus[name][i].fam for i in ai_i])
            m0[name] = {"det_auroc": round(float(roc_auc_score(y, prob)), 4),
                        "fam_acc": round(float((p == yt).mean()), 4),
                        "fam_bal": round(float(balanced_accuracy_score(yt, p)), 4)}
        results["m0"] = m0
        print(f"[r1] M0 探针: {json.dumps(m0, ensure_ascii=False)}", flush=True)

    # ---------- token 置换不变性 ----------
    d0 = corpus["val"][0]
    with torch.no_grad():
        ids0, mask0 = collate([d0], device)
        h0 = enc(ids0, mask0)                    # 固定编码器输出
        z0 = head(h0, mask0).cpu().numpy()
        perm = np.random.RandomState(0).permutation(h0.shape[1])   # 置换 token 行
        zp = head(h0[:, perm, :], mask0[:, perm]).cpu().numpy()
    results["perm_invariance_maxdiff"] = float(np.abs(z0 - zp).max())
    results["notes"] = {
        "task_hard": "数据无 task_id，L_task-hard 关闭（协议记录）",
        "stats_update": "初始化 + 每 epoch 末重算（训练中冻结；验证用冻结值）",
        "lambda_shrink": LAMBDA_SHRINK, "taus": [TAU_D, TAU_F], "lam": LAM,
        "backbone": "v1.0 d-det 编码器（runs/v0.4.1_covreg/last.pt），冻结",
        "deviation": ("① 概率形成时能量按维度归一（÷1537；等价温度 T=1537）——原始尺度下"
                      "高维 NLL 差 O(10²) 使 sigmoid/softmax 饱和、L_D 无梯度；"
                      "② b_test 作为'已见家族未见模型'测试；Microsoft/BigCode 全剔除训练；"
                      "③ λ、温度、权重未做任何验证集调优（固定值）。"),
    }
    results["timing_min"] = round((time.time() - t_start) / 60, 1)

    torch.save({"head": head.state_dict(), "bD": float(bD.detach().cpu()),
                "args": vars(args)}, OUT / "head.pt")
    np.savez_compressed(
        OUT / "stats.npz",
        **{k: np.asarray(v) for k, v in st.items() if k != "pi"},
        pi=st["pi"])
    (OUT / "results.json").write_text(json.dumps(results, ensure_ascii=False,
                                                 indent=2, default=str))
    print(f"[r1] 完成：{results['timing_min']} min → {OUT/'results.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
