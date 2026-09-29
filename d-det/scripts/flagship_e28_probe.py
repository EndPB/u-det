#!/usr/bin/env python
"""E28：E27 冻结表示的公平读出审计（不训练、不读 test/unseen、不新增数据）。

依据 `docx/d-det_E27复核与E28执行指导_2026-09-30.md`：
  · 冻结 E27 选定 epoch 的 w_q/b_q、编码器、LN/W1/W2；只导出 train/val 的 m_raw/μ/s；
  · 四臂读出：R0=m_raw、R1=μ、R2=s、R3=[μ;s]（同类 L2 正则化逻辑回归，sklearn lbfgs、
    tol=1e-6、max_iter=10000；C 网格 {0.01..10}；C_F* 按 val BalAcc、C_D* 按 val AUROC）；
  · 标准化 (13) 按表示×任务独立、只用训练折、std 下限 1e-2；R1/R3 的 μ 列必须逐位一致；
  · 复现检查：用 E27 保存的标准量与头复算其 val 指标；m_mu/s_mu 重算比对；
  · dup-mean 等价检查、centered cos / η_Δ 几何诊断、家族内配对 bootstrap（1000× seed=0）；
  · 逐家族召回 + 语言/长度分桶；test_accessed=false。

硬约束：只读 b_train/b_val parquet（过滤留出家族）；主键=(原文件 split, row_index)，md5 仅校验；
不 import flagship_e25_stage0（避免默认加载器）；不读 E25 features.npz。
冒烟写 runs/flagship_e28_smoke；正式写 runs/flagship_e28；已有产物默认拒绝覆盖。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
import warnings
from argparse import Namespace
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
import yaml
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (balanced_accuracy_score, f1_score, log_loss,
                             roc_auc_score)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from encoders import build_encoder  # noqa: E402
from models import build_model  # noqa: E402
from flagship_round1 import (Doc, SetPool, SEEN_LABS, UNSEEN_LABS,  # noqa: E402
                             lab_of, collate)

BIG = ROOT / "data/processed/semeval_big"
R1 = ROOT / "runs/flagship_r1"
R27 = ROOT / "runs/flagship_e27"
R28 = ROOT / "runs/flagship_e28"

C_GRID = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
STD_FLOOR = 1e-2
MAX_ITER = 10000
TOL = 1e-6
SEED = 0
N_BOOT = 1000
LEN_EDGES = [(8, 128), (129, 256), (257, 512), (513, 1024), (1025, 2048)]


def default_args(smoke: bool = False) -> Namespace:
    if smoke:
        return Namespace(smoke=True, max_len=2048, train_human=400, train_fam=60,
                         val_human=200, val_fam=40, unseen_cap=600, enc_bs=16,
                         batch=32)
    return Namespace(smoke=False, max_len=2048, train_human=8000, train_fam=1200,
                     val_human=1500, val_fam=500, unseen_cap=600, enc_bs=32,
                     batch=64)


# --------------------------------------------------------------------------- #
# 只读 train/val 的语料加载（复制 load_corpus_tagged 的 subset() rng 行为；
# 不读 b_test.parquet、不导出留出家族样本）
# --------------------------------------------------------------------------- #
def load_train_val(args) -> dict:
    rng = np.random.RandomState(0)

    def subset(path, human_cap, fam_cap, split_name):
        rows = pq.read_table(path, columns=["ids", "generator", "language"]).to_pylist()
        human, by_fam, unseen = [], {}, {}
        for ri, r in enumerate(rows):
            lab = lab_of(r["generator"])
            if lab == "UNKNOWN":
                continue
            ids = np.asarray(r["ids"][:args.max_len], dtype="int64")
            if len(ids) == 0:
                continue
            d = Doc(ids, int(lab != "Human"), lab, str(r["language"]))
            d.split = split_name
            d.gen = str(r["generator"])
            d.row = ri
            if lab == "Human":
                human.append(d)
            elif lab in SEEN_LABS:
                by_fam.setdefault(lab, []).append(d)
            elif lab in UNSEEN_LABS:
                unseen.setdefault(lab, []).append(d)
        rng.shuffle(human)
        out = {"human": human[:human_cap], "seen": []}
        for f in SEEN_LABS:
            lst = by_fam.get(f, [])
            rng.shuffle(lst)
            out["seen"].extend(lst[:fam_cap])
        for f, lst in unseen.items():          # 保持与旧实现相同的 rng 消耗（不导出）
            rng.shuffle(lst)
        return out

    tr = subset(BIG / "b_train.parquet", args.train_human, args.train_fam, "train")
    va = subset(BIG / "b_val.parquet", args.val_human, args.val_fam, "val")
    return {"train": tr["human"] + tr["seen"], "val": va["human"] + va["seen"]}


def ids_md5(ids) -> str:
    return hashlib.md5(np.ascontiguousarray(ids, dtype=np.int64).tobytes()).hexdigest()


# --------------------------------------------------------------------------- #
# 冻结前向（与 E27 to_tokens/stop_forward 完全一致）
# --------------------------------------------------------------------------- #
def stop_forward(ht, mask, wq, bq, m_t, s_t):
    hn = (ht - m_t) / s_t
    q = (hn @ wq + bq).squeeze(-1)
    q = q.masked_fill(mask == 0, -1e9)
    n = mask.sum(1)
    k = torch.clamp(torch.ceil(0.1 * n).long(), min=8)
    k = torch.minimum(k, n.long()).clamp(min=1)
    kmax = int(k.max())
    topq, topi = torch.topk(q, kmax, dim=1)
    keep = (torch.arange(kmax, device=q.device)[None, :] < k[:, None])
    a = torch.softmax(topq.masked_fill(~keep, -1e9), dim=1) * keep
    a = a / a.sum(1, keepdim=True).clamp(min=1e-9)
    ht_sel = torch.gather(ht, 1, topi[:, :, None].expand(-1, -1, ht.shape[2]))
    return (a[:, :, None] * ht_sel).sum(1)


@torch.no_grad()
def export_pass(enc, head, docs, bs, device, wq, bq, m_tT, s_tT):
    n = len(docs)
    raw = np.empty((n, 768), np.float32)
    mu = np.empty((n, 768), np.float32)
    st = np.empty((n, 768), np.float32)
    # 批序复刻 _batches(docs, bs, False)：按长度稳定排序后切块
    order = np.argsort([len(d.ids) for d in docs], kind="stable")
    chunks = [order[i:i + bs] for i in range(0, n, bs)]
    for ch in chunks:
        batch = [docs[i] for i in ch]
        ids, mask = collate(batch, device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            h = enc(ids, mask)
        hf = h.float()
        nv = mask.sum(1, keepdim=True).clamp(min=1)
        raw_b = hf.sum(1) / nv
        u = head.ln(hf)
        ht = (head.W2(F.gelu(head.W1(u))) + u) * mask.unsqueeze(-1)
        mu_b = ht.sum(1) / nv
        s_b = stop_forward(ht, mask, wq, bq, m_tT, s_tT)
        raw[ch] = raw_b.cpu().numpy()
        mu[ch] = mu_b.cpu().numpy()
        st[ch] = s_b.cpu().numpy()
    return raw, mu, st


# --------------------------------------------------------------------------- #
# 读出工具
# --------------------------------------------------------------------------- #
def fit_scaler(X):
    m = X.mean(0).astype(np.float64)
    s = np.maximum(X.std(0).astype(np.float64), STD_FLOOR)
    return m, s


def apply_scaler(X, m, s):
    return ((X - m) / s).astype(np.float32)


def fit_family(X_fit, y_fit, X_val, y_val, log):
    """多项逻辑回归：返回逐 C 结果 + C*（val BalAcc 最大、并列取小 C）。"""
    rows = []
    for C in C_GRID:
        t0 = time.perf_counter()
        with warnings.catch_warnings(record=True) as wl:
            warnings.simplefilter("always")
            clf = LogisticRegression(max_iter=MAX_ITER, tol=TOL, C=C).fit(X_fit, y_fit)
        conv = not any(issubclass(w.category, ConvergenceWarning) for w in wl)
        n_iter = int(np.max(clf.n_iter_))
        ptr = clf.predict_proba(X_fit)
        pva = clf.predict_proba(X_val)
        try:
            tr_ce = float(log_loss(y_fit, ptr, labels=list(clf.classes_)))
            va_ce = float(log_loss(y_val, pva, labels=list(clf.classes_)))
        except Exception:
            tr_ce = va_ce = None
        bal = float(balanced_accuracy_score(y_val, clf.predict(X_val)))
        sec = time.perf_counter() - t0
        rows.append({"C": C, "val_bal": bal, "n_iter": n_iter, "converged": conv,
                     "train_ce": tr_ce, "val_ce": va_ce, "sec": round(sec, 1)})
        log.append(f"family C={C:<5} n_iter={n_iter:<5} converged={conv} "
                   f"val_bal={bal:.4f} tr_ce={tr_ce and round(tr_ce, 4)} "
                   f"va_ce={va_ce and round(va_ce, 4)} ({sec:.1f}s)")
    best = max(rows, key=lambda r: (r["val_bal"], -r["C"]))
    C_star = best["C"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf = LogisticRegression(max_iter=MAX_ITER, tol=TOL, C=C_star).fit(X_fit, y_fit)
    log.append(f"family C*={C_star} 重拟合完成")
    return rows, C_star, clf


def fit_detection(X_fit, y_fit, X_val, y_val, log):
    rows = []
    for C in C_GRID:
        t0 = time.perf_counter()
        with warnings.catch_warnings(record=True) as wl:
            warnings.simplefilter("always")
            clf = LogisticRegression(max_iter=MAX_ITER, tol=TOL, C=C).fit(X_fit, y_fit)
        conv = not any(issubclass(w.category, ConvergenceWarning) for w in wl)
        n_iter = int(np.max(clf.n_iter_))
        ptr = clf.predict_proba(X_fit)[:, 1]
        pva = clf.predict_proba(X_val)[:, 1]
        try:
            tr_ce = float(log_loss(y_fit, ptr))
            va_ce = float(log_loss(y_val, pva))
        except Exception:
            tr_ce = va_ce = None
        au = float(roc_auc_score(y_val, pva))
        sec = time.perf_counter() - t0
        rows.append({"C": C, "val_auroc": au, "n_iter": n_iter, "converged": conv,
                     "train_ce": tr_ce, "val_ce": va_ce, "sec": round(sec, 1)})
        log.append(f"det    C={C:<5} n_iter={n_iter:<5} converged={conv} "
                   f"val_auroc={au:.4f} ({sec:.1f}s)")
    best = max(rows, key=lambda r: (r["val_auroc"], -r["C"]))
    C_star = best["C"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf = LogisticRegression(max_iter=MAX_ITER, tol=TOL, C=C_star).fit(X_fit, y_fit)
    log.append(f"det    C*={C_star} 重拟合完成")
    return rows, C_star, clf


def fam_eval(proba, y_true):
    pred = proba.argmax(1)
    rec = {}
    for c, lab in enumerate(SEEN_LABS):
        m = y_true == c
        if int(m.sum()) > 0:
            rec[lab] = round(float((pred[m] == c).mean()), 4)
    return {"acc": round(float((pred == y_true).mean()), 4),
            "bal": round(float(balanced_accuracy_score(y_true, pred)), 4),
            "n": int(len(y_true)), "recall": rec}


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    args = default_args(smoke=a.smoke)
    OUT = ROOT / "runs" / ("flagship_e28_smoke" if a.smoke else "flagship_e28")
    if OUT.exists() and any(OUT.iterdir()) and not a.force:
        print(f"[e28] 已存在产物，默认拒绝覆盖：{OUT}（--force 可覆盖）", flush=True)
        return 2
    OUT.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log_lines: list[str] = []
    solver_log: list[str] = []

    def say(msg: str):
        print(msg, flush=True)
        log_lines.append(msg)

    import sklearn
    commit = subprocess.run(
        ["git", "-C", str(ROOT.parent), "rev-parse", "HEAD"],
        capture_output=True, text=True).stdout.strip()
    say(f"[e28] commit {commit} | smoke={a.smoke} | device={device} | "
        f"sklearn={sklearn.__version__}")

    # ---- 1. 加载 E27 冻结件（不训练） ----
    ms_path = R27 / "model_state.pt"
    hp_path = R1 / "head.pt"
    enc_path = ROOT / "runs/v0.4.1_covreg/last.pt"
    sha = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
           for p in (ms_path, hp_path, enc_path)}
    ms = torch.load(ms_path, map_location="cpu", weights_only=False)
    say(f"[e28] E27 checkpoint selected={ms['selected']}（须为 {{'M':1,'T':1}}，"
        f"冒烟曾为 M@1/T@0）| sha256(model_state)={sha['model_state.pt'][:16]}")
    assert ms["selected"] == {"M": 1, "T": 1}, "E27 checkpoint 选择不符（疑似被冒烟覆盖）"
    wq = torch.as_tensor(ms["wq"], dtype=torch.float32, device=device)
    bq = torch.as_tensor(ms["bq"], dtype=torch.float32, device=device)
    m_tT = torch.as_tensor(ms["m_t"], dtype=torch.float32, device=device)
    s_tT = torch.as_tensor(ms["s_t"], dtype=torch.float32, device=device)
    m_mu, s_mu = np.asarray(ms["m_mu"]), np.asarray(ms["s_mu"])
    m_st, s_st = np.asarray(ms["m_st"]), np.asarray(ms["s_st"])
    armM = {k: np.asarray(v) for k, v in ms["armM"].items()}
    armT = {k: np.asarray(v) for k, v in ms["armT"].items()}

    with open(ROOT / "configs/ddet_v041.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    enc_cfg = dict(cfg["encoder"])
    enc_cfg["path"] = str(ROOT / enc_cfg["path"])
    dual = build_model("dual", encoder=build_encoder(**enc_cfg),
                       dim=768, pool=cfg["model"].get("pool", "mean"),
                       s2_rank=cfg["model"].get("s2_rank", 1))
    ck = torch.load(str(enc_path), map_location="cpu", weights_only=False)
    dual.load_state_dict(ck["state"], strict=False)
    enc = dual.encoder.to(device).eval()
    for p in enc.parameters():
        p.requires_grad = False
    head = SetPool(768, 512).to(device)
    head.load_state_dict(torch.load(hp_path, map_location=device)["head"])
    head.eval()
    for p in head.parameters():
        p.requires_grad = False

    # ---- 2. 只读 train/val 语料 ----
    corpus = load_train_val(args)
    say("[e28] 语料（仅 b_train/b_val，留出家族已过滤）："
        + " | ".join(f"{k}:{len(v)}" for k, v in corpus.items()))
    docs_tr, docs_va = corpus["train"], corpus["val"]
    counts = {}
    for k, v in corpus.items():
        fam_cnt = {}
        for d in v:
            fam_cnt[d.fam] = fam_cnt.get(d.fam, 0) + 1
        counts[k] = {"n": len(v), "family": fam_cnt,
                     "n_generators": len({d.gen for d in v})}
    if not a.smoke:
        assert len(docs_tr) == 17600 and len(docs_va) == 5372, \
            f"样本数与 E27 不符：{len(docs_tr)}/{len(docs_va)}"
        say(f"[e28] 样本清单与 E27 一致（17600/5372）✓")

    # ---- 3. 导出四表示（单次前向；GPU 预算独立记录） ----
    t0 = time.time()
    raw_tr, mu_tr, s_tr = export_pass(enc, head, docs_tr, args.batch, device,
                                      wq, bq, m_tT, s_tT)
    raw_va, mu_va, s_va = export_pass(enc, head, docs_va, args.batch, device,
                                      wq, bq, m_tT, s_tT)
    export_min = (time.time() - t0) / 60
    say(f"[e28] 导出完成 {export_min:.1f} min | raw/μ/s 形状 "
        f"{raw_tr.shape}/{mu_tr.shape}/{s_tr.shape}")

    # ---- 4. 复现检查（E27 标准量 + 原头） ----
    m_mu_diff = float(np.abs(mu_tr.mean(0) - m_mu).max())
    s_mu_re = np.maximum(mu_tr.std(0), STD_FLOOR)
    s_mu_diff = float(np.abs(s_mu_re - s_mu).max())
    say(f"[e28] μ 统计重算比：|Δm_mu|max={m_mu_diff:.2e} |Δs_mu|max={s_mu_diff:.2e}")
    zM_va = ((mu_va - m_mu) / s_mu).astype(np.float32)
    zT_va = np.concatenate([zM_va, ((s_va - m_st) / s_st).astype(np.float32)], 1)
    y_ai_va = np.array([d.y_ai for d in docs_va])
    ai_idx = np.array([i for i, d in enumerate(docs_va)
                       if d.y_ai == 1 and d.fam in SEEN_LABS])
    fam_true = np.array([SEEN_LABS.index(docs_va[i].fam) for i in ai_idx])

    def head_out(z, head_d):
        sd = z @ head_d["wD"][0] + head_d["bD"][0]
        af = z @ head_d["wF"].T + head_d["bF"]
        return sd, af

    repro = {}
    for tag, z, hd in (("M", zM_va, armM), ("T", zT_va, armT)):
        sd, af = head_out(z, hd)
        au = float(roc_auc_score(y_ai_va, sd))
        bal = float(balanced_accuracy_score(fam_true, af[ai_idx].argmax(1)))
        repro[tag] = {"val_det_auroc": round(au, 4), "val_fam_bal": round(bal, 4)}
        say(f"[e28] 复现 E27·{tag}：val det {au:.4f} | fam bal {bal:.4f}")
    if not a.smoke:
        e27 = json.loads((R27 / "metrics.json").read_text())
        for tag in ("M", "T"):
            ref = e27["arm_val"][tag]
            d1 = abs(repro[tag]["val_det_auroc"] - ref["det"]["auroc"])
            d2 = abs(repro[tag]["val_fam_bal"] - ref["fam"]["bal"])
            say(f"[e28] 复现差·{tag}：|Δdet|={d1:.4f} |Δfam bal|={d2:.4f}"
                f"（E27 报告 {ref['det']['auroc']}/{ref['fam']['bal']}）")
            repro[tag]["e27_ref"] = {"det": ref["det"]["auroc"],
                                     "fam_bal": ref["fam"]["bal"]}
            repro[tag]["diff"] = {"det": d1, "fam_bal": d2}

    # ---- 5. 四臂读出 ----
    arms_X = {"R0_mraw": (raw_tr, raw_va), "R1_mu": (mu_tr, mu_va),
              "R2_s": (s_tr, s_va),
              "R3_mus": (np.concatenate([mu_tr, s_tr], 1),
                         np.concatenate([mu_va, s_va], 1))}
    y_tr_ai = np.array([d.y_ai for d in docs_tr])
    ai_tr = np.array([d.y_ai == 1 and d.fam in SEEN_LABS for d in docs_tr])
    fam_tr = np.array([SEEN_LABS.index(d.fam) for d in docs_tr if
                       (d.y_ai == 1 and d.fam in SEEN_LABS)])
    y_va = np.array([d.y_ai for d in docs_va])
    fam_va = np.array([SEEN_LABS.index(docs_va[i].fam) for i in ai_idx])

    t_solver = time.time()
    results: dict = {}
    preds: dict = {}
    scalers: dict = {}
    for name, (Xtr, Xva) in arms_X.items():
        # 家族：训练 AI 行
        m_f, s_f = fit_scaler(Xtr[ai_tr])
        Ztr_f = apply_scaler(Xtr[ai_tr], m_f, s_f)
        Zva_f = apply_scaler(Xva[ai_idx], m_f, s_f)
        solver_log.append(f"== {name} family ==")
        fam_grid, C_F, clf_F = fit_family(Ztr_f, fam_tr, Zva_f, fam_va, solver_log)
        pf_va = clf_F.predict_proba(Zva_f)
        # 检测：全训练行
        m_d, s_d = fit_scaler(Xtr)
        Ztr_d = apply_scaler(Xtr, m_d, s_d)
        Zva_d = apply_scaler(Xva, m_d, s_d)
        solver_log.append(f"== {name} detection ==")
        det_grid, C_D, clf_D = fit_detection(Ztr_d, y_tr_ai, Zva_d, y_va, solver_log)
        pd_va = clf_D.predict_proba(Zva_d)[:, 1]
        results[name] = {
            "family": {"grid": fam_grid, "C_star": C_F, "val": fam_eval(pf_va, fam_va)},
            "detection": {"grid": det_grid, "C_star": C_D,
                          "val": {"auroc": round(float(roc_auc_score(y_va, pd_va)), 4),
                                  "f1@.5": round(float(f1_score(y_va, pd_va > 0.5,
                                                                average="macro")), 4)}},
            "converged_all": all(r["converged"] for r in fam_grid + det_grid)}
        scalers[name] = {"family": (m_f, s_f), "detection": (m_d, s_d)}
        preds[f"{name}_fam"] = pf_va.astype(np.float32)
        preds[f"{name}_det"] = pd_va.astype(np.float32)
        say(f"[e28] {name}：C_F*={C_F} val bal {results[name]['family']['val']['bal']}"
            f" | C_D*={C_D} val AUROC {results[name]['detection']['val']['auroc']}"
            f" | 全部收敛={results[name]['converged_all']}")

    # μ 列逐位一致性（R1 vs R3）
    ident = {}
    for task in ("family", "detection"):
        m1, s1 = scalers["R1_mu"][task]
        m3, s3 = scalers["R3_mus"][task]
        ident[task] = {"maxdiff_mean": float(np.abs(m1 - m3[:768]).max()),
                       "maxdiff_std": float(np.abs(s1 - s3[:768]).max())}
    say(f"[e28] R1/R3 μ 列标准化一致性：{json.dumps(ident)}")

    # ---- 6. dup-mean 等价检查（家族与检测；同 C，不搜索） ----
    dup = {}
    # family：R1 家族标量（AI 行）后的 μ 列
    Zdup_tr_f = apply_scaler(mu_tr[ai_tr], *scalers["R1_mu"]["family"])
    zva_f = apply_scaler(mu_va[ai_idx], *scalers["R1_mu"]["family"])
    C_ref_f = results["R1_mu"]["family"]["C_star"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf = LogisticRegression(max_iter=MAX_ITER, tol=TOL, C=C_ref_f).fit(
            np.concatenate([Zdup_tr_f / np.sqrt(2)] * 2, 1), fam_tr)
    p_dup = clf.predict_proba(np.concatenate([zva_f / np.sqrt(2)] * 2, 1))
    dup["family"] = {"C": C_ref_f,
                     "max_prob_diff": round(float(np.abs(
                         p_dup - preds["R1_mu_fam"]).max()), 6),
                     "bal_dup": round(float(balanced_accuracy_score(
                         fam_va, p_dup.argmax(1))), 4),
                     "bal_ref": results["R1_mu"]["family"]["val"]["bal"]}
    # detection：R1 检测标量（全行）后的 μ 列
    Zdup_tr_d = apply_scaler(mu_tr, *scalers["R1_mu"]["detection"])
    zva_d = apply_scaler(mu_va, *scalers["R1_mu"]["detection"])
    C_ref_d = results["R1_mu"]["detection"]["C_star"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf = LogisticRegression(max_iter=MAX_ITER, tol=TOL, C=C_ref_d).fit(
            np.concatenate([Zdup_tr_d / np.sqrt(2)] * 2, 1), y_tr_ai)
    p_dup = clf.predict_proba(np.concatenate([zva_d / np.sqrt(2)] * 2, 1))[:, 1]
    dup["detection"] = {"C": C_ref_d,
                        "max_prob_diff": round(float(np.abs(
                            p_dup - preds["R1_mu_det"]).max()), 6),
                        "auroc_dup": round(float(roc_auc_score(y_va, p_dup)), 4),
                        "auroc_ref": results["R1_mu"]["detection"]["val"]["auroc"]}
    say(f"[e28] dup-mean 等价检查：{json.dumps(dup, ensure_ascii=False)}")
    solver_min = (time.time() - t_solver) / 60
    say(f"[e28] 求解耗时 {solver_min:.1f} min（CPU；导出 {export_min:.1f} min GPU）")

    # ---- 7. 几何诊断（基于训练 AI） ----
    a_vec = mu_tr[ai_tr].mean(0)
    b_vec = s_tr[ai_tr].mean(0)

    def cos_eta(muX, sX):
        dm = muX - a_vec
        ds = sX - b_vec
        num = (dm * ds).sum(1)
        den = np.linalg.norm(dm, axis=1) * np.linalg.norm(ds, axis=1)
        ok = den > 0
        cos = float((num[ok] / den[ok]).mean()) if ok.any() else None
        eta = float((np.linalg.norm(ds - dm, axis=1) ** 2).sum()
                    / max((np.linalg.norm(ds, axis=1) ** 2).sum(), 1e-12))
        return {"cos_centered_mean": round(cos, 4) if cos is not None else None,
                "eta_delta": round(eta, 4), "n": int(len(muX)), "n_ok": int(ok.sum())}

    geometry = {"train_ai": cos_eta(mu_tr[ai_tr], s_tr[ai_tr]),
                "val_ai": cos_eta(mu_va[ai_idx], s_va[ai_idx]),
                "note": "cos 为逐样本中心化余弦均值 (17)；η_Δ 为 (18)；仅描述性，不用于筛选维度"}

    # ---- 8. 分桶（val；家族=AI 行、检测=全行） ----
    langs = np.array([d.lang for d in docs_va])
    ntoks = np.array([len(d.ids) for d in docs_va])
    fam_langs = langs[ai_idx]
    fam_ntoks = ntoks[ai_idx]
    buckets = {"family_by_language": {}, "family_by_length": {},
               "det_by_language": {}, "det_by_length": {}}
    for lang in sorted({l for l in set(langs.tolist()) if (langs == l).sum() >= 30}):
        det_m = langs == lang
        fam_m = fam_langs == lang
        if fam_m.sum() >= 20 and len(set(fam_va[fam_m].tolist())) >= 2:
            for name in arms_X:
                pr = preds[f"{name}_fam"][fam_m].argmax(1)
                buckets["family_by_language"].setdefault(lang, {})[name] = {
                    "bal": round(float(balanced_accuracy_score(fam_va[fam_m], pr)), 4),
                    "n": int(fam_m.sum())}
        if det_m.sum() >= 20 and len(set(y_va[det_m].tolist())) == 2:
            for name in arms_X:
                buckets["det_by_language"].setdefault(lang, {})[name] = {
                    "auroc": round(float(roc_auc_score(y_va[det_m],
                                                       preds[f"{name}_det"][det_m])), 4),
                    "n": int(det_m.sum())}
    for lo, hi in LEN_EDGES:
        det_m = (ntoks >= lo) & (ntoks <= hi)
        fam_m = (fam_ntoks >= lo) & (fam_ntoks <= hi)
        if fam_m.sum() >= 20 and len(set(fam_va[fam_m].tolist())) >= 2:
            for name in arms_X:
                pr = preds[f"{name}_fam"][fam_m].argmax(1)
                buckets["family_by_length"].setdefault(f"{lo}-{hi}", {})[name] = {
                    "bal": round(float(balanced_accuracy_score(fam_va[fam_m], pr)), 4),
                    "n": int(fam_m.sum())}
        if det_m.sum() >= 20 and len(set(y_va[det_m].tolist())) == 2:
            for name in arms_X:
                buckets["det_by_length"].setdefault(f"{lo}-{hi}", {})[name] = {
                    "auroc": round(float(roc_auc_score(y_va[det_m],
                                                       preds[f"{name}_det"][det_m])), 4),
                    "n": int(det_m.sum())}

    # ---- 9. 家族内配对 bootstrap（Δ_F = bal(R3) - bal(R1)） ----
    rng = np.random.RandomState(SEED)
    fam_idx = {c: np.where(fam_va == c)[0] for c in range(8)}
    deltas = np.empty(N_BOOT, np.float64)
    pred1 = preds["R1_mu_fam"].argmax(1)
    pred3 = preds["R3_mus_fam"].argmax(1)
    for b in range(N_BOOT):
        sel = np.concatenate([rng.choice(fam_idx[c], size=len(fam_idx[c]),
                                         replace=True) for c in range(8)])
        deltas[b] = (balanced_accuracy_score(fam_va[sel], pred3[sel])
                     - balanced_accuracy_score(fam_va[sel], pred1[sel]))
    d_point = (results["R3_mus"]["family"]["val"]["bal"]
               - results["R1_mu"]["family"]["val"]["bal"])
    boot = {"delta_F_point": round(d_point, 4),
            "ci95": [round(float(np.percentile(deltas, 2.5)), 4),
                     round(float(np.percentile(deltas, 97.5)), 4)],
            "boot_mean": round(float(deltas.mean()), 4),
            "n_boot": N_BOOT, "seed": SEED,
            "note": "C 与 epoch 均使用过该验证集 → 该区间不校正选择偏差，仅样本重采样不确定性"}
    say(f"[e28] Δ_F = {boot['delta_F_point']}，95% CI {boot['ci95']}")

    # ---- 10. 产物 ----
    np.savez_compressed(
        OUT / "features_train_val.npz",
        raw_train=raw_tr, raw_val=raw_va, mu_train=mu_tr, mu_val=mu_va,
        s_train=s_tr, s_val=s_va,
        n_tokens_train=np.array([len(d.ids) for d in docs_tr], np.int32),
        n_tokens_val=ntoks.astype(np.int32),
        rows_train=np.array([d.row for d in docs_tr], np.int64),
        rows_val=np.array([d.row for d in docs_va], np.int64),
        md5_train=np.array([ids_md5(d.ids) for d in docs_tr]),
        md5_val=np.array([ids_md5(d.ids) for d in docs_va]),
        fam_idx_train=np.array([SEEN_LABS.index(d.fam) if d.fam in SEEN_LABS
                                else -1 for d in docs_tr], np.int8),
        fam_idx_val=np.array([SEEN_LABS.index(d.fam) if d.fam in SEEN_LABS
                              else -1 for d in docs_va], np.int8),
        y_ai_train=y_tr_ai.astype(np.int8), y_ai_val=y_va.astype(np.int8),
        lang_train=np.array([d.lang for d in docs_tr]),
        lang_val=np.array([d.lang for d in docs_va]))
    np.savez_compressed(OUT / "predictions_val.npz",
                        ai_idx=ai_idx.astype(np.int64), **preds)
    h = hashlib.sha256()
    for sp, docs in (("train", docs_tr), ("val", docs_va)):
        for d in docs:
            h.update(f"{sp}:{d.row}:{ids_md5(d.ids)}\n".encode())
    manifest = {"commit": commit, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                "checkpoint_sha256": sha, "e27_selected": ms["selected"],
                "input_manifest_sha256": h.hexdigest(),
                "counts": counts, "dims": {"raw": 768, "mu": 768, "s": 768,
                                           "R3": 1536},
                "sklearn": sklearn.__version__, "test_accessed": False,
                "unseen_accessed": False,
                "notes": "只读 b_train/b_val；留出家族过滤；主键=(split,row)，md5 仅校验"}
    metrics = {"commit": commit, "smoke": a.smoke, "test_accessed": False,
               "sklearn": sklearn.__version__, "seed": SEED, "C_GRID": C_GRID,
               "budget": {"export_min": round(export_min, 1),
                          "solver_min": round(solver_min, 1)},
               "repro": repro, "mu_col_identity": ident, "arms": results,
               "dup_mean_check": dup, "bootstrap_delta_F": boot,
               "buckets": buckets,
               "solver_converged_all": all(v["converged_all"] for v in results.values())}
    (OUT / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=1))
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    (OUT / "geometry.json").write_text(json.dumps(geometry, ensure_ascii=False, indent=1))
    (OUT / "solver.log").write_text("\n".join(solver_log) + "\n")
    (OUT / "config.json").write_text(json.dumps({
        "args": vars(args), "C_GRID": C_GRID, "STD_FLOOR": STD_FLOOR,
        "solver": {"lbfgs": True, "tol": TOL, "max_iter": MAX_ITER,
                   "fit_intercept": True, "class_weight": None,
                   "family": "multinomial (默认)", "detection": "binomial"},
        "arms": {"R0": "m_raw", "R1": "mu(E24 ht 均值)", "R2": "s(冻结查询输出)",
                 "R3": "[mu;s]"},
        "rng": "RandomState(0) 复刻 load_corpus_tagged 的 subset 行为（仅 train/val）",
        "commit": commit}, ensure_ascii=False, indent=1))
    say(f"[e28] 完成 {time.time()-t_start:.1f}s；产物 {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
