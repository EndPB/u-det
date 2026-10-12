#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""d-det-v2 v0.1 · 统一评估

- 主表：F = K+1 Macro-F1 / BA / 每类 P-R-F1 / 混淆 / AI-only Macro-F1
        （AI-only 只对 AI 真值行计算；Human 预测作为误分类计入对应家族 FN）；
        D = pooled AUROC / Macro-F1 / BA / Human FPR / AI TPR（阈值仅由训练内校准子集冻结）；
        附 task-macro AUROC 与 Ours-F 派生检测（先验校正，附加读数）；
- 长度分桶（≤512 / 513-1024 / 1025-2048 / >2048）与 D 的长度匹配压力子集；
- AMMC 压力：comment / ws / rename / combo（同一 parent 配对；主臂 B1 与 Ours 系，C-P0 不参与）；
- 配对 bootstrap CI（按 task 聚类重采样）：Ours vs B0/B1/B2/B3/C-P0。

读数口径：dev 为已有暴露开发评估；test 仅在显式 --test 时读取（并记录）。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data as D          # noqa: E402
import model as M         # noqa: E402
import center as C        # noqa: E402
import train as TR        # noqa: E402

T0 = time.time()
STRESS = ["comment", "ws", "rename", "combo"]
CI_OTHERS = ["B0", "B1", "b2", "b3", "C-P0"]


def det_seed(*parts) -> int:
    return zlib.crc32("|".join(map(str, parts)).encode("utf-8")) % (2 ** 31)


# --------------------------------------------------------------------------- #
# 前向与输入准备
# --------------------------------------------------------------------------- #

def fwd_encoder(enc, tok, codes, chunk_len, max_chunks, batch=12):
    enc.eval()
    outs = []
    for s in range(0, len(codes), batch):
        sub = codes[s:s + batch]
        items = []
        for code in sub:
            chunks = D.encode_chunks(tok, code, chunk_len, max_chunks)
            Cn = len(chunks)
            ids = torch.zeros(Cn, chunk_len, dtype=torch.long)
            mask = torch.zeros(Cn, chunk_len, dtype=torch.long)
            for c, x in enumerate(chunks):
                n = x.shape[0]
                ids[c, :n] = x
                mask[c, :n] = 1
            items.append((ids, mask))
        Cmax = max(x[0].shape[0] for x in items)
        ids = torch.zeros(len(items), Cmax, chunk_len, dtype=torch.long)
        mask = torch.zeros(len(items), Cmax, chunk_len, dtype=torch.long)
        for k, (i_, m_) in enumerate(items):
            ids[k, :i_.shape[0]] = i_
            mask[k, :m_.shape[0]] = m_
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            h = enc(ids.cuda(), mask.cuda())
        outs.append(h.float().cpu())
    return torch.cat(outs, 0)


def base_logits(model, h):
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        return model.head(h.cuda()).float().cpu()


def load_stage1(wp, cfg, task, arm, seed):
    ck = wp["ckpt"] / f"{task}_{arm}_s{seed}" / "best.pt"
    if not ck.exists():
        return None
    st = torch.load(ck, map_location="cpu")
    m = M.Stage1Model(str(D.resolve(cfg["paths"]["encoder"])), 8 if task == "F" else 1).cuda()
    m.encoder.load_state_dict(st["encoder"])
    m.head.load_state_dict(st["head"])
    m.eval()
    return m


def load_head_state(wp, task, arm, seed):
    ck = wp["ckpt"] / f"{task}_{arm}_s{seed}" / "best.pt"
    return torch.load(ck, map_location="cpu") if ck.exists() else None


def load_refs(fdir):
    z = np.load(fdir / "refs.npz")
    folds = sorted(int(k[1:]) for k in z.files if k.startswith("W"))
    refs = {}
    for j in folds:
        r = C.RidgeRef(alphas=[1.0])
        r.W, r.mu, r.sd, r.ym = z[f"W{j}"], z[f"mu{j}"], z[f"sd{j}"], z[f"ym{j}"]
        refs[j] = r
    return refs


def probs_of(task, logits):
    l = torch.as_tensor(logits).float().cpu()
    if task == "F":
        return torch.softmax(l, -1).numpy()
    return torch.sigmoid(l.squeeze(-1)).numpy()


# --------------------------------------------------------------------------- #
# 指标
# --------------------------------------------------------------------------- #

def metrics_F(y, prob):
    pred = prob.argmax(-1)
    K = len(D.CLASSES_F)
    out = {"n": int(len(y)),
           "macro_f1": float(f1_score(y, pred, labels=list(range(K)), average="macro", zero_division=0)),
           "ba": float(np.mean([np.mean(pred[y == c] == c) if np.any(y == c) else np.nan for c in range(K)]))}
    ai = y > 0
    out["ai_only_macro_f1"] = float(f1_score(y[ai], pred[ai], labels=list(range(1, K)), average="macro", zero_division=0))
    rep = classification_report(y, pred, labels=list(range(K)), target_names=D.CLASSES_F,
                                output_dict=True, zero_division=0)
    out["per_class"] = {c: {k: float(rep[c][k]) for k in ("precision", "recall", "f1-score", "support")}
                        for c in D.CLASSES_F}
    out["confusion"] = confusion_matrix(y, pred, labels=list(range(K))).tolist()
    return out


def metrics_D(y, prob, thr=0.5):
    pred = (prob >= thr).astype(int)
    auroc = float(roc_auc_score(y, prob)) if len(np.unique(y)) > 1 else float("nan")
    return {"n": int(len(y)), "auroc": auroc, "threshold": float(thr),
            "macro_f1": float(f1_score(y, pred, labels=[0, 1], average="macro", zero_division=0)),
            "ba": float(0.5 * (np.mean(pred[y == 1] == 1) + np.mean(pred[y == 0] == 0))),
            "human_fpr": float(np.mean(pred[y == 0] == 1)), "ai_tpr": float(np.mean(pred[y == 1] == 1))}


def pick_threshold(y, prob):
    if len(np.unique(y)) < 2:
        return 0.5
    cand = np.unique(np.round(prob, 6))
    best, thr = -1.0, 0.5
    for t in cand:
        ba = 0.5 * (np.mean(prob[y == 1] >= t) + np.mean(prob[y == 0] < t))
        if ba > best:
            best, thr = ba, float(t)
    return thr


def metric_value(task, y, prob, thr=0.5):
    if task == "F":
        return float(f1_score(y, prob.argmax(-1), labels=list(range(len(D.CLASSES_F))), average="macro", zero_division=0))
    return float(roc_auc_score(y, prob)) if len(np.unique(y)) > 1 else float("nan")


def task_macro_auroc(y, prob, groups):
    vals = []
    for t in np.unique(groups):
        m = groups == t
        if len(np.unique(y[m])) > 1:
            vals.append(roc_auc_score(y[m], prob[m]))
    return float(np.mean(vals)) if vals else float("nan")


def boot_ci(task, y, pa, pb, groups, n_boot, seed):
    """配对 bootstrap（按 task 聚类重采样）：metric(pa) − metric(pb) 的 95% CI。"""
    uniq = np.unique(groups)
    idx_by = {t: np.where(groups == t)[0] for t in uniq}
    point = metric_value(task, y, pa) - metric_value(task, y, pb)
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n_boot):
        pick = rng.choice(len(uniq), size=len(uniq), replace=True)
        idx = np.concatenate([idx_by[uniq[j]] for j in pick])
        yy = y[idx]
        if task == "D" and len(np.unique(yy)) < 2:
            continue
        m0 = metric_value(task, yy, pa[idx])
        m1 = metric_value(task, yy, pb[idx])
        if np.isfinite(m0) and np.isfinite(m1):
            deltas.append(m0 - m1)
    if len(deltas) < int(0.5 * n_boot):
        return {"point": point, "lo": None, "hi": None, "valid": len(deltas), "note": "有效重采样不足"}
    d = np.asarray(deltas)
    return {"point": point, "lo": float(np.percentile(d, 2.5)),
            "hi": float(np.percentile(d, 97.5)), "valid": int(len(d))}


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description="d-det-v2 v0.1 统一评估")
    ap.add_argument("--task", choices=["F", "D"], required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--work-root", default=None)
    ap.add_argument("--splits", default="dev", help="逗号分隔：dev[,calib][,test]")
    ap.add_argument("--test", action="store_true", help="显式授权读取 test")
    ap.add_argument("--no-stress", action="store_true")
    ap.add_argument("--n-boot", type=int, default=None)
    args = ap.parse_args()

    cfg = D.load_cfg()
    wp = TR.work_paths(cfg, args.work_root)
    recs = D.load_records_file(wp["records"])
    fdir = wp["feat"] / f"{args.task}_s{args.seed}"
    assert fdir.exists(), f"缺少特征缓存 {fdir}（先运行 feat）"
    n_boot = args.n_boot or cfg["eval"]["bootstrap"]

    want = [s for s in args.splits.split(",") if s]
    if "test" in want and not args.test:
        raise SystemExit("读取 test 需要显式 --test")
    sets = {}
    for s in set(want) | {"calib"}:  # 校准子集始终计算（阈值/参考）
        sets[s] = D.subset_index(recs, "train", calib=True) if s == "calib" else D.subset_index(recs, s)
    y_of = {k: TR.class_ids(recs, v, args.task).numpy() for k, v in sets.items()}
    groups_of = {k: np.array([recs[i]["tid"] for i in v]) for k, v in sets.items()}

    tok = D.load_tokenizer(cfg)
    # 缓存读取（缺则现场计算）
    h_cache, base_cache, m5_cache = {}, {}, {}
    for k in sets:
        hp, lp, mp = fdir / f"h_{k}.pt", fdir / f"logits_{k}.pt", fdir / f"m5_{k}.pt"
        h_cache[k] = torch.load(hp, map_location="cpu").float() if hp.exists() else None
        base_cache[k] = torch.load(lp, map_location="cpu") if lp.exists() else None
        m5_cache[k] = torch.load(mp, map_location="cpu").float() if mp.exists() else None
    m_global = torch.load(fdir / "mglobal.pt", map_location="cpu")
    refs = load_refs(fdir)

    b1 = load_stage1(wp, cfg, args.task, "B1", args.seed)
    assert b1 is not None, "缺少 B1 checkpoint"
    sem = M.EncoderPool(str(D.resolve(cfg["paths"]["encoder"]))).cuda().eval()
    M.freeze_(sem)

    def inputs_for(k, codes=None):
        """返回 (h, base, m5)；codes 给了就现场算（压力视图），否则尽量用缓存。"""
        if codes is not None:
            h = fwd_encoder(b1.encoder, tok, codes, cfg["model"]["chunk_len"], cfg["model"]["max_chunks"])
            base = base_logits(b1, h)
            q = [D.content_proxy(c, recs[i]["lang"])[0] for c, i in zip(codes, sets[k])]
            s_t = fwd_encoder(sem, tok, q, cfg["model"]["chunk_len"], cfg["model"]["max_chunks"])
            m5 = C.predict_refs(refs, s_t)
            return h, base, m5
        h = h_cache[k]
        if h is None:
            h = fwd_encoder(b1.encoder, tok, [recs[i]["code"] for i in sets[k]],
                            cfg["model"]["chunk_len"], cfg["model"]["max_chunks"])
        base = base_cache[k] if base_cache[k] is not None else base_logits(b1, h)
        m5 = m5_cache[k]
        if m5 is None:
            q = [D.content_proxy(recs[i]["code"], recs[i]["lang"])[0] for i in sets[k]]
            s_t = fwd_encoder(sem, tok, q, cfg["model"]["chunk_len"], cfg["model"]["max_chunks"])
            m5 = C.predict_refs(refs, s_t)
        return h, base, m5

    # ---- 各臂干净集预测 ----
    probs = {}
    for arm in ("B0", "B1"):
        model = b1 if arm == "B1" else load_stage1(wp, cfg, args.task, arm, args.seed)
        if model is None:
            D.log(f"[eval] 跳过 {arm}（无 checkpoint）")
            continue
        probs[arm] = {}
        for k in sets:
            h, base, _ = inputs_for(k)
            if arm == "B1":
                lg = base
            else:
                h = fwd_encoder(model.encoder, tok, [recs[i]["code"] for i in sets[k]],
                                cfg["model"]["chunk_len"], cfg["model"]["max_chunks"])
                lg = base_logits(model, h)
            probs[arm][k] = probs_of(args.task, lg)

    arms2 = TR.F_ARMS if args.task == "F" else TR.D_ARMS
    heads = {}
    for arm in arms2:
        st = load_head_state(wp, args.task, arm, args.seed)
        if st is None:
            continue
        head = TR.make_head(args.task, arm, st["in_dim"], cfg).cuda()
        head.load_state_dict(st["head"]); head.eval()
        heads[arm] = head
        probs[arm] = {}
        for k in sets:
            h, base, m5 = inputs_for(k)
            with torch.no_grad():
                lg = TR.infer_logits(args.task, arm, head, h, base, m5, m_global)
            probs[arm][k] = probs_of(args.task, lg)

    p0npz = wp["root"] / "p0" / f"{args.task}_s{args.seed}.npz"
    if p0npz.exists():
        z = np.load(p0npz)
        probs["C-P0"] = {}
        for k in sets:
            key = f"fused_{k}"
            if key in z.files:
                p = z[key]
                probs["C-P0"][k] = p if args.task == "F" else (p[:, 1] if p.ndim == 2 else p)
    else:
        D.log("[eval] 无 C-P0 产物（可先运行 p0.py）")

    # ---- 主表 ----
    tables = {}
    for arm, pk in probs.items():
        tables[arm] = {}
        if args.task == "F":
            for k in sets:
                tables[arm][k] = metrics_F(y_of[k], pk[k])
        else:
            thr = pick_threshold(y_of["calib"], pk["calib"]) if "calib" in pk else 0.5
            for k in sets:
                tables[arm][k] = metrics_D(y_of[k], pk[k], thr)
                tables[arm][k]["task_macro_auroc"] = task_macro_auroc(y_of[k], pk[k], groups_of[k])

    derived = {}
    if args.task == "F" and "ours" in probs:
        lp_F = np.log(np.full(8, 1.0 / 8))
        lp_D = np.log(np.array([0.5] + [1.0 / 14] * 7))
        for k in sets:
            p = np.clip(probs["ours"][k], 1e-12, 1)
            adj = np.log(p) + (lp_D - lp_F)
            p_ai = 1.0 - np.exp(adj - np.logaddexp.reduce(adj, axis=1, keepdims=True))[:, 0]
            yD = np.array([0 if recs[i]["kind"] == "human" else 1 for i in sets[k]])
            if len(np.unique(yD)) > 1:
                derived[k] = {"auroc": float(roc_auc_score(yD, p_ai)),
                              "note": "Ours-F 派生检测（先验校正；附加读数）"}

    # ---- 配对 CI ----
    ci = {}
    ref = "ours"
    if ref in probs:
        rep_splits = [k for k in want if k != "calib" and k in sets]
        for other in CI_OTHERS:
            if other not in probs:
                continue
            for k in rep_splits:
                key = f"{ref}_vs_{other}@{k}"
                ci[key] = boot_ci(args.task, y_of[k], probs[ref][k], probs[other][k],
                                  groups_of[k], n_boot, seed=det_seed(args.task, k, other))

    # ---- 长度分桶 ----
    lens = TR.compute_token_lens(cfg, wp, recs)
    edges = cfg["eval"]["buckets"]
    buckets = {}
    for k in sets:
        bk = {}
        for i, ridx in enumerate(sets[k]):
            bk.setdefault(D.BUCKET_NAMES[D.bucket_of(int(lens[ridx]), edges)], []).append(i)
        rows = {}
        for b, ii in bk.items():
            ii = np.array(ii)
            yb = y_of[k][ii]
            r = {"n": int(len(ii))}
            for arm, pk in probs.items():
                if args.task == "F":
                    r[arm] = metrics_F(yb, pk[k][ii])["macro_f1"]
                elif len(np.unique(yb)) > 1:
                    r[arm] = metrics_D(yb, pk[k][ii], tables[arm][k]["threshold"])["auroc"]
                else:
                    fpr = float(np.mean((pk[k][ii] >= tables[arm][k]["threshold"]).astype(int) == 1)) if (yb == 0).all() else None
                    r[arm] = None
                    r[f"{arm}_human_fpr"] = fpr
            rows[b] = r
        buckets[k] = rows
    # D 长度匹配压力子集
    length_matched = {}
    if args.task == "D" and "dev" in sets:
        idx = sets["dev"]
        lm = []
        rng = np.random.default_rng(det_seed("lm", args.seed))
        for b in D.BUCKET_NAMES:
            ii = np.array([i for i, ridx in enumerate(idx)
                           if D.BUCKET_NAMES[D.bucket_of(int(lens[ridx]), edges)] == b])
            if len(ii) == 0:
                continue
            hi = ii[y_of["dev"][ii] == 0]
            ai = ii[y_of["dev"][ii] == 1]
            n = min(len(hi), len(ai))
            if n > 0:
                lm.append(rng.choice(hi, n, replace=False))
                lm.append(rng.choice(ai, n, replace=False))
        if lm:
            lmi = np.concatenate(lm)
            for arm, pk in probs.items():
                if len(np.unique(y_of["dev"][lmi])) > 1:
                    length_matched[arm] = {"n": int(len(lmi)),
                                           "auroc": float(roc_auc_score(y_of["dev"][lmi], pk["dev"][lmi]))}

    # ---- 压力（comment / ws / rename / combo）----
    stress = {}
    if not args.no_stress and "dev" in sets:
        idx = sets["dev"]
        yb = y_of["dev"]
        stress_arms = [a for a in ("B1", "ours", "b2", "b3") if a in probs]
        for variant in STRESS:
            codes, applied = [], []
            for i in idx:
                r = recs[i]
                if variant == "combo":
                    c = r["code"]
                    for name in ("comment", "ws", "rename"):
                        c2 = D.apply_transform(c, r["lang"], name)
                        if c2 is not None:
                            c = c2
                    codes.append(c); applied.append(c != r["code"])
                else:
                    c2 = D.apply_transform(r["code"], r["lang"], variant)
                    codes.append(c2 if c2 is not None else r["code"])
                    applied.append(c2 is not None)
            applied = np.array(applied)
            entry = {"n": int(len(idx)), "applied_rate": float(applied.mean())}
            h, base, m5 = inputs_for("dev", codes=codes)
            for arm in stress_arms:
                if arm == "B1":
                    pv = probs_of(args.task, base)
                else:
                    with torch.no_grad():
                        lg = TR.infer_logits(args.task, arm, heads[arm], h, base, m5, m_global)
                    pv = probs_of(args.task, lg)
                if args.task == "F":
                    entry[arm] = metrics_F(yb, pv)["macro_f1"]
                else:
                    entry[arm] = metrics_D(yb, pv, tables[arm]["dev"]["threshold"])["auroc"]
            stress[variant] = entry

    # ---- 输出 ----
    meta = {"git": TR.git_head(), "n_boot": n_boot, "work_root": str(wp["root"]),
            "y_counts": {k: {str(c): int((y_of[k] == c).sum()) for c in np.unique(y_of[k])} for k in sets},
            "time_s": round(time.time() - T0, 1)}
    out = {"task": args.task, "seed": args.seed, "splits": want,
           "tables": tables, "derived_detection": derived, "buckets": buckets,
           "length_matched": length_matched, "stress": stress, "ci": ci, "meta": meta}
    if args.work_root is None:
        outdir = TR.art_paths(cfg)["reports"] / args.task / f"s{args.seed}"
    else:
        outdir = wp["root"] / "reports" / f"{args.task}_s{args.seed}"
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "eval.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    sc = wp["root"] / "scores"
    sc.mkdir(parents=True, exist_ok=True)
    np.savez(sc / f"{args.task}_s{args.seed}_preds.npz",
             **{f"{arm}.{k}": pk[k] for arm, pk in probs.items() for k in sets},
             **{f"idx.{k}": np.array(sets[k]) for k in sets},
             **{f"y.{k}": y_of[k] for k in sets})

    D.log(f"[eval {args.task}/s{args.seed}] 完成 → {outdir/'eval.json'}（{time.time()-T0:.0f}s）")
    for arm, td in tables.items():
        row = td.get("dev", {})
        keys = ["macro_f1", "ai_only_macro_f1"] if args.task == "F" else ["auroc", "macro_f1", "human_fpr", "ai_tpr"]
        D.log(f"  {arm}: " + ", ".join(f"{k}={row.get(k):.4f}" for k in keys if isinstance(row.get(k), float)))


if __name__ == "__main__":
    main()
