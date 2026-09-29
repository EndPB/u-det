#!/usr/bin/env python
"""E25 调查 + 离线统一终评（门控 FAIL ⇒ 阶段 1 跳过；纯 CPU，不训练）。

指导文档要求（§4/§5）：
  ① 阶段 0 未过 M0 ⇒ 先调查 family/generator 混杂与表示信号（不掩盖负结果）；
  ② B0–B3 的 test_seen/unseen 统一评分（含检测双阈值、家族逐族召回±二项不确定性、
     unseen 检测 vs val Human 并按原始 split 分解、拒识两方向、语言/长度桶、generator 级召回）。

输出：runs/flagship_e25/e25_offline_and_invest.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from models.disc import DiscHead  # noqa: E402
from flagship_round1 import residualize, s_d_scalar, SEEN_LABS  # noqa: E402
from flagship_e25_stage0 import load_corpus_tagged  # noqa: E402
from argparse import Namespace  # noqa: E402

OUT = ROOT / "runs/flagship_e25"
ARGS = Namespace(smoke=False, max_len=2048, train_human=8000, train_fam=1200,
                 val_human=1500, val_fam=500, test_human=500, test_fam=200,
                 unseen_cap=600, enc_bs=32)


def softmax(a):
    m = a.max(1, keepdims=True); e = np.exp(a - m)
    return e / e.sum(1, keepdims=True)


def main() -> int:
    t0 = time.time()
    feats = dict(np.load(OUT / "features.npz", allow_pickle=True))
    meta = json.loads((OUT / "meta.json").read_text())
    st = dict(np.load(OUT.parent / "flagship_r1/stats.npz", allow_pickle=True))
    import torch
    bD = float(torch.load(OUT.parent / "flagship_r1/head.pt",
                          map_location="cpu", weights_only=False)["bD"])

    corpus = load_corpus_tagged(ARGS)
    y_tr = np.array([d.y_ai for d in corpus["train"]])
    ai_tr = np.array([d.y_ai == 1 for d in corpus["train"]])
    fam_tr = np.array([SEEN_LABS.index(d.fam) for d in corpus["train"] if d.y_ai])
    y_val = np.array([d.y_ai for d in corpus["val"]])
    ai_val = np.array([d.y_ai == 1 and d.fam in SEEN_LABS for d in corpus["val"]])
    fam_val = np.array([SEEN_LABS.index(d.fam) for d, s in zip(corpus["val"], ai_val) if s])
    y_te = np.array([d.y_ai for d in corpus["test_seen"]])
    ai_te = np.array([d.y_ai == 1 and d.fam in SEEN_LABS for d in corpus["test_seen"]])
    fam_te = np.array([SEEN_LABS.index(d.fam) for d, s in zip(corpus["test_seen"], ai_te) if s])
    ai_idx_te = np.where(ai_te)[0]
    zh_val = ~y_val.astype(bool)

    z_tr = feats["z_train"]; z_v = feats["z_val"]; z_t = feats["z_test_seen"]; z_u = feats["z_unseen"]
    rm_tr = feats["rm_train"]; rm_v = feats["rm_val"]; rm_t = feats["rm_test_seen"]; rm_u = feats["rm_unseen"]
    rF_tr = residualize(z_tr, st); rF_v = residualize(z_v, st)
    rF_t = residualize(z_t, st); rF_u = residualize(z_u, st)

    report = {"note": "门控 FAIL → 阶段 1 跳过；本文件 = 调查 + B0–B3 离线终评（只评一次）",
              "timing_min": None}

    def det_metrics(prob_v, prob_t):
        ths = np.linspace(0.05, 0.95, 37)
        f1s = [f1_score(y_val, (prob_v > t).astype(int), average="macro") for t in ths]
        th = float(ths[int(np.argmax(f1s))])
        return {"val_auroc": round(float(roc_auc_score(y_val, prob_v)), 4),
                "te_auroc": round(float(roc_auc_score(y_te, prob_t)), 4),
                "te_f1_th05": round(float(f1_score(y_te, (prob_t > 0.5).astype(int),
                                                   average="macro")), 4),
                "val_best_th": round(th, 3),
                "te_f1_valth": round(float(f1_score(y_te, (prob_t > th).astype(int),
                                                    average="macro")), 4)}

    def fam_metrics(pv, pt, pu):
        pred = pt.argmax(1)
        rec = {}
        for c in range(len(SEEN_LABS)):
            m = fam_te == c
            if m.sum() == 0:
                continue
            p = float((pred[m] == c).mean())
            se = float(np.sqrt(max(p * (1 - p), 1e-9) / m.sum()))
            rec[SEEN_LABS[c]] = {"n": int(m.sum()), "recall": round(p, 3),
                                 "se": round(se, 3)}
        seen_p = pv.max(1); unseen_p = pu.max(1)
        lab = np.array([0] * len(seen_p) + [1] * len(unseen_p))
        scr = np.concatenate([seen_p, unseen_p])
        return {"acc": round(float((pred == fam_te).mean()), 4),
                "balanced_acc": round(float(balanced_accuracy_score(fam_te, pred)), 4),
                "per_family_recall": rec,
                "rejection": {"auc_maxp": round(float(roc_auc_score(lab, scr)), 4),
                              "auc_neg_maxp": round(float(roc_auc_score(lab, -scr)), 4)}}

    def unseen_det(pv, pu):
        out = {"overall_auroc": round(float(roc_auc_score(
            np.concatenate([np.zeros(int(zh_val.sum())), np.ones(len(pu))]),
            np.concatenate([pv[zh_val], pu]))), 4)}
        splits = np.asarray(meta["unseen"]["split"])
        for sp in ("train", "val", "test"):
            m = splits == sp
            if m.any():
                out[f"auroc_{sp}"] = round(float(roc_auc_score(
                    np.concatenate([np.zeros(int(zh_val.sum())), np.ones(int(m.sum()))]),
                    np.concatenate([pv[zh_val], pu[m]]))), 4)
                out[f"n_{sp}"] = int(m.sum())
        return out

    arms = {}
    def run_arm(name, Xtr, Xv, Xt, Xu):
        sc = StandardScaler().fit(Xtr)
        lr_d = LogisticRegression(max_iter=3000, C=1.0).fit(sc.transform(Xtr), y_tr)
        pv = lr_d.predict_proba(sc.transform(Xv))[:, 1]
        pt = lr_d.predict_proba(sc.transform(Xt))[:, 1]
        pu = lr_d.predict_proba(sc.transform(Xu))[:, 1]
        entry = {"det": det_metrics(pv, pt), "unseen_det": unseen_det(pv, pu)}
        scf = StandardScaler().fit(Xtr[ai_tr])
        lr_f = LogisticRegression(max_iter=3000, C=1.0).fit(scf.transform(Xtr[ai_tr]), fam_tr)
        fv = lr_f.predict_proba(scf.transform(Xv[ai_val]))
        ft = lr_f.predict_proba(scf.transform(Xt[ai_te]))
        fu = lr_f.predict_proba(scf.transform(Xu))
        entry["fam_logreg"] = fam_metrics(fv, ft, fu)
        dh = DiscHead().fit(Xtr[ai_tr], fam_tr)
        c = np.asarray(dh.centers_)
        def smp(D):
            d2 = ((D[:, None, :] - c[None]) ** 2).sum(-1)
            return softmax(-d2)
        entry["fam_dischead"] = fam_metrics(smp(dh.transform(Xv[ai_val])),
                                            smp(dh.transform(Xt[ai_te])),
                                            smp(dh.transform(Xu)))
        print(f"[e25off] {name:<6} det te {entry['det']['te_auroc']} | "
              f"fam logreg bal {entry['fam_logreg']['balanced_acc']} / "
              f"disc bal {entry['fam_dischead']['balanced_acc']}", flush=True)
        arms[name] = entry
        return entry

    run_arm("B0", rm_tr, rm_v, rm_t, rm_u)
    run_arm("B1", z_tr, z_v, z_t, z_u)
    run_arm("B1mu", z_tr[:, :768], z_v[:, :768], z_t[:, :768], z_u[:, :768])
    run_arm("B3", rF_tr, rF_v, rF_t, rF_u)
    # B2：修复后温度能量（仅检测）
    sDv = s_d_scalar(z_v, st, bD); sDt = s_d_scalar(z_t, st, bD)
    pv2 = 1 / (1 + np.exp(-np.clip(sDv, -50, 50)))
    pt2 = 1 / (1 + np.exp(-np.clip(sDt, -50, 50)))
    pv2r = 1 / (1 + np.exp(-np.clip(s_d_scalar(rF_v, st, bD), -50, 50)))
    pt2r = 1 / (1 + np.exp(-np.clip(s_d_scalar(rF_t, st, bD), -50, 50)))
    arms["B2"] = {"det": det_metrics(pv2, pt2),
                  "det_on_rF_features": det_metrics(pv2r, pt2r)}
    print(f"[e25off] B2 det te {arms['B2']['det']['te_auroc']}（rF 上 "
          f"{arms['B2']['det_on_rF_features']['te_auroc']}）", flush=True)
    report["arms"] = arms

    # ---- 调查：z 分量分解（val；logreg bal） ----
    comps = {"mu": (0, 768), "logv": (768, 1536), "stop": (1536, 1537),
             "mu+logv": (0, 1536), "z": (0, 1537)}
    inv = {}
    for tag, (a, b) in comps.items():
        Xc_tr, Xc_v = z_tr[:, a:b], z_v[:, a:b]
        sc = StandardScaler().fit(Xc_tr)
        lrd = LogisticRegression(max_iter=3000, C=1.0).fit(sc.transform(Xc_tr), y_tr)
        det = round(float(roc_auc_score(y_val, lrd.predict_proba(sc.transform(Xc_v))[:, 1])), 4)
        d = max(1, b - a)
        if d >= 2 and d <= 1537:
            scf = StandardScaler().fit(Xc_tr[ai_tr])
            lrf = LogisticRegression(max_iter=3000, C=1.0).fit(
                scf.transform(Xc_tr[ai_tr]), fam_tr)
            pf = lrf.predict(scf.transform(Xc_v[ai_val]))
            fam = {"logreg_bal": round(float(balanced_accuracy_score(fam_val, pf)), 4)}
        else:
            fam = {"logreg_bal": None}
        inv[tag] = {"dim": d, "val_det_auroc": det, "val_fam": fam}
        print(f"[e25off] 分量 {tag:<8} dim {d:<5} det {det} | fam {fam}", flush=True)
    report["component_decomposition_val"] = inv

    # ---- 调查：generator 级信号（train 内 ≥100 样本的 generator，raw_mean 与 z） ----
    from collections import Counter
    gens = np.array([d.gen for d in corpus["train"] if d.y_ai])
    gv = np.array([d.gen for d in corpus["val"] if d.y_ai])
    cnt = Counter(gens.tolist())
    keep_gens = [g for g, n in cnt.items() if n >= 100]
    mtr = np.isin(gens, keep_gens)
    lab = {g: i for i, g in enumerate(sorted(keep_gens))}
    y_g = np.array([lab[g] for g in gens[mtr]])
    gen_inv = {}
    for tag, X in (("raw_mean", rm_tr[ai_tr][mtr]), ("z", z_tr[ai_tr][mtr])):
        lr = LogisticRegression(max_iter=3000, C=1.0).fit(
            StandardScaler().fit(X).transform(X), y_g)
        mval = np.isin(gv, keep_gens)
        Xval = (rm_v if tag == "raw_mean" else z_v)[ai_val][mval]
        pred = lr.predict(StandardScaler().fit(X).transform(Xval))
        yv = np.array([lab[g] for g in gv[mval]])
        gen_inv[tag] = {"n_generators": int(len(keep_gens)),
                        "val_gen_acc": round(float((pred == yv).mean()), 4),
                        "val_n": int(mval.sum())}
        print(f"[e25off] generator 级·{tag}: {gen_inv[tag]}", flush=True)
    report["generator_level_val"] = gen_inv
    report["family_generator_table"] = json.loads((OUT / "stage0.json").read_text()).get(
        "generators", {})

    # ---- 语言/长度桶（B0/B1/B1mu/B3，test_seen） ----
    ntok_te = np.asarray([int(x) for x in meta["test_seen"]["n_tokens"]])
    ntok_tr = np.asarray([int(x) for x in meta["train"]["n_tokens"]])
    qs = np.percentile(ntok_tr, [25, 50, 75])
    len_bucket = np.searchsorted(qs, ntok_te)
    lang_te = np.asarray(meta["test_seen"]["language"])

    def buckets(prob_t, fam_prob_t):
        out = {}
        combos = [("len", f"q{i}", len_bucket == i) for i in range(4)]
        combos += [("lang", lg, lang_te == lg) for lg in sorted(set(lang_te.tolist()))]
        for tag, key, msk in combos:
            if msk.sum() < 30:
                continue
            e = {"n": int(msk.sum())}
            yb = y_te[msk]
            if len(set(yb.tolist())) > 1:
                e["det_auroc"] = round(float(roc_auc_score(yb, prob_t[msk])), 4)
            m_ai = msk[ai_idx_te]
            if m_ai.sum() >= 10:
                e["fam_bal"] = round(float(balanced_accuracy_score(
                    fam_te[m_ai], fam_prob_t[m_ai].argmax(1))), 4)
            out[f"{tag}:{key}"] = e
        return out

    # 重跑各臂取概率（复用已保存的标准化逻辑，另存）
    bks = {}
    for name, Xtr, Xv, Xt, Xu in (("B0", rm_tr, rm_v, rm_t, rm_u),
                                  ("B1", z_tr, z_v, z_t, z_u),
                                  ("B1mu", z_tr[:, :768], z_v[:, :768], z_t[:, :768], z_u[:, :768]),
                                  ("B3", rF_tr, rF_v, rF_t, rF_u)):
        sc = StandardScaler().fit(Xtr)
        lr_d = LogisticRegression(max_iter=3000, C=1.0).fit(sc.transform(Xtr), y_tr)
        pt = lr_d.predict_proba(sc.transform(Xt))[:, 1]
        scf = StandardScaler().fit(Xtr[ai_tr])
        lr_f = LogisticRegression(max_iter=3000, C=1.0).fit(scf.transform(Xtr[ai_tr]), fam_tr)
        ft = lr_f.predict_proba(scf.transform(Xt[ai_te]))
        bks[name] = buckets(pt, ft)
    report["buckets_test_seen"] = bks
    # B2 的桶（检测）
    report["buckets_test_seen"]["B2"] = buckets(pt2, np.zeros((int(ai_te.sum()), 2)))
    report["buckets_test_seen"]["B2"] = {k: {kk: vv for kk, vv in v.items() if kk != "fam_bal"}
                                         for k, v in report["buckets_test_seen"]["B2"].items()}

    # ---- generator 级召回（test_seen） ----
    gen_t = np.asarray(meta["test_seen"]["generator"])
    def gen_recall(pred_ai):
        out = {}
        for g in sorted(set(gen_t[ai_idx_te].tolist())):
            m = gen_t[ai_idx_te] == g
            if m.sum() < 8:
                continue
            out[g] = {"n": int(m.sum()),
                      "acc": round(float((pred_ai[m] == fam_te[m]).mean()), 3)}
        return out
    gr = {}
    for name, Xtr, Xt in (("B0", rm_tr, rm_t), ("B1", z_tr, z_t)):
        scf = StandardScaler().fit(Xtr[ai_tr])
        lr_f = LogisticRegression(max_iter=3000, C=1.0).fit(scf.transform(Xtr[ai_tr]), fam_tr)
        gr[name] = gen_recall(lr_f.predict(scf.transform(Xt[ai_te])))
    report["generator_recall_test"] = gr

    report["timing_min"] = round((time.time() - t0) / 60, 1)
    (OUT / "e25_offline_and_invest.json").write_text(json.dumps(
        report, ensure_ascii=False, indent=2, default=str))
    print(f"[e25off] 完成 {report['timing_min']} min → "
          f"{OUT/'e25_offline_and_invest.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
