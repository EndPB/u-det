"""强 P0 补强（7ee36fb 修复裁定 §2.4）：lexical/style/meta 双侧 composition + 完整 symmetric pair P0。

每折（48 折沿用 plan）全部只在 train tasks 上拟合：
  1. 三种特征族（lexical=char_wb 2-4 hashed 1024；style=style+sizelen；meta）双侧 [c;i] 拼接，
     fit-only 标准化、**系列等权** LR(3 系列)、fit 内部 cross-fit（5 折按 task 分组）OOF 概率；
  2. 族关系分数 s_g=Σ_f p(i,f)p(j,f)（对称）；完整 symmetric pair P0 = 对称对特征
     [|Δlex|; lex和;|Δstyle|; style和;|Δmeta|; meta和] 的加权 SGD LR（同 q_sym 协议）；
  3. 融合规则 equal / LR-stack 在 train 上内层 5 折 CV 选择（平手→equal）；full_P0=4 成分融合；
  4. 主比较（与 7ee36fb 保存的 s_residual/s_compose 配对）：residual−full_P0、residual−compose（保留 +0.46pt 并列）；
     共享 task-cluster bootstrap 500（向量化 tie 版已合成审计）；AP 用 sklearn（修复口径）。
不重跑任何 7ee36fb 模型；仅新增 P0 成分拟合与分数。

输出：artifacts/relation_endpoint_balanced_strong_p0_2026-10-08/
"""
from __future__ import annotations

import gzip
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import core_r0_eval as ev  # noqa: E402

PLAN = ROOT / "d-det/artifacts/endpoint_balanced_relation_plan_2026-10-08"
BASE = ROOT / "d-det/artifacts/relation_endpoint_balanced_2026-10-08"
REF = BASE / "local"
OUT = ROOT / "d-det/artifacts/relation_endpoint_balanced_strong_p0_2026-10-08"
TEXTS = ROOT / "d-det/artifacts/r0_protocol_diff_2026-10-08/local/texts.jsonl.gz"
SEED = 20261008
LOG: list[str] = []


def log(m):
    print(m, flush=True)
    LOG.append(m)


def weighted_auc(y, s, w):
    order = np.argsort(-s, kind="mergesort")
    ys = y[order]; ss = s[order]; ws = w[order]
    Wp = ws[ys == 1].sum(); Wn = ws[ys == 0].sum()
    pos_w = np.where(ys == 1, ws, 0.0)
    is_new = np.empty(len(ss), bool); is_new[0] = True; is_new[1:] = ss[1:] != ss[:-1]
    seg_id = np.cumsum(is_new) - 1
    seg_sum = np.bincount(seg_id, weights=pos_w)
    cum_before = np.concatenate([[0.0], np.cumsum(seg_sum)[:-1]])
    contrib = float(np.where(ys == 0, ws * (cum_before[seg_id] + 0.5 * seg_sum[seg_id]), 0.0).sum())
    return contrib / (Wp * Wn + 1e-12)


def load_texts():
    texts = {}
    with gzip.open(TEXTS, "rt", encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            texts[o["i"]] = o["text"]
    return texts


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "logs").mkdir(exist_ok=True)
    (OUT / "local").mkdir(exist_ok=True)
    plan = json.loads((PLAN / "endpoint_balanced_plan.json").read_text(encoding="utf-8"))
    folds = plan["folds"]
    A = ev.load_assets()
    splits = {t: A["splits"][A["TASK_I"][t] * 2] for t in A["tasks"]}
    tr_tasks = [t for t in A["tasks"] if splits[t] == "train"]
    dv_tasks = [t for t in A["tasks"] if splits[t] == "dev"]
    nt, nv = len(tr_tasks), len(dv_tasks)
    series = plan["series"]
    log(f"[sp0] tasks train={nt} dev={nv}; folds={len(folds)}")
    texts = load_texts()
    from sklearn.feature_extraction.text import HashingVectorizer
    hv = HashingVectorizer(analyzer="char_wb", ngram_range=(2, 4), n_features=1024,
                           alternate_sign=False, norm="l2", dtype=np.float32)

    def unit_side_feats(members, tasks):
        """返回 dict 族 -> (nM, nt, F, 2)（双侧；lexical 批量 transform）。"""
        nM, nT_ = len(members), len(tasks)
        FST = A["style"].shape[1] + A["sizelen"].shape[1]
        lex = np.zeros((nM, nT_, 1024, 2), dtype=np.float32)
        st = np.zeros((nM, nT_, FST, 2), dtype=np.float64)
        mt = np.zeros((nM, nT_, A["meta"].shape[1], 2), dtype=np.float64)
        # 批量文本
        rows = []
        for m in members:
            M = A["MODEL_I"][m]
            for t in tasks:
                r0 = M * A["nT"] * 2 + A["TASK_I"][t] * 2
                rows.append(r0)
        txt_list = []
        for r0 in rows:
            txt_list.append(texts.get(r0, ""))
            txt_list.append(texts.get(r0 + 1, ""))
        Xlex = hv.transform(txt_list).astype(np.float32).toarray()
        Xlex = Xlex.reshape(nM * nT_, 2, 1024)
        lex[:, :, :, 0] = Xlex[:, 0, :].reshape(nM, nT_, 1024)
        lex[:, :, :, 1] = Xlex[:, 1, :].reshape(nM, nT_, 1024)
        for mi, m in enumerate(members):
            M = A["MODEL_I"][m]
            for k, t in enumerate(tasks):
                r0 = M * A["nT"] * 2 + A["TASK_I"][t] * 2
                st[mi, k, :, 0] = np.r_[A["style"][r0], A["sizelen"][r0]]
                st[mi, k, :, 1] = np.r_[A["style"][r0 + 1], A["sizelen"][r0 + 1]]
                mt[mi, k, :, 0] = A["meta"][r0]
                mt[mi, k, :, 1] = A["meta"][r0 + 1]
        return {"lexical": lex, "style": st, "meta": mt}

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import SGDClassifier

    def sgd_weighted(Xtr, ytr, wtr, Xdv, seeds=(0, 1, 2), epochs=5, bs=4096, alpha=1e-6, ret_train=False):
        Xtr = np.asarray(Xtr, dtype=np.float32); Xdv = np.asarray(Xdv, dtype=np.float32)
        classes = np.array([0, 1]); scores = np.zeros(len(Xdv)); tr_scores = np.zeros(len(Xtr))
        for seed in seeds:
            clf = SGDClassifier(loss="log_loss", alpha=alpha, random_state=seed)
            rng = np.random.default_rng(seed)
            for ep in range(epochs):
                perm = rng.permutation(len(Xtr))
                for i in range(0, len(perm), bs):
                    sel = perm[i:i + bs]
                    clf.partial_fit(Xtr[sel], ytr[sel], classes=classes, sample_weight=wtr[sel])
            scores += clf.decision_function(Xdv)
            if ret_train:
                tr_scores += clf.decision_function(Xtr)
        out = scores / len(seeds)
        if ret_train:
            return out, tr_scores / len(seeds)
        return out

    results = {"schema": "strong_p0_v1", "generated_utc": datetime.now(timezone.utc).isoformat(),
               "report_first_line": "强 P0 补强（lexical/style/meta composition + symmetric pair P0）",
               "folds": {}}
    for fid, fold in enumerate(folds):
        tf0 = time.time()
        M = fold["train_members"]
        tr_members = [m for s in series for m in M[s]]
        tr_series = np.array([series.index(s) for s in series for m in M[s] for _ in tr_tasks])
        ev_members = [m for s in series for m in plan["series_members"][s]]
        Ftr = unit_side_feats(tr_members, tr_tasks)
        Fev = unit_side_feats(ev_members, dv_tasks)
        # 训练对/评测对（沿用 plan 权重与结构）
        tr_pairs, tr_w = [], []
        for p in fold["train_pos"]:
            w0 = p["weight"] / nt
            for t in tr_tasks:
                tr_pairs.append((p["a"], p["b"], t)); tr_w.append(w0)
        for p in fold["train_neg"]:
            w0 = p["weight"] / nt
            for t in tr_tasks:
                tr_pairs.append((p["a"], p["b"], t)); tr_w.append(w0)
        w_tr = np.array(tr_w); w_tr = w_tr / w_tr.mean()
        n_pos_tr = len(fold["train_pos"]) * nt
        n_neg_tr = len(fold["train_neg"]) * nt
        y_tr = np.concatenate([np.ones(n_pos_tr), np.zeros(n_neg_tr)]).astype(np.float32)
        assert len(y_tr) == len(tr_pairs)
        ev_pairs, ev_w = [], []
        for p in fold["eval_pos"]:
            w0 = p["weight"] / nv
            for t in dv_tasks:
                ev_pairs.append((p["anchor"], p["partner"], t)); ev_w.append(w0)
        for p in fold["eval_neg"]:
            w0 = p["weight"] / nv
            for t in dv_tasks:
                ev_pairs.append((p["anchor"], p["partner"], t)); ev_w.append(w0)
        n_pos_ev = len(fold["eval_pos"]) * nv
        n_neg_ev = len(fold["eval_neg"]) * nv
        y_ev = np.concatenate([np.ones(n_pos_ev), np.zeros(n_neg_ev)])
        w_ev = np.array(ev_w)
        assert len(y_ev) == len(ev_pairs)
        midx_tr = {m: i for i, m in enumerate(tr_members)}
        midx_ev = {m: i for i, m in enumerate(ev_members)}
        tidx_tr = {t: i for i, t in enumerate(tr_tasks)}
        tidx_ev = {t: i for i, t in enumerate(dv_tasks)}
        # ---------- 每族：标准化 + 系列等权 LR + OOF ----------
        n_units_tr = len(tr_members) * nt
        family_scores = {}
        scalers = {}
        series_of_unit = tr_series
        unit_w = np.zeros(n_units_tr)
        for si in range(3):
            sel = series_of_unit == si
            unit_w[sel] = 1.0 / sel.sum()
        unit_w = unit_w / unit_w.mean()
        oof_list = {}
        full_list = {}
        for fam, F in Ftr.items():
            Xc = np.concatenate([F[:, :, :, 0], F[:, :, :, 1]], axis=2).reshape(n_units_tr, -1).astype(np.float64)
            sc = StandardScaler().fit(Xc)
            Xs = sc.transform(Xc)
            clf = LogisticRegression(max_iter=2000, C=1.0).fit(Xs, series_of_unit, sample_weight=unit_w)
            scalers[fam] = (sc, clf)
            # OOF（5 折按 task 分组）
            task_of = np.tile(np.arange(nt), len(tr_members))
            oof = np.zeros((n_units_tr, 3))
            for k in range(5):
                trm = (task_of % 5) != k; dvm = ~trm
                c2 = LogisticRegression(max_iter=2000, C=1.0).fit(Xs[trm], series_of_unit[trm], sample_weight=unit_w[trm])
                oof[dvm] = c2.predict_proba(Xs[dvm])
            oof_list[fam] = oof
            full_list[fam] = clf.predict_proba(Xs)
            # 族 pair 分数（eval）
            Xe = np.concatenate([Fev[fam][:, :, :, 0], Fev[fam][:, :, :, 1]], axis=2)
            Pev = clf.predict_proba(sc.transform(Xe.reshape(len(ev_members) * nv, -1).astype(np.float64)))
            def s_pair(P, pairs, midx, tidx):
                out = np.array([float(P[midx[a] * len(tidx) + tidx[t]] @ P[midx[b] * len(tidx) + tidx[t]])
                                for a, b, t in pairs])
                return out
            family_scores[fam] = {"ev": s_pair(Pev, ev_pairs, midx_ev, tidx_ev)}
        # OOF 的 train pair 分数（用 oof_list）
        def s_pair_oof(fam, pairs, midx, tidx):
            P = oof_list[fam]
            return np.array([float(P[midx[a] * nt + tidx[t]] @ P[midx[b] * nt + tidx[t]]) for a, b, t in pairs])
        for fam in Ftr:
            family_scores[fam]["tr_oof"] = s_pair_oof(fam, tr_pairs, midx_tr, tidx_tr)
        # ---------- symmetric pair P0（完整对称对特征） ----------
        def sym_pair_feats(F, pairs, midx, tidx, n_members):
            feats = []
            for fam in ("lexical", "style", "meta"):
                X = F[fam]
                flat = np.concatenate([X[:, :, :, 0], X[:, :, :, 1]], axis=2)  # (nM, nT, 2F)
                ia = np.array([midx[a] for a, b, t in pairs]) * len(tidx) + np.array([tidx[t] for a, b, t in pairs])
                ib = np.array([midx[b] for a, b, t in pairs]) * len(tidx) + np.array([tidx[t] for a, b, t in pairs])
                fa = flat.reshape(-1, flat.shape[-1])[ia]; fb = flat.reshape(-1, flat.shape[-1])[ib]
                feats.append(np.hstack([np.abs(fa - fb), fa + fb]))
            return np.hstack(feats).astype(np.float32)
        Xsym_tr = sym_pair_feats(Ftr, tr_pairs, midx_tr, tidx_tr, len(tr_members))
        Xsym_ev = sym_pair_feats(Fev, ev_pairs, midx_ev, tidx_ev, len(ev_members))
        mu, sd = Xsym_tr.mean(0), Xsym_tr.std(0); sd[sd < 1e-8] = 1.0
        s_pairLR_ev, s_pairLR_tr = sgd_weighted((Xsym_tr - mu) / sd, y_tr, w_tr, (Xsym_ev - mu) / sd, ret_train=True)
        # ---------- 融合（equal vs LR-stack；train 内层 5 折 CV 选择） ----------
        names = ["lexical", "style", "meta"]
        S_oof = np.stack([family_scores[f]["tr_oof"] for f in names], axis=1)
        S_ev = np.stack([family_scores[f]["ev"] for f in names], axis=1)
        Sp_ev = np.stack([S_ev[:, 0], S_ev[:, 1], S_ev[:, 2], s_pairLR_ev], axis=1)
        # 内层 CV：task 分组 5 折，比较 equal vs LR-stack（3 族与 4 成分分别）
        task_of_pair = np.array([tidx_tr[t] for a, b, t in tr_pairs])
        def inner_cv(M_features, y, w):
            aucs = {"equal": [], "lr": []}
            for k in range(5):
                dvm = (task_of_pair % 5) == k; trm = ~dvm
                eq_scores = M_features[dvm].mean(1)
                aucs["equal"].append(weighted_auc(y[dvm], eq_scores, w[dvm]))
                lr = LogisticRegression(max_iter=2000, C=1.0).fit(M_features[trm], y[trm], sample_weight=w[trm])
                aucs["lr"].append(weighted_auc(y[dvm], lr.decision_function(M_features[dvm]), w[dvm]))
            return {k: float(np.mean(v)) for k, v in aucs.items()}
        cv3 = inner_cv(S_oof, y_tr, w_tr)
        rule3 = "lr" if cv3["lr"] > cv3["equal"] else "equal"
        if rule3 == "lr":
            stack = LogisticRegression(max_iter=2000, C=1.0).fit(S_oof, y_tr, sample_weight=w_tr)
            s_fused3_ev = stack.decision_function(Sp_ev[:, :3])
        else:
            s_fused3_ev = Sp_ev[:, :3].mean(1)
        # full_P0（预声明）：4 成分**先各自 z 化再等权** mean（fit-only 尺度参数）。
        # 修正记录：首版直接用原始分数 mean，SGD pairLR 量纲 ~1e5 完全独裁（与单列 AUC 相同）——
        # 改为：3 族用 train OOF 分数的 mean/std、pairLR 用 train in-sample 分数的 mean/std 做 z 化。
        # 不采用 in-sample stack（pairLR 无 OOF，避免泄漏）；3 族规则融合作为 sensitivity 并列。
        z_ev = {}
        z_params = {}
        for k, v_tr, v_ev in (("lexical", family_scores["lexical"]["tr_oof"], S_ev[:, 0]),
                              ("style", family_scores["style"]["tr_oof"], S_ev[:, 1]),
                              ("meta", family_scores["meta"]["tr_oof"], S_ev[:, 2]),
                              ("pairLR", s_pairLR_tr, s_pairLR_ev)):
            m0, s0 = float(np.mean(v_tr)), float(np.std(v_tr))
            if s0 < 1e-12:
                s0 = 1.0
            z_ev[k] = (v_ev - m0) / s0
            z_params[k] = [m0, s0]
        s_fullP0_ev = np.mean([z_ev[k] for k in ("lexical", "style", "meta", "pairLR")], axis=0)
        # 参考分数（canonical）：从 7ee36fb 同折 npz 读 residual/compose，带对齐断言
        s_resid = None; s_compose = None
        zref = REF / f"eval_scores_fold{fid}.npz"
        if zref.exists():
            zr = np.load(zref)
            y_ref = np.asarray(zr["y"]).astype(int)
            if y_ref.shape == np.asarray(y_ev).shape and np.array_equal(y_ref, np.asarray(y_ev).astype(int)):
                s_resid = zr["s_residual"]; s_compose = zr["s_compose"]
                assert len(s_resid) == len(y_ev)
            else:
                log(f"[sp0] fold{fid}: ref 行序/长度不对齐——跳过参考")
        else:
            log(f"[sp0] fold{fid}: ref npz 缺失——跳过参考")
        out = {"fold": fid,
               "family_ev_auc": {f: weighted_auc(y_ev, family_scores[f]["ev"], w_ev) for f in names},
               "pairLR_ev_auc": weighted_auc(y_ev, s_pairLR_ev, w_ev),
               "fullP0_ev_auc": weighted_auc(y_ev, s_fullP0_ev, w_ev),
               "fused3_ev_auc": weighted_auc(y_ev, s_fused3_ev, w_ev),
               "compose_ref_auc": (None if s_compose is None else weighted_auc(y_ev, s_compose, w_ev)),
               "residual_ref_auc": (None if s_resid is None else weighted_auc(y_ev, s_resid, w_ev)),
               "fusion_rule3": rule3, "inner_cv3": cv3, "z_params": z_params,
               "runtime_s": time.time() - tf0}
        results["folds"][fid] = out
        save_kw = dict(y=y_ev, w=w_ev, s_fullP0=s_fullP0_ev, s_fused3=s_fused3_ev, s_pairLR=s_pairLR_ev,
                       **{f"s_{f}": family_scores[f]["ev"] for f in names})
        if s_resid is not None:
            save_kw["s_residual"] = s_resid; save_kw["s_compose"] = s_compose
        np.savez_compressed(OUT / "local" / f"scores_fold{fid}.npz", **save_kw)
        log(f"[sp0] fold{fid}: fam={ {f: round(out['family_ev_auc'][f],3) for f in names} } "
            f"pairLR={out['pairLR_ev_auc']:.4f} fullP0={out['fullP0_ev_auc']:.4f} "
            f"compose_ref={out['compose_ref_auc']:.4f} resid_ref={out['residual_ref_auc']:.4f} "
            f"rule={rule3} ({time.time()-tf0:.0f}s)")

    # ---------- 汇总：共享 task bootstrap（用保存的分数） ----------
    log("[sp0] shared bootstrap…")
    data = {}
    for fid in range(len(folds)):
        z = np.load(OUT / "local" / f"scores_fold{fid}.npz")
        data[fid] = z
    has_ref = all("s_residual" in z for z in data.values())
    reads = ["s_fullP0", "s_fused3", "s_pairLR", "s_lexical", "s_style", "s_meta"]
    if has_ref:
        reads += ["s_residual", "s_compose"]
    dnames = ["fullP0_minus_fused3"] + (["residual_minus_fullP0", "residual_minus_compose", "fullP0_minus_compose"] if has_ref else [])
    uniq_tasks = np.arange(nv)
    rng = np.random.default_rng(SEED)
    boot = {k: [] for k in reads}
    deltas = {k: [] for k in dnames}
    for b in range(500):
        pick = rng.choice(uniq_tasks, size=nv, replace=True)
        per = {k: [] for k in reads}
        per_d = {k: [] for k in dnames}
        for fid, z in data.items():
            y = z["y"]; w = z["w"]
            n = len(y)
            tcol = np.arange(n) % nv
            idx = np.concatenate([np.where(tcol == q)[0] for q in pick])
            for k in reads:
                per[k].append(weighted_auc(y[idx], z[k][idx], w[idx]))
            auc_fullP0 = weighted_auc(y[idx], z["s_fullP0"][idx], w[idx])
            auc_fused3 = weighted_auc(y[idx], z["s_fused3"][idx], w[idx])
            per_d["fullP0_minus_fused3"].append(auc_fullP0 - auc_fused3)
            if has_ref:
                auc_res = weighted_auc(y[idx], z["s_residual"][idx], w[idx])
                auc_comp = weighted_auc(y[idx], z["s_compose"][idx], w[idx])
                per_d["residual_minus_fullP0"].append(auc_res - auc_fullP0)
                per_d["residual_minus_compose"].append(auc_res - auc_comp)
                per_d["fullP0_minus_compose"].append(auc_fullP0 - auc_comp)
        for k in reads:
            boot[k].append(float(np.mean(per[k])))
        for k in dnames:
            deltas[k].append(float(np.mean(per_d[k])))
        if (b + 1) % 100 == 0:
            log(f"[sp0] bootstrap {b+1}/500 ({time.time()-t0:.0f}s)")

    def ci(v):
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    point = {}
    for k in reads:
        vals = [weighted_auc(z["y"], z[k], z["w"]) for z in data.values()]
        point[k] = float(np.mean(vals))
    summary = {"schema": "strong_p0_summary_v1", "generated_utc": datetime.now(timezone.utc).isoformat(),
               "point": point,
               "ci95": {k: ci(boot[k]) for k in reads},
               "deltas": {k: {"mean": float(np.mean(v)), "ci95": ci(v)} for k, v in deltas.items()},
               "fusion_rules": {fid: results["folds"][fid]["fusion_rule3"] for fid in results["folds"]}}
    (OUT / "metrics_strong_p0.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "metrics_strong_p0_folds.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    L = ["# 强 P0 补强（48 折；train/dev）", "",
         "| 读出 | 点值 | 95% CI |", "|---|---|---|"]
    for k in reads:
        L.append(f"| {k} | {point[k]:.4f} | [{summary['ci95'][k][0]:.4f}, {summary['ci95'][k][1]:.4f}] |")
    L += ["", "## 配对 Δ（同一 bootstrap）"]
    for k, v in summary["deltas"].items():
        L.append(f"- {k}: {v['mean']:+.4f} [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]")
    L += ["", "> 全部 P0 成分只在每折 train 上拟合（fit-only scaler、系列等权、task 分组 cross-fit）；",
          "> 融合选择=内层 5 折 CV（equal vs LR-stack）；7ee36fb 的 +0.46pt（residual−compose_canonical）保留并列。"]
    (OUT / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (OUT / "logs" / "sp0_run.log").write_text("\n".join(LOG) + "\n", encoding="utf-8")
    log(f"[sp0] done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
