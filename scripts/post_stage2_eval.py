"""post-stage2 Phase A 评估：A0 对照 / A-readout-1 冻结对象 / A-readout-2 重训 / paired delta。

依据《d-det_AutoDL_后续干预与主干探针指导_2026-10-08》§4-§5：
- A0：stage-1 缓存特征重放 22 对象 → 与 dev_scores_new.npz 对齐（不改动任何旧产物）；
- A-readout-1：stage-1b 保存对象在变换文本上仅 transform/predict（回答"原分类器对表面干预有多脆弱"）；
- A-readout-2：在每个变换文本上按 stage-1 原规则重训同一组固定读出（回答"干预后重新训练仍可读多少"）；
- 统计：500 次 task-cluster bootstrap（seed 20261008，同一折内跨变换/读出共享同一 picks 序列）；
- 全部标记 exploratory_train_dev_only=true；不读取 test。

用法：
  python scripts/post_stage2_eval.py --stage a0
  python scripts/post_stage2_eval.py --stage r1 --transform A1_format_norm [--folds-limit 1]
  python scripts/post_stage2_eval.py --stage r2 --transform A1_format_norm [--folds-limit 1]
  python scripts/post_stage2_eval.py --stage paired
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import variant_transfer_stage1_execute as s1  # noqa: E402

OUT = ROOT / "d-det/artifacts/post_stage2_intervention_2026-10-08"
S1 = ROOT / "d-det/artifacts/variant_transfer_stage1_2026-10-08"
S1B = ROOT / "d-det/artifacts/variant_transfer_stage1b_2026-10-08"
PRE = ROOT / "d-det/artifacts/variant_transfer_preflight_2026-10-08"
OBJ = S1B / "objects"

BOOT = 500
BOOT_SEED = 20261008
READOUTS = ("tfidf_char", "tfidf_word", "sem_base", "sem_small", "style_lr", "style_lgb",
            "metadata_only", "size_length_only", "P0_fusion", "P0_equal")
LOG: list[str] = []


def log(msg: str) -> None:
    print(msg, flush=True)
    LOG.append(msg)


def fold_list():
    return [(se, h, arm) for se in s1.SERIES for h in s1.SERIES[se] for arm in s1.ARM_NAMES]


def fold_key_of(se, h, arm):
    return f"{se}::heldout={s1.SIZE_LABEL[h]}::{arm}"


def make_picks(n_tasks: int, repeats: int = BOOT, seed: int = BOOT_SEED):
    rng = np.random.default_rng(seed)
    return [rng.choice(n_tasks, size=n_tasks, replace=True) for _ in range(repeats)]


def boot_stats(y, s, task_arr, picks):
    """给定共享 picks 的任务聚类 bootstrap；返回 point + CI（AUROC/AP/task-macro）。"""
    uniq = np.unique(task_arr)
    idx_by = {q: np.where(task_arr == q)[0] for q in uniq}
    a = np.empty(len(picks)); p = np.empty(len(picks)); tm = np.empty(len(picks))
    for k, pick in enumerate(picks):
        idx = np.concatenate([idx_by[uniq[q]] for q in pick])
        y2, s2 = y[idx], s[idx]
        a[k] = s1._auc(y2, s2)
        p[k] = s1._ap(y2, s2)
        tm[k] = s1.task_macro_auc(y2, s2, task_arr[idx])
    return {"n": int(len(y)), "n_tasks": int(len(uniq)), "pos": int(y.sum()), "neg": int(len(y) - y.sum()),
            "auroc": {"point": s1._auc(y, s), "ci95": [float(np.nanpercentile(a, 2.5)), float(np.nanpercentile(a, 97.5))]},
            "ap": {"point": s1._ap(y, s), "ci95": [float(np.nanpercentile(p, 2.5)), float(np.nanpercentile(p, 97.5))]},
            "task_macro_auroc": {"point": s1.task_macro_auc(y, s, task_arr),
                                 "ci95": [float(np.nanpercentile(tm, 2.5)), float(np.nanpercentile(tm, 97.5))]}}


def task_terms_fast(y, s, uniq, idx_by):
    """每任务 a_t（与 stage-2 task_terms 完全同口径：0<pos<n 才计算，否则 nan）。"""
    a = np.empty(len(uniq))
    for g, q in enumerate(uniq):
        sel = idx_by[q]
        yy = y[sel]
        if 0 < yy.sum() < len(sel):
            a[g] = s1._fast_auc(yy, s[sel])
        else:
            a[g] = np.nan
    return a


def paired_boot(y, yy_lists, idx_lists, s_orig, s_new, a_orig, a_new, picks):
    """同一 picks 下 paired delta（new − orig）。

    AUROC=行级重采样（与 stage-2 eval_domain 同式）；task-macro=重数保留修正口径
    M*=T^-1 Σ a_{t_j*}（与 stage-2 一致，替代早期去重版）。"""
    da = np.empty(len(picks)); dtm = np.empty(len(picks))
    for k, (yy, ii) in enumerate(zip(yy_lists, idx_lists)):
        da[k] = s1._fast_auc(yy, s_new[ii]) - s1._fast_auc(yy, s_orig[ii])
        dtm[k] = float(np.nanmean(a_new[picks[k]])) - float(np.nanmean(a_orig[picks[k]]))
    return {"auroc_delta": {"point": s1._auc(y, s_new) - s1._auc(y, s_orig),
                            "mean": float(np.nanmean(da)),
                            "ci95": [float(np.nanpercentile(da, 2.5)), float(np.nanpercentile(da, 97.5))],
                            "frac_le_0": float(np.mean(da <= 0)) if not np.any(np.isnan(da)) else None},
            "task_macro_delta": {"point": float(np.nanmean(a_new) - np.nanmean(a_orig)),
                                 "mean": float(np.nanmean(dtm)),
                                 "ci95": [float(np.nanpercentile(dtm, 2.5)), float(np.nanpercentile(dtm, 97.5))],
                                 "frac_le_0": float(np.mean(dtm <= 0)) if not np.any(np.isnan(dtm)) else None}}


# ------------------------------------------------------------------ 折数据准备
def build_fold_arrays(transform_id: str | None):
    """返回 samples、折数组（每折 tr_idx/dv_idx/y/tasks）。transform_id=None 用原始文本特征。"""
    split_of = s1.load_split()
    samples = s1.load_samples(split_of)
    n = len(samples)
    unit_rows = defaultdict(list)
    for i, r in enumerate(samples):
        unit_rows[r["unit"]].append(i)
    amend = json.loads((PRE / "r1_negative_set_amendment.json").read_text(encoding="utf-8"))
    per_fold_sel = {k: v["size_matched_negatives"]["selected"] for k, v in
                    amend["size_matched_arm"]["per_fold"].items()}
    folds = {}
    for se, h, arm in fold_list():
        key_core = f"{se}::heldout={s1.SIZE_LABEL[h]}"
        pos = [m for m in s1.SERIES[se] if m != h]
        neg_all = [m for ss in s1.SERIES if ss != se for m in s1.SERIES[ss]]
        neg = neg_all if arm == "size_mix" else per_fold_sel[key_core]
        units = pos + list(neg)
        tr = np.array(sorted(i for u in units for i in unit_rows[u] if samples[i]["split"] == "train"))
        dv = np.array(sorted(i for u in units for i in unit_rows[u] if samples[i]["split"] == "dev"))
        y_tr = np.array([0 if samples[i]["unit"] in neg else 1 for i in tr])
        y_dv = np.array([0 if samples[i]["unit"] in neg else 1 for i in dv])
        tasks_dv = np.array([samples[i]["task"] for i in dv])
        folds[fold_key_of(se, h, arm)] = {"tr": tr, "dv": dv, "y_tr": y_tr, "y_dv": y_dv, "tasks_dv": tasks_dv}
    return samples, folds


def load_features(transform_id: str | None):
    """transform_id=None → stage-1 原文本缓存；否则 local/features/{tid}。返回 dict。"""
    if transform_id is None:
        base = S1 / "features"
    else:
        base = OUT / "local/features" / transform_id
    sm = np.load(base / "style_meta.npz")
    es = np.load(base / "emb_codet5_small.npz")["emb"]
    eb = np.load(base / "emb_codet5_base.npz")["emb"]
    return {"style": sm["style"].astype(np.float64), "meta": sm["meta"].astype(np.float64),
            "sizelen": sm["sizelen"].astype(np.float64), "emb_small": es, "emb_base": eb}


def load_texts(transform_id: str) -> list[str]:
    import gzip
    p = OUT / "local/transformed_texts" / f"{transform_id}.jsonl.gz"
    out = []
    with gzip.open(p, "rt", encoding="utf-8") as f:
        for line in f:
            out.append(json.loads(line)["text"])
    return out


# ------------------------------------------------------------------ 预测（冻结对象）
def predict_object(obj, feats, texts, tr_idx, dv_idx):
    """仅 transform/predict：使用对象保存的 vectorizer/scaler/分类器。返回 name -> dev probs。"""
    dev = {}
    tr_texts = [texts[i] for i in tr_idx]
    dv_texts = [texts[i] for i in dv_idx]
    Xc_tr = obj["char_vec"].transform(tr_texts)
    Xc_dv = obj["char_vec"].transform(dv_texts)
    Xw_tr = obj["word_vec"].transform(tr_texts)
    Xw_dv = obj["word_vec"].transform(dv_texts)
    for name, Xtr, Xdv in (("tfidf_char", Xc_tr, Xc_dv), ("tfidf_word", Xw_tr, Xw_dv)):
        probs = np.zeros(Xdv.shape[0])
        for sd in obj["members"][name]["seeds"]:
            z = np.asarray(Xdv @ sd["coef"].T + sd["intercept"], dtype=np.float64).ravel()
            probs += expit(z)
        dev[name] = probs / len(obj["members"][name]["seeds"])
    st_tr, st_dv = feats["style"][tr_idx], feats["style"][dv_idx]
    eb_tr, eb_dv = feats["emb_base"][tr_idx], feats["emb_base"][dv_idx]
    es_tr, es_dv = feats["emb_small"][tr_idx], feats["emb_small"][dv_idx]
    mt_tr, mt_dv = feats["meta"][tr_idx], feats["meta"][dv_idx]
    sl_tr, sl_dv = feats["sizelen"][tr_idx], feats["sizelen"][dv_idx]
    for name, Fdv in (("sem_base", eb_dv), ("sem_small", es_dv), ("style_lr", st_dv),
                      ("metadata_only", mt_dv), ("size_length_only", sl_dv)):
        m = obj["members"][name]
        Z = m["scaler"].transform(Fdv)
        dev[name] = m["clf"].predict_proba(Z)[:, 1]
    m = obj["members"]["style_lgb"]
    dev["style_lgb"] = m["model"].predict_proba(st_dv)[:, 1]
    P5 = np.column_stack([dev[k] for k in s1.P0_MEMBERS])
    dev["P0_fusion"] = obj["members"]["P0_fusion"]["clf"].predict_proba(np.log(np.clip(P5, 1e-6, 1)))[:, 1]
    dev["P0_equal"] = P5.mean(1)
    return dev


# ------------------------------------------------------------------ A0
def stage_a0():
    t0 = time.time()
    samples, folds = build_fold_arrays(None)
    # float64 重算 style/meta/sizelen（避免 stage-1 float32 缓存量化误差）；emb 缓存本身即 float32 原精度
    texts = [r["code"] for r in samples]
    style_mat = np.array([s1.style_features(tx) for tx in texts], dtype=np.float64)
    feats = {"style": style_mat, "meta": style_mat[:, s1.META_IDX].copy(),
             "sizelen": np.array([[np.log10(s1.PARAM_B[r["unit"]]), style_mat[i, 0], style_mat[i, 1]]
                                  for i, r in enumerate(samples)], dtype=np.float64),
             "emb_small": np.load(S1 / "features/emb_codet5_small.npz")["emb"],
             "emb_base": np.load(S1 / "features/emb_codet5_base.npz")["emb"]}
    ref = np.load(S1B / "dev_scores_new.npz")
    report = {"schema": "post_stage2_a0_replay_check_v1",
              "generated_utc": datetime.now(timezone.utc).isoformat(),
              "note": "原始文本 + stage-1 缓存特征重放 22 对象；与 dev_scores_new.npz 对齐",
              "folds": {}}
    worst = 0.0
    for fk, fd in folds.items():
        pkl = OBJ / f"{fk.replace('::', '__').replace('=', '-')}.pkl"
        obj = pickle.loads(pkl.read_bytes())
        dev = predict_object(obj, feats, texts, fd["tr"], fd["dv"])
        # meta 校验
        meta_ref = ref[f"{fk}::__meta__"]
        meta_new = np.array([f"{samples[i]['unit']}|{samples[i]['task']}|{int(fd['y_dv'][j])}"
                             for j, i in enumerate(fd["dv"])])
        meta_ok = bool(np.array_equal(meta_ref, meta_new))
        maxd = 0.0
        per = {}
        for name in READOUTS:
            if name == "P0_equal":
                ref_v = ref[f"{fk}::P0_equal"]
            else:
                ref_v = ref[f"{fk}::{name}"]
            d = float(np.max(np.abs(ref_v - dev[name])))
            per[name] = d
            maxd = max(maxd, d)
        worst = max(worst, maxd)
        report["folds"][fk] = {"meta_ok": meta_ok, "max_abs_diff": maxd, "per_readout": per}
        log(f"  {fk}: max|Δ|={maxd:.3e} meta_ok={meta_ok}")
    report["worst_max_abs_diff"] = worst
    report["pass_1e-9"] = bool(worst <= 1e-9)
    report["note_precision"] = ("dev_scores_new.npz 由 stage-1b rebuild 以 float64 style 特征生成；"
                                "本重放同式 float64 重算；emb 为 float32 原精度。")
    report["runtime_seconds"] = time.time() - t0
    (OUT / "a0_replay_check.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"[a0] worst max|Δ|={worst:.3e} pass_1e-9={report['pass_1e-9']} in {time.time()-t0:.1f}s")


# ------------------------------------------------------------------ readout-1
def stage_r1(args):
    t0 = time.time()
    tid = args.transform
    samples, folds = build_fold_arrays(tid)
    feats = load_features(tid)
    texts = load_texts(tid)
    dev_scores = {}
    metrics = {"schema": "post_stage2_readout1_metrics_v1", "transform_id": tid,
               "generated_utc": datetime.now(timezone.utc).isoformat(),
               "protocol": {"readout": "frozen stage-1b objects, transform/predict only",
                            "bootstrap": f"task-cluster 500 seed {BOOT_SEED} shared picks"},
               "folds": {}}
    items = list(folds.items())
    if args.folds_limit:
        items = items[:args.folds_limit]
    for fk, fd in items:
        pkl = OBJ / f"{fk.replace('::', '__').replace('=', '-')}.pkl"
        obj = pickle.loads(pkl.read_bytes())
        dev = predict_object(obj, feats, texts, fd["tr"], fd["dv"])
        picks = make_picks(len(np.unique(fd["tasks_dv"])))
        fm = {}
        for name in READOUTS:
            fm[name] = boot_stats(fd["y_dv"], dev[name], fd["tasks_dv"], picks)
            dev_scores[f"{fk}::{name}"] = np.asarray(dev[name], dtype=np.float64)
        metrics["folds"][fk] = {"metrics": fm, "n_dev": int(len(fd["dv"])), "n_train": int(len(fd["tr"]))}
        log(f"  [{tid}] {fk}: P0_fusion AUROC {fm['P0_fusion']['auroc']['point']:.4f} "
            f"| tfidf_char {fm['tfidf_char']['auroc']['point']:.4f} "
            f"| style_lgb {fm['style_lgb']['auroc']['point']:.4f}")
    metrics["runtime_seconds"] = time.time() - t0
    np.savez_compressed(OUT / "local" / f"dev_scores_r1_{tid}.npz", **dev_scores)
    (OUT / f"metrics_readout1_{tid}.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"[r1:{tid}] done in {time.time()-t0:.1f}s folds={len(items)}")


# ------------------------------------------------------------------ readout-2
def stage_r2(args):
    t0 = time.time()
    tid = args.transform
    samples, folds = build_fold_arrays(tid)
    feats = load_features(tid)
    texts = load_texts(tid)
    style_mat, meta_mat, szfeat = feats["style"], feats["meta"], feats["sizelen"]
    emb_small, emb_base = feats["emb_small"], feats["emb_base"]
    dev_scores = {}
    metrics = {"schema": "post_stage2_readout2_metrics_v1", "transform_id": tid,
               "generated_utc": datetime.now(timezone.utc).isoformat(),
               "protocol": {"readout": "retrained on transformed train; dev selection by original rules",
                            "bootstrap": f"task-cluster 500 seed {BOOT_SEED} shared picks",
                            "note": "exploratory_train_dev_only=true"},
               "folds": {}}
    items = list(folds.items())
    if args.folds_limit:
        items = items[:args.folds_limit]
    for fk, fd in items:
        tr_idx, dv_idx = fd["tr"], fd["dv"]
        y_tr, y_dv, tasks_dv = fd["y_tr"], fd["y_dv"], fd["tasks_dv"]
        log(f"  [{tid}] {fk} train={len(tr_idx)} dev={len(dv_idx)}")
        tr_texts = [texts[i] for i in tr_idx]
        dv_texts = [texts[i] for i in dv_idx]
        t = time.time()
        cv_char = s1.tfidf_char().fit(tr_texts)
        Xc_tr, Xc_dv = cv_char.transform(tr_texts), cv_char.transform(dv_texts)
        cv_word = s1.tfidf_word().fit(tr_texts)
        Xw_tr, Xw_dv = cv_word.transform(tr_texts), cv_word.transform(dv_texts)
        log(f"    tfidf in {time.time()-t:.1f}s")
        P_tr, P_dv = {}, {}
        ptr, pdv, _ = s1.sgd_seeds(Xc_tr, y_tr, Xc_dv, y_dv)
        P_tr["tfidf_char"], P_dv["tfidf_char"] = ptr, pdv
        ptr, pdv, _ = s1.sgd_seeds(Xw_tr, y_tr, Xw_dv, y_dv)
        P_tr["tfidf_word"], P_dv["tfidf_word"] = ptr, pdv
        st_tr, st_dv = style_mat[tr_idx], style_mat[dv_idx]
        eb_tr, eb_dv = emb_base[tr_idx], emb_base[dv_idx]
        es_tr, es_dv = emb_small[tr_idx], emb_small[dv_idx]
        mt_tr, mt_dv = meta_mat[tr_idx], meta_mat[dv_idx]
        sl_tr, sl_dv = szfeat[tr_idx], szfeat[dv_idx]
        C_sel = {}
        for name, (Ftr, Fdv) in (("sem_base", (eb_tr, eb_dv)), ("sem_small", (es_tr, es_dv)),
                                 ("style_lr", (st_tr, st_dv)), ("metadata_only", (mt_tr, mt_dv)),
                                 ("size_length_only", (sl_tr, sl_dv))):
            ptr, pdv, C = s1.lr_select_auroc(Ftr, y_tr, Fdv, y_dv)
            P_tr[name], P_dv[name] = ptr, pdv
            C_sel[name] = C
        ptr, pdv = s1.lgb_fit(st_tr, y_tr, st_dv)
        P_tr["style_lgb"], P_dv["style_lgb"] = ptr, pdv
        Ptr5 = np.column_stack([P_tr[k] for k in s1.P0_MEMBERS])
        Pdv5 = np.column_stack([P_dv[k] for k in s1.P0_MEMBERS])
        ptr, pdv = s1.fusion_lr(Ptr5, y_tr, Pdv5)
        P_tr["P0_fusion"], P_dv["P0_fusion"] = ptr, pdv
        P_tr["P0_equal"], P_dv["P0_equal"] = Ptr5.mean(1), Pdv5.mean(1)
        picks = make_picks(len(np.unique(tasks_dv)))
        fm = {}
        for name in READOUTS:
            fm[name] = boot_stats(y_dv, P_dv[name], tasks_dv, picks)
            dev_scores[f"{fk}::{name}"] = np.asarray(P_dv[name], dtype=np.float64)
        metrics["folds"][fk] = {"metrics": fm, "C_selected": C_sel,
                                "n_dev": int(len(dv_idx)), "n_train": int(len(tr_idx))}
        log(f"    P0_fusion dev AUROC {fm['P0_fusion']['auroc']['point']:.4f}")
    metrics["runtime_seconds"] = time.time() - t0
    np.savez_compressed(OUT / "local" / f"dev_scores_r2_{tid}.npz", **dev_scores)
    (OUT / f"metrics_readout2_{tid}.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"[r2:{tid}] done in {time.time()-t0:.1f}s folds={len(items)}")


# ------------------------------------------------------------------ paired delta
def summarize_cross_folds(metrics, readout="P0_fusion"):
    """系列内折等权 → 系列等权；返回 {auroc_mean, tm_mean, auroc_folds...}。"""
    per_series = defaultdict(list)
    for fk, fv in metrics["folds"].items():
        se = fk.split("::")[0]
        per_series[se].append(fv["metrics"][readout]["auroc"]["point"])
    series_means = {se: float(np.mean(v)) for se, v in per_series.items()}
    all_mean = float(np.mean(list(series_means.values())))
    return series_means, all_mean, {se: len(v) for se, v in per_series.items()}


def build_fold_packs(folds):
    """每折预构建 uniq/idx_by/picks/idx_lists/yy_lists（跨变换与读出共享 → 同一抽样序列）。"""
    packs = {}
    for fk, fd in folds.items():
        tasks_dv = fd["tasks_dv"]
        uniq = np.unique(tasks_dv)
        idx_by = {q: np.where(tasks_dv == q)[0] for q in uniq}
        picks = make_picks(len(uniq))
        idx_lists = [np.concatenate([idx_by[uniq[q]] for q in pick]) for pick in picks]
        yy_lists = [fd["y_dv"][ii] for ii in idx_lists]
        packs[fk] = {"uniq": uniq, "idx_by": idx_by, "picks": picks,
                     "idx_lists": idx_lists, "yy_lists": yy_lists, "fd": fd}
    return packs


def stage_paired(args):
    t0 = time.time()
    samples, folds = build_fold_arrays(None)
    ref = np.load(S1B / "dev_scores_new.npz")
    packs = build_fold_packs(folds)
    out = {"schema": "post_stage2_paired_delta_v1",
           "generated_utc": datetime.now(timezone.utc).isoformat(),
           "protocol": ("paired task-cluster bootstrap (shared picks per fold); delta = transformed − original (dev); "
                        "task-macro multiplicity-preserving M*=T^-1 Σ a_{t_j*} (stage-2 task_terms consistent)"),
           "transforms": {}}
    for tid in ("A1_format_norm", "A2_comments_masked", "A3_literals_masked"):
        for rtag in ("r1", "r2"):
            p = OUT / "local" / f"dev_scores_{rtag}_{tid}.npz"
            if not p.exists():
                log(f"  skip {rtag}:{tid} (missing)")
                continue
            new = np.load(p)
            fam = out["transforms"].setdefault(tid, {}).setdefault(rtag, {"folds": {}})
            for fk, pack in packs.items():
                fd = pack["fd"]
                y_dv = fd["y_dv"]
                fmo = {}
                for name in READOUTS:
                    k = f"{fk}::{name}"
                    if k not in new.files:
                        continue
                    s_new = np.asarray(new[k], dtype=np.float64)
                    s_orig = np.asarray(ref[k], dtype=np.float64)
                    # dtype 一致性：sem 读出为 float32 路径（rankdata in float32 → 按原路径）
                    if name in ("sem_base", "sem_small"):
                        s_new = s_new.astype(np.float32)
                        s_orig = s_orig.astype(np.float32)
                    a_o = task_terms_fast(y_dv, s_orig, pack["uniq"], pack["idx_by"])
                    a_n = task_terms_fast(y_dv, s_new, pack["uniq"], pack["idx_by"])
                    fmo[name] = paired_boot(y_dv, pack["yy_lists"], pack["idx_lists"],
                                            s_orig, s_new, a_o, a_n, pack["picks"])
                fam["folds"][fk] = fmo
                if "P0_fusion" in fmo:
                    log(f"  [{tid}·{rtag}] {fk}: P0_fusion ΔAUROC "
                        f"{fmo['P0_fusion']['auroc_delta']['point']:+.4f}")
            pts = [fv.get("P0_fusion", {}).get("auroc_delta", {}).get("point")
                   for fv in fam["folds"].values()]
            pts = [x for x in pts if x is not None]
            log(f"  [{tid}·{rtag}] fold-mean ΔAUROC (P0_fusion) = {np.mean(pts):+.4f}")
        # 汇总每个 transform×rtag 的跨折 delta 均值（全读出）
        for rtag in ("r1", "r2"):
            if rtag not in out["transforms"].get(tid, {}):
                continue
            fam = out["transforms"][tid][rtag]
            summary = {}
            for name in READOUTS:
                pts = [fv[name]["auroc_delta"]["point"] for fv in fam["folds"].values() if name in fv]
                if pts:
                    summary[name] = {"auroc_delta_mean_over_folds": float(np.mean(pts)),
                                     "auroc_delta_min": float(np.min(pts)),
                                     "auroc_delta_max": float(np.max(pts)),
                                     "k_folds": len(pts)}
            fam["summary"] = summary
    out["runtime_seconds"] = time.time() - t0
    (OUT / "paired_delta.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"[paired] done in {time.time()-t0:.1f}s")


def stage_fix_tm(args):
    """将 r1/r2 metrics JSON 的 task_macro CI 修正为与 stage-2 相同的重数保留口径（M*=T^-1 Σ a）。"""
    t0 = time.time()
    samples, folds = build_fold_arrays(None)
    packs = build_fold_packs(folds)
    for tid in ("A1_format_norm", "A2_comments_masked", "A3_literals_masked"):
        for rtag in ("r1", "r2"):
            npz_p = OUT / "local" / f"dev_scores_{rtag}_{tid}.npz"
            met_p = OUT / f"metrics_readout{rtag[-1]}_{tid}.json"
            if not npz_p.exists() or not met_p.exists():
                continue
            new = np.load(npz_p)
            met = json.loads(met_p.read_text(encoding="utf-8"))
            n_fix = 0
            for fk, pack in packs.items():
                if fk not in met.get("folds", {}):
                    continue
                y_dv = pack["fd"]["y_dv"]
                for name in READOUTS:
                    k = f"{fk}::{name}"
                    if k not in new.files:
                        continue
                    s = np.asarray(new[k], dtype=np.float64)
                    if name in ("sem_base", "sem_small"):
                        s = s.astype(np.float32)
                    a_t = task_terms_fast(y_dv, s, pack["uniq"], pack["idx_by"])
                    M = np.array([np.nanmean(a_t[p]) for p in pack["picks"]])
                    item = met["folds"][fk]["metrics"].get(name)
                    if item is None:
                        continue
                    item["task_macro_auroc"]["ci95"] = [float(np.nanpercentile(M, 2.5)),
                                                        float(np.nanpercentile(M, 97.5))]
                    n_fix += 1
            met["task_macro_ci_note"] = ("CI corrected to multiplicity-preserving M*=T^-1 Σ a_{t_j*} "
                                         "(stage-2 task_terms consistent); point values unchanged.")
            met_p.write_text(json.dumps(met, ensure_ascii=False, indent=1), encoding="utf-8")
            log(f"  [fix_tm] {rtag}:{tid} fixed {n_fix} entries")
    log(f"[fix_tm] done in {time.time()-t0:.1f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["a0", "r1", "r2", "paired", "fix_tm"])
    ap.add_argument("--transform", default="")
    ap.add_argument("--folds-limit", type=int, default=0)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "logs").mkdir(exist_ok=True)
    if args.stage == "a0":
        stage_a0()
    elif args.stage == "r1":
        stage_r1(args)
    elif args.stage == "r2":
        stage_r2(args)
    elif args.stage == "paired":
        stage_paired(args)
    elif args.stage == "fix_tm":
        stage_fix_tm(args)
    (OUT / "logs" / f"eval_{args.stage}_{args.transform or 'all'}.log").write_text(
        "\n".join(LOG) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
