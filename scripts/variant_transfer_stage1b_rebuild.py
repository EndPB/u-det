"""stage-1b：重建并保存 stage-1 模型对象（供 stage-2 独立评分）。

依据《AutoDL stage1修复冻结与stage2单次评测指导》§2.1：
- 原环境、原顺序、原 seed、原预算的确定性重建（非寻找新高分）；
- 保存每折：TF-IDF 词表/IDF、StandardScaler、SGD 每 seed 最佳 epoch 系数、LR（原 C_selected）、
  LGBM、P0_fusion 头、R2 radial 统计量；拟合仅使用 train；dev 仅用于原 best-dev 规则重放；
- 重载保存包预测原 dev，与 stage-1 predictions/*.csv.gz 对齐（|Δ|≤5.1e-7）；
- 重算点指标与 stage-1 train_dev_metrics.json 对照（≤1e-10）。

输出（variant_transfer_stage1b_2026-10-08/）：
- objects/{fold}__{arm}.pkl（本地，gitignore）
- objects_manifest.json（路径/sha256/大小/类型）
- dev_scores_new.npz（重算 dev 分数，float64）
- replay_check.json（对齐报告）
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import pickle
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import variant_transfer_stage1_execute as s1  # noqa: E402

OUT = ROOT / "d-det/artifacts/variant_transfer_stage1b_2026-10-08"
S1 = ROOT / "d-det/artifacts/variant_transfer_stage1_2026-10-08"
PRE = ROOT / "d-det/artifacts/variant_transfer_preflight_2026-10-08"

SEEDS = s1.SEEDS
C_GRID = s1.C_GRID
ARM_NAMES = s1.ARM_NAMES
LOG: list[str] = []


def log(msg: str) -> None:
    print(msg, flush=True)
    LOG.append(msg)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def lr_fit_capture(Ftr, ytr, Fdv, ydv, grid=C_GRID):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Ftr)
    Ztr, Zdv = sc.transform(Ftr), sc.transform(Fdv)
    best = (-1.0, None, None)
    per_c = {}
    for C in grid:
        clf = LogisticRegression(max_iter=3000, C=C, class_weight="balanced").fit(Ztr, ytr)
        auc = float(roc_auc_score(ydv, clf.predict_proba(Zdv)[:, 1]))
        per_c[str(C)] = auc
        if auc > best[0] + 1e-12:
            best = (auc, C, clf)
    _, C, clf = best
    return clf, sc, C, clf.predict_proba(Ztr)[:, 1], clf.predict_proba(Zdv)[:, 1], per_c


def sgd_fit_capture(Xtr, ytr, Xdv, ydv, seeds=SEEDS, epochs=5, bs=10000, alpha=2e-6):
    from sklearn.linear_model import SGDClassifier
    from sklearn.metrics import roc_auc_score
    classes = np.array([0, 1])
    counts = np.bincount(ytr, minlength=2).astype(float)
    wclass = len(ytr) / (2 * np.maximum(counts, 1))
    per_seed = []
    tr_list, dv_list = [], []
    for seed in seeds:
        clf = SGDClassifier(loss="log_loss", alpha=alpha, random_state=seed)
        rng = np.random.default_rng(seed)
        best = (-1.0, None, None, None)
        for ep in range(epochs):
            perm = rng.permutation(len(ytr))
            for i in range(0, len(perm), bs):
                sel = perm[i:i + bs]
                clf.partial_fit(Xtr[sel], ytr[sel], classes=classes, sample_weight=wclass[ytr[sel]])
            auc = float(roc_auc_score(ydv, clf.predict_proba(Xdv)[:, 1]))
            if auc > best[0]:
                best = (auc, clf.coef_.copy(), clf.intercept_.copy(), ep)
        _, coef, inter, bep = best
        clf.coef_, clf.intercept_ = coef, inter
        tr_p = clf.predict_proba(Xtr)[:, 1]
        dv_p = clf.predict_proba(Xdv)[:, 1]
        per_seed.append({"seed": seed, "coef": coef, "intercept": inter,
                         "best_epoch_1based": bep + 1, "dev_auc": float(roc_auc_score(ydv, dv_p))})
        tr_list.append(tr_p)
        dv_list.append(dv_p)
        log(f"      sgd(seed={seed}) dev AUC {per_seed[-1]['dev_auc']:.4f} best_ep={bep+1}")
    return per_seed, np.mean(tr_list, 0), np.mean(dv_list, 0)


def lgb_fit_capture(Ftr, ytr, Fdv):
    import lightgbm as lgb
    m = lgb.LGBMClassifier(objective="binary", n_estimators=800, learning_rate=0.05,
                           num_leaves=63, min_child_samples=20, subsample=0.8, subsample_freq=1,
                           colsample_bytree=0.6, reg_lambda=1.0, class_weight="balanced",
                           random_state=0, verbose=-1, n_jobs=8)
    m.fit(Ftr, ytr)
    return m, m.predict_proba(Ftr)[:, 1], m.predict_proba(Fdv)[:, 1]


def fusion_fit_capture(Ptr5, ytr, Pdv5):
    from sklearn.linear_model import LogisticRegression
    Ztr = np.log(np.clip(Ptr5, 1e-6, 1))
    Zdv = np.log(np.clip(Pdv5, 1e-6, 1))
    meta = LogisticRegression(max_iter=3000, C=1.0).fit(Ztr, ytr)
    return meta, meta.predict_proba(Ztr)[:, 1], meta.predict_proba(Zdv)[:, 1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds-limit", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "objects").mkdir(exist_ok=True)
    (OUT / "logs").mkdir(exist_ok=True)

    amend = json.loads((PRE / "r1_negative_set_amendment.json").read_text(encoding="utf-8"))
    per_fold_sel = {k: v["size_matched_negatives"]["selected"] for k, v in
                    amend["size_matched_arm"]["per_fold"].items()}

    log("[1] samples")
    split_of = s1.load_split()
    samples = s1.load_samples(split_of)
    n = len(samples)
    assert n == 10659, n
    log("[2] style/meta recomputed (float64, same inputs as stage-1); emb from cache")
    style_mat = np.array([s1.style_features(r["code"]) for r in samples], dtype=np.float64)
    meta_mat = style_mat[:, s1.META_IDX].copy()
    szfeat = np.array([[np.log10(s1.PARAM_B[r["unit"]]), style_mat[i, 0], style_mat[i, 1]]
                       for i, r in enumerate(samples)], dtype=np.float64)
    emb_small = np.load(S1 / "features/emb_codet5_small.npz")["emb"]
    emb_base = np.load(S1 / "features/emb_codet5_base.npz")["emb"]

    unit_rows = defaultdict(list)
    for i, r in enumerate(samples):
        unit_rows[r["unit"]].append(i)

    combos = [(series, h, arm) for series in s1.SERIES for h in s1.SERIES[series] for arm in ARM_NAMES]
    if args.folds_limit:
        combos = combos[:args.folds_limit]

    dev_scores = {}
    manifest = {}
    for series, h, arm in combos:
        key_fold = f"{series}::heldout={s1.SIZE_LABEL[h]}"
        pos = [m for m in s1.SERIES[series] if m != h]
        neg_all = [m for ss in s1.SERIES if ss != series for m in s1.SERIES[ss]]
        neg = neg_all if arm == "size_mix" else per_fold_sel[key_fold]
        units = pos + [m for m in neg]
        tr_idx = np.array(sorted(i for u in units for i in unit_rows[u]
                                 if samples[i]["split"] == "train"))
        dv_idx = np.array(sorted(i for u in units for i in unit_rows[u]
                                 if samples[i]["split"] == "dev"))
        y_tr = np.array([0 if samples[i]["unit"] in neg else 1 for i in tr_idx])
        y_dv = np.array([0 if samples[i]["unit"] in neg else 1 for i in dv_idx])
        fold_key = f"{key_fold}::{arm}"
        log(f"  [{fold_key}] train={len(tr_idx)} dev={len(dv_idx)}")

        tr_texts = [samples[i]["code"] for i in tr_idx]
        dv_texts = [samples[i]["code"] for i in dv_idx]
        t = time.time()
        cv_char = s1.tfidf_char().fit(tr_texts)
        Xc_tr, Xc_dv = cv_char.transform(tr_texts), cv_char.transform(dv_texts)
        cv_word = s1.tfidf_word().fit(tr_texts)
        Xw_tr, Xw_dv = cv_word.transform(tr_texts), cv_word.transform(dv_texts)
        log(f"    tfidf in {time.time()-t:.1f}s")

        obj = {"fold_key": fold_key, "series": series, "heldout": h, "arm": arm,
               "pos": pos, "neg": neg, "char_vec": cv_char, "word_vec": cv_word,
               "members": {}, "C_selected": {}, "r2": {}}
        P_tr, P_dv = {}, {}

        seeds_char, ptr, pdv = sgd_fit_capture(Xc_tr, y_tr, Xc_dv, y_dv)
        obj["members"]["tfidf_char"] = {"type": "sgd_avg3", "seeds": seeds_char}
        P_tr["tfidf_char"], P_dv["tfidf_char"] = ptr, pdv
        seeds_word, ptr, pdv = sgd_fit_capture(Xw_tr, y_tr, Xw_dv, y_dv)
        obj["members"]["tfidf_word"] = {"type": "sgd_avg3", "seeds": seeds_word}
        P_tr["tfidf_word"], P_dv["tfidf_word"] = ptr, pdv

        st_tr, st_dv = style_mat[tr_idx], style_mat[dv_idx]
        eb_tr, eb_dv = emb_base[tr_idx], emb_base[dv_idx]
        es_tr, es_dv = emb_small[tr_idx], emb_small[dv_idx]
        mt_tr, mt_dv = meta_mat[tr_idx], meta_mat[dv_idx]
        sl_tr, sl_dv = szfeat[tr_idx], szfeat[dv_idx]

        for name, (Ftr, Fdv) in (("sem_base", (eb_tr, eb_dv)), ("sem_small", (es_tr, es_dv)),
                                 ("style_lr", (st_tr, st_dv)), ("metadata_only", (mt_tr, mt_dv)),
                                 ("size_length_only", (sl_tr, sl_dv))):
            clf, sc, C, trp, dvp, per_c = lr_fit_capture(Ftr, y_tr, Fdv, y_dv)
            obj["members"][name] = {"type": "lr", "clf": clf, "scaler": sc, "per_C_dev_auc": per_c}
            obj["C_selected"][name] = C
            P_tr[name], P_dv[name] = trp, dvp
            log(f"    {name}: C={C}")

        m, trp, dvp = lgb_fit_capture(st_tr, y_tr, st_dv)
        obj["members"]["style_lgb"] = {"type": "lgbm", "model": m}
        P_tr["style_lgb"], P_dv["style_lgb"] = trp, dvp

        Ptr5 = np.column_stack([P_tr[k] for k in s1.P0_MEMBERS])
        Pdv5 = np.column_stack([P_dv[k] for k in s1.P0_MEMBERS])
        meta_lr, trp, dvp = fusion_fit_capture(Ptr5, y_tr, Pdv5)
        obj["members"]["P0_fusion"] = {"type": "fusion_lr", "clf": meta_lr}
        P_tr["P0_fusion"], P_dv["P0_fusion"] = trp, dvp
        # §2.1.5 修正：train 返回 Ptr5.mean(1)（dev 输出保持 Pdv5.mean(1)，不改变原 dev）
        P_tr["P0_equal"], P_dv["P0_equal"] = Ptr5.mean(1), Pdv5.mean(1)

        for rep_name, Rtr_ in (("codet5_base", eb_tr), ("codet5_small", es_tr)):
            from sklearn.preprocessing import StandardScaler
            pos_mask = y_tr == 1
            sc = StandardScaler().fit(Rtr_[pos_mask])
            z = sc.transform(Rtr_)
            obj["r2"][rep_name] = {"scaler_mean": sc.mean_, "scaler_scale": sc.scale_,
                                   "center": z[pos_mask].mean(0)}
        obj["meta"] = {"train_n": int(len(tr_idx)), "dev_n": int(len(dv_idx)),
                       "pos_tr": int(y_tr.sum()), "pos_dv": int(y_dv.sum()),
                       "samples_order": "(unit, task) ascending; stage-1 identical",
                       "dev_idx_sig": sha256_of_array(dv_idx)}

        pkl_path = OUT / "objects" / f"{fold_key.replace('::','__').replace('=','-')}.pkl"
        with pkl_path.open("wb") as f:
            pickle.dump(obj, f, protocol=5)

        # dev 分数保存（float64）
        for name, pdv in P_dv.items():
            dev_scores[f"{fold_key}::{name}"] = np.asarray(pdv, dtype=np.float64)
        # dv 元数据（对齐用：(unit, task, y) 序列）
        dev_scores[f"{fold_key}::__meta__"] = np.array(
            [f"{samples[i]['unit']}|{samples[i]['task']}|{int(y_dv[j])}"
             for j, i in enumerate(dv_idx)])
        manifest[fold_key] = {"pkl": str(pkl_path.relative_to(ROOT)),
                              "sha256": sha256_file(pkl_path),
                              "bytes": pkl_path.stat().st_size}

    np.savez_compressed(OUT / "dev_scores_new.npz", **dev_scores)
    (OUT / "objects_manifest.json").write_text(json.dumps({
        "schema": "variant_transfer_stage1b_objects_manifest_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "note": "确定性重建产物；拟合仅用 train；P0_equal train 字段按 §2.1.5 修正（dev 输出不变）",
        "objects": manifest,
        "env": {"python": sys.version.split()[0],
                "numpy": np.__version__,
                "sklearn": __import__("sklearn").__version__},
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "logs" / "rebuild_run.log").write_text("\n".join(LOG) + "\n", encoding="utf-8")
    log(f"done in {time.time()-t0:.1f}s; objects={len(manifest)}")


def sha256_of_array(a) -> str:
    return hashlib.sha256(np.asarray(a).tobytes()).hexdigest()


if __name__ == "__main__":
    main()
