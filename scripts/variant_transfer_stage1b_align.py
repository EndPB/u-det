"""stage-1b 对齐与重载校验（§2.1.4 / §3.3）。

1) dev_scores_new（重建内存分数） vs stage-1 predictions/*.csv.gz：|Δ|≤5.1e-7；
2) 保存包重载 → 独立预测 dev（只 transform/predict） vs dev_scores_new：一致性核对；
3) 重算点指标（AUROC/AP/task-macro） vs stage-1 train_dev_metrics.json：|Δ|≤1e-10。

输出：variant_transfer_stage1b_2026-10-08/replay_check.json
"""
from __future__ import annotations

import csv
import gzip
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
TOL_CSV = 5.1e-7
TOL_POINT = 1e-10
TOL_RELOAD = 1e-12


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def predict_dev_from_object(obj, samples, style_mat, meta_mat, szfeat, emb_small, emb_base,
                            unit_rows, per_fold_sel):
    """只 transform/predict；与 stage-1 完全相同的 dev 行顺序。"""
    series, h, arm = obj["series"], obj["heldout"], obj["arm"]
    neg = obj["neg"]
    units = [m for m in s1.SERIES[series] if m != h] + [m for m in neg]
    dv_idx = np.array(sorted(i for u in units for i in unit_rows[u]
                             if samples[i]["split"] == "dev"))
    dv_texts = [samples[i]["code"] for i in dv_idx]
    Xc = obj["char_vec"].transform(dv_texts)
    Xw = obj["word_vec"].transform(dv_texts)
    st = style_mat[dv_idx]
    eb = emb_base[dv_idx]
    es = emb_small[dv_idx]
    mt = meta_mat[dv_idx]
    sl = szfeat[dv_idx]
    y = np.array([0 if samples[i]["unit"] in neg else 1 for i in dv_idx])
    scores = {}
    for name, X in (("tfidf_char", Xc), ("tfidf_word", Xw)):
        ps = []
        for sd in obj["members"][name]["seeds"]:
            d = (X @ sd["coef"].T + sd["intercept"]).ravel()
            ps.append(sigmoid(d))
        scores[name] = np.mean(ps, 0)
    for name, F in (("sem_base", eb), ("sem_small", es), ("style_lr", st),
                    ("metadata_only", mt), ("size_length_only", sl)):
        m = obj["members"][name]
        Z = m["scaler"].transform(F)
        scores[name] = m["clf"].predict_proba(Z)[:, 1]
    scores["style_lgb"] = obj["members"]["style_lgb"]["model"].predict_proba(st)[:, 1]
    P5 = np.column_stack([scores[k] for k in s1.P0_MEMBERS])
    Z5 = np.log(np.clip(P5, 1e-6, 1))
    scores["P0_fusion"] = obj["members"]["P0_fusion"]["clf"].predict_proba(Z5)[:, 1]
    scores["P0_equal"] = P5.mean(1)
    r2s = {}
    for rep, R in (("codet5_base", eb), ("codet5_small", es)):
        o = obj["r2"][rep]
        Z = (R - o["scaler_mean"]) / o["scaler_scale"]
        r2s[rep] = -np.linalg.norm(Z - o["center"], axis=1)
    return dv_idx, y, scores, r2s


def main() -> None:
    t0 = time.time()
    amend = json.loads((PRE / "r1_negative_set_amendment.json").read_text(encoding="utf-8"))
    per_fold_sel = {k: v["size_matched_negatives"]["selected"] for k, v in
                    amend["size_matched_arm"]["per_fold"].items()}
    split_of = s1.load_split()
    samples = s1.load_samples(split_of)
    style_mat = np.array([s1.style_features(r["code"]) for r in samples], dtype=np.float64)
    meta_mat = style_mat[:, s1.META_IDX].copy()
    szfeat = np.array([[np.log10(s1.PARAM_B[r["unit"]]), style_mat[i, 0], style_mat[i, 1]]
                       for i, r in enumerate(samples)], dtype=np.float64)
    emb_small = np.load(S1 / "features/emb_codet5_small.npz")["emb"]
    emb_base = np.load(S1 / "features/emb_codet5_base.npz")["emb"]
    unit_rows = defaultdict(list)
    for i, r in enumerate(samples):
        unit_rows[r["unit"]].append(i)

    dsn = np.load(OUT / "dev_scores_new.npz", allow_pickle=True)
    mj = json.loads((S1 / "train_dev_metrics.json").read_text(encoding="utf-8"))

    csv_vs_new = {}
    reload_vs_new = {}
    points_diff = {}
    worst_csv = (0.0, None)
    worst_point = (0.0, None)
    worst_reload = (0.0, None)
    over_csv = []

    folds = [k for k in mj["folds"]]
    for fold_key in folds:
        meta = dsn[f"{fold_key}::__meta__"]
        keys = [tuple(str(x).split("|")) for x in meta]  # (unit, task, y)
        # --- 1) csv vs new ---
        csv_map = {}
        with gzip.open(S1 / "predictions" / f"{fold_key.replace('::','__')}.csv.gz", "rt",
                       encoding="utf-8") as f:
            for r in csv.DictReader(f):
                csv_map[(r["readout"], r["unit"], r["task"])] = float(r["score"])
        fold_csv = {}
        for readout in s1.P0_MEMBERS + ("sem_small", "metadata_only", "size_length_only",
                                        "P0_fusion", "P0_equal"):
            arr = dsn[f"{fold_key}::{readout}"]
            mx = 0.0
            for j, (u, tk, yy) in enumerate(keys):
                d = abs(float(arr[j]) - csv_map[(readout, u, tk)])
                mx = max(mx, d)
            fold_csv[readout] = mx
            if mx > worst_csv[0]:
                worst_csv = (mx, f"{fold_key}::{readout}")
            if mx > TOL_CSV:
                over_csv.append({"fold": fold_key, "readout": readout, "max_abs_diff": mx})
        csv_vs_new[fold_key] = fold_csv

        # --- 2) reload vs new ---
        obj = pickle.load(open(OUT / "objects" / f"{fold_key.replace('::','__').replace('=','-')}.pkl", "rb"))
        dv_idx, y, scores, r2s = predict_dev_from_object(
            obj, samples, style_mat, meta_mat, szfeat, emb_small, emb_base, unit_rows, per_fold_sel)
        fold_reload = {}
        for readout in ("tfidf_char", "tfidf_word", "sem_base", "sem_small", "style_lr",
                        "style_lgb", "metadata_only", "size_length_only", "P0_fusion", "P0_equal"):
            mx = float(np.max(np.abs(scores[readout] - dsn[f"{fold_key}::{readout}"].astype(np.float64))))
            fold_reload[readout] = mx
            if mx > worst_reload[0]:
                worst_reload = (mx, f"{fold_key}::{readout}")
        reload_vs_new[fold_key] = fold_reload

        # --- 3) points vs metrics ---
        # 注：sem_base/sem_small 的原始分数路径为 float32（嵌入缓存 dtype）；scipy rankdata 对 float32
        # 输入返回 float32 秩→float32 算术。对齐需沿同 dtype 路径，否则产生 ~1e-9 级伪差异。
        fold_points = {}
        for readout in ("tfidf_char", "tfidf_word", "sem_base", "sem_small", "style_lr",
                        "style_lgb", "metadata_only", "size_length_only", "P0_fusion", "P0_equal"):
            arr = dsn[f"{fold_key}::{readout}"].astype(np.float64)
            arr_tm = arr.astype(np.float32) if readout in ("sem_base", "sem_small") else arr
            tasks = [k[1] for k in keys]
            a = s1._auc(y, arr)
            p = s1._ap(y, arr)
            tm = s1.task_macro_auc(y, arr_tm, tasks)
            old = mj["folds"][fold_key]["metrics"][readout]
            da = abs(a - old["auroc"]["point"])
            dp = abs(p - old["ap"]["point"])
            dtm = abs(tm - old["task_macro_auroc"]["point"])
            fold_points[readout] = {"auroc_diff": da, "ap_diff": dp, "task_macro_diff": dtm}
            for v, tag in ((da, "auroc"), (dp, "ap"), (dtm, "task_macro")):
                if v > worst_point[0]:
                    worst_point = (v, f"{fold_key}::{readout}:{tag}")
        points_diff[fold_key] = fold_points
        print(f"  {fold_key}: csv max {max(fold_csv.values()):.2e} reload max {max(fold_reload.values()):.2e} "
              f"points max {max(max(v.values()) for v in fold_points.values()):.2e}", flush=True)

    # R2 radial 复核（reload vs 原 r2 分数重算）
    r2_diff = {}
    for fold_key in folds:
        obj = pickle.load(open(OUT / "objects" / f"{fold_key.replace('::','__').replace('=','-')}.pkl", "rb"))
        dv_idx, y, scores, r2s = predict_dev_from_object(
            obj, samples, style_mat, meta_mat, szfeat, emb_small, emb_base, unit_rows, per_fold_sel)
        # 原 metrics 的 r2 只有指标；用 reload 分数重算 AUROC 与 metrics 对照
        tasks = [samples[i]["task"] for i in dv_idx]
        fold_r2 = {}
        for rep in ("codet5_base", "codet5_small"):
            a = s1._auc(y, r2s[rep])
            old = mj["folds"][fold_key]["r2"][rep]["euclid"]["auroc"]["point"]
            fold_r2[rep] = abs(a - old)
        r2_diff[fold_key] = fold_r2

    out = {
        "schema": "variant_transfer_stage1b_replay_check_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "tolerances": {"csv_vs_new": TOL_CSV, "reload_vs_new": TOL_RELOAD, "points_vs_metrics": TOL_POINT},
        "csv_vs_new": {"per_fold_readout_max": csv_vs_new, "over_tolerance": over_csv,
                       "worst": {"max_abs_diff": worst_csv[0], "where": worst_csv[1]}},
        "reload_vs_new": {"per_fold_readout_max": reload_vs_new,
                          "worst": {"max_abs_diff": worst_reload[0], "where": worst_reload[1]}},
        "points_vs_metrics": {"per_fold_readout": points_diff,
                              "worst": {"abs_diff": worst_point[0], "where": worst_point[1]}},
        "r2_reload_vs_metrics": r2_diff,
        "pass": {
            "csv_vs_new_within_tol": len(over_csv) == 0 and worst_csv[0] <= TOL_CSV,
            "reload_vs_new_within_tol": worst_reload[0] <= TOL_RELOAD,
            "points_within_tol": worst_point[0] <= TOL_POINT,
        },
        "dtype_path_note": "sem_base/sem_small 沿原 float32 路径（rankdata 对 float32 返回 float32 秩）；"
                           "分数本身逐位一致（float32 roundtrip 无损）；其余读出为 float64 路径",
        "runtime_seconds": time.time() - t0,
    }
    (OUT / "replay_check.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("PASS:", json.dumps(out["pass"]))
    print(f"runtime {out['runtime_seconds']:.0f}s")


if __name__ == "__main__":
    main()
