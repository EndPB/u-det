"""stage-2：一次性 test 评分（《stage1修复冻结与stage2单次评测指导》§4）。

- freeze-config：读取前生成配置（提交锚定）；
- evaluate：freeze_validation 全过后 → 写 ledger started → 读 1,881 test 行（一次）
  → 一次编码共用特征 → 22 包只 transform/predict → 统计（pooled AUROC/AP/正类率、
  task-macro（multiplicity 修正）、task-cluster bootstrap 500、seed 20261008 共享抽样序列、
  paired delta、seen-reference、控制 delta、分层）→ ledger finished → 关闭 test 开关。

无任何 fit/partial_fit；加载/评分失败即中止（不重训）。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import variant_transfer_stage1_execute as s1  # noqa: E402

OUT = ROOT / "d-det/artifacts/variant_transfer_stage2_2026-10-08"
B1 = ROOT / "d-det/artifacts/variant_transfer_stage1b_2026-10-08"
PRE = ROOT / "d-det/artifacts/variant_transfer_preflight_2026-10-08"
RECORDS = ROOT / "d-det/data/public_same_task_full_2026-10-07/records.jsonl"
SPLIT_CSV = ROOT / "d-det/artifacts/public_full_receive_2026-10-08/prereg/split_index.csv"
BOOT_SEED = 20261008
N_BOOT = 500
READOUTS10 = ("tfidf_char", "tfidf_word", "sem_base", "sem_small", "style_lr", "style_lgb",
              "metadata_only", "size_length_only", "P0_fusion", "P0_equal")
R2_REPS = ("codet5_base", "codet5_small")
N_TEST_ROWS = 1881


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def ci(arr):
    return [float(np.nanpercentile(arr, 2.5)), float(np.nanpercentile(arr, 97.5))]


def task_terms(y, s, tasks_d, uniq_t):
    a = np.empty(len(uniq_t))
    for g, t in enumerate(uniq_t):
        sel = tasks_d == t
        yy = y[sel]
        a[g] = s1._fast_auc(yy, s[sel]) if 0 < yy.sum() < sel.sum() else np.nan
    return a


def eval_domain(y, s, task_v, uniq_t, idx_by_t, picks):
    a_t = task_terms(y, s, task_v, uniq_t)
    A = np.empty(len(picks)); P = np.empty(len(picks)); M = np.empty(len(picks))
    for k, pick in enumerate(picks):
        ii = np.concatenate([idx_by_t[uniq_t[q]] for q in pick])
        A[k] = s1._auc(y[ii], s[ii])
        P[k] = s1._ap(y[ii], s[ii])
        M[k] = np.nanmean(a_t[pick])
    return {"point": {"auroc": s1._auc(y, s), "ap": s1._ap(y, s),
                      "task_macro": s1.task_macro_auc(y, s, task_v),
                      "n": int(len(y)), "pos": int(y.sum()), "pos_rate": float(y.mean())},
            "boot": {"auroc": A, "ap": P, "task_macro": M}}


def _sum_delta(d):
    return {"delta_mean": float(np.nanmean(d)), "ci95": ci(d), "frac_le_0": float((d <= 0).mean())}


def _support_gap(h, neg):
    hs = s1.PARAM_B[h]
    gaps = sorted(abs(np.log10(s1.PARAM_B[m]) - np.log10(hs)) for m in neg)
    return {"heldout_size_B": hs, "neg_min_abs_log10_gap": round(gaps[0], 4) if gaps else None,
            "neg_sizes_B": {m: s1.PARAM_B[m] for m in neg},
            "note": "匹配负集基于已见正成员选择，未必与 heldout size 匹配"}


def _jsonable_folds(results):
    out = {}
    for k, v in results.items():
        fold = {"n_heldout": v["n_heldout"], "n_seen": v["n_seen"], "n_neg": v["n_neg"],
                "support_gap": v["support_gap"], "readouts": {}, "r2": {},
                "control_deltas": v["control_deltas"]}
        for r, d in v["readouts"].items():
            item = {}
            for dom in ("heldout", "seen"):
                ev = d[dom]
                item[dom] = {"point": ev["point"],
                             "ci95": {"auroc": ci(ev["boot"]["auroc"]),
                                      "ap": ci(ev["boot"]["ap"]),
                                      "task_macro": ci(ev["boot"]["task_macro"])}}
            item["delta"] = {m: _sum_delta(d["delta_boot"][m]) for m in ("auroc", "ap", "task_macro")}
            fold["readouts"][r] = item
        for r, d in v["r2"].items():
            ev = d["heldout"]
            fold["r2"][r] = {"heldout": {"point": ev["point"],
                                         "ci95": {"auroc": ci(ev["boot"]["auroc"]),
                                                  "task_macro": ci(ev["boot"]["task_macro"])}}}
        out[k] = fold
    return out


def _stratify(results, scores_all, units, tasks, in_hard, combos):
    out = {}
    for series, h, arm in combos:
        if arm != "size_mix":
            continue
        fold_key = f"{series}::heldout={s1.SIZE_LABEL[h]}::{arm}"
        neg = [m for ss in s1.SERIES if ss != series for m in s1.SERIES[ss]]
        idxP = np.where(units == h)[0]; idxN = np.where(np.isin(units, neg))[0]
        s_all = scores_all[fold_key]["P0_fusion"]
        y = np.concatenate([np.ones(len(idxP)), np.zeros(len(idxN))])
        t_all = np.concatenate([tasks[idxP], tasks[idxN]])
        s = np.concatenate([s_all[idxP], s_all[idxN]])
        hard = np.array([in_hard.get(t, 0) for t in t_all])
        layer = {}
        for name, sel in (("in_hard", hard == 1), ("not_hard", hard == 0)):
            yy, ss = y[sel], s[sel]
            layer[name] = {"n": int(sel.sum()), "pos": int(yy.sum()),
                           "auroc": s1._auc(yy, ss) if 0 < yy.sum() < len(yy) else "not_estimable"}
        out[fold_key] = layer
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["freeze-config", "evaluate"], required=True)
    args = ap.parse_args()
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "logs").mkdir(exist_ok=True)

    if args.stage == "freeze-config":
        split_of = s1.load_split()
        test_tasks = sorted([t for t, sp in split_of.items() if sp == "test"])
        cfg = {
            "schema": "variant_transfer_stage2_config_frozen_v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "created_before_any_test_read": True,
            "data": {"records_sha256": "d786667a72f3f3864ea38115fd6ccfe8ac393141146d522673668f3730523210",
                     "members": sorted(s1.PARAM_B), "test_tasks": test_tasks,
                     "n_test_tasks": len(test_tasks), "expected_rows": 11 * len(test_tasks),
                     "split_index_sha256": sha256_file(SPLIT_CSV)},
            "readouts": list(READOUTS10), "r2_scored": list(R2_REPS),
            "r2_cosine": "invalid_zero_center（不评分）",
            "domain_definition": {
                "heldout_test": "positive=该折 heldout variant 的 171 行；negative=该折负集成员（每员 171 行）",
                "seen_test": "positive=目标系列已见成员（3/2 员）；negative 同负集；同评分包同批次不再拟合",
                "delta": "heldout_test − seen_test（共享抽样序列的 paired task CI）"},
            "arms": {"main": "size_mix", "sensitivity": "member-size-matched / length-unweighted"},
            "statistics": {"bootstrap_resamples": N_BOOT, "bootstrap_seed": BOOT_SEED,
                           "shared_pick_sequence_across_folds_arms": True,
                           "aggregation": "先系列内折等权，再系列间等权（CL/Q/DS）",
                           "stratification": {"hard_flag": "split_index.in_hard",
                                              "missing_class_rule": "not_estimable"}},
            "allowed_output_fields": ["fold", "arm", "readout", "unit_id", "task_id", "split", "label",
                                      "score", "code_sha256", "prediction_hash", "feature_hash", "model_hash"],
            "forbidden_outputs": ["代码正文", "题面", "测试用例", "凭据"],
        }
        (OUT / "stage2_config_frozen.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=1),
                                                       encoding="utf-8")
        print("wrote stage2_config_frozen.json")
        return

    # ================= evaluate =================
    fv = json.loads((B1 / "freeze_validation.json").read_text(encoding="utf-8"))
    assert fv["all_pass"], "freeze_validation 未全过，禁止读 test"
    cfg = json.loads((OUT / "stage2_config_frozen.json").read_text(encoding="utf-8"))
    assert cfg["created_before_any_test_read"] is True

    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                            cwd=ROOT).stdout.strip()
    ledger_path = OUT / "test_access_ledger.json"
    ledger = {
        "schema": "test_access_ledger_v2",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "local_timezone": "CST/UTC+8（系统 UTC 为准，不虚构预定时间）",
        "data": {"records_path": str(RECORDS.relative_to(ROOT)),
                 "records_sha256": cfg["data"]["records_sha256"],
                 "expected_test_rows": cfg["data"]["expected_rows"]},
        "split_index_sha256": cfg["data"]["split_index_sha256"],
        "config_frozen_sha256": sha256_file(OUT / "stage2_config_frozen.json"),
        "objects_manifest_sha256": sha256_file(B1 / "objects_manifest.json"),
        "scoring_code_commit": commit,
        "freeze_validation_all_pass": True,
        "note": "started 段于读取任何 test 行之前写入并落盘；finished 段评分结束后追加",
    }
    ledger_path.write_text(json.dumps(ledger, ensure_ascii=False, indent=1), encoding="utf-8")
    print("ledger started:", ledger["started_at_utc"], "commit", commit, flush=True)

    # ---- 读 test（一次）----
    split_of = s1.load_split()
    members = set(s1.PARAM_B)
    test_rows = []
    with RECORDS.open(encoding="utf-8") as f:
        for line in f:
            if "codellama--CodeLlama-" not in line and "Qwen--Qwen2.5-Coder-" not in line \
                    and "deepseek-ai--deepseek-coder-" not in line:
                continue
            d = json.loads(line)
            m = d.get("model_id")
            if m not in members or d.get("subset") != "full" or d.get("generation_mode") != "instruct":
                continue
            if split_of.get(d["task_id"]) != "test":
                continue
            test_rows.append({"unit": m, "task": d["task_id"], "code": d["solution"]})
    test_rows.sort(key=lambda r: (r["unit"], r["task"]))
    assert len(test_rows) == N_TEST_ROWS, len(test_rows)
    texts = [r["code"] for r in test_rows]
    units = np.array([r["unit"] for r in test_rows])
    tasks = np.array([r["task"] for r in test_rows])
    print("test rows:", len(test_rows), flush=True)
    rec_sha = sha256_file(RECORDS)
    assert rec_sha == cfg["data"]["records_sha256"]

    # ---- 特征（一次编码共用）----
    style_mat = np.array([s1.style_features(c) for c in texts], dtype=np.float64)
    meta_mat = style_mat[:, s1.META_IDX].copy()
    szfeat = np.array([[np.log10(s1.PARAM_B[u]), style_mat[i, 0], style_mat[i, 1]]
                       for i, u in enumerate(units)], dtype=np.float64)
    emb_small = s1.encode_codet5(s1.MODEL_SMALL, texts, device_batch=16)
    emb_base = s1.encode_codet5(s1.MODEL_BASE, texts, device_batch=8)
    feat_hash = {"style_sha256": hashlib.sha256(style_mat.tobytes()).hexdigest(),
                 "emb_small_sha256": hashlib.sha256(emb_small.tobytes()).hexdigest(),
                 "emb_base_sha256": hashlib.sha256(emb_base.tobytes()).hexdigest()}
    print("features done", flush=True)

    # ---- 评分（22 包；只 transform/predict）----
    amend = json.loads((PRE / "r1_negative_set_amendment.json").read_text(encoding="utf-8"))
    per_fold_sel = {k: v["size_matched_negatives"]["selected"] for k, v in
                    amend["size_matched_arm"]["per_fold"].items()}
    combos = [(series, h, arm) for series in s1.SERIES for h in s1.SERIES[series] for arm in s1.ARM_NAMES]
    scores_all, seed_scores_all, r2_score_all = {}, {}, {}
    for series, h, arm in combos:
        fold_key = f"{series}::heldout={s1.SIZE_LABEL[h]}::{arm}"
        obj = pickle.load((B1 / "objects" / f"{fold_key.replace('::','__').replace('=','-')}.pkl").open("rb"))
        Xc = obj["char_vec"].transform(texts)
        Xw = obj["word_vec"].transform(texts)
        sc, seeds = {}, {}
        for name, X in (("tfidf_char", Xc), ("tfidf_word", Xw)):
            ps = [sigmoid((X @ sd["coef"].T + sd["intercept"]).ravel()) for sd in obj["members"][name]["seeds"]]
            sc[name] = np.mean(ps, 0)
            seeds[name] = np.vstack(ps)
        for name, F in (("sem_base", emb_base), ("sem_small", emb_small), ("style_lr", style_mat),
                        ("metadata_only", meta_mat), ("size_length_only", szfeat)):
            m = obj["members"][name]
            sc[name] = m["clf"].predict_proba(m["scaler"].transform(F))[:, 1]
        sc["style_lgb"] = obj["members"]["style_lgb"]["model"].predict_proba(style_mat)[:, 1]
        P5 = np.column_stack([sc[k] for k in s1.P0_MEMBERS])
        sc["P0_fusion"] = obj["members"]["P0_fusion"]["clf"].predict_proba(np.log(np.clip(P5, 1e-6, 1)))[:, 1]
        sc["P0_equal"] = P5.mean(1)
        r2 = {}
        for rep, R in (("codet5_base", emb_base), ("codet5_small", emb_small)):
            o = obj["r2"][rep]
            Z = (R - o["scaler_mean"]) / o["scaler_scale"]
            r2[rep] = -np.linalg.norm(Z - o["center"], axis=1)
        scores_all[fold_key] = {k: np.asarray(v, dtype=np.float64) for k, v in sc.items()}
        seed_scores_all[fold_key] = {k: v.astype(np.float64) for k, v in seeds.items()}
        r2_score_all[fold_key] = {k: v.astype(np.float64) for k, v in r2.items()}
        print("  scored", fold_key, flush=True)

    # ---- 统计 ----
    uniq_t = np.unique(tasks)
    rng = np.random.default_rng(BOOT_SEED)
    picks = [rng.choice(len(uniq_t), size=len(uniq_t), replace=True) for _ in range(N_BOOT)]
    in_hard = {}
    with SPLIT_CSV.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            in_hard[r["task_id"]] = int(r["in_hard"])

    results = {}
    for series, h, arm in combos:
        key_fold = f"{series}::heldout={s1.SIZE_LABEL[h]}"
        fold_key = f"{key_fold}::{arm}"
        pos_seen = [m for m in s1.SERIES[series] if m != h]
        neg = per_fold_sel[key_fold] if arm == "size_matched" else \
            [m for ss in s1.SERIES if ss != series for m in s1.SERIES[ss]]
        mask_h = units == h
        mask_seen = np.isin(units, pos_seen)
        mask_neg = np.isin(units, neg)
        fold = {"n_heldout": int(mask_h.sum()), "n_seen": int(mask_seen.sum()), "n_neg": int(mask_neg.sum()),
                "readouts": {}, "r2": {}, "support_gap": _support_gap(h, neg)}

        def _domain(mask_pos):
            idxP = np.where(mask_pos)[0]
            idxN = np.where(mask_neg)[0]
            y = np.concatenate([np.ones(len(idxP)), np.zeros(len(idxN))])
            task_v = np.concatenate([tasks[idxP], tasks[idxN]])
            idx_by_t = {t: np.where(task_v == t)[0] for t in uniq_t}
            return y, task_v, idx_by_t, np.concatenate([idxP, idxN])

        yH, tvH, ibH, rowsH = _domain(mask_h)
        yS, tvS, ibS, rowsS = _domain(mask_seen)
        for readout in READOUTS10:
            s_all = scores_all[fold_key][readout]
            evH = eval_domain(yH, s_all[rowsH], tvH, uniq_t, ibH, picks)
            evS = eval_domain(yS, s_all[rowsS], tvS, uniq_t, ibS, picks)
            fold["readouts"][readout] = {
                "heldout": evH, "seen": evS,
                "delta_boot": {m: evH["boot"][m] - evS["boot"][m] for m in ("auroc", "ap", "task_macro")},
            }
        for rep in R2_REPS:
            s_all = r2_score_all[fold_key][rep]
            evH = eval_domain(yH, s_all[rowsH], tvH, uniq_t, ibH, picks)
            fold["r2"][rep] = {"heldout": evH}
        ctrl = {}
        s_fus = scores_all[fold_key]["P0_fusion"][rowsH]
        a_f = task_terms(yH, s_fus, tvH, uniq_t)
        for cname in ("P0_equal", "size_length_only", "metadata_only"):
            s_c = scores_all[fold_key][cname][rowsH]
            a_c = task_terms(yH, s_c, tvH, uniq_t)
            dA = np.empty(N_BOOT); dM = np.empty(N_BOOT)
            for k, pick in enumerate(picks):
                ii = np.concatenate([ibH[uniq_t[q]] for q in pick])
                dA[k] = s1._auc(yH[ii], s_fus[ii]) - s1._auc(yH[ii], s_c[ii])
                dM[k] = np.nanmean(a_f[pick]) - np.nanmean(a_c[pick])
            ctrl[cname] = {"auroc": _sum_delta(dA), "task_macro": _sum_delta(dM)}
        fold["control_deltas"] = ctrl
        results[fold_key] = fold
        print("  stats", fold_key, flush=True)

    # ---- 汇总（系列内折等权 → 系列间等权；boot 共享 picks）----
    series_names = list(s1.SERIES)
    summary = {}
    for readout in READOUTS10:
        item = {}
        for dom in ("heldout", "seen", "delta"):
            item[dom] = {}
            for metric in ("auroc", "ap", "task_macro"):
                per_series = []
                for series in series_names:
                    fkeys = [k for k in results if k.startswith(series)]
                    if dom == "delta":
                        arrs = [results[k]["readouts"][readout]["delta_boot"][metric] for k in fkeys]
                    else:
                        arrs = [results[k]["readouts"][readout][dom]["boot"][metric] for k in fkeys]
                    per_series.append(np.mean(arrs, axis=0))
                total = np.mean(per_series, axis=0)
                item[dom][metric] = {"mean": float(np.nanmean(total)), "ci95": ci(total)}
        summary[readout] = item

    strat = _stratify(results, scores_all, units, tasks, in_hard, combos)

    # ---- 输出 ----
    payload = {}
    for k, d in scores_all.items():
        for r, v in d.items():
            payload[f"{k}::{r}"] = v
        for r, v in r2_score_all[k].items():
            payload[f"{k}::R2_{r}"] = v
        for r, m in seed_scores_all[k].items():
            for i in range(m.shape[0]):
                payload[f"{k}::{r}::seed{i}"] = m[i]
    npz_path = OUT / "test_scores.npz"
    np.savez_compressed(npz_path, **payload)
    pred_hash = {"test_scores_npz_sha256": sha256_file(npz_path)}
    with (OUT / "test_rows_index.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["unit_id", "task_id", "split", "in_hard"])
        for i, u in enumerate(units):
            w.writerow([u, tasks[i], "test", in_hard.get(tasks[i], 0)])

    metrics_out = {
        "schema": "variant_transfer_stage2_metrics_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_status": "server_reconstruction_only",
        "original_bundle_verified": False,
        "claims_of_byte_identity": "forbidden",
        "test_read": True, "test_read_authority": "《stage1修复冻结与stage2单次评测指导》§4",
        "n_rows_scored": len(test_rows), "n_test_tasks": int(len(uniq_t)),
        "readouts": list(READOUTS10), "r2_scored": list(R2_REPS),
        "folds": _jsonable_folds(results),
        "summary": summary,
        "stratification": strat,
        "support_gap": {k: results[k]["support_gap"] for k in results},
        "features_hashes": feat_hash, "predictions_hashes": pred_hash,
        "runtime_seconds": time.time() - t0,
    }
    (OUT / "stage2_metrics.json").write_text(json.dumps(metrics_out, ensure_ascii=False, indent=1),
                                             encoding="utf-8")
    (OUT / "failures_and_invalids.json").write_text(json.dumps({
        "schema": "variant_transfer_stage2_failures_invalids_v1",
        "r2_cosine": {"status": "invalid_zero_center", "scored": False},
        "load_failures": [],
        "notes": ["bootstrap 频率非后验概率；CI 仅反映固定模型下的任务变异",
                  "arm 差异为不同负域敏感性，非同样本读出对比",
                  "size_length_only=oracle_size_length_control；metadata_only=code_layout_control"],
    }, ensure_ascii=False, indent=1), encoding="utf-8")

    ledger["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    ledger["n_test_rows_read"] = len(test_rows)
    ledger["features_hashes"] = feat_hash
    ledger["predictions_hashes"] = pred_hash
    ledger_path.write_text(json.dumps(ledger, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "execution_switches_stage2.json").write_text(json.dumps({
        "schema": "variant_transfer_execution_switches_stage2_v1",
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "test_read_allowed": False, "training_allowed": False, "generation_allowed": False,
        "authority": "test 单读完成（§4）；读后关闭；此后仅可对保存分数重算预声明统计，不得再选模型",
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print("stage-2 done in", round(time.time() - t0, 1), "s")


if __name__ == "__main__":
    main()
