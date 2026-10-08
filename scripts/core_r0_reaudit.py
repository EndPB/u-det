"""R0 复核（reaudit）：只用已保存分数（scores_*.npz + axis），不重读文本。

修正（依《R0复核_R1后训练候选与F0F2归因执行指导》§2）：
- task-cluster bootstrap 重数保留：索引按抽签序列逐次拼接；task-macro M_b=(1/|Q|)Σ a(q_j)（按序列,非去重）；
- model-balanced：在每个重采样样本内先 per-model 计再平均；
- 新增 model-cluster bootstrap（model-heldout 主 CI）；
- 汇总键规范：{rep}·{domain}·{readout}·{metric}；
- permutation check：随机翻转 50% 配对方向 → 应回 0.50。

输出：artifacts/r0_protocol_diff_reaudit_2026-10-08/
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

SRC = ROOT / "d-det/artifacts/r0_protocol_diff_2026-10-08"
OUT = ROOT / "d-det/artifacts/r0_protocol_diff_reaudit_2026-10-08"
NB = 500
SEED = 20261008
LOG: list[str] = []


def log(m):
    print(m, flush=True)
    LOG.append(m)


def acc_mb_fast(pos, inv_m, nm):
    """model-balanced：每 model 内 mean(pos)，再 model 平均（重数已在 pos 里）。"""
    sums = np.bincount(inv_m, weights=pos, minlength=nm)
    cnts = np.bincount(inv_m, minlength=nm)
    valid = cnts > 0
    if not valid.any():
        return np.nan
    return float(np.mean(sums[valid] / cnts[valid]))


def acc_tm_series(pos, inv_t):
    """task-macro（序列口径）：a_task[pick].mean() 的逐任务平均（供 bootstrap 内使用）。"""
    sums = np.bincount(inv_t, weights=pos)
    cnts = np.bincount(inv_t)
    with np.errstate(invalid="ignore"):
        a_task = sums / cnts
    return a_task


def run_domain(d_all, models, tasks, picks_task, picks_model):
    """d_all: dict readout -> d 数组。返回每读出的双 cluster 指标。"""
    uniq_m, inv_m = np.unique(models, return_inverse=True)
    uniq_t, inv_t = np.unique(tasks, return_inverse=True)
    nm, nt = len(uniq_m), len(uniq_t)
    idx_by_t = {q: np.where(inv_t == q)[0] for q in range(nt)}
    idx_by_m = {g: np.where(inv_m == g)[0] for g in range(nm)}
    out = {}
    for name, d in d_all.items():
        pos_base = (d > 0).astype(np.float64) + 0.5 * (d == 0).astype(np.float64)
        a_task = acc_tm_series(pos_base, inv_t)
        # task-cluster
        mb_t = np.empty(NB); tm_t = np.empty(NB); eff_t = np.empty(NB)
        for k, pick in enumerate(picks_task):
            idx = np.concatenate([idx_by_t[q] for q in pick])
            mb_t[k] = acc_mb_fast(pos_base[idx], inv_m[idx], nm)
            tm_t[k] = float(np.mean(a_task[pick]))
            eff_t[k] = len(np.unique(pick))
        # model-cluster
        mb_m = np.empty(NB); tm_m = np.empty(NB); eff_m = np.empty(NB)
        for k, pick in enumerate(picks_model):
            idx = np.concatenate([idx_by_m[g] for g in pick])
            mb_m[k] = acc_mb_fast(pos_base[idx], inv_m[idx], nm)
            tm_m[k] = float(np.mean(acc_tm_series(pos_base[idx], inv_t[idx])))
            eff_m[k] = len(np.unique(pick))
        out[name] = {
            "pair_acc_mb": acc_mb_fast(pos_base, inv_m, nm),
            "task_macro_seq": float(np.mean(a_task)),
            "task_cluster": {"ci95_mb": [float(np.nanpercentile(mb_t, 2.5)), float(np.nanpercentile(mb_t, 97.5))],
                             "ci95_tm": [float(np.nanpercentile(tm_t, 2.5)), float(np.nanpercentile(tm_t, 97.5))],
                             "effective_clusters_mean": float(np.mean(eff_t)), "n_clusters": nt},
            "model_cluster": {"ci95_mb": [float(np.nanpercentile(mb_m, 2.5)), float(np.nanpercentile(mb_m, 97.5))],
                              "ci95_tm": [float(np.nanpercentile(tm_m, 2.5)), float(np.nanpercentile(tm_m, 97.5))],
                              "effective_clusters_mean": float(np.mean(eff_m)), "n_clusters": nm},
            "n_units": int(len(d)), "n_models": int(nm), "n_tasks": int(nt),
        }
    return out


def main():
    t0 = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("--permutation", action="store_true", default=True)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "logs").mkdir(exist_ok=True)
    log("[reaudit] load results + scores (no text re-read)")

    domains = {}   # key -> {label, rep, split, fold, axis_models, axis_tasks}
    for p in sorted((SRC / "results").glob("*.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        rep = r.get("rep", "base")
        key = f"{rep}·{r['split']}_{r['fold']}"
        for label, dv in r["domains"].items():
            if label.startswith("_"):
                continue
            npz_p = SRC / "local" / f"scores_{r['split']}_{r['fold']}_{rep}_{label}.npz"
            if not npz_p.exists():
                continue
            domains[f"{rep}·{label}·f{r['fold']}"] = {
                "label": label, "rep": rep, "npz": npz_p,
                "models": dv.get("axis_models"), "tasks": dv.get("axis_tasks"),
                "split": r["split"], "fold": r["fold"]}
    log(f"  domains: {sorted(domains.keys())}")

    metrics = {"schema": "r0_reaudit_metrics_v1",
               "generated_utc": datetime.now(timezone.utc).isoformat(),
               "note": ("task-cluster=主 CI（task 域）；model-cluster 为 model-heldout 主 CI；"
                        "两者交叉敏感性仅作参考，不作为独立证据；500 次重采样非后验概率"),
               "bootstrap": {"n": NB, "seed": SEED, "multiplicity_preserved": True},
               "domains": {}}
    d_store = {}
    for key, dv in sorted(domains.items()):
        npz = np.load(dv["npz"])
        models = np.array(dv["models"]); tasks = np.array(dv["tasks"])
        d_all = {}
        for k in npz.files:
            if not k.endswith("::i"):
                continue
            name = k.split("::")[1]
            ck = f"{dv['label']}::{name}::c"
            if ck not in npz.files:
                continue
            d_all[name] = npz[k] - npz[ck]
        # picks（域内共享；与域样本量无关的固定 seed 序列）
        uniq_t = np.unique(tasks); uniq_m = np.unique(models)
        rng = np.random.default_rng(SEED)
        picks_task = [rng.choice(len(uniq_t), size=len(uniq_t), replace=True) for _ in range(NB)]
        rng2 = np.random.default_rng(SEED)
        picks_model = [rng2.choice(len(uniq_m), size=len(uniq_m), replace=True) for _ in range(NB)]
        met = run_domain(d_all, models, tasks, picks_task, picks_model)
        metrics["domains"][key] = {"label": dv["label"], "rep": dv["rep"],
                                   "split": dv["split"], "fold": dv["fold"], "readouts": met}
        d_store[key] = d_all
        log(f"  [{key}] u_mlp mb={met['u_mlp']['pair_acc_mb']:.4f} "
            f"taskCI={met['u_mlp']['task_cluster']['ci95_mb']} modelCI={met['u_mlp']['model_cluster']['ci95_mb']}")

    # ---- paired summary（规范键）----
    log("[reaudit] paired deltas (u_mlp − baselines; shared picks)")
    BASES = ["single_hc", "delta_pair", "u_linear", "p0_tfidf_char", "p0_tfidf_word",
             "p0_style_lr", "p0_style_lgb", "p0_metadata", "p0_size_length", "late_fusion"]
    summary = {"schema": "summary_paired_reaudit_v2",
               "generated_utc": datetime.now(timezone.utc).isoformat(),
               "key_format": "{representation=base|small}·{domain}·{readout=u_mlp}·{metric=pair_acc_mb|task_macro_seq}·vs={baseline}",
               "entries": {}, "aggregate_by_domain_family": {}}
    fam = {}
    for key, dv in sorted(domains.items()):
        d_all = d_store[key]
        if "u_mlp" not in d_all:
            continue
        npz = np.load(dv["npz"])
        models = np.array(dv["models"]); tasks = np.array(dv["tasks"])
        uniq_t_all, inv_t_all = np.unique(tasks, return_inverse=True)
        uniq_t = uniq_t_all
        rng = np.random.default_rng(SEED)
        picks = [rng.choice(len(uniq_t), size=len(uniq_t), replace=True) for _ in range(NB)]
        uniq_m, inv_m = np.unique(models, return_inverse=True)
        nm = len(uniq_m)
        idx_by_t = {q: np.where(inv_t_all == q)[0] for q in range(len(uniq_t))}
        du = d_all["u_mlp"]
        pos_u = (du > 0).astype(float) + 0.5 * (du == 0).astype(float)
        a_u = acc_tm_series(pos_u, inv_t_all)
        for b in BASES:
            if b not in d_all:
                continue
            db = d_all[b]
            pos_b = (db > 0).astype(float) + 0.5 * (db == 0).astype(float)
            a_b = acc_tm_series(pos_b, inv_t_all)
            pt_mb = acc_mb_fast(pos_u, inv_m, nm) - acc_mb_fast(pos_b, inv_m, nm)
            pt_tm = float(np.mean(a_u) - np.mean(a_b))
            dmb = np.empty(NB); dtm = np.empty(NB)
            for k, pick in enumerate(picks):
                idx = np.concatenate([idx_by_t[q] for q in pick])
                dmb[k] = acc_mb_fast(pos_u[idx], inv_m[idx], nm) - acc_mb_fast(pos_b[idx], inv_m[idx], nm)
                dtm[k] = float(np.mean(a_u[pick]) - np.mean(a_b[pick]))
            ekey = f"{dv['rep']}·{dv['label']}·u_mlp·vs={b}"
            summary["entries"][ekey] = {
                "pair_acc_mb": {"delta_point": float(pt_mb), "delta_mean": float(np.nanmean(dmb)),
                                "ci95": [float(np.nanpercentile(dmb, 2.5)), float(np.nanpercentile(dmb, 97.5))],
                                "frac_le_0": float(np.mean(dmb <= 0))},
                "task_macro_seq": {"delta_point": float(pt_tm), "delta_mean": float(np.nanmean(dtm)),
                                   "ci95": [float(np.nanpercentile(dtm, 2.5)), float(np.nanpercentile(dtm, 97.5))],
                                   "frac_le_0": float(np.mean(dtm <= 0))},
                "n_units": int(len(du)), "n_models": int(nm), "n_tasks": int(len(uniq_t))}
            fk = f"{dv['rep']}·{dv['label'] if dv['rep']=='base' else 'task_dev'}·{b}"
            fam.setdefault(fk, []).append(float(pt_mb))
    for fk, vals in sorted(fam.items()):
        summary["aggregate_by_domain_family"][fk] = {
            "k": len(vals), "mean_delta_point": float(np.mean(vals)),
            "min": float(np.min(vals)), "max": float(np.max(vals))}
    (OUT / "summary_paired_reaudit.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1),
                                                     encoding="utf-8")

    # ---- permutation check ----
    log("[reaudit] permutation check (50% pairs flipped; expect ≈0.50)")
    perm = {"schema": "r0_permutation_check_v1",
            "method": "每域独立：固定 seed 随机选择 50% 单元翻转 d 符号；重算 model-balanced acc（3 个重复）",
            "domains": {}}
    for key, dv in sorted(domains.items()):
        d_all = d_store[key]
        if "u_mlp" not in d_all:
            continue
        models = np.array(dv["models"])
        uniq_m, inv_m = np.unique(models, return_inverse=True)
        nm = len(uniq_m)
        vals = []
        for rep in range(3):
            rng = np.random.default_rng(SEED + rep)
            flip = rng.random(len(d_all["u_mlp"])) < 0.5
            d = d_all["u_mlp"].copy()
            d[flip] *= -1
            pos = (d > 0).astype(float) + 0.5 * (d == 0).astype(float)
            vals.append(acc_mb_fast(pos, inv_m, nm))
        perm["domains"][key] = {"flipped_acc_mb": vals, "mean": float(np.mean(vals))}
    perm["max_abs_dev_from_0.5"] = float(max(abs(np.mean(v["flipped_acc_mb"]) - 0.5)
                                             for v in perm["domains"].values()))
    perm["pass"] = bool(perm["max_abs_dev_from_0.5"] <= 0.01)
    (OUT / "permutation_check.json").write_text(json.dumps(perm, ensure_ascii=False, indent=1), encoding="utf-8")

    metrics["runtime_seconds"] = time.time() - t0
    (OUT / "metrics_reaudit.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "logs" / "reaudit_run.log").write_text("\n".join(LOG) + "\n", encoding="utf-8")
    log(f"[reaudit] done in {time.time()-t0:.1f}s; perm_pass={perm['pass']} "
        f"max_dev={perm['max_abs_dev_from_0.5']:.4f}")


if __name__ == "__main__":
    main()
