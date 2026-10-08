"""实验 A 汇总：共享 task-cluster bootstrap（500）+ 判词 + 报告。

读关系端点平衡目录的 local/eval_scores_fold*.npz（48 折），主 CI=共享 task-cluster bootstrap：
每次所有折使用同一有放回 task 序列，各折重算加权指标后按冻结平均（48 折均值）汇总；
配对 Δ（residual−compose 等）在同一 bootstrap 内计算。附 unweighted 均值、按 heldout
成员/系列的描述性统计、endpoint sanity 检查。

输出：artifacts/relation_endpoint_balanced_2026-10-08/{metrics_final.json, report.md}
"""
from __future__ import annotations

import glob
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def weighted_auc(y, s, w):
    """向量化加权 AUC（含 tie=0.5 处理；与段循环版等价）。"""
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


def weighted_ap(y, s, w):
    order = np.argsort(-s, kind="mergesort")
    ys = y[order]; ws = w[order]
    Wp = ws[ys == 1].sum()
    cum_tp = np.cumsum(np.where(ys == 1, ws, 0.0))
    cum_fp = np.cumsum(np.where(ys == 0, ws, 0.0))
    prec = cum_tp / (cum_tp + cum_fp + 1e-12)
    return float(np.where(ys == 1, ws * prec, 0.0).sum() / (Wp + 1e-12))


def unweighted_auc(y, s):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s))

OUT = ROOT / "d-det/artifacts/relation_endpoint_balanced_2026-10-08"
PLAN = ROOT / "d-det/artifacts/endpoint_balanced_relation_plan_2026-10-08"
SEED = 20261008
NB = 500
LOG: list[str] = []


def log(m):
    print(m, flush=True)
    LOG.append(m)


def main():
    t0 = time.time()
    plan = json.loads((PLAN / "endpoint_balanced_plan.json").read_text(encoding="utf-8"))
    folds = plan["folds"]
    data = {}
    for fid in range(len(folds)):
        p = OUT / "local" / f"eval_scores_fold{fid}.npz"
        if not p.exists():
            log(f"[stats] missing fold {fid} — abort")
            return
        z = np.load(p)
        y = z["y"]; w = z["w"]; tc = z["taskcol"].astype(int)
        reads = {k[2:]: z[k] for k in z.files if k.startswith("s_")}
        idx_by = {q: np.where(tc == q)[0] for q in np.unique(tc)}
        data[fid] = {"y": y, "w": w, "reads": reads, "idx_by": idx_by}
    log(f"[stats] loaded {len(data)} folds")

    read_names = ["compose", "q_sym", "P0_compose", "P0_pair", "residual",
                  "endpoint_anchor_size", "endpoint_partner_size", "endpoint_unorm_diff"]

    def fold_metrics(fid, idx=None):
        d = data[fid]
        y = d["y"][idx] if idx is not None else d["y"]
        w = d["w"][idx] if idx is not None else d["w"]
        out = {}
        for k in read_names:
            s = d["reads"][k]
            if idx is not None:
                s = s[idx]
            out[k] = weighted_auc(y, s, w)
        return out

    # 点估计（全量）
    point = {fid: fold_metrics(fid) for fid in data}
    mean_point = {k: float(np.mean([point[f][k] for f in data])) for k in read_names}
    unweighted_mean = {}
    for k in read_names:
        vals = []
        for fid in data:
            d = data[fid]
            vals.append(unweighted_auc(d["y"], d["reads"][k]))
        unweighted_mean[k] = float(np.mean(vals))

    # 共享 task-cluster bootstrap
    uniq_tasks = sorted({q for fid in data for q in data[fid]["idx_by"]})
    rng = np.random.default_rng(SEED)
    boot = {k: [] for k in read_names}
    deltas = {k: [] for k in ("residual_minus_compose", "residual_minus_P0_compose", "residual_minus_P0_pair",
                              "q_minus_compose")}
    n_done = 0
    for b in range(NB):
        pick = rng.choice(uniq_tasks, size=len(uniq_tasks), replace=True)
        per_fold = {}
        for fid in data:
            idx_by = data[fid]["idx_by"]
            idx = np.concatenate([idx_by[q] for q in pick if q in idx_by])
            per_fold[fid] = fold_metrics(fid, idx)
        for k in read_names:
            boot[k].append(float(np.mean([per_fold[f][k] for f in data])))
        deltas["residual_minus_compose"].append(float(np.mean([per_fold[f]["residual"] - per_fold[f]["compose"] for f in data])))
        deltas["residual_minus_P0_compose"].append(float(np.mean([per_fold[f]["residual"] - per_fold[f]["P0_compose"] for f in data])))
        deltas["residual_minus_P0_pair"].append(float(np.mean([per_fold[f]["residual"] - per_fold[f]["P0_pair"] for f in data])))
        deltas["q_minus_compose"].append(float(np.mean([per_fold[f]["q_sym"] - per_fold[f]["compose"] for f in data])))
        n_done += 1
        if n_done % 100 == 0:
            log(f"[stats] bootstrap {n_done}/{NB} ({time.time()-t0:.0f}s)")

    def ci(vals):
        return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]

    final = {"schema": "f2_balanced_final_v1", "generated_utc": datetime.now(timezone.utc).isoformat(),
             "n_folds": len(data), "bootstrap": {"unit": "task-cluster-shared", "n": NB, "seed": SEED},
             "weighted_mean": {k: {"point": mean_point[k], "ci95": ci(boot[k])} for k in read_names},
             "unweighted_mean": unweighted_mean,
             "deltas": {k: {"mean": float(np.mean(v)), "ci95": ci(v)} for k, v in deltas.items()},
             "endpoint_sanity": {
                 "anchor_mean": mean_point["endpoint_anchor_size"], "partner_mean": mean_point["endpoint_partner_size"],
                 "passed": bool(abs(mean_point["endpoint_anchor_size"] - 0.5) < 0.02 and abs(mean_point["endpoint_partner_size"] - 0.5) < 0.02)},
             "per_fold_point": {str(fid): point[fid] for fid in data}}

    # 按 heldout 成员/系列描述性统计
    by_member = {}
    for fid in data:
        for s, h in folds[fid]["heldout"].items():
            by_member.setdefault(h, {"series": s, "residual": [], "compose": []})
            by_member[h]["residual"].append(point[fid]["residual"])
            by_member[h]["compose"].append(point[fid]["compose"])
    final["by_heldout_member"] = {k: {"series": v["series"], "n_folds": len(v["residual"]),
                                      "residual_mean": float(np.mean(v["residual"])),
                                      "compose_mean": float(np.mean(v["compose"])),
                                      "delta_mean": float(np.mean(np.array(v["residual"]) - np.array(v["compose"])))}
                                  for k, v in by_member.items()}

    # 判词（§2.6）
    g = final["endpoint_sanity"]
    d_rc = final["deltas"]["residual_minus_compose"]
    verdict = None
    if not g["passed"]:
        verdict = "endpoint_balance_construction_failed"
    else:
        if d_rc["ci95"][0] > 0:
            verdict = "relation_increment_dev_evidence (当前三个已知系列)"
        elif d_rc["mean"] > 0:
            verdict = "relation_increment_positive_point_but_ci_includes_0"
        else:
            verdict = "relation_increment_not_observed"
    final["verdict"] = verdict
    (OUT / "metrics_final.json").write_text(json.dumps(final, ensure_ascii=False, indent=1), encoding="utf-8")

    L = ["# 实验 A：端点平衡关系增量（48 折，train/dev）", "",
         f"- 折数={len(data)}（4x4x3）；共享 task-cluster bootstrap（{NB}，seed {SEED}）", "",
         "| 读出 | 加权 mean | 95% CI | unweighted |", "|---|---|---|---|"]
    for k in read_names:
        wm = final["weighted_mean"][k]
        L.append(f"| {k} | {wm['point']:.4f} | [{wm['ci95'][0]:.4f}, {wm['ci95'][1]:.4f}] | {unweighted_mean[k]:.4f} |")
    L += ["", "## 配对 Δ（同一 bootstrap 内）", ""]
    for k, v in final["deltas"].items():
        L.append(f"- {k}: mean={v['mean']:+.4f} CI95 [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]")
    L += ["", f"## 判词\n- **verdict = `{verdict}`**",
          f"- endpoint sanity：anchor={g['anchor_mean']:.3f} / partner={g['partner_mean']:.3f} "
          f"（应≈0.5，passed={g['passed']}）", "",
          "> weighted=系列平衡的加权总体（pairwise concordance）；unweighted 另列。",
          "> 主比较为配对 Δ（residual−compose 等）；48 折共享 tasks，CI 用共享 task 序列。"]
    (OUT / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (OUT / "logs" / "stats.log").write_text("\n".join(LOG) + "\n", encoding="utf-8")
    log(f"[stats] verdict={verdict}; residual−compose={d_rc['mean']:+.4f} CI={d_rc['ci95']}")


if __name__ == "__main__":
    main()
