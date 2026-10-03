#!/usr/bin/env python
"""Round3：配对任务级 bootstrap（同 task 重采样；每 seed 差值 + CI95）。

对比（fuse 输出）：late_fusion vs lf_nosupcon；full_disentangle vs fd_nosemgrl / fd_nofpgrl / fd_noorth。
输入：artifacts/acl_dcan_round3_audit/mlp/{predictions.npz, metrics.json}
输出：artifacts/acl_dcan_round3_audit/stats_paired_bootstrap.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "acl_dcan_round3_audit"
PAIRS = [("late_fusion", "lf_nosupcon"), ("full_disentangle", "fd_nosemgrl"),
         ("full_disentangle", "fd_nofpgrl"), ("full_disentangle", "fd_noorth")]
B = 2000


def macro_f1(y, pred, K=6):
    import sklearn.metrics as sm
    return float(sm.f1_score(y, pred, average="macro", labels=list(range(K)), zero_division=0))


def main():
    d = np.load(OUT / "mlp" / "predictions.npz", allow_pickle=True)
    m = json.loads((OUT / "mlp" / "metrics.json").read_text())
    y = d["y_test"].astype(int)
    tasks = d["task_id_test"]
    seeds = sorted({r["seed"] for r in m["runs"]})
    task_list = sorted(set(tasks.tolist()))
    task_idx = {t: np.where(tasks == t)[0] for t in task_list}
    rng = np.random.default_rng(0)
    res = {"B": B, "n_tasks": len(task_list), "n_rows": int(len(y)), "contrasts": {}}
    for a, b in PAIRS:
        rows_a = []
        for seed in seeds:
            pa = d[f"{a}_s{seed}_fuse"].astype(np.float64)
            pb = d[f"{b}_s{seed}_fuse"].astype(np.float64)
            diff0 = macro_f1(y, pa.argmax(1)) - macro_f1(y, pb.argmax(1))
            deltas = np.empty(B)
            for i in range(B):
                samp = rng.choice(len(task_list), len(task_list), replace=True)
                idx = np.concatenate([task_idx[task_list[j]] for j in samp])
                deltas[i] = macro_f1(y[idx], pa[idx].argmax(1)) - macro_f1(y[idx], pb[idx].argmax(1))
            lo, hi = np.percentile(deltas, [2.5, 97.5])
            rows_a.append({"seed": seed, "diff": float(diff0), "ci95": [float(lo), float(hi)],
                           "ci_excludes_zero": bool(lo > 0 or hi < 0)})
        res["contrasts"][f"{a} - {b}"] = {"per_seed": rows_a,
                                          "mean_diff": float(np.mean([r["diff"] for r in rows_a]))}
    (OUT / "stats_paired_bootstrap.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    for k, v in res["contrasts"].items():
        print(f"[r3stats] {k}: mean {v['mean_diff']:+.4f} | " +
              " ".join(f"s{r['seed']}:{r['diff']:+.3f}[{r['ci95'][0]:+.3f},{r['ci95'][1]:+.3f}]"
                       for r in v["per_seed"]))


if __name__ == "__main__":
    main()
