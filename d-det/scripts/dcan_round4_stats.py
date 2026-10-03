#!/usr/bin/env python
"""Round4 A：配对任务级 bootstrap（raw − 各单变量变体；positive = 移除该线索后的性能下降）。

输入：artifacts/acl_dcan_round4/shortcuts_univar/{predictions.npz,metrics.json}
输出：artifacts/acl_dcan_round4/shortcuts_univar/stats.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / "artifacts" / "acl_dcan_round4" / "shortcuts_univar"
B = 2000
VARIANTS = ["ids_only", "strings_only", "comments_only", "ws_only", "all"]


def macro_f1(y, pred, K=6):
    import sklearn.metrics as sm
    return float(sm.f1_score(y, pred, average="macro", labels=list(range(K)), zero_division=0))


def main():
    d = np.load(A / "predictions.npz", allow_pickle=True)
    m = json.loads((A / "metrics.json").read_text())
    y = d["y_test"].astype(int)
    tasks = d["task_id_test"]
    task_list = sorted(set(tasks.tolist()))
    task_idx = {t: np.where(tasks == t)[0] for t in task_list}
    seeds = [p["seed"] for p in m["variants"]["raw"]["per_seed"]]
    rng = np.random.default_rng(42)
    out = {"B": B, "n_tasks": len(task_list), "n_rows": int(len(y)),
           "note": "diff = raw − variant（positive = 去掉该线索后掉分）；配对同任务重采样；"
                   "primary=best-dev 恢复口径；ep5 为 round2 同协议对照（仅均值）",
           "contrasts": {}}
    for v in VARIANTS:
        per_seed = []
        for s in seeds:
            praw = d[f"raw_s{s}_best_probs"].astype(np.float64)
            pv = d[f"{v}_s{s}_best_probs"].astype(np.float64)
            diff0 = macro_f1(y, praw.argmax(1)) - macro_f1(y, pv.argmax(1))
            deltas = np.empty(B)
            for i in range(B):
                samp = rng.choice(len(task_list), len(task_list), replace=True)
                idx = np.concatenate([task_idx[task_list[j]] for j in samp])
                deltas[i] = macro_f1(y[idx], praw[idx].argmax(1)) - macro_f1(y[idx], pv[idx].argmax(1))
            lo, hi = np.percentile(deltas, [2.5, 97.5])
            # ep5 对照
            praw5 = d[f"raw_s{s}_ep5_probs"].astype(np.float64)
            pv5 = d[f"{v}_s{s}_ep5_probs"].astype(np.float64)
            diff5 = macro_f1(y, praw5.argmax(1)) - macro_f1(y, pv5.argmax(1))
            per_seed.append({"seed": s, "diff_best": float(diff0), "ci95_best": [float(lo), float(hi)],
                             "ci_excludes_zero": bool(lo > 0 or hi < 0),
                             "diff_ep5": float(diff5)})
        out["contrasts"][f"raw - {v}"] = {
            "per_seed": per_seed,
            "mean_diff_best": float(np.mean([r["diff_best"] for r in per_seed])),
            "mean_diff_ep5": float(np.mean([r["diff_ep5"] for r in per_seed]))}
    (A / "stats.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    for k, v in out["contrasts"].items():
        print(f"[r4stats] {k}: best {v['mean_diff_best']:+.4f} (ep5 {v['mean_diff_ep5']:+.4f}) | " +
              " ".join(f"s{r['seed']}:{r['diff_best']:+.4f}[{r['ci95_best'][0]:+.4f},{r['ci95_best'][1]:+.4f}]"
                       f"{'*' if r['ci_excludes_zero'] else ''}" for r in v["per_seed"]))


if __name__ == "__main__":
    main()
