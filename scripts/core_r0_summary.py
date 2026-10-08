"""R0 paired 汇总：u_mlp vs 各基线（single/delta/P0）的 paired delta_mean + CI（共享 picks）。"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import core_r0_eval as ev  # noqa: E402

OUT = ev.OUT
RESULTS = OUT / "results"
BASES = ["single_hc", "delta_pair", "p0_tfidf_char", "p0_tfidf_word", "p0_style_lr",
         "p0_style_lgb", "p0_metadata", "p0_size_length", "late_fusion"]


def paired_delta(y_units_scores, base_scores, models, tasks, seeds=20261008, nb=500):
    """d = (si−sc)；返回 acc_u − acc_b 的 paired bootstrap（task-cluster；共享 picks）。"""
    du = y_units_scores  # (sc, si) tuple? 传入已算的 d
    db = base_scores
    uniq_t, inv_t = np.unique(tasks, return_inverse=True)
    uniq_m, inv_m = np.unique(models, return_inverse=True)

    def acc_mb(d, idx=None):
        if idx is not None:
            d = d[idx]; im = inv_m[idx]
        else:
            im = inv_m
        accs = []
        for g in range(len(uniq_m)):
            sel = im == g
            if sel.sum() == 0:
                continue
            accs.append(float(np.mean(d[sel] > 0) + 0.5 * np.mean(d[sel] == 0)))
        return float(np.mean(accs))

    point = acc_mb(du) - acc_mb(db)
    rng = np.random.default_rng(seeds)
    idx_by = {q: np.where(inv_t == q)[0] for q in range(len(uniq_t))}
    deltas = np.empty(nb)
    for k in range(nb):
        pick = rng.choice(len(uniq_t), size=len(uniq_t), replace=True)
        idx = np.concatenate([idx_by[q] for q in pick])
        deltas[k] = acc_mb(du, idx) - acc_mb(db, idx)
    return {"delta_point": float(point), "delta_mean": float(np.nanmean(deltas)),
            "ci95": [float(np.nanpercentile(deltas, 2.5)), float(np.nanpercentile(deltas, 97.5))],
            "frac_le_0": float(np.mean(deltas <= 0))}


def main():
    out = {"schema": "r0_paired_summary_v1",
           "note": "u_mlp − baseline；paired task-cluster bootstrap 500 (seed 20261008, 共享 picks)；model-balanced pair-acc 口径",
           "domains": {}, "aggregate": {}}
    per_key = defaultdict(list)
    for p in sorted(RESULTS.glob("*.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        if r.get("rep", "base") != "base":
            continue
        key = f"{r['split']}_{r['fold']}"
        for label, dv in r["domains"].items():
            if label.startswith("_") or "metrics" not in dv:
                continue
            met = dv["metrics"]
            if "u_mlp" not in met or "error" in met["u_mlp"]:
                continue
            # 载分数
            npz_p = OUT / "local" / f"scores_{r['split']}_{r['fold']}_base_{label}.npz"
            if not npz_p.exists():
                continue
            npz = np.load(npz_p)
            models = np.array(dv["axis_models"]); tasks = np.array(dv["axis_tasks"])
            d_u = npz["%s::u_mlp::i" % label] - npz["%s::u_mlp::c" % label]
            dom = {}
            for b in BASES:
                k = "%s::%s" % (label, b)
                if f"{k}::i" not in npz.files:
                    continue
                d_b = npz[f"{k}::i"] - npz[f"{k}::c"]
                dom[b] = paired_delta(d_u, d_b, models, tasks)
            out["domains"][key] = {"label": label, "n_units": met["u_mlp"]["n_units"],
                                   "u_mlp_mb": met["u_mlp"]["pair_acc_mb"], "deltas": dom}
            for b, dd in dom.items():
                per_key[(label.split("_")[0], b)].append(dd["delta_point"])
    # 跨折聚合
    for (sp, b), vals in sorted(per_key.items()):
        out["aggregate"][f"{sp}|u_mlp−{b}"] = {"mean_delta_point_over_kernels": float(np.mean(vals)),
                                               "min": float(np.min(vals)), "max": float(np.max(vals)),
                                               "k": len(vals)}
    (OUT / "summary_paired.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out["aggregate"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
