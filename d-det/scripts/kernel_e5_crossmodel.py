#!/usr/bin/env python
"""E5（mini）：跨家族"偏好位移"读数的一致性——Qwen r* vs DeepSeek r*。

直接检验 d-det.md §6 速查卡假设："s2 锚定人类偏好结构（跨实验室稳定）"：
若两个独立模型对的 r* 高度相关 → 偏好位移是一个**跨模型可复现的物理量**；
若低相关 → "位移"读数强依赖参照模型对，§3 的免标注原理需附加条件。
样本：E1 同 198 条（seed 固定复现）。输出：runs/kernel_e5/crossmodel.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from kernel_e1_rstar import build_samples, score_model  # noqa: E402

OUT = ROOT / "runs/kernel_e5"
DS = {"base": ROOT / "checkpoints/deepseek-coder-1.3b-base",
      "instruct": ROOT / "checkpoints/deepseek-coder-1.3b-instruct"}


def main() -> int:
    smp = build_samples()
    OUT.mkdir(parents=True, exist_ok=True)
    lp = {}
    for tag, path in DS.items():
        tj, mn = score_model(path, smp["code"])
        lp[tag] = (tj, mn)
        np.savez_compressed(OUT / f"ds_logp_{tag}.npz", total=tj, mean=mn)
        print(f"[e5] DeepSeek-{tag} 打分完成", flush=True)

    r_ds = lp["instruct"][0] - lp["base"][0]
    r_ds_m = lp["instruct"][1] - lp["base"][1]
    e1 = np.load(ROOT / "runs/kernel_e1/rstar_total.npz", allow_pickle=True)
    r_qw, y, lab, gen = e1["r"], e1["y"], e1["label"], e1["generator"]
    assert len(r_ds) == len(r_qw) == len(lab)
    m = lab > 0
    res = {
        "n": int(len(r_ds)), "n_machine": int(m.sum()),
        "corr_all_pearson": round(float(pearsonr(r_ds, r_qw)[0]), 4),
        "corr_machine_pearson": round(float(pearsonr(r_ds[m], r_qw[m])[0]), 4),
        "corr_machine_spearman": round(float(spearmanr(r_ds[m], r_qw[m])[0]), 4),
        "auc_ds_total": round(float(roc_auc_score(y, r_ds)), 4),
        "auc_ds_mean": round(float(roc_auc_score(y, r_ds_m)), 4),
        "auc_qwen_total_ref": round(float(roc_auc_score(y, r_qw)), 4),
    }
    # 族均值一致性（21 族）
    from collections import defaultdict
    gq, gd = defaultdict(list), defaultdict(list)
    for g, yi, a, b in zip(gen, m, r_qw, r_ds):
        if yi:
            gq[str(g)].append(float(a))
            gd[str(g)].append(float(b))
    common = [g for g in gq if g in gd and len(gq[g]) >= 2]
    if len(common) >= 5:
        mq = np.array([np.mean(gq[g]) for g in common])
        md = np.array([np.mean(gd[g]) for g in common])
        res["family_mean_corr"] = {"n_families": len(common),
                                   "pearson": round(float(pearsonr(mq, md)[0]), 4),
                                   "spearman": round(float(spearmanr(mq, md)[0]), 4)}
    (OUT / "crossmodel.json").write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print("[e5] 结果：", json.dumps(res, ensure_ascii=False), flush=True)
    print("[e5] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
