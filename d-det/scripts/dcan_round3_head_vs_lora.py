#!/usr/bin/env python
"""Round3：匹配条件 head-only（本轮）vs round2 LoRA 的配对任务级 bootstrap。

输入：
  artifacts/acl_dcan_round3_audit/head_only/predictions.npz（head_only_s{0..2}_probs）
  artifacts/acl_dcan_round2/trainable/predictions.npz（lora_s{0..2}_probs）
  y_test / task_id_test 取自 head_only predictions（同一 test 行序；与 round2 一致性
  由 checkpoint 复算校验确认）。
输出：artifacts/acl_dcan_round3_audit/head_vs_lora.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "acl_dcan_round3_audit"
B = 2000


def macro_f1(y, pred, K=6):
    import sklearn.metrics as sm
    return float(sm.f1_score(y, pred, average="macro", labels=list(range(K)), zero_division=0))


def main():
    dh = np.load(OUT / "head_only" / "predictions.npz", allow_pickle=True)
    dl = np.load(ROOT / "artifacts" / "acl_dcan_round2" / "trainable" / "predictions.npz",
                 allow_pickle=True)
    y = dh["y_test"].astype(int)
    tasks = dh["task_id_test"]
    n = len(y)
    assert all(dl[f"lora_s{s}_probs"].shape[0] == n for s in range(3)), "行数不一致"
    task_list = sorted(set(tasks.tolist()))
    task_idx = {t: np.where(tasks == t)[0] for t in task_list}
    rng = np.random.default_rng(1)
    per_seed = []
    for s in range(3):
        ph = dh[f"head_only_s{s}_probs"].astype(np.float64)
        pl = dl[f"lora_s{s}_probs"].astype(np.float64)
        f1h = macro_f1(y, ph.argmax(1))
        f1l = macro_f1(y, pl.argmax(1))
        diff0 = f1h - f1l
        deltas = np.empty(B)
        for i in range(B):
            samp = rng.choice(len(task_list), len(task_list), replace=True)
            idx = np.concatenate([task_idx[task_list[j]] for j in samp])
            deltas[i] = macro_f1(y[idx], ph[idx].argmax(1)) - macro_f1(y[idx], pl[idx].argmax(1))
        lo, hi = np.percentile(deltas, [2.5, 97.5])
        per_seed.append({"seed": s, "head_only_f1": f1h, "lora_f1": f1l,
                         "diff_head_minus_lora": float(diff0),
                         "ci95": [float(lo), float(hi)],
                         "ci_excludes_zero": bool(lo > 0 or hi < 0)})
    res = {"B": B, "n_tasks": len(task_list), "n_rows": n, "per_seed": per_seed,
           "mean_head_f1": float(np.mean([r["head_only_f1"] for r in per_seed])),
           "std_head_f1": float(np.std([r["head_only_f1"] for r in per_seed])),
           "mean_lora_f1": float(np.mean([r["lora_f1"] for r in per_seed])),
           "std_lora_f1": float(np.std([r["lora_f1"] for r in per_seed])),
           "mean_diff": float(np.mean([r["diff_head_minus_lora"] for r in per_seed])),
           "note": "配对任务级 bootstrap（同任务同重采样索引）；LoRA 数字使用 round2 "
                   "保存的逐 seed 概率（经 checkpoint 复算逐位校验）；两方案 encoder/"
                   "tokenize/线性头/CE/采样/head-lr/预算一致，差异仅在 adapter。"}
    (OUT / "head_vs_lora.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(f"[r3hvl] head-only {res['mean_head_f1']:.4f}±{res['std_head_f1']:.4f} vs "
          f"LoRA {res['mean_lora_f1']:.4f}±{res['std_lora_f1']:.4f} | mean diff "
          f"{res['mean_diff']:+.4f}")
    for r in per_seed:
        print(f"  s{r['seed']}: head {r['head_only_f1']:.4f} lora {r['lora_f1']:.4f} "
              f"diff {r['diff_head_minus_lora']:+.4f} [{r['ci95'][0]:+.4f},{r['ci95'][1]:+.4f}]"
              f"{' *' if r['ci_excludes_zero'] else ''}")


if __name__ == "__main__":
    main()
