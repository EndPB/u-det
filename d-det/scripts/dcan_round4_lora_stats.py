#!/usr/bin/env python
"""Round4 B：LoRA 预算扩展 vs round2 上限12ep 的配对任务级 bootstrap（+ 对 TF-IDF 对照）。

输入：
  artifacts/acl_dcan_round4/lora_extend/predictions.npz（lora_ext_s{0..2}_probs）
  artifacts/acl_dcan_round2/trainable/predictions.npz（lora_s{0..2}_probs）
  artifacts/acl_dcan_round1/p0/dcan/dcan_predictions.npz（char TF-IDF probs，对照）
输出：artifacts/acl_dcan_round4/lora_extend/stats.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / "artifacts" / "acl_dcan_round4" / "lora_extend"
B = 2000


def macro_f1(y, pred, K=6):
    import sklearn.metrics as sm
    return float(sm.f1_score(y, pred, average="macro", labels=list(range(K)), zero_division=0))


def main():
    d = np.load(A / "predictions.npz", allow_pickle=True)
    d2 = np.load(ROOT / "artifacts" / "acl_dcan_round2" / "trainable" / "predictions.npz", allow_pickle=True)
    y = d["y_test"].astype(int)
    tasks = d["task_id_test"]
    task_list = sorted(set(tasks.tolist()))
    task_idx = {t: np.where(tasks == t)[0] for t in task_list}
    rng = np.random.default_rng(7)
    out = {"B": B, "n_tasks": len(task_list), "n_rows": int(len(y)), "contrasts": {},
           "note": "diff = extended − baseline（positive = 延长预算后提升）；同任务配对重采样"}
    for s in range(3):
        pe = d[f"lora_ext_s{s}_probs"].astype(np.float64)
        p2 = d2[f"lora_s{s}_probs"].astype(np.float64)
        diff0 = macro_f1(y, pe.argmax(1)) - macro_f1(y, p2.argmax(1))
        deltas = np.empty(B)
        for i in range(B):
            samp = rng.choice(len(task_list), len(task_list), replace=True)
            idx = np.concatenate([task_idx[task_list[j]] for j in samp])
            deltas[i] = macro_f1(y[idx], pe[idx].argmax(1)) - macro_f1(y[idx], p2[idx].argmax(1))
        lo, hi = np.percentile(deltas, [2.5, 97.5])
        out["contrasts"][f"extended_s{s} - round2_s{s}"] = {
            "diff": float(diff0), "ci95": [float(lo), float(hi)],
            "excludes_zero": bool(lo > 0 or hi < 0),
            "extended_f1": macro_f1(y, pe.argmax(1)), "round2_f1": macro_f1(y, p2.argmax(1))}
    # 对 TF-IDF（round1 预测）对照（逐 seed）
    d1 = np.load(ROOT / "artifacts" / "acl_dcan_round1" / "p0" / "dcan" / "dcan_predictions.npz")
    ptf = d1["probs"].astype(np.float64)
    assert len(ptf) == len(y)
    out["tfidf_ref_f1"] = macro_f1(y, ptf.argmax(1))
    for s in range(3):
        pe = d[f"lora_ext_s{s}_probs"].astype(np.float64)
        diff0 = macro_f1(y, pe.argmax(1)) - out["tfidf_ref_f1"]
        deltas = np.empty(B)
        for i in range(B):
            samp = rng.choice(len(task_list), len(task_list), replace=True)
            idx = np.concatenate([task_idx[task_list[j]] for j in samp])
            deltas[i] = macro_f1(y[idx], pe[idx].argmax(1)) - macro_f1(y[idx], ptf[idx].argmax(1))
        lo, hi = np.percentile(deltas, [2.5, 97.5])
        out["contrasts"][f"extended_s{s} - tfidf_round1"] = {
            "diff": float(diff0), "ci95": [float(lo), float(hi)],
            "excludes_zero": bool(lo > 0 or hi < 0)}
    (A / "stats.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    for k, v in out["contrasts"].items():
        print(f"[r4bstats] {k}: {v['diff']:+.4f} [{v['ci95'][0]:+.4f},{v['ci95'][1]:+.4f}]"
              f"{' *' if v['excludes_zero'] else ''}")


if __name__ == "__main__":
    main()
