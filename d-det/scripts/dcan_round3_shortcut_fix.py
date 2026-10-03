#!/usr/bin/env python
"""Round3：Q1 捷径诊断的两处修正（不重训大基线）。

- 长度桶边界改为 **由 train 拟合**（char_count 四分位），再应用于 test（原先误用 test 分位数）；
- 所有 LR 重拟合记录 convergence（n_iter_/警告）；
- 明确 filtered TF-IDF 协议：固定 5 epoch、dev 仅记录、不恢复 best epoch。
输出：artifacts/acl_dcan_round3_audit/shortcuts_fix.json
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import dcan_four_models as r1  # noqa: E402

R2 = ROOT / "artifacts" / "acl_dcan_round2" / "shortcuts"
R1 = ROOT / "artifacts" / "acl_dcan_round1" / "p0" / "dcan"
OUT = ROOT / "artifacts" / "acl_dcan_round3_audit"
OUT.mkdir(parents=True, exist_ok=True)


def bucket_acc(pred, y, lens, edges):
    bid = np.digitize(lens, edges)
    return {f"q{i+1}": {"n": int((bid == i).sum()),
                        "acc": float((pred[bid == i] == y[bid == i]).mean()) if (bid == i).any() else None}
            for i in range(4)}


def main():
    rows = r1.load_rows()
    split = [r["task_split"] for r in rows]
    tr = np.array([i for i, s in enumerate(split) if s == "train"])
    te = np.array([i for i, s in enumerate(split) if s == "test"])
    lens = np.array([r["char_count"] for r in rows], dtype=float)
    edges = np.quantile(lens[tr], [0.25, 0.5, 0.75])  # train 拟合
    fam_idx = {f: i for i, f in enumerate(r1.FAMILIES)}
    y = np.array([fam_idx[r["family"]] for r in rows])
    out = {"train_quartile_edges": [float(x) for x in edges],
           "note": "长度桶边界由 train 拟合后应用于 test；推断按 test 顺序对齐既有预测",
           "methods": {}, "lr_convergence": {}}

    preds = np.load(R2 / "predictions.npz", allow_pickle=True)
    checks = {"metadata_only": "metadata_only_probs", "filtered_tfidf": "filtered_tfidf_probs",
              "codeT5_raw": "codeT5_raw_probs", "codeT5_center": "codeT5_center_probs",
              "codeT5_center_std": "codeT5_center_std_probs"}
    import sklearn.metrics as sm
    for name, key in checks.items():
        probs = preds[key].astype(np.float64)
        assert len(probs) == len(te)
        pred = probs.argmax(1)
        out["methods"][name] = {
            "macro_f1_check": float(sm.f1_score(y[te], pred, average="macro", zero_division=0)),
            "per_length_quartile_acc_train_edges": bucket_acc(pred, y[te], lens[te], edges)}
    d1 = np.load(R1 / "dcan_predictions.npz")
    out["methods"]["char_tfidf_round1_ref"] = {
        "macro_f1_check": float(sm.f1_score(y[te], d1["probs"].astype(np.float64).argmax(1),
                                            average="macro", zero_division=0)),
        "per_length_quartile_acc_train_edges": bucket_acc(d1["probs"].astype(np.float64).argmax(1),
                                                           y[te], lens[te], edges)}

    # LR convergence 重拟合（同 round2 口径）
    st_raw = np.array([r1.struct_features(r) for r in rows], dtype=np.float64)
    emb = r1.encode_semantics(rows).astype(np.float64)
    task = [r["task_id"] for r in rows]
    tm = {}
    for i, t in enumerate(task):
        tm.setdefault(t, []).append(i)
    zc = emb - np.array([emb[ix].mean(0) for ix in [tm[t] for t in task]])
    sdv = zc[tr].std(0) + 1e-9
    from sklearn.linear_model import LogisticRegression
    feats = {"metadata_only": (st_raw - st_raw[tr].mean(0)) / (st_raw[tr].std(0) + 1e-9),
             "codeT5_raw": emb, "codeT5_center": zc, "codeT5_center_std": zc / sdv}
    for name, X in feats.items():
        with warnings.catch_warnings(record=True) as wl:
            warnings.simplefilter("always")
            lr = LogisticRegression(max_iter=2000, C=0.1).fit(X[tr], y[tr])
        conv = not any("converge" in str(w.message).lower() for w in wl)
        out["lr_convergence"][name] = {"n_iter": int(lr.n_iter_[0]), "converged": bool(conv),
                                       "repro_macro_f1": float(sm.f1_score(
                                           y[te], lr.predict(X[te]), average="macro", zero_division=0))}
    out["filtered_tfidf_protocol"] = {
        "epochs": 5, "dev_logged_only": True, "best_epoch_restore": False,
        "note": "与 round2 一致：固定 5 epoch，不作 best-epoch 恢复（与 raw/center 的 LR 单次拟合不同类）"}
    (OUT / "shortcuts_fix.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print("[r3fix] written shortcuts_fix.json")
    for k, v in out["methods"].items():
        print(" ", k, round(v["macro_f1_check"], 4),
              {kk: (None if vv['acc'] is None else round(vv['acc'], 3))
               for kk, vv in v["per_length_quartile_acc_train_edges"].items()})


if __name__ == "__main__":
    main()
