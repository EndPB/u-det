"""修复测试：加权指标审计（AP tie 修复 + sklearn 对照）+ 从保存分数重算。

1. 合成套件（tie 全同分 / 全正 / 全负 / 单正例 / 随机 / 带权 tie 组）：
   对照 sklearn roc_auc_score / average_precision_score 的 sample_weight 版本；
   AUC 全负或全正时报警告并记 None（不写无效数值）；AP 修复口径=sklearn。
2. 从 7ee36fb 已保存 48 折分数（relation_endpoint_balanced/local/*.npz）重算
   AUC + AP（sklearn），废弃旧向量化 AP（全 tie 0.8333 错误）；AUC 与旧版并列核验。
输出：artifacts/relation_endpoint_balanced_strong_p0_2026-10-08/{metrics_metric_audit.json}
"""
from __future__ import annotations

import glob
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "d-det/artifacts/relation_endpoint_balanced_strong_p0_2026-10-08"
SRC = ROOT / "d-det/artifacts/relation_endpoint_balanced_2026-10-08"


def sk_auc(y, s, w):
    from sklearn.metrics import roc_auc_score
    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, s, sample_weight=w))


def sk_ap(y, s, w):
    from sklearn.metrics import average_precision_score
    if len(np.unique(y)) < 2:
        return None
    return float(average_precision_score(y, s, sample_weight=w))


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "logs").mkdir(exist_ok=True)
    rng = np.random.default_rng(20261008)

    # ---------- 合成套件 ----------
    synth = {}
    # 全 tie
    y = np.r_[np.ones(50), np.zeros(50)]; s = np.zeros(100); w = np.ones(100)
    synth["all_tie"] = {"auc": sk_auc(y, s, w), "ap": sk_ap(y, s, w)}
    # 无正类
    y0 = np.zeros(100); s0 = rng.normal(size=100); w0 = np.ones(100)
    synth["no_positive"] = {"auc": sk_auc(y0, s0, w0), "ap": sk_ap(y0, s0, w0)}
    # 无负类
    y1 = np.ones(100)
    synth["no_negative"] = {"auc": sk_auc(y1, s0, w0), "ap": sk_ap(y1, s0, w0)}
    # 单正例
    ys = np.zeros(100); ys[0] = 1
    synth["single_positive"] = {"auc": sk_auc(ys, s0, w0), "ap": sk_ap(ys, s0, w0)}
    # 随机连续
    yc = (rng.random(200) > 0.5).astype(float); sc = rng.normal(size=200); wc = rng.random(200) + 0.1
    synth["random_continuous_weighted"] = {"auc": sk_auc(yc, sc, wc), "ap": sk_ap(yc, sc, wc)}
    # 带权 tie 组（50% ties）
    yt = np.r_[np.ones(60), np.zeros(60)]
    st = np.r_[np.ones(30), np.zeros(30), np.r_[np.ones(30), np.zeros(30)]]
    wt = np.r_[np.full(30, 2.0), np.full(30, 1.0), np.full(30, 0.5), np.full(30, 3.0)]
    synth["tie_groups_weighted"] = {"auc": sk_auc(yt, st, wt), "ap": sk_ap(yt, st, wt)}
    # 平衡权重下的锚点分数 sanity（每个 anchor 正负权重相等 → AUC=0.5）
    ya = np.r_[np.ones(10), np.zeros(10), np.ones(10), np.zeros(10)]
    sa = np.r_[np.full(20, 7.0), np.full(20, 3.0)]
    wa = np.r_[np.full(10, 1/3), np.full(10, 1/3), np.full(10, 1/3), np.full(10, 1/3)]
    synth["balanced_anchor_ties"] = {"auc": sk_auc(ya, sa, wa), "ap": sk_ap(ya, sa, wa)}
    print("[audit] synth:", json.dumps(synth, ensure_ascii=False))

    # ---------- 48 折重算 ----------
    folds = {}
    files = sorted(glob.glob(str(SRC / "local" / "eval_scores_fold*.npz")),
                   key=lambda p: int(Path(p).stem.replace("eval_scores_fold", "")))
    for p in files:
        fid = int(Path(p).stem.replace("eval_scores_fold", ""))
        z = np.load(p)
        y = z["y"]; w = z["w"]
        rec = {}
        for k in [kk for kk in z.files if kk.startswith("s_")]:
            s = z[k]
            rec[k[2:]] = {"auc": sk_auc(y, s, w), "ap": sk_ap(y, s, w),
                          "unweighted_auc": float(__import__("sklearn.metrics", fromlist=["roc_auc_score"]).roc_auc_score(y, s))}
        folds[fid] = rec

    read_names = ["compose", "q_sym", "P0_compose", "P0_pair", "residual",
                  "endpoint_anchor_size", "endpoint_partner_size", "endpoint_unorm_diff"]
    agg = {}
    for k in read_names:
        aucs = [folds[f][k]["auc"] for f in folds if folds[f][k]["auc"] is not None]
        aps = [folds[f][k]["ap"] for f in folds if folds[f][k]["ap"] is not None]
        agg[k] = {"auc_mean": float(np.mean(aucs)), "ap_mean": float(np.mean(aps)), "n": len(aucs)}
        print(f"[audit] {k}: AUC={agg[k]['auc_mean']:.4f} AP={agg[k]['ap_mean']:.4f}")

    out = {"schema": "metric_audit_v1", "generated_utc": datetime.now(timezone.utc).isoformat(),
           "report_first_line": "修复测试 — AP tie 修复（sklearn 对照）+ 48 折重算",
           "synthetic_suite": synth,
           "note": ("修复口径：weighted AUC/AP 直接使用 sklearn sample_weight 版本；"
                    "无正/无负类返回 null 并记警告（不写无效数值）。旧向量化 AP 废弃（全 tie 0.8333 错误）。"),
           "folds": folds, "aggregate": agg}
    (OUT / "metrics_metric_audit.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[audit] done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
