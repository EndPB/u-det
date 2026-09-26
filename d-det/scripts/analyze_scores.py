#!/usr/bin/env python
"""离线分数分析（**不占 GPU**）：阈值扫描 / 长度分桶 / 2D GMM / 象限表 / 正交度。

输入：``train.py --eval --dump-scores`` 产出的逐样本分数文件：

    scores_m4_{val,test}.pt    {s1 (n,), s2 (n,r), label (n,), length (n,), meta}
    scores_pair_{val,test}.pt  {s1_plus/s2_plus/s1_minus/s2_minus, meta}

分析项（设计文档 §4 的联合结构与 §7 注记 2 的配套口径）：

    1. m4 阈值扫描   —— 固定 0.5 可能不是最优工作点（u-det 实测过六成差距来自阈值）
    2. m4 长度分桶   —— 防 C2 陷阱（长度与标签强混淆，总体指标会骗人）
    3. 2D 两成分 GMM —— 联合检测：π_A = 高 AI 成分的权重 / 纯度；硬分配一致率
    4. 象限表        —— 以**人类样本**为参考做 z 化，统计四象限的 AI 占比
                        （①高s1高s2=典型 instruct；②高s1低s2=base；③负 s2=反向倾斜；
                          ④低 s1=人类）
    5. 读出正交度    —— 从 best.pt 读 w1/w2 算 |cos|（配合训练日志的 cos(w1,w2)）

用法::

    python scripts/analyze_scores.py runs/v0.1.0
    python scripts/analyze_scores.py runs/v0.1.0 runs/v0.1.1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]

BUCKETS = [(0, 512), (512, 2048), (2048, 8192), (8192, 1 << 30)]


def load_dump(run_dir: Path, stream: str, split: str):
    for candidate in (split, "test", "val"):
        path = run_dir / f"scores_{stream}_{candidate}.pt"
        if path.exists():
            return torch.load(path, map_location="cpu", weights_only=False), candidate
    return None, None


def _f1(pred, label):
    tp = float(((pred == 1) & (label == 1)).sum())
    fp = float(((pred == 1) & (label == 0)).sum())
    fn = float(((pred == 0) & (label == 1)).sum())
    precision = tp / max(tp + fp, 1e-9)
    recall = tp / max(tp + fn, 1e-9)
    return 2 * precision * recall / max(precision + recall, 1e-9)


def analyze_m4(blob: dict):
    """返回 (metrics_dict, s1, s2, label)。"""
    s1 = blob["s1"].numpy().astype(np.float64)
    s2 = blob["s2"].numpy().astype(np.float64)
    label = blob["label"].numpy().astype(int)
    length = blob["length"].numpy().astype(int)
    prob = 1.0 / (1.0 + np.exp(-s1))

    out: dict = {"n": int(len(label)), "ai_ratio": float(label.mean())}
    try:
        out["auc"] = float(roc_auc_score(label, prob))
    except ValueError:
        out["auc"] = None
    out["acc@0.5"] = float(((prob >= 0.5).astype(int) == label).mean())
    out["f1@0.5"] = float(_f1((prob >= 0.5).astype(int), label))

    best = None
    for thr in np.arange(0.05, 1.0, 0.05):
        pred = (prob >= thr).astype(int)
        row = {"thr": float(thr), "acc": float((pred == label).mean()),
               "f1": float(_f1(pred, label))}
        if best is None or row["f1"] > best["f1"]:
            best = row
    out["best_threshold"] = best

    buckets = []
    for lo, hi in BUCKETS:
        mask = (length >= lo) & (length < hi)
        if mask.sum() == 0:
            continue
        sub_label, sub_prob = label[mask], prob[mask]
        entry = {
            "range": f"[{lo},{hi})", "n": int(mask.sum()),
            "ai_ratio": float(sub_label.mean()),
            "acc@0.5": float(((sub_prob >= 0.5).astype(int) == sub_label).mean()),
            "f1@0.5": float(_f1((sub_prob >= 0.5).astype(int), sub_label)),
            "auc": (float(roc_auc_score(sub_label, sub_prob))
                    if len(set(sub_label.tolist())) > 1 else None),
        }
        buckets.append(entry)
    out["length_buckets"] = buckets
    return out, s1, s2, label


def analyze_joint(s1, s2, label) -> dict:
    """2D 两成分 GMM（设计文档 §4 的联合检测）。"""
    if s2.ndim > 1:
        s2 = s2[:, 0]
    z1 = (s1 - s1.mean()) / (s1.std() + 1e-9)
    z2 = (s2 - s2.mean()) / (s2.std() + 1e-9)
    x = np.column_stack([z1, z2])

    from sklearn.mixture import GaussianMixture
    gmm = GaussianMixture(n_components=2, covariance_type="full",
                          n_init=5, random_state=0).fit(x)
    assign = gmm.predict(x)
    comps = []
    for k in range(2):
        mask = assign == k
        comps.append({
            "weight": float(gmm.weights_[k]),
            "n": int(mask.sum()),
            "ai_ratio": float(label[mask].mean()) if mask.sum() else None,
            "mean_z": [float(gmm.means_[k][0]), float(gmm.means_[k][1])],
        })
    ai_index = int(np.argmax([c["ai_ratio"] or 0.0 for c in comps]))
    hard_pred = (assign == ai_index).astype(int)
    return {
        "components": comps,
        "pi_A_weight": float(gmm.weights_[ai_index]),      # "AI 成分"的权重（≈协写比例）
        "pi_A_purity": comps[ai_index]["ai_ratio"],        # 该成分里 AI 的真实占比
        "hard_agreement": float((hard_pred == label).mean()),
    }


def analyze_quadrants(s1, s2, label) -> list:
    """以人类样本为参考 z 化，统计四象限（对应设计文档 §4 的象限语义）。"""
    if s2.ndim > 1:
        s2 = s2[:, 0]
    ref = label == 0
    if ref.sum() == 0:
        return []
    mu1, sd1 = s1[ref].mean(), s1[ref].std() + 1e-9
    mu2, sd2 = s2[ref].mean(), s2[ref].std() + 1e-9
    z1, z2 = (s1 - mu1) / sd1, (s2 - mu2) / sd2
    quads = []
    for name, (c1, c2) in (("I (+s1,+s2) 典型 instruct", (1, 1)),
                           ("II (+s1,-s2) base 型", (1, -1)),
                           ("III (-s1,-s2) 人类/低坍缩", (-1, -1)),
                           ("IV (-s1,+s2) 低坍缩高偏好", (-1, 1))):
        mask = (np.sign(z1) == c1) & (np.sign(z2) == c2)
        if mask.sum() == 0:
            continue
        quads.append({"quadrant": name, "n": int(mask.sum()),
                      "ai_ratio": float(label[mask].mean())})
    return quads


def analyze_pair(blob: dict) -> dict:
    s2_plus = blob["s2_plus"].numpy().astype(np.float64)
    s2_minus = blob["s2_minus"].numpy().astype(np.float64)
    if s2_plus.ndim > 1:
        gap = np.linalg.norm(s2_plus - s2_minus, axis=1)
    else:
        gap = (s2_plus - s2_minus).reshape(-1)
    return {
        "n": int(len(gap)),
        "dir_acc": float((gap > 0).mean()),
        "delta_mean": float(gap.mean()),
        "delta_median": float(np.median(gap)),
        "delta_q10": float(np.quantile(gap, 0.10)),
        "delta_q90": float(np.quantile(gap, 0.90)),
    }


def analyze_readout(run_dir: Path) -> dict | None:
    ckpt_path = run_dir / "best.pt"
    if not ckpt_path.exists():
        return None
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("state", {})
    w1 = state.get("w1.weight")
    w2 = state.get("w2.weight")
    if w1 is None or w2 is None:
        return None
    w1 = w1.reshape(-1).double()
    w2 = w2.double()
    cos = (w2 @ w1).abs() / (w2.norm(dim=1) * w1.norm() + 1e-12)
    return {"cos_abs_mean": float(cos.mean()), "cos_abs_max": float(cos.max()),
            "s2_rank": int(w2.shape[0])}


def run_one(run_dir: Path, out_json: bool = True) -> dict:
    report: dict = {"run": str(run_dir)}
    print(f"\n=== {run_dir} ===")

    blob, split = load_dump(run_dir, "m4", "test")
    if blob is not None:
        m4_out, s1, s2, label = analyze_m4(blob)
        print(f"[m4/{split}] n={m4_out['n']} AUC={m4_out['auc']} "
              f"acc@0.5={m4_out['acc@0.5']:.4f} f1@0.5={m4_out['f1@0.5']:.4f}")
        if m4_out["best_threshold"]:
            bt = m4_out["best_threshold"]
            print(f"[m4/{split}] 最佳阈值 thr={bt['thr']:.2f} → acc={bt['acc']:.4f} "
                  f"f1={bt['f1']:.4f}")
        for entry in m4_out["length_buckets"]:
            print(f"[m4/{split}] 分桶 {entry['range']:>11}  n={entry['n']:>5}  "
                  f"ai={entry['ai_ratio']:.2f}  acc@0.5={entry['acc@0.5']:.4f}  "
                  f"AUC={entry['auc'] if entry['auc'] is None else round(entry['auc'], 4)}")
        joint = analyze_joint(s1, s2, label)
        print(f"[联合 GMM] π_A(权重)={joint['pi_A_weight']:.3f}  "
              f"pi_A(纯度)={joint['pi_A_purity']}  硬分配一致率={joint['hard_agreement']:.4f}")
        for comp in joint["components"]:
            print(f"          成分 w={comp['weight']:.3f} n={comp['n']} "
                  f"AI占比={comp['ai_ratio']:.3f} 均值z={[round(v, 2) for v in comp['mean_z']]}")
        for quad in analyze_quadrants(s1, s2, label):
            print(f"[象限] {quad['quadrant']}  n={quad['n']:>5}  AI占比={quad['ai_ratio']:.3f}")
        report["m4"] = {**m4_out, "joint_gmm": joint,
                        "quadrants": analyze_quadrants(s1, s2, label)}
    else:
        print("[m4] 未找到分数文件（先跑 --eval --dump-scores）")

    pair_blob, pair_split = load_dump(run_dir, "pair", "test")
    if pair_blob is not None:
        pair_out = analyze_pair(pair_blob)
        print(f"[pair/{pair_split}] n={pair_out['n']} dir_acc={pair_out['dir_acc']:.4f} "
              f"Δ(mean/median)={pair_out['delta_mean']:.3f}/{pair_out['delta_median']:.3f}")
        report["pair"] = pair_out

    readout = analyze_readout(run_dir)
    if readout is not None:
        print(f"[读出] s2_rank={readout['s2_rank']}  |cos(w1,w2)| 均值={readout['cos_abs_mean']:.4f} "
              f"最大={readout['cos_abs_max']:.4f}")
        report["readout"] = readout

    if out_json:
        out_path = run_dir / "analysis.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        print(f"[分析] 结果已写入 {out_path}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="d-det 离线分数分析")
    parser.add_argument("runs", nargs="+", help="run 目录（可多个）")
    args = parser.parse_args()
    for item in args.runs:
        run_dir = Path(item)
        if not run_dir.is_absolute():
            run_dir = ROOT / run_dir
        if not run_dir.exists():
            print(f"[分析] 跳过不存在的目录：{run_dir}")
            continue
        run_one(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
