#!/usr/bin/env python
"""K1 密度感知判别头 + K2-lite 任务效应剥离（纯 CPU、零训练、只读缓存）。

背景：外部研究方案（DCAN/DMHM 线）的第一优先级验证——
  K1（DMHM 思想）：把"均值位置"判别（原型/线性）升级为"分布形状"判别
      （逐类协方差 → 马氏距离 / QDA）；收益预期来源：granite2b↔smollm2 这类
      "均值近"易混对（若其协方差形状不同，二阶矩可分）。
  K2-lite（DCAN 减法思想的闭式版）：逐题（跨族）中心化后再判别，直接检验
      "任务效应（题目难度噪声）占泛化 gap 多少"——它决定是否值得花 GPU 做
      训练版残差分离（K2-full）。

数据：runs/v0.5_disc/d_{qwen05,qwen15,ds13,yi15,granite2b,smollm2}.npz
      （Δ ∈ R^768 = v1.0 编码器池化特征差，一题一向量）
协议：按题 GroupKFold=5（同题同折，无泄漏）；标准化仅在训练折内 fit；
      所有变体共享同一折划分（排除折随机性）。

变体：
  V0 基线       DiscHead（闭式 LDA/shrinkage + 原型最近邻）——应复现 full 0.7626
  V1 QDA-768d   sklearn QDA(reg_param∈{0.05,0.2,0.5})，标准化 768 维
  V2 LDA5-QDA   DiscHead 投影（5 维）后 QDA(reg∈{0.05,0.2})
  V3 LW-马氏    逐类 Ledoit-Wolf 协方差 → 马氏距离最近类（DMHM 风格，不含 |Σ| 项）
  V4 LW-QDA     同 V3 协方差，但用全 QDA 分数（−½log|Σ_f| − ½d²，含"团大小"）
  V6 逐题中心化 仅 6 族共同题子集（329 题）：Δ̃ = Δ − mean_f(Δ) 后 DiscHead
                （对照 = 同子集的原始 DiscHead 基线）

判读（**预注册，出结果前不得修改**）：
  · K1 通过线：任一协方差变体（V1–V4）full acc ≥ 0.78（+2pt），或易混对互混数
    显著下降（granite2b↔smollm2 基线 182/90 条，降 ≥30%）
  · K2-lite 通过线：V6 相对同子集基线 acc +1.5pt（说明任务效应确为瓶颈来源）
  · 二分诊断：granite2b vs smollm2、qwen05 vs qwen15 的 2 类子实验（同变体集）
  · 若 K1/K2-lite 双双不通过 → 二阶矩/任务效应路线关闭，转向 K2-full（训练版）

运行：OMP_NUM_THREADS=8 /root/miniconda3/envs/udet/bin/python scripts/kernel_e15_density.py
输出：runs/kernel_e15/k1_k2lite.json + stdout 摘要表
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.disc import DiscHead  # noqa: E402

OUT = ROOT / "runs/v0.5_disc"
RES = ROOT / "runs/kernel_e15"
NAMES = ["qwen05", "qwen15", "ds13", "yi15", "granite2b", "smollm2"]
CONF_PAIRS = [("granite2b", "smollm2"), ("qwen05", "qwen15")]
QDA_REGS = (0.05, 0.2, 0.5)
LDA_QDA_REGS = (0.05, 0.2)


# --------------------------------------------------------------------------- #
def load_data(names):
    X, y, g = [], [], []
    for n in names:
        z = np.load(OUT / f"d_{n}.npz", allow_pickle=True)
        X.append(np.asarray(z["d"], dtype="float64"))
        t = [str(v) for v in z["tasks"]]
        y += [n] * len(t)
        g += t
    return np.vstack(X), np.array(y), np.array(g)


def lw_classify(Ztr, ytr, Zva, det: bool = False):
    """逐类 Ledoit-Wolf 协方差分类。

    det=False → 纯马氏距离最近类（V3，DMHM 风格）；
    det=True  → 全 QDA 分数（V4，含 −½log|Σ| 的"团大小"项）。
    """
    classes = np.unique(ytr)
    scores = np.zeros((len(Zva), len(classes)))
    for j, c in enumerate(classes):
        Zc = Ztr[ytr == c]
        mu = Zc.mean(0)
        lw = LedoitWolf().fit(Zc)
        S = (lw.covariance_ + lw.covariance_.T) / 2.0
        P = np.linalg.inv(S)
        D = Zva - mu
        d2 = np.einsum("ij,jk,ik->i", D, P, D)
        scores[:, j] = -0.5 * d2 - (0.5 * float(np.linalg.slogdet(S)[1]) if det else 0.0)
    return classes[scores.argmax(1)]


def run_cv(X, y, g):
    """共享折划分的交叉验证；返回 {变体名: 预测数组}。"""
    n = len(y)
    variant_names = (["V0_disc"] + [f"V1_qda768_r{r}" for r in QDA_REGS]
                     + [f"V2_lda5qda_r{r}" for r in LDA_QDA_REGS]
                     + ["V3_lwm_maha", "V4_lwm_qda"])
    preds = {k: np.empty(n, dtype=object) for k in variant_names}
    for tr, va in GroupKFold(n_splits=5).split(X, y, g):
        Xtr, Xva, ytr = X[tr], X[va], y[tr]
        # V0：闭式基线
        head = DiscHead().fit(Xtr, ytr)
        preds["V0_disc"][va] = head.predict(Xva)
        # 标准化（仅在训练折 fit）
        sc = StandardScaler().fit(Xtr)
        Ztr, Zva = sc.transform(Xtr), sc.transform(Xva)
        # V1：768 维 QDA（n<d 场景：svd solver 会因 rank 亏报错，改 eigen + shrinkage 正则）
        for r in QDA_REGS:
            q = QuadraticDiscriminantAnalysis(solver="eigen", shrinkage=r).fit(Ztr, ytr)
            preds[f"V1_qda768_r{r}"][va] = q.predict(Zva)
        # V2：先 LDA 投影到 5 维再 QDA
        Ztr5, Zva5 = head.transform(Xtr), head.transform(Xva)
        for r in LDA_QDA_REGS:
            q = QuadraticDiscriminantAnalysis(reg_param=r).fit(Ztr5, ytr)
            preds[f"V2_lda5qda_r{r}"][va] = q.predict(Zva5)
        # V3/V4：逐类 Ledoit-Wolf
        preds["V3_lwm_maha"][va] = lw_classify(Ztr, ytr, Zva, det=False)
        preds["V4_lwm_qda"][va] = lw_classify(Ztr, ytr, Zva, det=True)
    return {k: v.astype(str) for k, v in preds.items()}


def summarize(pred, y):
    rec = {str(c): float(((pred == c) & (y == c)).sum()) / max(1, int((y == c).sum()))
           for c in np.unique(y)}
    conf = {}
    for a, b in CONF_PAIRS:
        if a in rec and b in rec:
            conf[f"{a}->{b}"] = int(((y == a) & (pred == b)).sum())
            conf[f"{b}->{a}"] = int(((y == b) & (pred == a)).sum())
    return {
        "acc": round(float(accuracy_score(y, pred)), 4),
        "balanced_acc": round(float(balanced_accuracy_score(y, pred)), 4),
        "recall": {k: round(v, 3) for k, v in rec.items()},
        "conf": conf,
    }


def main() -> int:
    RES.mkdir(parents=True, exist_ok=True)
    report = {}

    # ---------- 全量 6 族：K1 变体 ----------
    print("[e15] 加载 6 族 Δ ...", flush=True)
    X, y, g = load_data(NAMES)
    print(f"[e15] X={X.shape}  y={len(y)}  tasks={len(set(g.tolist()))}", flush=True)
    preds = run_cv(X, y, g)
    report["full"] = {k: summarize(p, y) for k, p in preds.items()}
    print("\n===== 全量 6 族（按题 GroupKFold）=====")
    for k, m in report["full"].items():
        print(f"  {k:>16}: acc {m['acc']:.4f} / bal {m['balanced_acc']:.4f} / conf {m['conf']}")

    # ---------- 二分诊断：易混对子实验 ----------
    report["binary"] = {}
    for a, b in CONF_PAIRS:
        Xb, yb, gb = load_data([a, b])
        pb = run_cv(Xb, yb, gb)
        report["binary"][f"{a}_vs_{b}"] = {k: summarize(p, yb) for k, p in pb.items()}
        print(f"\n===== 二分：{a} vs {b}（n={len(yb)}）=====")
        for k, m in report["binary"][f"{a}_vs_{b}"].items():
            print(f"  {k:>16}: acc {m['acc']:.4f}")

    # ---------- K2-lite：逐题中心化（共同题子集） ----------
    common = None
    mats = []
    for n in NAMES:
        z = np.load(OUT / f"d_{n}.npz", allow_pickle=True)
        d = np.asarray(z["d"], dtype="float64")
        t = [str(v) for v in z["tasks"]]
        common = set(t) if common is None else (common & set(t))
        mats.append((d, t))
    common = sorted(common)
    arrs = []
    for d, t in mats:
        idx = {tt: i for i, tt in enumerate(t)}
        arrs.append(d[[idx[tt] for tt in common]])
    M = np.stack(arrs, axis=0)                       # (k, T, 768)
    Mc = M - M.mean(axis=0, keepdims=True)          # 逐题去跨族均值（K2-lite）
    X2 = np.vstack(list(M))
    X2c = np.vstack(list(Mc))
    y2 = np.concatenate([[n] * len(common) for n in NAMES])
    g2 = np.concatenate([np.array(common) for _ in NAMES])
    raw = DiscHead().cross_val(X2, y2, g2)
    cent = DiscHead().cross_val(X2c, y2, g2)
    report["k2lite"] = {
        "n_common_tasks": len(common), "n_samples": int(len(y2)),
        "raw": {k: round(float(v), 4) for k, v in raw.items()},
        "centered": {k: round(float(v), 4) for k, v in cent.items()},
        "delta_acc": round(float(cent["acc"] - raw["acc"]), 4),
    }
    print(f"\n===== K2-lite 逐题中心化（{len(common)} 共同题）=====")
    print(f"  原始 : {report['k2lite']['raw']}")
    print(f"  中心化: {report['k2lite']['centered']}  (Δacc={report['k2lite']['delta_acc']:+.4f})")

    out = RES / "k1_k2lite.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n[e15] 写出 {out}")
    print("[e15] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
