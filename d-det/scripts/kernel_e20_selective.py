#!/usr/bin/env python
"""E20：选择性中心化（E18-A 机理的构造性验证；纯 CPU 闭式）。

动机（E18-A）：任务中心 μ_t 极低秩，但正交补投影全面劣化——低秩"共享"部分与
判别方向共线（承载位置签名，削之伤信号）；中心化真收益在逐题特异残差。
本实验检验"伤害修复"空间：

    x̃ = x − (I − W·Wᵀ)·μ_t        （μ_t 中与判别方向 W 共线的分量**保留**）

协议（预注册）：
  - 载体 ×2：Δ 视图、[h⁺;h⁻] 联合视图（对齐 329 题，口径同 v0.7，按题 GroupKFold=5）
  - 对照三列：不扣 / 全扣（现行转导版）/ 选择性扣
  - W：训练折内 DiscHead(LDA) 的判别子空间（正交化基），取 r=5（主）与 r=3（对照）；
    W 估计源两版：raw（原始空间拟合）与 center（中心化空间拟合）——都防泄漏
  - 附加读数：‖WWᵀμ_t‖/‖μ_t‖ 跨题分布（共线成分占比；用全数据 W 拟合，仅报告用）

判读线（预注册）：
  selective ≥ full + 0.3pt    → 共线伤害真实且可修复（换默认为选择性扣）
  |full − selective| < 0.3pt  → 联合视图已自行绕开共线（保留全扣版）
  selective < full − 0.3pt    → 与 E18-A 矛盾（先查折内泄漏再复核）

诚实边界：转导版增强，不解除部署约束（E18-A 已在结构上定论）。

输出：runs/kernel_e20/e20.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from models.disc import DiscHead  # noqa: E402
from kernel_e17_framework import cv_disc, summarize, NAMES  # noqa: E402
from kernel_e17b_views import load_h_aligned  # noqa: E402

RES = ROOT / "runs/kernel_e20"


# --------------------------------------------------------------------------- #
def lda_basis(head: DiscHead, r: int) -> np.ndarray:
    """从（已拟合的）DiscHead 提取原始特征空间中的判别子空间正交基 Q (d, r)。"""
    d = head.scaler.mean_.shape[0]
    V = np.asarray(head.lda.scalings_, dtype="float64")
    if V.shape[0] != d and V.shape[1] == d:
        V = V.T
    if V.shape[0] != d:
        raise RuntimeError(f"scalings_ 形状异常：{V.shape}，d={d}")
    V = V[:, :min(r, V.shape[1])]
    V = V / head.scaler.scale_[:, None]          # 标准化逆变换 → 原始空间方向
    Q, _ = np.linalg.qr(V)                       # 正交化（span 不变）
    return Q


def build_view(HP, HM, tag):
    """返回 (M (k,T,d), X (kT,d), MU (T,d), task_of (kT,))。"""
    if tag == "delta":
        M = HP - HM
    else:
        M = np.concatenate([HP, HM], axis=2)     # [h⁺;h⁻] 联合
    k, T, d = M.shape
    X = np.vstack(list(M))                       # 家族主序
    MU = M.mean(axis=0)                          # (T, d) 逐题跨族均值
    task_of = np.arange(k * T) % T
    return M, X, MU, task_of


def run_selective(X, MU, task_of, y, g, r, w_mode):
    """选择性中心化：x̃ = x − (I−QQᵀ)μ_t（Q 为折内 W 的正交基）。"""
    pred = np.empty(len(y), dtype=object)
    for tr, va in GroupKFold(n_splits=5).split(X, y, g):
        Xtr, Xva = X[tr], X[va]
        if w_mode == "center":
            head = DiscHead().fit(Xtr - MU[task_of[tr]], y[tr])
        else:
            head = DiscHead().fit(Xtr, y[tr])
        Q = lda_basis(head, r)                   # (d, r) 正交基，仅用训练折
        Pmu = MU - (MU @ Q) @ Q.T                # (I − QQᵀ)μ_t  (T, d)
        pred[va] = DiscHead().fit(Xtr - Pmu[task_of[tr]], y[tr]).predict(
            Xva - Pmu[task_of[va]])
    return pred.astype(str)


def collinear_read(X, MU, y, r, w_mode):
    """共线成分占比 ‖QQᵀμ_t‖/‖μ_t‖ 的跨题分布（全数据拟合 W，仅报告）。"""
    if w_mode == "center":
        head = DiscHead().fit(X - MU[np.arange(len(y)) % len(MU)], y)
    else:
        head = DiscHead().fit(X, y)
    Q = lda_basis(head, r)
    comp = MU @ Q
    frac = np.linalg.norm(comp, axis=1) / np.maximum(np.linalg.norm(MU, axis=1), 1e-12)
    q = np.percentile(frac, [10, 50, 90])
    return {"mean": round(float(frac.mean()), 4),
            "p10": round(float(q[0]), 4), "median": round(float(q[1]), 4),
            "p90": round(float(q[2]), 4), "n_tasks": int(len(frac))}


# --------------------------------------------------------------------------- #
def main() -> int:
    RES.mkdir(parents=True, exist_ok=True)
    HP, HM, common = load_h_aligned()
    k, T, _ = HP.shape
    y = np.concatenate([[n] * T for n in NAMES])
    g = np.concatenate([np.array(common) for _ in NAMES])
    print(f"[e20] 对齐：{k} 族 × {T} 题 = {k*T} 样本", flush=True)

    report = {}
    for tag in ("delta", "joint"):
        M, X, MU, task_of = build_view(HP, HM, tag)
        entry = {}
        # 1) 不扣
        entry["no_center"] = summarize(cv_disc(X, y, g), y)
        # 2) 全扣（现行转导版）
        Xc = X - MU[task_of]
        entry["full_center"] = summarize(cv_disc(Xc, y, g), y)
        # 3) 选择性扣：r × W源
        for r in (5, 3):
            for wm in ("raw", "center"):
                key = f"selective_r{r}_{wm}W"
                pred = run_selective(X, MU, task_of, y, g, r, wm)
                entry[key] = summarize(pred, y)
                print(f"[e20] {tag:<6} {key:<22}: acc {entry[key]['acc']:.4f} / "
                      f"bal {entry[key]['balanced_acc']:.4f}", flush=True)
        # 4) 读数
        for r in (5, 3):
            for wm in ("raw", "center"):
                entry[f"collinear_frac_r{r}_{wm}W"] = collinear_read(X, MU, y, r, wm)
        # 5) 打印主对照
        print(f"[e20] {tag:<6} no_center            : acc {entry['no_center']['acc']:.4f}", flush=True)
        print(f"[e20] {tag:<6} full_center          : acc {entry['full_center']['acc']:.4f}", flush=True)
        report[tag] = entry

    # ---- 预注册判读（以 selective_r5_rawW 为主） ----
    verdicts = {}
    for tag in ("delta", "joint"):
        e = report[tag]
        full = e["full_center"]["acc"]
        sel = e["selective_r5_rawW"]["acc"]
        d = sel - full
        if d >= 0.003:
            v = f"+{d*100:.2f}pt → 共线伤害真实且可修复（候选新默认）"
        elif d > -0.003:
            v = f"{d*100:+.2f}pt → 无显著差（联合视图已自行绕开共线；保留全扣版）"
        else:
            v = f"{d*100:+.2f}pt → 劣于全扣，与 E18-A 矛盾（先查折内泄漏再复核）"
        verdicts[tag] = v
        print(f"[e20] 判读 {tag}: 选择性 {sel:.4f} vs 全扣 {full:.4f} → {v}", flush=True)

    report["verdicts"] = verdicts
    out = RES / "e20.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"[e20] 写出 {out}\n[e20] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
