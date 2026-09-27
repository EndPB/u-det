#!/usr/bin/env python
"""E21：S/A 交互的显式分解（E18-B"交互式互补"的定位；纯 CPU 闭式）。

检验 1（交互在哪）：类间散布矩阵 S_B 的非对角块能量
    η = 2·‖S_SA‖²_F / ‖S_B‖²_F   （[S;A]，S=(h⁺+h⁻)/2，A=(h⁺−h⁻)/2）
    判读：η≥0.2 交叉集中承载 / 0.05–0.2 显著但分散 / <0.05 交叉弱
    数据面：全量 5703（主）、对齐 329（原始+中心化；报告用）

检验 2（交互是什么）：最简显式交叉特征 ρ(x)=‖A_x‖/‖S_x‖（尺度不变量）
    预注册（qwen05/15 二分，基线=中心化联合视图 ≈0.904）：
      融合增量 ≥ +2pt 或 ρ 单独 AUC > 0.75 → 交互本质 = 尺度比值
      否则 → 分布式高阶结构（回看 η 定位）
    探索附加：granite2b/smollm2 二分、六分类 +ρ

检验 3（交互的族结构）：S-only / A-only 判别器的错误四格表按族分布
    （预期 granite2b/smollm2 贡献"都错"格——交叉结构正是易混对的解药）

输出：runs/kernel_e21/e21.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import cohen_kappa_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from kernel_e17_framework import cv_disc, summarize, load_full, NAMES  # noqa: E402
from kernel_e17b_views import load_h_aligned, center_stack  # noqa: E402

RES = ROOT / "runs/kernel_e21"


# --------------------------------------------------------------------------- #
def between_scatter_eta(X, y, balanced=False):
    """类间散布矩阵 S_B 的分块能量：η = 2‖S_SA‖²/‖S_B‖²。"""
    classes = sorted(set(map(str, y)))
    m = X.mean(axis=0)
    d1 = X.shape[1] // 2
    SB = np.zeros((X.shape[1], X.shape[1]))
    for c in classes:
        Xc = X[y == c]
        mc = Xc.mean(axis=0) - m
        w = 1.0 if balanced else float(len(Xc))
        SB += w * np.outer(mc, mc)
    SS = SB[:d1, :d1]; SA = SB[:d1, d1:]; AA = SB[d1:, d1:]
    eSS = float(np.sum(SS ** 2)); eAA = float(np.sum(AA ** 2)); eSA = float(np.sum(SA ** 2))
    total = eSS + eAA + 2 * eSA
    return {"eta": round(2 * eSA / total, 4),
            "frac_SS": round(eSS / total, 4), "frac_AA": round(eAA / total, 4),
            "frac_cross": round(2 * eSA / total, 4),
            "n": int(len(y)), "balanced": balanced}


def rho_of(S, A):
    """ρ(x) = ‖A_x‖ / ‖S_x‖（逐样本，一维尺度不变量）。"""
    ns = np.linalg.norm(S, axis=1)
    na = np.linalg.norm(A, axis=1)
    return na / np.maximum(ns, 1e-12)


def zscore(v):
    return (v - v.mean()) / max(float(v.std()), 1e-12)


# --------------------------------------------------------------------------- #
def main() -> int:
    RES.mkdir(parents=True, exist_ok=True)
    report = {}

    # ================= 检验 1：η（交互在哪） =================
    Xd, Xhp, Xhm, yf, gf = load_full()
    S_all = (Xhp + Xhm) / 2
    A_all = (Xhp - Xhm) / 2
    report["eta_full_raw"] = between_scatter_eta(np.hstack([S_all, A_all]), yf)
    report["eta_full_raw_balanced"] = between_scatter_eta(np.hstack([S_all, A_all]), yf,
                                                          balanced=True)
    print(f"[e21] η 全量: {report['eta_full_raw']}", flush=True)

    HP, HM, common = load_h_aligned()
    k, T, _ = HP.shape
    y = np.concatenate([[n] * T for n in NAMES])
    g = np.concatenate([np.array(common) for _ in NAMES])
    S_al = (HP + HM) / 2; A_al = (HP - HM) / 2
    report["eta_aligned_raw"] = between_scatter_eta(
        np.hstack([np.vstack(list(S_al)), np.vstack(list(A_al))]), y)
    report["eta_aligned_centered"] = between_scatter_eta(
        center_stack(S_al, A_al), y)
    print(f"[e21] η 对齐原始: {report['eta_aligned_raw']}", flush=True)
    print(f"[e21] η 对齐中心化: {report['eta_aligned_centered']}", flush=True)

    # ================= 检验 2：ρ（交互是什么） =================
    checks = {}
    # ---- 主：qwen05/15 二分 ----
    sel = np.isin(y, ["qwen05", "qwen15"])
    S2 = S_al[:, :, :][np.isin(NAMES, ["qwen05", "qwen15"])]
    A2 = A_al[np.isin(NAMES, ["qwen05", "qwen15"])]
    y2 = y[sel]; g2 = g[sel]
    Xc = center_stack(S2, A2)
    base = summarize(cv_disc(Xc, y2, g2), y2)
    rho = rho_of(np.vstack(list(S2)), np.vstack(list(A2)))
    auc = float(roc_auc_score((y2 == "qwen15").astype(int), rho))
    auc = max(auc, 1 - auc)
    Xf = np.hstack([Xc, zscore(rho)[:, None]])
    fused = summarize(cv_disc(Xf, y2, g2), y2)
    checks["qwen05_vs_qwen15"] = {
        "baseline_center": base, "rho_auc": round(auc, 4),
        "fused": fused, "delta_pt": round((fused["acc"] - base["acc"]) * 100, 2)}
    print(f"[e21] ρ·qwen05/15: 基线 {base['acc']:.4f} + ρ → {fused['acc']:.4f} "
          f"（Δ {checks['qwen05_vs_qwen15']['delta_pt']:+.2f}pt），ρ AUC {auc:.4f}", flush=True)

    # ---- 探索：granite2b/smollm2 二分 ----
    selG = np.isin(y, ["granite2b", "smollm2"])
    S2g = S_al[np.isin(NAMES, ["granite2b", "smollm2"])]
    A2g = A_al[np.isin(NAMES, ["granite2b", "smollm2"])]
    yg = y[selG]; gg = g[selG]
    Xcg = center_stack(S2g, A2g)
    baseg = summarize(cv_disc(Xcg, yg, gg), yg)
    rhog = rho_of(np.vstack(list(S2g)), np.vstack(list(A2g)))
    aucg = float(roc_auc_score((yg == "smollm2").astype(int), rhog))
    aucg = max(aucg, 1 - aucg)
    Xfg = np.hstack([Xcg, zscore(rhog)[:, None]])
    fusedg = summarize(cv_disc(Xfg, yg, gg), yg)
    checks["granite2b_vs_smollm2"] = {
        "baseline_center": baseg, "rho_auc": round(aucg, 4),
        "fused": fusedg, "delta_pt": round((fusedg["acc"] - baseg["acc"]) * 100, 2)}
    print(f"[e21] ρ·granite/smollm: 基线 {baseg['acc']:.4f} + ρ → {fusedg['acc']:.4f} "
          f"（Δ {checks['granite2b_vs_smollm2']['delta_pt']:+.2f}pt），ρ AUC {aucg:.4f}", flush=True)

    # ---- 探索：六分类 + ρ ----
    S6 = S_al; A6 = A_al
    Xc6 = center_stack(S6, A6)
    base6 = summarize(cv_disc(Xc6, y, g), y)
    rho6 = rho_of(np.vstack(list(S6)), np.vstack(list(A6)))
    Xf6 = np.hstack([Xc6, zscore(rho6)[:, None]])
    fused6 = summarize(cv_disc(Xf6, y, g), y)
    checks["six_family"] = {
        "baseline_center": base6, "fused": fused6,
        "delta_pt": round((fused6["acc"] - base6["acc"]) * 100, 2)}
    print(f"[e21] ρ·六分类: 基线 {base6['acc']:.4f} + ρ → {fused6['acc']:.4f} "
          f"（Δ {checks['six_family']['delta_pt']:+.2f}pt）", flush=True)
    report["rho_checks"] = checks

    # 预注册判读
    c = checks["qwen05_vs_qwen15"]
    if c["delta_pt"] >= 2.0 or c["rho_auc"] > 0.75:
        v = "成立：交互本质 = 尺度比值（ρ）——可写入理论节"
    else:
        v = "不成立：分布/高阶结构，回看 η 读数定位"
    report["rho_verdict"] = v
    print(f"[e21] ρ 判读: {v}", flush=True)

    # ================= 检验 3：错误四格表（交互的族结构） =================
    predS = cv_disc(np.vstack(list(S_al)), y, g)
    predA = cv_disc(np.vstack(list(A_al)), y, g)
    errS = predS != y; errA = predA != y
    tab = {}
    for c_ in NAMES:
        m = y == c_
        tab[c_] = {"both_right": int((m & ~errS & ~errA).sum()),
                   "S_only_wrong": int((m & errS & ~errA).sum()),
                   "A_only_wrong": int((m & ~errS & errA).sum()),
                   "both_wrong": int((m & errS & errA).sum())}
    report["error_cross_tab_aligned"] = {
        "S_acc": round(float((~errS).mean()), 4), "A_acc": round(float((~errA).mean()), 4),
        "kappa_err": round(float(cohen_kappa_score(errS, errA)), 4), "by_family": tab}
    print(f"[e21] 错误相关 kappa(对齐) = {report['error_cross_tab_aligned']['kappa_err']}"
          f"（S {(~errS).mean():.4f} / A {(~errA).mean():.4f}）", flush=True)
    for c_ in NAMES:
        t = tab[c_]
        print(f"[e21]   {c_:<10} 都错 {t['both_wrong']:>3} | S独错 {t['S_only_wrong']:>3} "
              f"| A独错 {t['A_only_wrong']:>3} | 都对 {t['both_right']:>3}", flush=True)

    out = RES / "e21.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"[e21] 写出 {out}\n[e21] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
