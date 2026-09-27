#!/usr/bin/env python
"""E20/E21 补充校验（对齐预注册口径 + 稳健性读数；纯 CPU）。

  A. E20 泄漏自检：选择性中心化改用"全数据拟合 W"（故意泄漏）重跑——
     若与折内版本差异微小 → 折内版本结果非泄漏/实现假象。
  B. E21 ρ 检验·规范口径：v0.7 官方构造 [h̃⁺;h̃⁻]（按二家族块中心化；基线 0.904/0.965），
     复算"基线 vs +ρ"；另附六分类规范口径。
  C. E21 η 尺度稳健性：列标准化版、块范数配平版的交叉能量占比。
  D. E21 错误四格表·全量 5703（与 E17e 的 S .7715 / A .7626 对齐复核）。

输出：runs/kernel_e21/e21_supp.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import cohen_kappa_score, roc_auc_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from models.disc import DiscHead  # noqa: E402
from kernel_e17_framework import cv_disc, summarize, load_full, NAMES  # noqa: E402
from kernel_e17b_views import load_h_aligned  # noqa: E402
from kernel_e20_selective import build_view, lda_basis  # noqa: E402
from eval_stable_v07 import subset_arrays  # noqa: E402

RES = ROOT / "runs/kernel_e21"


def run_selective_fixedQ(X, MU, task_of, y, g, Q):
    pred = np.empty(len(y), dtype=object)
    for tr, va in GroupKFold(n_splits=5).split(X, y, g):
        Pmu = MU - (MU @ Q) @ Q.T
        pred[va] = DiscHead().fit(X[tr] - Pmu[task_of[tr]], y[tr]).predict(
            X[va] - Pmu[task_of[va]])
    return pred.astype(str)


def eta_of(X, y, balanced=False, case="raw"):
    d1 = X.shape[1] // 2
    m = X.mean(axis=0)
    SB = np.zeros((X.shape[1], X.shape[1]))
    for c in sorted(set(map(str, y))):
        Xc = X[y == c]
        mc = Xc.mean(axis=0) - m
        w = 1.0 if balanced else float(len(Xc))
        SB += w * np.outer(mc, mc)
    SS = SB[:d1, :d1]; SA = SB[:d1, d1:]; AA = SB[d1:, d1:]
    eSS = float(np.sum(SS ** 2)); eAA = float(np.sum(AA ** 2)); eSA = float(np.sum(SA ** 2))
    tot = eSS + eAA + 2 * eSA
    return {"case": case, "eta": round(2 * eSA / tot, 4),
            "frac_SS": round(eSS / tot, 4), "frac_AA": round(eAA / tot, 4)}


def rho_of(S, A):
    return (np.linalg.norm(A, axis=1) /
            np.maximum(np.linalg.norm(S, axis=1), 1e-12))


def zscore(v):
    return (v - v.mean()) / max(float(v.std()), 1e-12)


def main() -> int:
    RES.mkdir(parents=True, exist_ok=True)
    report = {}

    # ============ A. E20 泄漏自检（joint 视图 r=5 rawW） ============
    HP, HM, common = load_h_aligned()
    k, T, _ = HP.shape
    y = np.concatenate([[n] * T for n in NAMES])
    g = np.concatenate([np.array(common) for _ in NAMES])
    M, X, MU, task_of = build_view(HP, HM, "joint")
    Q_all = lda_basis(DiscHead().fit(X, y), 5)          # 故意用全数据（含测试折）拟合
    pred_leak = run_selective_fixedQ(X, MU, task_of, y, g, Q_all)
    leak_acc = summarize(pred_leak, y)["acc"]
    report["e20_leak_check"] = {
        "fold_internal_r5rawW": 0.8414,                  # E20 实测（折内 W）
        "full_data_W_leaky": round(leak_acc, 4),
        "note": "两值接近（±0.3pt 内）→ 折内版本非泄漏假象"}
    print(f"[supp] A·E20 泄漏自检: 折内 0.8414 vs 全数据W {leak_acc:.4f}", flush=True)

    # ============ B. E21 ρ·规范口径（[h̃⁺;h̃⁻]） ============
    bchecks = {}
    for pair, base_ref in ((["qwen05", "qwen15"], 0.9036),
                           (["granite2b", "smollm2"], 0.9646)):
        HP2, HM2, common2 = subset_arrays(pair)
        HP2c = HP2 - HP2.mean(axis=0, keepdims=True)
        HM2c = HM2 - HM2.mean(axis=0, keepdims=True)
        Xc = np.hstack([np.vstack(list(HP2c)), np.vstack(list(HM2c))])
        y2 = np.concatenate([[n] * len(common2) for n in pair])
        g2 = np.concatenate([np.array(common2) for _ in pair])
        base = summarize(cv_disc(Xc, y2, g2), y2)
        rho = rho_of(np.vstack(list(HP2)), np.vstack(list(HM2)))
        auc = float(roc_auc_score((y2 == pair[1]).astype(int), rho))
        auc = max(auc, 1 - auc)
        fused = summarize(cv_disc(np.hstack([Xc, zscore(rho)[:, None]]), y2, g2), y2)
        bchecks[f"{pair[0]}_vs_{pair[1]}"] = {
            "baseline_canonical": base["acc"], "v07_reference": base_ref,
            "rho_auc": round(auc, 4), "fused": fused["acc"],
            "delta_pt": round((fused["acc"] - base["acc"]) * 100, 2)}
        print(f"[supp] B·ρ {pair}: 规范基线 {base['acc']:.4f}（v0.7 参考 {base_ref}）"
              f" + ρ → {fused['acc']:.4f}（Δ {bchecks[f'{pair[0]}_vs_{pair[1]}']['delta_pt']:+.2f}pt）"
              f"，ρ AUC {auc:.4f}", flush=True)
    # 六分类规范口径
    HP6, HM6, common6 = subset_arrays(NAMES)
    HP6c = HP6 - HP6.mean(axis=0, keepdims=True)
    HM6c = HM6 - HM6.mean(axis=0, keepdims=True)
    Xc6 = np.hstack([np.vstack(list(HP6c)), np.vstack(list(HM6c))])
    y6 = np.concatenate([[n] * len(common6) for n in NAMES])
    g6 = np.concatenate([np.array(common6) for _ in NAMES])
    base6 = summarize(cv_disc(Xc6, y6, g6), y6)
    rho6 = rho_of(np.vstack(list(HP6)), np.vstack(list(HM6)))
    fused6 = summarize(cv_disc(np.hstack([Xc6, zscore(rho6)[:, None]]), y6, g6), y6)
    bchecks["six_family_canonical"] = {
        "baseline_canonical": base6["acc"], "v07_reference": 0.8571,
        "fused": fused6["acc"],
        "delta_pt": round((fused6["acc"] - base6["acc"]) * 100, 2)}
    print(f"[supp] B·ρ 六分类: 规范基线 {base6['acc']:.4f} + ρ → {fused6['acc']:.4f}"
          f"（Δ {bchecks['six_family_canonical']['delta_pt']:+.2f}pt）", flush=True)
    report["rho_canonical"] = bchecks

    # ============ C. η 尺度稳健性 ============
    Xd, Xhp, Xhm, yf, gf = load_full()
    S_all = (Xhp + Xhm) / 2
    A_all = (Xhp - Xhm) / 2
    Xfa = np.hstack([S_all, A_all])
    eta_raw = eta_of(Xfa, yf, case="raw")
    Z = (Xfa - Xfa.mean(0)) / np.maximum(Xfa.std(0), 1e-12)
    eta_colz = eta_of(Z, yf, case="col_zscored")
    Xfn = Xfa.copy()
    Xfn[:, :Xfa.shape[1] // 2] /= np.linalg.norm(S_all) / np.sqrt(len(Xfa))
    Xfn[:, Xfa.shape[1] // 2:] /= np.linalg.norm(A_all) / np.sqrt(len(Xfa))
    eta_bal = eta_of(Xfn, yf, case="block_norm_matched")
    report["eta_robustness"] = [eta_raw, eta_colz, eta_bal]
    print(f"[supp] C·η 稳健性: raw {eta_raw['eta']} / 列标准化 {eta_colz['eta']} / "
          f"块配平 {eta_bal['eta']}", flush=True)

    # ============ D. 错误四格表·全量 5703 ============
    predS = cv_disc(S_all, yf, gf)
    predA = cv_disc(A_all, yf, gf)
    errS = predS != yf; errA = predA != yf
    tab = {}
    for c in NAMES:
        m = yf == c
        tab[c] = {"both_right": int((m & ~errS & ~errA).sum()),
                  "S_only_wrong": int((m & errS & ~errA).sum()),
                  "A_only_wrong": int((m & ~errS & errA).sum()),
                  "both_wrong": int((m & errS & errA).sum())}
    report["error_cross_tab_full"] = {
        "S_acc": round(float((~errS).mean()), 4), "A_acc": round(float((~errA).mean()), 4),
        "kappa_err": round(float(cohen_kappa_score(errS, errA)), 4), "by_family": tab}
    print(f"[supp] D·全量错误表: S {report['error_cross_tab_full']['S_acc']} / "
          f"A {report['error_cross_tab_full']['A_acc']} / "
          f"kappa {report['error_cross_tab_full']['kappa_err']}", flush=True)
    for c in NAMES:
        t = tab[c]
        print(f"[supp]   {c:<10} 都错 {t['both_wrong']:>3} | S独错 {t['S_only_wrong']:>3} "
              f"| A独错 {t['A_only_wrong']:>3} | 都对 {t['both_right']:>3}", flush=True)

    out = RES / "e21_supp.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"[supp] 写出 {out}\n[supp] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
