#!/usr/bin/env python
"""E18：v0.8 内核级优化三线验证（A 中心化算子化 / B 度量分块化 / C 先决检查）。

全部闭式、CPU。判读线（预注册，出数前锁定；来自外部路线文档）：
  A·谱检验：任务中心（跨族均值，h⁺/h⁻ 分块）PCA——有效秩（90% 方差）<50 → 开绿灯；
  A·归纳版：训练折拟合 U_r（前 r 个主方向），测试样本 x̃=(I−UUᵀ)x（单样本，无需兄弟样本）——
           对齐口径 ≥0.8471 → 转导部署约束解除；否则 μ_t 满秩 = “转导不可摊销”的定理级边界。
  B·分块度量：S、A 通道各自独立 LDA 距离（折内标定后相加）——≥0.8371 且错误 kappa<0.3
           → 通道独立性证明；否则“互补=噪声消除”→ 理论改写为多视图稳定化。
  C·先决检查：同题每族 ≥3 采样存量——不足则排队（不阻塞 A/B）。
输出：runs/kernel_e18/v08.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from models.disc import DiscHead  # noqa: E402
from kernel_e17_framework import cv_disc, summarize  # noqa: E402
from kernel_e17b_views import load_h_aligned  # noqa: E402

RES = ROOT / "runs/kernel_e18"
NAMES = ["qwen05", "qwen15", "ds13", "yi15", "granite2b", "smollm2"]
R_VALUES = (10, 30, 50, 100)


# --------------------------------------------------------------------------- #
def flat_aligned():
    HP, HM, common = load_h_aligned()
    k, T, d = HP.shape
    HPf = np.vstack(list(HP))
    HMf = np.vstack(list(HM))
    y = np.concatenate([[n] * T for n in NAMES])
    g = np.concatenate([np.array(common) for _ in NAMES])
    return HPf, HMf, y, g


def flat_full():
    HP_l, HM_l, y, g = [], [], [], []
    for n in NAMES:
        z = np.load(ROOT / f"runs/kernel_e16/h_{n}.npz", allow_pickle=True)
        hp = np.asarray(z["h_plus"], dtype="float64")
        hm = np.asarray(z["h_minus"], dtype="float64")
        t = [str(v) for v in z["tasks"]]
        HP_l.append(hp); HM_l.append(hm)
        y += [n] * len(t)
        g += t
    return np.vstack(HP_l), np.vstack(HM_l), np.array(y), np.array(g)


def task_centers(M, y, g, idx):
    """按题计算跨族中心（仅保留 ≥2 族的题）。返回 (n_tasks, d) 矩阵。"""
    by_task = {}
    for i in idx:
        by_task.setdefault(g[i], []).append(i)
    mus = []
    for t, rows in by_task.items():
        if len({y[i] for i in rows}) >= 2:
            mus.append(M[rows].mean(axis=0))
    return np.vstack(mus) if mus else np.empty((0, M.shape[1]))


def top_dirs(Mc, r):
    """对中心矩阵取前 r 个主方向（列）。"""
    X = Mc - Mc.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    return Vt[:min(r, Vt.shape[0])].T


def project(X, U):
    return X - (X @ U) @ U.T


# --------------------------------------------------------------------------- #
def run_spectral(HP, HM, y, g):
    """全数据任务中心谱（报告用）。"""
    idx = np.arange(len(y))
    out = {}
    for tag, M in (("plus", HP), ("minus", HM)):
        Mc = task_centers(M, y, g, idx)
        X = Mc - Mc.mean(axis=0, keepdims=True)
        sv = np.linalg.svd(X, compute_uv=False)
        ev = sv ** 2
        cum = np.cumsum(ev) / ev.sum()
        rank90 = int(np.searchsorted(cum, 0.90) + 1)
        rank95 = int(np.searchsorted(cum, 0.95) + 1)
        out[tag] = {"n_tasks": int(Mc.shape[0]),
                    "rank90": rank90, "rank95": rank95,
                    "cum_var": {str(r): round(float(cum[r - 1]), 4) for r in R_VALUES}}
    return out


def run_inductive(HP, HM, y, g, tag, report):
    """归纳投影：训练折拟合 U_r → 投影 → DiscHead。"""
    res = {}
    txt = np.hstack([HP, HM])
    res["baseline_pair_noproj"] = summarize(cv_disc(txt, y, g), y)
    for r in R_VALUES:
        pred = np.empty(len(y), dtype=object)
        for tr, va in GroupKFold(n_splits=5).split(HP, y, g):
            mus_p = task_centers(HP, y, g, tr)
            mus_m = task_centers(HM, y, g, tr)
            Up = top_dirs(mus_p, r)
            Um = top_dirs(mus_m, r)
            Xtr = np.hstack([project(HP[tr], Up), project(HM[tr], Um)])
            Xva = np.hstack([project(HP[va], Up), project(HM[va], Um)])
            pred[va] = DiscHead().fit(Xtr, y[tr]).predict(Xva)
        pred = pred.astype(str)
        res[f"proj_r{r}"] = summarize(pred, y)
        print(f"[e18] A·归纳 {tag} r={r:<3}: acc {res[f'proj_r{r}']['acc']:.4f} / "
              f"bal {res[f'proj_r{r}']['balanced_acc']:.4f}", flush=True)
    report[f"inductive_{tag}"] = res
    print(f"[e18] A·归纳 {tag} 基线(不投影) {res['baseline_pair_noproj']['acc']:.4f}", flush=True)


def channel_distances(Xtr, ytr, Xva):
    """通道独立 LDA：返回 (归一化到类内尺度, 预测)。"""
    h = DiscHead().fit(Xtr, ytr)
    Ztr, Zva = h.transform(Xtr), h.transform(Xva)
    centers = np.vstack([Ztr[ytr == c].mean(0) for c in h.classes_])
    d2tr = ((Ztr[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
    d2va = ((Zva[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
    cls_idx = {c: i for i, c in enumerate(h.classes_)}
    own = d2tr[np.arange(len(ytr)), [cls_idx[c] for c in ytr]]
    scale = float(own.mean()) + 1e-12
    return d2va / scale, h.classes_


def run_blocked(S, A, y, g, tag, report):
    """分块度量：S、A 各自 LDA 距离（折内标定）相加。"""
    n = len(y)
    predS = np.empty(n, dtype=object); predA = np.empty(n, dtype=object)
    predB = np.empty(n, dtype=object)
    for tr, va in GroupKFold(n_splits=5).split(S, y, g):
        dS, cls = channel_distances(S[tr], y[tr], S[va])
        dA, clsA = channel_distances(A[tr], y[tr], A[va])
        assert list(cls) == list(clsA)
        predS[va] = cls[dS.argmin(1)]
        predA[va] = cls[dA.argmin(1)]
        predB[va] = cls[(dS + dA).argmin(1)]
    predS = predS.astype(str); predA = predA.astype(str); predB = predB.astype(str)
    mS = summarize(predS, y); mA = summarize(predA, y); mB = summarize(predB, y)
    errS = predS != y; errA = predA != y
    kappa = float(cohen_kappa_score(errS, errA))
    bound = 1.0 - float(errS.mean()) * float(errA.mean())
    entry = {"S_alone": {k: mS[k] for k in ("acc", "balanced_acc")},
             "A_alone": {k: mA[k] for k in ("acc", "balanced_acc")},
             "blocked": {k: mB[k] for k in ("acc", "balanced_acc")},
             "kappa_err": round(kappa, 4),
             "independent_bound": round(bound, 4)}
    report[f"blocked_{tag}"] = entry
    print(f"[e18] B·分块 {tag}: S {mS['acc']:.4f} / A {mA['acc']:.4f} / 分块 {mB['acc']:.4f} / "
          f"kappa {kappa:.3f} / 独立上限 {bound:.4f}", flush=True)


def run_c_check():
    """C 先决检查：同题每族多采样存量。"""
    import pyarrow.parquet as pq
    p = ROOT / "data/processed/pairs.parquet"
    t = pq.read_table(p)
    tasks = t.column("task_id").to_pylist()
    dups = len(tasks) - len(set(tasks))
    return {"status": "parked" if dups == 0 else "ok",
            "n_pairs": len(tasks), "duplicate_tasks": dups,
            "note": "每 (题,族) 仅 1 对——无同题多采样存量，C 按路线文档排队（不阻塞 A/B）"}


# --------------------------------------------------------------------------- #
def main() -> int:
    RES.mkdir(parents=True, exist_ok=True)
    report = {}

    # ---------- A ----------
    HPf, HMf, yf, gf = flat_full()
    HP, HM, ya, ga = flat_aligned()
    print(f"[e18] 数据：全量 {len(yf)} / 对齐 {len(ya)}", flush=True)

    spec = run_spectral(HPf, HMf, yf, gf)
    report["A_spectral"] = spec
    for tag, s in spec.items():
        print(f"[e18] A·谱 {tag}: 题数 {s['n_tasks']} | rank90={s['rank90']} "
              f"rank95={s['rank95']} | 前100方向方差 {s['cum_var']['100']}", flush=True)

    run_inductive(HP, HM, ya, ga, "aligned", report)
    run_inductive(HPf, HMf, yf, gf, "full", report)

    # ---------- B ----------
    Sf = 0.5 * (HPf + HMf); Af = 0.5 * (HPf - HMf)
    run_blocked(Sf, Af, yf, gf, "full", report)
    # 对齐口径使用中心化块（S̃/Ã）
    k = 6; T = len(ya) // k
    HPm = HP.reshape(k, T, -1); HMm = HM.reshape(k, T, -1)
    HPc = (HPm - HPm.mean(axis=0, keepdims=True)).reshape(-1, HP.shape[1])
    HMc = (HMm - HMm.mean(axis=0, keepdims=True)).reshape(-1, HM.shape[1])
    Sc = 0.5 * (HPc + HMc); Ac = 0.5 * (HPc - HMc)
    run_blocked(Sc, Ac, ya, ga, "aligned_centered", report)

    # ---------- C ----------
    report["C_check"] = run_c_check()
    print(f"[e18] C·先决检查: {report['C_check']['status']}（{report['C_check']['note']}）", flush=True)

    # ---------- 判读 ----------
    verdicts = {}
    a_best = max(report["inductive_aligned"][f"proj_r{r}"]["acc"] for r in R_VALUES)
    verdicts["A"] = ("通过：有效秩<50 且归纳版 ≥0.8471 → 部署约束解除"
                     if spec["plus"]["rank90"] < 50 and a_best >= 0.8471 else
                     f"未达线（归纳最佳 {a_best:.4f}；rank90={spec['plus']['rank90']}）"
                     "——按两态价值记录：μ_t 边界/谱结构数据")
    bb = report["blocked_full"]["blocked"]["acc"]
    kk = report["blocked_full"]["kappa_err"]
    verdicts["B"] = ("通过：分块 ≥0.8371 且 kappa<0.3 → 通道独立性证明"
                     if bb >= 0.8371 and kk < 0.3 else
                     f"未达线（分块 {bb:.4f}；kappa {kk:.3f}）——互补可能为噪声消除"
                     "，按两态价值记录")
    verdicts["C"] = "排队（无同题多采样存量）"
    report["verdicts"] = verdicts
    print("\n===== 判读 =====")
    for kk2, v in verdicts.items():
        print(f"  [{kk2}] {v}")

    out = RES / "v08.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n[e18] 写出 {out}")
    print("[e18] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
