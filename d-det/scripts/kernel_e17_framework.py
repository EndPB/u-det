#!/usr/bin/env python
"""E17 框架/内核优化筛查（不新增数据；纯 CPU）。

在既有 6 族 Δ（runs/v0.5_disc/d_*.npz）与双侧特征（runs/kernel_e16/h_*.npz）上，
对"框架级"改进做**同折**对照筛查（按题 GroupKFold=5，与全部历史口径一致）：

  A. 中心化变体：跨族均值（现基线 0.7740）/ 留一均值 LOO（排除自身）/ 跨族中位数
  B. 中心化 + 子空间 QDA（现最佳 0.7791）与**概率集成**（DiscHead⊕QDA）
  C. **级联细化**：6 路判别 → 对易混对（granite2b,smollm2）、（qwen05,qwen15）用专用二分类重判
  D. **层次归因**：lab（qwen/ds/yi/ibm/hf）→ generation（qwen 内二分类）
  E. **视图扩展**（全量口径）：[Δ;h⁺]、[Δ;h⁻] vs 纯 Δ（0.7626）——检验"绝对位置"信息是否有增量

判读（预注册）：任一变体 aligned acc ≥ 0.785（现最佳 +0.5pt），或易混对互混降 ≥25% 且 bal 不降
→ 进正式验证；否则记为负结果。

输出：runs/kernel_e17/framework.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.disc import DiscHead  # noqa: E402

OUT = ROOT / "runs/v0.5_disc"
F16 = ROOT / "runs/kernel_e16"
RES = ROOT / "runs/kernel_e17"
NAMES = ["qwen05", "qwen15", "ds13", "yi15", "granite2b", "smollm2"]
CONF_PAIRS = [("granite2b", "smollm2"), ("qwen05", "qwen15")]
LAB = {"qwen05": "qwen", "qwen15": "qwen", "ds13": "ds", "yi15": "yi",
       "granite2b": "ibm", "smollm2": "hf"}
LAB2FAM = {"ds": "ds13", "yi": "yi15", "ibm": "granite2b", "hf": "smollm2"}


# --------------------------------------------------------------------------- #
def load_aligned():
    """返回家族块状 (M (6,T,768), y, g)。"""
    common = None
    mats = []
    for n in NAMES:
        z = np.load(OUT / f"d_{n}.npz", allow_pickle=True)
        d = np.asarray(z["d"], dtype="float64")
        t = [str(v) for v in z["tasks"]]
        common = set(t) if common is None else (common & set(t))
        mats.append((d, t))
    common = sorted(common)
    M = []
    for d, t in mats:
        idx = {tt: i for i, tt in enumerate(t)}
        M.append(d[[idx[tt] for tt in common]])
    M = np.stack(M, axis=0)                     # (6, T, 768)
    y = np.concatenate([[n] * len(common) for n in NAMES])
    g = np.concatenate([np.array(common) for _ in NAMES])
    return M, y, g


def load_full():
    """全量口径：Δ / h+ / h− 家族块状拼接（同序），返回 (Xdelta, Xhp, Xhm, y, g)。"""
    D, HP, HM, y, g = [], [], [], [], []
    for n in NAMES:
        zd = np.load(OUT / f"d_{n}.npz", allow_pickle=True)
        zh = np.load(F16 / f"h_{n}.npz", allow_pickle=True)
        dd = np.asarray(zd["d"], dtype="float64")
        hp = np.asarray(zh["h_plus"], dtype="float64")
        hm = np.asarray(zh["h_minus"], dtype="float64")
        t = [str(v) for v in zd["tasks"]]
        th = [str(v) for v in zh["tasks"]]
        assert t == th, f"{n} 任务顺序不一致"
        D.append(dd); HP.append(hp); HM.append(hm)
        y += [n] * len(t); g += t
    return (np.vstack(D), np.vstack(HP), np.vstack(HM),
            np.array(y), np.array(g))


def summarize(pred, y):
    rec = {str(c): round(float(((pred == c) & (y == c)).sum()) / max(1, int((y == c).sum())), 3)
           for c in np.unique(y)}
    conf = {}
    for a, b in CONF_PAIRS:
        if (y == a).any() and (y == b).any():
            conf[f"{a}->{b}"] = int(((y == a) & (pred == b)).sum())
            conf[f"{b}->{a}"] = int(((y == b) & (pred == a)).sum())
    return {"acc": round(float(accuracy_score(y, pred)), 4),
            "balanced_acc": round(float(balanced_accuracy_score(y, pred)), 4),
            "recall": rec, "conf": conf}


def folds_of(X, y, g):
    return list(GroupKFold(n_splits=5).split(X, y, g))


def cv_disc(X, y, g):
    pred = np.empty(len(y), dtype=object)
    for tr, va in folds_of(X, y, g):
        pred[va] = DiscHead().fit(X[tr], y[tr]).predict(X[va])
    return pred.astype(str)


def probs_matrix(head, X):
    """把 DiscHead.predict_proba 对齐到 NAMES 列序。"""
    P = head.predict_proba(X)
    out = np.zeros((len(X), len(NAMES)))
    for j, c in enumerate(head.classes_):
        out[:, NAMES.index(str(c))] = P[:, j]
    return out


def run_ensemble_qda(X, y, g, shrink=0.05):
    """返回 (pred_disc, pred_qda, pred_ens)。"""
    n = len(y)
    p1 = np.empty(n, dtype=object); p2 = np.empty(n, dtype=object)
    pe = np.empty(n, dtype=object)
    for tr, va in folds_of(X, y, g):
        h = DiscHead().fit(X[tr], y[tr])
        p1[va] = h.predict(X[va])
        Ztr, Zva = h.transform(X[tr]), h.transform(X[va])
        q = QDA(solver="eigen", shrinkage=shrink).fit(Ztr, y[tr])
        p2[va] = q.predict(Zva)
        Q = q.predict_proba(Zva)
        Pm = np.zeros((len(va), len(NAMES)))
        for j, c in enumerate(q.classes_):
            Pm[:, NAMES.index(str(c))] = Q[:, j]
        ens = 0.5 * probs_matrix(h, X[va]) + 0.5 * Pm
        pe[va] = [NAMES[i] for i in ens.argmax(1)]
    return p1.astype(str), p2.astype(str), np.array([str(v) for v in pe])


def run_cascade(X, y, g, pairs=CONF_PAIRS):
    """6 路 → （预测落进易混对时）二分类专用头重判。"""
    n = len(y)
    p6 = np.empty(n, dtype=object); pc = np.empty(n, dtype=object)
    for tr, va in folds_of(X, y, g):
        h6 = DiscHead().fit(X[tr], y[tr])
        base = h6.predict(X[va])
        p6[va] = base
        cur = np.array(base, dtype=object)
        for a, b in pairs:
            mtr = np.isin(y[tr], [a, b])
            hb = DiscHead().fit(X[tr][mtr], y[tr][mtr])
            sel = np.isin(base, [a, b])
            if sel.any():
                cur[sel] = hb.predict(X[va][sel])
        pc[va] = cur
    return p6.astype(str), np.array([str(v) for v in pc])


def run_hier(X, y, g):
    """lab（5 类）→ generation（qwen 内二分类）。"""
    labs = np.array([LAB[v] for v in y])
    n = len(y)
    pl = np.empty(n, dtype=object); pf = np.empty(n, dtype=object)
    for tr, va in folds_of(X, y, g):
        hl = DiscHead().fit(X[tr], labs[tr])
        lp = hl.predict(X[va])
        pl[va] = lp
        mq = np.isin(y[tr], ["qwen05", "qwen15"])
        hq = DiscHead().fit(X[tr][mq], y[tr][mq])
        out = np.empty(len(va), dtype=object)
        for i, lab in enumerate(lp):
            if lab == "qwen":
                out[i] = hq.predict(X[[va[i]]])[0]
            else:
                out[i] = LAB2FAM[lab]
        pf[va] = out
    return pl.astype(str), np.array([str(v) for v in pf])


def run_bags(X, y, g, B=20, frac=0.5, seed=0):
    rng = np.random.default_rng(seed)
    n, d = X.shape
    votes = np.zeros((n, len(NAMES)))
    for tr, va in folds_of(X, y, g):
        for _ in range(B):
            cols = rng.choice(d, size=int(d * frac), replace=False)
            hb = DiscHead().fit(X[tr][:, cols], y[tr])
            p = hb.predict(X[va][:, cols])
            for i, t in enumerate(va):
                votes[t, NAMES.index(str(p[i]))] += 1
    return np.array([NAMES[i] for i in votes.argmax(1)])


# --------------------------------------------------------------------------- #
def main() -> int:
    RES.mkdir(parents=True, exist_ok=True)
    report = {}
    M, y, g = load_aligned()
    k, T, d = M.shape
    print(f"[e17] 对齐：{k} 族 × {T} 题 = {k*T} 样本", flush=True)

    # ---- A. 中心化变体 ----
    m_all = M.mean(axis=0, keepdims=True)
    X_all = np.vstack(list(M - m_all))
    Mloo = np.empty_like(M)
    for kk in range(k):
        other = np.delete(M, kk, axis=0).mean(axis=0)
        Mloo[kk] = M[kk] - other
    X_loo = np.vstack(list(Mloo))
    m_med = np.median(M, axis=0, keepdims=True)
    X_med = np.vstack(list(M - m_med))

    report["center_mean"] = summarize(cv_disc(X_all, y, g), y)
    print(f"[e17] 中心化-均值(基线): {report['center_mean']['acc']}", flush=True)
    report["center_loo"] = summarize(cv_disc(X_loo, y, g), y)
    print(f"[e17] 中心化-LOO     : {report['center_loo']['acc']}", flush=True)
    report["center_median"] = summarize(cv_disc(X_med, y, g), y)
    print(f"[e17] 中心化-中位数  : {report['center_median']['acc']}", flush=True)

    # ---- B. QDA 与集成 ----
    p1, p2, pe = run_ensemble_qda(X_all, y, g)
    report["qda"] = summarize(p2, y)
    report["ens_disc_qda"] = summarize(pe, y)
    print(f"[e17] 中心化+QDA      : {report['qda']['acc']}", flush=True)
    print(f"[e17] 集成(Disc⊕QDA) : {report['ens_disc_qda']['acc']}", flush=True)

    # ---- C. 级联细化 ----
    pc6, pcas = run_cascade(X_all, y, g)
    report["cascade"] = summarize(pcas, y)
    print(f"[e17] 级联细化       : {report['cascade']['acc']} conf={report['cascade']['conf']}", flush=True)

    # ---- D. 层次归因 ----
    plab, phier = run_hier(X_all, y, g)
    report["hier"] = summarize(phier, y)
    report["hier"]["lab_acc"] = round(float(accuracy_score([LAB[v] for v in y], plab)), 4)
    print(f"[e17] 层次归因       : {report['hier']['acc']}（lab 层 {report['hier']['lab_acc']}）", flush=True)

    # ---- E. 视图扩展（全量口径）----
    Xd, Xhp, Xhm, yf, gf = load_full()
    ref = DiscHead().cross_val(Xd, yf, gf)
    report["full_delta_ref"] = {k2: round(float(v), 4) for k2, v in ref.items()}
    print(f"[e17] 全量 Δ 参考    : {report['full_delta_ref']}", flush=True)
    for tag, Xv in (("d_plus_hp", np.hstack([Xd, Xhp])),
                    ("d_plus_hm", np.hstack([Xd, Xhm])),
                    ("d_hp_hm", np.hstack([Xd, Xhp, Xhm]))):
        r = DiscHead().cross_val(Xv, yf, gf)
        report[tag] = {k2: round(float(v), 4) for k2, v in r.items()}
        print(f"[e17] 视图 {tag:<10}: acc {report[tag]['acc']}", flush=True)

    # ---- F. 袋装（对阵中心化基线）----
    pbag = run_bags(X_all, y, g)
    report["bagged_disc"] = summarize(pbag, y)
    print(f"[e17] 袋装(20×50%)  : {report['bagged_disc']['acc']}", flush=True)

    out = RES / "framework.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n[e17] 写出 {out}")
    print("[e17] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
