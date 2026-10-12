# -*- coding: utf-8 -*-
"""d-det-v2 v0.1 · 内容参考中心

- task_centers：按任务权重估计公共中心 μ（F: Human+族等权 1/8；D: Human 1/2、族 1/14）；
- RidgeRef：标准化 ridge（多输出），alpha 仅在内层按 task 分组选择；
- build_refs：外层 5 折交叉拟合 —— 每折 g 只看 4/5 折，生成其余 1 折的 OOF 参考；
  推理保留 5 个参考器（分别构造校正 logits 后平均，协议见《设计指导》§5.3）；
- global_mean：任务均匀的全局训练均值（B3 对照用）。

数值实现：numpy float32；X 标准化统计来自该折的训练部分。
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.model_selection import GroupKFold


def task_centers(h: torch.Tensor, recs, idx, task: str) -> dict:
    """按任务加权中心。h: [N, d]（与 recs 全表对齐），idx: 参与行下标。

    F: Human=1/8、每 family=1/8（各 1 generator）；D: Human=1/2、每族=1/14。
    缺行任务按实际行权重重归一化。
    """
    groups = {}
    for i in idx:
        groups.setdefault(recs[i]["tid"], []).append(i)
    centers = {}
    for tid, rows in groups.items():
        ws = []
        for i in rows:
            r = recs[i]
            if task == "F":
                ws.append(1.0 / 8)
            else:
                ws.append(0.5 if r["kind"] == "human" else 0.5 / 7)
        w = torch.tensor(ws, dtype=torch.float32)
        w = w / w.sum()
        centers[tid] = (h[rows] * w.unsqueeze(1)).sum(0)
    return centers


class RidgeRef:
    """标准化多输出 ridge；alpha 在内层以 GroupKFold(按 task) 选择（最小化中心预测 MSE）。"""

    def __init__(self, alphas, inner_folds=3):
        self.alphas = list(alphas)
        self.inner_folds = inner_folds
        self.mu = None
        self.sd = None
        self.W = None
        self.ym = None
        self.alpha = None

    def _fit_solve(self, Xz, Yc, alpha):
        d = Xz.shape[1]
        A = Xz.T @ Xz + alpha * np.eye(d, dtype=np.float32)
        W = np.linalg.solve(A, Xz.T @ Yc)
        return W

    def fit(self, X, Y, groups):
        X = np.asarray(X, dtype=np.float32)
        Y = np.asarray(Y, dtype=np.float32)
        self.mu = X.mean(0)
        self.sd = X.std(0) + 1e-6
        Xz = (X - self.mu) / self.sd
        ym = Y.mean(0)
        Yc = Y - ym

        # ---- 内层选择 alpha（按 task 分组）----
        uniq = np.unique(groups)
        k = int(min(self.inner_folds, len(uniq)))
        best, best_a = None, None
        if k >= 2:
            gkf = GroupKFold(n_splits=k)
            for a in self.alphas:
                errs, wsz = [], 0
                for tr, va in gkf.split(Xz, Yc, groups):
                    W = self._fit_solve(Xz[tr], Yc[tr], a)
                    pred = Xz[va] @ W
                    errs.append(np.mean((pred - Yc[va]) ** 2) * len(va))
                    wsz += len(va)
                e = sum(errs) / max(wsz, 1)
                if best is None or e < best:
                    best, best_a = e, a
        else:
            best_a = self.alphas[len(self.alphas) // 2]
        self.alpha = float(best_a)

        # ---- 全训练部分拟合（Xz 在训练部分零均值 ⇒ 截距即 ym）----
        self.W = self._fit_solve(Xz, Yc, self.alpha)
        self.ym = ym
        return self

    def predict(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        return ((X - self.mu) / self.sd) @ self.W + self.ym


def build_refs(task: str, recs, idx_main, s: torch.Tensor, h: torch.Tensor,
               folds: int, alphas, inner_folds):
    """5 折 OOF 参考。返回 (oof_m, refs, m_global, centers)。

    - s: [len(recs), ds] 全表对齐；h: [len(recs), d] 全表对齐（全局下标索引）；
    - oof_m: [len(idx_main), d] float32，每行=其所属折的 g 给出的参考（头训练用）；
    - refs: 5 个 RidgeRef（推理用，分别构造校正 logits 后平均）；
    - m_global: [1, d] 任务均匀全局均值；
    - centers: {tid: μ}（诊断/记录用）。
    """
    d = h.shape[-1]
    centers = task_centers(h, recs, idx_main, task)
    X = s[idx_main].numpy().astype(np.float32)
    Y = np.stack([centers[recs[i]['tid']].numpy() for i in idx_main]).astype(np.float32)
    groups = np.array([recs[i]['fold'] for i in idx_main])
    assert (groups >= 0).all(), "OOF 折只对 train 主训练 task 定义"

    oof = np.zeros((len(idx_main), d), dtype=np.float32)
    refs = {}
    for j in range(folds):
        tr = groups != j
        va = groups == j
        if tr.sum() == 0 or va.sum() == 0:
            continue
        g = RidgeRef(alphas, inner_folds).fit(X[tr], Y[tr], groups[tr])
        refs[j] = g
        oof[va] = g.predict(X[va])
    assert len(refs) >= 2, f"折数不足：{len(refs)}/{folds}"
    if len(refs) < folds:  # 冒烟/小样本时允许缺折（记录于调用方）
        pass

    m_global = torch.stack(list(centers.values())).mean(0, keepdim=True)
    return torch.from_numpy(oof), refs, m_global, centers


def predict_refs(refs, s: torch.Tensor) -> torch.Tensor:
    """用 5 个参考器预测任意样本：返回 [N, 5, d]。"""
    X = s.numpy().astype(np.float32)
    outs = [torch.from_numpy(refs[j].predict(X)) for j in sorted(refs)]
    return torch.stack(outs, dim=1)
