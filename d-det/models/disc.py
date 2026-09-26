"""判别式归因头（s2-v2 核心组件；依据内核实验 E10–E14）。

输入 = 配对位移 Δ = h(x+) − h(x−)（任意编码器的池化特征差，一题一向量）。
学   = 广义 LDA（类内白化 × 类间散度最大化；shrinkage 正则）→ r ≤ 族数−1 维判别子空间。
用   = 每族原型（投影空间中心）→ 最近原型归因 / softmax 概率。

设计依据（负结果一并固化）：
- 聚类形态证伪（E10/E12：Δ 空间不成簇——KMeans 无效）；
- 残差化非必需（E11：判别器自动找正交切分）；
- 全维白化反伤（E12：不做全维 Σ_W 白化，只做判别降维）；
- 小样本闭式 > 梯度（E13）：默认闭式解（LDA+shrinkage），梯度版留作大样本选项；
- 泛化 gap 是主瓶颈（in-sample 0.999 / 按题跨题 0.88）：评估必须用按题 GroupKFold。

接口约定（为扩族生成留好）：
- ``fit(deltas, families)``：任意族数 k（r = min(k−1, d_cap)）；
- 注册新族：直接重拟合（或 add_prototype 增量加原型后用最近原型判定）；
- Δ 的来源自由：任何编码器（本项目为 CodeT5 v1.0 池化 h；Qwen 亦可）。
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


class DiscHead:
    def __init__(self, shrink="auto", d_cap: int = 32, temperature: float = 1.0):
        self.shrink = shrink
        self.d_cap = d_cap
        self.temperature = temperature
        self.scaler: StandardScaler | None = None
        self.lda: LinearDiscriminantAnalysis | None = None
        self.classes_: np.ndarray | None = None
        self.centers_: np.ndarray | None = None  # (k, r) 投影空间原型
        self.families_: list | None = None

    # ------------------------------------------------------------------ #
    def fit(self, deltas: np.ndarray, families) -> "DiscHead":
        X = np.asarray(deltas, dtype="float64")
        y = np.asarray(families)
        self.scaler = StandardScaler().fit(X)
        Xs = self.scaler.transform(X)
        n_comp = min(len(set(y.tolist())) - 1, self.d_cap, X.shape[1])
        self.lda = LinearDiscriminantAnalysis(solver="eigen", shrinkage=self.shrink,
                                              n_components=max(1, n_comp)).fit(Xs, y)
        self.classes_ = self.lda.classes_
        Z = self.lda.transform(Xs)
        self.centers_ = np.vstack([Z[y == c].mean(0) for c in self.classes_])
        self.families_ = [str(c) for c in self.classes_]
        return self

    def transform(self, deltas: np.ndarray) -> np.ndarray:
        return self.lda.transform(self.scaler.transform(np.asarray(deltas, dtype="float64")))

    def predict(self, deltas: np.ndarray) -> np.ndarray:
        Z = self.transform(deltas)
        d2 = ((Z[:, None, :] - self.centers_[None, :, :]) ** 2).sum(-1)
        return self.classes_[d2.argmin(1)]

    def predict_proba(self, deltas: np.ndarray) -> np.ndarray:
        Z = self.transform(deltas)
        d2 = ((Z[:, None, :] - self.centers_[None, :, :]) ** 2).sum(-1)
        logits = -d2 / max(self.temperature, 1e-9)
        logits -= logits.max(1, keepdims=True)
        p = np.exp(logits)
        return p / p.sum(1, keepdims=True)

    # ------------------------------------------------------------------ #
    def cross_val(self, deltas: np.ndarray, families, groups=None, n_splits: int = 5) -> dict:
        X = np.asarray(deltas, dtype="float64")
        y = np.asarray(families)
        if groups is None:
            groups = np.arange(len(y))
        acc, bacc = [], []
        for tr, va in GroupKFold(n_splits=n_splits).split(X, y, groups):
            h = DiscHead(shrink=self.shrink, d_cap=self.d_cap).fit(X[tr], y[tr])
            p = h.predict(X[va])
            acc.append(accuracy_score(y[va], p))
            bacc.append(balanced_accuracy_score(y[va], p))
        return {"acc": float(np.mean(acc)), "balanced_acc": float(np.mean(bacc)),
                "n_splits": n_splits, "n": int(len(y)), "n_classes": int(len(set(y.tolist())))}

    # ------------------------------------------------------------------ #
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> "DiscHead":
        with open(path, "rb") as f:
            return pickle.load(f)
