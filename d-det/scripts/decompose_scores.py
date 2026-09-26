"""跨规模分数分解：s1 单独 / s2 单独 / 融合的 AUC 拆解 + 交叉融合矩阵。

输入：scale_scores*.npz（probe_scale_verify.py 产出；键 hum/ai4/plus/minus，
      每行 [s1, s2, ntok]；人类池固定为 m4-test python 人类 315 条）。
输出：
  1) 逐 run 逐 scope 分量表：s1 单独（编码器是否单独受损的关键读数）、s2*、fused、
     s1+L、s1+s2+L —— L = log1p(token 数)，长度对照；
  2) 交叉融合矩阵（仅在同一 pair 文件的 run 之间可组合）：行 = s1 来源，列 = s2 来源。

用法：
  python scripts/decompose_scores.py 'runs/*/scale_scores.npz'
  python scripts/decompose_scores.py 'runs/*/scale_scores_ds13.npz'
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold


def fused(y, X, repeats=4):
    aucs = []
    for tr, te in RepeatedStratifiedKFold(n_splits=5, n_repeats=repeats,
                                          random_state=0).split(X, y):
        clf = LogisticRegression(max_iter=2000).fit(X[tr], y[tr])
        aucs.append(roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1]))
    return float(np.mean(aucs)), float(np.std(aucs))


def main() -> int:
    pattern = sys.argv[1] if len(sys.argv) > 1 else "runs/*/scale_scores.npz"
    paths = sorted(glob.glob(pattern))
    if not paths:
        print("没有匹配的 npz：", pattern)
        return 1
    runs = {}
    for p in paths:
        z = np.load(p)
        runs[Path(p).parent.name + "/" + Path(p).stem] = {k: z[k] for k in z.files}
    print("## 载入", list(runs))

    scopes = (("ref(m4ai)", "ai4"), ("instruct", "plus"), ("base", "minus"))
    for name, z in runs.items():
        print(f"\n== {name} ==")
        for label, key in scopes:
            hum, oth = z["hum"], z[key]
            y = np.r_[np.zeros(len(hum)), np.ones(len(oth))]
            s1 = np.r_[hum[:, 0], oth[:, 0]]
            s2 = np.r_[hum[:, 1], oth[:, 1]]
            lt = np.log1p(np.r_[hum[:, 2], oth[:, 2]])
            a1 = roc_auc_score(y, s1)
            a2 = roc_auc_score(y, s2)
            a2 = max(a2, 1 - a2)
            f2, f2s = fused(y, np.stack([s1, s2], 1))
            fL, _ = fused(y, np.stack([s1, lt], 1))
            f2L, _ = fused(y, np.stack([s1, s2, lt], 1))
            print(f"  {label:12s} s1={a1:.4f} s2*={a2:.4f} fused={f2:.4f}±{f2s:.4f} "
                  f"s1+L={fL:.4f} s1+s2+L={f2L:.4f}")

    # 交叉融合矩阵（要求同一 scope 的样本长度一致：即同一 pair 文件的 run 集合）
    keys = list(runs)
    for label, key in scopes:
        lens = {k: (len(runs[k]["hum"]), len(runs[k][key])) for k in keys}
        if len(set(lens.values())) != 1:
            print(f"\n#### 交叉融合（{label}）：跳过（各 run 样本长度不同：{lens}）")
            continue
        print(f"\n#### 交叉融合（{label} vs 人）：行=s1来源 列=s2来源 ####")
        print(" " * 20 + "".join(f"{k.split('/')[0][:16]:>18s}" for k in keys))
        for a in keys:
            ya = np.r_[np.zeros(len(runs[a]["hum"])), np.ones(len(runs[a][key]))]
            s1a = np.r_[runs[a]["hum"][:, 0], runs[a][key][:, 0]]
            row = []
            for b in keys:
                s2b = np.r_[runs[b]["hum"][:, 1], runs[b][key][:, 1]]
                f, _ = fused(ya, np.stack([s1a, s2b], 1))
                row.append(f)
            print(f"{a.split('/')[0][:18]:>18s}" + "".join(f"{v:>18.4f}" for v in row))
    print("\n注：s2* 取双向最优；融合用原始 s2（LR 自行定符号）；L=log1p(tokens)。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
