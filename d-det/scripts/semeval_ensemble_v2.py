#!/usr/bin/env python
"""集成 v2：全成员 LightGBM 栈 + 层级化（Human 门 + 机器 10 类）+ 先验修正。

B 主口径（val-fit：cv5 估 val，全 val 拟合→test）：
  变体 full / core / hier / prior / 对照单模型。
C 刷新：同法（成员：c 直连 probs、z8_c、ngram_c、s1/s2、stats1/2）。
成员文件缺失自动跳过；行对齐用 y 校验。输出 runs/semeval_ensemble/results.json。
"""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pyarrow.parquet as pq
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
FEAT = ROOT / "runs/semeval_zeroshot/feat"
ZDIR = ROOT / "runs/semeval_r/z8"
Z2 = ROOT / "runs/semeval_r/z8v2"
PDIR = ROOT / "runs/semeval_r"
SDIR = ROOT / "data/processed/semeval"
OUT = ROOT / "runs/semeval_ensemble"

# B 任务神经 probs 成员：tag -> (val 文件, 优先 probs 键)
B_NEURAL = [
    ("b_main", PDIR / "probs_b_{s}.npz"),
    ("b_r4", PDIR / "probs_b_r4_{s}.npz"),
    ("b_r8", PDIR / "probs_b_r8_{s}.npz"),
    ("b_r11", PDIR / "probs_b_r11_{s}.npz"),
    ("b_r8i", PDIR / "probs_b_r8init_{s}.npz"),
    ("qwen_b", ROOT / "runs/semeval_qwen_b/probs_{s}.npz"),
    ("qwen_bbig", ROOT / "runs/semeval_qwen_b_big/probs_{s}.npz"),
    ("ctx_big", ROOT / "runs/semeval_b_big/probs_{s}.npz"),
    ("qwen7b", ROOT / "runs/semeval_qwen7b_b/probs_{s}.npz"),
    ("ds_b", ROOT / "runs/semeval_ds_b/probs_{s}.npz"),
]
C_NEURAL = [
    ("c_main", PDIR / "probs_c_{s}.npz"),
    ("qwen_c", ROOT / "runs/semeval_qwen_c/probs_{s}.npz"),
]


def lab(task: str, split: str) -> np.ndarray:
    return np.asarray(pq.read_table(SDIR / f"{task}_{split}.parquet",
                                    columns=["label"]).column("label").to_pylist())


def npz_probs(p: Path):
    if not p.exists():
        return None
    d = np.load(p, allow_pickle=True)
    for k in ("probs", "probs_val", "probs_test"):
        if k in d.files:
            return d[k].astype("float32")
    return None


def member_block(task: str, split: str, y_ref: np.ndarray):
    """返回 (X, names, singles)。singles: tag->(probs, y) 供对照。"""
    blocks, names, singles = [], [], {}

    def add(name, arr):
        if arr is None:
            return
        arr = np.asarray(arr, dtype="float32")
        if arr.shape[0] != len(y_ref):
            print(f"[ens] {name} 行数不符（{arr.shape[0]} vs {len(y_ref)}），跳过")
            return
        blocks.append(arr)
        names.append(name)

    for tag, tpl in (B_NEURAL if task == "b" else C_NEURAL):
        p = Path(str(tpl).format(s=split))
        arr = npz_probs(p)
        if arr is None:
            continue
        add(f"p:{tag}", arr)
        singles[tag] = arr

    # z8 v1 / v2（z + probs）
    for name, p, keys in (("z8_v1", ZDIR / f"{task}_{split}.npz", ("z8", "probs")),
                          ("z8v2", Z2 / f"{task}_{split}_base8.npz", ("z8", "probs")),
                          ("z8v2c", Z2 / f"{task}_{split}_center.npz", ("probs",))):
        if p.exists():
            d = np.load(p, allow_pickle=True)
            for k in keys:
                if k in d.files:
                    a = d[k].astype("float32")
                    if a.ndim == 2 and a.shape[0] > 10 and a.shape[1] > a.shape[0] * 10:
                        continue  # 疑似 h
                    add(f"{name}.{k}", a)

    # ngram
    for tag, p in ((f"ng_{task}", PDIR / f"ngram_{task}_{split}.npz"),):
        arr = npz_probs(p)
        if arr is not None:
            add(f"p:{tag}", arr)
            singles[tag] = arr

    # s1/s2
    fp = FEAT / f"{task}_{split}.npz"
    if fp.exists():
        d = np.load(fp, allow_pickle=True)
        add("s1", d["s1"].reshape(-1, 1).astype("float32"))
        add("s2", d["s2"].reshape(len(d["y"]), -1).astype("float32"))
    # stats1 / stats2
    for name, p in (("st1", SDIR / f"{task}_{split}_stats.npz"),
                    ("st2", SDIR / f"{task}_{split}_stats2.npz")):
        if p.exists():
            add(name, np.load(p)["X"].astype("float32"))
    X = np.hstack(blocks) if blocks else None
    return X, names, singles


def lgb_fit(X, y, seed=0):
    clf = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31,
                             min_child_samples=20, subsample=0.9, colsample_bytree=0.9,
                             class_weight="balanced", random_state=seed, verbose=-1)
    clf.fit(X, y)
    return clf


def f1(y, p) -> float:
    return round(float(f1_score(y, np.asarray(p).argmax(1), average="macro")), 4)


def cv_predict(X, y, n=5):
    oof = np.zeros((len(y), int(np.max(y)) + 1))
    skf = StratifiedKFold(n_splits=n, shuffle=True, random_state=0)
    for tr, va in skf.split(X, y):
        oof[va] = lgb_fit(X[tr], y[tr]).predict_proba(X[va])
    return oof


def prior_adjust(P, pi_s, pi_t):
    P = np.clip(P.astype("float64"), 1e-9, 1)
    return (P * (pi_t / pi_s)[None, :])


def run_task(task: str, res: dict):
    yv, yt = lab(task, "val"), lab(task, "test")
    Xv, names, _singles_v = member_block(task, "val", yv)
    Xt, _, singles = member_block(task, "test", yt)
    if Xv is None or Xt is None:
        print(f"[ens] {task}: 无可用成员", flush=True)
        return
    nC = int(max(yv.max(), yt.max())) + 1
    pi_s = np.bincount(yv, minlength=nC) / len(yv)          # 采样先验
    ytr_all = lab(task, "train")
    pi_t = np.bincount(ytr_all, minlength=nC) / len(ytr_all)  # 训练集频率≈真实分布
    out = {"members": names, "n_features": int(Xv.shape[1])}

    # 单模型对照
    out["singles"] = {tag: {"test_f1": f1(yt, p)} for tag, p in singles.items()}

    # A) full
    oof = cv_predict(Xv, yv)
    out["full_cv_val"] = f1(yv, oof)
    clf = lgb_fit(Xv, yv)
    Pte = clf.predict_proba(Xt)
    out["full_test"] = f1(yt, Pte)
    # D) full + 先验修正
    out["full_prior_test"] = f1(yt, prior_adjust(Pte, pi_s, pi_t))

    # B) core：去神经 probs（只 z/ng/base/stats）
    core_idx = [i for i, n in enumerate(names) if not n.startswith("p:") or
                n.startswith("p:ng_")]
    if len(core_idx) >= 3:
        Xc_v, Xc_t = Xv[:, core_idx], Xt[:, core_idx]
        out["core_cv_val"] = f1(yv, cv_predict(Xc_v, yv))
        out["core_test"] = f1(yt, lgb_fit(Xc_v, yv).predict_proba(Xc_t))

    # C) hier：门 + 机器 10/3 类
    def hier(P_test_source, Xa, Xb_):
        gate = lgb_fit(Xa, (yv == 0).astype(int))
        pg = gate.predict_proba(Xb_)[:, 1]
        m = yv > 0
        mclf = lgb_fit(Xa[m], yv[m] - 1)
        pm = mclf.predict_proba(Xb_)
        P = np.zeros((len(Xb_), nC))
        P[:, 0] = pg
        P[:, 1:] = (1 - pg)[:, None] * pm
        return P

    def hier_cv(Xa):
        oof = np.zeros((len(yv), nC))
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        for tr, va in skf.split(Xa, yv):
            Xtr, Xva = Xa[tr], Xa[va]
            ytr_ = yv[tr]
            gate = lgb_fit(Xtr, (ytr_ == 0).astype(int))
            pg = gate.predict_proba(Xva)[:, 1]
            m = ytr_ > 0
            mclf = lgb_fit(Xtr[m], ytr_[m] - 1)
            pm = mclf.predict_proba(Xva)
            oof[va, 0] = pg
            oof[va, 1:] = (1 - pg)[:, None] * pm
        return oof

    out["hier_cv_val"] = f1(yv, hier_cv(Xv))
    Phe = hier(None, Xv, Xt)
    out["hier_test"] = f1(yt, Phe)
    out["hier_prior_test"] = f1(yt, prior_adjust(Phe, pi_s, pi_t))
    res[task] = out
    print(f"[ens] {task}: full val(cv)={out['full_cv_val']} test={out['full_test']} | "
          f"prior={out['full_prior_test']} | hier val={out['hier_cv_val']} "
          f"test={out['hier_test']} prior={out['hier_prior_test']}", flush=True)


def main() -> int:
    res = {}
    for task in ("b", "c"):
        try:
            run_task(task, res)
        except Exception as e:
            res[task] = {"error": f"{type(e).__name__}: {e}"}
            print(f"[ens] {task} 失败：{e}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "results.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print("[ens] done -> runs/semeval_ensemble/results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
