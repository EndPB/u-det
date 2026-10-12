#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""d-det-v2 v0.1 · C-P0 强控制（词法/表面/概率融合）

组件（全部在训练侧拟合；不读取 dev/test 拟合任何参数）：
  1) word TF-IDF + 逻辑回归
  2) char_wb TF-IDF + 逻辑回归
  3) 布局/元数据（长度、空白、缩进、大小写等 + 语言 one-hot，全部部署可得）+ 逻辑回归
  4) B1 模型概率（阶段一基线）

融合规则（v0.1 首批实现协议，记录于 README/报告）：
  - 组件在 train 主训练集上拟合；校准子集从未参与任何组件训练；
  - 融合权重在`训练内校准子集`上拟合 —— 对该子集而言四个组件的预测均为 OOF；
  - dev/test 不参与成员选择与权重拟合；同时报告单组件最强值。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data as D          # noqa: E402
import train as TR        # noqa: E402

T0 = time.time()


def layout_features(recs, idx):
    """布局/元数据数值特征（部署可得）+ 语言 one-hot。"""
    rows = []
    for i in idx:
        r = recs[i]
        code = r["code"]
        lines = code.splitlines()
        n = len(lines)
        lens = [len(l) for l in lines] or [0]
        blank = sum(1 for l in lines if not l.strip())
        ind = [len(l) - len(l.lstrip(" ")) for l in lines if l.strip()]
        chars = max(len(code), 1)
        feat = [
            len(code), n, float(np.mean(lens)), float(np.max(lens)), float(np.std(lens)),
            blank / max(n, 1),
            float(np.mean(ind)) if ind else 0.0, float(np.max(ind)) if ind else 0.0,
            code.count("\t") / chars, code.count(" ") / chars,
            sum(c.isdigit() for c in code) / chars,
            sum(c.isalpha() for c in code) / chars,
            sum(c.isupper() for c in code) / chars,
            float(code.count("//")), float(code.count("/*")), float(code.count("#")),
            float(code.count('"')), float(code.count("'")),
            len(code) / max(n, 1),
        ]
        onehot = [1.0 if r["lang"] == L else 0.0 for L in D.LANG_ORDER]
        rows.append(feat + onehot)
    return np.asarray(rows, dtype=np.float32)


def labels_of(recs, idx, task):
    if task == "F":
        return np.array([D.CLASSES_F.index(recs[i]["fam"]) for i in idx])
    return np.array([0 if recs[i]["kind"] == "human" else 1 for i in idx])


def main():
    ap = argparse.ArgumentParser(description="C-P0 强控制")
    ap.add_argument("--task", choices=["F", "D"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--work-root", default=None)
    args = ap.parse_args()

    cfg = D.load_cfg()
    wp = TR.work_paths(cfg, args.work_root)
    recs = D.load_records_file(wp["records"])

    idx_tr = D.subset_index(recs, "train", main_only=True)
    idx_calib = D.subset_index(recs, "train", calib=True)
    idx_dev = D.subset_index(recs, "dev")
    sets = {"train": idx_tr, "calib": idx_calib, "dev": idx_dev}
    if cfg["data"].get("test_read"):
        sets["test"] = D.subset_index(recs, "test")

    y = {k: labels_of(recs, v, args.task) for k, v in sets.items()}
    texts = {k: [recs[i]["code"] for i in v] for k, v in sets.items()}

    # ---- 组件 1/2：word / char TF-IDF + LR ----
    p0 = cfg["p0"]
    vec_w = TfidfVectorizer(ngram_range=(1, 2), min_df=p0["min_df"], max_features=p0["max_features"],
                            sublinear_tf=True)
    Xw = vec_w.fit_transform(texts["train"])
    clf_w = LogisticRegression(C=p0["C"], max_iter=3000)
    clf_w.fit(Xw, y["train"])
    vec_c = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=p0["min_df"],
                            max_features=p0["max_features"], sublinear_tf=True)
    Xc = vec_c.fit_transform(texts["train"])
    clf_c = LogisticRegression(C=p0["C"], max_iter=3000)
    clf_c.fit(Xc, y["train"])

    # ---- 组件 3：布局/元数据 ----
    Ltr = layout_features(recs, idx_tr)
    sc = StandardScaler().fit(Ltr)
    clf_l = LogisticRegression(C=p0["C"], max_iter=3000)
    clf_l.fit(sc.transform(Ltr), y["train"])

    # ---- 组件 4：B1 概率 ----
    b1_prob = {}
    ck = wp["ckpt"] / f"{args.task}_B1_s{args.seed}" / "best.pt"
    assert ck.exists(), f"缺少 B1 checkpoint：{ck}"
    st = torch.load(ck, map_location="cpu")
    model = TR.M.Stage1Model(str(D.resolve(cfg["paths"]["encoder"])), 8 if args.task == "F" else 1).cuda()
    model.encoder.load_state_dict(st["encoder"])
    model.head.load_state_dict(st["head"])
    tok = D.load_tokenizer(cfg)
    for k, v in sets.items():
        _, probs = TR.eval_stage1(model, tok, recs, v, args.task, cfg)
        b1_prob[k] = probs
    del model
    torch.cuda.empty_cache()

    # ---- 各组件预测 ----
    comp = {"word": {}, "char": {}, "layout": {}, "b1": {}}
    for k in sets:
        comp["word"][k] = clf_w.predict_proba(vec_w.transform(texts[k]))
        comp["char"][k] = clf_c.predict_proba(vec_c.transform(texts[k]))
        comp["layout"][k] = clf_l.predict_proba(sc.transform(layout_features(recs, sets[k])))
        comp["b1"][k] = b1_prob[k]
        if args.task == "D":
            for name in comp:
                p = comp[name][k]
                comp[name][k] = p[:, 1] if p.ndim == 2 else p

    # ---- 融合（在校准子集上拟合，组件对校准子集均为 OOF）----
    def fuse_feats(k):
        parts = [np.asarray(comp[n][k]).reshape(len(y[k]), -1) for n in ("word", "char", "layout", "b1")]
        return np.concatenate(parts, axis=1)

    clf_f = LogisticRegression(C=p0["C"], max_iter=3000)
    clf_f.fit(fuse_feats("calib"), y["calib"])
    fused = {k: clf_f.predict_proba(fuse_feats(k)) for k in sets}

    # ---- 指标（dev 为主；校准集参考）----
    def score(task, y_true, prob):
        if task == "F":
            return {"macro_f1": float(f1_score(y_true, prob.argmax(-1), labels=list(range(len(D.CLASSES_F))), average="macro", zero_division=0))}
        p = prob[:, 1] if getattr(prob, "ndim", 1) == 2 else prob
        return {"auroc": float(roc_auc_score(y_true, p))}

    summary = {"task": args.task, "seed": args.seed, "n": {k: len(v) for k, v in sets.items()},
               "components_dev": {n: score(args.task, y["dev"], comp[n]["dev"]) for n in comp},
               "fused_dev": score(args.task, y["dev"], fused["dev"]),
               "fused_calib": score(args.task, y["calib"], fused["calib"]),
               "git": TR.git_head(), "time_s": round(time.time() - T0, 1)}
    D.log(f"[p0 {args.task}/s{args.seed}] fused_dev={summary['fused_dev']} "
          f"components_dev={ {n: v for n, v in summary['components_dev'].items()} }")

    outdir = wp["root"] / "p0"
    outdir.mkdir(parents=True, exist_ok=True)
    np.savez(outdir / f"{args.task}_s{args.seed}.npz",
             **{f"{n}_{k}": comp[n][k] for n in comp for k in sets},
             **{f"fused_{k}": fused[k] for k in sets})
    with open(outdir / f"{args.task}_s{args.seed}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    if args.work_root is None:  # 正式运行：记录进 artifacts
        apth = TR.art_paths(cfg)["runs"] / f"{args.task}_p0_s{args.seed}.json"
        apth.parent.mkdir(parents=True, exist_ok=True)
        with open(apth, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=1)
        D.log(f"[p0] 记录 → {apth}")
    D.log(f"[done] p0 用时 {time.time() - T0:.0f}s")


if __name__ == "__main__":
    main()
