#!/usr/bin/env python
"""ACL SOTA P0：三赛道强基线套件 v1（纯 CPU；authorbench 主赛道 + STACAD/Droid 外部赛道）。

赛道与切分（冻结，沿用既有 round 协议，不覆盖旧产物）：

- authorbench_dcan（主，task 划分）：data/h2_authorbench_dcan/core.jsonl；
  train/dev/test = task_split；cluster = task_id；6 families
- stacad_fold0（round4-C 协议）：subtrain 30k / file-level dev 3k / test = fold0 28992；
  cluster = file；7 classes；官方 105 特征矩阵与行空间逐行对齐（已校验）
- droid_fold0（generator 留出）：round4-C fold_plan；cluster = generator；7 families

基线族（P0 四条）：
1) 字符/词法 stylometry：char_wb(2,4) TF-IDF + word TF-IDF，均 SGD log_loss 5ep、
   类逆频率权重、best-dev 快照（round4-A 协议），3 seeds
2) CodeT5 监督基线：sem_lr（冻结 CodeT5 均值池化 + 线性 LR，统一协议，本套件训练）；
   另有复排行（round3 head_only / round4 lora 与 lora_ext，直接读取旧预测，只重算指标与 CI）
3) 结构/stylometry 统计：regex 提取器（命名/注释/格式/控制流/字面量统计）→ LR + LightGBM；
   STACAD 赛道用官方 105 特征（含 transformation/stylometric 视图）
4) 轻量集成：dev-only 拟合的 LR late-fusion stack（各基模型 dev 概率 → 测试一次）
   另报 seed-average 行（无拟合）

指标：macro-F1（主）+ balanced acc + per-class recall + ECE(15bin)；
macro-F1 报聚类 bootstrap 95% CI（A: task / B: file / C: generator，300 次重采样）。
test 不参与任何选择；dev 仅用于快照/融合权重。

输出：artifacts/acl_sota_p0/server_tracks/{track}/...；汇总见 acl_sota_p0_summary.json。
用法：
  OMP_NUM_THREADS=8 python scripts/acl_sota_p0_baselines.py --tracks authorbench_dcan stacad_fold0 droid_fold0
  python scripts/acl_sota_p0_baselines.py --smoke --tracks authorbench_dcan
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
OUT_BASE = ROOT / "artifacts" / "acl_sota_p0" / "server_tracks"
DATA = ROOT / "data"
SEEDS = [0, 1, 2]
TEXT_CAP = 6000


def log(msg: str) -> None:
    print(f"[p0] {msg}", flush=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------ 指标
def metrics(probs, y):
    import sklearn.metrics as sm
    pred = probs.argmax(1)
    conf = probs.max(1)
    correct = (pred == y).astype(float)
    bins = np.clip(np.digitize(conf, np.linspace(0, 1, 16)) - 1, 0, 14)
    ece = sum(np.mean(bins == b) * abs(correct[bins == b].mean() - conf[bins == b].mean())
              for b in range(15) if (bins == b).any())
    return {"n": int(len(y)),
            "macro_f1": float(sm.f1_score(y, pred, average="macro", zero_division=0)),
            "balanced_acc": float(sm.balanced_accuracy_score(y, pred)),
            "per_class_recall": [float(v) for v in sm.recall_score(y, pred, average=None, zero_division=0)],
            "ece": float(ece)}


def cluster_bootstrap_f1(probs, y, clusters, repeats=300, seed=20261004):
    import sklearn.metrics as sm
    cl = np.asarray([str(c) for c in clusters])
    uniq = np.unique(cl)
    idx_by_c = {c: np.where(cl == c)[0] for c in uniq}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(repeats):
        pick = rng.choice(len(uniq), size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_c[uniq[p]] for p in pick])
        vals.append(sm.f1_score(y[idx], probs[idx].argmax(1), average="macro", zero_division=0))
    v = np.array(vals)
    return {"ci95_low": float(np.percentile(v, 2.5)), "ci95_high": float(np.percentile(v, 97.5)),
            "boot_mean": float(v.mean())}


# ------------------------------------------------------------------ 结构/stylometry 特征（regex v1，语言无关）
_KW = ["if", "else", "elif", "for", "while", "switch", "case", "break", "continue",
       "return", "try", "catch", "throw", "class", "struct", "enum", "def", "import",
       "include", "package", "public", "private", "static", "const", "void", "new",
       "self", "this", "print", "printf", "cout", "input", "lambda", "async", "await",
       "yield", "match"]


def style_features(code: str) -> list[float]:
    import re
    lines = code.split("\n")
    n = max(1, len(lines))
    nchar = max(1, len(code))
    lens = [len(l) for l in lines]
    stripped = [l.strip() for l in lines]
    nonblank = [l for l in stripped if l]
    indent = [len(l) - len(l.lstrip(" ")) for l in nonblank]
    idents = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", code)
    idn = max(1, len(idents))
    names = set(idents)
    snake = sum(1 for w in idents if "_" in w and w.lower() == w)
    camel = sum(1 for w in idents if "_" not in w and any(c.isupper() for c in w[1:]))
    single = sum(1 for w in idents if len(w) == 1)
    upper_all = sum(1 for w in names if len(w) > 1 and w.isupper())
    return [
        nchar, n, np.mean(lens), np.max(lens), np.std(lens),
        sum(1 for l in lines if not l.strip()) / n,
        sum(1 for l in nonblank if l.startswith(("//", "#"))) / n,
        sum(1 for l in nonblank if l.startswith("/*") or l.startswith("*")) / n,
        code.count("/*") / n, code.count("*/") / n,
        np.mean(indent) if indent else 0.0, np.std(indent) if indent else 0.0,
        sum(1 for l in lines if "\t" in l) / n,
        sum(1 for l in lines if l != l.rstrip()) / n,
        sum(1 for l in lines if l.strip().endswith((";", "{"))) / n,
        len(idents) / n, np.mean([len(w) for w in idents]) if idents else 0.0,
        len(names) / n, single / idn, snake / idn, camel / idn, upper_all / max(1, len(names)),
        sum(1 for w in idents if w.startswith("_")) / idn,
        sum(1 for w in idents if any(ch.isdigit() for ch in w)) / idn,
        sum(c.isdigit() for c in code) / nchar, sum(c.isupper() for c in code) / nchar,
        code.count("_") / nchar, sum(1 for c in code if ord(c) > 127) / nchar,
        code.count(" ") / nchar, code.count("\t") / n,
        code.count("(") / n, code.count(")") / n, code.count("{") / n, code.count("}") / n,
        code.count("[") / n, code.count(";") / n, code.count(":") / n,
        code.count(",") / n, code.count("=") / n, code.count(".") / n,
        code.count("+") / n, code.count("-") / n, code.count("*") / n, code.count("/") / n,
        code.count("!") / n, code.count("<") / n, code.count(">") / n, code.count("&") / n,
        code.count("|") / n, code.count("%") / n,
        code.count('"') / n, code.count("'") / n, code.count("`") / n,
        code.count("\\n") / n, code.count("%s") / n, code.count("{}") / n, code.count("f\"") / n,
        sum(1 for l in nonblank if l.startswith("def ") or l.startswith("func ")) / n,
        sum(1 for l in nonblank if re.match(r"^\s*(if|for|while|switch)\b", l)) / n,
        sum(1 for l in nonblank if re.match(r"^\s*(return|yield)\b", l)) / n,
    ] + [len(re.findall(r"\b" + re.escape(k) + r"\b", code)) / n for k in _KW]


# ------------------------------------------------------------------ 通用训练组件
def tfidf_char():
    from sklearn.feature_extraction.text import TfidfVectorizer
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=5,
                           max_features=300000, sublinear_tf=True)


def tfidf_word():
    from sklearn.feature_extraction.text import TfidfVectorizer
    return TfidfVectorizer(token_pattern=r"[A-Za-z_][A-Za-z0-9_]*|\d+|\S",
                           ngram_range=(1, 2), min_df=3, max_features=200000,
                           sublinear_tf=True, lowercase=False)


def sgd_snapshot(Xtr, ytr, Xdv, ydv, Xte, yte, n_classes, seed, epochs=5, bs=10000, alpha=2e-6):
    """round4-A 协议：SGDClassifier log_loss 5ep、批 10000、类逆频率权重、best-dev 快照。"""
    import sklearn.metrics as sm
    from sklearn.linear_model import SGDClassifier
    counts = np.bincount(ytr, minlength=n_classes).astype(float)
    wclass = len(ytr) / (n_classes * np.maximum(counts, 1))
    clf = SGDClassifier(loss="log_loss", alpha=alpha, random_state=seed)
    classes = np.arange(n_classes)
    best = (-1.0, None, None)
    rng = np.random.default_rng(seed)
    for ep in range(epochs):
        perm = rng.permutation(len(ytr))
        for i in range(0, len(perm), bs):
            sel = perm[i:i + bs]
            clf.partial_fit(Xtr[sel], ytr[sel], classes=classes,
                            sample_weight=wclass[ytr[sel]])
        f1 = sm.f1_score(ydv, clf.predict(Xdv), average="macro", zero_division=0)
        if f1 > best[0]:
            best = (f1, clf.coef_.copy(), clf.intercept_.copy())
        log(f"    sgd seed{seed} ep{ep + 1}: dev F1 {f1:.4f}")
    coef, inter = best[1], best[2]
    clf.coef_, clf.intercept_ = coef, inter
    return clf.predict_proba(Xdv), clf.predict_proba(Xte), best[0]


def dense_lr(Ftr, ytr, Fdv, Fte, n_classes):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Ftr)
    lr = LogisticRegression(max_iter=3000, C=1.0).fit(sc.transform(Ftr), ytr)
    return lr.predict_proba(sc.transform(Fdv)), lr.predict_proba(sc.transform(Fte)), lr


def dense_lgb(Ftr, ytr, Fdv, Fte, n_classes, smoke=False):
    import lightgbm as lgb
    m = lgb.LGBMClassifier(objective="multiclass", num_class=n_classes,
                           n_estimators=50 if smoke else 800, learning_rate=0.05,
                           num_leaves=63, min_child_samples=20, subsample=0.8,
                           subsample_freq=1, colsample_bytree=0.6, reg_lambda=1.0,
                           class_weight="balanced", random_state=0, verbose=-1, n_jobs=8)
    m.fit(Ftr, ytr)
    return m.predict_proba(Fdv), m.predict_proba(Fte), m


def fusion_dev_lr(dev_probs: dict, te_probs: dict, y_dv, y_te, n_classes):
    """dev-only 拟合的 LR late fusion（log-prob 特征），test 一次。"""
    from sklearn.linear_model import LogisticRegression
    names = sorted(dev_probs)
    Zdv = np.hstack([np.log(np.clip(dev_probs[k], 1e-6, 1)) for k in names])
    Zte = np.hstack([np.log(np.clip(te_probs[k], 1e-6, 1)) for k in names])
    meta = LogisticRegression(max_iter=3000, C=1.0).fit(Zdv, y_dv)
    return meta.predict_proba(Zdv), meta.predict_proba(Zte), meta, names


# ------------------------------------------------------------------ 赛道数据
def build_authorbench(n_features_dense):
    import dcan_four_models as r1
    rows = r1.load_rows()
    mask = {s: np.array([r["task_split"] == s for r in rows]) for s in ("train", "dev", "test")}
    fam = r1.FAMILIES
    cls = {f: i for i, f in enumerate(fam)}
    y = np.array([cls[r["family"]] for r in rows])
    pack = {
        "name": "authorbench_dcan", "n_classes": 6, "classes": fam,
        "texts": [r["code"] for r in rows], "y": y, "mask": mask,
        "clusters_te": [r["task_id"] for r in rows if r["task_split"] == "test"],
        "provenance": {"rows": len(rows), "split_field": "task_split"},
    }
    if n_features_dense == "emb":
        z = np.load(ROOT / "runs" / "acl_dcan_round1" / "emb_dcan_ct5.npz", allow_pickle=True)
        emb = z["emb"]
        assert len(emb) == len(rows)
        pack["dense"] = emb
        pack["dense_name"] = "codet5_meanpool_768"
    pack["reuse"] = {
        "codet5_head_only": (ROOT / "artifacts" / "acl_dcan_round3_audit" / "head_only" / "predictions.npz",
                             "head_only_s{k}_probs", "round3 audit（round3_ft head_only 12ep）"),
        "codet5_lora_ext": (ROOT / "artifacts" / "acl_dcan_round4" / "lora_extend" / "predictions.npz",
                            "lora_ext_s{k}_probs", "round4 lora_extend（24ep 上限，best-dev）"),
    }
    return pack


def build_stacad():
    import dcan_round4_external as ext
    rows = ext.load_stacad()
    tr, dv, te, classes = ext.split_stacad(rows, 30000, 3000)
    z = np.load(ROOT / "data" / "stacad_v2" / "extracted" / "STACAD-v2" / "cache" /
                "corpus_v2" / "features_v2.npz", allow_pickle=True)
    X105 = z["X"].astype(np.float32)
    assert len(X105) == len(rows), "特征矩阵与行空间不一致"
    # 行空间逐行对齐（已校验 0/144958 mismatch）：按对象 id 建索引
    idx = {id(r): i for i, r in enumerate(rows)}
    m_tr = np.array([idx[id(r)] for r in tr])
    m_dv = np.array([idx[id(r)] for r in dv])
    m_te = np.array([idx[id(r)] for r in te])
    assert len(set(m_tr.tolist()) & set(m_dv.tolist())) == 0
    assert len(set(m_tr.tolist()) & set(m_te.tolist())) == 0
    assert len(set(m_dv.tolist()) & set(m_te.tolist())) == 0
    assert len(m_tr) == 30000 or len(m_tr) > 29000, f"subtrain size {len(m_tr)}"
    feats = ROOT / "artifacts" / "acl_dcan_round4" / "external" / "stacad_fold0" / "features.npz"
    fz = np.load(feats)
    F = np.zeros((len(rows), fz["Ftr"].shape[1]), dtype=np.float32)
    F[m_tr] = fz["Ftr"]
    F[m_dv] = fz["Fdv"]
    F[m_te] = fz["Fte"]
    assert np.allclose(F[m_tr], fz["Ftr"]) and np.allclose(F[m_dv], fz["Fdv"]) \
        and np.allclose(F[m_te], fz["Fte"]), "round4-C 特征与行空间不对齐"
    pack = {
        "name": "stacad_fold0", "n_classes": 7,
        "classes": ["gemini_3_flash", "gpt5_nano", "claude_3_haiku", "qwen3_coder_30b",
                    "deepseek_v3_2", "grok_4_fast", "devstral_2512"],
        "texts": [r["text"] for r in rows], "y": np.array([r["label"] for r in rows]),
        "mask": {"train": m_tr, "dev": m_dv, "test": m_te},
        "dense": X105, "dense_name": "stacad_official_105_features",
        "dense2": F, "dense2_name": "codet5_meanpool_768(fp16x2)",
        "clusters_te": [r["file"] for r in te],
        "provenance": {"rows": len(rows), "protocol": "round4-C: subtrain30k / file-dev3k / fold0 28992",
                       "features_v2_align": "0/144958 mismatch（key=(lang,file,fold,label)）"},
    }
    pack["reuse"] = {
        "tfidf_char_round4c": (ROOT / "artifacts" / "acl_dcan_round4" / "external" / "stacad_fold0" / "predictions.npz",
                               "tfidf_s{k}_probs", "round4-C TF-IDF（subtrain30k）"),
        "codet5_head_only": (ROOT / "artifacts" / "acl_dcan_round4" / "external" / "stacad_fold0" / "predictions.npz",
                             "head_only_s{k}_probs", "round4-C head-only"),
        "codet5_lora": (ROOT / "artifacts" / "acl_dcan_round4" / "external" / "stacad_fold0" / "predictions.npz",
                        "lora_s{k}_probs", "round4-C LoRA（2ep 上限）"),
    }
    return pack


def build_droid():
    import dcan_round4_external as ext
    fp = json.loads((DATA / "h2_droid_full_selected" / "fold_plan.json").read_text())
    tr, dv, te, classes = ext.load_droid(fp)
    rows = tr + dv + te
    off_tr, off_dv = 0, len(tr)
    off_te = len(tr) + len(dv)
    feats = ROOT / "artifacts" / "acl_dcan_round4" / "external" / "droid_fold0" / "features.npz"
    fz = np.load(feats)
    F = np.vstack([fz["Ftr"], fz["Fdv"], fz["Fte"]])
    assert F.shape[0] == len(rows), (F.shape, len(rows))
    cls = {c: i for i, c in enumerate(classes)}
    pack = {
        "name": "droid_fold0", "n_classes": len(classes), "classes": classes,
        "texts": [r["text"] for r in rows],
        "y": np.array([cls[r["family"]] for r in rows]),
        "mask": {"train": np.arange(off_tr, off_dv), "dev": np.arange(off_dv, off_te),
                 "test": np.arange(off_te, len(rows))},
        "dense2": F, "dense2_name": "codet5_meanpool_768(fp16x2)",
        "clusters_te": [te[i]["generator"] for i in range(len(te))],
        "provenance": {"rows": len(rows), "protocol": "round4-C: fold_0 generator-held-out"},
    }
    pack["reuse"] = {
        "tfidf_char_round4c": (ROOT / "artifacts" / "acl_dcan_round4" / "external" / "droid_fold0" / "predictions.npz",
                               "tfidf_s{k}_probs", "round4-C TF-IDF"),
        "codet5_head_only": (ROOT / "artifacts" / "acl_dcan_round4" / "external" / "droid_fold0" / "predictions.npz",
                             "head_only_s{k}_probs", "round4-C head-only"),
        "codet5_lora": (ROOT / "artifacts" / "acl_dcan_round4" / "external" / "droid_fold0" / "predictions.npz",
                        "lora_s{k}_probs", "round4-C LoRA（2ep 上限）"),
    }
    return pack


# ------------------------------------------------------------------ 主流程
def run_track(pack, out_dir: Path, smoke=False):
    t0 = time.time()
    n_classes = pack["n_classes"]
    texts = pack["texts"]
    y = pack["y"]
    m_tr, m_dv, m_te = (pack["mask"][s] for s in ("train", "dev", "test"))
    m_tr = np.where(m_tr)[0] if np.asarray(m_tr).dtype == bool else np.asarray(m_tr)
    m_dv = np.where(m_dv)[0] if np.asarray(m_dv).dtype == bool else np.asarray(m_dv)
    m_te = np.where(m_te)[0] if np.asarray(m_te).dtype == bool else np.asarray(m_te)
    Ttr = [texts[i] for i in m_tr]
    Tdv = [texts[i] for i in m_dv]
    Tte = [texts[i] for i in m_te]
    ytr, ydv, yte = y[m_tr], y[m_dv], y[m_te]
    if smoke:
        Ttr, ytr = Ttr[:4000], ytr[:4000]
        Tdv, ydv = Tdv[:1000], ydv[:1000]
        Tte, yte = Tte[:1000], yte[:1000]
    cl_te = pack["clusters_te"][:len(yte)] if smoke else pack["clusters_te"]
    log(f"{pack['name']}: train {len(Ttr)} dev {len(Tdv)} test {len(Tte)} classes {n_classes}")

    dv_probs, te_probs, rows = {}, {}, {}

    def add(name, dvp, tep, prov, seeds_f1=None):
        dv_probs[name] = dvp
        te_probs[name] = tep
        m = metrics(tep, yte)
        m.update({"provenance": prov, "dev_macro_f1": float(
            __import__("sklearn.metrics", fromlist=["f1_score"]).f1_score(ydv, dvp.argmax(1), average="macro", zero_division=0))})
        rows[name] = m

    # 1) 字符/词法 stylometry
    for tag, vec in (("tfidf_char", tfidf_char()), ("tfidf_word", tfidf_word())):
        Xtr = vec.fit_transform(Ttr)
        Xdv = vec.transform(Tdv)
        Xte = vec.transform(Tte)
        log(f"  {tag}: vocab {Xtr.shape[1]}")
        dvs, tes, f1s = [], [], []
        for s in SEEDS:
            dvp, tep, bf1 = sgd_snapshot(Xtr, ytr, Xdv, ydv, Xte, yte, n_classes, s)
            dvs.append(dvp)
            tes.append(tep)
            f1s.append(bf1)
        np.savez_compressed(out_dir / f"{tag}_probs.npz",
                            dv=np.mean(dvs, 0).astype(np.float16), te=np.mean(tes, 0).astype(np.float16),
                            **{f"te_s{s}": tes[s].astype(np.float16) for s in range(len(tes))},
                            y_dv=ydv, y_te=yte)
        add(tag, np.mean(dvs, 0), np.mean(tes, 0), f"SGD 5ep best-dev，3 seeds（dev F1 {['%.4f' % f for f in f1s]}）")

    # 2) sem_lr（冻结编码器 + 线性头，统一协议；优先 768d 编码器视图）
    if "dense2" in pack or "dense" in pack:
        key = "dense2" if "dense2" in pack else "dense"
        F = pack[key]
        Ftr, Fdv, Fte = F[m_tr], F[m_dv], F[m_te]
        if smoke:
            Ftr, Fdv, Fte = Ftr[:4000], Fdv[:1000], Fte[:1000]
        dvp, tep, _ = dense_lr(Ftr.astype(np.float32), ytr, Fdv.astype(np.float32), Fte.astype(np.float32), n_classes)
        np.savez_compressed(out_dir / "sem_lr_probs.npz", dv=dvp.astype(np.float16),
                            te=tep.astype(np.float16), y_dv=ydv, y_te=yte)
        add("sem_lr", dvp, tep, f"冻结 {pack[key + '_name']} + StandardScaler+LR(C=1)（统一协议）")

    # 3) 结构/stylometry 统计
    if pack["name"] == "stacad_fold0":
        F = pack["dense"]
        Ftr, Fdv, Fte = F[m_tr], F[m_dv], F[m_te]
        if smoke:
            Ftr, Fdv, Fte = Ftr[:4000], Fdv[:1000], Fte[:1000]
        dvp, tep, _ = dense_lr(Ftr, ytr, Fdv, Fte, n_classes)
        np.savez_compressed(out_dir / "feats105_lr_probs.npz", dv=dvp.astype(np.float16),
                            te=tep.astype(np.float16), y_dv=ydv, y_te=yte)
        add("feats105_lr", dvp, tep, "官方 105 特征 + 标准化 + LR（round4-C 子训练协议内）")
        dvp, tep, _ = dense_lgb(Ftr, ytr, Fdv, Fte, n_classes, smoke)
        np.savez_compressed(out_dir / "feats105_lgb_probs.npz", dv=dvp.astype(np.float16),
                            te=tep.astype(np.float16), y_dv=ydv, y_te=yte)
        add("feats105_lgb", dvp, tep, "官方 105 特征 + LightGBM（800 树；官方协议映射）")
    else:
        t1 = time.time()
        Str = np.array([style_features(t) for t in Ttr], dtype=np.float32)
        Sdv = np.array([style_features(t) for t in Tdv], dtype=np.float32)
        Ste = np.array([style_features(t) for t in Tte], dtype=np.float32)
        log(f"  style_features dim {Str.shape[1]} ({time.time() - t1:.0f}s)")
        dvp, tep, _ = dense_lr(Str, ytr, Sdv, Ste, n_classes)
        np.savez_compressed(out_dir / "style_lr_probs.npz", dv=dvp.astype(np.float16),
                            te=tep.astype(np.float16), y_dv=ydv, y_te=yte)
        add("style_lr", dvp, tep, f"regex stylometry {Str.shape[1]}d + 标准化 + LR")
        dvp, tep, _ = dense_lgb(Str, ytr, Sdv, Ste, n_classes, smoke)
        np.savez_compressed(out_dir / "style_lgb_probs.npz", dv=dvp.astype(np.float16),
                            te=tep.astype(np.float16), y_dv=ydv, y_te=yte)
        add("style_lgb", dvp, tep, f"regex stylometry {Str.shape[1]}d + LightGBM(800)")

    # 4) 轻量集成（dev-only LR stack）+ 等权均值对照
    fus_dv, fus_te, meta, names = fusion_dev_lr(dv_probs, te_probs, ydv, yte, n_classes)
    add("fusion_lr", fus_dv, fus_te, "dev-only LR stack（dev=拟合集"
        "，dev F1 非 out-of-sample）: " + ",".join(names))
    mean_te = np.mean([te_probs[k] for k in names], 0)
    mean_dv = np.mean([dv_probs[k] for k in names], 0)
    add("mean_ensemble", mean_dv, mean_te, "等权概率均值: " + ",".join(names))

    # 5) 复排行（旧预测只重算指标 + CI）
    for name, (path, keypat, prov) in pack["reuse"].items():
        if not path.exists():
            log(f"  reuse {name}: missing {path}")
            continue
        z = np.load(path, allow_pickle=True)
        seeds = []
        for s in range(3):
            k = keypat.format(k=s)
            if k in z.files:
                seeds.append(np.asarray(z[k], dtype=np.float64))
        if not seeds:
            continue
        y_test_ref = np.asarray(z["y_test"])
        if len(y_test_ref) != len(yte):
            log(f"  reuse {name}: y_test size mismatch ({len(y_test_ref)} vs {len(yte)}) — 跳过")
            continue
        assert np.array_equal(y_test_ref, yte), f"{name}: y_test 顺序不一致"
        avg = np.mean(seeds, 0)
        m = metrics(avg, yte)
        import sklearn.metrics as sm
        f1s = [float(sm.f1_score(y_test_ref, s.argmax(1), average="macro", zero_division=0)) for s in seeds]
        m.update({"provenance": prov + f"（{len(seeds)} seeds；只重算指标）",
                  "seed_macro_f1": f1s})
        rows[name] = m

    # 6) CI（聚类 bootstrap）对关键行
    for name in list(rows):
        probe = te_probs.get(name)
        if probe is None:
            # 复排行
            path, keypat, _ = pack["reuse"].get(name, (None, None, None))
            if path is None or not path.exists():
                continue
            z = np.load(path, allow_pickle=True)
            seeds = [np.asarray(z[keypat.format(k=s)], dtype=np.float64) for s in range(3)
                     if keypat.format(k=s) in z.files]
            if not seeds:
                continue
            probe = np.mean(seeds, 0)
        ci = cluster_bootstrap_f1(probe, yte, cl_te)
        rows[name].update(ci)

    json.dump({"track": pack["name"], "n_classes": n_classes, "classes": pack["classes"],
               "n_train": int(len(Ttr)), "n_dev": int(len(Tdv)), "n_test": int(len(Tte)),
               "smoke": smoke, "seeds": SEEDS, "provenance": pack["provenance"],
               "rows": rows},
              open(out_dir / "metrics.json", "w"), indent=2, ensure_ascii=False)
    # 保存全部 dev/test 概率（解码用）
    save = {"y_dv": ydv, "y_te": yte}
    for k in dv_probs:
        save[f"dv__{k}"] = dv_probs[k].astype(np.float16)
        save[f"te__{k}"] = te_probs[k].astype(np.float16)
    np.savez_compressed(out_dir / "predictions_all.npz", **save)
    log(f"{pack['name']} done in {(time.time() - t0) / 60:.1f} min")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", nargs="+", default=["authorbench_dcan", "stacad_fold0", "droid_fold0"])
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    OUT_BASE.mkdir(parents=True, exist_ok=True)
    env = {"python": sys.version.split()[0], "numpy": np.__version__, "platform": platform.platform(),
           "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
           "note": "纯 CPU 阶段（tfidf/LR/LGBM/融合）；OMP 放宽不适用于神经实验"}
    try:
        import sklearn, lightgbm
        env["sklearn"] = sklearn.__version__
        env["lightgbm"] = lightgbm.__version__
    except Exception as e:
        env["import_error"] = repr(e)
    for track in args.tracks:
        out_dir = OUT_BASE / (track + ("_smoke" if args.smoke else ""))
        out_dir.mkdir(parents=True, exist_ok=True)
        json.dump({"script": "scripts/acl_sota_p0_baselines.py", "track": track,
                   "smoke": args.smoke, "seeds": SEEDS, "text_cap": TEXT_CAP,
                   "sgd": "log_loss alpha=2e-6 bs=10000 5ep class-inverse weights best-dev",
                   "fusion": "LR C=1 on dev log-probs", "ci": "cluster bootstrap 300"},
                  open(out_dir / "config.json", "w"), indent=2, ensure_ascii=False)
        json.dump(env, open(out_dir / "env.json", "w"), indent=2)
        if track == "authorbench_dcan":
            pack = build_authorbench("emb")
        elif track == "stacad_fold0":
            pack = build_stacad()
        elif track == "droid_fold0":
            pack = build_droid()
        else:
            raise SystemExit(f"unknown track {track}")
        run_track(pack, out_dir, args.smoke)
    log("ALL TRACKS DONE")


if __name__ == "__main__":
    main()
