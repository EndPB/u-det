#!/usr/bin/env python
"""P0 强基线：字符 n-gram TF-IDF + 线性分类器（ACL 数据集合 v1，四源统一实现）。

对应指导：d-det/docx/d-det_AutoDL_服务器端AI下一阶段执行指导_2026-10-03.md 第三步。
- 全流式读取（parquet batch / JSONL 逐行），不用 pandas.read_*；
- 词表在训练集抽样上拟合（固定词表分块 transform，控制内存）；
- SGDClassifier(log_loss) partial_fit，按类逆频率样本权重；逐 epoch 在 dev/val 上选 pass；
- 关闭集/任务留出/文件留出/generator 留出四套协议见各 source；
- 指标：macro-F1、balanced acc、逐类召回、混淆矩阵、ECE(top-1, 15 bins)、分组（generator/family/language）准确率。

用法：
  python scripts/p0_tfidf_baseline.py --source dcan
  python scripts/p0_tfidf_baseline.py --source aicd_t2 --smoke
  python scripts/p0_tfidf_baseline.py --source stacad
  python scripts/p0_tfidf_baseline.py --source droid
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DEFAULT_OUT = ROOT / "artifacts" / "acl_dcan_round1" / "p0"

TEXT_CAP = 6000          # 每篇代码截断字符数（控制向量化开销）
CHUNK = 10000            # transform/SGD 分块大小
STACAD_MODELS = ["gemini_3_flash", "gpt5_nano", "claude_3_haiku", "qwen3_coder_30b",
                 "deepseek_v3_2", "grok_4_fast", "devstral_2512"]


def log(msg: str) -> None:
    print(f"[p0] {msg}", flush=True)


# ------------------------------------------------------------------ 数据源
def rows_aicd_t2():
    """yield (text, label, split, groups, extras)；split ∈ {train,validation,test}。"""
    import pyarrow.parquet as pq
    base = DATA / "acl_attribution_collection_v1" / "raw" / "aicd" / "T2"
    for shard in sorted(base.glob("*.parquet")):
        split = shard.name.split("-", 1)[0]
        pf = pq.ParquetFile(shard)
        for batch in pf.iter_batches(batch_size=4096, columns=["code", "label"], use_threads=False):
            codes = batch.column("code").to_pylist()
            labs = batch.column("label").to_pylist()
            for c, l in zip(codes, labs):
                yield (c or "")[:TEXT_CAP], int(l), split, {}, {}


def rows_dcan():
    p = DATA / "h2_authorbench_dcan" / "core.jsonl"
    with p.open() as fh:
        for line in fh:
            d = json.loads(line)
            yield ((d.get("code") or "")[:TEXT_CAP], d["family"], d["task_split"],
                   {"model": d["model_name"]}, {"task_id": d["task_id"]})


def rows_stacad(folds):
    """STACAD：5 折文件级（folds.npy 与 corpus 写入顺序逐 pair 对齐）。"""
    cdir = DATA / "stacad_v2" / "extracted" / "STACAD-v2" / "data" / "corpus_v2"
    langs = ["py", "java", "c", "cpp", "php", "go", "cs"]
    gidx = 0
    for lg in langs:
        with (cdir / f"{lg}.jsonl").open() as fh:
            for line in fh:
                d = json.loads(line)
                f = int(folds[gidx]); gidx += 1
                yield ((d.get("llm_src") or "")[:TEXT_CAP], int(d["label"]),
                       f"fold{f}", {"lang": lg, "model": STACAD_MODELS[int(d["label"]) - 1]}, {})


def rows_droid(fold_plan):
    p = DATA / "h2_droid_full_selected" / "core.jsonl"
    with p.open() as fh:
        for line in fh:
            d = json.loads(line)
            if d.get("Label") != "MACHINE_GENERATED":
                continue
            yield ((d.get("Code") or "")[:TEXT_CAP], d["Model_Family"], d["split_source"],
                   {"generator": d["Generator"], "language": d["Language"]}, {})


# ------------------------------------------------------------------ 工具
def load_rows(rows_iter):
    """物化来源（流式读取；文本驻留内存，不落盘）。返回 (texts, labels, splits, groups)。"""
    texts, labels, splits, groups_l = [], [], [], []
    for text, label, split, groups, _extras in rows_iter:
        texts.append(text)
        labels.append(label)
        splits.append(split)
        groups_l.append(groups)
    return texts, labels, splits, groups_l


def subset(T, L, S, G, pred):
    idx = [i for i, s in enumerate(S) if pred(s, G[i])]
    return ([T[i] for i in idx], [L[i] for i in idx], [G[i] for i in idx])


def sample_stride(texts, cap):
    if len(texts) <= cap:
        return texts
    stride = max(1, len(texts) // cap)
    return texts[::stride][:cap]


def cap_tuple(tup, cap, seed=7):
    n = len(tup[0])
    if n <= cap:
        return tup
    idx = sorted(np.random.RandomState(seed).choice(n, cap, replace=False))
    return tuple([x[i] for i in idx] for x in tup)


def transform_chunks(vec, texts_iter, chunk=CHUNK):
    buf = []
    for t in texts_iter:
        buf.append(t)
        if len(buf) >= chunk:
            yield vec.transform(buf)
            buf = []
    if buf:
        yield vec.transform(buf)


def eval_metrics(probs: np.ndarray, y: np.ndarray, classes: list, groups: dict | None = None):
    from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
    pred = probs.argmax(1)
    out = {
        "n": int(len(y)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "balanced_acc": float(balanced_accuracy_score(y, pred)),
        "per_class_recall": {classes[i]: float((pred[y == i] == i).mean()) if (y == i).any() else None
                             for i in range(len(classes))},
        "confusion_matrix": confusion_matrix(y, pred, labels=list(range(len(classes)))).tolist(),
        "ece_top1_15": float(ece_top1(probs, y)),
    }
    if groups:
        gb = {}
        for name, arr in groups.items():
            gb[name] = {}
            for g in sorted(set(arr.tolist())):
                m = np.array([a == g for a in arr])
                gb[name][g] = {"n": int(m.sum()), "acc": float((pred[m] == y[m]).mean())}
        out["groups"] = gb
    return out


def ece_top1(probs: np.ndarray, y: np.ndarray, bins: int = 15) -> float:
    conf = probs.max(1)
    pred = probs.argmax(1)
    acc = (pred == y).astype(np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    e = 0.0
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        m = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if not m.any():
            continue
        e += (m.sum() / len(y)) * abs(acc[m].mean() - conf[m].mean())
    return e


def fit_vectorizer(texts, max_vocab: int):
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=5,
                          max_features=max_vocab, sublinear_tf=True, lowercase=False,
                          dtype=np.float32)
    vec.fit(texts)
    return vec


# ------------------------------------------------------------------ 主流程
def run_experiment(args, classes, train, dev, test, tag, out_dir):
    """train/dev/test = (texts, labels, groups)；dev 可为 None。全内存、流式读取。"""
    from sklearn.linear_model import SGDClassifier
    from sklearn.metrics import f1_score
    if args.smoke:
        train = cap_tuple(train, 30000)
        dev = cap_tuple(dev, 10000) if dev is not None else None
        test = cap_tuple(test, 20000)
    t0 = time.time()
    tr_texts, tr_lab, _ = train
    cls_idx = {c: i for i, c in enumerate(classes)}
    K = len(classes)
    N = len(tr_texts)
    tr_y = np.array([cls_idx[v] for v in tr_lab])
    counts = Counter(tr_lab)
    w = np.array([N / (K * max(1, counts[v])) for v in tr_lab], dtype=np.float64)
    sample_texts = sample_stride(tr_texts, args.vocab_sample)
    log(f"[{tag}] train {N}; counts {dict(counts)}; vocab sample {len(sample_texts)}")
    vec = fit_vectorizer(sample_texts, args.max_vocab)
    log(f"[{tag}] vocab {len(vec.vocabulary_)}; {args.epochs} epochs over chunks...")

    dev_cache = None
    if dev is not None:
        dv_texts, dv_lab, _ = dev
        dev_cache = (vec.transform(dv_texts), np.array([cls_idx[v] for v in dv_lab]))

    clf = SGDClassifier(loss="log_loss", alpha=args.alpha, max_iter=1, tol=None,
                        learning_rate="optimal", average=False, random_state=0)
    per_epoch = []
    for ep in range(1, args.epochs + 1):
        t1 = time.time()
        rng = np.random.RandomState(100 + ep)
        order = rng.permutation(N)
        n = 0
        for i in range(0, N, CHUNK):
            idxs = order[i:i + CHUNK]
            X = vec.transform([tr_texts[j] for j in idxs])
            clf.partial_fit(X, tr_y[idxs], classes=np.arange(K), sample_weight=w[idxs])
            n += len(idxs)
        rec = {"epoch": ep, "n": n, "sec": round(time.time() - t1, 1)}
        if dev_cache is not None:
            pr = clf.predict_proba(dev_cache[0])
            rec["dev_macro_f1"] = float(f1_score(dev_cache[1], pr.argmax(1),
                                                 average="macro", zero_division=0))
        per_epoch.append(rec)
        log(f"[{tag}] epoch {ep}: {rec}")

    te_texts, te_lab, te_groups = test
    te_y = np.array([cls_idx[v] for v in te_lab])
    probs = np.concatenate([clf.predict_proba(X)
                            for X in transform_chunks(vec, iter(te_texts))])
    garr = None
    if te_groups and te_groups[0]:
        garr = {k: np.array([g[k] for g in te_groups]) for k in sorted(te_groups[0].keys())}
    m = eval_metrics(probs, te_y, classes, garr)
    m.update({"vocab": len(vec.vocabulary_), "train_n": N, "n_epochs": args.epochs,
              "wall_sec": round(time.time() - t0, 1)})
    np.savez_compressed(out_dir / f"{tag}_predictions.npz",
                        probs=probs.astype(np.float16), y=te_y.astype(np.int16))
    log(f"[{tag}] TEST macro-F1 {m['macro_f1']:.4f} balAcc {m['balanced_acc']:.4f} "
        f"ECE {m['ece_top1_15']:.4f} ({m['wall_sec']}s)")
    return m, per_epoch


def run_dcan(args, out_dir):
    classes = ["claude", "deepseek", "gemini", "llama", "openai", "qwen"]
    T, L, S, G = load_rows(rows_dcan())
    train = subset(T, L, S, G, lambda s, g: s == "train")
    dev = subset(T, L, S, G, lambda s, g: s == "dev")
    test = subset(T, L, S, G, lambda s, g: s == "test")
    m, pe = run_experiment(args, classes, train, dev, test, "dcan", out_dir)
    return {"dcan": {"test": m, "epochs": pe}}


def run_aicd(args, out_dir):
    classes = list(range(12))  # numeric_id 0..11（标签映射未核验前只报数字 ID）
    T, L, S, G = load_rows(rows_aicd_t2())
    train = subset(T, L, S, G, lambda s, g: s == "train")
    dev = subset(T, L, S, G, lambda s, g: s == "validation")
    test = subset(T, L, S, G, lambda s, g: s == "test")
    m, pe = run_experiment(args, classes, train, dev, test, "aicd_t2", out_dir)
    return {"aicd_t2": {"test": m, "epochs": pe,
                        "note": "numeric_id only; label map pending official verification"}}


def run_stacad(args, out_dir):
    folds = np.load(DATA / "stacad_v2" / "extracted" / "STACAD-v2" / "data" / "corpus_v2" / "folds.npy")
    classes = list(range(1, 8))  # label 1..7 = 7 个 LLM 模型
    T, L, S, G = load_rows(rows_stacad(folds))
    results = {}
    for k in range(args.folds):
        tag = f"stacad_fold{k}"
        train = subset(T, L, S, G, lambda s, g, k=k: s != f"fold{k}")
        test = subset(T, L, S, G, lambda s, g, k=k: s == f"fold{k}")
        m, pe = run_experiment(args, classes, train, None, test, tag, out_dir)
        results[f"fold{k}"] = {"test": m, "epochs": pe}
    f1s = [v["test"]["macro_f1"] for v in results.values()]
    results["mean_macro_f1"] = float(np.mean(f1s))
    results["std_macro_f1"] = float(np.std(f1s))
    return {"stacad_v2": results}


def run_droid(args, out_dir):
    fp = json.loads((DATA / "h2_droid_full_selected" / "fold_plan.json").read_text())
    classes = sorted({fam for spec in fp["folds"].values() for fam in spec})  # 7 个机器家族
    T, L, S, G = load_rows(rows_droid(None))
    results = {}
    for fk, spec in fp["folds"].items():
        tg = {g for v in spec.values() for g in v["train_generators"]}
        hg = {g for v in spec.values() for g in v["heldout_generators"]}
        train = subset(T, L, S, G, lambda s, g, tg=tg: s == "train" and g["generator"] in tg)
        dev = subset(T, L, S, G, lambda s, g, tg=tg: s == "dev" and g["generator"] in tg)
        test = subset(T, L, S, G, lambda s, g, hg=hg: s == "test" and g["generator"] in hg)
        m, pe = run_experiment(args, classes, train, dev, test, f"droid_{fk}", out_dir)
        results[fk] = {"test": m, "epochs": pe}
    return {"h2_droid_full_selected": results}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True,
                    choices=["aicd_t2", "dcan", "stacad", "droid"])
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=2e-6)
    ap.add_argument("--max-vocab", type=int, default=300000)
    ap.add_argument("--vocab-sample", type=int, default=100000)
    ap.add_argument("--folds", type=int, default=5, help="STACAD 折数（smoke 可设 1）")
    args = ap.parse_args()
    if args.smoke:
        args.epochs = min(args.epochs, 2)
        args.vocab_sample = min(args.vocab_sample, 20000)
        args.max_vocab = min(args.max_vocab, 100000)
        if args.source == "stacad":
            args.folds = 1
    out_dir = Path(args.out) / args.source
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if args.source == "dcan":
        res = run_dcan(args, out_dir)
    elif args.source == "aicd_t2":
        res = run_aicd(args, out_dir)
    elif args.source == "stacad":
        res = run_stacad(args, out_dir)
    else:
        res = run_droid(args, out_dir)
    res["meta"] = {"script": "scripts/p0_tfidf_baseline.py", "source": args.source,
                   "args": vars(args), "wall_sec": round(time.time() - t0, 1),
                   "vectorizer": "char_wb(2,4) min_df=5 sublinear_tf",
                   "text_cap_chars": TEXT_CAP}
    (out_dir / "metrics.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    log(f"done {args.source} in {res['meta']['wall_sec']}s -> {out_dir}")


if __name__ == "__main__":
    main()
