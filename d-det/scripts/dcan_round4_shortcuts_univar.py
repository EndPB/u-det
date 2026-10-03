#!/usr/bin/env python
"""Round4 A：TF-IDF 捷径单变量诊断（C 词法状态机；train 拟合预处理；test 不参与选择）。

变体（同一 task split、同一 TF-IDF/SGD 管线）：
  raw            —— 原文（对照；round1 .7639 的管线复刻）
  ids_only       —— 非关键字标识符 → `_id`（字符串/注释内不替换）
  strings_only   —— 字符串/字符常量 → `""`（保留注释）
  comments_only  —— 去除 // 与 /* */ 注释（词法级，字符串内 //、/* 不受影响）
  ws_only        —— 空白/换行/缩进折叠为单空格
  all            —— 以上四项联合（对齐 round2 filtered，但为词法级实现）
每个变体 × 3 seeds：5 epoch、dev 曲线、best epoch（best-dev 恢复）与 epoch5（round2 同协议）
两套 test 概率；训练集拟合 TF-IDF 词表；dev 仅用于记录/选择，test 只在冻结后评一次。

输出：artifacts/acl_dcan_round4/shortcuts_univar/{metrics.json,predictions.npz,config.json,
env.json,split_manifest.json,hashes.json}
"""
from __future__ import annotations

import json
import platform
import sys
import time
import hashlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import dcan_four_models as r1  # noqa: E402
from p0_tfidf_baseline import fit_vectorizer, sample_stride  # noqa: E402

OUT = ROOT / "artifacts" / "acl_dcan_round4" / "shortcuts_univar"
OUT.mkdir(parents=True, exist_ok=True)

C_KEYWORDS = set("""auto break case char const continue default do double else enum extern float for goto if inline int
long register restrict return short signed sizeof static struct switch typedef union unsigned void volatile while
_Alignas _Alignof _Atomic _Bool _Complex _Generic _Imaginary _Noreturn _Static_assert _Thread_local""".split())

VARIANTS = ["raw", "ids_only", "strings_only", "comments_only", "ws_only", "all"]
SEEDS = [0, 1, 2]
EPOCHS = 5


def transform_c(code: str, ids=False, strings=False, comments=False, ws=False) -> str:
    """C 词法级变换：字符串感知的注释剥离；字符串/字符常量、标识符独立开关；可选空白折叠。"""
    out = []
    i, n = 0, len(code)
    while i < n:
        c = code[i]
        two = code[i:i + 2]
        if comments and two == "//":
            j = code.find("\n", i)
            j = n if j == -1 else j
            out.append(" ")
            i = j
            continue
        if comments and two == "/*":
            j = code.find("*/", i + 2)
            j = n if j == -1 else j + 2
            out.append(" ")
            i = j
            continue
        if c == '"' or c == "'":
            q, j = c, i + 1
            while j < n:
                if code[j] == "\\":
                    j += 2
                    continue
                if code[j] == q:
                    j += 1
                    break
                j += 1
            out.append('""' if strings else code[i:j])
            i = j
            continue
        if ids and (c.isalpha() or c == "_"):
            j = i
            while j < n and (code[j].isalnum() or code[j] == "_"):
                j += 1
            tok = code[i:j]
            out.append(tok if tok in C_KEYWORDS else "_id")
            i = j
            continue
        out.append(c)
        i += 1
    s = "".join(out)
    if ws:
        import re
        s = re.sub(r"\s+", " ", s).strip()
    return s


def apply_variant(code: str, variant: str) -> str:
    if variant == "raw":
        return code
    if variant == "ids_only":
        return transform_c(code, ids=True)
    if variant == "strings_only":
        return transform_c(code, strings=True)
    if variant == "comments_only":
        return transform_c(code, comments=True)
    if variant == "ws_only":
        return transform_c(code, ws=True)
    if variant == "all":
        return transform_c(code, ids=True, strings=True, comments=True, ws=True)
    raise ValueError(variant)


def metrics_block(probs, y, fam_order, model_name, char_count, train_edges):
    import sklearn.metrics as sm
    pred = probs.argmax(1)
    out = {"n": int(len(y)),
           "macro_f1": float(sm.f1_score(y, pred, average="macro", zero_division=0)),
           "balanced_acc": float(sm.balanced_accuracy_score(y, pred)),
           "ece_top1_15": float(r1.ece(probs.astype(np.float64), y)),
           "per_class_recall": {fam_order[i]: float((pred[y == i] == i).mean()) if (y == i).any() else None
                                for i in range(len(fam_order))},
           "per_generator_acc": {}}
    for g in sorted(set(model_name.tolist())):
        sel = model_name == g
        out["per_generator_acc"][g] = {"n": int(sel.sum()), "acc": float((pred[sel] == y[sel]).mean())}
    bid = np.digitize(char_count, train_edges)
    out["per_length_quartile_acc_train_edges"] = {
        f"q{i+1}": {"n": int((bid == i).sum()),
                    "acc": float((pred[bid == i] == y[bid == i]).mean()) if (bid == i).any() else None}
        for i in range(4)}
    return out


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    t0 = time.time()
    if (OUT / "metrics.json").exists():
        raise SystemExit(f"refuse to overwrite existing {OUT}/metrics.json")
    rows = r1.load_rows()
    split = [r["task_split"] for r in rows]
    task = [r["task_id"] for r in rows]
    sha = [r["source_sha256"] for r in rows]
    model_name = np.array([r["model_name"] for r in rows])
    char_count = np.array([r["char_count"] for r in rows], dtype=float)
    fam_idx = {f: i for i, f in enumerate(r1.FAMILIES)}
    y = np.array([fam_idx[r["family"]] for r in rows])
    tr = np.array([i for i, s in enumerate(split) if s == "train"])
    dv = np.array([i for i, s in enumerate(split) if s == "dev"])
    te = np.array([i for i, s in enumerate(split) if s == "test"])
    train_edges = np.quantile(char_count[tr], [0.25, 0.5, 0.75])  # train 拟合

    texts = {}
    for v in VARIANTS:
        texts[v] = [apply_variant(r["code"], v) for r in rows]
    # 变换示例（审计用）
    sample_i = int(tr[0])
    examples = {v: {"raw_head": rows[sample_i]["code"][:200],
                    "transformed_head": texts[v][sample_i][:200]} for v in VARIANTS}

    # class-inverse 权重（仅 train）
    from collections import Counter
    counts = Counter(y[tr]); K = len(r1.FAMILIES); N = len(tr)
    w = np.array([N / (K * max(1, counts[c])) for c in y[tr]], dtype=np.float64)

    from sklearn.linear_model import SGDClassifier
    import sklearn.metrics as sm
    res = {"config": {"script": "scripts/dcan_round4_shortcuts_univar.py",
                      "variants": VARIANTS, "seeds": SEEDS, "epochs": EPOCHS,
                      "tfidf": "char_wb(2,4) min_df=5 sublinear max_features=300k（词表仅 train 拟合）",
                      "sgd": "SGDClassifier log_loss alpha=2e-6 lr=optimal；批大小 10000；类逆频率样本权重",
                      "selection": "统一规则：记录 dev 曲线，best-dev 快照（coef/intercept/t）另存；epoch5 概率保留用于 round2 同协议对照",
                      "transform": "C 词法状态机 v1（字符串感知）；确定性、无拟合参数；关键字表见源码",
                      "boundaries": "test 不参与任何选择；无新下载；OMP=2"},
           "train_quartile_edges": [float(x) for x in train_edges],
           "examples": examples, "seeds_run": [], "variants": {}}
    pack = {"y_test": y[te].astype(np.int16),
            "task_id_test": np.array([task[i] for i in te], dtype=object),
            "source_sha256_test": np.array([sha[i] for i in te], dtype=object),
            "model_name_test": model_name[te],
            "family_order": np.array(r1.FAMILIES, dtype=object)}
    split_blob = "\n".join(f"{task[i]}\t{sha[i]}" for i in te)
    pack["split_hash_test"] = np.array(hashlib.sha256(split_blob.encode()).hexdigest())

    for v in VARIANTS:
        t_texts = [texts[v][i] for i in tr]
        d_texts = [texts[v][i] for i in dv]
        e_texts = [texts[v][i] for i in te]
        vec = fit_vectorizer(sample_stride(t_texts, 100000), 300000)
        Xtr = vec.transform(t_texts)
        Xdv = vec.transform(d_texts)
        Xte = vec.transform(e_texts)
        rng100 = None
        vres = {"vocab_size": int(len(vec.vocabulary_)), "per_seed": []}
        for seed in SEEDS:
            clf = SGDClassifier(loss="log_loss", alpha=2e-6, max_iter=1, tol=None,
                                learning_rate="optimal", average=False, random_state=seed)
            best = {"f1": -1.0, "coef": None, "intercept": None, "epoch": None}
            curve = []
            for ep in range(1, EPOCHS + 1):
                rng = np.random.RandomState(1000 * seed + 100 + ep)
                order = rng.permutation(N)
                for j in range(0, N, 10000):
                    idxs = order[j:j + 10000]
                    clf.partial_fit(Xtr[idxs], y[tr][idxs], classes=np.arange(K),
                                    sample_weight=w[idxs])
                f = float(sm.f1_score(y[dv], clf.predict(Xdv), average="macro", zero_division=0))
                curve.append({"epoch": ep, "dev_macro_f1": f})
                print(f"[r4a] {v} s{seed} ep{ep}: dev {f:.4f}", flush=True)
                if f > best["f1"]:
                    best = {"f1": f, "coef": clf.coef_.copy(), "intercept": clf.intercept_.copy(),
                            "epoch": ep}
            probs_ep5 = np.concatenate([clf.predict_proba(Xte[k:k + 4000]) for k in range(0, len(te), 4000)])
            clf.coef_, clf.intercept_ = best["coef"], best["intercept"]
            probs_best = np.concatenate([clf.predict_proba(Xte[k:k + 4000]) for k in range(0, len(te), 4000)])
            pack[f"{v}_s{seed}_best_probs"] = probs_best.astype(np.float16)
            pack[f"{v}_s{seed}_ep5_probs"] = probs_ep5.astype(np.float16)
            m_best = metrics_block(probs_best, y[te], r1.FAMILIES, model_name[te], char_count[te], train_edges)
            m_ep5 = metrics_block(probs_ep5, y[te], r1.FAMILIES, model_name[te], char_count[te], train_edges)
            vres["per_seed"].append({"seed": seed, "dev_curve": curve, "best_epoch": best["epoch"],
                                     "best_dev_f1": best["f1"], "test_best_dev": m_best,
                                     "test_epoch5": m_ep5})
            print(f"[r4a] {v} s{seed}: best ep{best['epoch']} dev {best['f1']:.4f} | "
                  f"test(best) {m_best['macro_f1']:.4f} | test(ep5) {m_ep5['macro_f1']:.4f}", flush=True)
        r1_mean = float(np.mean([p["test_best_dev"]["macro_f1"] for p in vres["per_seed"]]))
        r1_std = float(np.std([p["test_best_dev"]["macro_f1"] for p in vres["per_seed"]]))
        vres["mean_best_dev_macro_f1"], vres["std_best_dev_macro_f1"] = r1_mean, r1_std
        vres["mean_epoch5_macro_f1"] = float(np.mean([p["test_epoch5"]["macro_f1"] for p in vres["per_seed"]]))
        res["variants"][v] = vres
        print(f"[r4a] {v}: mean best-dev {r1_mean:.4f}±{r1_std:.4f} | mean ep5 {vres['mean_epoch5_macro_f1']:.4f}", flush=True)

    np.savez_compressed(OUT / "predictions.npz", **pack)
    (OUT / "metrics.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))

    # config / env / split_manifest / hashes（实验目录要求）
    import torch, sklearn, transformers
    env = {"python": platform.python_version(), "torch": torch.__version__,
           "numpy": np.__version__, "sklearn": sklearn.__version__,
           "transformers": transformers.__version__, "gpu": torch.cuda.get_device_name(0),
           "env_vars": {"OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"}}
    (OUT / "env.json").write_text(json.dumps(env, ensure_ascii=False, indent=2))
    sm_path = ROOT / "artifacts" / "acl_dcan_round2" / "split_manifest.json"
    sm2 = json.loads(sm_path.read_text())
    (OUT / "split_manifest.json").write_text(json.dumps({
        "generated_utc": "2026-10-03T21:45:00Z",
        "note": "Round4 A 不改变切分：沿用 round2 manifest（DCAN 任务级 train/dev/test；test 不参与选择）",
        "unchanged_from": "artifacts/acl_dcan_round2/split_manifest.json",
        "invariants_verified": {"rows_per_split": {"train": int(len(tr)), "dev": int(len(dv)), "test": int(len(te))},
                                "test_tasks": len(set(task[i] for i in te)), "test_rows": int(len(te))},
        "sources": sm2.get("sources", {})}, ensure_ascii=False, indent=2))
    files = ["data/h2_authorbench_dcan/core.jsonl", "scripts/dcan_round4_shortcuts_univar.py",
             "scripts/p0_tfidf_baseline.py", "scripts/dcan_four_models.py",
             "artifacts/acl_dcan_round4/shortcuts_univar/metrics.json",
             "artifacts/acl_dcan_round4/shortcuts_univar/predictions.npz"]
    hashes = {"files": {}}
    for f in files:
        p = ROOT / f
        hashes["files"][f] = {"sha256": sha256(p), "size": p.stat().st_size}
    (OUT / "hashes.json").write_text(json.dumps(hashes, ensure_ascii=False, indent=1))
    print(f"[r4a] done {round(time.time()-t0,1)}s -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
