#!/usr/bin/env python
"""Round2 Q1：TF-IDF 优势来源诊断（h2_authorbench_dcan，任务级留出；test 不参与选择）。

对照（同一任务级划分、同一 test）：
  A char TF-IDF —— 复用 round1 (.7639) 预测做分组/长度桶复核（不重训）；
  B metadata-only —— 31 维结构统计 → 标准化 → LogisticRegression；
  C filtered TF-IDF —— 去注释/字符串、标识符→_id、保留 C 关键字与结构 → 同 round1 TF-IDF 管线（5ep）；
  D frozen CodeT5 768d → LR（z_raw / z_center / z_center_std；center 为任务内转导，非单样本部署）。
输出：artifacts/acl_dcan_round2/shortcuts/{metrics.json, predictions.npz}
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import dcan_four_models as r1  # noqa: E402

OUT = ROOT / "artifacts" / "acl_dcan_round2" / "shortcuts"
OUT.mkdir(parents=True, exist_ok=True)
R1 = ROOT / "artifacts" / "acl_dcan_round1"

C_KEYWORDS = set("""auto break case char const continue default do double else enum extern float for goto if inline int
long register restrict return short signed sizeof static struct switch typedef union unsigned void volatile while
_Alignas _Alignof _Atomic _Bool _Complex _Generic _Imaginary _Noreturn _Static_assert _Thread_local""".split())
import re  # noqa: E402

_re_block = re.compile(r"/\*.*?\*/", re.S)
_re_line = re.compile(r"//[^\n]*")
_re_str = re.compile(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'')
_re_ident = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def skeleton(code: str) -> str:
    s = _re_block.sub(" ", code)
    s = _re_line.sub(" ", s)
    s = _re_str.sub('""', s)
    s = _re_ident.sub(lambda m: m.group(0) if m.group(0) in C_KEYWORDS else "_id", s)
    return re.sub(r"\s+", " ", s)


def metrics(probs, y, model_name, char_count, families, lang="C"):
    import sklearn.metrics as sm
    pred = probs.argmax(1)
    fam = np.array(families)
    out = {"n": int(len(y)),
           "macro_f1": float(sm.f1_score(y, pred, average="macro", zero_division=0)),
           "balanced_acc": float(sm.balanced_accuracy_score(y, pred)),
           "ece_top1_15": float(r1.ece(probs.astype(np.float64), y)),
           "per_class_recall": {fam[i]: float((pred[y == i] == i).mean()) if (y == i).any() else None
                                for i in range(len(fam))},
           "per_generator_acc": {}}
    for g in sorted(set(model_name.tolist())):
        sel = model_name == g
        out["per_generator_acc"][g] = {"n": int(sel.sum()), "acc": float((pred[sel] == y[sel]).mean())}
    # 长度桶（train 四分位为界）
    qs = np.quantile(char_count, [0.25, 0.5, 0.75])
    binid = np.digitize(char_count, qs)
    out["per_length_quartile_acc"] = {f"q{i+1}": {"n": int((binid == i).sum()),
                                                  "acc": float((pred[binid == i] == y[binid == i]).mean())}
                                      for i in range(4)}
    out["language"] = {"langs": [lang], "note": "C-only（常量）"}
    return out


def main():
    t0 = time.time()
    rows = r1.load_rows()
    fam = [r["family"] for r in rows]
    split = [r["task_split"] for r in rows]
    task = [r["task_id"] for r in rows]
    model_name = np.array([r["model_name"] for r in rows])
    char_count = np.array([r["char_count"] for r in rows], dtype=float)
    tr = np.array([i for i, s in enumerate(split) if s == "train"])
    te = np.array([i for i, s in enumerate(split) if s == "test"])
    dv = np.array([i for i, s in enumerate(split) if s == "dev"])
    cls = {f: i for i, f in enumerate(r1.FAMILIES)}
    y = np.array([cls[f] for f in fam])

    st_raw = np.array([r1.struct_features(r) for r in rows], dtype=np.float64)
    emb = r1.encode_semantics(rows)  # 缓存命中
    res = {"config": {"script": "scripts/dcan_round2_shortcuts.py",
                      "split": "task-level train/dev/test (unchanged from round1)",
                      "classifier_LR": "multinomial lbfgs C=0.1 max_iter=2000 (train only)",
                      "tfidf": "char_wb(2,4) min_df=5 sublinear max_features=300k; SGD log_loss 5ep 类逆频率权重",
                      "note": "char TF-IDF 数字复用 round1（不重训）；z_center 为任务内转导表示"},
           "methods": {}}
    pack = {}

    # B metadata-only
    mu, sd = st_raw[tr].mean(0), st_raw[tr].std(0) + 1e-9
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(max_iter=2000, C=0.1).fit((st_raw[tr] - mu) / sd, y[tr])
    probs = lr.predict_proba((st_raw[te] - mu) / sd)
    res["methods"]["metadata_only"] = metrics(probs, y[te], model_name[te], char_count[te], r1.FAMILIES)
    pack["metadata_only_probs"] = probs.astype(np.float16)
    print("[q1] metadata_only:", round(res["methods"]["metadata_only"]["macro_f1"], 4), flush=True)

    # C filtered TF-IDF
    from p0_tfidf_baseline import fit_vectorizer, sample_stride, transform_chunks
    tr_texts = [skeleton(rows[i]["code"]) for i in tr]
    te_texts = [skeleton(rows[i]["code"]) for i in te]
    dv_texts = [skeleton(rows[i]["code"]) for i in dv]
    vec = fit_vectorizer(sample_stride(tr_texts, 100000), 300000)
    counts = Counter(y[tr]); K = len(r1.FAMILIES); N = len(tr)
    w = np.array([N / (K * max(1, counts[c])) for c in y[tr]], dtype=np.float64)
    from sklearn.linear_model import SGDClassifier
    clf = SGDClassifier(loss="log_loss", alpha=2e-6, max_iter=1, tol=None, learning_rate="optimal",
                        average=False, random_state=0)
    Xd = vec.transform(dv_texts)
    per_ep = []
    for ep in range(1, 6):
        rng = np.random.RandomState(100 + ep)
        order = rng.permutation(N)
        for i in range(0, N, 10000):
            idxs = order[i:i + 10000]
            clf.partial_fit(vec.transform([tr_texts[j] for j in idxs]), y[tr][idxs],
                            classes=np.arange(K), sample_weight=w[idxs])
        import sklearn.metrics as sm
        f = sm.f1_score(y[dv], clf.predict(Xd), average="macro", zero_division=0)
        per_ep.append({"epoch": ep, "dev_macro_f1": float(f)})
        print(f"[q1] filtered-tfidf epoch {ep}: dev {f:.4f}", flush=True)
    probs = np.concatenate([clf.predict_proba(X) for X in transform_chunks(vec, iter(te_texts))])
    res["methods"]["filtered_tfidf"] = metrics(probs, y[te], model_name[te], char_count[te], r1.FAMILIES)
    res["methods"]["filtered_tfidf"]["epochs"] = per_ep
    pack["filtered_tfidf_probs"] = probs.astype(np.float16)
    print("[q1] filtered_tfidf test:", round(res["methods"]["filtered_tfidf"]["macro_f1"], 4), flush=True)

    # D frozen CodeT5 reps (raw/center/center_std)
    z = emb.astype(np.float64)
    task_mean = {}
    for i, t in enumerate(task):
        task_mean.setdefault(t, []).append(i)
    tm = {t: z[ix].mean(0) for t, ix in task_mean.items()}
    zc = z - np.array([tm[t] for t in task])
    sdv = zc[tr].std(0) + 1e-9
    for name, X in [("codeT5_raw", z), ("codeT5_center", zc), ("codeT5_center_std", zc / sdv)]:
        lr = LogisticRegression(max_iter=2000, C=0.1).fit(X[tr], y[tr])
        probs = lr.predict_proba(X[te])
        res["methods"][name] = metrics(probs, y[te], model_name[te], char_count[te], r1.FAMILIES)
        pack[f"{name}_probs"] = probs.astype(np.float16)
        print(f"[q1] {name}:", round(res["methods"][name]["macro_f1"], 4), flush=True)
    res["methods"]["codeT5_center"]["note"] = "任务内去均值（使用 test 任务兄弟输出）—— 转导口径，非单样本部署"
    res["methods"]["codeT5_center_std"]["note"] = "任务内去均值 + 训练任务逐维标准差 —— 转导口径"

    # A char TF-IDF（round1 预测复核，含分组/长度桶）
    try:
        d = np.load(R1 / "p0" / "dcan" / "dcan_predictions.npz")
        probs0, y0 = d["probs"].astype(np.float64), d["y"].astype(int)
        if len(y0) == len(te) and np.all(y0 == y[te]):
            res["methods"]["char_tfidf_round1_ref"] = metrics(probs0, y0, model_name[te],
                                                             char_count[te], r1.FAMILIES)
            res["methods"]["char_tfidf_round1_ref"]["note"] = "round1 P0 预测复用（非重训）"
        else:
            res["methods"]["char_tfidf_round1_ref"] = {"error": "order mismatch"}
    except Exception as exc:  # noqa: BLE001
        res["methods"]["char_tfidf_round1_ref"] = {"error": repr(exc)}

    pack["y_test"] = y[te].astype(np.int16)
    pack["model_name_test"] = model_name[te].astype(object)
    pack["char_count_test"] = char_count[te]
    np.savez_compressed(OUT / "predictions.npz", **pack)
    res["env"] = {"python": sys.version.split()[0], "wall_sec": round(time.time() - t0, 1)}
    (OUT / "metrics.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(f"[q1] done {round(time.time()-t0,1)}s -> {OUT}")


if __name__ == "__main__":
    main()
