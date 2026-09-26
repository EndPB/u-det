#!/usr/bin/env python
"""TF-IDF char n-gram 基模型（A/B/C）：家族词法指纹的非神经基线。

- 文本 = 前 3000 字符 + 后 1000 字符；char_wb 2-4 gram，LR(balanced)；
- train 5 折 OOF probs（供 train-fit 栈）+ 全量 train 上的 val/test probs；
输出：runs/semeval_r/ngram_{task}_{split}.npz
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from semeval_stats import FILES, RAW, replay_picks, test_kept  # noqa: E402

OUT = ROOT / "runs/semeval_r"


def texts_for(task: str, split: str, tok) -> tuple[list[str], np.ndarray]:
    path = RAW / FILES[task][("train", "val", "test").index(split)]
    t = pq.read_table(path, columns=["code", "label"])
    codes = t.column("code").to_pylist()
    labs = np.asarray(t.column("label").to_pylist())
    idx = test_kept(path) if split == "test" else replay_picks()[(task, split)]
    X, y = [], []
    for i in idx:
        c = codes[i]
        if len(tok(c, add_special_tokens=False)["input_ids"]) < 8:
            continue
        X.append(c[:3000] + " <tail> " + c[-1000:])
        y.append(int(labs[i]))
    return X, np.asarray(y)


def main() -> int:
    tok = AutoTokenizer.from_pretrained(str(ROOT / "checkpoints/codet5-base"))
    picks = replay_picks()
    for task in ("a", "b", "c"):
        try:
            Xtr, ytr = texts_for(task, "train", tok)
            Xva, yva = texts_for(task, "val", tok)
            Xte, yte = texts_for(task, "test", tok)
            vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4),
                                  max_features=300000, min_df=2, sublinear_tf=True)
            Ttr = vec.fit_transform(Xtr)
            Tva = vec.transform(Xva)
            Tte = vec.transform(Xte)
            clf = LogisticRegression(max_iter=1000, C=4.0, class_weight="balanced")
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
            oof = np.zeros((len(ytr), int(ytr.max()) + 1), dtype="float32")
            for tr_i, va_i in skf.split(Ttr, ytr):
                c = LogisticRegression(max_iter=1000, C=4.0, class_weight="balanced")
                c.fit(Ttr[tr_i], ytr[tr_i])
                oof[va_i] = c.predict_proba(Ttr[va_i]).astype("float32")
            clf.fit(Ttr, ytr)
            OUT.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(OUT / f"ngram_{task}_train.npz", probs=oof, y=ytr)
            np.savez_compressed(OUT / f"ngram_{task}_val.npz",
                                probs=clf.predict_proba(Tva).astype("float32"), y=yva)
            np.savez_compressed(OUT / f"ngram_{task}_test.npz",
                                probs=clf.predict_proba(Tte).astype("float32"), y=yte)
            from sklearn.metrics import f1_score
            f1v = f1_score(yva, clf.predict(Tva), average="macro")
            f1t = f1_score(yte, clf.predict(Tte), average="macro")
            print(f"[ngram] {task}: val={f1v:.4f} test={f1t:.4f}", flush=True)
        except Exception as e:
            print(f"[ngram] {task}: FAILED {type(e).__name__}: {e}", flush=True)
    print("[ngram] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
