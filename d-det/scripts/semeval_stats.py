#!/usr/bin/env python
"""A/B/C 全 split 手工特征（结构/风格统计，取自 semeval.md 选手做法）——与处理后 parquet 行严格对齐。

对齐：复现 semeval_prepare.py 抽样 rng 序列（seed=0；任务序 a→b→c，每任务 train 抽→val 抽）
+ test 全收（SHA1 去重）+ tokenized<8 过滤；每 split 行数对 parquet 逐一断言。
输出：data/processed/semeval/{task}_{split}_stats.npz（X float32, names）。CPU；fail-soft。
"""

from __future__ import annotations

import hashlib
import math
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/SemEval-2026-Task13"
OUT = ROOT / "data/processed/semeval"
FILES = {
    "a": ("task_a/task_a_training_set_1.parquet", "task_a/task_a_validation_set.parquet",
          "task_a/task_a_test_set_sample.parquet"),
    "b": ("task_b/task_b_training_set.parquet", "task_b/task_b_validation_set.parquet",
          "task_b/task_b_test_set_sample.parquet"),
    "c": ("task_c/task_c_training_set_1.parquet", "task_c/task_c_validation_set.parquet",
          "task_c/task_c_test_set_sample.parquet"),
}
NAMES = ["r_void", "blank_run_max", "trail_blank", "log_lines", "log_chars",
         "line_mean", "line_std", "line_max", "tab_frac", "sp4_frac", "sp_frac",
         "mixed_indent", "trail_ws_frac", "crlf", "cr_only", "char_entropy",
         "comment_frac", "docstring", "ident_len"]


def labels_of(path: Path) -> np.ndarray:
    return np.asarray(pq.read_table(path, columns=["label"]).column("label").to_pylist())


def replay_picks() -> dict:
    rng = random.Random(0)
    picks = {}
    for task in ("a", "b", "c"):
        for split_i, split in ((0, "train"), (1, "val")):
            lab = labels_of(RAW / FILES[task][split_i])
            pools = {int(l): np.where(lab == l)[0].tolist() for l in sorted(set(lab.tolist()))}
            picked: list[int] = []
            if task == "a":
                qs = ((0, 16000), (1, 16000)) if split == "train" else ((0, 6000), (1, 6000))
                for l, q in qs:
                    picked += rng.sample(pools[l], min(q, len(pools[l])))
            elif task == "c":
                q = 10000 if split == "train" else 3000
                for l in sorted(pools):
                    picked += rng.sample(pools[l], min(q, len(pools[l])))
            else:
                qh, qm = (6000, 1200) if split == "train" else (3000, 500)
                picked += rng.sample(pools[0], min(qh, len(pools[0])))
                for l in sorted(pools):
                    if l == 0:
                        continue
                    picked += rng.sample(pools[l], min(qm, len(pools[l])))
            picks[(task, split)] = sorted(picked)
    return picks


def test_kept(path: Path) -> list[int]:
    seen, kept = set(), []
    pf = pq.ParquetFile(path)
    i = 0
    for batch in pf.iter_batches(batch_size=8192, columns=["code"]):
        for c in batch.column("code").to_pylist():
            h = hashlib.sha1(c.encode("utf-8", "ignore")).hexdigest()
            if h not in seen:
                seen.add(h)
                kept.append(i)
            i += 1
    return kept


def stats_one(code: str) -> list[float]:
    lines = code.split("\n")
    n = max(len(lines), 1)
    ne = sum(1 for ln in lines if not ln.strip())
    run = mx = 0
    for ln in lines:
        if not ln.strip():
            run += 1
            mx = max(mx, run)
        else:
            run = 0
    tb = 0
    for ln in reversed(lines):
        if not ln.strip():
            tb += 1
        else:
            break
    lens = np.array([len(ln) for ln in lines], dtype=float)
    tab = sum(1 for ln in lines if ln.startswith("\t"))
    sp4 = sum(1 for ln in lines if ln.startswith("    ") and not ln.startswith("\t"))
    sp = sum(1 for ln in lines if ln.startswith(" ") and not ln.startswith("    "))
    mixed = 1.0 if (tab > 0 and (sp4 + sp) > 0) else 0.0
    tws = sum(1 for ln in lines if ln.strip() and ln != ln.rstrip())
    crlf = 1.0 if "\r\n" in code else 0.0
    cr = 1.0 if ("\r" in code and "\r\n" not in code) else 0.0
    cnt = Counter(code)
    tot = max(sum(cnt.values()), 1)
    ent = -sum((v / tot) * math.log2(v / tot) for v in cnt.values())
    com = sum(1 for ln in lines if ln.lstrip().startswith(("#", "//", "/*", "*")))
    doc = 1.0 if ('"""' in code or "'''" in code) else 0.0
    idents = re.findall(r"[A-Za-z_][A-Za-z_0-9]*", code)
    ilen = float(np.mean([len(x) for x in idents])) if idents else 0.0
    return [ne / n, float(mx), float(tb), math.log1p(len(lines)), math.log1p(len(code)),
            lens.mean(), lens.std(), lens.max(), tab / n, sp4 / n, sp / n, mixed,
            tws / n, crlf, cr, ent, com / n, doc, ilen]


def main() -> int:
    tok = AutoTokenizer.from_pretrained(str(ROOT / "checkpoints/codet5-base"))
    picks = replay_picks()
    for task in ("a", "b", "c"):
        for split in ("train", "val", "test"):
            try:
                path = RAW / FILES[task][("train", "val", "test").index(split)]
                npq = pq.ParquetFile(
                    OUT / f"{task}_{split}.parquet").metadata.num_rows
                idx = test_kept(path) if split == "test" else picks[(task, split)]
                codes = pq.read_table(path, columns=["code"]).column("code").to_pylist()
                rows = []
                for i in idx:
                    c = codes[i]
                    if len(tok(c, add_special_tokens=False)["input_ids"]) < 8:
                        continue
                    rows.append(stats_one(c))
                X = np.asarray(rows, dtype="float32")
                assert X.shape[0] == npq, (task, split, X.shape[0], npq)
                np.savez_compressed(OUT / f"{task}_{split}_stats.npz", X=X,
                                    names=np.array(NAMES, dtype=object))
                print(f"[stats] {task}_{split}: n={X.shape[0]} d={X.shape[1]} 对齐✅", flush=True)
            except Exception as e:  # fail-soft
                print(f"[stats] {task}_{split}: FAILED {type(e).__name__}: {e}", flush=True)
    print("[stats] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
