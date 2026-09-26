#!/usr/bin/env python
"""SemEval-2026 Task 13 数据准备：清洗 → 分层子集采样 → CodeT5 分词（头尾截断）→ parquet。

输出（默认 data/processed/semeval/）：
    {task}_train.parquet / {task}_val.parquet / {task}_test.parquet
    字段：ids(list<int>) label generator language n_tokens(原始长度)

策略（参考 docx/semeval.md 的“不平衡/泛化”结论）：
- 清洗：按 code SHA1 去重（各 split 内部）、丢弃 <8 token 的空/微片段；
- A（二分类，轻不平衡）：平衡采样（human/AI 各 ~16K）；
- C（4 类，中等不平衡）：四类各 ~10K；
- B（11 类，极端不平衡 225:1）：human 6K + 每家族 ≤1.2K（大幅缓和后配合 sqrt 类权重）；
- val 评估子集分层采样（保留全体不现实；报告时注明口径）；test 全收（官方镜像样本 1K）；
- 截断：>max_tokens 时取 头 768 + 尾 256（长代码的头尾预算/换行边界思路来自竞赛队伍经验）。

用法：
    python scripts/semeval_prepare.py --stats-only            # 只打印长度/分布统计
    python scripts/semeval_prepare.py                         # 全量产出
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RAW = ROOT / "data/raw/SemEval-2026-Task13"
OUT = ROOT / "data/processed/semeval"
COLS = ["code", "label", "generator", "language"]

FILES = {
    "a": ("task_a/task_a_training_set_1.parquet", "task_a/task_a_validation_set.parquet",
          "task_a/task_a_test_set_sample.parquet"),
    "b": ("task_b/task_b_training_set.parquet", "task_b/task_b_validation_set.parquet",
          "task_b/task_b_test_set_sample.parquet"),
    "c": ("task_c/task_c_training_set_1.parquet", "task_c/task_c_validation_set.parquet",
          "task_c/task_c_test_set_sample.parquet"),
}


def iter_rows(path: Path):
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=4096, columns=COLS):
        codes = batch.column("code").to_pylist()
        labels = batch.column("label").to_pylist()
        gens = batch.column("generator").to_pylist()
        langs = batch.column("language").to_pylist()
        yield from zip(codes, labels, gens, langs)


def scan(path: Path):
    """一趟扫描：去重 + 记录 (行号, label, generator, language)。"""
    seen = set()
    kept = []
    dup = short = 0
    for i, (code, label, gen, lang) in enumerate(iter_rows(path)):
        h = hashlib.sha1(code.encode("utf-8", "ignore")).hexdigest()
        if h in seen:
            dup += 1
            continue
        seen.add(h)
        kept.append((i, int(label), str(gen), str(lang)))
    return kept, {"dup": dup, "kept": len(kept)}


def sample_train(task: str, kept: list, rng: random.Random) -> list:
    by_label = defaultdict(list)
    for idx, lab, gen, lang in kept:
        by_label[lab].append(idx)
    picked = []
    if task == "a":                                     # 平衡：human/AI 各 16K
        for lab, quota in ((0, 16000), (1, 16000)):
            pool = by_label[lab]
            picked += rng.sample(pool, min(quota, len(pool)))
    elif task == "c":                                   # 四类各 10K
        for lab in sorted(by_label):
            pool = by_label[lab]
            picked += rng.sample(pool, min(10000, len(pool)))
    else:                                               # b：human 6K + 每家族 ≤1.2K
        pool = by_label[0]
        picked += rng.sample(pool, min(6000, len(pool)))
        for lab in sorted(by_label):
            if lab == 0:
                continue
            pool = by_label[lab]
            picked += rng.sample(pool, min(1200, len(pool)))
    return sorted(picked)


def sample_val(task: str, kept: list, rng: random.Random) -> list:
    by_label = defaultdict(list)
    for idx, lab, gen, lang in kept:
        by_label[lab].append(idx)
    picked = []
    if task == "a":
        for lab, quota in ((0, 6000), (1, 6000)):
            pool = by_label[lab]
            picked += rng.sample(pool, min(quota, len(pool)))
    elif task == "c":
        for lab in sorted(by_label):
            pool = by_label[lab]
            picked += rng.sample(pool, min(3000, len(pool)))
    else:
        pool = by_label[0]
        picked += rng.sample(pool, min(3000, len(pool)))
        for lab in sorted(by_label):
            if lab == 0:
                continue
            pool = by_label[lab]
            picked += rng.sample(pool, min(500, len(pool)))
    return sorted(picked)


def encode_rows(path: Path, keep_idx: set, tok, max_tokens: int, head: int, tail: int):
    rows = {"ids": [], "label": [], "generator": [], "language": [], "n_tokens": []}
    for i, (code, label, gen, lang) in enumerate(iter_rows(path)):
        if i not in keep_idx:
            continue
        ids = tok(code, add_special_tokens=False)["input_ids"]   # ★ 对齐训练侧口径
        n = len(ids)
        if n < 8:
            continue
        if n > max_tokens:
            ids = ids[:head] + ids[-tail:]
        rows["ids"].append(ids)
        rows["label"].append(int(label))
        rows["generator"].append(str(gen))
        rows["language"].append(str(lang))
        rows["n_tokens"].append(int(n))
    return rows


def save(rows: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table({
        "ids": pa.array(rows["ids"], type=pa.list_(pa.int32())),
        "label": pa.array(rows["label"], type=pa.int16()),
        "generator": pa.array(rows["generator"], type=pa.string()),
        "language": pa.array(rows["language"], type=pa.string()),
        "n_tokens": pa.array(rows["n_tokens"], type=pa.int32()),
    })
    pq.write_table(table, str(path), compression="zstd")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(RAW))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--tok", default="checkpoints/codet5-base")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--head", type=int, default=768)
    ap.add_argument("--tail", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stats-only", action="store_true")
    args = ap.parse_args()
    raw, out = Path(args.raw), Path(args.out)
    tok = AutoTokenizer.from_pretrained(str(ROOT / args.tok))

    if args.stats_only:
        rng = random.Random(args.seed)
        lens = {}
        for task in ("a", "b", "c"):
            path = raw / FILES[task][0]
            codes = []
            gens = Counter()
            for i, (code, label, gen, lang) in enumerate(iter_rows(path)):
                if i % 97 == 0:                        # 抽样（前 12 万行内约 1237 条）
                    codes.append(code)
                    gens[str(gen)] += 1
                if i >= 120000:
                    break
            ls = sorted(len(tok(c, add_special_tokens=False)["input_ids"]) for c in codes)
            q = lambda p: ls[min(len(ls) - 1, int(p * len(ls)))]
            lens[task] = {"n_sampled": len(ls), "p50": q(0.5), "p90": q(0.9),
                          "p99": q(0.99), "max": ls[-1], "generators_top": gens.most_common(5)}
            print(f"[stats] {task}: {lens[task]}", flush=True)
        print(json.dumps(lens, ensure_ascii=False), flush=True)
        return 0

    rng = random.Random(args.seed)
    meta = {}
    for task in ("a", "b", "c"):
        tr_p, va_p, te_p = (raw / f for f in FILES[task])
        print(f"\n===== task {task} =====")
        tr_kept, tr_stat = scan(tr_p)
        va_kept, va_stat = scan(va_p)
        print(f"[scan] train kept={tr_stat['kept']} dup={tr_stat['dup']} | "
              f"val kept={va_stat['kept']} dup={va_stat['dup']}")

        tr_idx = set(sample_train(task, tr_kept, rng))
        va_idx = set(sample_val(task, va_kept, rng))
        te_idx = set(i for i, *_ in scan(te_p)[0])          # test 全收

        tr_rows = encode_rows(tr_p, tr_idx, tok, args.max_tokens, args.head, args.tail)
        va_rows = encode_rows(va_p, va_idx, tok, args.max_tokens, args.head, args.tail)
        te_rows = encode_rows(te_p, te_idx, tok, args.max_tokens, args.head, args.tail)
        for rows, name in ((tr_rows, "train"), (va_rows, "val"), (te_rows, "test")):
            save(rows, out / f"{task}_{name}.parquet")
            lab = Counter(rows["label"])
            print(f"[save] {task}_{name}: n={len(rows['ids'])} labels={dict(sorted(lab.items()))} "
                  f"langs={dict(Counter(rows['language']).most_common(8))}")
            meta[f"{task}_{name}"] = {"n": len(rows["ids"]), "labels": dict(lab)}
    with open(out / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print("\n[done] ->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
