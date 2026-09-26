#!/usr/bin/env python
"""B 大训练集：human 20K + 机器每族 ≤4K（覆盖全部生成器）→ CodeT5 与 Qwen 双分词。

输出：
  data/processed/semeval_big/b_train.parquet（CodeT5 ids）+ b_val/b_test（复制）
  data/processed/semeval_qwen_big/b_train.parquet（Qwen ids）+ b_val/b_test（复制）
"""

from __future__ import annotations

import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/SemEval-2026-Task13/task_b/task_b_training_set.parquet"
BIG = ROOT / "data/processed/semeval_big"
BIGQ = ROOT / "data/processed/semeval_qwen_big"


def sample_rows() -> list[tuple[str, int, str, str]]:
    t = pq.read_table(RAW, columns=["code", "label", "generator", "language"])
    codes = t.column("code").to_pylist()
    labs = t.column("label").to_pylist()
    gens = t.column("generator").to_pylist()
    langs = t.column("language").to_pylist()
    by_lab_gen = defaultdict(lambda: defaultdict(list))
    for i, (l, g) in enumerate(zip(labs, gens)):
        by_lab_gen[int(l)][str(g)].append(i)
    rng = random.Random(123)
    picked: list[int] = []
    human_pool = [i for gl in by_lab_gen[0].values() for i in gl]
    picked += rng.sample(human_pool, min(20000, len(human_pool)))
    for lab in sorted(by_lab_gen):
        if lab == 0:
            continue
        gens_map = by_lab_gen[lab]
        tot = sum(len(v) for v in gens_map.values())
        quota = min(4000, tot)
        acc = 0
        items = sorted(gens_map.items())
        for j, (g, idxs) in enumerate(items):
            remain = max(quota - acc, 0)
            want = remain if j == len(items) - 1 else int(round(quota * len(idxs) / tot))
            take = min(len(idxs), want, remain)
            picked += rng.sample(idxs, take)
            acc += take
    rows = [(codes[i], int(labs[i]), str(gens[i]), str(langs[i])) for i in picked]
    rng.shuffle(rows)
    lab = Counter(r[1] for r in rows)
    gen = Counter(r[2] for r in rows)
    print(f"[big] n={len(rows)} labels={dict(sorted(lab.items()))}", flush=True)
    print(f"[big] 生成器覆盖 {len(gen)} 个", flush=True)
    return rows


def tokenize(rows, tok, task_dir: Path, name: str):
    ids_l, lab_l, gen_l, lang_l, ntok = [], [], [], [], []
    for code, lab, gen, lang in rows:
        ids = tok(code, add_special_tokens=False)["input_ids"]
        n = len(ids)
        if n < 8:
            continue
        if n > 1024:
            ids = ids[:768] + ids[-256:]
        ids_l.append(ids)
        lab_l.append(lab)
        gen_l.append(gen)
        lang_l.append(lang)
        ntok.append(n)
    task_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({
        "ids": pa.array(ids_l, type=pa.list_(pa.int32())),
        "label": pa.array(lab_l, type=pa.int16()),
        "generator": pa.array(gen_l, type=pa.string()),
        "language": pa.array(lang_l, type=pa.string()),
        "n_tokens": pa.array(ntok, type=pa.int32()),
    }), task_dir / f"{name}.parquet", compression="zstd")
    print(f"[big] {name}: n={len(ids_l)} -> {task_dir}", flush=True)


def main() -> int:
    rows = sample_rows()
    tok5 = AutoTokenizer.from_pretrained(str(ROOT / "checkpoints/codet5-base"))
    tokq = AutoTokenizer.from_pretrained(str(ROOT / "checkpoints/qwen2.5-coder-1.5b-instruct"))
    tokenize(rows, tok5, BIG, "b_train")
    tokenize(rows, tokq, BIGQ, "b_train")
    for src_dir, dst_dir in ((ROOT / "data/processed/semeval", BIG),
                             (ROOT / "data/processed/semeval_qwen", BIGQ)):
        for split in ("val", "test"):
            src, dst = src_dir / f"b_{split}.parquet", dst_dir / f"b_{split}.parquet"
            if not dst.exists():
                shutil.copyfile(src, dst)
        print(f"[big] {dst_dir} val/test 就位", flush=True)
    print("[prepare_big] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
