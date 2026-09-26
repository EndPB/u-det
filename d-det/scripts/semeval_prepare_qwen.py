#!/usr/bin/env python
"""Qwen 骨干用的重分词数据（行选择与 CodeT5 版严格对齐）。

复现 semeval_prepare.py 的行抽取（rng 重放 + CodeT5<8 token 过滤，见 semeval_stats.py），
ids 改用 Qwen2.5-Coder tokenizer（头 768 + 尾 256 = 1024）。
输出：data/processed/semeval_qwen/{task}_{split}.parquet（schema 同 semeval 版）
"""

from __future__ import annotations

import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from semeval_stats import FILES, RAW, replay_picks, test_kept  # noqa: E402

OUT = ROOT / "data/processed/semeval_qwen"
COLS = ["code", "label", "generator", "language"]


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="checkpoints/qwen2.5-coder-1.5b-instruct",
                    help="目标分词器（默认 Qwen；DeepSeek 用 checkpoints/deepseek-coder-1.3b-instruct）")
    ap.add_argument("--out", default="data/processed/semeval_qwen")
    args = ap.parse_args()
    out_dir = ROOT / args.out
    tok5 = AutoTokenizer.from_pretrained(str(ROOT / "checkpoints/codet5-base"))
    tokq = AutoTokenizer.from_pretrained(str(ROOT / args.tokenizer))
    picks = replay_picks()
    print("[replay] 抽取索引完成", flush=True)
    for task in ("a", "b", "c"):
        for split in ("train", "val", "test"):
            path = RAW / FILES[task][("train", "val", "test").index(split)]
            target = pq.ParquetFile(
                ROOT / f"data/processed/semeval/{task}_{split}.parquet").metadata.num_rows
            idx = test_kept(path) if split == "test" else picks[(task, split)]
            t = pq.read_table(path, columns=COLS)
            codes = t.column("code").to_pylist()
            labs = t.column("label").to_pylist()
            gens = t.column("generator").to_pylist()
            langs = t.column("language").to_pylist()
            ids_l, lab_l, gen_l, lang_l, ntok = [], [], [], [], []
            for i in idx:
                c = codes[i]
                if len(tok5(c, add_special_tokens=False)["input_ids"]) < 8:
                    continue
                qids = tokq(c, add_special_tokens=False)["input_ids"]
                n = len(qids)
                if n > 1024:
                    qids = qids[:768] + qids[-256:]
                ids_l.append(qids)
                lab_l.append(int(labs[i]))
                gen_l.append(str(gens[i]))
                lang_l.append(str(langs[i]))
                ntok.append(int(n))
            assert len(ids_l) == target, (task, split, len(ids_l), target)
            out_dir.mkdir(parents=True, exist_ok=True)
            pq.write_table(pa.table({
                "ids": pa.array(ids_l, type=pa.list_(pa.int32())),
                "label": pa.array(lab_l, type=pa.int16()),
                "generator": pa.array(gen_l, type=pa.string()),
                "language": pa.array(lang_l, type=pa.string()),
                "n_tokens": pa.array(ntok, type=pa.int32()),
            }), out_dir / f"{task}_{split}.parquet", compression="zstd")
            print(f"[qwen-ids] {task}_{split}: n={len(ids_l)} 对齐✅", flush=True)
    print("[prepare_qwen] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
