#!/usr/bin/env python
"""准备 CoDET-M4 数据：概览统计 -> 划分 train/val/test -> 保存到 data/processed。

用法::

    python scripts/prepare_data.py                       # 全量划分并保存
    python scripts/prepare_data.py --sample 20000        # 抽样调试
    python scripts/prepare_data.py --tokenize            # 额外保存分词结果（较慢）
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from udet.data import (  # noqa: E402
    CLEANED_CODE,
    dataset_summary,
    find_parquet,
    load_codet_m4,
    split_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="准备 CoDET-M4 数据")
    parser.add_argument("--raw-dir", default=str(PROJECT_ROOT / "data" / "raw" / "CoDET-M4"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "data" / "processed"))
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample", type=int, default=None, help="仅使用前 N 条（调试用）")
    parser.add_argument("--tokenize", action="store_true", help="额外保存 tokenize 结果")
    parser.add_argument("--tokenizer", default=str(PROJECT_ROOT / "checkpoints" / "codet5-base"))
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--num-proc", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    parquet = find_parquet(args.raw_dir)
    print(f"[prepare_data] 读取: {parquet}")
    ds = load_codet_m4(path=parquet)

    if args.sample:
        ds = ds.select(range(min(args.sample, len(ds))))
    print(f"[prepare_data] 样本数: {len(ds)}")

    # ---- 概览 ----
    summary = dataset_summary(ds)
    print("[prepare_data] 数据概览:")
    print(json.dumps(summary, ensure_ascii=False, indent=2)[:3000])

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.sample:
        summary["sampled"] = args.sample
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # ---- 划分 ----
    splits = split_dataset(
        ds,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        use_existing_split=args.sample is None,
    )
    for name, sub in splits.items():
        print(f"[prepare_data] {name:>10}: {len(sub):>7} 条")

    # ---- 可选：分词 ----
    if args.tokenize:
        from udet.encoders import build_tokenizer
        from udet.data import tokenize_dataset

        tokenizer = build_tokenizer(args.tokenizer)
        splits = splits.map(
            lambda x: tokenize_dataset(x, tokenizer, text_column=CLEANED_CODE, max_length=args.max_length, num_proc=args.num_proc),
            desc="tokenize",
        )
        print("[prepare_data] 已生成 input_ids / attention_mask")

    # ---- 保存（datasets arrow 格式，可 memory-map，适合大文件）----
    splits.save_to_disk(str(out_dir / "codet_m4"))
    print(f"[prepare_data] 已保存到: {out_dir / 'codet_m4'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
