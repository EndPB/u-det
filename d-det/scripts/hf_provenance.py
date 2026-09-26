#!/usr/bin/env python
"""HF 数据集溯源检查：HF test 分片与官方 B 数据（test 样本/train/val）的重叠。"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
HF = ROOT / "data/raw/hf_preprocessed"
OFF = ROOT / "data/raw/SemEval-2026-Task13/task_b"


def norm(c: str) -> str:
    return c.replace("\r\n", "\n").replace("\r", "\n").strip()


def hashes(path: Path, col: str = "code", limit: int | None = None, norm_it: bool = True) -> set:
    codes = pq.read_table(path, columns=[col]).column(col).to_pylist()
    if limit:
        codes = codes[:limit]
    out = set()
    for c in codes:
        s = norm(c) if norm_it else c
        out.add(hashlib.sha1(s.encode("utf-8", "ignore")).hexdigest())
    return out


def main() -> int:
    # 官方参照集
    off_te1k = hashes(OFF / "task_b_test_set_sample.parquet")
    off_tr5k = hashes(OFF / "task_b_training_set.parquet", limit=5000)
    off_va5k = hashes(OFF / "task_b_validation_set.parquet", limit=5000)
    print(f"official: test1k={len(off_te1k)} train5k={len(off_tr5k)} val5k={len(off_va5k)}")

    for shard in range(4):
        p = HF / f"test-{shard}.parquet"
        if not p.exists():
            print(f"test-{shard}: 未下载")
            continue
        h = hashes(p)
        n = pq.ParquetFile(p).metadata.num_rows
        print(f"test-{shard}: n={n} uniq={len(h)} | "
              f"∩test1k={len(h & off_te1k)}/1000 | "
              f"∩train5k={len(h & off_tr5k)}/5000 | "
              f"∩val5k={len(h & off_va5k)}/5000")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
