"""Stream-audit the downloaded AICD parquet shards without loading a dataset.

The public AICD parquet files contain only ``code`` and numeric ``label``.
This audit records the physical schema, split sizes, label counts and code
length summaries. It deliberately does not decode or materialize the full
corpus.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq


def quantiles(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {"min": None, "p25": None, "median": None, "p75": None, "p95": None, "max": None}
    values = sorted(values)
    def q(p: float):
        if len(values) == 1:
            return values[0]
        pos = (len(values) - 1) * p
        lo = math.floor(pos)
        hi = math.ceil(pos)
        if lo == hi:
            return values[lo]
        return values[lo] + (values[hi] - values[lo]) * (pos - lo)
    return {"min": values[0], "p25": q(.25), "median": q(.5), "p75": q(.75), "p95": q(.95), "max": values[-1]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("d-det/data/acl_attribution_collection_v1/raw/aicd"))
    ap.add_argument("--out", type=Path, default=Path("d-det/data/acl_attribution_collection_v1/raw/aicd_audit.json"))
    ap.add_argument("--reservoir", type=int, default=20000)
    args = ap.parse_args()
    files = sorted(args.root.glob("**/*.parquet"))
    if not files:
        raise SystemExit(f"No parquet files under {args.root}")

    rng = random.Random(20261003)
    result: dict = {
        "root": str(args.root.resolve()),
        "files": [],
        "configs": {},
        "total_rows": 0,
        "schema_variants": [],
        "warnings": [
            "The public parquet schema exposes only code and numeric label; family/generator/language metadata is not present in these shards.",
            "Numeric label IDs must not be interpreted as family names without an external release mapping.",
        ],
    }
    schema_seen = set()

    for path in files:
        rel = path.relative_to(args.root).as_posix()
        parts = path.relative_to(args.root).parts
        config = parts[0] if len(parts) > 1 else "unknown"
        split = path.stem.rsplit("-", 1)[0]
        pf = pq.ParquetFile(path)
        schema = [(f.name, str(f.type), f.nullable) for f in pf.schema_arrow]
        schema_key = json.dumps(schema, sort_keys=True)
        schema_seen.add(schema_key)
        labels: Counter[str] = Counter()
        lengths: list[int] = []
        rows = 0
        null_code = 0
        empty_code = 0
        for batch in pf.iter_batches(batch_size=8192, columns=["code", "label"]):
            codes = batch.column(0).to_pylist()
            labs = batch.column(1).to_pylist()
            for code, label in zip(codes, labs):
                rows += 1
                labels[str(label)] += 1
                if code is None:
                    null_code += 1
                    length = 0
                else:
                    length = len(code)
                    if length == 0:
                        empty_code += 1
                if len(lengths) < args.reservoir:
                    lengths.append(length)
                else:
                    j = rng.randrange(rows)
                    if j < args.reservoir:
                        lengths[j] = length

        cfg = result["configs"].setdefault(config, {"splits": {}, "rows": 0, "labels": Counter()})
        split_info = {
            "rows": rows,
            "labels": dict(sorted(labels.items(), key=lambda kv: int(kv[0]))),
            "code_length_sample_n": len(lengths),
            "code_length_sample_quantiles": quantiles(lengths),
            "null_code": null_code,
            "empty_code": empty_code,
        }
        cfg["splits"][split] = split_info
        cfg["rows"] += rows
        cfg["labels"].update(labels)
        result["total_rows"] += rows
        result["files"].append({"path": rel, "bytes": path.stat().st_size, "rows": rows, "schema": schema})

    for cfg in result["configs"].values():
        cfg["labels"] = dict(sorted(cfg["labels"].items(), key=lambda kv: int(kv[0])))
    result["schema_variants"] = [json.loads(x) for x in sorted(schema_seen)]
    result["file_count"] = len(files)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"files": len(files), "rows": result["total_rows"], "out": str(args.out.resolve())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
