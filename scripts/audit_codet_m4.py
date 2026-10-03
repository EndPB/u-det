"""Stream-audit CoDET-M4 metadata and write a compact provenance summary."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq


def main():
    root = Path("d-det/data/codet_m4")
    path = root / "dataset_without_comments.parquet"
    pf = pq.ParquetFile(path)
    counters = {k: Counter() for k in ("language", "model", "target", "split", "source")}
    rows = 0
    null_code = 0
    for batch in pf.iter_batches(batch_size=8192, columns=["code", "language", "model", "target", "split", "source"]):
        cols = {name: batch.column(i).to_pylist() for i, name in enumerate(["code", "language", "model", "target", "split", "source"])}
        for i, code in enumerate(cols["code"]):
            rows += 1
            null_code += int(code is None or not str(code).strip())
            for name in counters:
                counters[name][str(cols[name][i])] += 1
    result = {
        "dataset": "DaniilOr/CoDET-M4",
        "revision": "4d4e665037cb797cb5381c0b95c1d33e30420b8e",
        "path": str(path),
        "rows": rows,
        "row_groups": pf.metadata.num_row_groups,
        "schema": [(f.name, str(f.type)) for f in pf.schema_arrow],
        "null_or_empty_code": null_code,
        "counts": {name: dict(sorted(counter.items())) for name, counter in counters.items()},
        "role": "external structural/source-fingerprint control; not a same-task semantic-pair corpus",
    }
    (root / "audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"rows": rows, "models": len(counters["model"]), "languages": dict(counters["language"]), "out": str((root/'audit.json').resolve())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
