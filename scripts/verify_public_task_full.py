"""Streaming integrity checks for the complete public corpus."""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FULL = ROOT / "d-det/data/public_same_task_full_2026-10-07"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    manifest = json.loads((FULL / "manifest.json").read_text(encoding="utf-8"))
    counts = Counter()
    tasks = defaultdict(set)
    units = set()
    seen = set()
    malformed = []
    with (FULL / "records.jsonl").open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = (row.get("source"), row.get("member"), row.get("task_id"), row.get("line", row.get("code_sha256")))
            if key in seen:
                malformed.append({"line": line_no, "reason": "duplicate_record_key", "key": key})
            seen.add(key)
            source = row.get("source")
            task = row.get("task_id")
            unit = "|".join(str(row.get(k, "")) for k in ("source", "model_id", "generation_mode", "subset", "backend", "temperature"))
            if source not in {"evalplus", "bigcodebench"} or not task or not row.get("asset_sha256"):
                malformed.append({"line": line_no, "reason": "missing_required_fields"})
            counts[source] += 1
            tasks[source].add(task)
            units.add(unit)
    report = {
        "schema": "public_same_task_full_integrity_v1",
        "manifest_row_counts": manifest.get("row_counts"),
        "observed_row_counts": dict(counts),
        "task_counts": {k: len(v) for k, v in tasks.items()},
        "exact_generator_units": len(units),
        "duplicate_or_malformed_count": len(malformed),
        "sample_errors": malformed[:20],
        "records_sha256": sha256(FULL / "records.jsonl"),
        "code_executed": False,
    }
    (FULL / "full_integrity_check.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    if counts.get("evalplus") != 8778 or counts.get("bigcodebench") != 287476:
        raise SystemExit("full corpus row count mismatch")
    if malformed:
        raise SystemExit("malformed or duplicate records")


if __name__ == "__main__":
    main()
