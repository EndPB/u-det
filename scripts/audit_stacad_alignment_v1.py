"""Audit the existing compact STACAD same-task alignment package."""
from __future__ import annotations

import collections
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "d-det" / "data" / "h2_stacad_alignment_v1"
OUT = DATA / "audit_index.json"


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def main() -> None:
    core_rows = 0
    core_tasks = set()
    core_splits = collections.Counter()
    core_generators = collections.Counter()
    core_languages = collections.Counter()
    core_code_hashes = set()
    core_empty = 0
    for obj in read_jsonl(DATA / "core.jsonl"):
        core_rows += 1
        core_tasks.add(obj["task_id"])
        core_splits[obj["task_split"]] += 1
        core_generators[obj["generator"]] += 1
        core_languages[obj["language"]] += 1
        code = obj.get("code") or ""
        core_empty += not bool(code.strip())
        core_code_hashes.add(obj.get("code_sha256") or hashlib.sha256(code.encode()).hexdigest())

    task_rows = 0
    task_ids = set()
    task_splits = {}
    for obj in read_jsonl(DATA / "task_index.jsonl"):
        task_rows += 1
        task_ids.add(obj["task_id"])
        task_splits[obj["task_id"]] = obj["task_split"]

    pair_rows = 0
    pair_ids = set()
    pair_tasks = set()
    pair_types = collections.Counter()
    pair_positive = 0
    pair_same_family = 0
    for obj in read_jsonl(DATA / "pair_index.jsonl"):
        pair_rows += 1
        pair_ids.add(obj["pair_id"])
        pair_tasks.add(obj["task_id"])
        pair_types[obj["pair_type"]] += 1
        pair_positive += bool(obj.get("positive_for_family_invariance"))
        pair_same_family += obj["left_family"] == obj["right_family"]

    split_overlap = []
    split_sets = collections.defaultdict(set)
    for task_id, split in task_splits.items():
        split_sets[split].add(task_id)
    names = sorted(split_sets)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            overlap = split_sets[a] & split_sets[b]
            if overlap:
                split_overlap.append({"left": a, "right": b, "count": len(overlap)})

    result = {
        "schema": "h2_stacad_alignment_v1_audit",
        "core": {
            "rows": core_rows,
            "tasks": len(core_tasks),
            "splits": dict(sorted(core_splits.items())),
            "generators": dict(sorted(core_generators.items())),
            "languages": dict(sorted(core_languages.items())),
            "empty_code": core_empty,
            "distinct_code_sha256": len(core_code_hashes),
        },
        "task_index": {"rows": task_rows, "distinct_task_ids": len(task_ids)},
        "pairs": {
            "rows": pair_rows,
            "distinct_pair_ids": len(pair_ids),
            "distinct_tasks": len(pair_tasks),
            "types": dict(sorted(pair_types.items())),
            "positive_for_family_invariance_true": pair_positive,
            "same_family_pairs": pair_same_family,
        },
        "checks": {
            "core_task_index_match": core_tasks == task_ids,
            "core_nonempty_unique_code": core_rows == len(core_code_hashes) and core_empty == 0,
            "pair_ids_unique": pair_rows == len(pair_ids),
            "pairs_reference_known_tasks": pair_tasks <= task_ids,
            "all_pairs_cross_family": pair_same_family == 0,
            "all_pairs_negative": pair_positive == 0,
            "no_task_split_overlap": not split_overlap,
        },
        "task_split_overlap": split_overlap,
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["checks"], ensure_ascii=False))


if __name__ == "__main__":
    main()
