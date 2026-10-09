"""Server-side reconstruction of the code-conditioned round inputs (2026-10-09).

Rebuilds from the server's public corpus copy:
  - records_train_dev.jsonl : 11 members x 969 train/dev tasks, instruct mode (10,659 rows)
  - family_series_admission.json : 3 observed series -> 11 members (from the frozen
    endpoint-balanced plan registry; family_is_confirmed=false for all)
  - input_hashes.json : input hashes, row-order hash, split/member-mapping hashes,
    ordered code SHA-256 list hash (per round guidance section 1).

Mode choice evidence (server-side): instruct-mode reconstruction gives exactly
10,659/10,659 AST-parseable rows and 10,602/10,659 with top-level entry point,
matching the local preflight numbers; complete mode gives 10,594 entry points.

No prompt/test/canonical-solution fields are read; only
(model_id, task_id, split, solution, solution_sha256, provenance) are written.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def ordered_hash(values) -> str:
    return sha256_bytes(("\n".join(values) + "\n").encode("utf-8"))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--records", type=Path,
                    default="d-det/data/public_same_task_full_2026-10-07/records.jsonl")
    ap.add_argument("--split-csv", type=Path,
                    default="d-det/artifacts/public_full_receive_2026-10-08/prereg/split_index.csv")
    ap.add_argument("--series-plan", type=Path,
                    default="d-det/artifacts/endpoint_balanced_relation_plan_2026-10-08/endpoint_balanced_plan.json")
    ap.add_argument("--out", type=Path,
                    default="d-det/artifacts/code_conditioned_design_2026-10-09/inputs")
    a = ap.parse_args()

    a.out.mkdir(parents=True, exist_ok=True)

    whitelist = {}
    with a.split_csv.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["split"] in ("train", "dev"):
                whitelist[r["task_id"]] = r["split"]
    assert len(whitelist) == 969, len(whitelist)

    plan = json.loads(a.series_plan.read_text(encoding="utf-8"))
    series_members = plan["series_members"]
    series_of = {m: s for s, ms in series_members.items() for m in ms}
    assert len(series_of) == 11, len(series_of)

    rows = []
    with a.records.open(encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d.get("model_id") not in series_of:
                continue
            if d.get("subset") != "full":
                continue
            if d.get("task_id") not in whitelist:
                continue
            mode = d.get("generation_mode")
            if mode != "instruct":
                continue
            code = d.get("solution")
            assert isinstance(code, str), d["member"]  # 2 rows are genuinely empty strings
            assert sha256_bytes(code.encode("utf-8")) == d["solution_sha256"], d["member"]
            rows.append({
                "model_id": d["model_id"],
                "series": series_of[d["model_id"]],
                "task_id": d["task_id"],
                "split": whitelist[d["task_id"]],
                "code": code,
                "solution_sha256": d["solution_sha256"],
                "generation_mode": "instruct",
                "subset": "full",
                "source": d.get("source"),
                "asset": d.get("asset"),
            })

    keys = [(r["model_id"], r["task_id"]) for r in rows]
    assert len(keys) == len(set(keys)), "duplicate (model, task) rows"
    assert len(rows) == 11 * 969, len(rows)
    n_by_split = {s: sum(r["split"] == s for r in rows) for s in ("train", "dev")}
    assert n_by_split == {"train": 11 * 798, "dev": 11 * 171}, n_by_split

    records_out = a.out / "records_train_dev.jsonl"
    with records_out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    admission = {
        "schema": "family_series_admission_v1",
        "created": "2026-10-09",
        "source_status": "server_reconstruction_only",
        "family_is_confirmed": False,
        "admission_note": ("observed series are official release families; members are size "
                           "variants within an observed series; family_is_confirmed=false for all"),
        "series": [{"series": s, "members": [{"model_id": m} for m in ms]}
                   for s, ms in series_members.items()],
    }
    (a.out / "family_series_admission.json").write_text(
        json.dumps(admission, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # row order / split / member / code hashes
    hashes = {
        "schema": "code_conditioned_inputs_hashes_v1",
        "generated_utc": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc).isoformat(),
        "row_order": "input records.jsonl order (filtered)",
        "mode": "instruct",
        "input_sha256": {
            "records.jsonl": sha256_file(a.records),
            "split_index.csv": sha256_file(a.split_csv),
            "endpoint_balanced_plan.json": sha256_file(a.series_plan),
        },
        "output_sha256": {
            "records_train_dev.jsonl": sha256_file(records_out),
            "family_series_admission.json": sha256_file(a.out / "family_series_admission.json"),
        },
        "counts": {"rows": len(rows), "tasks": len(whitelist), "members": len(series_of),
                   "split_rows": n_by_split,
                   "split_tasks": {s: sum(1 for t, sp in whitelist.items() if sp == s)
                                   for s in ("train", "dev")}},
        "row_order_sha256": ordered_hash([f"{m}\t{t}" for m, t in keys]),
        "task_split_sha256": {
            s: ordered_hash(sorted(t for t, sp in whitelist.items() if sp == s))
            for s in ("train", "dev")},
        "member_mapping_sha256": ordered_hash(
            [f"{m}\t{series_of[m]}" for m in sorted(series_of)]),
        "code_sha256_list_sha256": ordered_hash([r["solution_sha256"] for r in rows]),
        "member_row_counts": {m: sum(r["model_id"] == m for r in rows) for m in sorted(series_of)},
    }
    (a.out / "input_hashes.json").write_text(
        json.dumps(hashes, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows), "tasks": len(whitelist), "members": len(series_of),
                      "row_order_sha256": hashes["row_order_sha256"][:16],
                      "records_sha256": hashes["output_sha256"]["records_train_dev.jsonl"][:16]}))
    # write SHA256SUMS for the inputs dir
    files = sorted(p for p in a.out.glob("*") if p.name != "SHA256SUMS.txt")
    (a.out / "SHA256SUMS.txt").write_text(
        "".join(f"{sha256_file(p)}  {p.name}\n" for p in files), encoding="utf-8")


if __name__ == "__main__":
    main()
