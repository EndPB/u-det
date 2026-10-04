"""Audit h2_pair_benchmark_v1."""
from __future__ import annotations
import hashlib, json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if not (ROOT / "d-det" / "data" / "h2_pair_benchmark_v1").exists():
    ROOT = Path(__file__).resolve().parents[2]  # 脚本位于 d-det/scripts 时回到仓库根
DATA = ROOT / "d-det/data/h2_pair_benchmark_v1"
VIEWS = {"raw", "ids_only", "strings_only", "comments_only", "all"}

def digest(text): return hashlib.sha256(text.encode("utf-8")).hexdigest()

def jaccard(left, right):
    a = set(re.findall(r"[A-Za-z_][A-Za-z_0-9]*", left)); b = set(re.findall(r"[A-Za-z_][A-Za-z_0-9]*", right))
    return len(a & b) / len(a | b) if a or b else 0.0

def main() -> None:
    summary = json.loads((DATA / "summary.json").read_text(encoding="utf-8"))
    source_paths = {
        "h2_pairs_v2_same_family": ROOT / "d-det/data/h2_pairs_v2/same_family_pairs.jsonl",
        "authorbench_hardneg_full": ROOT / "d-det/data/h2_authorbench_dcan_hardneg_pairs_v1/same_task_cross_family_pairs.jsonl",
        "llm_codegen_v2_hardneg_full": ROOT / "d-det/data/h2_llm_codegen_v2_hardneg_pairs_v1/same_task_cross_family_pairs.jsonl",
    }
    source_hashes = {}
    for name, path in source_paths.items():
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""): h.update(chunk)
        source_hashes[name] = h.hexdigest()
    ids, rows, counts, task_labels = set(), [], Counter(), Counter(); invalid = 0
    task_splits = defaultdict(set); raw_hash_splits = defaultdict(set)
    with (DATA / "pairs.jsonl").open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip(): continue
            row = json.loads(line); pid = row.get("pair_id"); source = row.get("source"); label = row.get("pair_label")
            bad = not pid or pid in ids or source not in {"authorbench_dcan", "llm_codegen_v2"} or label not in {0, 1}
            positive = label == 1
            if positive:
                bad = bad or row["pair_type"] != "same_family_cross_generator" or row["left_family"] != row["right_family"] or row.get("family") != row["left_family"] or row["positive_for_family_invariance"] is not True
            else:
                bad = bad or row["pair_type"] != "same_task_cross_family_hard_negative" or row["left_family"] == row["right_family"] or row["positive_for_family_invariance"] is not False
            bad = bad or row["left_generator"] == row["right_generator"] or not row.get("task_id")
            left, right = row.get("left_views", {}), row.get("right_views", {})
            lh, rh = row.get("left_view_code_sha256", {}), row.get("right_view_code_sha256", {})
            bad = bad or set(left) != VIEWS or set(right) != VIEWS or set(lh) != VIEWS or set(rh) != VIEWS
            if set(left) == VIEWS and set(right) == VIEWS and set(lh) == VIEWS and set(rh) == VIEWS:
                bad = bad or any(not left[k].strip() or not right[k].strip() or lh[k] != digest(left[k]) or rh[k] != digest(right[k]) for k in VIEWS)
                raw_hash_splits[(source, lh["raw"])].add(row.get("task_split")); raw_hash_splits[(source, rh["raw"])].add(row.get("task_split"))
            ids.add(pid); rows.append(row); counts[(source, row.get("task_split"), label)] += 1; task_labels[(source, row.get("task_id"), label)] += 1
            task_splits[(source, row.get("task_id"))].add(row.get("task_split"))
            invalid += int(bad)
    balanced_rows = [json.loads(line) for line in (DATA / "llm_family_balanced_pairs.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    family_counts = Counter((r["task_split"], r["family"]) for r in balanced_rows if r["pair_label"] == 1)
    balanced_counts = Counter((r["task_split"], r["pair_label"]) for r in balanced_rows)
    expected = {"authorbench_dcan": {"train": 1, "dev": 1, "test": 1}, "llm_codegen_v2": {"train": 1, "dev": 1, "test": 1}}
    task_rows = [json.loads(line) for line in (DATA / "task_balanced_pairs.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    task_cells = Counter((r["source"], r["task_split"], r["task_id"], r["pair_label"]) for r in task_rows)
    main_by_id = {r["pair_id"]: r for r in rows}
    family_rows = [json.loads(line) for line in (DATA / "family_pair_balanced_pairs.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    family_ids = set(); family_invalid = 0
    family_cells = Counter((r["source"], r["task_split"], r["task_id"], r["pair_label"]) for r in family_rows)
    family_counts = Counter((r["source"], r["task_split"], tuple(sorted((r["left_family"], r["right_family"])))) for r in family_rows if r["pair_label"] == 0)
    for row in family_rows:
        pid = row["pair_id"]
        family_invalid += int(pid in family_ids or pid not in main_by_id or row != main_by_id.get(pid))
        family_ids.add(pid)
    triplet_rows = [json.loads(line) for line in (DATA / "triplet_index.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    triplet_ids = set(); triplet_invalid = 0
    for row in triplet_rows:
        tid = row["triplet_id"]; triplet_invalid += int(tid in triplet_ids or row["anchor_pair_id"] not in main_by_id or row["positive_pair_id"] not in main_by_id or row["negative_pair_id"] not in main_by_id)
        triplet_ids.add(tid)
        pos = main_by_id.get(row["positive_pair_id"]); neg = main_by_id.get(row["negative_pair_id"])
        if pos and neg:
            triplet_invalid += int(pos["pair_label"] != 1 or neg["pair_label"] != 0 or pos["source"] != row["source"] or neg["source"] != row["source"] or pos["task_id"] != row["task_id"] or neg["task_id"] != row["task_id"] or pos["task_split"] != row["task_split"] or neg["task_split"] != row["task_split"] or row["anchor_family"] != pos["left_family"] or row["positive_family"] != pos["right_family"] or row["negative_family"] == row["anchor_family"])
    lexical_rows = [json.loads(line) for line in (DATA / "lexical_hardneg_index.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    lexical_ids = set(); lexical_invalid = 0
    for row in lexical_rows:
        sid = row["selection_id"]; lexical_invalid += int(sid in lexical_ids or row["pair_id"] not in main_by_id)
        lexical_ids.add(sid)
        pair = main_by_id.get(row["pair_id"])
        if pair:
            ids_score = jaccard(pair["left_views"]["ids_only"], pair["right_views"]["ids_only"]); strings_score = jaccard(pair["left_views"]["strings_only"], pair["right_views"]["strings_only"])
            lexical_invalid += int(pair["pair_label"] != 0 or pair["source"] != row["source"] or pair["task_id"] != row["task_id"] or pair["task_split"] != row["task_split"] or abs(ids_score - row["ids_jaccard"]) > 1e-12 or abs(strings_score - row["strings_jaccard"]) > 1e-12 or row["mean_jaccard"] != 0.5 * (row["ids_jaccard"] + row["strings_jaccard"]))
    multi_rows = [json.loads(line) for line in (DATA / "task_multi_negative_index.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    multi_ids = set(); multi_invalid = 0
    for row in multi_rows:
        sid = row["selection_id"]
        multi_invalid += int(sid in multi_ids or not row["positive_pair_ids"] or not row["negative_pair_ids"])
        multi_ids.add(sid)
        positive_ids, negative_ids = row["positive_pair_ids"], row["negative_pair_ids"]
        multi_invalid += int(len(positive_ids) != row["positive_count"] or len(negative_ids) != row["negative_count"])
        multi_invalid += int(len(set(positive_ids)) != len(positive_ids) or len(set(negative_ids)) != len(negative_ids))
        multi_invalid += int(any(pid not in main_by_id or main_by_id[pid]["pair_label"] != 1 for pid in positive_ids))
        multi_invalid += int(any(pid not in main_by_id or main_by_id[pid]["pair_label"] != 0 for pid in negative_ids))
        for pid in positive_ids + negative_ids:
            if pid in main_by_id:
                pair = main_by_id[pid]
                multi_invalid += int(pair["source"] != row["source"] or pair["task_split"] != row["task_split"] or pair["task_id"] != row["task_id"])
        listed_neg_ids = [x["pair_id"] for x in row["negative_family_pairs"]]
        multi_invalid += int(listed_neg_ids != negative_ids)
        for item in row["negative_family_pairs"]:
            pair = main_by_id.get(item["pair_id"])
            multi_invalid += int(pair is None or pair["left_family"] != item["left_family"] or pair["right_family"] != item["right_family"] or pair["left_generator"] != item["left_generator"] or pair["right_generator"] != item["right_generator"])
    fold_summary = json.loads((DATA / "generator_fold_summary.json").read_text(encoding="utf-8"))
    fold_ids = {f["fold_id"] for f in fold_summary["folds"]}; fold_seen = Counter(); fold_invalid = 0
    for line in (DATA / "generator_fold_index.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip(): continue
        row = json.loads(line); key = (row["fold_id"], row["pair_id"]); fold_seen[key] += 1
        bad = row["fold_id"] not in fold_ids or row["role"] not in {"train", "dev", "test"} or row["task_split"] != row["role"]
        endpoints = {row["left_generator"], row["right_generator"]}
        if row["role"] in {"train", "dev"}: bad = bad or row["heldout_generator"] in endpoints
        else: bad = bad or row["heldout_generator"] not in endpoints
        fold_invalid += int(bad)
    checks = {
        "summary_pair_count": len(rows) == summary["pair_records"],
        "source_hashes_match": source_hashes == summary["source_hashes"],
        "summary_positive_count": sum(r["pair_label"] for r in rows) == summary["positive_records"],
        "pair_ids_unique": len(ids) == len(rows),
        "records_valid": invalid == 0,
        "source_split_balanced": all(counts[(s, sp, 0)] == counts[(s, sp, 1)] for s in expected for sp in expected[s]),
        "source_split_nonempty": all(counts[(s, sp, 1)] > 0 for s in expected for sp in expected[s]),
        "no_cross_source_task_collision": not any(sum(1 for r in rows if r["task_id"] == tid and r["source"] != source) for source, tid, _ in task_labels),
        "task_split_disjoint": all(len(v) == 1 for v in task_splits.values()),
        "raw_view_hash_split_disjoint": all(len(v) == 1 for v in raw_hash_splits.values()),
        "llm_family_balanced_three_families": all(len({family_counts[(sp, fam)] for fam in ("google", "meta", "mistral")}) == 1 for sp in ("train", "dev", "test")),
        "llm_family_balanced_labels_equal": all(balanced_counts[(sp, 0)] == balanced_counts[(sp, 1)] for sp in ("train", "dev", "test")),
        "task_balanced_one_row_per_cell": all(n == 1 for n in task_cells.values()),
        "task_balanced_exact_subset": all(r["pair_id"] in main_by_id and r == main_by_id[r["pair_id"]] for r in task_rows),
        "llm_family_balanced_exact_subset": all(r["pair_id"] in main_by_id and r == main_by_id[r["pair_id"]] for r in balanced_rows),
        "task_balanced_labels_and_tasks_equal": all({(r["task_id"], r["pair_label"]) for r in task_rows if r["source"] == s and r["task_split"] == sp and r["pair_label"] == 0} and {r["task_id"] for r in task_rows if r["source"] == s and r["task_split"] == sp and r["pair_label"] == 0} == {r["task_id"] for r in task_rows if r["source"] == s and r["task_split"] == sp and r["pair_label"] == 1} for s in expected for sp in expected[s]),
        "triplet_index_valid": triplet_invalid == 0 and len(triplet_ids) == summary["triplets"]["records"],
        "lexical_hardneg_index_valid": lexical_invalid == 0 and len(lexical_ids) == summary["lexical_hardneg"]["records"],
        "task_multi_negative_index_valid": multi_invalid == 0 and len(multi_ids) == summary["task_multi_negative"]["records"],
        "family_pair_balanced_exact_subset": family_invalid == 0,
        "family_pair_balanced_one_row_per_task_label": all(n == 1 for n in family_cells.values()),
        "family_pair_balanced_labels_equal": all(sum(1 for r in family_rows if r["source"] == s and r["task_split"] == sp and r["pair_label"] == 0) == sum(1 for r in family_rows if r["source"] == s and r["task_split"] == sp and r["pair_label"] == 1) for s, sp in {(r["source"], r["task_split"]) for r in family_rows}),
        "family_pair_balanced_negative_pairs_equal": all(len({n for (s, sp, pair), n in family_counts.items() if s == source and sp == split}) == 1 for source, split in {(r["source"], r["task_split"]) for r in family_rows}),
        "family_pair_balanced_summary_count": len(family_rows) == summary["family_pair_balanced"]["records"],
        "generator_fold_ids_valid": len(fold_ids) == fold_summary["fold_count"] and len(fold_seen) == sum(fold_seen.values()),
        "generator_fold_rows_valid": fold_invalid == 0,
        "unsupported_folds_explicitly_marked": all(not f["trainable_positive_support"] for f in fold_summary["folds"] if len(f["seen_generators"]) == 1),
    }
    audit = {"schema": "h2_pair_benchmark_v1_audit", "rows": len(rows), "by_source_split_label": {f"{s}|{sp}|label={y}": n for (s, sp, y), n in sorted(counts.items())}, "task_balanced_rows": len(task_rows), "triplet_rows": len(triplet_rows), "triplet_invalid_records": triplet_invalid, "lexical_hardneg_rows": len(lexical_rows), "lexical_hardneg_invalid_records": lexical_invalid, "task_multi_negative_rows": len(multi_rows), "task_multi_negative_invalid_records": multi_invalid, "family_pair_balanced_rows": len(family_rows), "family_pair_balanced_invalid_records": family_invalid, "family_pair_balanced_by_source_split_family_pair": {f"{s}|{sp}|{'~'.join(pair)}": n for (s, sp, pair), n in sorted(family_counts.items())}, "llm_family_balanced_rows": len(balanced_rows), "llm_family_balanced_by_split_label": {f"{sp}|label={y}": n for (sp, y), n in sorted(balanced_counts.items())}, "generator_fold_count": fold_summary["fold_count"], "generator_fold_index_rows": sum(fold_seen.values()), "generator_fold_invalid_rows": fold_invalid, "invalid_records": invalid, "checks": checks, "notes": ["Pair views are deterministic transforms of the same endpoints.", "Use task-clustered evaluation; pair rows are not independent.", "AuthorBench positive pairs are OpenAI-only; the separate LLM view is the family-balanced diagnostic.", "The family-pair-balanced view is a strict diagnostic subset and may omit a source/split family pair when task overlap prevents a complete matching.", "Google and Mistral generator-heldout folds have no train/dev positive support and are diagnostic-only."]}
    (DATA / "audit_index.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"checks": checks, "rows": len(rows), "invalid_records": invalid}, ensure_ascii=False))
    if not all(checks.values()): raise SystemExit(1)

if __name__ == "__main__": main()
