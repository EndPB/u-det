"""Stream canonical data and validate actual support before any new ACL training.

No model imports, score computation, resplitting, or changes to source datasets.
Metadata-only counts include test: this is protocol preparation, not model selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CANONICAL = {
    "authorbench_complete": "h2_authorbench/core.jsonl",
    "authorbench_dcan": "h2_authorbench_dcan/core.jsonl",
    "llm_codegen_v2": "h2_llm_codegen_v2/paired_examples.jsonl",
    "stacad_alignment": "h2_stacad_alignment_v1/core.jsonl",
    "droid_full": "h2_droid_full_selected/core.jsonl",
    "droid_subset": "h2_droid_subset/core.jsonl",
    # These two controls use the derived package's explicit train/val/test
    # files rather than a single core.jsonl.  Keep the split files separate in
    # the audit so a missing core file cannot hide an available package.
    "codet_m4_control": [
        "codet_m4_balanced_control_v1/train.jsonl",
        "codet_m4_balanced_control_v1/val.jsonl",
        "codet_m4_balanced_control_v1/test.jsonl",
    ],
    "aicd_numeric_control": [
        "aicd_t2_numeric_balanced_v1/train.jsonl",
        "aicd_t2_numeric_balanced_v1/validation.jsonl",
        "aicd_t2_numeric_balanced_v1/test.jsonl",
    ],
}


def stream(path):
    with path.open("rb") as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                yield number, line, json.loads(line)


def first(row, *keys):
    return next((row[k] for k in keys if row.get(k) not in (None, "")), None)


def summarize(path):
    paths = path if isinstance(path, list) else [path]
    paths = [Path(p) for p in paths]
    digest = hashlib.sha256()
    counts, missing, labels = Counter(), Counter(), Counter()
    family_generators, tasks, hashes, task_generators = (defaultdict(set) for _ in range(4))
    gen_family = defaultdict(set)
    rows = 0
    for current in paths:
        for number, line, row in stream(current):
            digest.update(line)
            rows += 1
            family = first(row, "family", "Model_Family")
            generator = first(row, "generator", "model_name", "Generator", "model")
            task = first(row, "task_id", "prompt_id")
            split = first(row, "task_split", "split_source", "split") or "unknown"
            language = first(row, "language", "Language")
            label = first(row, "Label", "label", "target")
            code = first(row, "code", "Code", "c_code")
            if not code and isinstance(row.get("views"), dict):
                code = first(row["views"], "code_only", "raw")
            for name, value in (("family", family), ("generator", generator), ("task_id", task),
                                ("language", language), ("base_model", row.get("base_model")),
                                ("model_role", row.get("model_role")), ("repository_id", row.get("repository_id"))):
                missing[name] += value is None
            labels[str(label)] += 1
            counts[f"{split}|{family}|{generator}|{language}"] += 1
            if family and generator:
                family_generators[family].add(generator)
                gen_family[generator].add(family)
            if task:
                tasks[task].add(split)
                if family and generator:
                    task_generators[(task, split, family)].add(generator)
            if isinstance(code, str) and code.strip():
                hashes[hashlib.sha256(code.encode("utf-8")).hexdigest()].add(split)
    positive = Counter()
    for (_, split, family), gens in task_generators.items():
        positive[f"{split}|{family}"] += len(gens) * (len(gens) - 1) // 2
    task_leaks = {t: sorted(s) for t, s in tasks.items() if len(s) > 1}
    code_leaks = {h: sorted(s) for h, s in hashes.items() if len(s) > 1}
    multigen = {f: sorted(g) for f, g in sorted(family_generators.items())}
    return {
        "path": [p.relative_to(ROOT).as_posix() for p in paths] if len(paths) > 1 else paths[0].relative_to(ROOT).as_posix(),
        "bytes": sum(p.stat().st_size for p in paths),
        "sha256_nonblank_lines": digest.hexdigest(), "rows": rows,
        "missing": dict(missing), "labels": dict(labels), "cells": dict(sorted(counts.items())),
        "family_generators": multigen, "generator_family_conflicts": {g: sorted(f) for g, f in gen_family.items() if len(f) > 1},
        "tasks": len(tasks), "task_cross_split_count": len(task_leaks), "task_cross_split_examples": dict(list(task_leaks.items())[:10]),
        "code_cross_split_count": len(code_leaks), "code_cross_split_examples": dict(list(code_leaks.items())[:10]),
        "same_task_cross_generator_candidates": dict(sorted(positive.items())),
        "families_with_at_least_3_generators": [f for f, g in multigen.items() if len(g) >= 3],
        "explicit_base_role_coverage": rows - max(missing["base_model"], missing["model_role"]),
        "base_instruct_causal_claim_ready": False,
        "h2_task_aware_candidate": bool(tasks and positive and not task_leaks),
        "caveat": "Candidate counts are metadata upper bounds; identical code/prompt mismatch and generator-heldout positive support need separate checks. Family names remain source-specific. No model role is inferred from model-name spelling.",
    }


def audit_pair_folds(data):
    pair_path = data / "h2_pair_benchmark_v1/pairs.jsonl"
    fold_path = data / "h2_pair_benchmark_v1/generator_fold_index.jsonl"
    pairs, ids, invalid, task_splits = {}, set(), [], defaultdict(set)
    for number, _, row in stream(pair_path):
        pid = row.get("pair_id")
        pos = row.get("pair_label") == 1
        same = row.get("left_family") == row.get("right_family")
        checks = [bool(pid), pid not in ids, row.get("pair_label") in (0, 1), pos == same,
                  row.get("positive_for_family_invariance") == pos,
                  row.get("left_generator") != row.get("right_generator"), bool(row.get("task_id"))]
        if not all(checks):
            invalid.append(number)
        ids.add(pid)
        pairs[pid] = {k: row.get(k) for k in ("source", "task_id", "task_split", "pair_label", "left_generator", "right_generator")}
        task_splits[(row.get("source"), row.get("task_id"))].add(row.get("task_split"))
    folds = defaultdict(lambda: {"counts": Counter(), "gens": defaultdict(set), "tasks": defaultdict(set), "seen_ids": set(), "invalid": []})
    for number, _, row in stream(fold_path):
        f = folds[row["fold_id"]]
        role, heldout, pid = row["role"], row["heldout_generator"], row["pair_id"]
        endpoints = {row["left_generator"], row["right_generator"]}
        base = pairs.get(pid)
        valid = base is not None and all(row.get(k) == v for k, v in (base or {}).items())
        valid = valid and pid not in f["seen_ids"] and role in {"train", "dev", "test"}
        valid = valid and ((heldout not in endpoints and row["task_split"] == role) if role in {"train", "dev"}
                           else heldout in endpoints and row["task_split"] == "test")
        if not valid:
            f["invalid"].append(number)
        f["seen_ids"].add(pid)
        f["counts"][f"{role}|{row['pair_label']}"] += 1
        f["gens"][role].update(endpoints)
        f["tasks"][role].add(row["task_id"])
    results = []
    for name, f in sorted(folds.items()):
        task_overlap = {f"{a}|{b}": len(f["tasks"][a] & f["tasks"][b]) for a, b in combinations(["train", "dev", "test"], 2)}
        support = all(f["counts"][f"{role}|{label}"] > 0 for role in ("train", "dev", "test") for label in (0, 1))
        results.append({"fold_id": name, "counts": dict(f["counts"]), "invalid_rows": len(f["invalid"]),
                        "task_overlap": task_overlap, "binary_training_support": support,
                        "admitted": support and not f["invalid"] and not any(task_overlap.values()),
                        "task": "same-family pair verification, NOT K-class family attribution"})
    return {"pairs": len(pairs), "invalid_pair_rows": invalid[:10], "invalid_pair_count": len(invalid),
            "task_split_conflicts": sum(len(s) > 1 for s in task_splits.values()), "folds": results,
            "admitted_count": sum(r["admitted"] for r in results),
            "test_access": "Metadata and positive-support audit only. No scores or hyperparameter selection."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "d-det/artifacts/acl_local_stage_a_2026-10-07")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    matrices = {}
    for name, relative in CANONICAL.items():
        relatives = relative if isinstance(relative, list) else [relative]
        paths = [ROOT / "d-det/data" / item for item in relatives]
        matrices[name] = summarize(paths) if all(path.exists() for path in paths) else {"path": relative, "status": "missing"}
        print(name, matrices[name].get("rows", "missing"), flush=True)
    pair_audit = audit_pair_folds(ROOT / "d-det/data")
    report = {"schema": "acl_local_stage_a_v1", "utc": datetime.now(timezone.utc).isoformat(),
              "python": platform.python_version(), "data_roles": matrices, "pair_fold_audit": pair_audit,
              "training_started": False, "source_files_modified": False,
              "access_boundary": "Complete streaming metadata/hash inspection; no test metric evaluation.",
              "next_gate": "Register features/P0 controls and assess same-base role metadata before starting training."}
    (args.out / "data_role_matrix.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# 本机阶段 A：数据角色与留出支持审计", "", "流式读取；未训练模型、未调参、未计算 test 分数。", "",
             "| 数据 | 行数 | family | task | ≥3 generator 的 family | 跨 split task/hash |", "|---|---:|---:|---:|---|---|"]
    for name, r in matrices.items():
        if "rows" not in r:
            lines.append(f"| {name} | missing | — | — | — | — |")
        else:
            lines.append(f"| {name} | {r['rows']} | {len(r['family_generators'])} | {r['tasks']} | {', '.join(r['families_with_at_least_3_generators']) or 'none'} | {r['task_cross_split_count']}/{r['code_cross_split_count']} |")
    lines += ["", f"Pair fold：{len(pair_audit['folds'])} 折中 {pair_audit['admitted_count']} 折具备 train/dev/test 双标签支持。",
              "这些折只测 same-family pair verification，不能写成 K 类归因。", "",
              "所有现有数据都不能仅凭 model name 推断共同 base / post-training 因果控制。重复 hash 和缺失字段见 JSON。", "",
              "## 重跑", "", "`python scripts/audit_acl_local_readiness.py`"]
    (args.out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "admitted_pair_folds": pair_audit["admitted_count"]}), flush=True)
    if pair_audit["invalid_pair_count"] or any(f["invalid_rows"] for f in pair_audit["folds"]):
        raise SystemExit("Invalid existing pair/fold metadata: inspect output before training")


if __name__ == "__main__":
    main()
