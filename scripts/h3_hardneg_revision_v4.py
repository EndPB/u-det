#!/usr/bin/env python3
"""H3 v4: quota-constrained minimum-cost hard-negative view.

This is a CPU-only continuation of v3.  It solves a small bipartite
assignment per split: every task contributes exactly three distinct AI
generators, every generator receives its exact global quota, and the chosen
edges minimize length plus masked-token distance.  No project provenance is
inferred and test is never opened.
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import LinearConstraint, Bounds, milp
from scipy.sparse import lil_matrix

ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / "d-det/data/h3_stacad_revision_v1"
V3_SCRIPT = ROOT / "scripts/h3_hardneg_revision_v3.py"
OUT = ROOT / "d-det/artifacts/h3_hardneg_revision_v4_2026-10-11"
DERIVED = ROOT / "d-det/data/h3_stacad_revision_v4"
SEED = 20261011

# Import the tested v3 helpers without duplicating token masking and probes.
sys.path.insert(0, str(V3_SCRIPT.parent))
from h3_hardneg_revision_v3 import (  # noqa: E402
    auc,
    cosine_candidates,
    dump,
    length_probe,
    lexical_probe,
    load,
)


def select_min_cost(rows: list[dict], max_ai: int = 3) -> tuple[list[dict], dict]:
    by_task: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_task[row["task_id"]].append(row)
    task_ids = sorted(by_task, key=lambda task: hashlib.sha256(task.encode()).hexdigest())
    groups = []
    generators = sorted({str(row.get("generator")) for row in rows if row["label"] == 1})
    for task_id in task_ids:
        group = by_task[task_id]
        humans = [r for r in group if r["label"] == 0]
        ais = [r for r in group if r["label"] == 1]
        if len(humans) != 1 or len(ais) != len(generators):
            raise ValueError(f"task support is not exactly one human plus all generators: {task_id}")
        ranked = cosine_candidates(humans[0], ais)
        by_gen = {str(row.get("generator")): (float(score), row) for score, row in ranked}
        groups.append((task_id, humans[0], by_gen))
    total_slots = len(groups) * max_ai
    quotas = {gen: total_slots // len(generators) + int(i < total_slots % len(generators)) for i, gen in enumerate(generators)}
    edges = [(task_index, gen, score, row) for task_index, (_, _, by_gen) in enumerate(groups) for gen, (score, row) in by_gen.items()]
    n = len(edges)
    c = np.asarray([score for _, _, score, _ in edges], dtype=float)
    task_index = {task_id: i for i, (task_id, _, _) in enumerate(groups)}
    gen_index = {gen: i for i, gen in enumerate(generators)}
    matrix = lil_matrix((len(groups) + len(generators), n), dtype=float)
    for j, (ti, gen, _, _) in enumerate(edges):
        matrix[ti, j] = 1.0
        matrix[len(groups) + gen_index[gen], j] = 1.0
    lower = np.asarray([max_ai] * len(groups) + [quotas[g] for g in generators], dtype=float)
    upper = lower.copy()
    result = milp(c=c, integrality=np.ones(n), bounds=Bounds(0, 1), constraints=LinearConstraint(matrix.tocsr(), lower, upper), options={"time_limit": 180, "presolve": True})
    if not result.success:
        raise RuntimeError(f"MILP failed: {result.message}")
    chosen: dict[str, list[tuple[float, dict]]] = defaultdict(list)
    used = Counter()
    for value, (ti, gen, score, row) in zip(result.x, edges):
        if value > 0.5:
            chosen[groups[ti][0]].append((score, row))
            used[gen] += 1
    selected = []
    manifests = {}
    for task_id, human, _ in groups:
        picks = sorted(chosen[task_id], key=lambda item: (item[0], item[1]["row_id"]))
        for role, score, row in [("human", 0.0, human)] + [("ai", score, row) for score, row in picks]:
            copy = dict(row)
            copy["eval_view"] = "hardneg_v4"
            copy["hardneg_score"] = float(score)
            copy["selection_method"] = "quota_constrained_min_cost_milp"
            copy["project_provenance_status"] = "unavailable_in_stacad_alignment"
            selected.append(copy)
        manifests[task_id] = {"human": human["row_id"], "ai": [row["row_id"] for _, row in picks], "generators": [row.get("generator") for _, row in picks], "scores": [float(score) for score, _ in picks]}
    return selected, {"tasks": manifests, "generator_quotas": quotas, "generator_counts": dict(used), "objective": "0.70*log_length_delta + 0.30*(1-masked_token_cosine)"}


def write_sums(directory: Path) -> None:
    lines = []
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS.txt":
            lines.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(directory).as_posix()}")
    (directory / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    DERIVED.mkdir(parents=True, exist_ok=True)
    train = [r for r in load(V1 / "train_all.jsonl") if r.get("task_split") == "train"]
    dev = [r for r in load(V1 / "dev_clean.jsonl") if r.get("task_split") == "dev"]
    train_rows, train_manifest = select_min_cost(train)
    dev_rows, dev_manifest = select_min_cost(dev)
    for path, data in ((DERIVED / "hardneg_train_v4.jsonl", train_rows), (DERIVED / "hardneg_dev_v4.jsonl", dev_rows)):
        path.write_text("".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in data), encoding="utf-8")
    dump(OUT / "train_manifest.json", {**train_manifest, "rows": len(train_rows)})
    dump(OUT / "dev_manifest.json", {**dev_manifest, "rows": len(dev_rows)})
    metrics = {
        "clean_dev_reference": {"raw": lexical_probe(train, dev, "raw"), "masked": lexical_probe(train, dev, "masked_tokens"), "length": length_probe(train, dev)},
        "hardneg_v4": {"raw": lexical_probe(train_rows, dev_rows, "raw"), "masked": lexical_probe(train_rows, dev_rows, "masked_tokens"), "length": length_probe(train_rows, dev_rows)},
        "counts": {"train_rows": len(train_rows), "train_tasks": len(train_manifest["tasks"]), "dev_rows": len(dev_rows), "dev_tasks": len(dev_manifest["tasks"])},
        "generator_counts": {"train": dict(Counter(r.get("generator") for r in train_rows)), "dev": dict(Counter(r.get("generator") for r in dev_rows))},
    }
    dump(OUT / "metrics.json", metrics)
    dump(OUT / "provenance_audit.json", {"status": "unavailable", "required_fields": ["repository_id", "project_id", "source_url"], "evidence": "all STACAD corpus rows inspected; only source_ref=file/line is available in alignment"})
    dump(OUT / "config.json", {"seed": SEED, "selection": "exact task sum=3 and exact global generator quotas, minimum-cost MILP", "objective": "0.70*log_length_delta + 0.30*(1-masked_token_cosine)", "test_deserialized": False, "gpu_run": False, "project_provenance": "unavailable"})
    (OUT / "hypothesis.md").write_text("""# H3 v4 quota-constrained hard-negative hypothesis\n\nA minimum-cost assignment can reduce length and masked-shape differences without allowing generator identity to select the evaluation rows.  Each task contributes one human and three distinct generators, with exact global generator quotas.  STACAD has no repository/project identity in the supplied alignment; this view therefore supports task-heldout claims only.\n\nThe untouched clean-dev reference remains reported.  This CPU pass does not authorize GPU training and never opens test.\n""", encoding="utf-8")
    m = metrics["hardneg_v4"]
    c = metrics["clean_dev_reference"]
    (OUT / "report.md").write_text(f"""# H3 v4 quota-constrained hard-negative CPU revision\n\n`project_provenance=unavailable`; this is a task-heldout diagnostic view.\n\n| view | raw task-macro | masked task-macro | length task-macro |\n|---|---:|---:|---:|\n| untouched clean-dev | {c['raw']['task_macro_auroc']:.4f} | {c['masked']['task_macro_auroc']:.4f} | {c['length']['task_macro_auroc']:.4f} |\n| hard-negative v4 | {m['raw']['task_macro_auroc']:.4f} | {m['masked']['task_macro_auroc']:.4f} | {m['length']['task_macro_auroc']:.4f} |\n\nThe assignment is exact-quota and CPU-only.  A formal GPU manifest remains forbidden unless the project provenance and all pre-registered gates are satisfied.\n""", encoding="utf-8")
    (OUT / "commands.txt").write_text("python scripts/h3_hardneg_revision_v4.py\n", encoding="utf-8")
    (OUT / "env.json").write_text(json.dumps({"python": sys.version, "platform": platform.platform(), "numpy": np.__version__}, indent=2) + "\n", encoding="utf-8")
    (OUT / "git_head.txt").write_text(subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip() + "\n", encoding="utf-8")
    write_sums(OUT)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
