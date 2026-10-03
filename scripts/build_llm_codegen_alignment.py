"""Build a same-CWE cross-model alignment core from LLM-CodeGen CSVs."""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


RAW = Path("d-det/data/llm_codegen_raw")
OUT = Path("d-det/data/h2_llm_codegen")


def parse_response(text: str) -> dict:
    try:
        return ast.literal_eval(text)
    except Exception:
        try:
            return json.loads(text)
        except Exception:
            return {}


def content_from_response(obj: dict) -> str:
    msg = obj.get("message")
    if isinstance(msg, dict):
        return str(msg.get("content", ""))
    return str(obj.get("content", ""))


def family(name: str) -> str:
    s = name.lower().replace("-", "")
    if "gemini" in s or "gemni" in s:
        return "google"
    if "llama" in s:
        return "meta"
    if "mistral" in s or "codestral" in s:
        return "mistral"
    if "codegemma" in s:
        return "google"
    if "codellama" in s:
        return "meta"
    if "granite" in s:
        return "ibm"
    if "phi" in s:
        return "microsoft"
    return name


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # The HF repo mirrors some CSVs at two paths. Deduplicate by file hash.
    seen_file_hash = set()
    records = []
    for path in sorted(RAW.glob("**/*.csv")):
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest in seen_file_hash:
            continue
        seen_file_hash.add(digest)
        model_dir = path.parent.name
        with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as f:
            for row in csv.DictReader(f):
                resp = parse_response(row.get("response", ""))
                model = str(resp.get("model") or model_dir)
                scenario = "secure" if path.name.startswith("secure_") else "simple"
                code = content_from_response(resp)
                records.append({
                    "task_id": f"{scenario}:{row.get('cweid')}",
                    "prompt": row.get("prompt", ""), "code": code,
                    "model_name": model, "family": family(model), "language": "C",
                    "scenario": scenario, "cwe_id": row.get("cweid"),
                    "source_file": path.relative_to(RAW).as_posix(),
                    "source_sha256": hashlib.sha256(code.encode("utf-8")).hexdigest(),
                })
    by_task = defaultdict(list)
    for row in records:
        by_task[row["task_id"]].append(row)
    selected_tasks = sorted(t for t, rows in by_task.items() if len({r["family"] for r in rows}) >= 2)
    rng = random.Random(20261003)
    shuffled = selected_tasks[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    split = {t: ("train" if i < round(.70*n) else "dev" if i < round(.85*n) else "test") for i, t in enumerate(shuffled)}
    selected = []
    for task in selected_tasks:
        for row in sorted(by_task[task], key=lambda r: r["model_name"]):
            row = dict(row); row["task_split"] = split[task]; selected.append(row)
    with (OUT / "core.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for row in selected:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    with (OUT / "task_index.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for task in selected_tasks:
            rows = by_task[task]
            f.write(json.dumps({"task_id": task, "prompt": rows[0]["prompt"], "scenario": rows[0]["scenario"], "cwe_id": rows[0]["cwe_id"], "task_split": split[task], "models": sorted({r["model_name"] for r in rows}), "families": sorted({r["family"] for r in rows})}, ensure_ascii=False, separators=(",", ":")) + "\n")
    summary = {
        "dataset": "LLM-CodeGen same-CWE alignment core", "source": "codesbyusman/LLM-CodeGen",
        "selection": "deduplicate mirrored CSVs; retain tasks with at least two model families",
        "unique_csv_files": len(seen_file_hash), "rows": len(selected), "tasks": len(selected_tasks),
        "models": sorted({r["model_name"] for r in selected}), "families": sorted({r["family"] for r in selected}),
        "scenarios": sorted({r["scenario"] for r in selected}), "task_split_counts": dict(Counter(split.values())),
        "family_counts": dict(Counter(r["family"] for r in selected)),
        "why_dcan_ready": ["Same CWE and scenario gives a concrete semantic proxy across models.", "Two prompt regimes provide a simple domain-shift control.", "Task-level splits prevent identical CWE/scenario groups crossing train and test."],
        "limitations": ["C-only and security prompts may induce lexical shortcuts.", "Model names/families are inferred from response metadata and directory names; verify before publication.", "This is a supplementary alignment core, not a replacement for STACAD or AuthorBench."],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"rows": len(selected), "tasks": len(selected_tasks), "models": summary["models"], "out": str(OUT.resolve())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
