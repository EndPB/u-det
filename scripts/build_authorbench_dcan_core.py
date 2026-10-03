"""Build a larger same-prompt, cross-family alignment core from AuthorBench."""

from __future__ import annotations

import hashlib
import json
import random
import zipfile
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path("d-det/data/h2_authorbench_dcan")
ZIP = Path("d-det/data/h2_authorbench/LLM-AuthorBench.json.zip")
FAMILY = {
    "claude-3.5-haiku": "claude",
    "gpt-4o": "openai", "gpt-4.1": "openai", "gpt-4o-mini": "openai",
    "gemini-2.5-flash-preview-05-20": "gemini",
    "qwen-2.5-72b-instruct": "qwen", "deepseek-chat": "deepseek",
    "llama-3.3-70b-instruct": "llama",
}


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP) as z:
        raw = json.loads(z.read("LLM-AuthorBench.json"))
    by_prompt_model = defaultdict(list)
    for idx, row in enumerate(raw):
        model = row.get("model_name")
        if model in FAMILY:
            by_prompt_model[(row["prompt"], model)].append((idx, row))

    by_prompt = defaultdict(list)
    for (prompt, model), candidates in by_prompt_model.items():
        # A deterministic representative avoids counting repeated API samples
        # as independent semantic examples.
        idx, row = min(candidates, key=lambda x: x[1].get("SHA256_checksum", ""))
        by_prompt[prompt].append((idx, row, len(candidates)))

    selected_prompts = sorted(
        prompt for prompt, rows in by_prompt.items()
        if len({FAMILY[row["model_name"]] for _, row, _ in rows}) >= 2
    )
    rng = random.Random(20261003)
    shuffled = selected_prompts[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train, n_dev = round(n * .70), round(n * .15)
    split = {p: ("train" if i < n_train else "dev" if i < n_train + n_dev else "test") for i, p in enumerate(shuffled)}

    rows_out = []
    for prompt in selected_prompts:
        task_id = "authorbench_" + hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:12]
        for idx, row, replicate_count in sorted(by_prompt[prompt], key=lambda x: x[1]["model_name"]):
            rows_out.append({
                "task_id": task_id, "prompt": prompt, "code": row["c_code"],
                "model_name": row["model_name"], "family": FAMILY[row["model_name"]],
                "language": "C", "source": "LLM-AuthorBench", "source_row_index": idx,
                "source_sha256": row.get("SHA256_checksum"), "replicate_count": replicate_count,
                "char_count": row.get("char_count"), "num_lines": row.get("num_lines"),
                "nloc": row.get("nloc"), "cyclomatic_complexity": row.get("CC"),
                "token_size": row.get("token_size"), "task_split": split[prompt],
            })

    with (ROOT / "core.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for row in rows_out:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    with (ROOT / "task_index.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for prompt in selected_prompts:
            rows = by_prompt[prompt]
            f.write(json.dumps({
                "task_id": "authorbench_" + hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:12],
                "prompt": prompt, "language": "C", "task_split": split[prompt],
                "num_models": len(rows), "families": sorted({FAMILY[x[1]["model_name"]] for x in rows}),
            }, ensure_ascii=False, separators=(",", ":")) + "\n")
    summary = {
        "dataset": "LLM-AuthorBench DCAN alignment core",
        "source_archive": str(ZIP),
        "source_archive_sha256": hashlib.sha256(ZIP.read_bytes()).hexdigest(),
        "selection": "one deterministic row per model per prompt; prompt must contain at least two distinct families",
        "rows": len(rows_out), "tasks": len(selected_prompts),
        "families": sorted({r["family"] for r in rows_out}),
        "models": sorted({r["model_name"] for r in rows_out}),
        "language": ["C"],
        "task_split_counts": dict(Counter(split.values())),
        "family_counts": dict(Counter(r["family"] for r in rows_out)),
        "model_counts": dict(Counter(r["model_name"] for r in rows_out)),
        "why_dcan_ready": [
            "Each selected task has outputs from at least two model families, enabling same-semantics cross-family positives.",
            "task_id is a hard grouping key for task-held-out evaluation.",
            "Repeated prompt-model samples are collapsed to one deterministic representative and replicate_count is retained.",
        ],
        "limitations": [
            "C-only; most families still have one generator.",
            "Prompt equivalence is a semantic proxy, not a formal proof that outputs solve exactly the same behavior.",
            "Use STACAD and Droid for multi-language and generator-rich external validation.",
        ],
    }
    (ROOT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"rows": len(rows_out), "tasks": len(selected_prompts), "out": str(ROOT.resolve())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
