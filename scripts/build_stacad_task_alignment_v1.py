"""Build a compact, file-split, same-task STACAD alignment package.

STACAD has one human file and paraphrases from up to seven generators.  The
package keeps complete seven-generator tasks for multilingual semantic control,
without pretending that different vendors are same-family positive pairs.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "d-det/data/stacad_v2/extracted/STACAD-v2/data/corpus_v2"
DEFAULT_OUT = ROOT / "d-det/data/h2_stacad_alignment_v1"
MODELS = (
    "claude_3_haiku", "deepseek_v3_2", "devstral_2512", "gemini_3_flash",
    "gpt5_nano", "grok_4_fast", "qwen3_coder_30b",
)
FAMILY = {
    "claude_3_haiku": "claude",
    "deepseek_v3_2": "deepseek",
    "devstral_2512": "mistral",
    "gemini_3_flash": "gemini",
    "gpt5_nano": "openai",
    "grok_4_fast": "grok",
    "qwen3_coder_30b": "qwen",
}
LANGS = ("c", "cpp", "cs", "go", "java", "php", "py")


def task_key(row: dict) -> str:
    return f"{row['lang']}:{row['file_name']}"


def canonical_split(raw: str) -> str:
    return "dev" if raw == "val" else raw


def score(seed: int, key: str) -> int:
    return int.from_bytes(hashlib.blake2b(f"{seed}|{key}".encode(), digest_size=8).digest(), "big")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-per-language", type=int, default=300)
    ap.add_argument("--dev-per-language", type=int, default=100)
    ap.add_argument("--test-per-language", type=int, default=60)
    ap.add_argument("--seed", type=int, default=20261004)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    paths = [SOURCE / f"{lang}.jsonl" for lang in LANGS]

    tasks = {}
    rows_seen = 0
    for path in paths:
        with path.open("r", encoding="utf-8") as src:
            for line_no, line in enumerate(src, 1):
                row = json.loads(line)
                rows_seen += 1
                key = task_key(row)
                item = tasks.setdefault(key, {
                    "task_key": key,
                    "lang": row["lang"],
                    "file_name": row["file_name"],
                    "source_split": canonical_split(row["split_v1"]),
                    "models": set(),
                    "rows": [],
                })
                item["models"].add(row["paraphrased_by"])
                item["rows"].append({"path": path, "line_no": line_no, "row": row})

    complete = {k: v for k, v in tasks.items() if set(MODELS).issubset(v["models"]) and v["source_split"] in {"train", "dev", "test"}}
    requested = {"train": args.train_per_language, "dev": args.dev_per_language, "test": args.test_per_language}
    selected_keys = set()
    quota = {}
    for split, n in requested.items():
        quota[split] = {}
        for lang in LANGS:
            keys = sorted((k for k, v in complete.items() if v["source_split"] == split and v["lang"] == lang), key=lambda k: score(args.seed, k))
            take = min(n, len(keys))
            quota[split][lang] = {"requested": n, "available": len(keys), "selected": take}
            selected_keys.update(keys[:take])

    args.out.mkdir(parents=True, exist_ok=True)
    core_path = args.out / "core.jsonl"
    task_path = args.out / "task_index.jsonl"
    pair_path = args.out / "pair_index.jsonl"
    selected_tasks = sorted(selected_keys)
    record_count = 0
    pair_count = Counter()
    with core_path.open("w", encoding="utf-8", newline="\n") as core, task_path.open("w", encoding="utf-8", newline="\n") as task_out, pair_path.open("w", encoding="utf-8", newline="\n") as pair_out:
        for key in selected_tasks:
            task = complete[key]
            rows_by_model = {x["row"]["paraphrased_by"]: x for x in task["rows"] if x["row"]["paraphrased_by"] in MODELS}
            # The source may contain at most one retained row per model/file.
            if len(rows_by_model) != len(MODELS):
                continue
            anchor = rows_by_model[MODELS[0]]["row"]
            human = anchor["human_src"]
            human_hash = hashlib.sha256(human.encode("utf-8")).hexdigest()
            task_out.write(json.dumps({
                "task_id": f"stacad:{key}", "task_key": key,
                "language": task["lang"], "file_name": task["file_name"],
                "task_split": task["source_split"], "models": list(MODELS),
                "families": [FAMILY[m] for m in MODELS],
                "human_sha256": human_hash, "human_code": human,
                "source_ref": f"{rows_by_model[MODELS[0]]['path'].relative_to(ROOT).as_posix()}#L{rows_by_model[MODELS[0]]['line_no']}",
            }, ensure_ascii=False, separators=(",", ":")) + "\n")
            for model in MODELS:
                entry = rows_by_model[model]
                row = entry["row"]
                code = row["llm_src"]
                rec = {
                    "record_id": f"stacad:{key}:{model}", "task_id": f"stacad:{key}", "task_key": key,
                    "task_split": task["source_split"], "language": task["lang"], "file_name": task["file_name"],
                    "generator": model, "family": FAMILY[model], "label": row.get("label"),
                    "human_sha256": human_hash, "code": code,
                    "code_sha256": hashlib.sha256(code.encode("utf-8")).hexdigest(),
                    "source_ref": f"{entry['path'].relative_to(ROOT).as_posix()}#L{entry['line_no']}",
                }
                core.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n")
                record_count += 1
            for left, right in itertools.combinations(MODELS, 2):
                pair_type = "same_task_cross_generator"
                pair = {
                    "pair_id": f"pair:{hashlib.sha1(f'{key}|{left}|{right}'.encode()).hexdigest()[:20]}",
                    "task_id": f"stacad:{key}", "task_key": key, "task_split": task["source_split"],
                    "pair_type": pair_type, "positive_for_family_invariance": False,
                    "left_generator": left, "right_generator": right,
                    "left_family": FAMILY[left], "right_family": FAMILY[right],
                }
                pair_out.write(json.dumps(pair, ensure_ascii=False, separators=(",", ":")) + "\n")
                pair_count[pair_type] += 1

    split_counts = Counter()
    lang_counts = Counter()
    for key in selected_tasks:
        if key in complete:
            split_counts[complete[key]["source_split"]] += 1
            lang_counts[(complete[key]["source_split"], complete[key]["lang"])] += 1
    summary = {
        "dataset_id": "h2_stacad_alignment_v1",
        "purpose": "Multilingual same-task multi-generator semantic control for family attribution",
        "source": "local STACAD-v2 corpus_v2",
        "source_rows_scanned": rows_seen,
        "complete_source_tasks": len(complete),
        "selected_tasks": len(selected_tasks), "selected_rows": record_count,
        "models": list(MODELS), "families": [FAMILY[m] for m in MODELS], "languages": list(LANGS),
        "split_counts": dict(split_counts),
        "language_split_counts": {f"{s}/{l}": n for (s, l), n in sorted(lang_counts.items())},
        "pair_counts": dict(pair_count),
        "quota": quota,
        "selection": "complete seven-model tasks only; deterministic hash ranking within source file-level split and language",
        "important_boundary": "all generators are different vendors; no same_family_cross_generator positives exist in this package",
        "limitations": [
            "Human file is the shared semantic anchor; model outputs are paraphrases, not formal equivalence proof.",
            "The seven generators represent distinct vendor families, so same-task pairs are cross-family controls, not H2 positive pairs.",
            "Use source split as file-level train/dev/test; never random-split rows from the same file.",
            "Do not merge these family labels with AuthorBench or LLM-CodeGen labels without a source-specific table.",
        ],
        "recommended_use": [
            "multilingual task-conditioned family attribution",
            "cross-generator semantic control and centered representations",
            "external validation of H2 mechanisms after positive-pair experiments on LLM-CodeGen/AuthorBench",
        ],
    }
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.out / "split_manifest.json").write_text(json.dumps({"dataset_id": summary["dataset_id"], "split_counts": dict(split_counts), "language_split_counts": summary["language_split_counts"], "file_group_key": "lang:file_name", "source_split": "STACAD split_v1 with val renamed dev"}, ensure_ascii=False, indent=2), encoding="utf-8")
    readme = """# h2_stacad_alignment_v1

这是从本地 STACAD-v2 构建的紧凑多语言同任务对齐集。每个任务保留同一 human 文件的 7 个 generator paraphrase，使用原始 file-level split。

- `core.jsonl`：每个模型一行，包含代码和 family/generator 元数据。
- `task_index.jsonl`：每个任务一行，包含 human anchor 和任务信息。
- `pair_index.jsonl`：同任务跨 generator 对照 pair。
- `summary.json` / `split_manifest.json`：计数与协议。

这组 7 个 generator 各自属于不同厂商，因此 pair 不是 H2 的同家族正对；它用于多语言语义控制、任务中心化和外部验证。H2 正对仍应来自 LLM-CodeGen/AuthorBench 的 `same_family_cross_generator`。
"""
    (args.out / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps({"out": str(args.out.resolve()), "source_rows": rows_seen, "complete_tasks": len(complete), "selected_tasks": len(selected_tasks), "rows": record_count, "pairs": sum(pair_count.values())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
