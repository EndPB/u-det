"""Build a small, provenance-first manifest for the local attribution collection.

This does not copy large code fields or merge incompatible label spaces.  It
records where each source lives, its official split/metadata guarantees and
the intended role in the family-attribution experiments.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("d-det/data/acl_attribution_collection_v1")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def size(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


def main() -> None:
    audit = json.loads((ROOT / "raw/aicd_audit.json").read_text(encoding="utf-8"))
    droid = json.loads(Path("d-det/data/h2_droid_full_selected/summary.json").read_text(encoding="utf-8"))
    authorbench = json.loads(Path("d-det/data/h2_authorbench/summary.json").read_text(encoding="utf-8"))
    stacad_summary = json.loads(Path("d-det/data/stacad_v2/extracted/STACAD-v2/data/corpus_v2/summary.json").read_text(encoding="utf-8"))
    stacad_zip = Path("d-det/data/stacad_v2/stacad-v2.zip")
    if not stacad_zip.exists():
        stacad_zip = Path("d-det/data/stacad_v2/stacad-v2.zip.part")

    sources = [
        {
            "id": "aicd_bench",
            "status": "local_raw_complete",
            "role": "primary_scale_family_attribution",
            "source_url": "https://huggingface.co/datasets/AICD-bench/AICD-Bench",
            "revision": json.loads((ROOT / "raw/aicd_api.json").read_text(encoding="utf-8")).get("sha", "b7dc6d82257de6e0146d1e7b0831abd7088633af"),
            "license": "CC BY-NC 4.0; research use and original terms apply",
            "configs": {
                name: {"rows": info["rows"], "splits": info["splits"]}
                for name, info in audit["configs"].items()
            },
            "raw_root": "raw/aicd",
            "bytes": sum(int(x["bytes"]) for x in audit["files"]),
            "schema": ["code", "label"],
            "metadata_warning": "Numeric labels are retained exactly; the public parquet does not carry family, generator or language columns. Do not use names until the official mapping/protocol is verified.",
            "paper_protocol": "Task 2 is 12-way family attribution (human plus 11 model families); official split sizes are preserved by config/split.",
        },
        {
            "id": "droid_collection_v2_selected",
            "status": "local_selected_complete",
            "role": "generator_rich_cross_generator_h2",
            "source_url": "https://huggingface.co/datasets/project-droid/DroidCollection",
            "revision": droid.get("source_revision"),
            "license": "Preserve the source dataset terms; local research package only",
            "raw_root": "../h2_droid_full_selected",
            "bytes": sum(size(p) for p in Path("d-det/data/h2_droid_full_selected").glob("*")),
            "rows": droid.get("counts_selected_output"),
            "families": droid.get("selected_families"),
            "metadata": droid.get("columns"),
            "metadata_warning": "No public task_id/prompt_id; use for generator/family robustness and diagnostic controls, not same-task causal claims.",
        },
        {
            "id": "stacad_v2",
            "status": "local_archive_and_data_complete",
            "role": "paired_task_repository_ood_and_paraphrase",
            "source_url": "https://zenodo.org/records/22809889",
            "revision": "zenodo:22809889",
            "license": "See the bundled STACAD-v2 LICENSE and README; retain original provenance",
            "raw_root": "../stacad_v2/extracted/STACAD-v2/data",
            "archive": "../stacad_v2/stacad-v2.zip",
            "archive_bytes": size(stacad_zip),
            "archive_sha256": sha256(stacad_zip) if stacad_zip.exists() else None,
            "rows": stacad_summary.get("kept_pairs"),
            "files": stacad_summary.get("kept_files"),
            "models": list(stacad_summary.get("per_model", {}).keys()),
            "languages": list(stacad_summary.get("per_language", {}).keys()),
            "fields": ["lang", "file_name", "human_src", "llm_src", "paraphrased_by", "label", "split_v1"],
            "split_guarantee": "Five deterministic file-level folds; never random-split pairs across the same file.",
        },
        {
            "id": "llm_authorbench_task_subset",
            "status": "local_selected_complete",
            "role": "same_task_cross_family_control",
            "source_url": "https://github.com/LLMauthorbench/LLMauthorbench",
            "revision": "local-pinned-summary",
            "license": "See the upstream repository and archive terms",
            "raw_root": "../h2_authorbench",
            "bytes": sum(size(p) for p in Path("d-det/data/h2_authorbench").glob("*")),
            "rows": authorbench.get("rows"),
            "tasks": authorbench.get("tasks"),
            "families": authorbench.get("families"),
            "models": authorbench.get("models"),
            "languages": authorbench.get("language"),
            "split_guarantee": "Primary split is task-level; generator holdouts are only a within-OpenAI diagnostic because other families have one generator.",
        },
        {
            "id": "llm_authorbench_dcan_core",
            "status": "local_selected_complete",
            "role": "same_semantics_cross_family_alignment",
            "source_url": "https://github.com/LLMauthorbench/LLMauthorbench",
            "revision": "same-local-pinned-archive",
            "license": "See the upstream repository and archive terms",
            "raw_root": "../h2_authorbench_dcan",
            "bytes": sum(size(p) for p in Path("d-det/data/h2_authorbench_dcan").glob("*")),
            "rows": 9498,
            "tasks": 2715,
            "families": ["claude", "deepseek", "gemini", "llama", "openai", "qwen"],
            "selection": "one deterministic sample per model per prompt; prompt must contain at least two distinct families",
            "split_guarantee": "Prompt/task groups are kept intact across train/dev/test.",
            "limitation": "C-only and prompt equivalence is a semantic proxy; external validation remains required.",
        },
        {
            "id": "codet_m4",
            "status": "local_raw_complete",
            "role": "external_structural_source_fingerprint_control",
            "source_url": "https://huggingface.co/datasets/DaniilOr/CoDET-M4",
            "revision": "4d4e665037cb797cb5381c0b95c1d33e30420b8e",
            "license": "See upstream dataset card",
            "raw_root": "../codet_m4",
            "bytes": size(Path("d-det/data/codet_m4/dataset_without_comments.parquet")),
            "rows": 500552,
            "languages": ["python", "java", "cpp"],
            "models": ["codellama", "gpt", "llama3.1", "nxcode", "qwen1.5", "human", "unknown/None"],
            "metadata": ["code", "language", "model", "split", "target", "source", "features", "cleaned_code"],
            "limitation": "No reliable same-task cross-generator pairing; use as structural/source-fingerprint control, not semantic alignment evidence.",
        },
        {
            "id": "llm_codegen_alignment_core",
            "status": "local_selected_complete",
            "role": "same_cwe_cross_model_alignment",
            "source_url": "https://huggingface.co/datasets/codesbyusman/LLM-CodeGen",
            "revision": "8a0af8344ca67da4370076815a1d161145c3204c",
            "license": "CC BY 4.0 per upstream card",
            "raw_root": "../llm_codegen_raw",
            "processed_root": "../h2_llm_codegen",
            "bytes": sum(size(p) for p in Path("d-det/data/h2_llm_codegen").glob("*")),
            "rows": 1515,
            "tasks": 168,
            "models": 9,
            "families": ["google", "ibm", "meta", "microsoft", "mistral"],
            "scenarios": ["simple", "secure"],
            "limitation": "C-only; CWE/security prompt semantics can create lexical shortcuts, so use as supplementary alignment evidence.",
        },
    ]

    external = [
        {
            "id": "codet_m4",
            "status": "external_pointer_only",
            "role": "binary_detection_and_hybrid_control",
            "url": "https://huggingface.co/datasets/DaniilOr/CoDET-M4",
            "reason": "Useful for external detection checks, but it is not a same-task multi-generator family benchmark.",
        },
        {
            "id": "codemirage",
            "status": "external_pointer_only",
            "role": "unseen_generator_and_language_detection_control",
            "url": "https://huggingface.co/datasets/HanxiGuo/CodeMirage",
            "reason": "Binary human/AI benchmark; preserve its non-redistribution terms and do not merge its labels into family attribution.",
        },
        {
            "id": "dcan_code_fingerprints",
            "status": "method_reference_only",
            "role": "method_reference",
            "url": "https://github.com/mtt500/DCAN",
            "reason": "Disentangled source-agnostic/source-specific representation is a candidate method family, not an additional local corpus.",
        },
        {
            "id": "multiaigcd",
            "status": "candidate_not_downloaded",
            "role": "future_same_problem_multilingual_scenario_control",
            "url": "https://github.com/BasakDemirok/MultiAIGCD",
            "reason": "The paper reports 800 problems, 6 LLMs, 3 languages and 3 usage scenarios, but the official repository currently says the data archive is not yet published; do not regenerate it locally without an explicit decision.",
        },
    ]

    manifest = {
        "collection_id": "acl_attribution_collection_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "A provenance-first local research collection for SOTA family attribution from post-training differences.",
        "index": {
            "path": "index.jsonl",
            "stats": "index_stats.json",
            "records": 304622,
            "policy": "Metadata and stable source references only; code is not duplicated and label spaces are not merged."
        },
        "design": {
            "primary_benchmark": "AICD Task 2 raw official splits, numeric labels kept intact",
            "cross_generator_control": "Droid selected rows with explicit generator/family metadata",
            "same_task_control": "AuthorBench task-level split",
            "paired_repository_ood": "STACAD-v2 file-level folds and paraphrase pairs",
            "evaluation_rule": "Never concatenate label spaces; report per-source metrics and a preregistered cross-source transfer table.",
        },
        "sources": sources,
        "external_pointers": external,
        "known_gaps": [
            "AICD raw parquet lacks family/generator/language columns; recover the official label map before publishing class names.",
            "Droid lacks prompt/task IDs, so it cannot alone prove task-conditioned invariance.",
            "AuthorBench is C-only and has multi-generator coverage mainly within OpenAI.",
            "STACAD is paraphrase-paired and therefore measures a different but valuable repository/task shift.",
            "Licences differ; this package is for local/server research and should not be redistributed as a new combined dataset without checking every upstream licence.",
        ],
        "recommended_training_order": [
            "1. Reproduce AICD Task 2 baseline using numeric IDs and official splits.",
            "2. Train task-conditioned/multi-view attribution on AuthorBench and STACAD, using group-safe folds.",
            "3. Evaluate generator-held-out family transfer on Droid and report unknown-family rejection.",
            "4. Use CoDET-M4/CodeMirage only as external binary or robustness controls.",
        ],
    }
    out = ROOT / "collection_manifest.json"
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(out.resolve()), "sources": len(sources), "external_pointers": len(external)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
