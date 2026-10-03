"""Build a streaming, code-free metadata index over the local sources."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path


OUT = Path("d-det/data/acl_attribution_collection_v1")
INDEX = OUT / "index.jsonl"


def write_entry(f, entry):
    f.write(json.dumps(entry, ensure_ascii=False, separators=(",", ":")) + "\n")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    stats = Counter()
    with INDEX.open("w", encoding="utf-8") as out:
        # AICD is intentionally indexed at shard level: the public parquet
        # does not expose stable row IDs or provenance beyond numeric labels.
        audit = json.loads((OUT / "raw/aicd_audit.json").read_text(encoding="utf-8"))
        for item in audit["files"]:
            rel = item["path"]
            cfg = rel.split("/", 1)[0]
            split = Path(rel).name.split("-", 1)[0]
            write_entry(out, {
                "record_id": f"aicd:{rel}", "dataset_id": "aicd_bench",
                "granularity": "shard", "config": cfg, "split": split,
                "row_count": item["rows"], "data_ref": f"raw/aicd/{rel}",
                "label_space": "aicd_numeric_pending", "metadata_complete": False,
            })
            stats["aicd_shards"] += 1

        codet_audit = json.loads(Path("d-det/data/codet_m4/audit.json").read_text(encoding="utf-8"))
        write_entry(out, {
            "record_id": "codet_m4:dataset_without_comments.parquet",
            "dataset_id": "codet_m4", "granularity": "shard",
            "row_count": codet_audit["rows"], "data_ref": "../codet_m4/dataset_without_comments.parquet",
            "label_space": "codet_m4_model_target", "metadata_complete": True,
        })
        stats["codet_m4_shards"] += 1

        # Droid: retain explicit metadata and source row hash, never duplicate Code.
        droid_path = Path("d-det/data/h2_droid_full_selected/core.jsonl")
        with droid_path.open(encoding="utf-8") as src:
            for line_no, line in enumerate(src):
                row = json.loads(line)
                rid = row.get("source_row_sha1") or hashlib.sha1(line.encode()).hexdigest()
                write_entry(out, {
                    "record_id": f"droid:{rid}", "dataset_id": "droid_collection_v2_selected",
                    "granularity": "record", "split": row.get("split_source"),
                    "generator": row.get("Generator"), "family": row.get("Model_Family"),
                    "generation_mode": row.get("Generation_Mode"), "label": row.get("Label"),
                    "language": row.get("Language"), "source": row.get("Source"),
                    "data_ref": f"../h2_droid_full_selected/core.jsonl#L{line_no+1}",
                    "source_row_sha1": row.get("source_row_sha1"), "metadata_complete": True,
                })
                stats["droid_records"] += 1

        # AuthorBench: task_id is the grouping key for same-task evaluation.
        ab_path = Path("d-det/data/h2_authorbench/core.jsonl")
        with ab_path.open(encoding="utf-8") as src:
            for line_no, line in enumerate(src):
                row = json.loads(line)
                digest = row.get("source_sha256") or hashlib.sha256(line.encode()).hexdigest()
                write_entry(out, {
                    "record_id": f"authorbench:{digest}", "dataset_id": "llm_authorbench_task_subset",
                    "granularity": "record", "split": row.get("task_split"),
                    "task_id": row.get("task_id"), "generator": row.get("model_name"),
                    "family": row.get("family"), "language": row.get("language"),
                    "replicate_count": row.get("replicate_count"),
                    "data_ref": f"../h2_authorbench/core.jsonl#L{line_no+1}",
                    "source_sha256": row.get("source_sha256"), "metadata_complete": True,
                })
                stats["authorbench_records"] += 1

        # Expanded DCAN core: same source archive, but retain only prompts
        # represented by at least two distinct families.
        dcan_path = Path("d-det/data/h2_authorbench_dcan/core.jsonl")
        with dcan_path.open(encoding="utf-8") as src:
            for line_no, line in enumerate(src):
                row = json.loads(line)
                digest = row.get("source_sha256") or hashlib.sha256(line.encode()).hexdigest()
                write_entry(out, {
                    "record_id": f"authorbench_dcan:{digest}", "dataset_id": "llm_authorbench_dcan_core",
                    "granularity": "record", "split": row.get("task_split"),
                    "task_id": row.get("task_id"), "generator": row.get("model_name"),
                    "family": row.get("family"), "language": row.get("language"),
                    "replicate_count": row.get("replicate_count"),
                    "data_ref": f"../h2_authorbench_dcan/core.jsonl#L{line_no+1}",
                    "source_sha256": row.get("source_sha256"), "metadata_complete": True,
                })
                stats["authorbench_dcan_records"] += 1

        codegen_path = Path("d-det/data/h2_llm_codegen/core.jsonl")
        with codegen_path.open(encoding="utf-8") as src:
            for line_no, line in enumerate(src):
                row = json.loads(line)
                digest = row.get("source_sha256") or hashlib.sha256(line.encode()).hexdigest()
                write_entry(out, {
                    "record_id": f"llm_codegen:{digest}", "dataset_id": "llm_codegen_alignment_core",
                    "granularity": "record", "split": row.get("task_split"),
                    "task_id": row.get("task_id"), "generator": row.get("model_name"),
                    "family": row.get("family"), "language": row.get("language"),
                    "scenario": row.get("scenario"), "cwe_id": row.get("cwe_id"),
                    "data_ref": f"../h2_llm_codegen/core.jsonl#L{line_no+1}",
                    "source_sha256": row.get("source_sha256"), "metadata_complete": True,
                })
                stats["llm_codegen_records"] += 1

        # STACAD: one JSON row contains a human/LLM pair. Keep file-level group.
        stacad_root = Path("d-det/data/stacad_v2/extracted/STACAD-v2/data/corpus_v2")
        for path in sorted(stacad_root.glob("*.jsonl")):
            if path.name == "summary.json":
                continue
            lang = path.stem
            with path.open(encoding="utf-8") as src:
                for line_no, line in enumerate(src):
                    row = json.loads(line)
                    pair_key = f"{lang}:{row.get('file_name')}:{row.get('paraphrased_by')}:{line_no}"
                    digest = hashlib.sha1(pair_key.encode()).hexdigest()
                    write_entry(out, {
                        "record_id": f"stacad:{digest}", "dataset_id": "stacad_v2",
                        "granularity": "pair", "language": row.get("lang", lang),
                        "group_id": row.get("file_name"), "generator": row.get("paraphrased_by"),
                        "family": "unknown", "label": row.get("label"),
                        "split": row.get("split_v1"),
                        "data_ref": f"../stacad_v2/extracted/STACAD-v2/data/corpus_v2/{path.name}#L{line_no+1}",
                        "metadata_complete": True,
                    })
                    stats["stacad_pairs"] += 1

    (OUT / "index_stats.json").write_text(json.dumps({"records": sum(stats.values()), "by_source": dict(stats)}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"index": str(INDEX.resolve()), "records": sum(stats.values()), "by_source": dict(stats)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
