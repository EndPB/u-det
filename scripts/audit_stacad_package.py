#!/usr/bin/env python
"""Stream-only audit for the STACAD Zenodo replication zip.

The archive is inspected without extracting the 3.8 GB unpacked package and
without loading JSONL records or source code into memory.  The output is a
small index and summary for deciding which parts belong on AutoDL.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path


LABELS = {
    1: "gemini_3_flash",
    2: "gpt_5_nano",
    3: "claude_3_haiku",
    4: "qwen3_coder_30b",
    5: "deepseek_v3",
    6: "grok_4_fast",
    7: "devstral",
}


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("zip_path", type=Path)
    ap.add_argument("--out", type=Path, default=Path("stacad_audit"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    files = []
    with zipfile.ZipFile(args.zip_path) as z:
        for info in z.infolist():
            n = info.filename.replace("\\", "/")
            if re.search(r"STACAD/data/task[12]/[^/]+\.jsonl$", n):
                files.append(info)
        if not files:
            raise RuntimeError("No STACAD/data/task1 or task2 JSONL members found")

        summary = {
            "archive": str(args.zip_path),
            "archive_bytes": args.zip_path.stat().st_size,
            "archive_md5": md5_file(args.zip_path),
            "members": [],
            "tasks": {},
            "label_map": LABELS,
            "notes": [
                "Streamed from zip; no extraction and no full source-code load.",
                "STACAD labels are concrete models, not family/generator pairs.",
            ],
        }
        index_path = args.out / "records_index.jsonl"
        with index_path.open("w", encoding="utf-8") as idx:
            for info in files:
                task = re.search(r"/data/(task1|task2)/", info.filename).group(1)
                counts = Counter()
                languages = Counter()
                generators = Counter()
                missing = Counter()
                rows = 0
                seen_file = set()
                duplicate_file = 0
                with z.open(info) as raw:
                    for line_no, line in enumerate(raw, 1):
                        if not line.strip():
                            continue
                        row = json.loads(line)
                        rows += 1
                        lang = str(row.get("lang", ""))
                        fname = str(row.get("file_name", ""))
                        gen = str(row.get("paraphrased_by", ""))
                        label = row.get("label", "")
                        if fname in seen_file:
                            duplicate_file += 1
                        seen_file.add(fname)
                        languages[lang] += 1
                        generators[gen] += 1
                        counts[str(label)] += 1
                        for k in ("lang", "file_name", "human_src", "llm_src", "paraphrased_by", "label"):
                            if k not in row or row[k] in (None, ""):
                                missing[k] += 1
                        idx.write(json.dumps({
                            "task": task, "member": info.filename, "line": line_no,
                            "lang": lang, "file_name": fname, "paraphrased_by": gen,
                            "label": label,
                        }, ensure_ascii=False) + "\n")
                summary["members"].append({
                    "name": info.filename, "compressed_bytes": info.compress_size,
                    "uncompressed_bytes": info.file_size, "rows": rows,
                    "duplicate_file_name": duplicate_file,
                    "labels": dict(counts), "languages": dict(languages),
                    "generators": dict(generators), "missing": dict(missing),
                })
                task_sum = summary["tasks"].setdefault(task, {"rows": 0, "labels": Counter(), "languages": Counter(), "generators": Counter()})
                task_sum["rows"] += rows
                for k, target in (("labels", counts), ("languages", languages), ("generators", generators)):
                    for key, value in target.items():
                        task_sum[k][key] += value
        for task_sum in summary["tasks"].values():
            for k in ("labels", "languages", "generators"):
                task_sum[k] = dict(task_sum[k])
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"members": len(files), "tasks": sorted(summary["tasks"]), "index": str(index_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
