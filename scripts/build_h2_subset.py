"""Build a small H2-ready subset from DroidCollection via hf-mirror.

The source shards are downloaded one at a time, read by row group, and
deleted after processing. No model, tokenizer, GPU, or full in-memory table
is used. The output is JSONL plus compact metadata/count reports.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq
import requests


BASE = "https://hf-mirror.com/datasets/project-droid/DroidCollection/resolve/9a42843be99456a3af70d2b9c9a53a4d7624598f/data"
SHARDS = {
    "train": ["train-00000-of-00003.parquet", "train-00001-of-00003.parquet", "train-00002-of-00003.parquet"],
    "dev": ["dev-00000-of-00001.parquet"],
    "test": ["test-00000-of-00001.parquet"],
}


def get_json(url: str):
    r = requests.get(url, timeout=(20, 60))
    r.raise_for_status()
    return r.json()


def download(url: str, dst: Path) -> None:
    tmp = dst.with_suffix(dst.suffix + ".part")
    if tmp.exists():
        tmp.unlink()
    with requests.get(url, stream=True, timeout=(20, 120)) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", "0"))
        got = 0
        t0 = time.time()
        with tmp.open("wb") as f:
            for block in r.iter_content(4 * 1024 * 1024):
                if block:
                    f.write(block)
                    got += len(block)
                    if total and got % (32 * 1024 * 1024) < len(block):
                        print(f"  downloaded {got/1e6:.0f}/{total/1e6:.0f} MB", flush=True)
        if total and got != total:
            raise IOError(f"short download: {got} != {total}")
    # Windows Defender/indexers can briefly hold the completed temporary file.
    # Retry the atomic rename instead of keeping the whole shard in memory.
    last_error = None
    for attempt in range(12):
        try:
            tmp.replace(dst)
            break
        except PermissionError as exc:
            last_error = exc
            time.sleep(1.0)
    else:
        raise last_error
    print(f"  done {dst.name}: {got/1e6:.1f} MB in {time.time()-t0:.1f}s", flush=True)


def key_for(row: dict) -> str:
    raw = "|".join(str(row.get(k, "")) for k in ("Code", "Generator", "Source", "Language"))
    return hashlib.sha1(raw.encode("utf-8", "ignore")).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/h2_droid_subset")
    ap.add_argument("--per-generator", type=int, default=240)
    ap.add_argument("--min-generators", type=int, default=2)
    ap.add_argument("--max-families", type=int, default=8)
    ap.add_argument("--max-rows", type=int, default=24000)
    args = ap.parse_args()

    out = Path(args.out)
    raw = out / "_raw"
    out.mkdir(parents=True, exist_ok=True)
    raw.mkdir(parents=True, exist_ok=True)
    core_path = out / "core.jsonl"
    diag_path = out / "diagnostic_hybrid_adversarial.jsonl"

    # First pass: count only metadata, still streaming row groups.
    counts = Counter()
    fam_gen = defaultdict(Counter)
    gen_label = defaultdict(Counter)
    fam_lang = defaultdict(Counter)
    seen = 0
    cols = ["Code", "Generator", "Generation_Mode", "Source", "Language", "Sampling_Params", "Rewriting_Params", "Label", "Model_Family"]
    for split, names in SHARDS.items():
        for name in names:
            dst = raw / name
            print(f"[metadata] {split}/{name}", flush=True)
            download(f"{BASE}/{name}", dst)
            pf = pq.ParquetFile(dst)
            for batch in pf.iter_batches(batch_size=8192, columns=cols):
                data = batch.to_pydict()
                n = len(data["Code"])
                for i in range(n):
                    label = data["Label"][i]
                    fam = data["Model_Family"][i] or "human"
                    gen = data["Generator"][i] or "human"
                    counts[(split, label)] += 1
                    gen_label[gen][label] += 1
                    if label == "MACHINE_GENERATED":
                        fam_gen[fam][gen] += 1
                        fam_lang[fam][data["Language"][i]] += 1
                    seen += 1
            pf.close()
            dst.unlink()

    # Select families with at least min-generators and substantial machine data.
    eligible = []
    for fam, gens in fam_gen.items():
        usable = {g: n for g, n in gens.items() if n >= args.per_generator}
        if len(usable) >= args.min_generators:
            eligible.append((sum(usable.values()), fam, usable))
    eligible.sort(reverse=True)
    selected = eligible[: args.max_families]
    if not selected:
        raise RuntimeError("No family has enough distinct generators; inspect counts.json")
    selected_families = {fam for _, fam, _ in selected}
    selected_generators = {fam: set(gens) for _, fam, gens in selected}
    print("selected families:", [(fam, sorted(gens)) for _, fam, gens in selected], flush=True)

    # Second pass: download again and reservoir-like deterministic quota by
    # split/family/generator/label. The source is streamed and deleted again.
    quotas = Counter()
    rows = {"train": [], "dev": [], "test": []}
    diagnostics = []
    diagnostic_quotas = Counter()
    diagnostic_limit = 1000  # keep refined/adversarial diagnostics balanced
    seen_keys = set()
    for split, names in SHARDS.items():
        for name in names:
            dst = raw / name
            print(f"[select] {split}/{name}", flush=True)
            download(f"{BASE}/{name}", dst)
            pf = pq.ParquetFile(dst)
            for batch in pf.iter_batches(batch_size=4096, columns=cols):
                data = batch.to_pydict()
                n = len(data["Code"])
                for i in range(n):
                    label = data["Label"][i]
                    fam = data["Model_Family"][i] or "human"
                    gen = data["Generator"][i] or "human"
                    if label in {"MACHINE_REFINED", "MACHINE_GENERATED_ADVERSARIAL"}:
                        if diagnostic_quotas[label] < diagnostic_limit:
                            diagnostics.append({k: data[k][i] for k in cols})
                            diagnostic_quotas[label] += 1
                        continue
                    if label == "MACHINE_GENERATED":
                        if fam not in selected_families or gen not in selected_generators[fam]:
                            continue
                    elif label == "HUMAN_GENERATED":
                        # Human controls are balanced per selected family only
                        # by source/language; they carry family=human.
                        pass
                    else:
                        continue
                    # Keep bounded human controls and per-generator machine rows.
                    quota_key = (split, label, fam if label == "MACHINE_GENERATED" else "human", gen)
                    limit = args.per_generator if label == "MACHINE_GENERATED" else args.per_generator * max(1, len(selected_families))
                    if quotas[quota_key] >= limit:
                        continue
                    row = {k: data[k][i] for k in cols}
                    row.update({"split_source": split, "source_row_sha1": key_for(row)})
                    if row["source_row_sha1"] in seen_keys:
                        continue
                    seen_keys.add(row["source_row_sha1"])
                    rows[split].append(row)
                    quotas[quota_key] += 1
                    if sum(len(v) for v in rows.values()) >= args.max_rows:
                        break
                if sum(len(v) for v in rows.values()) >= args.max_rows:
                    break
            pf.close()
            dst.unlink()
            if sum(len(v) for v in rows.values()) >= args.max_rows:
                break
        if sum(len(v) for v in rows.values()) >= args.max_rows:
            break

    for path, payload in [(core_path, [r for s in ("train", "dev", "test") for r in rows[s]]), (diag_path, diagnostics)]:
        with path.open("w", encoding="utf-8") as f:
            for row in payload:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "source": "project-droid/DroidCollection",
        "source_revision": "9a42843be99456a3af70d2b9c9a53a4d7624598f",
        "mirror": "https://hf-mirror.com",
        "selected_families": {fam: sorted(gens) for _, fam, gens in selected},
        "counts_all_source": {f"{a}|{b}": n for (a, b), n in counts.items()},
        "counts_selected_output": {s: len(rows[s]) for s in rows},
        "diagnostic_rows": len(diagnostics),
        "diagnostic_counts_by_label": dict(diagnostic_quotas),
        "columns": cols + ["split_source", "source_row_sha1"],
        "notes": [
            "Streaming row-group selection; raw shards deleted after each pass.",
            "DroidCollection has no public task_id/prompt_id; this subset is H2 candidate, not proof of same-task pairing.",
            "Primary core keeps HUMAN_GENERATED and MACHINE_GENERATED; refined/adversarial rows are diagnostic only.",
            "Rebuild generator-held-out folds on the server after inspecting per-generator counts.",
        ],
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.rmtree(raw, ignore_errors=True)
    print(json.dumps({"out": str(out), "rows": {k: len(v) for k, v in rows.items()}, "diagnostic": len(diagnostics)}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
