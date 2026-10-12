from pathlib import Path
import gzip
import hashlib
import json

import pyarrow.parquet as pq


ROOT = Path("d-det/data/mainline_abc_200k_2026-10-12")


def convert_one(task: str, split: str) -> None:
    src = ROOT / task / f"{split}.parquet"
    dst = ROOT / task / f"{split}.jsonl.gz"
    count = 0
    with gzip.open(dst, "wt", encoding="utf-8", newline="\n") as out:
        for batch in pq.ParquetFile(src).iter_batches(
            columns=["code", "label", "code_sha256"], batch_size=4096
        ):
            cols = batch.to_pydict()
            for code, label, code_sha in zip(
                cols["code"], cols["label"], cols["code_sha256"]
            ):
                out.write(
                    json.dumps(
                        {"code": code, "label": int(label), "code_sha256": code_sha},
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                count += 1
    print(f"{task}/{split}: {count} rows, {dst.stat().st_size} bytes")


def main() -> None:
    for task in "ABC":
        for split in ("train", "validation"):
            convert_one(task, split)

    manifest_path = ROOT / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = {}
    for path in sorted(ROOT.rglob("*")):
        if path.is_file() and path.name not in {"manifest.json", "build_progress.json"}:
            rel = path.relative_to(ROOT).as_posix()
            files[rel] = {
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "bytes": path.stat().st_size,
            }
    manifest["files"] = files
    manifest["standard_library_bundle"] = {
        "format": "jsonl.gz",
        "reader": "gzip + json; no pyarrow required",
        "files": [f"{task}/{split}.jsonl.gz" for task in "ABC" for split in ("train", "validation")],
    }
    manifest["manifest_revision"] = "jsonl_gzip_fallback_2026-10-12"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print("manifest_sha256", hashlib.sha256(manifest_path.read_bytes()).hexdigest())
    print("file_count", len(files))


if __name__ == "__main__":
    main()
