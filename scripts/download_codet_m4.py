"""Download the compact public CoDET-M4 parquet from hf-mirror."""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path

import requests


ROOT = Path("d-det/data/codet_m4")
DATASET = "DaniilOr/CoDET-M4"
REVISION = "4d4e665037cb797cb5381c0b95c1d33e30420b8e"
FILE = "dataset_without_comments.parquet"
URL = f"https://hf-mirror.com/datasets/{DATASET}/resolve/{REVISION}/{FILE}"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(ROOT).free < 20 * 1024**3:
        raise RuntimeError("Keep at least 20 GiB free")
    meta = requests.get(f"https://hf-mirror.com/api/datasets/{DATASET}", timeout=60).json()
    out = ROOT / FILE
    part = ROOT / (FILE + ".part")
    with requests.get(URL, stream=True, timeout=(30, 180), allow_redirects=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", "0"))
        got = 0
        with part.open("wb") as f:
            for block in r.iter_content(8 * 1024 * 1024):
                if block:
                    f.write(block)
                    got += len(block)
                    if got % (64 * 1024 * 1024) < len(block):
                        print(round(got / 1e6, 1), "MB", flush=True)
    if total and got != total:
        raise RuntimeError(f"download size {got} != {total}")
    digest = sha256(part)
    part.replace(out)
    (ROOT / "manifest.json").write_text(json.dumps({
        "dataset": DATASET, "revision": REVISION, "endpoint": "https://hf-mirror.com",
        "file": FILE, "bytes": got, "sha256": digest,
        "source_api": meta.get("sha"), "downloaded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "license": "See upstream dataset card; do not merge label spaces without provenance.",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"file": str(out.resolve()), "bytes": got, "sha256": digest}, ensure_ascii=False))


if __name__ == "__main__":
    main()
