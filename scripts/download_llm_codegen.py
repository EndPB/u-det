"""Download the small metadata CSV portion of the public LLM-CodeGen set."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import requests


DATASET = "codesbyusman/LLM-CodeGen"
REVISION = "8a0af8344ca67da4370076815a1d161145c3204c"
ROOT = Path("d-det/data/llm_codegen_raw")


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    api = requests.get(f"https://hf-mirror.com/api/datasets/{DATASET}", timeout=60).json()
    files = [x["rfilename"] for x in api.get("siblings", []) if x.get("rfilename", "").endswith(".csv")]
    manifest = {"dataset": DATASET, "revision": REVISION, "endpoint": "https://hf-mirror.com", "files": {}}
    for rel in files:
        dst = ROOT / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://hf-mirror.com/datasets/{DATASET}/resolve/{REVISION}/{rel}"
        r = requests.get(url, timeout=(30, 120))
        r.raise_for_status()
        dst.write_bytes(r.content)
        manifest["files"][rel] = {"bytes": len(r.content), "sha256": hashlib.sha256(r.content).hexdigest()}
    (ROOT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"files": len(files), "bytes": sum(v["bytes"] for v in manifest["files"].values()), "out": str(ROOT.resolve())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
