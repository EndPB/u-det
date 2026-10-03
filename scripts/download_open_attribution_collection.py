"""Download open attribution benchmark shards with streaming and checksums.

The script keeps the original Parquet files and writes a manifest. It never
loads a dataset table into memory. HF traffic defaults to hf-mirror.com.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import errno
import time
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "d-det" / "data" / "acl_attribution_collection_v1" / "raw"
HF_BASE = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com").rstrip("/")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def get_json(url: str) -> dict:
    r = requests.get(url, timeout=(20, 60))
    r.raise_for_status()
    return r.json()


def download(url: str, dst: Path, expected_size: int | None) -> dict:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and (expected_size is None or dst.stat().st_size == expected_size):
        return {
            "status": "existing",
            "bytes": dst.stat().st_size,
            "sha256": sha256(dst),
        }
    part = dst.with_suffix(dst.suffix + ".part")
    if part.exists():
        part.unlink()
    got = 0
    started = time.time()
    with requests.get(url, stream=True, timeout=(30, 180), allow_redirects=True) as r:
        r.raise_for_status()
        with part.open("wb") as f:
            for block in r.iter_content(4 * 1024 * 1024):
                if block:
                    f.write(block)
                    got += len(block)
                    if got % (128 * 1024 * 1024) < len(block):
                        print(f"[download] {dst.name}: {got / 1e9:.2f} GB", flush=True)
    if expected_size is not None and got != expected_size:
        raise RuntimeError(f"short download for {dst}: {got} != {expected_size}")
    last_error = None
    for attempt in range(20):
        try:
            part.replace(dst)
            break
        except PermissionError as exc:
            last_error = exc
            time.sleep(1.0)
    else:
        raise last_error
    return {
        "status": "downloaded",
        "bytes": got,
        "sha256": sha256(dst),
        "seconds": round(time.time() - started, 2),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--configs", nargs="+", default=["T1", "T2", "T3"],
                    choices=["T1", "T2", "T3"])
    ap.add_argument("--min-free-gib", type=float, default=20.0)
    args = ap.parse_args()

    disk_root = Path(args.out.anchor or args.out.drive or str(ROOT))
    free = shutil.disk_usage(disk_root).free / 1024**3
    if free < args.min_free_gib:
        raise RuntimeError(f"Only {free:.1f} GiB free; refusing download")

    api_url = f"{HF_BASE}/api/datasets/AICD-bench/AICD-Bench"
    api = get_json(api_url)
    (args.out / "aicd_api.json").parent.mkdir(parents=True, exist_ok=True)
    (args.out / "aicd_api.json").write_text(
        json.dumps(api, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    wanted = []
    for item in api.get("siblings", []):
        name = item.get("rfilename", "")
        if name.endswith(".parquet") and any(name.startswith(c + "/") for c in args.configs):
            wanted.append(name)
    wanted.sort()
    if not wanted:
        raise RuntimeError("No AICD parquet shards found in API response")

    manifest = {
        "dataset": "AICD-bench/AICD-Bench",
        "revision": api.get("sha"),
        "endpoint": HF_BASE,
        "configs": args.configs,
        "files": {},
    }
    for name in wanted:
        url = f"{HF_BASE}/datasets/AICD-bench/AICD-Bench/resolve/{api['sha']}/{name}"
        dst = args.out / "aicd" / name
        # HEAD follows the mirror's signed Xet redirect and gives the exact size.
        h = requests.head(url, allow_redirects=True, timeout=(20, 60))
        h.raise_for_status()
        size = int(h.headers.get("content-length", "0")) or None
        print(f"[start] {name} expected={size}", flush=True)
        rec = download(url, dst, size)
        rec.update({"url": url, "revision": api["sha"]})
        manifest["files"][name] = rec
        print(f"[done] {name} {rec['bytes'] / 1e6:.1f} MB {rec['sha256']}", flush=True)

    manifest["total_bytes"] = sum(v["bytes"] for v in manifest["files"].values())
    (args.out / "aicd_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "files": len(wanted),
        "bytes": manifest["total_bytes"],
        "revision": api["sha"],
        "manifest": str(args.out / "aicd_manifest.json"),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
