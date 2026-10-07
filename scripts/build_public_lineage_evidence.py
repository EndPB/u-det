"""系列证据（官方资料拉取，服务器侧重构版）：pins + 原文 + sha256。

说明：指导方交接包中的 official_docs/ 与来源 JSON 未上传（expected manifest 16/16 缺失，
见 public_full_followup_2026-10-08/expected_manifest_check.json）。本脚本按指导 §2 给出的
三个官方 URL 直接拉取原始资料并固定 commit，输出到 official_docs_server_fetch/；
原版 curated 文件到达后再做逐文件核对。
"""
from __future__ import annotations

import hashlib
import json
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "d-det/artifacts/public_full_followup_2026-10-08/official_docs_server_fetch"
DOCS = [
    {"name": "CodeLlama-Instruct", "repo": "meta-llama/codellama",
     "raw": "https://raw.githubusercontent.com/meta-llama/codellama/main/MODEL_CARD.md",
     "out": "CodeLlama-MODEL_CARD.md"},
    {"name": "Qwen2.5-Coder-Instruct", "repo": "huggingface/Qwen2.5-Coder",
     "raw": "https://raw.githubusercontent.com/huggingface/Qwen2.5-Coder/main/README.md",
     "out": "Qwen2.5-Coder-README.md"},
    {"name": "DeepSeek-Coder-v1-Instruct", "repo": "deepseek-ai/DeepSeek-Coder",
     "raw": "https://raw.githubusercontent.com/deepseek-ai/DeepSeek-Coder/main/README.md",
     "out": "DeepSeek-Coder-README.md"},
]
CONTESTS_PROTO = {"name": "CodeContests-schema", "repo": "google-deepmind/code_contests",
                  "raw": "https://raw.githubusercontent.com/google-deepmind/code_contests/main/contest_problem.proto",
                  "out": "contest_problem.proto"}


def fetch(url: str, timeout=60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def repo_pin(repo: str) -> dict:
    try:
        data = json.loads(fetch(f"https://api.github.com/repos/{repo}/commits?per_page=1"))
        c = data[0]
        return {"commit": c["sha"], "commit_date": c["commit"]["committer"]["date"]}
    except Exception as e:
        return {"commit": None, "error": f"{type(e).__name__}: {e}"}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sources = {"schema": "public_lineage_official_docs_sources_server_fetch_v1",
               "fetched_utc": datetime.now(timezone.utc).isoformat(),
               "note": "服务器侧重构：交接包 curated official_docs/ 未上传；本目录为按 §2 URL 的原始拉取+commit pin",
               "docs": []}
    for d in DOCS + [CONTESTS_PROTO]:
        t0 = time.time()
        pin = repo_pin(d["repo"])
        body = fetch(d["raw"])
        p = OUT / d["out"]
        p.write_bytes(body)
        sources["docs"].append({
            "name": d["name"], "repo": d["repo"], "url": d["raw"], "pin": pin,
            "path": str(p.relative_to(ROOT)), "bytes": len(body),
            "sha256": hashlib.sha256(body).hexdigest(), "fetch_seconds": round(time.time() - t0, 1),
        })
        print("fetched", d["name"], len(body), "bytes; pin:", str(pin.get("commit"))[:12])
    (OUT / "sources.json").write_text(json.dumps(sources, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("sources.json written")


if __name__ == "__main__":
    main()
