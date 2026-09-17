#!/usr/bin/env python
"""下载编码器权重（默认 CodeT5-base）。

    python scripts/download_encoder.py                          # codet5-base
    python scripts/download_encoder.py --model Salesforce/codet5-large
    python scripts/download_encoder.py --model microsoft/graphcodebert-base --name graphcodebert-base

下载完成后会在 ``checkpoints/<name>/`` 下得到完整的 transformers 权重目录，
``udet.encoders.build_encoder`` 与 ``build_tokenizer`` 可直接指向该目录。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "Salesforce/codet5-base"
DEFAULT_ENDPOINT = "https://hf-mirror.com"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="下载编码器预训练权重")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HF 模型 ID 或本地目录")
    parser.add_argument("--name", default=None, help="本地目录名（默认取模型 ID 的最后一段）")
    parser.add_argument("--output-root", default=str(PROJECT_ROOT / "checkpoints"))
    parser.add_argument("--endpoint", default=os.environ.get("HF_ENDPOINT", DEFAULT_ENDPOINT))
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument(
        "--allow-patterns",
        nargs="*",
        default=["*.json", "*.txt", "*.model", "pytorch_model*.bin", "*.safetensors", "*.md"],
        help="需要下载的文件模式（默认不含 flax/tf 权重）",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    os.environ["HF_ENDPOINT"] = args.endpoint
    from huggingface_hub import snapshot_download

    name = args.name or args.model.rstrip("/").split("/")[-1]
    output_dir = Path(args.output_root) / name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[download_encoder] 模型: {args.model}")
    print(f"[download_encoder] 端点: {args.endpoint}")
    print(f"[download_encoder] 目标: {output_dir}")

    snapshot_download(
        repo_id=args.model,
        local_dir=str(output_dir),
        allow_patterns=args.allow_patterns,
        token=args.token,
        max_workers=4,
    )

    # ---- 汇总 ----
    files = sorted(p for p in output_dir.rglob("*") if p.is_file())
    total = sum(p.stat().st_size for p in files) / 1024 / 1024
    print(f"[download_encoder] 完成，共 {len(files)} 个文件，合计 {total:.1f} MB")
    for p in files:
        print(f"    {p.relative_to(output_dir)}  {p.stat().st_size / 1024 / 1024:8.1f} MB")

    info = {"repo_id": args.model, "local_dir": str(output_dir), "endpoint": args.endpoint}
    with (output_dir.parent / f"{name}.json").open("w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    weights = [p for p in files if p.suffix in {".bin", ".safetensors"}]
    if not weights:
        print("[download_encoder][警告] 未发现权重文件，请检查 --allow-patterns", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
