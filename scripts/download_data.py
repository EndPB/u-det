#!/usr/bin/env python
"""下载 CoDET-M4 数据集（全量）。

说明：
    * 数据集只有一个 parquet 文件（约 437 MB，500,552 行），下载即"全量"；
    * AutoDL 国内机器默认使用 ``hf-mirror.com`` 镜像（无需代理）；
    * 若已 ``source /etc/network_turbo``，也可以 ``--endpoint https://huggingface.co`` 走官方源。

用法::

    python scripts/download_data.py
    python scripts/download_data.py --output-dir data/raw/CoDET-M4 --endpoint https://hf-mirror.com
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO = "DaniilOr/CoDET-M4"
DEFAULT_FILENAME = "dataset_without_comments.parquet"
DEFAULT_ENDPOINT = "https://hf-mirror.com"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="下载 CoDET-M4 数据集")
    parser.add_argument("--repo-id", default=DEFAULT_REPO, help="HuggingFace 数据集仓库 ID")
    parser.add_argument("--filename", default=DEFAULT_FILENAME, help="要下载的 parquet 文件名（默认全量文件）")
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "data" / "raw" / "CoDET-M4"),
        help="下载目录",
    )
    parser.add_argument("--endpoint", default=os.environ.get("HF_ENDPOINT", DEFAULT_ENDPOINT), help="HF 端点（镜像）")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"), help="HF 访问令牌（公开数据集无需）")
    parser.add_argument("--all-files", action="store_true", help="下载仓库内所有文件（README 等）")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # 必须在导入 huggingface_hub 之前设置端点，否则镜像不生效
    os.environ["HF_ENDPOINT"] = args.endpoint
    from huggingface_hub import snapshot_download

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[download_data] 仓库   : {args.repo_id}")
    print(f"[download_data] 端点   : {args.endpoint}")
    print(f"[download_data] 目标目录: {output_dir}")

    allow_patterns = None if args.all_files else [args.filename, "*.md", ".gitattributes"]

    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        local_dir=str(output_dir),
        allow_patterns=allow_patterns,
        token=args.token,
        max_workers=4,
    )

    parquet = output_dir / args.filename
    if not parquet.exists():
        print(f"[download_data][错误] 未找到文件: {parquet}", file=sys.stderr)
        return 1

    size_mb = parquet.stat().st_size / 1024 / 1024
    print(f"[download_data] 完成: {parquet} ({size_mb:.1f} MB)")

    # ---- 校验行数 ----
    try:
        import pyarrow.parquet as pq

        meta = pq.ParquetFile(parquet).metadata
        print(f"[download_data] 校验: rows={meta.num_rows}, columns={meta.num_columns}")
        print(f"[download_data] 字段: {meta.schema.names}")
    except Exception as exc:  # noqa: BLE001
        print(f"[download_data] 跳过校验（{exc}）")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
