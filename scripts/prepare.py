#!/usr/bin/env python
"""数据与权重准备（单文件脚本）。

用法::

    python scripts/prepare.py data       # 下载 CoDET-M4 全量数据集（约 437 MB）
    python scripts/prepare.py encoder    # 下载 CodeT5-base 编码器权重（约 850 MB）
    python scripts/prepare.py all        # 全部下载

说明：
    * 默认走 hf-mirror.com 镜像（国内直连可用）；
    * 如需官方源，先 ``source /etc/network_turbo``，再 ``--endpoint https://huggingface.co``。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_REPO = "DaniilOr/CoDET-M4"
DATA_FILE = "dataset_without_comments.parquet"
ENCODER_REPO = "Salesforce/codet5-base"
ENCODER_PATTERNS = ["*.json", "*.txt", "pytorch_model*.bin", "*.safetensors", "*.model"]


def download_data(endpoint: str, out_dir: Path) -> Path:
    """下载 CoDET-M4 全量 parquet（500,552 行）。"""
    from huggingface_hub import snapshot_download

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[data] {DATA_REPO} -> {out_dir}")
    snapshot_download(
        repo_id=DATA_REPO,
        repo_type="dataset",
        local_dir=str(out_dir),
        allow_patterns=[DATA_FILE, "*.md"],
        max_workers=4,
    )
    path = out_dir / DATA_FILE
    if not path.exists():
        raise FileNotFoundError(f"未找到 {path}")
    print(f"[data] 完成：{path} ({path.stat().st_size / 1048576:.1f} MB)")
    return path


def download_encoder(endpoint: str, out_dir: Path, repo: str = ENCODER_REPO) -> Path:
    """下载编码器权重（默认 CodeT5-base）。"""
    from huggingface_hub import snapshot_download

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[encoder] {repo} -> {out_dir}")
    snapshot_download(
        repo_id=repo,
        local_dir=str(out_dir),
        allow_patterns=ENCODER_PATTERNS,
        max_workers=4,
    )
    weights = [p for p in out_dir.glob("*") if p.suffix in {".bin", ".safetensors"}]
    if not weights:
        raise FileNotFoundError(f"{out_dir} 中未找到权重文件")
    size = sum(p.stat().st_size for p in out_dir.rglob("*") if p.is_file()) / 1048576
    print(f"[encoder] 完成：{out_dir}（{len(weights)} 个权重文件，合计 {size:.1f} MB）")
    return out_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="下载 CoDET-M4 数据集与编码器权重")
    parser.add_argument("target", choices=["data", "encoder", "all"], help="下载目标")
    parser.add_argument("--endpoint", default=os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"),
                        help="HF 端点（默认 hf-mirror 镜像）")
    parser.add_argument("--data-dir", default=str(ROOT / "data" / "raw" / "CoDET-M4"))
    parser.add_argument("--encoder-dir", default=str(ROOT / "checkpoints" / "codet5-base"))
    parser.add_argument("--encoder-repo", default=ENCODER_REPO)
    args = parser.parse_args()

    # 必须在导入 huggingface_hub 之前设置端点
    os.environ["HF_ENDPOINT"] = args.endpoint
    print(f"[prepare] 端点：{args.endpoint}")
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        print("请先安装依赖：pip install -r requirements.txt", file=sys.stderr)
        return 1

    if args.target in ("data", "all"):
        download_data(args.endpoint, Path(args.data_dir))
    if args.target in ("encoder", "all"):
        download_encoder(args.endpoint, Path(args.encoder_dir), args.encoder_repo)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
