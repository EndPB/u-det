#!/usr/bin/env python
"""数据与权重准备（单文件脚本）。

用法::

    python scripts/prepare.py data       # 下载 CoDET-M4 全量数据集（约 437 MB）
    python scripts/prepare.py encoder    # 下载 CodeT5-base 编码器权重（约 850 MB）
    python scripts/prepare.py hybrid     # 下载 HybridCodeAuthorship（8 个 CSV，约 408 MB）并转 parquet
    python scripts/prepare.py all        # data + encoder + hybrid

说明：
    * HF 默认走 hf-mirror.com 镜像（国内直连可用）；官方源需 ``source /etc/network_turbo``；
    * hybrid 在 GitHub 上，必须先 ``source /etc/network_turbo``（或手动设 http_proxy/https_proxy）。
"""

from __future__ import annotations

import argparse
import ast
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_REPO = "DaniilOr/CoDET-M4"
DATA_FILE = "dataset_without_comments.parquet"
ENCODER_REPO = "Salesforce/codet5-base"
ENCODER_PATTERNS = ["*.json", "*.txt", "pytorch_model*.bin", "*.safetensors", "*.model"]

# HybridCodeAuthorship：行级/片段级 AI 代码标注（8 个 CSV，按 part 分片）
HYBRID_BASE = "https://raw.githubusercontent.com/CapitalOne-Research/c1-hybrid-code-authorship/main/data"
HYBRID_FILES = [f"HybridCodeAuthorship_part_{i}.csv" for i in range(1, 9)]
HYBRID_KEEP = [
    "RecordId", "ModelId", "Language", "GitHubUrl",
    "AICode", "Attribution", "LineNumber", "AILineProportion",
]


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


def download_hybrid(out_dir: Path, force: bool = False) -> Path:
    """下载 HybridCodeAuthorship 的 8 个 CSV 并合并转成 parquet。

    GitHub 需要代理：``source /etc/network_turbo``（或手动设置 http_proxy/https_proxy）。
    只保留训练需要的列（AICode + 行级 Attribution/LineNumber + 元信息）；
    用标准库 csv 解析（pyarrow 的 CSV 解析器对该文件的引号/换行处理有问题）。
    """
    import csv

    import pyarrow as pa
    import pyarrow.parquet as pq

    csv.field_size_limit(10**9)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in HYBRID_FILES:
        dst = out_dir / name
        if dst.exists() and dst.stat().st_size > 0 and not force:
            print(f"[hybrid] 已存在，跳过：{name}")
            continue
        url = f"{HYBRID_BASE}/{name}"
        print(f"[hybrid] 下载 {url}")
        with urllib.request.urlopen(url, timeout=120) as resp, open(dst, "wb") as f:
            while chunk := resp.read(1 << 20):
                f.write(chunk)
        print(f"[hybrid] {name}: {dst.stat().st_size / 1048576:.1f} MB")

    schema = pa.schema([
        ("RecordId", pa.string()), ("ModelId", pa.string()), ("Language", pa.string()),
        ("GitHubUrl", pa.string()), ("AICode", pa.string()),
        ("Attribution", pa.list_(pa.string())), ("LineNumber", pa.list_(pa.int32())),
        ("AILineProportion", pa.float32()),
    ])
    out_path = out_dir / "hybrid.parquet"
    rows = bad_lines = bad_attribution = 0
    batch, batch_size = [], 512

    def flush(writer):
        if not batch:
            return
        writer.write_table(pa.table({
            "RecordId": pa.array([r["RecordId"] for r in batch], pa.string()),
            "ModelId": pa.array([r["ModelId"] for r in batch], pa.string()),
            "Language": pa.array([r["Language"] for r in batch], pa.string()),
            "GitHubUrl": pa.array([r["GitHubUrl"] for r in batch], pa.string()),
            "AICode": pa.array([r["AICode"] for r in batch], pa.string()),
            "Attribution": pa.array([r["Attribution"] for r in batch], pa.list_(pa.string())),
            "LineNumber": pa.array([r["LineNumber"] for r in batch], pa.list_(pa.int32())),
            "AILineProportion": pa.array([r["AILineProportion"] for r in batch], pa.float32()),
        }, schema=schema))
        batch.clear()

    with pq.ParquetWriter(out_path, schema) as writer:
        for name in HYBRID_FILES:
            with open(out_dir / name, newline="", encoding="utf-8") as f:
                for rec in csv.DictReader(f):
                    code = rec["AICode"]
                    try:
                        attribution = ast.literal_eval(rec["Attribution"])
                        linenos = ast.literal_eval(rec["LineNumber"])
                    except (ValueError, SyntaxError):
                        bad_lines += 1
                        continue
                    text = (code or "").replace("\r\n", "\n")
                    if len(text.split("\n")) != len(attribution):     # 行数与标注数不一致
                        bad_attribution += 1
                    batch.append({
                        "RecordId": rec["RecordId"], "ModelId": rec["ModelId"],
                        "Language": rec["Language"], "GitHubUrl": rec["GitHubUrl"],
                        "AICode": text, "Attribution": [str(a) for a in attribution],
                        "LineNumber": [int(x) for x in linenos],
                        "AILineProportion": float(rec["AILineProportion"] or 0.0),
                    })
                    rows += 1
                    if len(batch) >= batch_size:
                        flush(writer)
            print(f"[hybrid] {name}: 累计 {rows} 行")
        flush(writer)

    print(f"[hybrid] 完成：{out_path}（{rows} 行，{out_path.stat().st_size / 1048576:.1f} MB）")
    if bad_lines or bad_attribution:
        print(f"[hybrid] 注意：解析失败 {bad_lines} 行；行数与 Attribution 数不一致 {bad_attribution} 行"
              f"（后者在 build_subset 里按短的一侧截断）")
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="下载 CoDET-M4 / HybridCodeAuthorship 数据集与编码器权重")
    parser.add_argument("target", choices=["data", "encoder", "hybrid", "all"], help="下载目标")
    parser.add_argument("--endpoint", default=os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"),
                        help="HF 端点（默认 hf-mirror 镜像）")
    parser.add_argument("--data-dir", default=str(ROOT / "data" / "raw" / "CoDET-M4"))
    parser.add_argument("--hybrid-dir", default=str(ROOT / "data" / "raw" / "HybridCodeAuthorship"))
    parser.add_argument("--encoder-dir", default=str(ROOT / "checkpoints" / "codet5-base"))
    parser.add_argument("--encoder-repo", default=ENCODER_REPO)
    parser.add_argument("--force", action="store_true", help="hybrid：重新下载已存在的 CSV")
    args = parser.parse_args()

    # 必须在导入 huggingface_hub 之前设置端点
    os.environ["HF_ENDPOINT"] = args.endpoint
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        print("请先安装依赖：pip install -r requirements.txt", file=sys.stderr)
        return 1

    if args.target in ("data", "hybrid", "all"):
        if args.target != "hybrid":
            print(f"[prepare] HF 端点：{args.endpoint}")
    if args.target in ("data", "all"):
        download_data(args.endpoint, Path(args.data_dir))
    if args.target in ("encoder", "all"):
        download_encoder(args.endpoint, Path(args.encoder_dir), args.encoder_repo)
    if args.target in ("hybrid", "all"):
        download_hybrid(Path(args.hybrid_dir), force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
