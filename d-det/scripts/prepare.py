#!/usr/bin/env python
"""数据与权重准备（d-det 版，单文件脚本）。

与 u-det 版的区别：d-det 只需要两样**新**资产（数据 / 编码器权重已随项目复制）：

    pairmdl  同族 base / instruct 小模型对（用于生成 s2 的零标注配对数据）
    prompts  代码任务 prompt 集（生成配对的题目来源）

用法::

    python scripts/prepare.py pairmdl                    # 默认 Qwen2.5-Coder-0.5B 家族（约 1.9 GB）
    python scripts/prepare.py pairmdl --pair-name qwen2.5-coder-1.5b
    python scripts/prepare.py prompts                    # bigcodebench + humaneval（约 15 MB）
    python scripts/prepare.py all

说明：
    * HF 走 hf-mirror.com 镜像（国内直连可用，无需代理）；官方源需 ``source /etc/network_turbo``；
    * 只下载推理必需文件（safetensors / config / tokenizer）；
    * 既有资产（data/processed、checkpoints/codet5-base）随项目复制，本脚本不管。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# 同族模型对：name -> (base_repo, instruct_repo)
# 全部已验证 hf-mirror 可达、非 gated（见 docx/d-det-v0.1.md 的调研记录）。
MODEL_PAIRS = {
    "qwen2.5-coder-0.5b": ("Qwen/Qwen2.5-Coder-0.5B", "Qwen/Qwen2.5-Coder-0.5B-Instruct"),
    "qwen2.5-coder-1.5b": ("Qwen/Qwen2.5-Coder-1.5B", "Qwen/Qwen2.5-Coder-1.5B-Instruct"),
    "smollm2-360m": ("HuggingFaceTB/SmolLM2-360M", "HuggingFaceTB/SmolLM2-360M-Instruct"),
}

# prompt 来源数据集：repo -> 本地子目录名
PROMPT_REPOS = [
    ("bigcode/bigcodebench", "bigcodebench"),
    ("openai/openai_humaneval", "humaneval"),
]

MODEL_PATTERNS = ["*.json", "*.txt", "*.safetensors", "*.model"]


def download_pair_models(endpoint: str, name: str, ckpt_root: Path) -> None:
    """下载一对同族 base / instruct 模型到 checkpoints/<name>-{base,instruct}/。"""
    from huggingface_hub import snapshot_download

    base_repo, instruct_repo = MODEL_PAIRS[name]
    for repo, suffix in ((base_repo, "base"), (instruct_repo, "instruct")):
        out_dir = ckpt_root / f"{name}-{suffix}"
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[pairmdl] {repo} -> {out_dir}")
        snapshot_download(
            repo_id=repo,
            local_dir=str(out_dir),
            allow_patterns=MODEL_PATTERNS,
            max_workers=4,
        )
        size = sum(p.stat().st_size for p in out_dir.rglob("*") if p.is_file()) / 1048576
        print(f"[pairmdl] 完成：{out_dir}（{size:.1f} MB）")


def download_prompts(endpoint: str, out_root: Path) -> None:
    """下载 prompt 数据集（bigcodebench / humaneval）到 data/raw/prompts/<子目录>/。"""
    from huggingface_hub import snapshot_download

    for repo, sub in PROMPT_REPOS:
        out_dir = out_root / sub
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[prompts] {repo} -> {out_dir}")
        snapshot_download(
            repo_id=repo,
            repo_type="dataset",
            local_dir=str(out_dir),
            max_workers=4,
        )
        size = sum(p.stat().st_size for p in out_dir.rglob("*") if p.is_file()) / 1048576
        print(f"[prompts] 完成：{out_dir}（{size:.1f} MB）")


def main() -> int:
    parser = argparse.ArgumentParser(description="d-det：下载模型对 / prompt 集")
    parser.add_argument("target", choices=["pairmdl", "prompts", "all"], help="下载目标")
    parser.add_argument("--pair-name", default="qwen2.5-coder-0.5b", choices=sorted(MODEL_PAIRS),
                        help="模型对名称（见脚本顶部 MODEL_PAIRS）")
    parser.add_argument("--endpoint", default=os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"),
                        help="HF 端点（默认 hf-mirror 镜像）")
    parser.add_argument("--ckpt-dir", default=str(ROOT / "checkpoints"))
    parser.add_argument("--prompts-dir", default=str(ROOT / "data" / "raw" / "prompts"))
    args = parser.parse_args()

    # 必须在导入 huggingface_hub 之前设置端点
    os.environ["HF_ENDPOINT"] = args.endpoint
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        print("请先安装依赖：pip install -r requirements.txt", file=sys.stderr)
        return 1

    print(f"[prepare] HF 端点：{args.endpoint}")
    if args.target in ("pairmdl", "all"):
        download_pair_models(args.endpoint, args.pair_name, Path(args.ckpt_dir))
    if args.target in ("prompts", "all"):
        download_prompts(args.endpoint, Path(args.prompts_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
