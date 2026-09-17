"""分词器与小工具：与编码器配套使用。"""

from __future__ import annotations

from typing import Any, Optional

from transformers import AutoTokenizer, PreTrainedTokenizerBase


def build_tokenizer(
    model_name_or_path: str,
    use_fast: bool = True,
    local_files_only: Optional[bool] = None,
    **kwargs: Any,
) -> PreTrainedTokenizerBase:
    """构建分词器。

    CodeT5 使用 SentencePiece 分词器，仓库中通常只有 ``spiece.model``；
    若环境缺少 ``sentencepiece`` 或转换失败，会自动回退到慢速分词器。

    Args:
        model_name_or_path: 本地目录（如 ``checkpoints/codet5-base``）或 Hub 名称。
        use_fast: 是否优先使用 fast tokenizer。
        local_files_only: 是否强制离线加载；默认对本地路径自动启用。
    """
    if local_files_only is None:
        local_files_only = "/" in model_name_or_path or model_name_or_path.startswith(".")

    try:
        return AutoTokenizer.from_pretrained(
            model_name_or_path, use_fast=use_fast, local_files_only=local_files_only, **kwargs
        )
    except Exception as exc:  # noqa: BLE001 - 回退到慢速分词器
        if not use_fast:
            raise
        print(f"[build_tokenizer] fast tokenizer 加载失败（{exc}），回退到慢速实现。")
        return AutoTokenizer.from_pretrained(
            model_name_or_path, use_fast=False, local_files_only=local_files_only, **kwargs
        )
