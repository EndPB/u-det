"""CoDET-M4 数据集加载与处理。

数据集：https://huggingface.co/datasets/DaniilOr/CoDET-M4 （单文件 parquet，约 437 MB，50 万条）

字段说明（与 HF 上的 parquet 一致）：
    * ``code``          —— 原始代码片段；
    * ``cleaned_code``  —— 去除注释后的代码（推荐作为模型输入）；
    * ``language``      —— 编程语言；
    * ``model``         —— 生成该样本的模型（人类样本可能为空 / 特定标记）；
    * ``target``        —— 标签列（如 ``human`` / ``machine``）；
    * ``source``        —— 样本来源数据集；
    * ``split``         —— 官方划分标记；
    * ``features``      —— 8 个手工统计特征（平均函数长度、可维护性指数等）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

# ---- 列名常量 ----
CODE = "code"
CLEANED_CODE = "cleaned_code"
LANGUAGE = "language"
MODEL = "model"
TARGET = "target"
SOURCE = "source"
SPLIT = "split"
FEATURES = "features"
INDEX = "__index_level_0__"

ALL_COLUMNS = [CODE, LANGUAGE, MODEL, SPLIT, TARGET, SOURCE, FEATURES, CLEANED_CODE]

DEFAULT_RAW_FILENAME = "dataset_without_comments.parquet"


def find_parquet(raw_dir: Union[str, Path], filename: str = DEFAULT_RAW_FILENAME) -> Path:
    """在目录中定位数据文件；找不到时给出清晰提示。"""
    raw_dir = Path(raw_dir)
    path = raw_dir / filename
    if path.exists():
        return path
    candidates = sorted(raw_dir.glob("*.parquet")) if raw_dir.exists() else []
    if candidates:
        return candidates[0]
    raise FileNotFoundError(
        f"未找到数据集文件: {path}\n"
        f"请先运行: python scripts/download_data.py --output-dir {raw_dir}"
    )


def load_codet_m4(
    path: Union[str, Path, None] = None,
    raw_dir: Union[str, Path, None] = None,
    columns: Optional[Sequence[str]] = None,
    cache_dir: Optional[Union[str, Path]] = None,
):
    """加载 CoDET-M4 为 ``datasets.Dataset``。

    Args:
        path: parquet 文件路径（优先）。
        raw_dir: 数据目录（当 ``path`` 为空时在该目录内查找）。
        columns: 需要保留的列，默认全部。
        cache_dir: datasets 缓存目录（可选）。

    Returns:
        ``datasets.Dataset``（单 train split）。
    """
    from datasets import load_dataset

    if path is None:
        if raw_dir is None:
            raise ValueError("必须提供 path 或 raw_dir 之一")
        path = find_parquet(raw_dir)
    path = Path(path)

    ds = load_dataset(
        "parquet",
        data_files=str(path),
        split="train",
        cache_dir=str(cache_dir) if cache_dir else None,
    )
    if columns is not None:
        keep = [c for c in columns if c in ds.column_names]
        ds = ds.select_columns(keep)
    return ds


def split_dataset(
    ds,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    split_column: str = SPLIT,
    use_existing_split: bool = True,
):
    """划分 train / validation / test。

    优先使用数据集自带的 ``split`` 列（若取值覆盖 train/valid/test）；
    否则进行随机划分。

    Returns:
        ``datasets.DatasetDict``，键为 ``train`` / ``validation`` / ``test``。
    """
    from datasets import DatasetDict

    if use_existing_split and split_column in ds.column_names:
        values = set(np.unique(ds[split_column]).tolist())
        normalized = {str(v).lower() for v in values}
        has_train = any(v in {"train", "training"} for v in normalized)
        has_val = any(v in {"valid", "validation", "val", "dev"} for v in normalized)
        has_test = any(v in {"test", "testing"} for v in normalized)
        if has_train and (has_val or has_test):
            mapping = {
                "train": {"train", "training"},
                "validation": {"valid", "validation", "val", "dev"},
                "test": {"test", "testing"},
            }
            out = {}
            for target_name, keys in mapping.items():
                sub = ds.filter(lambda x: str(x[split_column]).lower() in keys, desc=f"筛选 {target_name}")
                if len(sub) > 0:
                    out[target_name] = sub
            if "train" in out and ("validation" in out or "test" in out):
                return DatasetDict(out)
            print("[split_dataset] 自带 split 列不完整，改用随机划分。")

    shuffled = ds.shuffle(seed=seed)
    n_total = len(shuffled)
    n_test = int(n_total * test_ratio)
    n_val = int(n_total * val_ratio)
    n_train = n_total - n_val - n_test
    return DatasetDict(
        {
            "train": shuffled.select(range(0, n_train)),
            "validation": shuffled.select(range(n_train, n_train + n_val)),
            "test": shuffled.select(range(n_train + n_val, n_total)),
        }
    )


def tokenize_dataset(
    dataset,
    tokenizer,
    text_column: str = CLEANED_CODE,
    max_length: int = 512,
    num_proc: Optional[int] = 1,
    batched: bool = True,
):
    """把文本列批量编码为 ``input_ids`` / ``attention_mask``。"""

    def _encode(examples: Dict[str, List[str]]) -> Dict[str, List[List[int]]]:
        return tokenizer(
            examples[text_column],
            truncation=True,
            max_length=max_length,
            padding=False,  # 动态 padding 交给 collator
        )

    return dataset.map(
        _encode,
        batched=batched,
        num_proc=num_proc,
        desc="tokenize",
        remove_columns=None,
    )


def dataset_summary(ds, batch_size: int = 50000) -> Dict[str, Any]:
    """统计类别分布、语言分布等，便于快速了解数据。

    采用分批计数（而非一次性取出整列），内存占用低，适合小内存实例。
    """
    from collections import Counter

    summary: Dict[str, Any] = {"num_rows": len(ds)}

    fields = {
        "target_counts": TARGET,
        "language_counts": LANGUAGE,
        "model_counts": MODEL,
        "source_counts": SOURCE,
        "split_counts": SPLIT,
    }
    counters = {col: Counter() for col in fields.values() if col in ds.column_names}
    if not counters:
        return summary

    for start in range(0, len(ds), batch_size):
        batch = ds[start : start + batch_size]
        for col, counter in counters.items():
            counter.update(str(v) for v in batch[col])

    for key, col in fields.items():
        if col in counters:
            counter = counters[col]
            top = 20 if col in (LANGUAGE, MODEL, SOURCE) else None
            summary[key] = dict(counter.most_common(top) if top else counter.most_common())
    return summary
