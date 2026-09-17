"""数据模块：按名称切换数据集（单文件类实现，子类放同目录单文件）。

    from dataio import build_dataset, collate

    ds = build_dataset("m4", file="data/processed/m4.parquet", split="train", report=rep, max_length=512)
    ds = build_dataset("hybrid", file="data/processed/hybrid.parquet", split="val", train=False)
    loader = DataLoader(ds, batch_size=8, collate_fn=collate)

新增数据集：在 dataio/ 下新建单文件（继承 base.BaseTokenDataset），并在下面 DATASETS 加一行。
"""

from __future__ import annotations

import math

import torch

from .base import BaseTokenDataset
from .hybrid import HybridDataset
from .m4 import M4Dataset

DATASETS = {
    "m4": M4Dataset,
    "hybrid": HybridDataset,
}

IGNORE = -100


def list_datasets():
    """列出所有可用数据集名称。"""
    return sorted(DATASETS)


def build_dataset(name: str, **kwargs):
    """按名称构建数据集（cfg 里的 data.name / split 会传到这里）。"""
    if name not in DATASETS:
        raise KeyError(f"未知数据集 {name!r}，可选：{list_datasets()}")
    return DATASETS[name](**kwargs)


def collate(batch: list[dict], pad_id: int = 0, multiple: int = 1) -> dict:
    """组装 batch。v0.2 起一律 **batch=1**（FFT 是全局算子，padding 会污染频谱语义），
    因此默认不对长度做任何对齐（multiple=1）；保留参数以备小样本批量评测。"""
    longest = max(len(b["input_ids"]) for b in batch)
    length = int(math.ceil(longest / multiple) * multiple)

    size = len(batch)
    input_ids = torch.full((size, length), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((size, length), dtype=torch.long)
    tok_labels = torch.full((size, length), IGNORE, dtype=torch.long)
    for row, item in enumerate(batch):
        n = len(item["input_ids"])
        input_ids[row, :n] = torch.tensor(item["input_ids"], dtype=torch.long)
        attention_mask[row, :n] = 1
        if item["tok_labels"] is not None:
            tok_labels[row, :n] = torch.tensor(item["tok_labels"], dtype=torch.long)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "tok_labels": tok_labels,
        "labels": torch.tensor([b["label"] for b in batch], dtype=torch.long),
        "codes": [b["code"] for b in batch],
        "meta": [b["meta"] for b in batch],
    }
