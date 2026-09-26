"""数据模块：按名称切换数据集（单文件类实现，子类放同目录单文件）。

    m4    —— 人类 / AI 样本级标签（s1 的 BCE 监督）
    pair  —— 同 prompt 的 base / instruct 输出对（s2 的 hinge margin 监督）
    hybrid —— token/行级人机标签（tok_head 的 token BCE；评测 line/chunk/token F1）

用法::

    from dataio import build_dataset, COLLATES

    ds = build_dataset("m4", file="data/processed/m4.parquet", split="train")
    loader = DataLoader(ds, batch_size=1, collate_fn=COLLATES["m4"])

新增数据集：在 dataio/ 下新建单文件（继承 base.BaseDataset），
并在下面 DATASETS / COLLATES 各加一行。
"""

from __future__ import annotations

import torch

from .base import BaseDataset
from .hybrid import HybridDataset
from .m4 import M4Dataset
from .pair import PairDataset, PairXFDataset

DATASETS = {
    "m4": M4Dataset,
    "pair": PairDataset,
    "hybrid": HybridDataset,
    "pairxf": PairXFDataset,
}


def list_datasets():
    """列出所有可用数据集名称。"""
    return sorted(DATASETS)


def build_dataset(name: str, **kwargs):
    """按名称构建数据集（cfg 里的流名会传到这里）。"""
    if name not in DATASETS:
        raise KeyError(f"未知数据集 {name!r}，可选：{list_datasets()}")
    return DATASETS[name](**kwargs)


def _pad(rows: list, pad_id: int = 0):
    """把一批 token 序列 pad 到批内最长，返回 (ids, mask) 两个 long 张量。"""
    length = max(len(r) for r in rows)
    ids = torch.full((len(rows), length), pad_id, dtype=torch.long)
    mask = torch.zeros((len(rows), length), dtype=torch.long)
    for row, seq in enumerate(rows):
        n = len(seq)
        ids[row, :n] = torch.tensor(seq, dtype=torch.long)
        mask[row, :n] = 1
    return ids, mask


def collate_m4(batch: list) -> dict:
    input_ids, attention_mask = _pad([b["input_ids"] for b in batch])
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": torch.tensor([b["label"] for b in batch], dtype=torch.long),
        "codes": [b["code"] for b in batch],
        "meta": [b["meta"] for b in batch],
    }


def collate_pair(batch: list) -> dict:
    ids_plus, mask_plus = _pad([b["input_ids_plus"] for b in batch])
    ids_minus, mask_minus = _pad([b["input_ids_minus"] for b in batch])
    return {
        "input_ids_plus": ids_plus,
        "attention_mask_plus": mask_plus,
        "input_ids_minus": ids_minus,
        "attention_mask_minus": mask_minus,
        "codes_plus": [b["code_plus"] for b in batch],
        "codes_minus": [b["code_minus"] for b in batch],
        "meta": [b["meta"] for b in batch],
    }


def collate_pairxf(batch: list) -> dict:
    """pairxf：同 task_id 的两个族各一对 base/instruct（共 4 条序列）。"""
    ids_pa, mask_pa = _pad([b["input_ids_plus_a"] for b in batch])
    ids_ma, mask_ma = _pad([b["input_ids_minus_a"] for b in batch])
    ids_pb, mask_pb = _pad([b["input_ids_plus_b"] for b in batch])
    ids_mb, mask_mb = _pad([b["input_ids_minus_b"] for b in batch])
    return {
        "input_ids_plus_a": ids_pa,
        "attention_mask_plus_a": mask_pa,
        "input_ids_minus_a": ids_ma,
        "attention_mask_minus_a": mask_ma,
        "input_ids_plus_b": ids_pb,
        "attention_mask_plus_b": mask_pb,
        "input_ids_minus_b": ids_mb,
        "attention_mask_minus_b": mask_mb,
        "meta": [b["meta"] for b in batch],
    }


def collate_hybrid(batch: list) -> dict:
    """hybrid：tok_labels 的 pad 位补 -100（IGNORE，与 u-det 评测口径一致）。"""
    input_ids, attention_mask = _pad([b["input_ids"] for b in batch])
    tok_labels = torch.full_like(input_ids, -100)
    for row, b in enumerate(batch):
        n = len(b["tok_labels"])
        tok_labels[row, :n] = torch.tensor(b["tok_labels"], dtype=torch.long)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "tok_labels": tok_labels,
        "labels": torch.tensor([b["label"] for b in batch], dtype=torch.long),
        "line_of_token": [b["line_of_token"] for b in batch],
        "line_label": [b["line_label"] for b in batch],
        "codes": [b["code"] for b in batch],
        "meta": [b["meta"] for b in batch],
    }


COLLATES = {
    "m4": collate_m4,
    "pair": collate_pair,
    "hybrid": collate_hybrid,
    "pairxf": collate_pairxf,
}
