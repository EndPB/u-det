"""HybridCodeAuthorship：行级/片段级标注数据集（单文件实现）。

原始数据：data/raw/HybridCodeAuthorship/hybrid.parquet（scripts/prepare.py hybrid 生成，
字段 RecordId / ModelId / Language / GitHubUrl / AICode / Attribution / LineNumber / AILineProportion）。

token 级标签由行级 Attribution 映射而来：每个 token 取"起始字符所在行"的标签
（AI=1 / Human=0），特殊 token 与 padding 记 -100。行级评测用 line_of_token + line_label。

用法::

    from dataio import build_dataset

    ds = build_dataset("hybrid", file="data/processed/hybrid.parquet", split="val", report=..., train=False)
    item = ds[0]        # input_ids / tok_labels / line_of_token / line_label / label / code / meta
"""

from __future__ import annotations

import bisect

import pyarrow.parquet as pq

from .base import BaseTokenDataset

RAW = "data/raw/HybridCodeAuthorship/hybrid.parquet"
IGNORE = -100


def read_hybrid(path: str = RAW) -> list[dict]:
    """读原始 hybrid parquet（约 1 万条，直接全量进内存）。"""
    table = pq.read_table(path)
    return table.to_pylist()


def tokenize_with_line_labels(tokenizer, code: str, attribution: list[str],
                              bos_id: int = 1, eos_id: int = 2) -> dict:
    """整段代码分词 -> 每 token 的行标签（-100 忽略），并保留行级标签用于评测。"""
    enc = tokenizer(code, add_special_tokens=False, return_offsets_mapping=True)
    lines = code.split("\n")
    starts = [0]
    for ln in lines[:-1]:
        starts.append(starts[-1] + len(ln) + 1)

    n_lines = min(len(lines), len(attribution))       # 行数不一致时按短的截断
    line_of_token = []
    for start, _ in enc["offset_mapping"]:
        line = bisect.bisect_right(starts, start) - 1
        line_of_token.append(min(max(line, 0), max(n_lines - 1, 0)))

    ids = [bos_id] + list(enc["input_ids"]) + [eos_id]
    labels = [1 if attribution[i] == "AI" else 0 for i in line_of_token]
    return {
        "input_ids": ids,
        "tok_labels": [IGNORE] + labels + [IGNORE],
        "line_of_token": [-1] + line_of_token + [-1],
        "line_label": [1 if a == "AI" else 0 for a in attribution[:n_lines]],
    }


class HybridDataset(BaseTokenDataset):
    """行级数据集：样本级标签恒为 1（AICode 含 AI 片段），token 级标签来自行级 Attribution。"""

    def __init__(self, file: str = "data/processed/hybrid.parquet", split: str = "train", **kwargs):
        super().__init__(**kwargs)
        table = pq.read_table(file)
        self.ids, self.tok, self.codes, self.labels, self.meta = [], [], [], [], []
        self.line_of_token, self.line_label = [], []
        for i, s in enumerate(table.column("split").to_pylist()):
            if split not in ("all", "*", s):
                continue
            self.ids.append(table.column("input_ids")[i].as_py())
            self.tok.append(table.column("tok_labels")[i].as_py())
            self.line_of_token.append(table.column("line_of_token")[i].as_py())
            self.line_label.append(table.column("line_label")[i].as_py())
            self.codes.append(table.column("code")[i].as_py())
            self.labels.append(int(table.column("label")[i].as_py()))
            self.meta.append({
                "record_id": table.column("RecordId")[i].as_py(),
                "model_id": table.column("ModelId")[i].as_py(),
            })

    def __getitem__(self, index: int) -> dict:
        item = super().__getitem__(index)
        if not self.train:      # 评测需要行级信息（此时不做窗口裁剪，报告前缀补 -1 占位）
            pad = len(item["input_ids"]) - len(self.ids[index])
            item["line_of_token"] = [-1] * pad + list(self.line_of_token[index])
            item["line_label"] = list(self.line_label[index])
        return item
