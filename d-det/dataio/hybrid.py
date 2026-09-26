"""hybrid 数据集（token/行级人机标注；HybridCodeAuthorship）。

u-det 同源（data/processed/hybrid.parquet 已随数据复制进 d-det）：
- tok_labels：每个 token 取"起始字符所在行"的标签（AI=1/Human=0；-100=忽略）；
- line_of_token / line_label：行级评测用；
- 样本标签恒为 1（文件含 AI 段）——不进样本头损失，只做 token 级监督。

与 u-det 版的差异：去掉 report（报告前缀）依赖；batch=1、不截断，
窗口语义交给分块编码器（codet5blk）。
"""

from __future__ import annotations

import pyarrow.parquet as pq

from .base import BaseDataset

IGNORE = -100


class HybridDataset(BaseDataset):
    """token/行级数据集。attrs：ids / tok / codes / labels / meta / line_of_token / line_label。"""

    def __init__(self, file: str = "data/processed/hybrid.parquet", split: str = "train", **kwargs):
        super().__init__(**kwargs)
        self.tok, self.line_of_token, self.line_label = [], [], []
        table = pq.read_table(file)
        cols = set(table.column_names)
        index = [i for i, s in enumerate(table.column("split").to_pylist())
                 if split in ("all", "*", s)]
        for i in index:
            self.ids.append(table.column("input_ids")[i].as_py())
            self.tok.append(table.column("tok_labels")[i].as_py())
            self.line_of_token.append(table.column("line_of_token")[i].as_py())
            self.line_label.append(table.column("line_label")[i].as_py())
            self.codes.append(table.column("code")[i].as_py())
            self.labels.append(int(table.column("label")[i].as_py()))
            rec = {}
            if "RecordId" in cols:
                rec["record_id"] = table.column("RecordId")[i].as_py()
            if "ModelId" in cols:
                rec["model_id"] = table.column("ModelId")[i].as_py()
            self.meta.append(rec)

    def __getitem__(self, index: int) -> dict:
        return {
            "input_ids": list(self.ids[index]),
            "tok_labels": list(self.tok[index]),
            "line_of_token": list(self.line_of_token[index]),
            "line_label": list(self.line_label[index]),
            "label": int(self.labels[index]),
            "code": self.codes[index],
            "meta": dict(self.meta[index]),
        }
