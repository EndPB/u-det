"""CoDET-M4 样本级二分类数据集（s1 的 BCE 监督）。

只读 u-det 侧 ``scripts/build_subset.py`` 预分词后的 ``data/processed/m4.parquet``
（已复制进 d-det），字段：split / code / input_ids / target / language / model。
human=0 / ai=1。

⚠️ 已知数据特性（继承自 u-det lessons C2）：**长度与标签强混淆**
（AI 样本全部 <2048 token，4096+ 全为 human；纯长度规则的 val acc 就有 0.659）。
因此 s1 的评测必须带**长度分桶**（scripts/analyze_scores.py），不能只看总体指标。
"""

from __future__ import annotations

import pyarrow.parquet as pq

from .base import BaseDataset

TARGETS = {"human": 0, "ai": 1}


class M4Dataset(BaseDataset):
    """样本级数据集（human=0 / ai=1）。"""

    def __init__(self, file: str = "data/processed/m4.parquet", split: str = "train", **kwargs):
        super().__init__(**kwargs)
        table = pq.read_table(file)
        index = [i for i, s in enumerate(table.column("split").to_pylist())
                 if split in ("all", "*", s)]
        for i in index:
            self.ids.append(table.column("input_ids")[i].as_py())
            self.codes.append(table.column("code")[i].as_py())
            self.labels.append(TARGETS[table.column("target")[i].as_py()])
            self.meta.append({
                "language": table.column("language")[i].as_py(),
                "model": table.column("model")[i].as_py(),
            })

    def __getitem__(self, index: int) -> dict:
        return {
            "input_ids": list(self.ids[index]),
            "label": int(self.labels[index]),
            "code": self.codes[index],
            "meta": dict(self.meta[index]),
        }
