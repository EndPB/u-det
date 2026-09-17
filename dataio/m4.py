"""CoDET-M4：样本级二分类数据集（单文件实现）。

原始数据：data/raw/CoDET-M4/dataset_without_comments.parquet（50 万行，多语言、多生成模型）。
由 scripts/build_subset.py 过滤 + 均衡 + 预分词后写入 data/processed/m4.parquet，
训练时只读后者（字段：split / code / input_ids / target / language / model）。
"""

from __future__ import annotations

import pyarrow.parquet as pq

from .base import BaseTokenDataset

RAW = "data/raw/CoDET-M4/dataset_without_comments.parquet"
TARGETS = {"human": 0, "ai": 1}
M4_COLUMNS = ["cleaned_code", "target", "language", "model", "split"]


def iter_m4(path: str = RAW, columns=None, batch_size: int = 8192):
    """流式读取原始 parquet（按 row group 分批，避免整表物化）。"""
    parquet = pq.ParquetFile(path)
    columns = list(columns or M4_COLUMNS)
    for batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
        yield batch.to_pydict()


class M4Dataset(BaseTokenDataset):
    """样本级数据集（human=0 / ai=1），供样本级任务与 human 类。"""

    def __init__(self, file: str = "data/processed/m4.parquet", split: str = "train", **kwargs):
        super().__init__(**kwargs)
        table = pq.read_table(file)
        index = [
            i for i, s in enumerate(table.column("split").to_pylist())
            if split in ("all", "*", s)
        ]
        self.ids, self.codes, self.labels, self.meta = [], [], [], []
        for i in index:
            self.ids.append(table.column("input_ids")[i].as_py())
            self.codes.append(table.column("code")[i].as_py())
            self.labels.append(TARGETS[table.column("target")[i].as_py()])
            self.meta.append({
                "language": table.column("language")[i].as_py(),
                "model": table.column("model")[i].as_py(),
            })
        self.tok = None
