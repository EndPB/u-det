"""base / instruct 配对数据集（s2 的 hinge margin 监督，**零人类标注**）。

数据由 ``scripts/gen_pairs.py`` 生成：同一 prompt 下同族 base 与 instruct 模型的输出对。
schema（data/processed/pairs.parquet）：

    split / task_id / family / language / x_plus / x_minus /
    input_ids_plus / input_ids_minus / n_tokens_plus / n_tokens_minus

约定：**x_plus = instruct 输出、x_minus = base 输出**
（训练目标 s2(x₊) ≥ s2(x₋) + margin，见 docx/d-det.md §3 的免标注估计原理）。

注意：配对双方都是"AI 输出"，**不参与 s1 的人机 BCE** —— 两个损失各自的监督信号
来自不同数据（人/AI 边界 vs AI 内部差分）、统计上互不重叠。〔2026-09-23 修正：消融 B（λc=0）实测两读出会自然共线（|cos|≈0.74）——「监督信号不重叠」≠「读出自然正交」；正交需 cos² 正则显式强制，见 docx/d-det-v0.1.md §5.6-B 与 v0.2.md。〕
"""

from __future__ import annotations

import pyarrow.parquet as pq

from .base import BaseDataset


class PairDataset(BaseDataset):
    def __init__(self, file: str = "data/processed/pairs.parquet", split: str = "train", **kwargs):
        super().__init__(**kwargs)
        table = pq.read_table(file)
        index = [i for i, s in enumerate(table.column("split").to_pylist())
                 if split in ("all", "*", s)]
        self.ids_plus, self.ids_minus = [], []
        self.codes_plus, self.codes_minus = [], []
        for i in index:
            self.ids_plus.append(table.column("input_ids_plus")[i].as_py())
            self.ids_minus.append(table.column("input_ids_minus")[i].as_py())
            self.codes_plus.append(table.column("x_plus")[i].as_py())
            self.codes_minus.append(table.column("x_minus")[i].as_py())
            self.meta.append({
                "task_id": table.column("task_id")[i].as_py(),
                "family": table.column("family")[i].as_py(),
                "language": table.column("language")[i].as_py(),
            })

    def __len__(self) -> int:
        return len(self.ids_plus)

    def __getitem__(self, index: int) -> dict:
        return {
            "input_ids_plus": list(self.ids_plus[index]),
            "input_ids_minus": list(self.ids_minus[index]),
            "code_plus": self.codes_plus[index],
            "code_minus": self.codes_minus[index],
            "meta": dict(self.meta[index]),
        }


class PairXFDataset(BaseDataset):
    """跨族配对（s2 的"方向引力"监督，v0.4）：同 task_id 在两个文件（族）各取一对。

    用途：λxfam · (1 − cos(Δh_A, Δh_B))，把不同族的 base→instruct 位移方向聚拢到同一
    根轴。两文件须由 scripts/gen_pairs.py 用同一 prompt 池生成（同 task_id 的 split 一致）。

    口径备注：若其中一族也是后续探针的目标族（如 qwen1.5），该族从"未见族"变为
    "已见族" —— 评测文档需明示（qwen15=已见；ds13 仍 held-out）。
    """

    def __init__(self, files: list | tuple = (), split: str = "train", **kwargs):
        super().__init__(**kwargs)
        files = list(files)
        if len(files) < 2:
            raise ValueError("PairXFDataset 需要至少两个配对文件")
        tables = [pq.read_table(f) for f in files[:2]]
        maps = []
        for t in tables:
            m = {}
            for i in range(t.num_rows):
                if split in ("all", "*", t.column("split")[i].as_py()):
                    m.setdefault(t.column("task_id")[i].as_py(), i)
            maps.append(m)
        common = sorted(set(maps[0]) & set(maps[1]))
        self.ids_plus_a, self.ids_minus_a = [], []
        self.ids_plus_b, self.ids_minus_b = [], []
        self.codes_plus_a, self.codes_minus_a = [], []
        self.codes_plus_b, self.codes_minus_b = [], []
        for tid in common:
            i, j = maps[0][tid], maps[1][tid]
            self.ids_plus_a.append(tables[0].column("input_ids_plus")[i].as_py())
            self.ids_minus_a.append(tables[0].column("input_ids_minus")[i].as_py())
            self.ids_plus_b.append(tables[1].column("input_ids_plus")[j].as_py())
            self.ids_minus_b.append(tables[1].column("input_ids_minus")[j].as_py())
            self.codes_plus_a.append(tables[0].column("x_plus")[i].as_py())
            self.codes_minus_a.append(tables[0].column("x_minus")[i].as_py())
            self.codes_plus_b.append(tables[1].column("x_plus")[j].as_py())
            self.codes_minus_b.append(tables[1].column("x_minus")[j].as_py())
            self.meta.append({
                "task_id": tid,
                "family_a": tables[0].column("family")[i].as_py(),
                "family_b": tables[1].column("family")[j].as_py(),
                "language": tables[0].column("language")[i].as_py(),
            })

    def __len__(self) -> int:
        return len(self.ids_plus_a)

    def __getitem__(self, index: int) -> dict:
        return {
            "input_ids_plus_a": list(self.ids_plus_a[index]),
            "input_ids_minus_a": list(self.ids_minus_a[index]),
            "input_ids_plus_b": list(self.ids_plus_b[index]),
            "input_ids_minus_b": list(self.ids_minus_b[index]),
            "meta": dict(self.meta[index]),
        }
