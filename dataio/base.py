"""数据集基类（单文件）：统一"报告前缀 + 组装"的取数逻辑。

v0.2 起**不截断代码**：每条样本整段进入模型（频域 U-Net 对长度无要求，训练/评测均 batch=1）。

子类只需把 processed parquet 读进以下列表：
    self.ids     每条的代码 token（不含报告）
    self.tok     每条的 token 标签（-100 忽略；样本级数据为 None）
    self.codes   每条原始代码文本（生成报告用）
    self.labels  每条样本级标签
    self.meta    每条附加信息（dict，可选）
"""

from __future__ import annotations

from typing import List, Optional

import torch.utils.data as tud


class BaseTokenDataset(tud.Dataset):
    def __init__(self, report=None, train: bool = True, seed: int = 0):
        self.report = report
        self.train = train
        self.seed = seed
        self.ids: List[List[int]] = []
        self.tok: Optional[List[List[int]]] = None
        self.codes: List[str] = []
        self.labels: List[int] = []
        self.meta: List[dict] = []

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> dict:
        ids = list(self.ids[index])                 # 整段代码，不截断
        tok = None if self.tok is None else list(self.tok[index])

        if self.report is not None:                 # 报告 token 前缀（整段代码统计，不含标签）
            prefix = self.report.ids(self.codes[index])
            if prefix:
                ids = prefix + ids
                tok = None if tok is None else [-100] * len(prefix) + tok

        return {
            "input_ids": ids,
            "tok_labels": tok,
            "label": int(self.labels[index]),
            "code": self.codes[index],
            "meta": dict(self.meta[index]) if index < len(self.meta) else {},
        }
