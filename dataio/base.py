"""数据集基类（单文件）：统一"报告前缀 + 窗口裁剪 + 组装"的取数逻辑。

子类只需把 processed parquet 读进以下列表：
    self.ids     每条的代码 token（不含报告）
    self.tok     每条的 token 标签（-100 忽略；样本级数据为 None）
    self.codes   每条原始代码文本（生成报告用）
    self.labels  每条样本级标签
    self.meta    每条附加信息（dict，可选）
"""

from __future__ import annotations

import random
from typing import List, Optional

import torch.utils.data as tud


class BaseTokenDataset(tud.Dataset):
    def __init__(self, report=None, max_length: int = 512, train: bool = True, seed: int = 0):
        self.report = report
        self.max_length = int(max_length)
        self.train = train
        self.seed = seed
        self.epoch = 0                       # 由 train.py 每个 epoch 置一次（窗口随机但可复现）
        self.ids: List[List[int]] = []
        self.tok: Optional[List[List[int]]] = None
        self.codes: List[str] = []
        self.labels: List[int] = []
        self.meta: List[dict] = []

    def __len__(self) -> int:
        return len(self.ids)

    # ------------------------------------------------------------------ #
    def _crop(self, ids: List[int], tok: Optional[List[int]], rng: random.Random):
        """训练时超长样本随机截取一段窗口（评测时返回全量，由评测循环滑窗）。"""
        if not self.train or len(ids) <= self.max_length:
            return ids, tok
        start = rng.randrange(0, len(ids) - self.max_length + 1)
        end = start + self.max_length
        return ids[start:end], None if tok is None else tok[start:end]

    def __getitem__(self, index: int) -> dict:
        ids = list(self.ids[index])
        tok = None if self.tok is None else list(self.tok[index])

        if self.report is not None:           # 报告 token 前缀（整段代码统计，不含标签）
            prefix = self.report.ids(self.codes[index])
            if prefix:
                ids = prefix + ids
                tok = None if tok is None else [-100] * len(prefix) + tok

        rng = random.Random((self.seed, self.epoch, index).__hash__())
        ids, tok = self._crop(ids, tok, rng)
        return {
            "input_ids": ids,
            "tok_labels": tok,
            "label": int(self.labels[index]),
            "code": self.codes[index],
            "meta": dict(self.meta[index]) if index < len(self.meta) else {},
        }
