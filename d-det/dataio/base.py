"""数据集基类（单文件）：公共容器 + 取数约定。

约定（与 u-det 一致）：**不截断** —— 样本整段进入模型（编码器分块处理，长度无上限）；
训练 / 评测恒 batch=1，用梯度累积凑有效批大小。
"""

from __future__ import annotations

from typing import List

import torch.utils.data as tud


class BaseDataset(tud.Dataset):
    """公共字段：ids / codes / labels / meta。子类负责填充与 __getitem__。"""

    def __init__(self, train: bool = True, seed: int = 0):
        self.train = train
        self.seed = seed
        self.ids: List[List[int]] = []
        self.codes: List[str] = []
        self.labels: List[int] = []
        self.meta: List[dict] = []

    def __len__(self) -> int:
        return len(self.ids)
