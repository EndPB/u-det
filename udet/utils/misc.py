"""杂项工具：随机种子、参数量统计、格式化。"""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    """固定所有随机种子，保证实验可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def human_format(num: float) -> str:
    """把大数字格式化为 ``1.2M`` / ``3.4B`` 形式。"""
    for unit in ["", "K", "M", "B", "T"]:
        if abs(num) < 1000:
            return f"{num:3.1f}{unit}".strip()
        num /= 1000.0
    return f"{num:.1f}P"


def count_parameters(module: torch.nn.Module, trainable_only: bool = False) -> int:
    """统计模型参数量。"""
    params = module.parameters()
    if trainable_only:
        return sum(p.numel() for p in params if p.requires_grad)
    return sum(p.numel() for p in params)


def get_device(prefer: str = "auto") -> torch.device:
    """选择设备；无卡实例上会自动回落到 CPU。"""
    if prefer == "cpu":
        return torch.device("cpu")
    if prefer == "cuda" or prefer == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if prefer == "cuda":
            print("[get_device] CUDA 不可用，回落到 CPU。")
        return torch.device("cpu")
    return torch.device(prefer)
