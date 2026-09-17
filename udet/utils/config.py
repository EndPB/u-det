"""YAML 配置读写工具（不依赖 hydra，轻量、可读）。"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Union

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _prepare_path(cfg: Mapping[str, Any], root: Union[str, Path]) -> dict:
    """把配置中的相对路径（含 ``/`` 的字符串）解析为绝对路径，方便在任意工作目录运行。"""
    root = Path(root)
    out: dict = {}
    for key, value in cfg.items():
        if isinstance(value, MutableMapping):
            out[key] = _prepare_path(value, root)
        elif isinstance(value, str) and ("/" in value) and not value.startswith(("/", "http")):
            out[key] = str(root / value)
        else:
            out[key] = value
    return out


def load_config(
    path: Union[str, Path],
    resolve_paths: bool = True,
    root: Union[str, Path, None] = None,
) -> dict:
    """加载 YAML 配置文件。

    Args:
        path: 配置文件路径。
        resolve_paths: 是否将 ``data/raw`` 这类相对路径转换为绝对路径（相对项目根目录）。
        root: 自定义项目根目录，默认为仓库根目录。

    Returns:
        配置字典。
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"配置文件格式错误（应为字典）: {path}")
    if resolve_paths:
        cfg = _prepare_path(cfg, root or PROJECT_ROOT)
    cfg["_config_path"] = str(path)
    return cfg


def save_config(cfg: Mapping[str, Any], path: Union[str, Path]) -> None:
    """保存配置到 YAML（自动创建父目录）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = {k: v for k, v in cfg.items() if not str(k).startswith("_")}
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(clean, f, allow_unicode=True, sort_keys=False)


def get_by_path(cfg: Mapping[str, Any], key_path: str, default: Any = None) -> Any:
    """按 ``"encoder.name"`` 形式读取嵌套配置。"""
    node: Any = cfg
    for key in key_path.split("."):
        if not isinstance(node, Mapping) or key not in node:
            return default
        node = node[key]
    return node


def deep_update(base: Mapping[str, Any], updates: Mapping[str, Any]) -> dict:
    """递归合并配置（updates 覆盖 base），返回新字典。"""
    out = copy.deepcopy(dict(base))
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out
