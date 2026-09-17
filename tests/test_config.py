"""配置文件与数据工具的小型测试（不依赖数据集下载）。"""

from __future__ import annotations

from pathlib import Path

from udet.utils import deep_update, get_by_path, load_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_load_default_config():
    cfg = load_config(PROJECT_ROOT / "configs" / "udet_base.yaml")
    assert cfg["encoder"]["name"] == "codet5"
    assert cfg["model"]["name"] == "unet1d"
    # 相对路径已解析为绝对路径
    assert Path(cfg["data"]["raw_dir"]).is_absolute()
    assert cfg["data"]["raw_dir"].endswith("data/raw/CoDET-M4")


def test_get_by_path_and_deep_update():
    cfg = {"a": {"b": 1, "c": {"d": 2}}}
    assert get_by_path(cfg, "a.b") == 1
    assert get_by_path(cfg, "a.c.d") == 2
    assert get_by_path(cfg, "a.x", default="fallback") == "fallback"

    merged = deep_update(cfg, {"a": {"b": 9, "e": 3}})
    assert merged["a"]["b"] == 9
    assert merged["a"]["c"]["d"] == 2
    assert merged["a"]["e"] == 3
    assert cfg["a"]["b"] == 1  # 原配置不被修改
