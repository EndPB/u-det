"""编码器注册表：按名称构建 / 列举编码器。

用法::

    from udet.encoders import build_encoder, list_encoders

    print(list_encoders())              # ['codet5', 'hf', ...]
    enc = build_encoder("codet5", model_name_or_path="checkpoints/codet5-base")

新增编码器::

    from udet.encoders.registry import register_encoder

    @register_encoder("my_encoder")
    class MyEncoder(BaseEncoder):
        ...
"""

from __future__ import annotations

from typing import Callable, Dict, List, Type, Union

from .base import BaseEncoder

# 注册表：名称 -> 编码器类
ENCODER_REGISTRY: Dict[str, Type[BaseEncoder]] = {}


def register_encoder(name: str) -> Callable[[Type[BaseEncoder]], Type[BaseEncoder]]:
    """类装饰器：将编码器注册到全局注册表。"""

    def decorator(cls: Type[BaseEncoder]) -> Type[BaseEncoder]:
        if not issubclass(cls, BaseEncoder):
            raise TypeError(f"{cls.__name__} 必须继承 BaseEncoder")
        if name in ENCODER_REGISTRY and ENCODER_REGISTRY[name] is not cls:
            raise KeyError(f"编码器名称冲突: {name!r} 已注册为 {ENCODER_REGISTRY[name].__name__}")
        ENCODER_REGISTRY[name] = cls
        return cls

    return decorator


def list_encoders() -> List[str]:
    """返回已注册的编码器名称列表。"""
    return sorted(ENCODER_REGISTRY.keys())


def get_encoder_cls(name: str) -> Type[BaseEncoder]:
    if name not in ENCODER_REGISTRY:
        raise KeyError(f"未知编码器 {name!r}，可选: {list_encoders()}")
    return ENCODER_REGISTRY[name]


def build_encoder(cfg: Union[str, dict] = None, name: str = None, **overrides) -> BaseEncoder:
    """构建编码器。

    支持三种调用方式::

        build_encoder("codet5", model_name_or_path="checkpoints/codet5-base")
        build_encoder({"name": "codet5", "model_name_or_path": "..."})
        build_encoder(cfg["encoder"])          # 直接传配置字典

    配置字典中的 ``name`` 字段用于选择编码器，其余字段作为关键字参数传入构造器；
    ``overrides`` 中的参数优先级最高。
    """
    if isinstance(cfg, str):
        cfg = {"name": cfg, **overrides}
    elif cfg is None:
        cfg = {"name": name or "codet5", **overrides}
    else:
        cfg = {**cfg, **overrides}

    cfg = dict(cfg)
    enc_name = cfg.pop("name", None) or name
    if enc_name is None:
        raise ValueError("必须通过 cfg['name'] 或 name 参数指定编码器名称")

    cls = get_encoder_cls(enc_name)
    return cls(**cfg)
