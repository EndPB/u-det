"""模型模块：U-Net（1D / 2D）及统一构建入口。

    from udet.models import build_unet, list_unets

    unet = build_unet("unet1d", in_channels=768, out_channels=2, features=[64, 128, 256, 512])

新增自定义网络时，用 ``@register_unet("name")`` 注册即可被配置系统识别。
"""

from __future__ import annotations

from typing import Callable, Dict, List, Type, Union

import torch.nn as nn

from .unet1d import UNet1D, build_unet1d
from .unet2d import UNet2D, build_unet2d

UNET_REGISTRY: Dict[str, Type[nn.Module]] = {
    "unet1d": UNet1D,
    "unet2d": UNet2D,
}


def register_unet(name: str) -> Callable[[Type[nn.Module]], Type[nn.Module]]:
    """类装饰器：注册自定义网络结构。"""

    def decorator(cls: Type[nn.Module]) -> Type[nn.Module]:
        if name in UNET_REGISTRY and UNET_REGISTRY[name] is not cls:
            raise KeyError(f"网络名称冲突: {name!r}")
        UNET_REGISTRY[name] = cls
        return cls

    return decorator


def list_unets() -> List[str]:
    return sorted(UNET_REGISTRY.keys())


def build_unet(cfg: Union[str, dict] = None, name: str = None, **overrides) -> nn.Module:
    """统一构建入口，用法与 :func:`udet.encoders.build_encoder` 一致。

        build_unet("unet1d", in_channels=768, out_channels=2)
        build_unet({"name": "unet2d", "in_channels": 3, "out_channels": 2})
        build_unet(config["model"])
    """
    if isinstance(cfg, str):
        cfg = {"name": cfg, **overrides}
    elif cfg is None:
        cfg = {"name": name or "unet1d", **overrides}
    else:
        cfg = {**cfg, **overrides}

    cfg = dict(cfg)
    unet_name = cfg.pop("name", None) or name
    if unet_name is None:
        raise ValueError("必须通过 cfg['name'] 或 name 参数指定网络名称")
    if unet_name not in UNET_REGISTRY:
        raise KeyError(f"未知网络 {unet_name!r}，可选: {list_unets()}")

    cls = UNET_REGISTRY[unet_name]
    return cls(**cfg)


__all__ = [
    "UNet1D",
    "UNet2D",
    "build_unet1d",
    "build_unet2d",
    "build_unet",
    "register_unet",
    "list_unets",
    "UNET_REGISTRY",
]
