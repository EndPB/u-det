"""编码器模块：统一接口 + 注册表 + 具体实现。

切换编码器只需要改配置 / 改一个名字::

    from udet.encoders import build_encoder, build_tokenizer, list_encoders

    print(list_encoders())                       # ['codet5', 'hf']
    encoder = build_encoder("codet5", model_name_or_path="checkpoints/codet5-base")
    tokenizer = build_tokenizer("checkpoints/codet5-base")
"""

from .base import BaseEncoder, EncoderOutput, freeze_module
from .registry import (
    ENCODER_REGISTRY,
    build_encoder,
    get_encoder_cls,
    list_encoders,
    register_encoder,
)
from .tokenizer import build_tokenizer

# 导入具体实现以触发注册（顺序无关）
from . import codet5  # noqa: F401,E402
from . import hf_encoder  # noqa: F401,E402

__all__ = [
    "BaseEncoder",
    "EncoderOutput",
    "freeze_module",
    "ENCODER_REGISTRY",
    "register_encoder",
    "build_encoder",
    "get_encoder_cls",
    "list_encoders",
    "build_tokenizer",
    "codet5",
    "hf_encoder",
]
