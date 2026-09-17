"""网络模块：按名称切换（U-Net 主干 + 任务头 + 瓶颈模块 + 直接二分类基线）。

    from models import build_unet, build_mid, SampleHead, TokenHeads, PooledClassifier

    mid = build_mid("transformer", dim=512, layers=2, heads=8)     # 或 build_mid("none")
    unet = build_unet("unet1d", in_channels=768, features=[64, 128, 256, 512], mid=mid)
    unet2 = build_unet("unet2d", in_channels=3, out_channels=2)    # 图像
    sample_head = SampleHead(512, hidden=256)
    token_heads = TokenHeads(list(reversed([64, 128, 256, 512])))
    clf = PooledClassifier(encoder, pooling="mean")                  # 直接二分类基线

新增网络：在 models/ 下新建单文件，并在这里加一个 build_xxx 函数或映射。
"""

from .baseline import PooledClassifier
from .heads import SampleHead, TokenHeads
from .transformer import MidTransformer, build_mid
from .unet import UNet, build_unet, downsample_mask

__all__ = [
    "UNet", "build_unet", "downsample_mask",
    "MidTransformer", "build_mid",
    "SampleHead", "TokenHeads", "PooledClassifier",
]
