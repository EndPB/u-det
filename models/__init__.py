"""网络模块：按名称切换（当前为 U-Net 单文件实现）。

    from models import build_unet

    unet = build_unet("unet1d", in_channels=768, out_channels=2)   # 序列
    unet = build_unet("unet2d", in_channels=3, out_channels=2)     # 图像

新增网络：在 models/ 下新建单文件，并在这里加一个 build_xxx 函数或映射。
"""

from .unet import UNet, build_unet

__all__ = ["UNet", "build_unet"]
