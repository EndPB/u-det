"""网络模块：按名称切换（频域 U-Net 主干 + 任务头 + 直接二分类基线）。

    from models import build_hier, SampleHead, TokenHeads, PooledClassifier

    net = build_hier(dim=768, depth=4, mid=8, heads=12)      # 频域 U-Net（v0.2 主干）
    sample_head = SampleHead(768, hidden=256)
    token_heads = TokenHeads([768] * 5)                      # 5 个尺度：L/16 ... L
    clf = PooledClassifier(encoder, pooling="mean")          # 直接二分类基线

新增网络：在 models/ 下新建单文件，并在这里加一个 build_xxx 函数或映射。
"""

from .baseline import PooledClassifier
from .heads import SampleHead, TokenHeads
from .hier import FrequencyUNet, build_hier

__all__ = ["FrequencyUNet", "build_hier", "SampleHead", "TokenHeads", "PooledClassifier"]
