"""网络模块：按名称切换（v0.2 频域 U-Net / v0.3 Transformer 编解码器 + 任务头 + 位置探针 + 直接二分类基线）。

    from models import build_hier, SampleHead, TokenHeads, PooledClassifier, PositionProbes

    net = build_hier("hier", dim=768, depth=4, mid=8, heads=12)      # v0.2 频域 U-Net
    net = build_hier("codec", dim=768, depth=4, mid=4, heads=12)     # v0.3/v0.4 窗口注意力编解码器
    sample_head = SampleHead(768, hidden=256)
    token_heads = TokenHeads([768] * 5)                          # 5 个尺度
    probes = PositionProbes(768, spans=(4, 16, 32, 64))          # v0.4 下采样位置探针
    clf = PooledClassifier(encoder, pooling="mean")              # 直接二分类基线

新增网络：在 models/ 下新建单文件，并在这里加一个 build_xxx 函数或映射。
"""

from .baseline import PooledClassifier
from .heads import SampleHead, TokenHeads
from .hier import FrequencyUNet, build_hier as _build_frequnet
from .hier2 import TransformerCodec, build_codec
from .probes import PositionProbe, PositionProbes, position_targets


def build_hier(name: str = "hier", **kwargs):
    """按名称构建主干：``hier``/``frequnet``（v0.2 频域 U-Net）/ ``codec``（v0.3 Transformer 编解码器）。"""
    if name in ("hier", "frequnet"):
        return _build_frequnet(name, **kwargs)
    if name in ("codec", "hier2", "transformer"):
        return build_codec(name, **kwargs)
    raise KeyError(f"未知主干 {name!r}，可选：['hier', 'codec']")


__all__ = ["FrequencyUNet", "TransformerCodec", "build_hier", "build_codec",
           "SampleHead", "TokenHeads", "PooledClassifier",
           "PositionProbe", "PositionProbes", "position_targets"]
