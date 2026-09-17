"""编码器模块：按名称一键切换。

内置两种（对应同一套 CodeT5 权重）：

    codet5tok  逐 token 独立编码（长度=1）→ 预计算 (vocab, D) 查表；
               训练零编码开销、长度无上限，U-Det 默认。
    codet5     上下文编码（整段进编码器，受 512 位置 / attention 限制）。

新增编码器只需两步：
    1) 在 encoders/ 下新建一个单文件（如 unixcoder.py），实现一个 nn.Module：
         - 属性 hidden_size
         - forward(input_ids, attention_mask=None, ...) -> (B, L, D)
    2) 在下面 ENCODERS 字典里加一行映射。

用法::

    from encoders import build_encoder, list_encoders

    print(list_encoders())                                   # ['codet5', 'codet5tok']
    encoder = build_encoder("codet5tok", path="checkpoints/codet5-base")
"""

from .codet5 import CodeT5Encoder, CodeT5TokenEncoder

# 编码器注册表：名称 -> 类（切换编码器只需要改这里的 name）
ENCODERS = {
    "codet5": CodeT5Encoder,
    "codet5tok": CodeT5TokenEncoder,
}


def list_encoders():
    """列出所有可用编码器名称。"""
    return sorted(ENCODERS)


def build_encoder(name: str = "codet5", **kwargs):
    """按名称构建编码器（cfg 里的 encoder.name 会传到这里）。"""
    if name not in ENCODERS:
        raise KeyError(f"未知编码器 {name!r}，可选：{list_encoders()}")
    return ENCODERS[name](**kwargs)
