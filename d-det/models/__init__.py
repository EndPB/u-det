"""模型模块：按名称构建（新增主干 = 新建单文件 + MODELS 加一行）。

d-det 首版只有 ``dual``（轻量双分数模型：编码器 → 池化 → 双读出）。
后续主干对照（如 u-det 的 codec 多尺度主干）按同一接口扩展：
``forward(input_ids, attention_mask) -> (s1, s2)``。
"""

from .scores import DualScoreModel, GatedAttentionPool
from .disc import DiscHead

# 模型注册表：名称 -> 类（切换主干只需要改这里的 name）
MODELS = {
    "dual": DualScoreModel,
}


def list_models():
    """列出所有可用模型名称。"""
    return sorted(MODELS)


def build_model(name: str = "dual", **kwargs):
    """按名称构建模型（cfg 里的 model.name 会传到这里）。"""
    if name not in MODELS:
        raise KeyError(f"未知模型 {name!r}，可选：{list_models()}")
    return MODELS[name](**kwargs)
