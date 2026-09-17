# U-Det

基于 **CodeT5 编码器 + U-Net** 的代码检测（token 级）实验框架。

- 编码器可插拔：内置 CodeT5（默认），也可一键切换到任意 HuggingFace 编码器（BERT / GraphCodeBERT / T5 系…）
- U-Net 现成可用：1D（token 序列）与 2D（图像）两个版本，结构清晰，便于后续魔改
- 数据：CoDET-M4（50 万条代码样本，含 `cleaned_code` 去注释代码与 8 个手工特征）
- 环境：Python 3.12 + PyTorch (CUDA 12.8)，适配 RTX 3080 Ti（12 GB）

> 当前仓库为**初始化阶段**：数据、编码器权重、网络骨架、工程结构已就绪，训练/评测代码待后续迭代。

---

## 1. 快速开始

### 1.1 创建环境

```bash
conda create -n udet python=3.12 -y
conda activate udet
pip install -r requirements.txt          # 已配置阿里云 pip 镜像
```

依赖要点：

| 组件 | 版本 | 说明 |
| --- | --- | --- |
| Python | 3.12 | 与 AutoDL 基础镜像一致 |
| torch | 2.9.1 | PyPI 默认 wheel 自带 **CUDA 12.8** 运行时，适配 RTX 3080 Ti |
| transformers | 4.57.x | 最后的 4.x 稳定线，CodeT5 兼容性最佳 |

> 无卡实例（CPU 模式）同样可以安装：torch 的 cu128 wheel 在 CPU 上正常运行，切到有卡模式即可直接使用 GPU。

### 1.2 下载数据与权重

```bash
python scripts/download_data.py       # CoDET-M4 全量（~437MB，默认走 hf-mirror 镜像）
python scripts/prepare_data.py        # 统计 + 划分 train/val/test
python scripts/download_encoder.py    # CodeT5-base 权重（~0.9GB）
```

### 1.3 冒烟测试

```bash
python scripts/smoke_test.py          # 编码器 + U-Net 全链路形状自检（CPU 可跑）
python scripts/smoke_test.py --tiny   # 无权重时用迷你随机模型快速验证
pytest -q                             # 单元测试
```

---

## 2. 目录结构

```
u-det/
├── configs/
│   └── udet_base.yaml          # 统一配置（数据 / 编码器 / 网络 / 训练）
├── data/                       # 数据目录（不入 git，见 data/README.md）
│   ├── raw/CoDET-M4/           # 原始 parquet
│   └── processed/              # 划分后的 arrow 数据集
├── checkpoints/                # 编码器权重（不入 git，见 checkpoints/README.md）
│   └── codet5-base/
├── scripts/
│   ├── download_data.py        # 下载 CoDET-M4
│   ├── prepare_data.py         # 数据统计与划分
│   ├── download_encoder.py     # 下载编码器权重
│   └── smoke_test.py           # 端到端形状自检
├── udet/                       # 主包
│   ├── encoders/               # ★ 编码器：统一接口 + 注册表 + 实现
│   │   ├── base.py             #   BaseEncoder / EncoderOutput
│   │   ├── registry.py         #   build_encoder / register_encoder
│   │   ├── codet5.py           #   CodeT5 封装
│   │   ├── hf_encoder.py       #   任意 HF 编码器封装
│   │   └── tokenizer.py        #   分词器
│   ├── models/                 # ★ 网络：U-Net
│   │   ├── blocks.py           #   DoubleConv / Down / Up / OutConv
│   │   ├── unet1d.py           #   序列 U-Net（token 级）
│   │   └── unet2d.py           #   经典 2D U-Net
│   ├── data/codet_m4.py        # 数据集加载 / 划分 / 分词
│   └── utils/                  # 配置、种子、参数量等
└── tests/                      # 单元测试（CPU 可跑）
```

---

## 3. 编码器：一键切换

所有编码器实现统一接口 `BaseEncoder`（输入 `input_ids` / `attention_mask`，输出 `EncoderOutput`），
通过注册表按名字构建：

```python
from udet.encoders import build_encoder, build_tokenizer, list_encoders

print(list_encoders())                       # ['codet5', 'hf']

# 方式一：CodeT5（默认，推荐）
encoder = build_encoder("codet5", model_name_or_path="checkpoints/codet5-base")
tokenizer = build_tokenizer("checkpoints/codet5-base")

# 方式二：任意 HF 编码器（自动处理 T5 等 encoder-decoder 模型）
encoder = build_encoder("hf", model_name_or_path="microsoft/graphcodebert-base")

# 方式三：直接读配置
from udet.utils import load_config
cfg = load_config("configs/udet_base.yaml")
encoder = build_encoder(cfg["encoder"])
```

`EncoderOutput` 包含：

| 字段 | 形状 | 说明 |
| --- | --- | --- |
| `last_hidden_state` | `(B, L, D)` | token 级表示，喂给 U-Net |
| `hidden_states` | `list[(B, L, D)]` | 各层隐状态（可作多尺度跳连） |
| `attention_mask` | `(B, L)` | 供 mask 处理 |
| `pooled_output` | `(B, D)` | 句级平均池化 |

常用能力：`freeze()` / `freeze_layers(n)` / `layer_selection="all"|[0,3,6,9]` /
`torch_dtype="bfloat16"` / `gradient_checkpointing=True`。

新增编码器：

```python
from udet.encoders.registry import register_encoder
from udet.encoders.base import BaseEncoder

@register_encoder("my_encoder")
class MyEncoder(BaseEncoder):
    @property
    def hidden_size(self) -> int: ...
    def forward(self, input_ids=None, attention_mask=None, **kw): ...
```

---

## 4. U-Net：现成可魔改

### 4.1 序列版 `unet1d`（默认）

```python
from udet.models import build_unet

unet = build_unet("unet1d", in_channels=768, out_channels=2, features=[64, 128, 256, 512])

feats = encoder_out.last_hidden_state.transpose(1, 2)   # (B, 768, L)
logits = unet(feats, mask=encoder_out.attention_mask)   # (B, 2, L)
```

特点：

- 奇数长度序列自动对齐（`ceil_mode` 池化 + 插值），不会因长度不对而报错；
- `mask` 传入后自动把 padding 位置输出置零；
- `deep_supervision=True` 返回多尺度 logits 列表，可直接做深监督损失；
- `return_features=True` 返回各解码层特征，方便接额外的头。

### 4.2 经典 2D 版 `unet2d`

```python
unet2d = build_unet("unet2d", in_channels=3, out_channels=5, features=[64, 128, 256, 512], bilinear=False)
y = unet2d(x)          # (B, 5, H, W)，任意 H/W 自动对齐
```

### 4.3 魔改入口

| 想改什么 | 到哪里改 |
| --- | --- |
| 卷积块 / 归一化 / 激活 | `udet/models/blocks.py` |
| 跳连方式（注意力、门控…） | `unet1d.py` 的 `Up1d` / `unet2d.py` 的 `Up2d` |
| 输出头（多任务、CRF…） | `unet1d.py` 的 `outc` / `forward` |
| 整体结构 | 新写一个类并用 `@register_unet("name")` 注册 |

---

## 5. 配置

`configs/udet_base.yaml` 涵盖数据、编码器、网络、训练四部分；
`load_config()` 会自动把 `data/raw` 这类相对路径解析为**绝对路径**，因此可以在任意目录运行脚本。

```bash
python - <<'PY'
from udet.utils import load_config
from udet.encoders import build_encoder
from udet.models import build_unet

cfg = load_config("configs/udet_base.yaml")
encoder = build_encoder(cfg["encoder"])
unet = build_unet(cfg["model"], in_channels=encoder.hidden_size)
print(encoder, unet)
PY
```

---

## 6. 常见问题（AutoDL 环境）

| 问题 | 解决 |
| --- | --- |
| HuggingFace 下载慢/失败 | 脚本默认使用 `hf-mirror.com`；也可 `source /etc/network_turbo` 后 `--endpoint https://huggingface.co` |
| 需要访问 GitHub | `source /etc/network_turbo`（学术加速，仅加速 GitHub/HF） |
| 无卡实例内存小（2 GB） | `prepare_data.py` 采用分批统计；加载大数据用 memory-map（arrow）而非 pandas 全量读 |
| 切到有卡模式后 torch 报 CUDA 版本错误 | `python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"` 应输出 `2.9.1 12.8 True` |
| 显存不够（3080 Ti 12 GB） | 开启 `gradient_checkpointing=True`、降低 `max_length`、`torch_dtype="bfloat16"`（Ampere 支持） |

---

## 7. 路线图

- [x] 工程骨架 + conda 环境（Python 3.12 / CUDA 12.8）
- [x] CoDET-M4 全量数据下载与划分
- [x] 编码器模块（CodeT5 权重就绪，支持切换）
- [x] U-Net（1D / 2D）与单元测试
- [ ] 训练 / 验证循环（魔改 U-Net 后接入）
- [ ] 评测指标与可视化
- [ ] 消融实验与配置管理

---

## 8. 数据与许可

- 数据集：CoDET-M4（MIT License，见 `data/README.md`）
- 编码器权重：Salesforce/codet5-base（BSD-3-Clause）
- 本项目代码：MIT
