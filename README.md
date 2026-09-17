# U-Det

CodeT5 编码器 + U-Net 的代码检测实验。

当前为初始化阶段：数据（CoDET-M4 全量）、CodeT5 权重、U-Net 骨架、编码器切换模块已就绪，训练/评测代码待后续迭代。

## 目录结构

```
u-det/
├── configs/udet_base.yaml     # 配置：数据 / 编码器 / 网络
├── encoders/                  # 编码器模块（单文件实现，按名称切换）
│   ├── __init__.py            #   build_encoder() 切换入口 + 注册表
│   └── codet5.py              #   CodeT5 编码器（默认）
├── models/                    # 网络模块（单文件实现）
│   ├── __init__.py            #   build_unet() 切换入口
│   └── unet.py                #   U-Net（1D 序列 / 2D 图像通用，待魔改）
├── scripts/prepare.py         # 单文件脚本：下载数据与编码器权重
├── data/raw/CoDET-M4/         # 数据集（不入库）
└── checkpoints/codet5-base/   # 编码器权重（不入库）
```

## 环境

| 组件 | 版本 |
| --- | --- |
| Python | 3.12 |
| torch | 2.9.1（PyPI 默认 wheel 自带 CUDA 12.8） |
| transformers | 4.57.x |
| GPU 目标 | RTX 3080 Ti 12GB / CUDA 12.8 驱动 |

```bash
conda create -n udet python=3.12 -y
conda activate udet
pip install -r requirements.txt
```

## 数据与权重准备

```bash
python scripts/prepare.py data       # CoDET-M4 全量：437MB / 500,552 行 / 单 parquet
python scripts/prepare.py encoder    # CodeT5-base 权重：约 850MB（含分词器文件）
```

默认走 `hf-mirror.com` 镜像（国内直连可用）；如需官方源，先 `source /etc/network_turbo`，
再加 `--endpoint https://huggingface.co`。

数据集字段：`code`（原始）、`cleaned_code`（去注释，推荐输入）、`language`、`model`、`target`、
`source`、`split`、`features`（8 个手工统计特征）。

## 编码器切换

```python
from encoders import build_encoder, list_encoders

list_encoders()                                         # ['codet5']
encoder = build_encoder("codet5", path="checkpoints/codet5-base")

hidden = encoder(input_ids, attention_mask)                              # (B, L, 768)
layers = encoder(input_ids, attention_mask, output_hidden_states=True)   # 各层隐状态（可选）
```

- 统一接口：`hidden_size` 属性 + `forward(input_ids, attention_mask) -> (B, L, D)`；
- 支持 `freeze=True` / `dtype="bfloat16"` / `gradient_checkpointing=True`；
- **新增编码器**：在 `encoders/` 下新建单文件，然后在 `encoders/__init__.py` 的 `ENCODERS` 中加一行。

## U-Net

```python
from models import build_unet

unet = build_unet("unet1d", in_channels=768, out_channels=2, features=[64, 128, 256, 512])
logits = unet(hidden.transpose(1, 2), mask=attention_mask)     # (B, 2, L)
```

- 同一个 `UNet` 类支持 `dim=1`（序列，输入 `(B,C,L)`）与 `dim=2`（图像，输入 `(B,C,H,W)`）；
- 奇数长度 / 非 2 次幂尺寸自动对齐；序列版可用 `mask` 把 padding 位置输出置零；
- `deep_supervision=True` 返回多尺度 logits 列表；`return_features=True` 返回各解码层特征；
- **魔改入口**：`DoubleConv`（卷积块）、`Down`（下采样）、`Up`（跳连）、`UNet.forward`（整体）。

## 自检

```bash
python - <<'PY'
import torch, yaml
from encoders import build_encoder
from models import build_unet

cfg = yaml.safe_load(open("configs/udet_base.yaml"))
encoder = build_encoder(**cfg["encoder"])                  # encoder.name = codet5
unet = build_unet(**cfg["model"])                          # model.name = unet1d

ids = torch.randint(0, 1000, (2, 131))                     # 奇数长度也支持
mask = torch.ones_like(ids)
hidden = encoder(ids, mask)
logits = unet(hidden.transpose(1, 2), mask=mask)
print("encoder:", tuple(hidden.shape), "-> unet:", tuple(logits.shape))
PY
```

无 GPU 时同样可以跑（自动在 CPU 上计算）。

## 备注（AutoDL）

- HuggingFace 直连不通 → 使用 hf-mirror（`scripts/prepare.py` 默认）；
- GitHub 操作需先 `source /etc/network_turbo`；
- 无卡实例只有 2GB 内存：读大数据请用 memory-map（如 `pyarrow` / `datasets`），避免一次性全量载入；
- 切到有卡模式后验证 GPU：
  `python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"` → 期望 `2.9.1 12.8 True`。

## 数据与许可

- 数据集：CoDET-M4（MIT License，<https://huggingface.co/datasets/DaniilOr/CoDET-M4>）
- 编码器权重：Salesforce/codet5-base（BSD-3-Clause）
