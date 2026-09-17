# U-Det

AI 代码检测（二分类）实验：**逐 token 冻结编码 + 1D U-Net 双任务**。

- 每 token 独立过冻结 CodeT5（长度=1，无上下文）→ 预计算成 `vocab×768` 查表（LUT），
  训练零编码开销、输入长度无上限（绕开 `n_positions=512` 与 attention $O(L^2)$）；
- 1D U-Net 下采样/上采样替代长上下文滑动窗口，瓶颈插 Transformer 做双向语义交互；
- 瓶颈出**样本级**分类；各上采样层出 **token 级**分类（多尺度深监督：标签按尺度池化，
  对应不同文本长度精度，用跳跃连接引导上下采样）；
- 代码前注入**短报告**（结构偏置 / 词汇可预测性 / 句法方差）；
- 数据：CoDET-M4 均衡子集（human 类 + 样本级）+ HybridCodeAuthorship（行级标注 -> token 级引导），
  两流**并行、每个 optimizer step 同时参与**（同一模型、同一次反向与更新）。

## 目录结构

```
u-det/
├── train.py                     # 主干：训练 / 验证 / 评测（双流并行 + 多尺度损失）
├── configs/udet_base.yaml       # U-Det 配置：data / encoder / model / heads / report / loss / train
├── configs/baseline_codet5.yaml # 基线配置：CodeT5（微调）+ 池化 + 线性头（纯样本级）
├── encoders/                    # 编码器（按名称切换，单文件实现）
│   ├── __init__.py              #   build_encoder() 注册表
│   └── codet5.py                #   codet5tok（逐 token 查表，默认）/ codet5（整段上下文）
├── models/                      # 网络（单文件实现）
│   ├── __init__.py              #   build_unet / build_mid / SampleHead / TokenHeads / PooledClassifier
│   ├── unet.py                  #   U-Net（1D/2D，mid 插槽，return_features 给多尺度特征）
│   ├── transformer.py           #   瓶颈 Transformer（双向交互，正弦位置编码）
│   ├── heads.py                 #   样本级头（池化+MLP）/ 多尺度 token 头（1x1 卷积）
│   └── baseline.py              #   直接二分类基线：编码器 + masked 池化 + 线性/MLP 头
├── dataio/                      # 数据（单文件实现）
│   ├── __init__.py              #   build_dataset() 注册表 + collate（动态 padding）
│   ├── base.py                  #   基类：报告前缀 + 窗口裁剪 + 组装
│   ├── m4.py                    #   CoDET-M4 样本级（human/ai）
│   └── hybrid.py                #   HybridCodeAuthorship 行级（token 标签 + 行级评测信息）
├── report/                      # 报告注入
│   ├── __init__.py              #   build_report() 注册表（handcrafted / none）
│   └── handcrafted.py           #   三类手工统计 -> 一行短报告
├── scripts/
│   ├── prepare.py               # 下载数据与权重（m4 / encoder / hybrid）
│   ├── build_subset.py          # 均衡子集 + 预分词 + 划分（写 data/processed/）
│   └── compare.py               # 汇总 runs/ 下的 val/test 指标
├── data/                        # 原始与处理后数据（不入库）
└── checkpoints/                 # 编码器权重与 LUT 缓存（不入库）
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

## 数据准备

```bash
python scripts/prepare.py data       # CoDET-M4 全量：437MB / 500,552 行
python scripts/prepare.py encoder    # CodeT5-base 权重：约 850MB
source /etc/network_turbo            # GitHub 需要代理
python scripts/prepare.py hybrid     # HybridCodeAuthorship：8 个 CSV 约 408MB -> hybrid.parquet

python scripts/build_subset.py       # 均衡子集 + 预分词 + 划分（约 3~5 分钟）
# 产出：data/processed/m4.parquet（human/AI 各 1w）、data/processed/hybrid.parquet（全量 1w+）
```

`build_subset.py` 的均衡策略：长度分桶（token 数，8 档）× 类别（human/AI）× 语言（java/python/cpp），
二次均衡后按 **code 哈希**切 train/val/test = 8:1:1（同一段代码不跨集）；hybrid 按 RecordId 分组切分。

数据集字段：CoDET-M4 用 `cleaned_code`（去注释）作输入，`target` 为 human/ai；
HybridCodeAuthorship 用 `AICode` + 逐行 `Attribution`（AI/Human，映射为 token 级标签）。

## 训练 / 评测

```bash
python train.py --config configs/udet_base.yaml --tag base            # 训练（默认 3 epoch）
python train.py --config configs/udet_base.yaml --eval --ckpt runs/base/best.pt
python train.py --limit 200 --epochs 1 --max-length 512 --tag smoke  # 冒烟
```

产物：`runs/<tag>/{config.yaml, metrics.csv, metrics.jsonl, best.pt}`（best 按 `train.monitor` 选）。

指标：m4 -> 样本级 ACC/F1；hybrid -> 行级 P/R/F1、token 级 F1、片段级 F1（连续 AI 行段按 IoU≥0.5 逐文件匹配）。
超长样本训练时随机截窗，评测时滑窗（stride = max_length/2）合并。

### 首版基线（3 epoch，约 8 分钟，9.19M 可训练参数）

| 数据集/任务 | 指标 | val | test |
| --- | --- | --- | --- |
| m4 样本级（1926/2001 条） | ACC / F1 | 0.850 / 0.864 | **0.849 / 0.867** |
| hybrid 行级（797/883 条） | P / R / F1 | 0.494 / 0.426 / 0.458 | **0.496 / 0.454 / 0.474** |
| hybrid token 级 | F1 | 0.513 | 0.547 |
| hybrid 片段级 | F1 | 0.130 | 0.127 |

片段级 F1 偏低是本架构下一步的主要改进点（预测碎片化），可尝试：token 概率滑动平滑后再阈值、
片段级后处理（合并/最小段长）、或调大 `loss.scale_weights` 里粗尺度的权重。

## 直接二分类基线（CodeT5 微调 + 池化）

```bash
# 基线：整段代码过 CodeT5（微调）-> masked 池化 -> 线性头（纯样本级二分类）
python train.py --config configs/baseline_codet5.yaml --tag base_codet5_ft
python train.py --config configs/baseline_codet5.yaml --eval --ckpt runs/base_codet5_ft/best.pt
# 带报告：--report handcrafted --max-length 448（448 + 53 报告 ≤ 512）
# 冻结对照：--freeze-encoder --lr 1e-3
python scripts/compare.py        # 汇总 runs/ 下所有实验的 val/test 指标
```

### 对比（同一份 m4 子集、同样 3 epoch / 1004 step、L=512、bf16；test 集）

| 实验 | 模型 | 编码器 | m4 ACC | m4 F1 | hybrid 行级 F1 | hybrid token F1 |
| --- | --- | --- | --- | --- | --- | --- |
| `base_codet5_ft` | CodeT5 + 池化 + 线性 | codet5（微调 109.6M） | **0.975** | **0.976** | — | — |
| `udet_m4` | U-Det（样本级单任务） | codet5tok（LUT 可训练 24.7M） | 0.948 | 0.949 | — | — |
| `base_lut` | U-Det（样本级+token 级） | codet5tok（LUT 可训练） | 0.859 | 0.874 | **0.542** | **0.621** |
| `base`（v0.1） | U-Det（样本级+token 级） | codet5tok（LUT 冻结） | 0.849 | 0.867 | 0.474 | 0.547 |

结论：
- **样本级**任务上，微调上下文编码器 + 池化明显强于 U-Det（0.976 vs 0.949）——逐 token 无上下文特征
  在单任务上信息不足；
- **token 级 / 片段级**任务只有 U-Det 能做（基线受 512 位置限制，无法直接输出 token 级标注）；
- **不冻结编码器**对 U-Det 很关键：行级 F1 0.474 → 0.542、token F1 0.547 → 0.621；
- 加了 hybrid 流后 m4 样本级会下降（用 `--no-hybrid` 可回到 0.949），因为 hybrid 的样本级标签恒为 AI
  且长序列 token 任务占主导。

## 消融开关（都在 yaml 里，或用命令行覆盖）

| 开关 | 取值 | 说明 |
| --- | --- | --- |
| `encoder.name` | `codet5tok` / `codet5` | 逐 token 查表 / 整段上下文编码 |
| `encoder.freeze` | `false` / `true` | false = 微调编码器（codet5tok 时 LUT 变可训练嵌入表） |
| `train.lr_encoder` | 如 `1e-4` / `2e-5` | 编码器单独学习率（参数分组） |
| `model.name` | `unet1d` / `codet5cls` | U-Det / 直接二分类基线（`--model`） |
| `model.mid.name` | `transformer` / `none` | 瓶颈是否做双向交互（`--mid none`） |
| `report.name` | `handcrafted` / `none` | 是否注入手工统计报告（`--report none`） |
| `loss.token` | 0 / 1.0 | token 级监督权重（0 = 纯样本级） |
| `train.streams` | `[m4, hybrid]` | 双流；`--no-m4` / `--no-hybrid` 单流 |
| `model.features` | 如 `[64,128,256,512]` | U-Net 深度/宽度（len-1 = 下采样次数） |

## 实现要点

- **逐 token 编码器**（`encoders/codet5.py:CodeT5TokenEncoder`）：分块跑冻结编码器得到 `(vocab, D)` 查表，
  首次约 2.4s 并缓存到 `data/processed/codet5_lut.pt`，之后启动直接加载（不加载 850MB 权重）。
- **多尺度损失**（`train.py:token_loss`）：各解码层 logits 长度为 $L/2^k$，标签用 mask-aware 平均池化
  到对应长度作为软标签（不同文本长度精度），权重 `loss.scale_weights` 由粗到细。
- **报告注入**（`report/handcrafted.py`）：空行率/缩进一致性/尾部空行 + 字符与二元组香农熵 +
  命名惯例多样性/行长方差，渲染成一行短文本（数值量化，≤`report.max_tokens`），拼在代码 token 前，
  报告位置不计 token 损失（-100）。
- **行级标签对齐**（`dataio/hybrid.py`）：整段分词（fast tokenizer + offset mapping），
  每个 token 取起始字符所在行的 `Attribution`（AI=1 / Human=0）。

## 编码器 / 网络 / 数据 / 报告切换接口

```python
from encoders import build_encoder, list_encoders      # ['codet5', 'codet5tok']
from models import build_unet, build_mid               # 'unet1d'/'unet2d'，'none'/'transformer'
from dataio import build_dataset, collate              # 'm4' / 'hybrid'
from report import build_report                        # 'handcrafted' / 'none'
```

新增编码器 / 网络 / 数据集 / 报告：各自目录下新建单文件，在对应注册表里加一行映射即可。

## 备注（AutoDL）

- HuggingFace 直连不通 → `scripts/prepare.py` 默认走 hf-mirror；GitHub 需 `source /etc/network_turbo`；
- pip 走阿里云镜像；开代理后 pip 更慢，装依赖前 `unset http_proxy https_proxy`；
- 本机 `OMP_NUM_THREADS` 环境变量异常（libgomp 报错），运行时建议显式 `OMP_NUM_THREADS=8`；
- 逐 token 查表后显存占用很低：L=2048 / batch=4 前向+反向峰值约 0.35GB（12GB 卡可再加长）。
- CoDET-M4 原始 parquet 自带 `split` 列（train 37.4w / val 4.4w / test 4.4w），当前子集
  **未采用**（按 code 哈希重切，防跨集泄漏）；如需对齐官方划分，可在 `build_subset.py` 里保留该列并改用。

## 数据与许可

- 数据集：CoDET-M4（MIT，<https://huggingface.co/datasets/DaniilOr/CoDET-M4>）
- 数据集：HybridCodeAuthorship（Apache-2.0，<https://github.com/CapitalOne-Research/c1-hybrid-code-authorship>）
- 编码器权重：Salesforce/codet5-base（BSD-3-Clause）
