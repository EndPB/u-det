# U-Det

AI 代码检测实验：**逐 token 无上下文编码 + 序列级编解码主干 + 样本级/token 级双任务**。

- 每 token 独立过 CodeT5（长度=1，无跨 token attention）→ 特征对长度无假设，**整段代码不截断**；
- 主干把"下采样/上采样"做成**可学算子**，用多级压缩替代长上下文滑动窗口；
- **样本级**分类接在主干最底部（瓶颈）；各上采样层出 **token 级**分类（多尺度深监督）；
- 代码前注入**短报告**（结构偏置 / 词汇可预测性 / 句法方差）；
- 数据：CoDET-M4（样本级）+ HybridCodeAuthorship（行级标注 → token 级），两流**并行**、
  每个 optimizer step 同时参与。

当前默认版本为 **v0.3**（`configs/udet_v03.yaml`）。

## 版本演进与结果（test 集，同一份数据/切分）

| 版本 | 主干 | 编码器 | m4 sample F1 | hybrid line F1 | chunk F1 | token F1 |
| --- | --- | --- | --- | --- | --- | --- |
| 基线 | — | CodeT5 整段上下文（**微调**） | **0.9756** | — | — | — |
| 基线 | — | CodeT5 整段上下文（冻结） | 0.7651 | — | — | — |
| v0.1 | 卷积 U-Net（窗口训练 + 滑窗评测） | 查表 LUT（冻结） | 0.8665 | 0.4740 | 0.1269 | 0.5465 |
| v0.1 | 同上 | 查表 LUT（可训练） | 0.8744 | 0.5416 | 0.1499 | 0.6208 |
| v0.2 | 频域 U-Net（FFT 上下采样） | 查表 LUT（可训练） | 0.7686 | 0.6341 | 0.3094 | 0.6666 |
| v0.2 | 同上 | CodeT5 + **peft LoRA** | 0.7705 | 0.6535 | **0.3698** | 0.6801 |
| **v0.3** | **窗口注意力编解码器** | CodeT5 + peft LoRA | **0.8863** | 0.6462 | 0.3056 | **0.6864** |

- 主干参数量：v0.1 约 9M（LUT 冻结）/ v0.2 **122.35M** / **v0.3 42.52M**（栈内 4 级共享权重）。
- 逐版本设计说明与逐组件参数量见 `docx/u-det-v0.1.md`、`docx/u-det-v0.2.md`、`docx/u-det-v0.3.md`；
  总设计见 `docx/u-det.md`。

## 目录结构

```
u-det/
├── train.py                     # 主干：训练 / 验证 / 评测（双流并行 + 多尺度损失 + 梯度累积）
├── configs/
│   ├── udet_v03.yaml            # ★ 当前默认：窗口注意力编解码器（v0.3）
│   ├── udet_lora.yaml           # v0.2 频域 U-Net + CodeT5 LoRA
│   └── udet_base.yaml           # v0.2 频域 U-Net + 可训练 LUT
├── encoders/                    # 编码器（按名称切换，单文件实现）
│   ├── __init__.py              #   build_encoder() 注册表
│   └── codet5.py                #   codet5tok（逐 token 查表）/ codet5lora（逐 token + LoRA）
│                                #   / codet5（整段上下文，受 512 限制）
├── models/                      # 网络（单文件实现）
│   ├── __init__.py              #   build_hier() 按名称分发
│   ├── hier.py                  #   v0.2 频域 U-Net（name=hier）
│   ├── hier2.py                 #   v0.3 窗口注意力编解码器（name=codec）★
│   ├── heads.py                 #   样本级头 + 5 尺度 token 头
│   └── baseline.py              #   直接二分类基线：编码器 + masked 池化 + 线性/MLP 头
├── dataio/                      # 数据（单文件实现）
│   ├── __init__.py              #   build_dataset() 注册表 + collate
│   ├── base.py                  #   基类：报告前缀 + 组装（不截断）
│   ├── m4.py                    #   CoDET-M4 样本级（human/ai）
│   └── hybrid.py                #   HybridCodeAuthorship 行级（token 标签 + 行级评测信息）
├── report/                      # 报告注入（handcrafted / none）
├── scripts/
│   ├── prepare.py               # 下载数据与权重（m4 / encoder / hybrid）
│   ├── build_subset.py          # 均衡子集 + 预分词 + 划分（写 data/processed/）
│   ├── compare.py               # 汇总 runs/ 下所有实验的 val/test 指标
│   ├── diag_m4.py               # 诊断：按长度分桶 + 各尺度特征的线性探针
│   └── probe_m4.py              # 诊断：train 拟合 → val 评测的线性探针
├── docx/                        # 设计与实验报告
├── data/                        # 原始与处理后数据（不入库）
└── checkpoints/                 # 编码器权重与 LUT 缓存（不入库）
```

## 环境

| 组件 | 版本 |
| --- | --- |
| Python | 3.12 |
| torch | 2.9.1（PyPI 默认 wheel 自带 CUDA 12.8） |
| transformers | 4.57.x |
| peft | ≥0.17（`codet5lora` 需要） |
| GPU 目标 | RTX 3080 Ti 12GB |

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
python scripts/prepare.py hybrid     # HybridCodeAuthorship -> hybrid.parquet

python scripts/build_subset.py       # 均衡子集 + 预分词 + 划分（约 3~5 分钟）
# 产出：data/processed/m4.parquet（human/AI 各 1w）、data/processed/hybrid.parquet
```

`build_subset.py` 的均衡策略：长度分桶（token 数）× 类别（human/AI）× 语言（java/python/cpp），
按 **code 哈希**切 train/val/test = 8:1:1（同一段代码不跨集）；hybrid 按 RecordId 分组切分。
**不做任何截断**。

## 训练 / 评测

```bash
# v0.3（默认）：窗口注意力编解码器 + 逐 token CodeT5 LoRA
OMP_NUM_THREADS=8 python train.py --config configs/udet_v03.yaml --tag v03
python train.py --config configs/udet_v03.yaml --eval --ckpt runs/v03/best.pt

# 冒烟
OMP_NUM_THREADS=8 python train.py --config configs/udet_v03.yaml --tag smoke --limit 40 --epochs 1

# 汇总所有实验
python scripts/compare.py
```

产物：`runs/<tag>/{config.yaml, metrics.csv, metrics.jsonl, best.pt, eval.json}`。

指标：m4 → 样本级 ACC/F1；hybrid → 行级 P/R/F1、token 级 F1、片段级 F1（连续 AI 行段按 IoU≥0.5 匹配）。
训练与评测一律 **batch=1 整段**（不截断、不滑窗）。

> **`train.monitor` 注意**：默认 `mean` 只统计**标签有意义**的流（`train.sample_streams`）的样本级 F1
> 加上各流的行级 F1。hybrid 的样本标签恒为 1，其 `sample_f1` 会随 token 级变好而下降，
> 计入会把真正更好的 epoch 压下去（v0.3 首跑就踩过，见 `docx/u-det-v0.3.md` §6.4）。

## 消融开关

| 开关 | 取值 | 说明 |
| --- | --- | --- |
| `encoder.name` | `codet5tok` / `codet5lora` / `codet5` | 逐 token 查表 / 逐 token + LoRA / 整段上下文 |
| `encoder.freeze` | `true` / `false` | `codet5tok`：LUT 固定 buffer / 可训练嵌入表；`codet5lora`：固定 LoRA / 训练 LoRA |
| `encoder.targets` | `[v, o, wi, wo]` | LoRA 目标层。**逐 token（seq_len=1）时 q/k 对输出无影响**（softmax 单元素恒为 1），只能挂 v/o/wi/wo |
| `encoder.chunk` / `compile` | `2048` / `true` | 分块大小（内部补齐到固定长度）与 `torch.compile` |
| `train.lr_encoder` | `5e-5` | 编码器/LoRA 单独学习率（**3e-4 会一 epoch 改写编码器特征**） |
| `model.name` | `codec` / `hier` / `codet5cls` | v0.3 / v0.2 / 直接二分类基线 |
| `model.share` | `true` / `false` | 采样栈内 4 级是否共享同一个 Block |
| `model.divs` / `wins` | `[4,4,2,2]` / `[16,16,16,32]` | 各级下采样除数（累计 64×）与感受野 |
| `report.name` | `handcrafted` / `none` | 是否注入手工统计报告 |
| `loss.token` | `0` / `1.0` | token 级监督权重（0 = 纯样本级） |
| `train.sample_streams` | `[m4]` | 只在标签有意义的流上算样本级 CE |
| `train.streams` | `[m4, hybrid]` | 双流；`--no-m4` / `--no-hybrid` 单流 |

## 实现要点

- **逐 token 编码器**：`codet5tok` 分块跑冻结 CodeT5 得到 `(vocab, D)` 查表并缓存
  （`data/processed/codet5_lut.pt`）；`codet5lora` 保持同样语义但每步现算，用 peft LoRA 让梯度回流进
  CodeT5（`lora_B` 零初始化 ⇒ 起点与冻结 LUT 逐位相同）。
- **v0.3 采样块**（`models/hier2.py`）：`out = 算子捷径 + 注意力校正 + FFN`；
  下采样捷径 = 同窗口均值池化、kv = 细层自己；**上采样捷径 = 最近邻重复、kv = 下采样前的细层 skip**
  —— 上采样的跨注意力就是跳连。窗口越界两侧补 `[PAD]` 零向量，无需注意力掩码。
- **v0.2 采样核**（`models/hier.py`）：rfft → 逐频率低秩复数混合 → 低半频截断（下）/ 谱零延拓（上）
  → irfft → 裁剪。**FFT 一律在 ≥ 真实长度的 2 的幂上做**（否则 cuFFT plan 无界增长会吃光显存）。
- **多尺度损失**（`train.py:token_loss`）：各尺度 logits 长度不同，标签用 mask-aware 平均池化到对应
  长度作为软标签，权重 `loss.scale_weights` 由粗到细。
- **报告注入**（`report/handcrafted.py`）：空行率/缩进一致性 + 字符与二元组香农熵 + 命名惯例多样性，
  渲染成一行短文本（≤`report.max_tokens`）拼在代码 token 前，报告位置不计 token 损失（-100）。
- **行级标签对齐**（`dataio/hybrid.py`）：整段分词（fast tokenizer + offset mapping），每个 token 取
  起始字符所在行的 `Attribution`（AI=1 / Human=0）。

## 切换接口

```python
from encoders import build_encoder, list_encoders      # ['codet5', 'codet5lora', 'codet5tok']
from models import build_hier                          # 'codec'（v0.3）/ 'hier'（v0.2）
from dataio import build_dataset, collate              # 'm4' / 'hybrid'
from report import build_report                        # 'handcrafted' / 'none'
```

新增编码器 / 网络 / 数据集 / 报告：各自目录下新建单文件，在对应注册表里加一行映射即可。

## 备注（AutoDL）

- HuggingFace 直连不通 → `scripts/prepare.py` 默认走 hf-mirror；GitHub 需 `source /etc/network_turbo`；
- pip 走阿里云镜像；开代理后 pip 更慢，装依赖前 `unset http_proxy https_proxy`；
- 本机 `OMP_NUM_THREADS` 环境变量异常（libgomp 报错），运行时显式加 `OMP_NUM_THREADS=8`；
- 长序列用 `torch.cuda.cufft_plan_cache.max_size` 与"2 的幂内部 FFT"控制 plan 数量；
- CoDET-M4 原始 parquet 自带 `split` 列，当前子集未采用（按 code 哈希重切，防跨集泄漏）。

## 已知数据特性

- **hybrid 样本标签恒为 1**：`build_subset.py` 只保留含 AI 行的文件 ⇒ 样本级 CE 只在 m4 上回传；
- **m4 长度与标签强混淆**：ai 样本全部 <2048 token，≥4096 的样本 100% 是 human
  ⇒ 单靠长度规则 val acc 就有 0.659，报告样本级指标时需注意。

## 数据与许可

- 数据集：CoDET-M4（MIT，<https://huggingface.co/datasets/DaniilOr/CoDET-M4>）
- 数据集：HybridCodeAuthorship（Apache-2.0，<https://github.com/CapitalOne-Research/c1-hybrid-code-authorship>）
- 编码器权重：Salesforce/codet5-base（BSD-3-Clause）
