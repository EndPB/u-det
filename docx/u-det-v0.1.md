# U-Det v0.1 技术报告

> 版本：v0.1（2026-09-17）｜代码：`/root/autodl-tmp/u-det`｜硬件：RTX 3080 Ti 12GB（AutoDL 有卡模式）
> 数据：CoDET-M4 均衡子集（样本级）+ HybridCodeAuthorship（行级 → token 级）
> 主结果（不冻结 LUT，3 epoch ≈ 8 分钟）：m4 test F1 **0.874**，hybrid test 行级 F1 **0.542**（token F1 0.621）
> 首版冻结 LUT 对照：m4 0.867 / hybrid 行级 0.474；直接二分类基线（微调 CodeT5 + 池化）m4 test F1 **0.976**

---

## 0. 速览

| 项 | 数值 |
| --- | --- |
| 任务 | AI 代码检测（二分类）：**样本级** + **token 级**（同一模型、同时训练） |
| 编码器 | 逐 token 独立过 CodeT5（长度=1）→ 预计算查表（LUT）；**默认不冻结**（LUT 作为 24.7M 可训练嵌入表），可切回冻结 |
| 主干 | 1D U-Net（`features=[64,128,256,512]`，3 次下采样）+ 瓶颈 2 层 Transformer |
| 任务头 | 样本级头（瓶颈池化 + MLP，132,866 参数）+ token 级头（4 尺度 1×1 卷积，964 参数） |
| 可训练参数 | U-Net+头 **9,185,478**（9.19M；其中 U-Net 9,051,648 = 98.5%、两头 133,830 = 1.5%）；LUT 可训练时再 +24,652,800 → **33.84M** |
| 输入长度 | 无上限（逐 token 编码绕开 `n_positions=512` 与 $O(L^2)$）；默认 m4 512 / hybrid 2048 |
| 显存 | 前向 220–237 MiB；训练步（前向+反向，L=2048 / B=4）峰值 **0.35 GB** |
| 速度 | 0.08 s/it（每 it = m4 batch 16 + hybrid batch 4）→ 1726 it/epoch ≈ 2.4 分钟 |
| 评测 | 滑窗合并（window=2048，stride=1024）：样本 ACC/F1、行级 P/R/F1、token F1、片段 F1（IoU≥0.5） |

---

## 1. 任务与整体设计

### 1.1 任务定义

给定一段代码 $c$（token 序列 $x_{1:L}$），同时输出

- **样本级**：$P(\text{AI}\mid c)\in[0,1]$（整段代码是否含 AI 生成内容）；
- **token 级**：$P(\text{AI}\mid x_i, c)\in[0,1],\ i=1..L$（每个 token 是否属于 AI 生成片段），并由 token 概率聚合出行级 / 片段级判定。

### 1.2 数据流总览

```
代码文本 c
  │  ① 手工统计（结构偏置 / 词汇可预测性 / 句法方差）
  ├─► 报告文本 r ──┐
  │                │ ② CodeT5 RoBERTa-BPE 分词（报告 r 在前、代码 c 在后）
  │                ▼
  │        token 序列 [r_ids | x_1..x_L] (+<s>/</s>)
  │                │ ③ 逐 token 独立过编码器 → 查表 LUT (32100×768)
  │                ▼
  │        H ∈ R^{B×L×768} ──transpose──► R^{B×768×L}
  │                │ ④ 1D U-Net：inc → down×3 → 【瓶颈 Transformer】 → up×3
  │                ▼
  │   bottleneck (B,512,L/8) ──┬─► ⑤ 样本级头 → 2 logits
  │                            └─► ⑥ 4 个尺度的 token 头（含 bottleneck 自身）
  ▼
损失 L = λ_s·CE(样本) + λ_t·Σ_k w_k·BCE(尺度 k，软标签)
```

### 1.3 与 `docx/u-det.md` 设想的对应关系

| 设想 | 状态 | 落地形式 |
| --- | --- | --- |
| 参考 1D U-Net，用下采样替代长上下文滑动窗口 | ✅ | `models/unet.py`，3 次下采样；训练期长度上限仅受显存约束 |
| U-Net 中间（最底部）接样本级分类 | ✅ | `models/heads.py:SampleHead` 作用在 bottleneck 上 |
| 上采样 + 跳跃连接回来做 token 级分类 | ✅ | `models/heads.py:TokenHeads`，4 个尺度（L/8、L/4、L/2、L） |
| 上采样每层按不同文本长度精度算损失 | ✅ | `train.py:token_loss`：标签按尺度平均池化 → 软标签 BCE |
| 样本级与 token 级一起端到端训练 | ✅ | 双流并行，每个 optimizer step 两流同时反向 |
| 模块可插拔（便于消融） | ✅ | 4 张注册表：编码器 / 网络+瓶颈 / 数据集 / 报告 |
| 编码器使用 CodeT5 | ✅ | **逐 token 独立**编码（见 §2.2），不冻结（LUT 可训练） |
| 代码前注入短报告（三类统计） | ✅ | `report/handcrafted.py`，实测 53 token |
| 片段级数据用 c1-hybrid-code-authorship | ✅ | 行级 Attribution → token 级标签（8,584 条） |
| m4 提供 human 类、hybrid 提供 token 级引导 | ✅ | m4 human/AI 各 1w；hybrid 作 AI 类 + 行级监督 |
| 高质量均衡子集（类别/长度均衡） | ✅ | 长度 8 桶 × 语言 3 类 × 类别均衡，哈希切分 |
| U-Net 中间用 Transformer 双向交互 | ✅ | `models/transformer.py:MidTransformer`（2 层） |
| 中间改用 CodeT5 解码器自回归 | ⛔ 未做 | 后续版本（§8） |

---

## 2. 模型逐组件详解（重点）

### 2.1 输入表示层

**（a）报告文本**（`report/handcrafted.py`，无参数，纯预处理）

模板固定为一行（数值量化以压缩词表）：

```
STATS blank_ratio={} indent_consistency={} trailing_blank={} char_entropy={}
      bigram_entropy={} naming_diversity={} line_length_var={}
```

| 分组 | 统计量 | 定义 | 精度 |
| --- | --- | --- | --- |
| 结构偏置 | `blank_ratio` | 空行数 / 总行数 $R_{void}$ | 3 位小数 |
| | `indent_consistency` | 缩进量非零的行中，缩进为"最常见缩进单位"整数倍的比例 | 3 位 |
| | `trailing_blank` | 文末连续空行数（整数） | 整数 |
| 词汇可预测性 | `char_entropy` | 字符分布香农熵（bit） | 2 位 |
| | `bigram_entropy` | 相邻字符二元组分布香农熵（bit） | 2 位 |
| 句法方差 | `naming_diversity` | 标识符命名惯例（snake/upper/pascal/camel/other）分布的归一化熵，0=单一惯例、1=完全混用 | 3 位 |
| | `line_length_var` | 行字符长度的方差 | 1 位 |

- 统计**针对整段代码**（与窗口无关），报告 token 上限 `report.max_tokens=64`，实测样例 53 个 token。
- 消融开关：`report.name = handcrafted | none`。

**（b）分词**：CodeT5 的 RoBERTa-BPE（vocab 32,100，`<pad>=0 <s>=1 </s>=2 <unk>=3 <mask>=4`）；
报告 token 在**代码 token 之前**拼接，序列首尾加 `<s>`/`</s>`。

**（c）token 级标签**（仅 hybrid）：
用 fast tokenizer 的 `offset_mapping` 得到每个 token 的字符区间，取**起始字符所在行**的 `Attribution`（`AI→1`、`Human→0`）；
`<s>`/`</s>`、报告 token、padding 一律置 **-100**（不参与 token 损失）。
行级标签与 `line_of_token` 一并落盘，供行级/片段级评测使用（与 token 标签一一对应）。

### 2.2 逐 token 编码器 `CodeT5TokenEncoder`（`encoders/codet5.py`，`encoder.name=codet5tok`）

**动机**：把"长上下文"完全交给 U-Net，编码器只提供与上下文无关的 token 特征。

**原理**：每个 token 单独送进编码器（序列长度 1）。此时自注意力只有一个 key：
$\mathrm{softmax}(0)\cdot V(x) = V(x)$，注意力不产生 token 间交互，于是编码器退化为一个**固定的逐 token 函数** $f(\cdot)$：

$$f(t) = \mathrm{Encoder}([t])_{0}\in\mathbb{R}^{768}$$

由于 $f$ 与上下文、位置无关，可**预先**对全部 32,100 个词表项算一次，得到查表 $E\in\mathbb{R}^{32100\times768}$，
前向即 $H_{i} = E[x_i]$（一次 `F.embedding`）。

**关键属性**

| 项 | 值 |
| --- | --- |
| 训练期参数 | `freeze=true`：**0**（LUT 为非持久 buffer，不进 state_dict）；`freeze=false`：24,652,800（LUT 为 `nn.Parameter`，参与微调） |
| LUT 规模 | $32100\times768 = 24{,}652{,}800$ 个数 = **94.0 MiB**（fp32） |
| 构建方式 | 分块 8192 个 token 一次前向（`chunk=8192`），仅用编码器最后一层隐状态（`layer=-1`，可配） |
| 构建耗时 | **2.4 s**（GPU），缓存至 `data/processed/codet5_lut.pt`；之后启动不加载 850MB 权重 |
| 离线使用的编码器 | T5 encoder **109,607,040 参数**（词嵌入 24,652,800 ＋ 12 层 × 7,079,424 ＋ 首层相对位置偏置 384 ＋ 末层 LayerNorm 768），仅用于一次性构建 LUT |
| padding | 按 attention_mask 置零（避免卷积看到"伪 token"特征） |
| 消融对照 | `encoder.name=codet5`（整段上下文编码，受 512 位置限制） |

**收益**：编码器代价从 $O(L^2)$ 降到 $O(L)$ 查表；输入长度实际无上限（无位置嵌入、无注意力长度约束）。

**两种模式（`encoder.freeze`）**

| 模式 | LUT 形态 | 训练期参数量 | ckpt 大小 | 适用 |
| --- | --- | --- | --- | --- |
| `freeze=true`（v0.1 首版） | 非持久 buffer（`persistent=False`） | 0 | 35 MB | 快速消融 / 纯结构验证 |
| `freeze=false`（当前默认） | `nn.Parameter`（用编码器逐 token 特征初始化） | 24,652,800（94 MiB fp32） | 129 MB | 微调编码器（推荐） |

> “不冻结编码器”在逐 token 查表架构下的含义就是 **LUT 可训练**：等价于一个 32100×768 的嵌入表微调，
> 但参数量只有 109.6M 上下文编码器的 22%，也没任何 attention 开销。微调时用 `train.lr_encoder`
> （默认 1e-4，与 U-Net 的 1e-3 分组）以参数组形式单独给学习率。
> 实测效果（§6.4）：hybrid 行级 F1 0.474 → **0.542**、token F1 0.547 → **0.621**。

### 2.3 U-Net 主干 `models/unet.py`

配置：`features=[64,128,256,512]`（$\text{depth}=3$）、`kernel_size=3`、`use_batchnorm=True`、`dropout=0`、
`out_channels=None`（本模型不用 U-Net 自带的输出头，改用独立的两个任务头，避免死参数）。

**结构**：`inc`(DoubleConv) → `down×3`(MaxPool2d/1d k=2 + DoubleConv) → **瓶颈 mid** → `up×3`(最近邻插值 + cat 跳连 + DoubleConv)。
每个 `DoubleConv` = `Conv(k3) → BN → ReLU → Conv(k3) → BN → ReLU`，重复两次卷积。

**各组件通道数与参数量（精确）**

| 组件 | 结构细节 | 参数量 | 占比 |
| --- | --- | --- | --- |
| `unet.inc` | 768→64 k3：147,456 ＋ BN：128 ＋ 64→64 k3：12,288 ＋ BN：128 | **160,000** | 1.74% |
| `unet.downs.0` | MaxPool + 64→128 k3：24,576 ＋ BN：256 ＋ 128→128 k3：49,152 ＋ BN：256 | **74,240** | 0.81% |
| `unet.downs.1` | MaxPool + 128→256 k3：98,304 ＋ BN：512 ＋ 256→256 k3：196,608 ＋ BN：512 | **295,936** | 3.22% |
| `unet.downs.2` | MaxPool + 256→512 k3：393,216 ＋ BN：1,024 ＋ 512→512 k3：786,432 ＋ BN：1,024 | **1,181,696** | 12.86% |
| `unet.mid` | 见 §2.4 | **6,305,792** | **68.65%** |
| `unet.ups.0` | 插值 + cat(512+256=768)→256 k3：589,824 ＋ BN：512 ＋ 256→256 k3：196,608 ＋ BN：512 | **787,456** | 8.57% |
| `unet.ups.1` | 插值 + cat(256+128=384)→128 k3：147,456 ＋ BN：256 ＋ 128→128 k3：49,152 ＋ BN：256 | **197,120** | 2.15% |
| `unet.ups.2` | 插值 + cat(128+64=192)→64 k3：36,864 ＋ BN：128 ＋ 64→64 k3：12,288 ＋ BN：128 | **49,408** | 0.54% |
| **U-Net 小计** | | **9,051,648** | **98.54%** |

**其他实现点**

- 长度对齐：MaxPool 用 `ceil_mode=True` 兼容奇数长度；上采样用最近邻插值再按跳连尺寸裁剪，跳连用 `torch.cat` 拼接（通道维）。
- padding 安全：上采样后不额外插值到 padding 区域；`mask` 只在 `inc` 之前的特征上以置零方式体现，并通过 `downsample_mask`（逐级 2× MaxPool，ceil_mode）下采样后传给瓶颈模块与样本头。
- `return_features=True` 返回 `(logits, scales, bottleneck)`；`scales` 保持**原生分辨率**（不插值回全长），供多尺度 token 头使用。
- `deep_supervision=True` 仍保留（`aux_heads`）但本版本未启用（用 `TokenHeads` 代替，两者不叠加）。

### 2.4 瓶颈双向交互 `models/transformer.py:MidTransformer`

位置：`downs` 之后、`ups` 之前（即 U-Net 最底部），输入输出均为 $(B,C,L')$，$L'=\lceil L/8\rceil$。

| 项 | 配置 |
| --- | --- |
| 维度 | $d_{model}=512$（= `features[-1]`，与瓶颈通道一致） |
| 层数 | 2（pre-LN） |
| 注意力头 | 8（每头 64 维） |
| FFN | `mlp_ratio=4` → 隐层 2048，激活 GELU |
| dropout | 0.0（注意力与 MLP 都未启用） |
| 位置编码 | **正弦编码**（固定、无参数、长度无关，预计算 8192 并可自动扩展） |
| padding | 由 `key_padding_mask`（`~mask`）屏蔽 |
| 末尾 | 一层 LayerNorm |

**单层参数量明细（×2 层 + 末尾 LN）**

| 子模块 | 形状 | 参数量 |
| --- | --- | --- |
| `norm1` (LayerNorm) | (512,) ×2 | 1,024 |
| `attn.in_proj_weight` | (1536, 512) | 786,432 |
| `attn.in_proj_bias` | (1536,) | 1,536 |
| `attn.out_proj.weight` | (512, 512) | 262,144 |
| `attn.out_proj.bias` | (512,) | 512 |
| `norm2` (LayerNorm) | (512,) ×2 | 1,024 |
| `mlp.0` (Linear 512→2048) | (2048, 512) + (2048,) | 1,050,624 |
| `mlp.3` (Linear 2048→512) | (512, 2048) + (512,) | 1,049,088 |
| **单层合计** | | **3,152,384** |
| 2 层 + 末尾 `norm` (1,024) | | **6,305,792** |

> 单层里注意力 1,050,624（33.3%）、FFN 2,099,712（66.6%）、LN 2,048（0.06%）——
> `mlp_ratio=4` 使 FFN 是标准 Transformer 里参数的主项，全模型 68.65% 的参数都在这 2 层里。
> 若需瘦身，优先调 `model.mid.layers` / `mlp_ratio`，而不是动卷积层。

### 2.5 样本级头 `SampleHead`（`models/heads.py`）

作用于瓶颈输出 $(B,512,L')$：

1. **masked 平均池化**：$\bar{h}_d = \frac{\sum_l m_l h_{d,l}}{\sum_l m_l}$，$m$ 为下采样后的有效位 mask → $(B,512)$
2. `LayerNorm(512)` → `Linear(512→256)` → `GELU` → `Dropout(0)` → `Linear(256→2)`

| 子模块 | 形状 | 参数量 |
| --- | --- | --- |
| `norm` | (512,) ×2 | 1,024 |
| `net.0` | (256, 512) + (256,) | 131,328 |
| `net.3` | (2, 256) + (2,) | 514 |
| **合计** | | **132,866** |

> 设计取舍：当前用最简的 masked mean pooling（无额外参数、稳定）。可选升级为注意力池化 / `[CLS]` 查询 token，
> 或把瓶颈 Transformer 的层数加深——但要留意样本级信号从瓶颈回传到 512 维 FFN 的梯度强度。

### 2.6 token 级头 `TokenHeads`（`models/heads.py`）

4 个尺度各一个 `Conv1d(kernel_size=1)`，输入为 `[bottleneck] + scales`（**由粗到细**）：

| 头 | 输入通道 | 输入长度 | 形状 | 参数量 |
| --- | --- | --- | --- | --- |
| `heads.0` | 512（bottleneck） | $L/8$ | (1, 512, 1) + (1,) | 513 |
| `heads.1` | 256（`scales[0]`） | $L/4$ | (1, 256, 1) + (1,) | 257 |
| `heads.2` | 128（`scales[1]`） | $L/2$ | (1, 128, 1) + (1,) | 129 |
| `heads.3` | 64（`scales[2]`） | $L$ | (1, 64, 1) + (1,) | 65 |
| **合计** | | | | **964** |

输出每 token 1 个 logit（配合 `BCEWithLogitsLoss`）。1×1 卷积不破坏分辨率，故各尺度 logits 与对应特征同长度。

### 2.7 全模型参数量总表

| 组件 | 参数量 | 占比 |
| --- | --- | --- |
| `unet.mid`（瓶颈 Transformer，2 层） | 6,305,792 | 68.65% |
| `unet.downs.2` | 1,181,696 | 12.86% |
| `unet.ups.0` | 787,456 | 8.57% |
| `unet.downs.1` | 295,936 | 3.22% |
| `unet.ups.1` | 197,120 | 2.15% |
| `unet.inc` | 160,000 | 1.74% |
| `sample_head`（含 LN） | 132,866 | 1.45% |
| `unet.downs.0` | 74,240 | 0.81% |
| `unet.ups.2` | 49,408 | 0.54% |
| `token_heads`（4 个 1×1 卷积） | 964 | 0.01% |
| **U-Net + 任务头小计** | **9,185,478** | 100% |
| 编码器（离线构建 LUT） | 0（训练期）／109,607,040（构建期） | — |
| LUT | 24,652,800 个数 = 94.0 MiB（`freeze=false` 时为可训练参数） | — |
| **总计（默认 `freeze=false`）** | **33,838,278（33.84M）** | — |

- 张量个数：82 个可训练张量（其中 U-Net 68、样本头 6、token 头 8，且无死参数：`out_channels=None` 避免
  了未被使用的 U-Net 输出头）；`freeze=false` 时再加 1 个 `encoder.lut`。
- checkpoint 大小：LUT 冻结 约 35 MB；LUT 可训练 约 129 MB（含 BN running 统计量）；
  对照：微调整段 CodeT5 的基线为 418 MB。

### 2.8 前向张量形状流（实测）

| 阶段 | L=512, B=16（m4） | L=2048, B=4（hybrid） |
| --- | --- | --- |
| 编码器输出 | (16, 512, 768) | (4, 2048, 768) |
| transpose → `inc` 输入 | (16, 768, 512) | (4, 768, 2048) |
| `inc` 输出 | (16, 64, 512) | (4, 64, 2048) |
| `downs.0` 输出 | (16, 128, 256) | (4, 128, 1024) |
| `downs.1` 输出 | (16, 256, 128) | (4, 256, 512) |
| `downs.2` 输出 = 瓶颈输入 | (16, 512, 64) | (4, 512, 256) |
| `mid` 输出 = **bottleneck** | (16, 512, 64) | (4, 512, 256) |
| `ups.0` / `scales[0]` | (16, 256, 128) | (4, 256, 512) |
| `ups.1` / `scales[1]` | (16, 128, 256) | (4, 128, 1024) |
| `ups.2` / `scales[2]` | (16, 64, 512) | (4, 64, 2048) |
| 样本级 logits | (16, 2) | (4, 2) |
| token 级 logits（由粗到细） | (16,1,64) / (16,1,128) / (16,1,256) / (16,1,512) | (4,1,256) / (4,1,512) / (4,1,1024) / (4,1,2048) |

### 2.9 显存与速度

| 场景 | 峰值显存 |
| --- | --- |
| 前向 L=512 / B=16 | 220 MiB |
| 前向 L=2048 / B=4 | 237 MiB |
| 前向+反向（训练步，L=2048 / B=4，bf16） | **0.35 GB** |
| 训练速度 | 0.08 s/it（含 m4 B=16 + hybrid B=4 两次前向反向与一次更新）≈ 12.5 it/s |

> 显存主要被 U-Net 的中间激活占用（$64\times L$ 那一层），瓶颈 Transformer 因 $L'=L/8$ 开销极小。
> 12GB 卡上仍有很大余量：可把 `batch_size.hybrid` 提到 16–32、或 `max_length` 提到 4096+。

---

## 3. 损失函数（`train.py`）

### 3.1 总损失

$$\mathcal{L} = \lambda_s\,\mathcal{L}_{\text{sample}} + \lambda_t\,\mathcal{L}_{\text{token}}$$

- $\lambda_s = $ `loss.sample`（默认 1.0）、$\lambda_t = $ `loss.token`（默认 1.0；置 0 退化为纯样本级）
- 每个 step 在**同一个**总损失上反传（两流 batch 的损失相加），随后一次 `opt.step()`

### 3.2 样本级项

$$\mathcal{L}_{\text{sample}} = \mathrm{CE}\big(\text{logits}_{(B,2)},\ y\big)$$
（可选 `loss.class_weights` 类别加权，默认不加权；m4 的两类均衡，hybrid 的 $y\equiv1$）

### 3.3 token 级项（多尺度 + 软标签）

设最细尺度标签为 $y\in\{0,1,-100\}^{L}$（-100 = 忽略），尺度 $k$ 的下采样倍率 $s_k\in\{8,4,2,1\}$，
则该尺度长度为 $L_k=L/s_k$。用**mask-aware 平均池化**得到该尺度的软标签：

$$\tilde y^{(k)}_j = \frac{\sum_{i\in \text{bin}_j} m_i\, y_i}{\sum_{i\in \text{bin}_j} m_i},\qquad
w^{(k)}_j = \mathbb{1}\Big[\textstyle\sum_{i\in \text{bin}_j} m_i > 0\Big]$$

$$\mathcal{L}_{\text{token}} = \frac{1}{\sum_k \lambda_k}\sum_{k}\lambda_k\cdot
\frac{\sum_j w^{(k)}_j\,\mathrm{BCE}_{\text{logits}}\big(z^{(k)}_j,\tilde y^{(k)}_j\big)}{\sum_j w^{(k)}_j}$$

| 尺度 $k$ | 特征来源 | 长度 | 权重 $\lambda_k$（`loss.scale_weights`） |
| --- | --- | --- | --- |
| 0（最粗） | `bottleneck` | $L/8$ | 0.25 |
| 1 | `scales[0]` | $L/4$ | 0.50 |
| 2 | `scales[1]` | $L/2$ | 0.75 |
| 3（最细） | `scales[2]` | $L$ | 1.00 |

要点：

- 粗尺度用**软标签**（该窗口内 AI token 占比），而不是硬投票——与"不同文本长度精度"的表述一致，
  也让细尺度与粗尺度的梯度方向自然一致；
- 报告 token、`<s>`/`</s>`、padding 全为 -100，池化时分母自动把它们剔除；
- m4 的 batch 完全没有 token 标签 → 该分支返回 `None`，不产生任何 token 梯度（纯样本级监督）。

---

## 4. 训练方案

### 4.1 双流并行（澄清"步数不同"）

两个任务**同时**训练，机制是：

```
每个 optimizer step：
    batch_m4     = next(m4_stream)        # 16 条，窗口 ≤512
    batch_hybrid = next(hybrid_stream)    # 4 条，窗口 ≤2048
    loss = (样本级 CE over batch_m4 ∪ batch_hybrid) + (多尺度 token BCE over batch_hybrid)
    loss.backward()                       # 梯度累加
    clip_grad_norm_(1.0); opt.step(); scheduler.step()
```

- 两个 forward、一次 backward、一次更新：**同一模型、同一 optimizer、同一 step**，不存在先后或交替。
- "步数不同"只影响 epoch 记账：m4 loader 每 epoch 1004 步（16,073/16），hybrid 1726 步（6,904/4）。
  取 max 作 epoch 长度（1726），较短的 m4 用 `cycle()` 从头再来一遍（该 epoch 中 m4 被使用 1.72 遍）。
  若改用 min，则每 epoch 有 722 步的 hybrid 数据被丢弃。
- 为什么不合成一个 DataLoader：两源的窗口长度不同（512 vs 2048）。合成后同一 batch 里只要有一条 hybrid 长样本，
  同 batch 的 m4 样本会被 pad 到 2048，约 **25%** 的算力浪费在 padding 上（本配置 16:4 混合）。
  若要"单一数据集"的写法，可加 `mixed` 数据集 + 按源配比的 batch sampler，代价即上述 padding。

### 4.2 超参数（`configs/udet_base.yaml`）

| 项 | 值 | 说明 |
| --- | --- | --- |
| epochs | 3 | 共 5,178 step |
| batch_size | m4 16 / hybrid 4 | 两源各自独立 |
| max_length（训练） | m4 512 / hybrid 2048 | 超长样本**随机截窗**（同一 epoch 内可复现） |
| eval_max_length | 2048 | 滑窗 stride = 1024 |
| optimizer | AdamW，`lr=1e-3`；参数组：U-Net+头 1e-3，编码器 `lr_encoder`（LUT 1e-4 / 上下文编码器 2e-5） | 编码器冻结时只有一组 |
| scheduler | OneCycleLR，`pct_start=0.05` | 峰值在第 259 步，末段退火到 ~0 |
| 精度 | bf16 autocast | BatchNorm 保持 fp32 |
| 梯度裁剪 | 1.0 | 全局范数 |
| 模型选择 | `monitor=mean` | mean(m4 val F1, hybrid val 行级 F1) → 本次 best = epoch 1 |
| 其他 | `num_workers=4`，`seed=0`，drop_last=True | |

### 4.3 训练曲线（val）

| epoch | m4 val（ACC / F1） | hybrid val（行级 F1 / token F1 / 片段 F1） | train loss（累计均值） | 本 epoch 耗时 |
| --- | --- | --- | --- | --- |
| 0 | 0.733 / 0.779 | 0.5197 / 0.5848 / 0.1592 | 0.953（sample 0.375 / token 0.578） | 127 s |
| 1 | **0.850 / 0.864** | 0.4575 / 0.5132 / 0.1295 | 0.726（0.203 / 0.523） | 135 s |
| 2 | 0.820 / 0.840 | 0.4689 / 0.5276 / 0.1304 | 0.630（0.136 / 0.495） | 137 s |

观察：样本级任务随训练稳步上升；token 级在 epoch 0 达到峰值后略回落（高学习率下 token 头在"高召回
(p=0.39, r=0.77)"与"平衡"之间摆动），末段 lr 退火后趋稳。

---

## 5. 数据

### 5.1 构成（`data/processed/`）

| 数据集 | 条数 | token 均值 / 中位 / 最大 | 关键分布 | train/val/test |
| --- | --- | --- | --- | --- |
| `m4.parquet`（CoDET-M4 均衡子集） | 20,000 | 777 / 275 / 4,096 | 类别 human 10,000 / ai 10,000；语言 python 6,340 / java 6,842 / cpp 6,818 | 16,073 / 2,001 / 1,926 |
| `hybrid.parquet`（HybridCodeAuthorship） | 8,584 | 1,814 / 1,272 / 10,475 | 行数均值 185；AI 行占比 0.308；模型 gpt-oss-120b 3,399 / llama-4-scout 1,582 / llama-3.3-70b 3,603 | 6,904 / 797 / 883 |

每 epoch 可见监督量：**样本级标签 22,977 个**（m4 16,073 + hybrid 6,904）、
**token 级标签 12.57M 个**（hybrid train 共 1,283,213 行，其中 AI 行 391,882，30.5%）；
m4 train 12.52M token（仅样本级标签）。两源每 epoch 的 token 量几乎相同（12.52M vs 12.57M）。

### 5.2 均衡与切分（`scripts/build_subset.py`）

- **两级筛选**：先按字符长度桶 × 语言做蓄水池采样（每格 1200 条），预分词得到精确 token 长度，
  再按 8 个 token 长度桶（32/64/128/256/512/1024/2048/4096）× 语言 × 类别做 **waterfilling 均衡**。
- m4 结果：human 每个（长度桶 × 语言）格 ~417 条（非常均匀，各桶 1,244–1,251）；
  ai 桶 0–4 每格 ~657 条，桶 5–7 因 M4 本身缺少长 AI 样本而稀疏（429 / 3 / 0）——属于数据固有分布。
- **去重**：以 `blake2b(code)` 全量去重（跨类别同文只留一条，避免同段代码既是 human 又是 ai）。
- **切分**：按内容哈希取模 10 → train/val/test = 8/1/1，同一段代码不会跨集；
  hybrid 按 `RecordId` 分组（同一原文件的不同生成模型样本进同一集，防止泄漏）。
- 未采用 CoDET-M4 自带的 `split` 列（train 37.4w / val 4.4w / test 4.4w），原因见 §8。
- hybrid 过滤：删除无 AI 行的样本 1,904 条（占 18.2%）；3 条行数与 Attribution 数不一致的按短侧截断。

---

## 6. 评测协议与结果

### 6.1 滑窗合并

- 样本长度 > `eval_max_length` 时按 window=2048、stride=1024 切片；
- **样本级**：各窗口 logits 求平均后 argmax；
- **token 级**：重叠区概率按累积均值合并；
- **报告前缀**：多窗口样本只在该样本的**第一个窗口**注入报告（避免重复注入）；其余窗口无报告前缀，
  属于已知的输入不完全一致（单窗口样本即全部 m4 样本不受影响）。
- **行级**：一个 token 的概率按其在文件中的行聚合（`line_of_token`），行概率 = 该行 token 概率均值，>0.5 判 AI；
- **片段级**：把连续 AI 行归并为片段，预测片段与标注片段按 **IoU ≥ 0.5** 贪心匹配（**逐文件统计**，
  避免文件边界拼出假片段），汇总后算 P/R/F1。

### 6.2 首版结果（LUT 冻结，best.pt = epoch 1；不冻结版本见 §6.4）

| 数据集（test 条数） | 指标 | val | **test** |
| --- | --- | --- | --- |
| m4（1,926 / val 2,001） | ACC | 0.850 | **0.849** |
| | F1 | 0.864 | **0.867** |
| hybrid（883 / val 797） | 行级 P | 0.494 | **0.496** |
| | 行级 R | 0.426 | **0.454** |
| | 行级 F1 | 0.458 | **0.474** |
| | token F1 | 0.513 | **0.547** |
| | 片段 P / R / F1 | 0.181 / 0.101 / 0.130 | **0.174 / 0.100 / 0.127** |

参照：HybridCodeAuthorship 论文里 AIGCode Detector 的最好成绩为行级 F1 0.48、片段级 F1 0.56
（该工作是纯检测器基线，训练/评测协议未必完全一致，仅作量级参照）。本版本行级已接近，**片段级差距明显**。

### 6.3 片段级 F1 偏低的诊断

1. **评测口径苛刻**：一个 20 行的正确片段若被判成 2 段（IoU 各 0.5）就全部不计分；预测碎片化会被双重惩罚
   （精确率与召回率同时下降）。
2. **无后处理**：直接对逐 token 概率取 0.5 阈值，行与行之间概率抖动会导致片段碎裂；
   没有做平滑、最小段长、或片段级合并。
3. **训练/评测窗口不一致**：训练用随机窗口（≤2048），评测用滑窗重叠区**均匀平均**，
   窗口边界处的概率更不稳定，恰好在片段边界产生假分裂。
4. **损失层面缺少片段一致性**：token 级 BCE 逐点独立，没有对"相邻 token 标签应一致"建模
   （无 CRF / 转移项 / 片段级 IoU 损失）。
5. **类别不平衡**：AI 行占 30.8%（全量）/ 30.5%（train），BCE 未加权 → 边界行（信息最少）更容易被压向负类，导致召回偏低（r=0.45）。

### 6.4 与“CodeT5 直接二分类”基线的对比（不冻结编码器）

基线 `models/baseline.py:PooledClassifier`：整段代码 → CodeT5（n_positions=512，**微调**）→ masked 平均池化
→ `LayerNorm + Linear(768→2)`；协议与 U-Det 一致（同一 m4 子集、3 epoch × 1,004 step、L=512 截窗、bf16、AdamW+OneCycle）。

| 实验 | 模型 | 可训练参数 | m4 test ACC | m4 test F1 | hybrid 行级 F1 | token F1 | chunk F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `base_codet5_ft` | CodeT5 → mean pool → Linear | 109.6M | **0.975** | **0.976** | — | — | — |
| `udet_m4` | U-Det（样本级单任务） | 24.7M(LUT)+9.19M | 0.948 | 0.949 | — | — | — |
| `base_lut` | U-Det（样本级+token 级） | 24.7M(LUT)+9.19M | 0.859 | 0.874 | **0.542** | **0.621** | 0.150 |
| `base`（首版） | U-Det（双任务，LUT 冻结） | 9.19M | 0.849 | 0.867 | 0.474 | 0.547 | 0.127 |

验证集 F1 曲线：

| epoch | 基线（m4） | U-Det 单任务（m4） | U-Det 双任务（m4 / hybrid 行级） | 首版冻结 LUT（m4 / hybrid） |
| --- | --- | --- | --- | --- |
| 0 | 0.9562 | 0.9168 | 0.8595 / 0.5185 | 0.7790 / 0.5197 |
| 1 | 0.9723 | 0.9333 | 0.8849 / 0.5394 | 0.8643 / 0.4575 |
| 2 | 0.9742 | 0.9495 | 0.8692 / 0.5308 | 0.8396 / 0.4689 |

结论：

1. **样本级单任务上，微调上下文编码器 + 池化明显强于 U-Det**（0.976 vs 0.949，差 2.6 分）：逐 token
   无上下文特征 + 9.19M 主干 在单任务上信息/容量不够；U-Det 的优势场景不在这里。
2. **token 级 / 片段级任务只有 U-Det 能原生做**：基线受 512 位置限制，无法直接输出全文件 token 级标注
   （要做得加“分窗 + 拼接”后处理）。
3. **不冻结编码器对 U-Det 很关键**：行级 F1 0.474 → **0.542**、token F1 0.547 → **0.621**（+6.8 / +7.4 分），
   而 LUT 只增加 24.7M 参数（上下文编码器的 22%）。
4. **双任务会拉低 m4 样本级**：单任务 0.949 → 双任务 0.874（-7.5 分），因为 hybrid 的样本级标签恒为 AI
   且长序列 token 任务主导梯度 → 下一步可调 `loss.sample` 权重 / 两流采样比 / 先单任务预热再双任务。

---

## 7. 消融开关与用法

```bash
# 训练 / 评测 / 冒烟
OMP_NUM_THREADS=8 python train.py --config configs/udet_base.yaml --tag base
OMP_NUM_THREADS=8 python train.py --config configs/udet_base.yaml --eval --ckpt runs/base/best.pt
OMP_NUM_THREADS=8 python train.py --limit 200 --epochs 1 --max-length 512 --tag smoke
```

| 开关 | 取值 | 作用 | 已实测 |
| --- | --- | --- | --- |
| `encoder.name` | `codet5tok` / `codet5` | 逐 token 查表 / 整段上下文编码 | 默认前者 |
| `encoder.freeze` | `false` / `true` | 是否微调编码器（codet5tok 时 = LUT 变可训练嵌入表） | ✅ 行级 F1 0.474 → 0.542 |
| `train.lr_encoder` | `1e-4` / `2e-5` | 编码器单独学习率（参数组） | ✅ |
| `model.name` | `unet1d` / `codet5cls` | U-Det / 直接二分类基线（`--model`） | ✅ §6.4 |
| `model.mid.name` | `transformer` / `none` | 瓶颈是否做双向交互 | ✅ 参数 9.19M → **2.88M** |
| `report.name` | `handcrafted` / `none` | 是否注入手工统计报告 | ✅ |
| `loss.token` | 1.0 / 0 | token 级监督开关 | ✅ |
| `loss.scale_weights` | 4 元列表 | 各尺度权重（由粗到细） | ✅ |
| `train.streams` | `[m4, hybrid]` | `--no-m4` / `--no-hybrid` 单流 | ✅ |
| `model.features` | 如 `[64,128,256,512]` | U-Net 深度/宽度（深度 = len−1） | — |
| `report.max_tokens` | 64 | 报告长度上限 | — |

---

## 8. 局限与下一步

**建模**
1. 片段级指标是最大短板：优先试 (a) 概率平滑（移动平均 / 中值滤波）后再阈值、(b) 最小段长与相邻段合并、
   (c) 加权 BCE 或 Dice 类损失、(d) 片段级 IoU 感知损失、(e) 相邻 token 一致性（CRF / 转移矩阵）。
2. 逐 token 编码丢弃了 CodeT5 的上下文表示，全靠 U-Net 学 —— **§6.4 已量化**：同数据单任务下比微调
   CodeT5 + 池化的基线低 2.6 分 F1。可试的补救：(a) 让 U-Net 输入拼上编码器的上下文特征（冻结上下文编码器
   + 逐 token 特征双通道）；(b) 逐 token 路径再拼相邻 k 个 token 的编码（局部窗口，牺牲一点长度自由度）；
   (c) 把 U-Net 加深/加宽以提升容量。
3. 瓶颈 Transformer 占了 68.65% 参数却只作用在 $L/8$ 的序列上；可做 `layers`、`mlp_ratio`、`heads` 的
   规模消融，或试"瓶颈前先线性压维"以降低参数量。
4. 样本级头目前是 masked mean pooling；可对比注意力池化或查询 token。
5. `docx/u-det.md` 里"U-Net 中间改用 CodeT5 解码器自回归"尚未实现——可作为 v0.2 的探索方向。

**数据**
6. 当前子集未采用 CoDET-M4 自带 `split` 列（train 37.4w/val 4.4w/test 4.4w）。若要和公开基线对齐，
   应在 `build_subset.py` 中保留 `raw_split` 并按官方划分评测。
7. m4 的长 AI 样本天然稀缺（桶 5–7 只有 429/3/0 条），长样本上的样本级能力目前只能靠 human 类与 hybrid 支撑。
8. hybrid 只用了 `AICode`（含 AI 行），未使用其 `HumanCode`（纯人类原文件）；后者可作为额外 human 类或
   "零 AI 行"的 token 级负样本（当前 hybrid 内部已含 69% 人类行，故不是硬需求）。

**工程**
9. 训练时 GPU 利用率仅 ~14%，瓶颈在 CPU 侧（报告分词 + collate）。优化方向：把报告 token 在
   `build_subset.py` 里预计算并落盘（代价是报告模板变更需重建），或提高 `num_workers`。
10. 训练/评测的窗口策略可以更细：训练也按"窗口中心加权"而非均匀权重，与滑窗评测对齐。

---

## 附录 A. 代码结构

```
u-det/
├── train.py                     # 主干：UDet 组装 + 双流并行训练 + 多尺度损失 + 滑窗评测
├── configs/udet_base.yaml       # 全部开关与超参
├── encoders/{__init__,codet5}.py        # 注册表；CodeT5Encoder（上下文）/ CodeT5TokenEncoder（LUT）
├── models/{__init__,unet,transformer,heads}.py   # U-Net / 瓶颈 Transformer / 两个任务头
├── dataio/{__init__,base,m4,hybrid}.py  # 注册表 + collate；基类（报告前缀+窗口裁剪）；两数据源
├── report/{__init__,handcrafted}.py     # 报告注册表 + 三类手工统计
└── scripts/{prepare,build_subset}.py    # 下载（HF/GitHub）+ 均衡子集构建与预分词
```

## 附录 B. 关键文件对应（本文档数字来源）

| 数字 | 来源 |
| --- | --- |
| 参数量明细 | `build_model()` 后遍历 `named_parameters()` 逐张量统计 |
| 形状流 / 显存 | 前向实测（L=512/B=16、L=2048/B=4），`torch.cuda.max_memory_allocated()` |
| 数据统计 | `data/processed/{m4,hybrid}.parquet` 逐列统计 |
| 结果 | `runs/base/metrics.jsonl`（val）与 `--eval --ckpt runs/base/best.pt`（val/test） |
| 训练耗时 | `runs/base/metrics.csv`（0.08 s/it） |

## 附录 C. 复现步骤

```bash
# 0. 环境：conda env udet（Python 3.12, torch 2.9.1+cu128, transformers 4.57.6）
# 1. 数据与权重
python scripts/prepare.py data            # CoDET-M4（HF 镜像）
python scripts/prepare.py encoder         # CodeT5-base
source /etc/network_turbo && python scripts/prepare.py hybrid   # HybridCodeAuthorship（GitHub）
python scripts/build_subset.py            # 均衡子集 + 预分词（约 3~5 分钟）
# 2. 训练 + 评测
OMP_NUM_THREADS=8 python train.py --config configs/udet_base.yaml --tag base
OMP_NUM_THREADS=8 python train.py --config configs/udet_base.yaml --eval --ckpt runs/base/best.pt
```
