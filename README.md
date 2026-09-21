# U-Det

> ## ⚠️ 动手之前先读 [`docx/lessons.md`](docx/lessons.md)
>
> 里面是本项目**踩过的坑与由此总结出的规则**，按「方法论 / 评测与产物 / 数据特性 /
> 工程运维」分类，每条都注明证据来源。
>
> * **A 类（方法论）每一条都让本项目的结论翻过车** —— 改代码或跑任何对比之前务必先扫一遍；
> * **B 类（评测与产物）全是「静默出错」** —— 不报错，但结果是垃圾；写评测相关代码前必读。

AI 代码检测实验：**分块上下文编码 + 序列级编解码主干 + 样本级/token 级双任务**。

- 每 **128 个 token 为一个块**过 CodeT5（块内 attention 真实存在、块间独立）→ 特征对长度无假设，**整段代码不截断**；
- 主干把"下采样/上采样"做成**可学算子**，用多级压缩替代长上下文滑动窗口；
- **样本级**分类读**全部 5 个尺度**（各自池化后拼接）；各尺度出 **token 级**分类（多尺度深监督），最细尺度额外拼接编码器全长特征；
- 代码前注入**短报告**（结构偏置 / 词汇可预测性 / 句法方差）；
- 数据：CoDET-M4（样本级）+ HybridCodeAuthorship（行级标注 → token 级），两流**并行**、
  每个 optimizer step 同时参与。

两个版本各赢一半，**互为补充**（都只跑 4 epoch，与基线同预算）：

* **v0.4.5**（`configs/udet_v045.yaml`）：line **0.7980** / chunk **0.5114** / token **0.8281**
  ⇒ ★ **三项 token 级指标首次全部反超微调 CodeT5 滑窗基线**（0.7978 / 0.5081 / 0.8270），
  且 `[2048,8192)` 长文档桶（241 样本）也首次反超（line +0.57）；代价是 m4 掉到 0.9693（−0.43）。
* **v0.4.4**（`configs/udet_v044.yaml`）：m4 **0.9775** ⇒ ★ **首次在 m4 上反超基线**。
* 两者都是**参数少 16M、长度无上限**；§8.10 显示差距里**有六成来自固定阈值 0.5**。

🟡 **v0.4.7 / v0.4.8 已完成但只跑了 2 轮**（† 标记的那两行）—— 不能与 4 轮的版本直接比。
**同 4 轮预算下，v0.4.5 是唯一在 hybrid 三项上全面超过微调基线的版本**
（行级 +0.02 / 段级 +0.33 / token 级 +0.11，代价是文档级 −0.43）；
而提文档级的版本（v0.4.4 / v0.4.6 / v0.4.8）都把 token 三项推回基线以下。

★ **把这五次改动排在一起看，符号一致（详见 `docx/u-det-v0.4.md` §8.15.1）：
三个“给文档级加**表达力**”的改动全部拉低 token 三项；
唯一“给 token 级加通路”的改动拉低文档级；
而唯一改 **loss 监督强度** 的改动（锚点 L/8）则只涨段级。
⇒ 这不是“哪个算子更好”，而是**两个目标在共享主干上互相竞争**（结构性耦合）。
⇒ 下一步应**解耦**（旁路改额外一路头 / 文档级梯度加缩放），而不是继续加表达力。

★ **编码器层数分配也已被零训练探针否掉一半**（`scripts/probe_layers.py`，§8.16）：
**文档级信息在第 5 层就饱和**，而且在**瓶颈分辨率（L/64）下同样饱和**
⇒ 瓶颈缺的是**位置数（带宽）**，不是加工层数 ⇒ **“把尾部 k 层搬到瓶颈”没有依据**。
再看“删”的真实指标代价（拿训练好的主干 + 一个固定的 768×768 线性补偿来量，四项均值）：
**k=2 → −0.57、k=3 → −0.71、k=4 → −1.28** ⇒ **k=3 是最佳折中**（与 k=2 同价、多腾一层）。
⇒ 该试的是让 `divs` 对短文档自适应（少下一次采样），而不是往瓶颈堆层。

★ **v0.4.9 已实现、自检通过（待跑）**：把「结构性耦合」的三项做成**单变量**开关
（§8.17）：
① token 旁路改「**额外一路头**」（**分别监督**、只把均值用于报告）；
② 文档级 → 主干的**梯度缩放 λ**（只改梯度不改前向，λ=0 即 stop-gradient）；
③ 瓶颈**权重移植**（拷 CodeT5 blocks 8-11，**参数与结构不变**，保真度 cos 0.86；
若允许瓶颈改用 T5 语义 RMS+ReLU 则 0.997）。
另新增通用 CLI `--set 段.键=值`，所以四条变体**共用一份配置**，不会配置漂移。
✅ 启动队列：`setsid nohup bash scripts/queue_v049.sh > /tmp/queue_v049.log 2>&1 &`
（4 × 2 epoch ≈ 9 h；⚠️ 2 轮只能排方向，胜出者必须续到 4 轮才能引用）

旧两版的两项改动（长度权重、报告向量）取值均由零训练探针定下：
① `loss.token_weight_mode: length`；② `report.mode: vector`。
探针实测：报告前缀在近一半样本（长度<256）上**占掉瓶颈一半的位置**。

## 版本演进与结果（test 集，同一份数据/切分）

| 版本 | 主干 | 编码器 | m4 sample F1 | hybrid line F1 | chunk F1 | token F1 |
| --- | --- | --- | --- | --- | --- | --- |
| v0.4.9（+ 额外一路头 / λ / 权重移植） | 同上 | 同上 | 🟡 待跑 | 🟡 待跑 | 🟡 待跑 | 🟡 待跑 |
| CodeT5 滑窗基线 | — | CodeT5 **全量微调**，window=512 / stride 按流（同数据同指标同预算） | 0.9736 | 0.7978 | 0.5081 | 0.8270 |
| base_codet5_ft（旧口径） | — | CodeT5 整段上下文（512 截断） | 0.9756 | — | — | — |
| 基线 | — | CodeT5 整段上下文（冻结） | 0.7651 | — | — | — |
| **v0.4.8**（+ 文档级池化 ABMIL） | 同上 | 同上 | **0.9748**† | 0.7767 | 0.4791 | 0.8092 |
| **v0.4.7**（+ 锚点随长度缩放 N_min=L/8） | 同上 | 同上 | 0.9713† | 0.7843 | 0.4884 | 0.8141 |
| **v0.4.6**（+ 长度感知尺度权重 + 报告改文档级向量） | 同上 | 同上 | 0.9745 | 0.7909 | 0.5043 | 0.8229 |
| ★ **v0.4.5**（+ token 头旁路） | 窗口注意力编解码器 + 门控 | 分块 CodeT5（K=128）+ LoRA | 0.9693 | ★ **0.7980** | ★ **0.5114** | ★ **0.8281** |
| ★ **v0.4.4**（+ 多尺度样本头） | 同上 | 同上 | ★ **0.9775** | 0.7878 | 0.5076 | 0.8187 |
| v0.4.2 | 同上 | 分块 CodeT5 + LoRA | 0.9700 | 0.7895 | 0.5035 | 0.8190 |
| v0.4.1 | 同上 | 逐 token CodeT5 + LoRA | 0.9408 | 0.7139 | 0.3970 | 0.7489 |
| v0.4.0 | 同上 | 逐 token CodeT5 + LoRA | 0.9380 | 0.7071 | 0.3907 | 0.7431 |
| v0.3.1 | 同上（无门控） | 逐 token CodeT5 + LoRA | 0.9307 | 0.6922 | 0.3733 | 0.7279 |
| v0.3 | 窗口注意力编解码器（栈内共享） | CodeT5 + peft LoRA | 0.8863 | 0.6462 | 0.3056 | 0.6864 |
| v0.2 | 频域 U-Net（FFT 上下采样） | CodeT5 + peft LoRA | 0.7705 | 0.6535 | 0.3698 | 0.6801 |
| v0.2 | 同上 | 查表 LUT（可训练） | 0.7686 | 0.6341 | 0.3094 | 0.6666 |
| v0.1 | 卷积 U-Net | 查表 LUT（可训练） | 0.8744 | 0.5416 | 0.1499 | 0.6208 |
| v0.1 | 同上 | 查表 LUT（冻结） | 0.8665 | 0.4740 | 0.1269 | 0.5465 |

> 除第一行外的「同数据同指标同预算」基准见 `docx/u-det-v0.4.md` §8.6；
> v0.4.x 的逐代差距分析见 §8.9 / §8.10 / §8.11；⚠️ 2 epoch 的读数在本项目已**六次**指向错误结论（lessons A1）。

- 主干参数量：v0.1 约 9M（LUT 冻结）/ v0.2 **122.35M** / **v0.3 42.52M**（栈内 4 级共享权重）/ **v0.4.4 93.65M**（含门控与多尺度样本头）/ **v0.4.5 93.65M**（+ 769，token 头旁路）。
- 逐版本设计说明与逐组件参数量见 `docx/u-det-v0.1.md`、`docx/u-det-v0.2.md`、
  `docx/u-det-v0.3.md`、**`docx/u-det-v0.4.md`**（最新）；总设计见 `docx/u-det.md`；
- ★ **踩过的坑与规则见 `docx/lessons.md`**（改代码前必读）。

## 目录结构

```
u-det/
├── train.py                     # 主干：训练 / 验证 / 评测（双流并行 + 多尺度损失 + 梯度累积）
├── configs/
│   ├── udet_v049.yaml           # v0.4.9：token 旁路改「额外一路头」（单变量）
│   ├── udet_v048.yaml           # v0.4.8：文档级池化改 ABMIL
│   ├── udet_v047.yaml           # v0.4.7：尺度权重锚点随长度缩放（N_min = L/8）
│   ├── udet_v046.yaml           # v0.4.6：长度感知尺度权重 + 报告改文档级向量
│   ├── udet_v045.yaml           # v0.4.5：+ token 头旁路（hybrid 三项最好的版本）
│   ├── udet_v044.yaml           # v0.4.4：多尺度样本头（m4 最好的版本）
│   ├── udet_v04_cons.yaml       # v0.4.1：门控 + A/B2/B3 损失
│   ├── udet_v042.yaml           # v0.4.2：+ 分块编码器（static+compile）
│   ├── udet_v043.yaml           # v0.4.3：单任务消融（负结果）
│   ├── udet_v04.yaml            # v0.4.0：门控
│   ├── baseline_codet5_tok.yaml # CodeT5 滑窗基线（同口径对照基准）
│   ├── udet_v03.yaml            # v0.3：窗口注意力编解码器（栈内共享）
│   ├── udet_lora.yaml           # v0.2 频域 U-Net + CodeT5 LoRA
│   └── udet_base.yaml           # v0.2 频域 U-Net + 可训练 LUT
├── encoders/                    # 编码器（按名称切换，单文件实现）
│   ├── __init__.py              #   build_encoder() 注册表
│   └── codet5.py                #   codet5tok（逐 token 查表）/ codet5lora（逐 token + LoRA）
│                                #   / codet5blk（★ 分块 K=128 + LoRA）/ codet5（整段，受 512 限制）
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
│   ├── compare.py               #   汇总 runs/ 下所有实验的 val/test 指标
│   ├── analyze_raw.py           #   ★ 离线分析 raw_*.pt：长度分桶 + 阈值扫描（不占 GPU）
│   ├── probe_scales.py          #   ★ 零训练探针：逐尺度 token 表现 + 尺度权重离线定标
│   ├── probe_report.py          #   ★ 零训练探针：报告统计量信息量（CPU）+ 报告污染量化
│   ├── probe_layers.py          #   ★★ 零训练探针：**逐层编码器信息量**（P5 token / P6 文档
│   │                            #      / P7 cut-and-stitch，回答“尾部能删几层”）
│   ├── probe_lut_layers.py      #   逐层 LUT 的无上下文文档级判别力（旧路径）
│   ├── check_v046.py            #   自检：长度权重 + 报告向量，含**向后兼容逐位回归**
│   ├── check_v049.py            #   ★ 自检：额外一路头 + 梯度缩放 + 瓶颈移植
│   │                            #     （`--encoder` 会用**真实瓶颈输入**量移植保真度）
│   ├── queue_v0*.sh             #   接力队列（追加 epoch / 批量落盘）
│   ├── rebuild_probe_stitch.py  #   从日志重建被覆盖的探针产物（B2 事后补救）
│   ├── watchdog_run.sh          #   夜间看门狗（参数化：`<TAG> <CFG> <EPOCHS>`；PID 单例；
│   │                            #   判据含“eval.json 比 best.pt 新”，**从不杀进程**）
│   ├── diag_m4.py               #   诊断：按长度分桶 + 各尺度特征的线性探针
│   └── probe_m4.py              #   诊断：train 拟合 → val 评测的线性探针
├── docx/                        # 设计与实验报告
│   ├── report.md               #   ★ 面向专家读者的现状汇报（模型 / 实验 / 结果 / 下一步）
│   ├── lessons.md              #   ★★ 经验教训（动手前必读）
│   └── u-det*.md               #   总设计 + 逐版本报告（最新 u-det-v0.4.md）
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
# v0.4.4（当前最优）：分块 CodeT5 + LoRA + 窗口注意力编解码器（门控）+ 多尺度样本头
OMP_NUM_THREADS=8 python train.py --config configs/udet_v044.yaml --tag v0.4.4
python train.py --config configs/udet_v044.yaml --eval --ckpt runs/v0.4.4/best.pt

# 续跑追加 epoch（metrics.csv 里已出现的 epoch 不重复计，见 lessons D3）
OMP_NUM_THREADS=8 python train.py --config configs/udet_v044.yaml --tag v0.4.4 \
  --resume runs/v0.4.4/last.pt --epochs 4

# 冒烟（★ 只写 eval_limit<N>.json，绝不覆盖 eval.json，见 lessons B2）
OMP_NUM_THREADS=8 python train.py --config configs/udet_v044.yaml --tag smoke --limit 40 --epochs 1

# 落盘全量原始概率（供离线分析）
python train.py --config configs/udet_v044.yaml --eval --ckpt runs/v0.4.4/best.pt --dump-raw

# 离线分析（纯 CPU、不占 GPU）：长度分桶 + 阈值扫描，见 §8.10
python scripts/analyze_raw.py runs/v0.4.4 runs/v0.4.2

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
| `encoder.name` | `codet5blk` / `codet5tok` / `codet5lora` / `codet5` | **分块 K=128（v0.4.2 起默认）** / 逐 token 查表 / 逐 token + LoRA / 整段上下文 |
| `encoder.freeze` | `true` / `false` | `codet5tok`：LUT 固定 buffer / 可训练嵌入表；`codet5lora`/`codet5blk`：固定 LoRA / 训练 LoRA |
| `encoder.targets` | `[v, o, wi, wo]` | LoRA 目标层。**逐 token（seq_len=1）时 q/k 对输出无影响**（softmax 单元素恒为 1），只能挂 v/o/wi/wo；**分块后块内 attention 真实存在，q/k 重新有意义**（尚未启用，见 §8.11） |
| `encoder.block` / `block_batch` | `128` / `16` | 分块编码器的块长与打包批大小（每前向 ≤2048 token） |
| `encoder.static` / `compile` | `true` / `true` | `static` 把块数补齐到 `block_batch` 的整数倍以取得**固定形状**——这是 `compile` 生效的前提（否则反而 1.62× 变慢，见 lessons A6） |
| `encoder.ckpt` | `true` / `false` | 分块前向是否逐片重算以省显存 |
| `train.lr_encoder` | `5e-5` | 编码器/LoRA 单独学习率（**3e-4 会一 epoch 改写编码器特征**） |
| `model.name` | `codec` / `hier` / `codet5cls` | v0.3+ / v0.2 / 直接二分类基线 |
| `model.share` | `true` / `false` | 采样栈内 4 级是否共享同一个 Block |
| `model.gate` / `gate_init` | `true` / `-1.0` | 上采样融合用**门控**（`g*out+(1-g)*skip`）而非加法；`gate_init=-1` 使起点≈纯跳连 |
| `model.divs` / `wins` | `[4,4,2,2]` / `[16,16,16,32]` | 各级下采样除数（累计 64×）与感受野 |
| `model.sample_only` | `true` / `false` | 只留样本级通路、跳过全部上采样（v0.4.3 消融：省 35M 参数但掉 2.3 分，**已否决**） |
| `heads.sample_levels` | `1` / `5` | 样本头读取的尺度数。`1`=只看瓶颈；`5`=**每尺度各自池化后拼接**（v0.4.4 起默认，宽度 768→3840） |
| `heads.token_bypass` | `true` / `false` | 最细尺度 token 头额外拼接**编码器全长逐 token 特征**（768→1536，v0.4.5） |
| `heads.report_dim` / `report_proj` | `7` / `32` | ★ v0.4.6：样本头接入**文档级报告向量**（`0` = 不接，与历史**逐位一致**） |
| `report.name` | `handcrafted` / `none` | 是否注入手工统计报告 |
| `report.mode` | `prefix` / `vector` / `none` | ★ v0.4.6：`prefix`（默认、历史行为，报告作为 token 前缀进序列）/ `vector`（**不进序列**，7 维数值喂文档级头） |
| `report.stats_mean` / `stats_std` | 7 个数 | v0.4.6：7 维统计量的**固定**标准化常数（在 m4 train 上算得，见 `scripts/probe_report.py`） |
| `loss.token` | `0` / `1.0` | token 级监督权重（0 = 纯样本级） |
| `loss.token_weight_mode` | `static` / `length` | ★ v0.4.6：`static`（默认、历史行为）/ `length`（$w_i=\sigma((\log_2 n_i-\log_2 N_{\min})/\tau)$，短文档自动向高分辨率尺度集中） |
| `loss.token_weight_nmin` / `token_weight_tau` | `64` / `0.5` | ★ v0.4.6：`length` 模式的锚点与温度（取值由零训练探针扫得，见 §8.12.5） |
| `train.sample_streams` | `[m4]` | 只在标签有意义的流上算样本级 CE |
| `train.streams` | `[m4, hybrid]` | 双流；`--no-m4` / `--no-hybrid` 单流 |

## 实现要点

- **分块编码器**（`encoders/codet5.py: CodeT5BlockEncoder`，v0.4.2 起默认）：把整段切成
  `block=128` 的块、打包成 `(block_batch, 128)` 一批过 LoRA CodeT5 —— **块内 attention 真实存在，
  块间彼此独立**。★ **块的输入/输出向量数严格 1:1**（K 进 K 出，不池化、不重叠、不跨样本边界），
  所以对下游完全透明（仍返回 `(B, L, D)`，`models/hier2.py` 一行都不用改）；长度也仍然无上限 ——
  既不受 `n_positions=512` 限制，也没有整段 O(L²) attention。只有**末块**用 `pad_id` 补齐并在
  attention 里屏蔽，输出再按 mask 置零；`static: true` 额外把**块数**补齐到 `block_batch` 的整数倍
  以取得固定形状 —— 这是 `compile` 能生效的前提（见 lessons A6）。
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
  ⇒ 单靠长度规则 val acc 就有 0.659，报告样本级指标时需注意；**因此 m4 的 `>2048` 分桶毫无意义**
  （F1 退化成 0/0，见 lessons C2）。
- **短文档才是当前瓶颈**：v0.4.4 在 hybrid 三个短桶上全面落后基线、只在 `[8192,∞)`（**仅 5 个样本**）
  反超 ⇒ 详见 `docx/u-det-v0.4.md` §8.10.2，方法论见 lessons A4。
- 更多数据陷阱（恒定标签、退化流会污染 `monitor`）见 **`docx/lessons.md` C 类**。

## 数据与许可

- 数据集：CoDET-M4（MIT，<https://huggingface.co/datasets/DaniilOr/CoDET-M4>）
- 数据集：HybridCodeAuthorship（Apache-2.0，<https://github.com/CapitalOne-Research/c1-hybrid-code-authorship>）
- 编码器权重：Salesforce/codet5-base（BSD-3-Clause）
