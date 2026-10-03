# d-det · ACL 数据集合 v1 第二轮报告：三个诊断问题（捷径 / 消融 / 可训练表示）

> ⚠️ **评测修正横幅（2026-10-03 · round3 审计）**：本报告 MLP 相关数字（late-fusion/full-disentangle 及其消融、semantic/fingerprint 分支）使用旧的表示提取流程（含 Dropout/BN 的模块未切 `eval()`），已被修正版取代：late-fusion **.7381±.0043**、full-disentangle **.7356±.0021**、lf−SupCon 差值 **+1.1pt（配对 CI 跨零）**、其余机制项无一致效应；LoRA 数字（.7286±.0140）经 checkpoint 逐位复算有效。**MLP 相关结论以 `docx/d-det-acl-dcan-round3-audit-report.md` 为准**；本报告正文数字保留原样存档。

> 执行规范：`docx/d-det_AutoDL_3420857之后服务器AI继续指导_2026-10-03.md`。
> 脚本：`scripts/dcan_round2_shortcuts.py`（Q1）、`scripts/dcan_round2_ablations.py`（Q2）、`scripts/dcan_round2_trainable.py`（Q3）。
> 产物：`artifacts/acl_dcan_round2/{split_manifest,env}.json + shortcuts/ + ablations/ + trainable/ + README.md`；唯一 checkpoint `runs/acl_dcan_round2/best_lora.pt`（1.2MB，不入库）。round1 产物未改动。
> 定位：**诊断轮，不写 SOTA**（同数据/同切分/同 backbone/3 seeds 且 generator-held-out 不下降的条件未满足）；与 round1 及 E35/E36/v2/AuthorBench 分开陈述。

## 一句话结论

在 `h2_authorbench_dcan` 任务级留出上：**① TF-IDF（.7639）的优势不来自长度/语言/元数据捷径，而依赖代码内容 token**（metadata-only .4753、去标识符/字符串后 .4572、冻结表示线性探针 .4844；纯骨架+模板只剩 ≈.46–.48）；**② 五个单变量开关没有一个使 DCAN 实质超过 .713**（lf−SupCon .7153±.0069 仅噪声级、fd 去任一机制 ≈ .707），机制项（SupCon/两种 GRL/协方差）在本设置下均无可见贡献——瓶颈不在被消融项；**③ 可训练表示里 LoRA 最优**：3 seeds **.7286±.0140**（> late-fusion .7134 > head-only .6881），但**仍低于同切分 TF-IDF .7639**（差 3.5pt），且 file/generator-held-out 本轮未新增实验，无"同时改善"证据。

## 1 Q1 · TF-IDF 优势是否来自捷径（任务留出；test 不参与选择）

同一任务级 train/dev/test；除 char TF-IDF（round1 复用 .7639）外全部在本轮新算：

| 方法 | macro-F1 | balAcc | ECE | 长度桶 acc（q1→q4） |
|---|---:|---:|---:|---|
| char TF-IDF（参照，复用） | **.7639** | .7755 | .079 | .673 / .709 / .788 / .874 |
| metadata-only（31 维结构统计 → LR） | .4753 | .4876 | .038 | .519 / .547 / .516 / .649 |
| filtered TF-IDF（去注释/字符串、标识符→`_id`） | .4572 | .4698 | .341 | .407 / .390 / .420 / .608 |
| CodeT5 raw 768d → LR | .4844 | .4976 | .140 | .555 / .544 / .547 / .704 |
| CodeT5 任务内中心化 → LR | .4522 | .4626 | .124 | .558 / .533 / .522 / .644 |
| CodeT5 中心化+训练任务标准化 → LR | .6376 | .6358 | .137 | .626 / .629 / .613 / .715 |

- **读法**：语言是常量（C-only），长度桶显示"更长更好"但 q1 的 .67（char TF-IDF）已远超机会 .167 ⇒ 长度不是主因；metadata-only .475（2.8×机会）说明结构信号可观但不是优势主体；**把标识符/字符串/任务词剥离后，char TF-IDF 从 .764 掉到 .457（−30.7pt）**——优势的主体来自内容 token（可能是任务语义/命名习惯与 family 的耦合，而非纯粹"风格"）。
- **转导口径声明**：z_center/z_center_std 使用 test 任务的兄弟输出（同 task 内去均值/标准化），**不是单样本部署方法**；raw 是可比的无转导参照（.4844）。center_std 的提升（.638）与 AuthorBench 机制审计（z_center_std 增益）方向一致，但仍 < TF-IDF。
- 逐 family/generator/长度桶明细见 `shortcuts/metrics.json`。

## 2 Q2 · DCAN 失败的首要原因（单变量消融，3 seeds）

变体与原实现**逐行对齐**（一次只关一项；lr/wd/采样/seed/评测不变；dev 规则与 round1 相同并记录在案）。融合输出（fuse）为主对照：

| 变体（关闭项） | fused macro-F1 | vs 基线 | dev 规则 |
|---|---:|---:|---|
| round1 late-fusion（含 SupCon；参照） | .7134 ± .0060 | — | 双分支平均 |
| **lf − SupCon** | **.7153 ± .0069** | +0.2pt（噪声级） | 双分支平均 |
| fd − semantic GRL | .7073 ± .0084 | −0.0pt vs fd .7030 | fingerprint head |
| fd − fingerprint GRL | .7074 ± .0090 | +0.0pt vs fd | fingerprint head |
| fd − 协方差项 | .7071 ± .0088 | +0.0pt vs fd | fingerprint head |

- **没有任何单变量修改使 DCAN 实质超过 .713**；SupCon、semantic-GRL、fingerprint-GRL、正交项四个机制在全部 3 seeds 上均可移除而不改变结果（±0.01 内），即**失败首因不在这些机制**。结合 round1（融合 > 单分支、disentangle ≯ late-fusion）与 Q1（TF-IDF 依赖内容 token），瓶颈更符合"融合表示容量/目标与内容信号利用方式的差距"，而不是某个正则项的开关。
- 过程透明：初版消融把 round1 的 `CE(head_s)` 误删，导致语义分支崩溃（.15）——已按"先逐行对齐原实现、再关单机制"修复并重跑（本表为修复后结果）。
- 逐 generator 召回、balanced acc、ECE（含每变体每 seed）见 `ablations/metrics.json`。

## 3 Q3 · 仅在现有 CodeT5 资产上的可训练表示（无新下载）

顺序按要求做小规模冒烟（1 seed/3ep）→ 淘汰 → 正式复现：

| 方案 | 冒烟 dev→test（3ep） | 正式（3 seeds）macro-F1 | 备注 |
|---|---|---|---|
| head-only（冻结 mean-pool；round1 参照） | — | .6881 ± .0131 | 40ep 预算 |
| last_block（block[-1]+final LN+head） | dev .17→.49；test .4867 | 未正式（慢热，被淘汰） | 65s/ep |
| **LoRA(r8, q/v)+head** | dev .48→.61；test .6067 | **.7286 ± .0140**（.7088/.7372/.7399） | 299,526 参数；~10GB VRAM；~22min/seed |

- LoRA 逐 seed：balAcc .7325/.7539/.7538（均值 .7467）；ECE .108/.060/.031（均值 .066）；dev 在 ep9–12 仍缓慢上升（**预算截断，非调优上限**）。
- 逐 generator（seed2）：gemini .933 / gpt-4.1 .868 / llama .866 / claude .864 / qwen .635 / gpt-4o-mini .586 / deepseek .574 / gpt-4o .469。
- 结论：**训练表示确实是目前最有希望的方向（> late-fusion .7134 > head-only .6881），但 3 seeds 平均仍低于 TF-IDF .7639**；在 12 epoch 小预算与单一数据集下不足以写"优于基线"。
- 唯一 checkpoint：`runs/acl_dcan_round2/best_lora.pt`（最佳 seed 的 LoRA+head，1.2MB）。

## 4 对指导四个输出问题的回答

1. **TF-IDF 的优势是否主要来自任务/长度/语言捷径？** —— 不是主要来自长度/语言；**主要来自内容 token**（去标识符/字符串后 −30.7pt）；metadata（结构）单独有 .48 的上限，说明"捷径"存在但不是优势的主体。
2. **哪一个单变量修改使 DCAN 超过 .713？** —— **没有**（lf−SupCon 仅 +0.2pt 噪声级；fd 三变体 ≤ .707）。被消融的五个机制都没有可测贡献。
3. **是否同时改善 task-/file-/generator-held-out？** —— 本轮只有 task-held-out 有新实验，且没有任何修改超过 TF-IDF；file/generator 未新增实验（不引用未测结论）；**无"同时改善"证据**。
4. **是否仍然低于 .7639 的同切分 TF-IDF？** —— **是**：全部消融变体 .70–.72、LoRA .7286±.0140，均低于 .7639（差距 ≥3.5pt）。

## 5 边界与诚实性声明

- 未下载任何新模型/权重；未复制全量 embedding；未把 parquet 转 JSONL；未保存每 seed 完整模型（仅 1 个 LoRA checkpoint，1.2MB）。
- 系统评估仅 `h2_authorbench_dcan`（C-only、单划分协议）；Droid/AICD/STACAD 未新增实验（沿用 round1 数字与规则，AICD 仍只用 numeric-id，跨 split 哈希风险 1,848 条记录在案）。
- test 任务不参与任何选择；dev 仅 epoch 选择；z_center 系列明确标注为**转导**；LoRA 预算 ≤12ep（dev 未完全平台化），不构成调优上限结论。
- 未同时调 lr/温度/维度/门控/损失权重（消融仅关单机制）；未改采样与评测预算。
- 不作 SOTA 声明：同数据/同切分/同 backbone + 3 seeds + generator-held-out 不下降的条件未满足。

## 6 复现与交付清单

```bash
cd /root/autodl-tmp/u-det/d-det
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
P=/root/miniconda3/envs/udet/bin/python
$P scripts/dcan_round2_shortcuts.py                                   # ~46s
$P scripts/dcan_round2_ablations.py --seeds 3 --epochs 40             # ~228s
$P scripts/dcan_round2_trainable.py --smoke --scheme both             # ~562s（筛选记录 smoke_metrics.json）
$P scripts/dcan_round2_trainable.py --scheme lora --seeds 3 --epochs 12   # ~4024s
```

交付对照：`artifacts/acl_dcan_round2/`（config 说明在 README + env.json + split_manifest.json + metrics.json ×3 + 压缩预测 ×3 + 唯一 checkpoint 路径 + 本报告）。
