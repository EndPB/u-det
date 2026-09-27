# d-det：从 RLHF 闭式解出发的双维 AI 代码检测

> 在 `u-det` 仓库内创建的新项目（复用同一 repo 与 conda 环境），**代码与 u-det 零依赖**。
> 方法：把检测信号按 $p_{\text{RLHF}} \propto p_{\text{base}}\, e^{r/\beta}$ 强制分解为
> **$s_1$ 众数坍缩度**（法向；人 vs AI 监督）与 **$s_2$ 偏好位移度**（切向；
> base vs instruct 差分，**零人类标注**），在 $(s_1, s_2)$ 平面上联合完成
> 检测、伪装识别与家族归因。完整推导见 [`docx/d-det.md`](docx/d-det.md)。

> ★ **冻结方案 v1.0（2026-09-24）**：后续汇报与 SemEval 工作以
> [`docx/d-det-v1.0-freeze.md`](docx/d-det-v1.0-freeze.md) 为准（权重 = `runs/v0.4.1_covreg/last.pt`，双轴融合决策）。

---

## 与 u-det 的关系

- **同一个 git 仓库、同一个 conda 环境（`udet`）**：环境安装一次、资产复用；
- **代码独立**：`d-det/` 不 import `u-det/` 的任何模块。复制进来的组件
  （`encoders/`、`dataio/` 的模板、`requirements.txt`）在本目录内独立演化；
- **数据与权重完全独立**：`data/processed/`（m4 / hybrid / LUT）与
  `checkpoints/codet5-base/` 已复制进本目录（约 1.1 GB），不需要软链或相对路径；
- 新增资产（配对数据、Qwen 模型对）也只落在本目录。

## 目录结构

```
d-det/
├── train.py                  # 训练/评测主入口（流式：m4 BCE + hybrid token BCE + pair margin + MLM 辅助）
├── configs/ddet_base.yaml    # 首版配置（v0.1 LoRA）
├── configs/ddet_v030.yaml    # v0.3 全参微调 + 四路监督
├── configs/ddet_v040.yaml    # v0.4 几何损失（斥力/Fisher/跨族方向；当前最新）
├── configs/ddet_v041.yaml    # v0.4 + 去冗余白化对照臂
├── encoders/                 # 复制自 u-det（codet5/codet5tok/codet5lora/codet5blk）+ `full_ft` 分支（v0.3）
├── models/scores.py          # DualScoreModel：编码器 → 池化 → s1/s2 双读出 + tok_head（v0.3）
├── dataio/                   # m4 + pair + hybrid（token）+ pairxf（跨族 Δ；v0.4）
├── scripts/
│   ├── prepare.py            # 下载同族 base/instruct 模型对 + prompt 集（hf-mirror）
│   ├── gen_pairs.py          # 生成配对数据（--pair-name 支持多族；零标注监督来源）
│   ├── selfcheck.py          # 接线自检（起点等价 / 首步梯度 / 参数清单）
│   ├── probe_sensitivity.py  # 选择性敏感性（v2：信息保持型扰动 + 口径修复）
│   ├── probe_fragments.py    # 片段任务探针（函数 → 首/中/尾片段；v0.2）
│   ├── probe_scale_verify.py # 跨规模/跨族核验（s1/s2/融合 + 重复 CV + npz 落盘；v0.2+）
│   ├── decompose_scores.py   # 分数分解（s1 单独/融合/交叉融合矩阵；v0.3 归因）
│   ├── analyze_scores.py     # 离线：阈值扫描 / 长度分桶 / 2D GMM / 象限 / 正交度
│   ├── compare_scores.py     # 两个 run 的分数结构对比（相关性 / 错误重叠；v0.2）
│   ├── calibrate_margin.py   # 斥力 margin 标定（P5 分位；v0.4）
│   ├── check_v040_grads.py   # 新损失首步梯度冒烟（死锁家族第 4 课；v0.4）
│   └── queue_*.sh            # 后台接力队列（v0.1…v0.4；v0.4 起支持自动衔接）
├── docx/                     # 设计文档 + 版本报告（v0.1–v0.4 / v1.0-freeze / semeval / kernel）
├── data/processed/           # m4 / pairs（0.5B）/ pairs_qwen15 / hybrid / codet5_lut
└── checkpoints/              # codet5-base + Qwen2.5-Coder 0.5B/1.5B 模型对
```

## 当前状态（2026-09-26）

| 项 | 状态 |
|---|---|
| 代码（训练/数据/脚本/配置） | ✅ 全链路跑通（selfcheck→生成→训练→评测→分析） |
| 设计文档 | ✅ `docx/d-det.md`（含实现映射与工程决策记录） |
| 数据：m4（s1 监督） | ✅ 16107/1991/1940（train/val/test） |
| 数据：pairs（s2 监督） | ✅ qwen-0.5B **1303 对**（1048/124/131）｜qwen-1.5B 394 对｜**DeepSeek-1.3B 400 对** |
| 数据：hybrid（token 级） | ✅ 就位（v0.3 启用：tok_head + line/chunk/token 评测） |
| 权重：codet5-base + Qwen 0.5B/1.5B 对 + DeepSeek-1.3B 对 | ✅ 全部就位 |
| 训练 / 验证 | ✅ **v0.1.0 完成**：m4-test acc 0.9552 / AUC 0.9881，pair dir_acc 1.0（见 `docx/d-det-v0.1.md` §5） |
| 消融 | ✅ A（--no-pair）：单流 m4-test 0.9567（s2≡0）｜✅ B（loss.orth=0）：**\|cos\|=0.74，正交需正则**（见 v0.1 文档 §5.6） |
| v0.2：片段任务 | ⚠️ 可行但**非差异化**（单头基线同样/更好；见 §1.3/§3.6） |
| v0.2：消融 C / 跨族泛化 | ✅ 全部完成：C3 abmil（test 0.9582）｜C4：s2 零训练泛化 1.5B（dir_acc 97.2%）｜⚠️ 基线对照（§3.6）：未超 abA 口径（已被 v0.3 翻越） |
| **v0.3.0（全参微调 + 四路监督）** | ✅ **2ep：m4/test F1 0.9776（acc 0.9773 / AUC 0.9966）——首次正超真基线；s2 首次可迁移**。4ep：F1 0.9756（m4 过拟合，hybrid 仍爬：line 0.7845 / token 0.8131）。见 `docx/d-det-v0.3.md` |
| **v0.4.0（几何损失 = 斥力+Fisher+跨族方向）** | ✅ 2ep：m4/test **F1 0.9766**；片段 **mid 0.9026（自 4ep +1.9pt）**、head/tail 新最佳；hybrid line **0.7881**（微超 u-det v0.4.4）；**s2\* ≈ 0.9998/1.0（跨族）**；⚠️ instruct s1 回落 0.86。见 `docx/d-det-v0.4.md` |
| **v0.4.1（+ 去冗余白化对照臂）** | ✅ 2ep：m4/test **F1 0.9775**（追平 u-det v0.4.4 / 2ep 峰值）；hybrid 0.7877 / 0.5008 / 0.8167；白化未修复 s1-OOD。见 `docx/d-det-v0.4.md` |
| **SemEval-2026 Task 13 适配（外部基准）** | ✅ **六杠杆总攻（2026-09-25）**：**B-test 0.420**（分层集成：Qwen1.5B 骨干 val .665 + 60K 数据 + 19 成员栈，旧 .304）；**C-test 0.827**（Qwen 单臂 .810 已超旧集成）；**A 0.781**（等权三源融合：s1+rvoid+Qwen 探针，超旧 0.749）；低秩 s2 **r\*=8**。详见 `docx/d-det-semeval.md` §6–§8 + `docx/d-det.md` 修订注记 |
| **v0.5（判别头工程化 + 扩族 3→6）** | ✅ 配对 Δ 判别头（`models/disc.py` `DiscHead`，闭式 LDA/shrinkage）：3 族 0.766 → **6 族跨题 0.7626 / 共同题 0.7239**（5 族对照 0.8121 / 0.7605）；混淆呈"相近对"结构（granite2b↔smollm2 互混）；归因=判别式经内核 E1–E14 实锤。见 `docx/d-det-kernel.md` §5.1–§5.2 |
| **v0.6（视图升级，E15–E17）** | ✅ **归因主视角=双侧联合视图 [h⁺;h⁻]**（对称/反对称分量单独都弱、联合才强，E17e）：非转导全量 **0.8271**（+6.5pt）；转导中心化 [h̃⁺;h̃⁻] 对齐 **0.8571**（+8.3pt）；易混对二分中心化 0.90–0.97；E11 二族 0.977（解 P1 0.98 之谜）；中心化增益 k=2–6 稳定。规范脚本 `scripts/disc_v06_views.py` → `runs/v0.6_views/`。理论一页=§3.12。见 `docx/d-det-kernel.md` §3.9–§3.12 |
| **v0.7（稳定版冻结，三件套）** | ✅ 核心改进=双侧联合视图（**零训练/零调参/确定性**）：**二分类 4 组** .734→**.904** / .850→**.965** / .780→**.873** / .885→**.977**；**四分类 3 组** .762→**.904** / .840→**.929** / .769→**.892**；**归因** .763→**.827**（对齐 .774→**.857**）。套件 `scripts/eval_stable_v07.py` → `runs/v0.7_stable/`；独立说明文档 `docx/d-det-v0.7-standalone.md`（零基础可读） |

## 快速开始

### 无卡模式（历史阶段）

初版代码在无卡模式下写成，**未运行任何 python**；此节保留作为阶段说明。当前已进入有卡阶段（下）。

### 有卡模式（命令存档，已按序执行完毕）

```bash
cd /root/autodl-tmp/u-det/d-det
export OMP_NUM_THREADS=8                      # 本项目环境变量异常，必须显式设置
# conda activate udet  （或直接使用 /root/miniconda3/envs/udet/bin/python）

# 1) 下载同族模型对 + prompt 集（hf-mirror，无需代理）
python scripts/prepare.py pairmdl             # Qwen2.5-Coder-0.5B{,-Instruct}，约 1.9 GB
python scripts/prepare.py prompts             # bigcodebench + humaneval，约 15 MB

# 2) 接线自检（起点等价 / 首步梯度；不通过就不要往下走）
python scripts/selfcheck.py --config configs/ddet_base.yaml

# 3) 生成配对数据（先小规模估时，再全量）
python scripts/gen_pairs.py --limit 100 --out data/processed/pairs_smoke.parquet
python scripts/gen_pairs.py --out data/processed/pairs.parquet

# 4) 冒烟训练（小样本 1 epoch；先单流 s1、再联合）
python train.py --config configs/ddet_base.yaml --tag smoke_s1 --no-pair --limit 200 --epochs 1
python train.py --config configs/ddet_base.yaml --tag smoke --limit 200 --epochs 1 \
    --set data.pair_file=pairs_smoke.parquet

# 5) 正式训练（2 轮起步；判停看"周期末端"是否还创新高，不看 loss —— u-det 教训 A1/A2）
python train.py --config configs/ddet_base.yaml --tag v0.1.0

# 6) 评测 + 落盘逐样本分数（离线分析的数据来源）
python train.py --config configs/ddet_base.yaml --eval --ckpt runs/v0.1.0/best.pt --dump-scores

# 7) 离线分析（不占 GPU）与敏感性验证
python scripts/analyze_scores.py runs/v0.1.0
python scripts/probe_sensitivity.py --config configs/ddet_base.yaml --ckpt runs/v0.1.0/best.pt

# 8) v0.3 全参微调（四路监督：m4 BCE + hybrid token BCE + pair hinge + MLM；每 epoch ≈2.2h）
python train.py --config configs/ddet_v030.yaml --tag v0.3.0_fullft
python train.py --config configs/ddet_v030.yaml --tag v0.3.0_fullft \
    --resume runs/v0.3.0_fullft/last.pt --epochs 2            # 续跑 +2ep（累计 4ep）
python train.py --config configs/ddet_v030.yaml --eval --ckpt runs/v0.3.0_fullft/best.pt --dump-scores
python scripts/probe_fragments.py    --config configs/ddet_v030.yaml --ckpt runs/v0.3.0_fullft/best.pt --n 500
python scripts/probe_scale_verify.py --config configs/ddet_v030.yaml --ckpt runs/v0.3.0_fullft/best.pt \
    --tag v0.3.0_fullft --pair-file data/processed/pairs_qwen15.parquet --npz runs/v0.3.0_fullft/scale_scores.npz

# 9) v0.4 几何损失两臂（基线 = v0.3.0 4ep 的 last.pt；各 2ep；预算纪律：单方法 ≤4ep）
python scripts/calibrate_margin.py                       # P5 标定 margin（读 runs/v0.3.0_fullft/scores_m4_test.pt）
python scripts/check_v040_grads.py                       # 新损失首步梯度冒烟（CPU，须 PASS）
python train.py --config configs/ddet_v040.yaml --tag v0.4.0_geom \
    --resume runs/v0.3.0_fullft/last.pt --epochs 2 --set loss.rep_margin=4.528
python train.py --config configs/ddet_v041.yaml --tag v0.4.1_covreg \
    --resume runs/v0.3.0_fullft/last.pt --epochs 2 --set loss.rep_margin=4.528
bash scripts/queue_v040.sh                               # （存档）自动接力：等待→冒烟→两臂→探针→汇总
```

## 数据说明

- **m4**（`data/processed/m4.parquet`，2 万条）：human/AI 样本级标签 → $s_1$ 的 BCE。
  ⚠️ 已知陷阱（u-det lessons C2）：**长度与标签强混淆**（AI 全部 <2048 token，
  4096+ 全为 human）。评测必须带长度分桶（`analyze_scores.py` 已内置）。
- **pairs**（`data/processed/pairs.parquet`，1303 对）：同 prompt 下
  **instruct 输出（$x_+$）/ base 输出（$x_-$）** 的配对 → $s_2$ 的 hinge margin。
  约定：`x_plus` = instruct、`x_minus` = base；生成时 base 走 completion 前缀、
  instruct 走 chat 模板（各自最自然的用法，同一道题）。
- **hybrid.parquet**：行级标注 + "同文件人/AI 改写对"；v0.3 起启用 token 级监督（tok_head）与 line/chunk/token 评测。
- **SemEval-2026 Task 13**（`data/raw/SemEval-2026-Task13/`，~1.2GB，2026-09-24 下载）：A 二分类（含未见语言/域设置）/ B 家族归因（11 类）/ C 混合检测（Human/Machine/Hybrid/Adversarial）；指标 macro F1；适配实验与报告见 `docx/d-det-semeval.md`（三臂 + 冻结探针 + R_void 复现 + HF 500K 分布验证）。

## 工程铁律（继承自 `../docx/lessons.md`，违反过都出过事）

1. checkpoint 存 `model.state_dict()` 的可训练子集（**不是** `named_parameters`）；
2. `--eval` 必须用**训练时那份 config**（本项目的防呆会比对 `encoder/model` 段并中止）；
3. `--limit` 的冒烟评测写 `eval_limit<N>.json`，**绝不覆盖**正式产物；
4. 架构 / 数据对比**对齐 epoch 预算**；2 轮的数字只能用来排优先级；
5. 后台任务日志直接重定向（`> log 2>&1`），不要过管道；
6. 每次操作前 `export OMP_NUM_THREADS=8`；
7. **以 best.pt 为契约的下游必须设保底**：续跑臂若 monitor 未超 resume 分，`best.pt` 可能不存在
   （v0.4.1 实测：0.9232 ≤ 0.9233）——训练侧已保底（best=last），脚本侧仍建议 `[ -f best.pt ] || CKPT=last.pt`。

## 文档索引

- `docx/d-det.md` —— 设计文档（推导 + 实现映射 + 工程决策记录）
- `docx/d-det-v0.1.md` —— v0.1 实现清单、数据调研、执行记录与结果（§5：指标/分析/敏感性/消融）
- `docx/d-det-v0.2.md` —— v0.2：片段任务、敏感性 v2（口径修复+归一化）、消融 C、工程记录
- `docx/d-det-v0.3.md` —— v0.3：全参微调 + 四路监督（2ep 首超真基线 / 4ep 过拟合判读 / s2 可迁移）
- `docx/d-det-v0.4.md` —— v0.4：几何损失两臂（斥力/Fisher/跨族方向/去冗余）+ SemEval-2026 Task 13 外部基准附录
- `docx/d-det-v1.0-freeze.md` —— **冻结方案（v1.0）**：单一最优配置 + 双轴融合协议 + 交叉融合证据 + SemEval 适配计划
- `docx/d-det-semeval.md` —— **SemEval-2026 适配报告**：三臂基线 + 冻结探针/R_void + 六杠杆总攻（Qwen/大数据/集成）+ 改进方向 §8
- `docx/d-det-kernel.md` —— **内核实验报告（E 系列，2026-09-25）**：真实 $r^*$ 检验、双轴几何诊断、判别式归因与收内散间（E1–E14，含负结果）；§5.1–§5.2 = v0.5 判别头工程化与 3→6 族扩族验证
- `docx/d-det-kernel-handoff.md` —— **内核改进研究任务书**（自包含、面向接触不到代码的外部 AI：理论/管道/E1–E14 结论/v0.5 数字/六大缺口/E15–E20 候选与假说/论文关键词/数据一览；对方出思路与实验设计，团队执行并反馈）
- `docx/d-det-v0.7-standalone.md` —— **v0.7 稳定版独立说明书**（零基础读者：背景扫盲/公式逐符号解释/维度/数据/方法/三大任务结果表/术语表；不依赖任何其他文档）
- `../docx/lessons.md` —— u-det 教训库（通用工程铁律，两项目共用）
