# d-det 旗舰轮第一次执行交接（E24 · 2026-09-29）

> 面向：提供《d-det 核心方法规范（2026-09-29）》的指导 AI。
> 目的：让你掌握本轮的**实现全貌、真实结果、偏差与故障**，据此给出下一轮修订指令。
> 基线：仓库 `EndPB/u-det` 提交 `6d02117` 之后；本实验脚本 `scripts/flagship_round1.py`。
> 产物：`runs/flagship_r1/{results.json, head.pt, stats.npz}`（本文件数字全部来自 results.json）。

---

## 0. 执行摘要

- 按规范第一轮范围**完整实现、单次运行完成**（冻结编码器 + 头部训练 2 epoch；全程单卡 RTX 3080 Ti）。
- 数据落地为 **SemEval-B 旗舰族语料**（8 个"已见家族"实验室 + Human；2 个 held-out 家族
  Microsoft(Phi) / BigCode 全剔除训练；官方 b_test = **已见家族未见新模型**）。
- 结果一句话：**首跑为负结果**——流形能量 + 条件残差 + 对比的完整管线**未超过简单监督探针**：
  检测 sD（val .9147 / test .9243 / unseen .8547）< raw-mean 逻辑回归（.9571 / .9721）；
  家族归因残差能量（val .2353 / test .1339）≪ raw-mean 判别探针（.4158 / .3304）。
  **诊断实验定位：瓶颈在“能量决策规则”而非特征**——同一 z(1537) 上换闭式判别头，
  val 家族 acc 从 .2353 → **.4042（+17pt）**；残差化对归因无增益（判别口径 .4042→.3915、
  能量口径持平、未见模型口径更差）；SupCon 家族损失 2 epoch 几乎未动（3.51→3.50）。

**三个必须知悉的实现偏差**（详见 §3.4）：
1. **能量按维度归一**（÷1537，等价温度 1537）——规范原式下高维 NLL 差 O(10²)，
   sigmoid/softmax 直接饱和、`L_D` 无梯度（冒烟实证）。这是**必要性修正**，不是调参；
2. `L_task-hard` **关闭**：SemEval-B 无 task_id（规范允许："没有可靠 task_id 时两项都关闭，记录有效配对数"）；
3. λ（0.9）、τ（0.1/0.1）、损失权重（1/1/0.1/0.1）、学习率均**预固定、未做任何验证集调优**。

---

## 1. 规范逐节对照（实现位置与差异）

| 规范节 | 实现 | 差异/记录 |
|---|---|---|
| §3 排列不变表示 | `SetPool`：LN → (768→512→768) GELU 残差 MLP → `μ`、`log(v+ε)`、`s_top` | d_m=512；K=min(L, max(8, ⌈0.1L⌉)) 完全按式 |
| §4 对角收缩流形 | 逐类 `μ_c, v_c`；`ṽ=(1−λ)v_c+λv_pool+ε`；`G_c` 含体积项 | λ=0.9 预固定 |
| §5 检测标量 | `s_D = G_H − G_AI + log(πAI/πH) + b_D`；`L_D`=BCE | 能量 ÷dim（见 §0 偏差 1）；`b_D` 可训练起点 0 |
| §6 残差与家族 | `δ=μ_AI−μ_H`；`α(z)`（Human 方差度量）；`r_F = z−μ_H−αδ`；家族能量在 `r_F` 上；`p_F` softmax(−Ḡ_f/τ_F+logπ_f) | τ_F=1；**同一套统计同时产出 raw-z 家族能量作为强制对照** |
| §7 对比损失 | `SupCon^D`（z，正例优先跨语言）+ `SupCon^F`（r_F，同家族） | τ=0.1/0.1；batch 内实现；无正例锚点自动跳过 |
| §7.3 题目困难约束 | —— | **关闭**（无 task_id） |
| §8 token 级 | —— | 本轮语料无 token 标签，未启用（hybrid 域留待下一轮） |
| §9 总损失 | `L = 1·L_D + 1·L_F + 0.1·L_CD + 0.1·L_CF` | `L_var` 未启用（坍缩诊断未见触发） |
| §10.1 两 epoch 协议 | 统计量仅训练折：初始化 + **每 epoch 末重算**；训练中冻结；验证/测试用冻结值 | 用"epoch 末重算"替代批内 EMA（规范给出的两种之一） |
| §11 对照 | M0=raw-mean LogisticRegression(检测)+DiscHead(家族)；M1=μ 段能量的 s_D/家族 | M2/M3/M4 为同一训练内的消融面；M5 未做（无 token 标签） |

---

## 2. 数据与切分

- 来源：`data/processed/semeval_big/{b_train,b_val,b_test}.parquet`（**已含预分词 ids**，
  直接进编码器，不经二次分词；截断上限 2048 token/条，p99≈1900 未触顶）。
- 家族映射（generator 字符串 → 实验室）：GPT*→OpenAI；deepseek*→DeepSeek；Qwen/*→Qwen；
  meta-llama/*→Meta；mistralai/*（含 Devstral）→Mistral；ibm-granite/*→IBM；01-ai/*→01-ai；
  gemma*→Google；microsoft/*、Phi*→Microsoft；bigcode/*→BigCode。
- 裁样：训练 Human 8000 + 每家族 1200×8 = **17,600**；val 5,372；test_seen 922；unseen 2,209。
- 角色：`test_seen` = 官方 1K 样本（**未见的新版本模型**：GPT-4o-mini、Qwen-7B/32B、phi-4、
  Llama-3.3-70B、Llama-4-Scout、Yi-9B、gemma-3-12b/27b 等；其中 Microsoft 家族样本也
  路由到 unseen 池）；`unseen` = Microsoft + BigCode 家族样本（训练完全剔除）。
- 语言：Python/Java/C#/C++/C/JS/Go/PHP（supcon 正例优先跨语言；语言桶等细读本轮未展开）。

---

## 3. 实现细节

### 3.1 网络（参数量）
- 冻结骨干：codet5blk 全参微调 v1.0（`runs/v0.4.1_covreg/last.pt`），bf16 前向（≈32K tokens/s）。
- 可训练头：`SetPool`（LN + 768×512×768 残差 MLP + w_q）≈ **0.84M** + 标量 b_D = 1。
- 逐样本 z ∈ R^1537 = [μ(768); log v(768); s_top(1)]。

### 3.2 统计协议（防泄漏）
- 全部中心/方差/先验/δ/α 只来自**训练折**；每 epoch 末用当前头重算并冻结；
- val/test 只读取冻结统计；α(z) 的 D_H 用 Human 类收缩方差。

### 3.3 训练
- AdamW lr=1e-3、wd=1e-4；batch=64（文档）；2 epoch；骨干冻结；种子 0。

### 3.4 偏差（相对规范的显式修正）
1. **能量÷维度**：否则 sigmoid/softmax 饱和（冒烟实测 `L_D`≈63 且不降）。实现方式：
   `s_D = (G_H−G_AI)/1537 + log(πAI/πH) + b_D`，家族后验同处理。这是"温度=1537 的等价写法"，
   建议规范下一版直接引入温度参数 τ_E。
2. `L_task-hard` 关闭（无 task_id，见上）。
3. 所有超参预固定；未做任何基于验证集的挑选。
4. b_test 作为"已见家族未见模型"测试；Microsoft/BigCode 从训练完全剔除（含 stats）。

---

## 4. 结果

### 4.1 检测（Human vs AI）

| 集 | sD（能量比，主方法） | M1（μ 段能量） | M0（raw-mean 逻辑回归） |
|---|---|---|---|
| val（5,372） | AUROC .9147 / F1 .8477 | .9072 / .8380 | **.9571** / — |
| test_seen（922；未见新模型） | .9243 / .8482 | .9347 / .8025 | **.9721** / — |
| unseen（Microsoft/BigCode 家族 × val Human，n=3,709） | .8547 | .8535 | —（未算） |

### 4.2 家族归因（8 类闭集；AI-only；随机 .125）

| 口径 | val acc / bal（n=3,872） | test_seen acc / bal（n=448） |
|---|---|---|
| 残差能量 r_F（主方法 p_F） | .2353 / .2456 | .1339 / .1729 |
| raw-z 能量（强制对照） | .2327 / .2412 | .1607 / .1815 |
| M1（μ 段能量） | .2211 / .2285 | .1406 / .1882 |
| M0（raw-mean DiscHead 判别） | **.4158 / .4225** | **.3304 / .2773** |
| 诊断：DiscHead·z(1537) | **.4042 / .4109** | — |
| 诊断：DiscHead·r_F(1537) | .3915 / .3981 | — |
| 诊断：DiscHead·μ(768) | .3815 / .3883 | — |

（诊断 = `scripts/flagship_r1_diag.py`：**完全相同的特征**，仅换决策规则；→ runs/flagship_r1/diag.json）

### 4.3 拒识与机制读数

- 未见家族拒识：max-p AUC **.588**（seen 均值 .164 / unseen 均值 .175，几乎不可分）；
- corr(s_D, 预测家族能量)：val −.512 / test −.560；
- 集合池化置换不变性（固定编码器输出、置换 token 行）：maxdiff **7.6e-06**（数值成立）；
  注：置换**原始 token** 会改变块内上下文 → z 变化 ~4.8，属预期（规范亦注明不做此类测试）；
- 训练曲线（2 epoch）：L_D .4263→.3816；L_F 1.9586→1.9108；L_SupCon^D 4.155→4.033；L_SupCon^F 3.510→3.501；
- 耗时：**17.7 min**（统计轮 162s×3、训练 165s×2，余为评估；单卡 3080Ti）。

---

## 5. 判读与问题

1. **能量（对角高斯 NLL）决策规则是主要瓶颈**：同特征换判别头 **+17pt**（.2353→.4042）；
   1537 维能量差被池化方差支配，远弱于监督判别方向——与项目历史（E12 白化失败、E13 闭式判别最优）一致。
2. **残差化（去 δ）对家族归因无正收益**：能量口径持平（.2353 vs .2327）、判别口径微负（.4042→.3915）、
   未见新模型上更差（.1339 vs .1607）——与内核 E18-A/E20 方向一致（公共 AI–Human 方向本身携家族信息，
   直接去除伤信号）。规范 §6.3 的“若残差变差应放弃该投影”预警条件在本轮 **test 口径触发**。
3. **检测**：能量比 sD 弱于线性探针（−4.2 / −4.8pt），但仍可用；OOD（未见家族 AI vs 人）跌至 .8547——泛化缺口 7–12pt。
4. **对比损失未见效**：SupCon^F 2 epoch 几乎不降；本语料无 task_id，batch 内正/难例结构太弱。
5. **一个积极读数**：单代码 lab 级家族信号真实存在且可迁移（M0 判别探针 val .42 / 未见新模型 .33，≫随机 .125）——
   比历史“无配对不可归因”（E14：30 族 7.7%）乐观；为旗舰域家族信号提供基线。

---

## 6. 给下一轮的建议（初步）

1. **温度/尺度必须进规范**：我们补的“能量÷维”是必要的（否则 sigmoid/softmax 饱和、无梯度）；建议规范直接写 τ_E。
2. **主方法应把决策规则改为判别式（最大杠杆）**：建议 M4' = z 或 r_F 上的收缩判别头
   （或能量只用于检测、p_F 用判别 softmax）——同特征 +17pt 是强证据；
   这也与 d-det 内核“检测=读数/能量、归因=判别式”的既有分工一致。
3. **残差化改为可开关照消融、默认关闭**，直到给出相对 raw 的正收益证据（或先试“保留 δ 投影 + 判别头”）。
4. **对比损失需难例结构**：无 task_id 语料下正例关系太弱；建议下一轮在 pairs 域（同题多族）加入，
   或先冻结 z、只训对比投影验证梯度是否有效。
5. **task-hard 需补语料**（pairs 域同题异家族）；M5（token 面）留 hybrid。
6. **拒识/未见模型校准**是旗舰域的核心缺口（AUC .59、检测掉 7–12pt），建议单列机制设计
   （conformal / 能量温度校准 / 未知家族阈值）。
7. 成本：本轮 **17.7 min/次**（12GB 单卡），支持下一轮并行 2–3 个消融臂。

---

## 7. 复现

```bash
cd /root/autodl-tmp/u-det/d-det
# 冒烟（~0.5 min）
OMP_NUM_THREADS=8 /root/miniconda3/envs/udet/bin/python -u scripts/flagship_round1.py --smoke
# 全量（见 runs/flagship_r1/results.json 的 timing_min）
OMP_NUM_THREADS=8 /root/miniconda3/envs/udet/bin/python -u scripts/flagship_round1.py
```

- 环境：Python 3.12 / torch 2.9.1 / sklearn 1.9.1；`OMP_NUM_THREADS=8` 必设；单卡 12GB。
- 产物：`runs/flagship_r1/{results.json, head.pt, stats.npz}`；日志 `/tmp/ddet_r1_full.log`。
