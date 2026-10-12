# d-det：代码条件下的后训练差异与来源归因

论文工作稿（依据总账 v2.2；v2.5 仅作为弱探索补充）

版本：2026-10-12  
状态：研究稿；不替代总账，不触发新实验。

本文只整理 d-det 自身的有效实验。没有把 u-det 检测工程线并入，也没有把 v2.6–v2.8 的新公式、资料库讨论、路线讨论或未运行计划写入正文。主要依据是 d-det/docx/archive/d-det_ACL总账_v2.2目标口径修正前快照_2026-10-11.md，以及相应 artifacts 中的 manifest、metrics、report、脚本和 SHA256 清单。

## 摘要

d-det 研究代码样本中的来源和生成过程差异。核心问题不是在随机切分上把代码分类得更高，而是检验冻结代码表示是否包含可迁移的来源关系，以及这种关系在任务、生成器、格式和长度变化下是否仍然存在。本文将实验对象分为四个相互独立的轴：单样本 Human/AI 检测、AI 内部来源/家族可读性、同父或同题条件下的后训练处理差异、以及 task/project/generator-heldout 转移。双侧表示与任务中心化只在明确的配对或转导协议中使用，不被包装为默认单样本部署。

研究从 KL 约束的后训练接口出发，使用同题双侧表示 u=[h+;h−]、对称和反对称坐标 S、A，并比较冻结表示、线性读出、任务中心化、关系残差、强词法/布局控制和受控代码变体。实验覆盖 d-det v0.1–v1.0、E1–E21、E24–E36、DCAN/强 P0、H2 pair、C0–C3 代码条件批次，以及 H3 v1–v5 的数据和增强诊断。

主要结果是：冻结表示中确实存在来源可读性和配对关系可读性，但当前数据、预算和 generator-heldout 协议下没有稳定的实用方法增量。本文唯一主模型 `relation residual` 的配对加权 AUROC 为 `.8501`，完整强 P0 为 `.8486`，差值 `+.0014` 的 95% CI 跨 0；因此主方法增量未被确认。C0–C3 没有超过 C0；H3 数据闸门持续为 revise_data，AMMC 和局部微调没有证明稳定增益。论文因此把可读性存在与归因方法泛化严格分开，并完整报告失败路线、实现错误和数据边界。

## 0. 主模型、对照与主效果（先读这里）

本文只把一个方法称为**主模型（Ours-H2）**：在同题、同 family、不同 generator 的配对上，使用冻结 CodeT5 的双侧代码表示，先扣除独立端点/来源组成，再学习 task-conditioned 的 **relation residual**。其评估是端点平衡的 48 折、按 task cluster 拟合和 bootstrap；因此它是 H2 的配对归因方法，不是单样本 Human/AI 检测器，也不是把 C0 或 H3 的检测分数改名为 d-det 主模型。

| 标记 | 模型/实验 | 输入与协议 | 在本文中的角色 |
|---|---|---|---|
| **OURS / 主模型** | `relation residual` | 双侧 CodeT5 表示；同题同 family 异 generator；端点平衡 48 折；task-cluster cross-fit | 唯一的主方法结果 |
| **CONTROL / 强对照** | `full P0` | lexical/style/meta 与对称 pair 成分的完整组合；同一 48 折 | 主模型必须超过的匹配强基线 |
| **CONTROL / 简化对照** | `independent source composition` | 仅独立端点/来源组成，不含 relation residual | 解释小增量来自哪里 |
| **BASELINE** | `TF-IDF`、冻结余弦、线性头、`char_center_std`、AuthorBench fusion | 各自固定的单视图或任务条件输入 | 基线或外部强控制，均不是我们的主模型 |
| **BASELINE / ABLATION** | `C0`、`C1`、`C2`、`C3` | 代码条件检测批次；C0 是冻结 canonical baseline，C1–C3 是代码/静态代理/一致性消融 | 独立的代码条件诊断线，不是 H2 主模型 |
| **DIAGNOSTIC** | E1–E21、E24–E36、H3 GPU、AMMC、局部微调 | 几何、来源可读性、任务分离或反捷径协议 | 用来解释边界，不承担主方法结论 |

### 主效果到底是什么

主模型在开发集上的**配对加权 AUROC**为 **`.8501`**（不是 C0/H3 使用的 task-macro AUROC）。同一折、同一 task-cluster bootstrap 下，完整强 P0 为 **`.8486`**，所以主模型相对完整强对照的差值为 **`+.0014`（约 +0.14pt，95% CI `[−.0097,+.0129]`）**，区间跨过 0；相对较弱的 independent source composition 的差值为 **`+.0046`（约 +0.46pt，95% CI `[+.0018,+.0075]`）**，但这个比较不能替代 full P0。故本文的主效果是一个**受限的负结果**：双侧关系信号可读，但 relation residual 没有在完整强控制之后证明稳定、实用的方法增量。

下面几个数字不要误读为主效果：`C0=.9585` 是另一条代码条件检测基线；`C1–C3` 是其消融；`E17/E17d` 的 `pair full=.8271`、`center=.8571` 是配对/转导诊断；H3 的 `.6723/.6761/.6765` 是 GPU 诊断。它们分别回答检测、配对可读性或反捷径问题，不能与 `.8501 vs .8486` 的主模型比较混成一张总榜。

为避免读者把“有可读性”看成“方法有效”，全文结果表使用以下标签：**OURS**=主模型，**BASELINE**=基线，**CONTROL**=匹配控制，**ABLATION**=主模型或检测线的消融，**DIAGNOSTIC**=只用于解释的实验。

## 1. 研究问题

### 1.1 检测

给定单段代码或严格同题 Human/AI 数据，判断代码是否来自人类或机器。该轴回答检测问题，不自动回答来源家族问题。

### 1.2 AI 内部来源和家族可读性

在 AI 子集中，判断 observed source unit、generator 或有证据的 family。随机切分上的高分只能称闭集来源可读性；只有 generator-heldout、task-heldout 或 project-heldout 才支持相应迁移主张。

### 1.3 后训练处理差异

比较同一父 checkpoint、同题输入和可追溯运行条件下的处理差异。不同基座、模型规模、提示协议或模板不能直接称为纯后训练差异。

### 1.4 转移

测量 task、project 或 generator 留出后的性能损失。随机切分不能替代这些外推设置；同题参考或任务中心化属于转导接口，必须和单样本结果分开。

## 2. 数学接口

对参考策略 π0、奖励 r(t,y) 和 β>0：

\[
\max_\pi E_{y\sim\pi}[r(t,y)]-\beta KL(\pi\|\pi_0),\qquad
\pi^*(y|t)=\frac{\pi_0(y|t)e^{r(t,y)/\beta}}{Z(t)}.
\]

所以

\[
\log\pi^*(y|t)-\log\pi_0(y|t)=r(t,y)/\beta-\log Z(t).
\]

同题差分可以抵消归一化项，但该式不推出共同基座、加性表示、正交子空间或跨 generator 不变性。

对同题双侧表示 h+、h−，定义

\[
u=[h^+;h^-],\qquad S=(h^++h^-)/2,\qquad A=(h^+-h^-)/2.
\]

任务中心化为

\[
r_i=h_i-\mu_{task(i)}.
\]

它可以减少题目内容，但需要同题参考，因此属于 transductive 诊断。

H3 的历史联合目标为

\[
L=L_D+\lambda_F L_F+\lambda_{inv}L_{inv},\qquad
L_{inv}=|P|^{-1}\sum_{(c,T(c))\in P}\|p_D(c)-p_D(T(c))\|_2^2.
\]

只有 AI 行使用 source 头；变体继承 parent 标签并留在同一 split。

## 3. 数据和证据规则

每条记录至少保留 task、solution、human_or_ai、generator/source、model role、language、split、父样本和代码哈希。缺失字段写明 unavailable，不由文件名推断。

| 数据 | 有效用途 | 不能承担的主张 |
|---|---|---|
| STACAD alignment | 同题 Human/AI 候选、任务控制、H3 | 同家族正对；project-heldout |
| AuthorBench/DCAN | task-heldout 来源/家族诊断和强 P0 | 与其他数据拼 family |
| LLM-CodeGen v2 | 有支持折的 same-task/same-family pair 诊断 | 伪造空折或跨数据拼接 |
| DroidCollection | generator-rich 外部压力 | task-aware H2/H3 |
| CoDET-M4 | 外部检测和结构/来源控制 | 与其他包共享 family 标签 |
| AICD T2 | numeric-only 外部控制 | 把数字标签改写成 d-det family |

### 3.1 数据集构建明细

本文使用的数据不是一个混合大表，而是按研究问题分别构建的多个数据角色。不同数据包的标签空间、任务粒度和来源语义不合并。

| 数据包 | 原始来源与固定版本 | 构建规则 | 最终规模 | 标签/字段 | 允许实验 |
|---|---|---|---:|---|---|
| STACAD alignment v1 | 本地 STACAD-v2 corpus_v2 | 只保留完整七模型任务；按原始 file-level split；每任务保留一个 Human anchor 与七个模型输出；语言按配额抽样 | 3,180 tasks；22,260 AI 行；7 languages | human anchor、model、family、generator、language、task、split、代码哈希 | 同题 Human/AI、任务中心、跨语言控制 |
| AuthorBench task-aware | LLM-AuthorBench 固定压缩包 | 只保留八个模型均有输出的 prompt；每个 prompt-model 只取一行；任务整组切分 | 239 tasks；1,912 行；167/36/36 train/dev/test | 六个 family；八个 observed model；C-only；replicate_count | task-heldout family、有限 OpenAI generator-heldout |
| AuthorBench-DCAN | AuthorBench 的 DCAN 派生包 | 先按任务和来源审计，再执行保守 skeleton 重复剔除；不跨数据包拼 family | 9,498 行；2,715 tasks | family/source、task、代码视图、任务中心索引 | 强 P0、任务条件和 shortcut 诊断 |
| LLM-CodeGen v2 pairs | LLM-CodeGen v2 code_tasks.jsonl | 修复 Gemini 响应解析；只保留同任务、同 family、不同 generator 的非空正对；每对生成五个对齐视图 | 695 full pairs；447 balanced pairs；train/dev/test=475/122/98 | family、generator、task/group、code_only/ids_only/strings_only/comments_only/all | H2 配对与视图审计 |
| DroidCollection v2 | project-droid/DroidCollection revision 9a42843 | 流式读取 parquet；保留选定 family 的机器行；删除 exact duplicate；每 source split 的 Human 控制封顶 1,000/family；独立保留 hybrid/adversarial 文件 | 146,718 primary rows；125,718 machine、21,000 human；32 machine generators、7 families | Human/Machine、family、generator、language、split、row hash | generator-heldout 外部归因压力 |
| CoDET-M4 control | DaniilOr/CoDET-M4 pinned revision | 每个 split/model/language cell train 封顶 500，val/test 封顶 200；保留原 source/model，不跨包重命名 | 18,870 行级控制包 | code、model、language、target、source、结构特征 | 外部检测、来源和结构控制 |
| AICD T2 numeric control | AICD-Bench T2 pinned parquet | 流式 parquet；每数字标签 train/val/test 封顶 3,000/300/1,000；保留原整数，不解析 family | 48,877 行级控制 | code、整数 label、split、源文件和源行 | numeric-only 外部控制 |

STACAD 的 3,180 个任务固定为 train 2,100、dev 660、test 420；train 每语言 300，dev 每语言最多 100，test 每语言 60。七个 generator 属于不同厂商，因此 66,780 个 same-task cross-generator pair 是语义控制，不是 same-family 正对。AuthorBench 的 239 个任务固定为 167/36/36，只有 OpenAI family 同时有 GPT-4.1、GPT-4o 和 GPT-4o-mini 三个 generator；其余 family 只有一个 generator。LLM-CodeGen full pair family 计数为 Google 142、Meta 406、Mistral 147，balanced 版本为 142/158/147。

STACAD 派生包沿用数据与文档 CC BY 4.0、代码 MIT 的许可记录；所有派生包保留 source archive hash、split manifest 和源行哈希。本文不把不同数据包的 family 名称合并，也不把模型发布目录推断为行为等价或共同 base。

DroidCollection 没有公开 task_id 或 prompt_id，因而不支持 same-task paired generalization。其 diagnostic_hybrid_adversarial.jsonl 的 2,000 行不进入 primary fit。AICD 公开字段只有 code 和 integer label，label_map_pending 在核实前不改名为 family。

### 3.2 数据构建与排除流程

数据构建固定为以下顺序：

1. 固定来源 revision、许可证和源文件哈希；
2. 流式读取，记录源文件、源行和原始代码哈希；
3. 解析 task、solution、generator、family、language 和 split 字段；
4. 先按 task/project/solution cluster 切分，禁止同一 cluster 跨 split；
5. 执行 exact、whitespace-normalized、lexical/skeleton 重复审计；
6. 只在 train 内生成 comment stripping、safe whitespace 和 identifier rename 变体；
7. 为每个 parent-child 保存 parent_id、变换类型、before/after hash、AST/parser 状态、changed token/line 数和 reject reason；
8. 生成 data_role、fold_plan、support matrix 和 SHA256SUMS；
9. 冻结模型、阈值和 manifest 后才读取独立评估；
10. 评估后关闭 test_read，禁止用暴露数据重新筛选任务或调参。

变体继承 parent 的标签和 split，不产生新 family，不产生新的 Hybrid/Adversarial 标签。STACAD 中同一 Human anchor 与七个 AI 输出不被当成七个独立任务；AuthorBench 的 replicate_count 只用于敏感性分析，不当作独立 task 数；Droid 的 Human 行是控制样本，不是与机器行的同题配对。

### 3.3 数据角色和主张边界

| 实验主张 | 最低数据条件 | 本文可用数据 | 明确不能写成 |
|---|---|---|---|
| Human/AI 检测 | 真实 Human 与 AI 标签；task/solution 不跨 split | STACAD alignment | family attribution 或项目泛化 |
| 已见 source/family 读出 | source/family 标签和 task-heldout | AuthorBench、DCAN、LLM-CodeGen | 未见 generator 泛化 |
| same-family 配对 | 同 task、同 family、不同 generator 的真实 pair | LLM-CodeGen v2、AuthorBench OpenAI 辅助折 | STACAD 跨厂商 pair |
| generator-heldout | heldout generator 不进训练，且 task 也隔离 | Droid、合格 LLM-CodeGen/AuthorBench 折 | project-heldout |
| project-heldout | 可验证 project/repository ID | 当前数据不足 | 用 task_id 或 source_ref 代替 project |

先按 task 或项目边界切分，再生成 parent-child 变体。变体不改变标签，不产生新的 family 或创作类别。所有实验均记录 train/dev/test 读取状态、阈值范围、模型哈希和逐行/逐任务指标。

### 3.4 统一实验配置

| 项目 | 固定口径 |
|---|---|
| 编码器 | 冻结 CodeT5-small 或实验包指定的冻结表示；局部微调单独标注 |
| 输入 | 代码原文或预注册视图；删除视图不代表只保留该语义类别 |
| 截断 | 实验包固定的 512-token head/tail 或完整词法输入；不同输入预算不直接比较 |
| 读出 | 线性头、逻辑回归、MLP 或预注册融合；不能事后换读出器 |
| 阈值 | 只在 train 内验证集或预先独立 calibration 集选择 |
| 随机性 | 每个正式配置记录 seed、初始化哈希和折成员；重复 seed 不是独立数据集 |
| 变体 | parent 每 epoch 曝光次数、变换概率、接受率和拒绝原因固定记录 |
| 评估 | row/pooled 与 task-macro 同时报告；配对比较按 task cluster bootstrap |
| 开关 | 历史主线均记录 test_read、generation、weights_downloaded、code_execution |

### 3.5 指标

对每条代码行的分数，pooled row AUROC 衡量全体行的排序；对每个包含两类的任务分别计算 AUROC 后取均值，得到 task-macro AUROC。BA 和 Macro-F1 依赖冻结阈值，不能从 test 反调。family 任务同时报告各类 F1、宏平均、混淆矩阵和每类支持数。配对增量以共同 task 为 bootstrap cluster，报告 delta 的点估计和 95% CI；不能用两个四舍五入点估计直接相减替代 paired delta。

## 4. 主要实验结果（按主模型优先阅读）

### 4.1 d-det 自身早期版本（BASELINE / HISTORICAL）

d-det v0.1 的 m4 test F1 为 .9557、AUC .9881，pair direction accuracy 为 1.0；去掉正交约束后 |cos|=.74，说明正交项需要显式控制。v0.2 的 head/tail/mid 片段结果为 .966/.937/.928。v0.3–v1.0 的 m4 F1 约 .9756–.9776，hybrid line/chunk/token 约 .784/.50/.817；这条线证明检测头可以工作，但没有提供 family attribution 的跨生成器证据。

| 版本 | m4 F1 | line F1 | chunk F1 | token F1 | 结论 |
|---|---:|---:|---:|---:|---|
| v0.1 | .9557 | — | — | — | 配对方向可读；正交项影响明显 |
| v0.2 | — | .966 head | .937 tail | .928 mid | 片段任务可行但存在位置差异 |
| v0.3 | .9776 | .7845 | .4848 | .8131 | 全参检测可工作 |
| v0.4/v1.0 | .9766/.9775 | .7881/.7877 | .5008 | .8167 | 与旧强基线处于同一范围 |

### 4.2 E1–E21：从公共检测轴到配对几何（DIAGNOSTIC）

E1 的 reward ratio 单族 AUC 约 .81，ANOVA p=8e-9，说明明确谱系条件下存在偏好位移。E2/E5/E6 的多参照读出约 .86，但跨实验室约 .13。E8/E9 的位移投影 AUC .845 且几乎与检测轴重合，说明公共位移主要是检测轴，不是独立 family 轴。

E10–E14 的 KMeans 约 .52–.60、判别 LR 约 .88、高维白化接近机会，表明 family 更像判别半空间而不是自然聚类。E15/E16 的高维 QDA 不稳，局部中心最高约 .78，单 delta 学习约 .75 且不稳定。E17/E17d 的 pair full .8271、delta .7626、center .8571，表明联合双侧交互重要，但 center 是转导接口。E18/E19 说明 S/A 单独只能恢复联合信息的一部分。E20/E21 的 cross-scatter η=.41–.50，简单一维比值接近机会，支持分布式交互。

| 实验组 | 输入/方法 | 主要读数 | 解释 |
|---|---|---:|---|
| E1 | 单族 reward ratio | AUC .81；ANOVA p=8e-9 | 有谱系条件下存在偏好位移 |
| E2/E5/E6 | 多参照、跨实验室 | .86；.13 | 公共位移不稳定 |
| E8/E9 | 位移投影 | AUC .845 | 主要重现检测轴 |
| E10–E14 | KMeans、LR、白化 | .52–.60；.88；近机会 | family 更像判别半空间 |
| E15/E16 | QDA、局部中心、delta | .15–.51；约 .78；约 .75 | 高维和单 delta 不稳 |
| E17/E17d | pair、delta、center | .8271；.7626；.8571 | 联合交互重要，center 为转导 |
| E18–E21 | S/A、scatter | S/A 低于联合；η=.41–.50 | 不能用单一手工比值替代交互 |

### 4.3 E24–E36、Stage B 和 Droid（DIAGNOSTIC / TRANSFER CONTROL）

E24–E28 的能量、查询和显著通道路线没有超过公平基线；s_top 零梯度导致的早期解释已撤回。E29–E30 观察到来源可读性，但 generator 迁移损失大。E31–E33 的残差、SupCon 和暖启动表明短预算下表示漂移是主要问题。E35 的 cross-generator 增量约 +.28/+ .30pt，E36 pooled 为 −.17/−.20pt，Droid v2 更大数据仍接近机会。因此停止继续堆叠几何损失。

| 阶段 | 主要配置 | 结果 | 关闭原因 |
|---|---|---:|---|
| E24–E28 | energy/query/saliency | 低于公平基线 | s_top 零梯度和方向矛盾 |
| E29–E30 | generator-heldout 来源读出 | 有可读性，迁移损失大 | 不能支持稳定泛化 |
| E31–E33 | residual、SupCon、warm start | B0 .5879/.5855；残差更低 | 短预算表示漂移 |
| E35 | cross-generator 增量 | +.28/+ .30pt | 低于实用阈值 |
| E36/Droid v2 | 7-family generator-heldout | pooled −.17/−.20pt；接近机会 | 停止继续堆叠几何损失 |

### 4.4 DCAN、强 P0 和任务条件（BASELINE / CONTROL）

DCAN round1–4 修复了 eval/train mode、BN、checkpoint 和 zero_grad 等静默错误。LoRA 可改善 task-heldout，但低于 AuthorBench fusion。去 identifier、comment、string 的单变量审计显示 lexical shortcut 可拆分，但联合清洗不能直接归因于某个机制。强 P0 的 AuthorBench task-heldout Macro-F1 为 .8388，word TF-IDF 为 .8200；内容模型未稳定超过 P0。

| 基线/方法 | 数据协议 | 指标 | 结果 |
|---|---|---|---:|
| AuthorBench fusion P0 | task-heldout | Macro-F1 | .8388 |
| word TF-IDF | 同一 task-heldout | Macro-F1 | .8200 |
| 冻结 CodeT5-small | AuthorBench family | test BA / Macro-F1 | .5972 / .5760 |
| char_center_std | 同一 family 任务划分 | BA / Macro-F1 | .6435 / .6672 |
| Droid generator-heldout | 7-family | BA/读出约 | .1645，机会 .1429 |

### 4.5 H2 pair round1/round2（BASELINE / PAIR DIAGNOSTIC）

pair 包包含 4,692 条记录、27 项检查、0 个 invalid。AuthorBench 与 LLM-CodeGen 的 raw、ids、strings、comments、all 视图显示冻结表示和线性头有不同程度的 pair 可读性，但学习式距离没有在两个 source 上同时超过门槛，清洗视图也没有稳定增益。该结果不能写成已经学到 family 语义，也不能和 K 类 attribution F1 横比。

| source | view | TF-IDF BA | 冻结余弦 BA | 线性头 BA | 学习距离 BA |
|---|---|---:|---:|---:|---:|
| AuthorBench | raw/ids/strings/comments/all | .532/.538/.511/.530/.513 | .584/.538/.584/.561/.555 | .804/.766/.789/.819/.808 | .740/.677/.695/.713/.694 |
| LLM-CodeGen | raw/ids/strings/comments/all | .704/.689/.689/.658/.561 | .612/.531/.607/.617/.541 | .648/.713/.658/.645/.692 | .672/.653/.590/.627/.621 |

pair rows 按 task/group 聚类，不能把每一行当独立样本；family map 部分来自模型发布目录，不能扩展成行为等价证明。

本节的 pair round1/round2 是主模型之前的可读性和视图审计，表中的 `学习距离` 不是主模型。主模型结果见下一节；这样可以区分“冻结表示中存在可读性”与“relation residual 超过完整强控制”。

### 4.6 H2 主模型：relation residual 与完整强 P0（OURS / MAIN RESULT）

H2 主模型只在有真实同题、同 family、不同 generator 配对的 LLM-CodeGen/AuthorBench 支持折上定义。模型输入是双侧冻结 CodeT5 表示及其 task-conditioned relation 特征；训练和评估按 task cluster cross-fit，端点边际平衡，避免把 partner 或 endpoint 身份直接当作关系。`independent source composition` 是指定的简化控制，`full P0` 是包含 lexical/style/meta 和对称 pair 成分的完整强控制。主比较必须看 `relation residual − full P0`，而不是只看 relation residual − composition。

| 角色 | 方法 | 配对加权 AUROC | 与主模型的 paired 差值 | 95% CI | 判读 |
|---|---|---:|---:|---|---|
| **OURS** | relation residual | **.8501** | — | — | 本文唯一主模型 |
| **CONTROL** | full P0 | .8486 | **+.0014（+0.14pt）** | `[−.0097,+.0129]` | 区间跨 0，未证实增量 |
| **CONTROL** | independent source composition | .8455 | **+.0046（+0.46pt）** | `[+.0018,+.0075]` | 较弱控制，不能替代 full P0 |
| **CONTROL** | fused3 | .8225 | +.0276 | 未作为主比较 | 组合不完整，不作强基线 |

因此，本文关于主模型的正式结论是：**relation residual 在指定简化 composition 上有小幅开发集增量，但相对完整强 P0 没有稳定方法增量，也没有达到预设的 1pt 实用阈值。** 这一结论不否认来源关系可读性，只限制方法增量的强度和证据等级。

### 4.7 C0–C3 代码条件批次（BASELINE / ABLATION）

C0 canonical dev row/task-macro 为 .9442/.9585。C0 冻结重验 198/198 通过。C1 code-only/full、C2 静态代理和 C3 一致性正则相对 C0 均无稳定增量，主要差异约 −2.70 至 −4.41pt，每项正折不超过 1/11；C2 相对 C1 full 的 .09pt 为空效应；变体 AST/接口破坏为 0。

裁定为 code_conditioned_increment_not_observed，证据等级为 canonical diagnostic。C0–C3 关闭，不增加容量、温度、轮数或损失项补救。

| job | 配置 | task-macro AUROC | 相对 C0 | 正折 | 状态 |
|---|---|---:|---:|---:|---|
| C0 | frozen canonical baseline | .9585 | — | — | frozen |
| C1 | code-only MLP | 约 .9315 | −2.70pt | 1/11 | closed |
| C1 | full MLP | 约 .9279 | −3.04pt | 1/11 | closed |
| C2 | λv=.1 static proxy | 约 .9290 | −2.95pt | 1/11 | closed |
| C3 | λinv=.1 consistency | 约 .9253 | −3.31pt | 1/11 | closed |

C0 的 198/198 digest 重验通过；C1–C3 使用相同 train/dev、折和评估接口。test_read、generation、weights_downloaded 和 code_execution 均为 false。

| C1 视图 | head | row AUROC | task-macro AUROC | 相对 C0 的 task-macro |
|---|---|---:|---:|---:|
| code_only | linear / MLP | .9132 / .9212 | .9282 / .9315 | −.0303 / −.0270 |
| prompt_only | linear / MLP | .5000 / .5000 | .5000 / .5000 | 构造性无信息 |
| ht_hy | linear / MLP | .8759 / .9181 | .9310 / .9285 | −.0275 / −.0299 |
| full | linear / MLP | .8603 / .9138 | .9142 / .9279 | −.0441 / −.0304 |
| psi_only | linear / MLP | .5580 / .5530 | .5993 / .5823 | −.3596 / −.3761 |

C1 full-MLP 相对 code-only-MLP 的任务内差异为 −.0034，CI [−.0078,+.0015]，题面条件化没有带来增量。C2 为 full-MLP 加 λv=.1 静态代理，task-macro .9290，相对 C0 −.0295，相对 C1 full +.0009，CI [−.0006,+.0025]。C3 为 full-MLP 加 λinv=.1 一致性，task-macro .9253，相对 C0 −.0331。

C3 的 train-only 变体采样为 comment requested 2,103/accepted 1,677、whitespace 3,161/accepted 1,076、rename 4,271/accepted 3,666；accepted 变体 AST 破坏和接口改变均为 0。变换后归因 AUROC 为 comment .871、whitespace .850、rename .895。该安全性结果不等于检测增量。

### 4.8 H3 v1–v4 数据闸门（DIAGNOSTIC / DATA GATE）

初始 length-only .8459、lexical-only .9261，metadata/source 约机会，因此判 revise_data。v1 覆盖七语言 parser 22,048 行，balanced length .8231、lexical .9084；v2 clean/controlled length .7248/.6679，lexical .9694/.9585；v3 hard-negative length .7109、masked lexical .6578；v4 hard-negative length .6907、masked lexical .6343，clean length 仍 .7248。不同协议不可互换，也没有证明捷径完全消失。

| 变体类型 | 请求数 | 接受数 | 主要拒绝原因 | 安全审计 |
|---|---:|---:|---|---|
| comment/docstring | 2,103 | 1,677 | no_comment 或无可删除内容 | accepted AST/接口保持 |
| whitespace | 3,161 | 1,076 | no_change | accepted AST/接口保持 |
| identifier rename | 4,271 | 3,666 | ast_changed、reflection、no_safe_candidate | accepted AST/接口保持 |

v1 的七语言 parser 审计覆盖 22,048 行，排除 5 个任务；跨 split exact、whitespace、token 和 skeleton 残留为 0。v3 的 train/dev 规模为 8,384/2,636 行，对应 2,096/659 task，并按七 generator 配额平衡；project provenance 仍 unavailable。v4 的 hard-negative 视图使用 dev 内容进行挑战匹配，因此只能作探索性压力集，不能重新命名为独立确认集。

### 4.9 H3 GPU、AMMC 和局部微调（DIAGNOSTIC / EXPLORATORY）

GPU 诊断在 RTX 3080 Ti 完成 24/24：detection-only .6723，joint .6761，joint-invariance .6765，source-only .1473；同折 lexical row 控制 .7536。该实例化关闭。

v5 固定 parent 曝光的组合增强在两折 clean/hard-negative 上 task-macro 变化约 −.15 至 +.24pt，CI 跨 0；冻结表示和局部 encoder 微调没有稳定增强。局部微调出现 task-macro 高、pooled row 低的分歧，说明任务内排序不能代替整体检测。v5 只作为弱探索。

| H3 阶段 | 训练/控制数据 | length probe | lexical probe | 主要结论 |
|---|---|---:|---:|---|
| 初始闸门 | train/dev；metadata/source 控制 | .8459 | .9261 | revise_data |
| v1 | 22,048 parser rows；7 languages | .8231 | .9084 | parser 完成，闸门未过 |
| v2 | clean/controlled dev | .7248/.6679 | .9694/.9585 | 协议读数不可互换 |
| v3 | 8,384 train / 2,636 dev；hard negative | .7248/.7109 | masked .6547/.6578 | generator quota 修正 |
| v4 | 精确 quota；challenge view | .7248/.6907 | masked .6343 | clean length 仍未过闸门 |

| H3 v5 eval | original joint TM | composed joint TM | paired delta |
|---|---:|---:|---:|
| fold 0 clean | .9184 | .9179 | −.05pt |
| fold 0 hard-negative | .9058 | .9044 | −.15pt |
| fold 1 clean | .8606 | .8613 | +.06pt |
| fold 1 hard-negative | .8408 | .8431 | +.24pt |

v5 组合变体使 clean raw lexical 从 .9115/.8625 降到 .8902/.8526，但模型 task-macro 没有稳定改善；requested-mask 与 accepted-mask 也不同（约 .5047/.4905 与 .5587/.5543），因此不能把变体接受率变化解释为模型学到不变性。

| fold/eval | original TM / row / BA | composed TM / row / BA | ΔTM，pt |
|---|---|---|---:|
| fold 0 clean | .9570 / .7372 / .5154 | .9535 / .7348 / .5147 | −.354 |
| fold 0 hard-negative | .9562 / .7155 / .5132 | .9510 / .7176 / .5135 | −.525 |
| fold 1 clean | .9317 / .6977 / .6034 | .9251 / .6953 / .5789 | −.664 |
| fold 1 hard-negative | .9197 / .6251 / .5637 | .9139 / .6267 / .5495 | −.577 |

### 4.10 v2.5 捷径干预补充（DIAGNOSTIC / WEAK SUPPLEMENT）

该实验继承 v4 的选样历史，只在 train 内按 task/generator 隔离，使用冻结 CodeT5 的确定性读出，没有读取 dev/test，不是完整 d-det 微调主表。两折各比较 original、AMMC、AMMC nuisance projection、同秩 random projection、nuisance-only、lexical raw 和 lexical masked。

| arm | fold 0 original task AUROC | fold 0 composed task AUROC | fold 1 original task AUROC | fold 1 composed task AUROC |
|---|---:|---:|---:|---:|
| original | .9080 | .9160 | .9332 | .9157 |
| AMMC | .9120 | .9173 | .9375 | .9277 |
| nuisance projection | .9005 | .9041 | .9258 | .9130 |
| random projection | .9107 | .9094 | .9351 | .9219 |
| lexical raw | .8744 | .8285 | .8294 | .7846 |
| lexical masked | .6808 | .6808 | .6309 | .6309 |

AMMC 相对 original 的 paired task-macro 差异为 fold 0 +.0040、fold 1 +.0043；composed 为 +.0013、+.0120，均为单次 fit 且 CI 跨 0。nuisance projection 相对 original 为 −.0075、−.0074，相对同秩 random projection 为 −.0102、−.0093，也不能解释为成功消除捷径。composed 改变率在人类和 AI 行上分别为 94.36% 和 94.28%，但 comment 改变率为 68.85% 和 45.90%，其改变指示的 AUROC 为 .3853，翻向后为 .6147，说明变换接受率本身可能成为线索。

该补充只能支持一个有限结论：切分主要防止泄漏，平衡削弱标签相关，AMMC 提供压力视图；它们没有证明更强语义特征已经涌现。

## 5. 主结论

1. 固定数据和协议下存在来源可读性与配对关系可读性。
2. 这些可读性没有在当前 generator-heldout、强 P0 和整体读出条件下转化为稳定方法增量。
3. 任务中心化、双侧交互和 AMMC 变体是诊断工具，不是已证实的单样本因果机制。
4. C0–C3 的代码条件化后训练差异在当前实例化中没有超过 C0。
5. H3 数据闸门和增强结果支持继续区分数据捷径、检测鲁棒性和来源读出，不支持把某一失败扩大成所有后训练差异不存在。

## 6. 经验、总结和教训

### 6.1 方法

- 来源可读性、family attribution、后训练差异和 generator transfer 必须分开。
- 双侧联合优于单一 delta，但只在配对协议中有意义。
- 任务中心化提高部分读出，却改变部署接口，不能隐藏其转导性质。
- 正交、SupCon、残差和一致性损失的下降不是因素识别证明。
- 小增量必须和完整强 P0 比较，不能只和弱 baseline 比较。

### 6.2 数据

- STACAD 的同题 Human/AI 适合检测和任务控制，但 same_family_pairs=0，不能伪造同家族正对。
- Droid 没有公开 task/prompt ID，只能做外部 generator 压力。
- 不同数据包不能直接拼 family 标签；AICD 数字标签不能命名为模型家族。
- 先切分后增强；parent/child 不跨 split；重复和 parser 过滤写入 manifest。
- 词法强不等于结果无效，但必须和同折强控制与变体压力共同报告。

### 6.3 工程和统计

- train/eval mode、dropout、BN、checkpoint、zero_grad、tokenizer、截断和 TF-IDF fit scope 的错误足以改变结论。
- row 与 task-macro 回答不同问题；CI 按 task/project cluster bootstrap。
- score digest 不能代替逐行预测；哈希通过不能代替科学闸门。
- test 暴露后不能换 split 或重命名制造新确认集。
- 负结果只适用于当前数据、预算和协议。

# 附录 A：有实际结果的实验顺序

| 顺序 | 实验 | 结果摘要 | 保留方式 |
|---:|---|---|---|
| 1 | d-det v0.1 | m4 .9557、AUC .9881、pair dir 1.0 | 历史结果 |
| 2 | d-det v0.2 | head/tail/mid .966/.937/.928 | 片段诊断 |
| 3 | d-det v0.3–v1.0 | m4 .9756–.9776，hybrid 约 .788/.50/.817 | 检测基线 |
| 4 | E1 | 单族 reward ratio AUC .81 | 谱系条件诊断 |
| 5 | E2/E5/E6 | 多参照约 .86，跨实验室约 .13 | 公共轴诊断 |
| 6 | E8/E9 | 投影 AUC .845，与检测轴重合 | 非独立 family 轴 |
| 7 | E10–E14 | KMeans .52–.60，LR .88，白化近机会 | 判别几何 |
| 8 | E15/E16 | QDA 不稳，局部中心约 .78，delta 约 .75 | 局部诊断 |
| 9 | E17/E17d | full .8271，delta .7626，center .8571 | 配对/转导 |
| 10 | E18–E21 | S/A 低于联合，scatter η .41–.50 | 交互诊断 |
| 11 | E24–E28 | 能量/显著路线未超过基线 | 关闭 |
| 12 | E29/E30 | 来源可读，generator 迁移损失大 | 限定诊断 |
| 13 | E31–E33 | 残差/SupCon 无稳定增量，表示漂移 | 关闭 |
| 14 | E35/E36/Droid | +.28/.30pt 后 pooled −.17/−.20pt | H2 负边界 |
| 15 | DCAN round1–4 | 修复静默错误；LoRA 低于强 P0 | 工程/强基线 |
| 16 | H2 pair round1 | 4,692 records，27 checks，0 invalid | 数据审计 |
| 17 | H2 pair round2 | 冻结表示可读，学习距离未过门槛 | 负结果 |
| 18 | C0 | canonical .9442/.9585，198/198 重验 | 基准 |
| 19 | C1/C2/C3 | 相对 C0 无稳定增量 | 关闭 |
| 20 | H3 初始闸门 | length .8459、lexical .9261 | revise_data |
| 21 | H3 v1 | parser 22,048 行；length .8231 | 数据修订 |
| 22 | H3 v2 | clean/controlled length .7248/.6679 | 数据修订 |
| 23 | H3 v3 | hard-negative length .7109、masked lexical .6578 | 数据修订 |
| 24 | H3 v4 | hard-negative length .6907、masked lexical .6343 | 探索挑战 |
| 25 | H3 GPU | .6723/.6761/.6765，source .1473 | 关闭 |
| 26 | H3 v5 AMMC | task delta −.15 至 +.24pt，CI 跨 0 | 弱探索 |
| 27 | v5 局部微调 | task-macro 与 pooled row 分歧 | 弱探索 |
| 28 | v2.5 本机捷径干预 | AMMC 与 nuisance projection 未显示稳定增量 | 仅补充 |

# 附录 B：证据文件索引

| 主题 | 主要证据 |
|---|---|
| d-det 早期表示和配对 | d-det/artifacts 中 E1–E36、flagship、pair 和 DCAN 目录 |
| H2 pair | d-det/artifacts/h2_*、d-det/data/h2_* |
| C0–C3 | d-det/artifacts/code_conditioned_*、mainline_c0c3_batch_2026-10-10 |
| H3 数据 | d-det/artifacts/h3_data_gate_*、h3_local_revision_*、h3_hardneg_revision_* |
| H3 GPU/AMMC | d-det/artifacts/h3_gpu_diagnostic_batch_2026-10-11、h3_ammc_* |
| v2.5 弱补充 | d-det/artifacts/local_shortcut_intervention_2026-10-11 |

所有数字以对应目录中的 metrics、report、commands、git_head 和 SHA256SUMS 为准。

# 附录 C：停止裁定

- 论文依据 v2.2；v2.5 只作为勉强成立的探索补充。
- u-det 不进入本文。
- v2.6–v2.8 新增的文献驱动公式、格式通道新叙事和未运行计划不进入本文。
- C0–C3 保留为已完成且关闭的历史主线。
- H3 保留数据闸门、GPU 诊断、AMMC 和局部微调负结果。
- 不启动 AutoDL，不读取 test，不继续追加实验。
