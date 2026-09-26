# SemEval-2026 Task 13 竞赛总结
## 一、竞赛概览
**SemEval-2026 Task 13: Detecting Machine-Generated Code with Multiple Programming Languages, Generators, and Application Scenarios**
检测机器生成代码，覆盖**多种编程语言、多种生成器、多种应用场景**。包含三个子任务（Subtask A/B/C），核心挑战是**泛化能力**（未知语言、未知生成器、未知领域）。
### 参赛队伍与最终排名
| 队伍 | Subtask A | Subtask B | Subtask C | 主要方法 |
|------|-----------|-----------|-----------|----------|
| **TeleAI** | **1st** (0.99708) | **1st** (0.50819) | 2nd (0.73332) | 全参数微调 Qwen3-30B + 多级集成 |
| **Young DSMLKZ** | - | - | **1st** (0.78546) | MIL-UniXcoder + Meta-Stacking + 手工特征 |
| **Yuvan Ramesh** | - | 3rd (0.44882) | 3rd (0.71398) | 任务自适应集成 |
| **UIT_AMMC** | 3rd (0.80188) | - | - | 结构化格式签名 + QLoRA + 多智能体辩论 |
| **SYSUpporter** | 4th (0.43399) | - | - | 风格信号 + 语言感知截断 |
| **Alejandro Mosquera** | 2nd (0.82409) | - | - | - |
| **TocToc** | - | 2nd (0.45595) | - | - |
| **CEIA** | 4th (0.75318) | - | - | - |
| **Brazil AI** | - | - | 4th (0.69770) | - |
| **mcdok** | - | - | 5th (0.68643) | - |
---
## 二、三个子任务的数据集与核心问题
### Subtask A: 二分类机器代码检测
**任务定义**： 给定一段代码，判断是完全人类生成还是机器生成。
**泛化方向**（从训练&验证到测试，**两种均可能未知**）：
- **语言泛化**： C++/Python/Java → Go/PHP/C#/C/JavaScript
- **领域泛化**： 算法代码 → 研究代码、通用部署代码
**核心问题**：
- 长度跨度大（max-min = 24 ~ 34,346 tokens）
- 语言和领域双重分布偏移
**数据统计**：
| Split | 生成器 | #样本 | 语言 | 领域 |
|-------|--------|-------|------|------|
| Train | Human | 238,475 | Python/Java/C++ | 算法代码 |
| Train | AI | 261,525 | Python/Java/C++ | 算法代码 |
| Validation | Human | 47,695 | Python/Java/C++ | 算法代码 |
| Validation | AI | 52,305 | Python/Java/C++ | 算法代码 |
| **Test** | Human | 390,054 | C/C#/C++/Go/Java/JS/Python | 算法+研究+通用部署 |
| **Test** | AI | 109,946 | C/C#/C++/Go/Java/JS/Python | 算法+研究+通用部署 |
---
### Subtask B: 多分类作者归因
**任务定义**： 给定一段代码，预测其归属为人类或 10 种模型家族，**共 11 分类**。数据集共含 **45 个生成器**，平均每个家族 4 个。
**泛化方向**： Test 中存在**已见家族的未见模型**（如 Test 中 Qwen 有 12 个模型，训练只有 5 个）。
**核心问题**：
- 人类和机器**极度不平衡**（训练：442K vs ~50K）
- 模型间长度分布不同
- 测试集中存在 **4 种 (模型, 语言) 组合未见**
**数据统计**（按家族）：
| 家族 | Train | Validation | Test |
|------|-------|------------|------|
| Human | 442,096 | 88,490 | 243,769 |
| GPT | 10,810 (1模型) | 2,154 (1) | 104,304 (2) |
| Qwen | 8,993 (5) | 1,755 (5) | 26,459 (12) |
| Llama | 8,197 (5) | 1,695 (5) | 35,411 (13) |
| IBM-Granite | 8,127 (3) | 1,579 (3) | 11,202 (7) |
| Phi | 5,783 (3) | 1,118 (3) | 20,981 (7) |
| Mistral | 4,608 (2) | 895 (2) | 14,185 (5) |
| DeepSeek | 4,162 (3) | - | 9,674 (6) |
| Yi | 3,029 (2) | 650 (2) | 8,471 (4) |
| StarCoder | 2,227 (3) | 847 (3) | 3,631 (6) |
| Gemma | 1,968 (3) | 372 (3) | 21,913 (8) |
| BigCode | - | 445 (3) | - |
---
### Subtask C: 细粒度代码分类
**任务定义**： 给定一段代码，判断它属于**四类**中的哪一类：
1. **完全人类编写** (Human)
2. **完全机器生成** (AI)
3. **混合** (Hybrid): LLM 部分补全或重写
4. **对抗** (Adversarial): 通过"人类风格"提示词，或通过 RLHF 对抗训练模拟人类代码
**核心问题**：
- **类别不平衡**（Human 训练样本 485K vs 其他类）
- 语义上 Hybrid 和 Adversarial 与 Human 难以区分
**数据统计**：
| Split | Label | #样本 | #生成器 |
|-------|-------|-------|---------|
| Train | Human | 485,483 | 1 |
| Train | AI | 210,471 | 78 |
| Train | Hybrid | 85,520 | 28 |
| Train | Adversarial | 118,526 | 17 |
| Validation | Human | 107,885 | 1 |
| Validation | AI | 46,770 | 78 |
| Validation | Hybrid | 19,006 | 28 |
| Validation | Adversarial | 26,339 | 17 |
| **Test** | Human | 334,770 | 1 |
| **Test** | AI | 58,510 | 78 |
| **Test** | Hybrid | 73,889 | 29 |
| **Test** | Adversarial | 32,831 | 17 |
---
## 三、各队伍主要解决思路
### 3.1 TeleAI（A 1st, B 1st, C 2nd）
**总体思路**： 数据为中心 + 全参数微调 + 多级集成。报告中很多细节未披露，各步骤独立进行。
**(a) 数据分析与增强**
- 类别不平衡 → **重复过采样**
- 长度分布 → **统一截断和填充**
**(b) 提示优化**
- 设计多种提示模板（含显式任务指令、类别描述、输出格式约束）
- 在验证集上迭代评估，通过**剪枝和早停**机制选出最优模板
**(c) 全参数微调**
- 基座模型： **Qwen3-30B-A3B-Instruct**（参数量远大于 UniXcoder/CodeT5 等）
- 各任务**独立全参数微调**。作者认为代码生成检测需要**深入修改模型的内部表征**
**(d) 集成与投票**
- 聚合多个 checkpoint 的预测（数据量 vs 参数量较小，易过拟合；选择不同超参、平台期附近 Epoch）
- **Soft Voting**： 平均各类别概率
- **Hard Voting**： 多数投票制
- **Logits-based Voting**： 基于逻辑值融合
- **LightGBM Stacking**： 将多个基模型在验证集上的概率向量/logits 作为特征，训练元分类器
**成绩**： A=0.99708 (1st), B=0.50819 (1st), C=0.73332 (2nd)
---
### 3.2 UIT_AMMC（A 3rd）
**核心假设**： 判别信号已从"个人风格"迁移到 RLHF 优化可读性形成的**空白对齐工件 (Whitespace Alignment Artifact)**。语义趋同时排版信号仍可区分作者，且**语言无关**。
**关键统计特征**（所有特征作为提示词注入模型上下文）：
1. **结构偏置**： 空行率 R_void = 空行数/总行数；缩进一致性
2. **词汇可预测性**： 字符/二元组香农熵
3. **句法方差**： 命名惯例方差、行长方差
**R_void 工件的统计显著性**：
| Split | Human μ | AI μ | Cohen's d | p-value |
|-------|---------|------|-----------|---------|
| Train | 11.7% | 15.4% | 0.32 | <0.001 |
| Test | 7.25% | 16.50% | 0.97 | <0.001 |
**训练方法（Algorithm 1: Robust Instruction Tuning）**：
- **QLoRA 微调 Qwen2.5-Coder-14B-Instruct**（LoRA rank=64, α=64）
- 数据按**项目划分**避免代码片段泄漏到训练/验证两侧
- **随机增强**（独立触发）：
  - 注释剥离 p=0.2
  - 空白扰动 p=0.3（插入/删除空行至原密度 ±10%）
  - 标识符重命名 p=0.4
- 优化 next-token prediction loss
**对比推理（Algorithm 2: Contrastive Inference）**：
- 对原代码提取特征，其中 R_void 缩放（在融合时相对 S_sem 偏小）
- 假设代码是 Human/AI，通过提示词模板计算平均 NLL，做差并归一化得模型判为 AI 的概率
- 融合空行率得到最终判断： `S_final = λ·S_sem + (1-λ)·R̂_void (λ=0.82)`
- 阈值按已见语言用 **Youden 指数**独立校准： Python 0.53, Java 0.56, C++ 0.54, **OOD 全局默认 0.54**
**多智能体辩论（Algorithm 3: ACE-inspired）**：
- 对**决策边界附近的模糊样本**进一步讨论
- AI 方 (Advocate)、Human 方 (Skeptic) 各自解释并交给裁判 (Judge)
- 仅对 `|S_final - τ(l)| < ε` 的样本触发辩论
- 不涉及多轮辩论、演化等
**消融结果**：
| 配置 | Test F1 | Δ |
|------|---------|---|
| 纯 R_void（阈值法） | 0.679 | - |
| SFT (Qwen-Coder-14B) | 0.740 | +0.061 |
| SFT + 统计特征入 Prompt | 0.785 | +0.106 |
| **完整级联 (+ 辩论)** | **0.802** | **+0.123** |
---
### 3.3 Team Yuvan（B 3rd, C 3rd）
**Subtask A — 发现关键陷阱并绕开**：
- 直接微调 CodeBERT/Qwen 在测试集效果极差，**标签翻转反而提升**：
  | Model | Val F1 | Test F1 | Flipped |
  |-------|--------|---------|---------|
  | CodeBERT (125M) | 0.995 | 0.245 | 0.561 |
  | Qwen (1.5B) | 0.998 | 0.269 | 0.587 |
  | GraphCodeBERT+LR | 0.900 | 0.328 | — |
- 结论： 不能依赖脆弱的词汇线索，改用 **41 维 AST 特征 + LightGBM**（Test 0.638）
- 最终方案： **0.95·AST + 0.05·翻转后的 Qwen**
**Subtask B**:
- B 数据集各类样本数分布**极度不均**，使用**加权交叉熵**（w_c = N/(K·n_c)）
- **Qwen2.5-Coder-1.5B-Instruct** 上 LoRA 微调
**Subtask C**:
- Qwen-2.5-coder、CodeBERT+头尾分块、TF-IDF+LR **加权投票**
- 权重以步长 0.05 单纯形网格搜索，发现最优解**平坦**：
  - Qwen only: Val 0.909, Test 0.710
  - Qwen+CodeBERT: Val 0.910, Test 0.711
  - Qwen+TF-IDF: Val 0.910, Test 0.713
  - **All three**: Val 0.911, **Test 0.714**
---
### 3.4 SYSUpporter（B 4th）
**发现的关键问题**： 类别极端不平衡、代码长度分布不同、部分风格特征会被分词器破坏、测试集存在未见组合和分布偏移。
**Imbalance-Aware Training**：
- **损失层**： Focal Loss (γ=2.0)、类权重 w_c = 1/√n_c
- **池化层**： 对编码器最后层输出进行**注意力掩码 GeM 池化**（指数 p 初始 3，可学习）
- **正则化层**： **Multi-Sample Dropout**，5 个并行 dropout 分支，dropout 率从 0.1 线性递增到 0.18
**Language-Aware Truncation**：
- 按语言确定**头尾预算**： 从前往后、从后往前截取设定 token 数，保留关键区域
- **换行边界对齐**： 截断点移到附近 20 tokens 内的换行处
- **动态窗口采样**： 训练时 10% 使用随机位置连续窗口（数据增强）
**Efficient Batching**：
- 长度 std=1,662（min=24, max=34,346），随机组 batch 组内差异过大
- **按长度排序分组，组内打乱**，减少 padding 浪费
**Stylistic Feature Engineering**（每组取 1 个放在提示词开头）：
- **行尾组**： `[CRLF]` (Windows \r\n)、`[CR]` (旧 Mac \r)
- **缩进组**： `[TAB_INDENT]`、`[MIXED_INDENT]`、`[SPACE4]`、`[SPACE_OTHER]`
- **尾随空格组**： `[TRAILING_WS]`
- **功能 token**： `[LANG]`、`[CODE]`、`[TRUNC]`、`[WINDOW]`
- 示例： `[LANG] Python [SPACE4] [TRAILING_WS] [CODE] (头 811) [TRUNC] (尾 203)`
---
### 3.5 YoungDSMLKZ（C 1st）
**核心观点**： AI 代码检测**本质是长上下文问题**。
**数据切分**： 4:1 切分为 dev-train（用于 OOF、调参）和 dev-eval（用于模型选择、路由阈值调优）。
**三大视角融合**：
**① 风格度量视角 (Stylometric)**
- LLM 会留下**确定性、启发式的痕迹**（格式、词法、注释习惯）
- 提取 **200+ 维特征** + **XGBoost**
- 包含： 格式类（空行率、注释比例、缩进）、词法类（字符熵、平均标识符长度、行长变异系数）、AST 结构、LLM 对话残留
**② 局部视角 (Local)**
- 微调 **UniXcoder 512**、**MazgaBERT 1024**（基于 ModernBERT-base 微调）
- 输出 4 维向量的对数概率作为特征
- 5 折 OOF 分别训练两个 CatBoost
- 拼接 OOF 特征 + 之前对数概率 → 训练一个 CatBoost（考虑非线性交互与类别不平衡）
**③ 全局视角 (Global)**
- **滑窗 512 步幅 256**，切成 N 块独立通过 UniXcoder
- 对每个隐藏维度**最大池化**
- 融合向量通过**类别权重的 MIL-CatBoost**
**Class-Routing Decision Rule（关键创新）**：
- 按类别**分配最擅长的子系统**：
  - Human → XGBoost
  - Hybrid → Meta-Stacking
  - AI 和 Adversarial → MIL-CatBoost
- 依据： 每类分配 **dev-eval 上召回率最高**的子系统
- **推理优先级**： Adversarial → AI → Human → Hybrid
- 若无一命中，回退到 MIL-CatBoost
---
## 四、跨队伍洞察
### 共识与共性做法
1. **不平衡处理**： 过采样（TeleAI）、加权损失（Yuvan, SYSU）、Focal Loss（SYSU）
2. **长度处理**： 统一截断填充（TeleAI）、按语言截断（SYSU）、滑窗（Young DSMLKZ）
3. **LLM 基座选择**： Qwen 系列被广泛使用（Qwen3-30B / Qwen2.5-Coder-14B / Qwen2.5-Coder-1.5B）
4. **统计/结构特征**： R_void（空行率）、熵、缩进、注释等风格特征在多个队伍中出现
5. **集成/融合**： 几乎所有头部队伍都使用了多模型或多 checkpoint 集成
### 关键洞察与陷阱
1. **分布偏移是最大障碍**（Yuvan 发现微调模型测试集上标签翻转反而更好）
2. **词汇线索脆弱，结构线索更稳健**（多个队伍采用 AST、格式、空白特征）
3. **长上下文问题本质**（Young DSMLKZ 用 MIL + 滑窗处理长代码）
4. **按类别路由到最擅长子系统**可显著提升（Young DSMLKZ 的 Class-Routing）
5. **人类 vs AI 的可分性**在 Subtask A 中很高（TeleAI 0.997），但**细粒度分类**（B: 0.508, C: 0.785）依然很难
### 难度梯度
- **Subtask A**（二分类）： 最高分 0.997，相对容易
- **Subtask C**（四分类）： 最高分 0.785，中等难度
- **Subtask B**（11 分类+未见模型）： 最高分仅 0.508，**最难**


# 类别不平衡处理专题总结
你说得对，不平衡是该竞赛的**核心挑战之一**（三个子任务都明确列出），我之前分散在各队伍里没有系统梳理。下面单独展开。
---
## 一、不平衡全景：三个子任务的量化情况
### Subtask A（轻微）
| Split | Human | AI | 比例 |
|-------|-------|-----|------|
| Train | 238,475 | 261,525 | ~1 : 1.1（基本平衡）|
| **Test** | 390,054 | 109,946 | **~3.6 : 1**（训练平衡、测试偏斜 → 隐性陷阱）|
### Subtask B（极端，最难）
| 对比 | 数值 | 比例 |
|------|------|------|
| Train: Human vs Gemma（最大家族差距）| 442,096 vs 1,968 | **~225 : 1** |
| Train: Human vs AI 家族均值（~6K）| 442,096 | **~74 : 1** |
| Test: Human vs StarCoder | 243,769 vs 3,631 | **~67 : 1** |
这是三个子任务中不平衡最严重的，也是最高 Macro-F1 仅 0.508 的重要原因。
### Subtask C（中等）
| Split | Human | AI | Hybrid | Adversarial | 最不平衡对 |
|-------|-------|-----|--------|-------------|-----------|
| Train | 485,483 | 210,471 | 85,520 | 118,526 | Human : Hybrid ≈ **5.7 : 1** |
| Test | 334,770 | 58,510 | 73,889 | 32,831 | **训练时 Hybrid 最少，测试时 Adversarial 最少**（不平衡结构本身发生了漂移）|
**关键背景**：评测指标是 **Macro F-score**，每个类别权重相同 → 训练中的不平衡会直接压制少数类的召回，进而重创 Macro-F1。
---
## 二、各队伍的不平衡处理策略
### 1. TeleAI（A 1st / B 1st / C 2nd）—— 数据层：过采样
**重复过采样（Repeated Oversampling）**
- 对少数类样本进行有放回的重复采样，直至各类别量级对齐
- 属于**数据层面**的重采样策略，作用于训练输入分布
- 风险：过采样会放大过拟合，因此 TeleAI 配套了**多 checkpoint 聚合**（软/硬投票、Logits 融合、LightGBM Stacking）来对冲——过采样的副本被不同超参/不同 Epoch 的模型看到，集成降低方差
**同时处理的孪生问题**：长度分布差异 → 统一截断和填充（长度也是一种“结构性的不平衡”）
---
### 2. Yuvan（B 3rd）—— 损失层：逆频率加权交叉熵
**公式**：
$$w_c = \frac{N}{K \cdot n_c}$$
- $N$ = 总样本数，$K$ = 类别数（11），$n_c$ = 第 c 类样本数
- 即**逆频率归一化权重**：每个类别的权重使得该类在损失中的总贡献与其类别数解耦（人多的类权重小、人少的类权重大，加权后各类期望贡献相等）
- 承载方式：在 **Qwen2.5-Coder-1.5B-Instruct 上 LoRA 微调**时，用该权重替换交叉熵中每类的系数
- 特点：**单一、简洁**的损失层方案，没有额外的采样或集成机制
---
### 3. SYSUpporter（B 4th）—— 损失层 + 正则化层的双管齐下
这是对不平衡处理**最系统化**的队伍，在"Imbalance-Aware Training"下专门列出三个组件：
**① Focal Loss（γ = 2.0）**
$$FL(p_t) = -(1-p_t)^{\gamma}\log(p_t)$$
- 作用机制：对已分对的简单样本（$p_t$ 高）动态降权 $(1-p_t)^\gamma$，把梯度预算集中在**难分样本**上
- 与不平衡的关系：多数类大量“简单样本”主导梯度时，Focal Loss 自动抑制它们，等效放大少数类难样本的话语权
**② 平滑类权重**
$$w_c = \frac{1}{\sqrt{n_c}}$$
- 注意与 Yuvan 的区别：**取平方根**而非直接逆频率
- $1/\sqrt{n_c}$ 是比 $1/n_c$ 更**温和**的加权（对 n=1968 的 Gemma，权重是 1/n 的 ~44 倍）
- 实践含义：极端不平衡下（225:1），全逆频率加权会导致少数类梯度爆炸、训练不稳，平方根是常用的折中
**③ Multi-Sample Dropout（5 个并行分支，dropout 率 0.1 → 0.18 线性递增）**
- 同一样本过 5 条不同 dropout 的分支，损失取平均
- 作用：强正则化，防止模型在过采样的少数类样本/多数类冗余模式上记忆化；不同 dropout 率提供多样化的决策边界
**辅助配套**（与不平衡间接相关）：
- GeM 池化（可学习指数 p）：对长样本更鲁棒的池化，缓解长度不平衡带来的表征偏差
- 类权重与 Focal 叠加 = **重加权 + 难例聚焦**的双重机制
---
### 4. YoungDSMLKZ（C 1st）—— 分类器层 + 决策层的隐性平衡
YoungDSMLKZ 没有显式使用采样或损失加权，而是把不平衡处理**下沉到分类器和路由决策**中：
**① 类别权重内置到树模型**
- 全局视角：融合向量通过**设置了类别权重的 MIL-CatBoost**
- 局部视角：Meta-Stacking 的 CatBoost 训练时明确“**考虑到存在非线性交互与类别不平衡**”
- 即 CatBoost/XGBoost 的 `class_weights` / `scale_pos_weight` 类参数
**② Class-Routing Decision Rule（最有特色的隐式平衡手段）**
规则：**每类分配 dev-eval 上召回率最高的子系统**
- Human → XGBoost
- Hybrid → Meta-Stacking
- AI 和 Adversarial → MIL-CatBoost
这与不平衡的关系是**直接的**：
- 优化目标从“整体准确率”变为“**每类召回率最大化**"——这正是 Macro-F1 的逐类分解
- 少数类（如 Hybrid，训练仅 85K）不再被多数类淹没，因为它的预测由**专门为其召回率优化**的子系统负责
**③ 推理优先级体现少数类优先**
$$\text{Adversarial} \rightarrow \text{AI} \rightarrow \text{Human} \rightarrow \text{Hybrid}$$
- 优先判定少数/难类，无一命中才回退到 MIL-CatBoost
- 防止多数类（Human）的“吸走效应”——若先判 Human，大量边界样本会被错误吞入多数类
---
## 三、策略对比与洞察
| 队伍 | 层级 | 具体机制 | 适用场景 |
|------|------|----------|----------|
| TeleAI | **数据层** | 重复过采样 + 集成对冲 | 平衡-中度不平衡，数据量充足 |
| Yuvan | **损失层** | 逆频率加权 $N/(K \cdot n_c)$ | 中度不平衡，方案简洁 |
| SYSUpporter | **损失层 + 正则层** | Focal(γ=2) + $1/\sqrt{n_c}$ + Multi-Sample Dropout | **极端不平衡**（225:1），双重机制 |
| YoungDSMLKZ | **分类器层 + 决策层** | 类权重 CatBoost + 按类召回路由 + 少数类优先 | 与 Macro-F1 目标对齐，多子系统场景 |
### 值得注意的细节
1. **两种加权公式的选择**：Yuvan 用 $1/n_c$（全逆频率），SYSU 用 $1/\sqrt{n_c}$（平方根平滑）。面对 B 子任务 225:1 的极端比例，**平方根加权更稳健**——全逆频率会让少数类单样本损失放大数百倍，容易训练崩坏。
2. **Focal Loss 与类权重可叠加**：SYSU 同时使用两者。Focal 解决“简单样本淹没梯度”，类权重解决“类别频次不均”，针对的是不平衡的两个不同侧面。
3. **过采样必须配套正则化/集成**：TeleAI 的过采样 + 多 checkpoint 投票、SYSU 的 Multi-Sample Dropout，都说明**数据层重采样会放大过拟合**，需要后端机制对冲。
4. **YoungDSMLKZ 的启示**：当评测指标是 Macro-F1 时，**按类召回率路由**可能是比损失加权更直接的对齐方式——它把“平衡”写进了架构决策而非损失函数。这可能是其在 C 子任务（类别不平衡 + 类间语义高度相似）拿下第一的关键因素之一。
5. **不平衡结构本身会漂移**：Subtask C 中训练最少的 Hybrid（85K）在测试时反超 AI，而训练第三多的 Adversarial 在测试中垫底（32K）。**仅按训练集频率加权可能在测试集上失效**，这也是 YoungDSMLKZ 用 dev-eval 而非训练频率来定路由的原因。