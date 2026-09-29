# d-det 核心方法规范：旗舰模型来源差异的流形与对比分离

版本：2026-09-29  
基线：`u-det/d-det`，提交 `6d02117`  
状态：待服务器端执行验证的主方法规范，不是已验证结论。

## 理论起点：后训练分布

KL 约束的奖励优化为：

\[
\max_q\; \mathbb E_q[r(y,t)]
-\beta\,\mathrm{KL}(q(\cdot\mid t)\|p_0(\cdot\mid t)).
\]

最优分布满足：

\[
p_*(y\mid t)=
\frac{p_0(y\mid t)\exp(r(y,t)/\beta)}{Z(t)},
\qquad
Z(t)=\sum_y p_0(y\mid t)\exp(r(y,t)/\beta).
\]

因此：

\[
\log p_*(y\mid t)-\log p_0(y\mid t)
=\frac{r(y,t)}{\beta}-\log Z(t).
\]

两个旗舰模型 \(f,g\) 的输出差异可写成：

\[
\log p_f-\log p_g
=\Delta_{pre}^{f,g}
+\Delta_{post}^{f,g}
-\Delta_Z^{f,g}.
\]

我们看不到这些概率，所以学习代码表示中的可观测来源几何：

\[
\Phi(x_{t,f})
=\mu_t+\delta_{AI}+\rho_f(t)+\epsilon.
\]

\(\delta_{AI}\) 是总体 AI–Human 差异，\(\rho_f\) 是控制题目和总体来源后剩余的家族/后训练相关差异。跨旗舰模型数据还包含预训练、架构、提示和解码因素，因此不能声称 \(\rho_f\) 是纯后训练因果效应。

## DCAN 与 DMHM 的参考边界

- **DCAN** 提供任务特征分离的启发：本方法用标量 \(s_D\) 承担检测，用去除总体 AI–Human 方向后的 \(r_F\) 承担家族归因；不预设低秩归因瓶颈或两个独立子空间。
- **DMHM** 提供流形感知的启发：本方法采用对角收缩方差、带体积项的能量和条件家族残差；第一轮不声称已实现完整的局部/全局混合协方差。
- 两者提供分离和流形建模的参考；本文主创新仍是基于不同旗舰模型代码产物的后训练相关来源差异分离。



### 维度总览

| 对象 | 维度 | 作用 |
|---|---:|---|
| CodeT5 token 表示 \(h_i\) | \(768\) | 上下文代码特征 |
| 均值 \(\mu\) | \(768\) | 全局统计 |
| 方差 \(v\) | \(768\) | token 表示离散程度 |
| 显著位置分数 \(s_{top}\) | \(1\) | 关键 token 来源证据 |
| 样本表示 \(z=[\mu;\log(v+\epsilon);s_{top}]\) | \(1537\) | 流形和对比学习输入 |
| 检测分数 \(s_D\) | \(1\) | Human/AI 二分类 |
| 家族残差 \(r_F\) | \(1537\) | 家族归因几何 |
| 家族后验 \(p_F\) | \(|\mathcal F|\) | AI 家族分类 |

---

## 0. 核心研究问题

目标是分离两个来源差异：

1. **AI–Human 差异**：代码是否进入机器生成分布；
2. **AI 家族差异**：在机器代码内部，来自哪个模型家族。

理论公式和因果边界见“理论起点：后训练分布”。实现目标是：先估计总体 AI–Human 差异，再检验去除该公共方向后是否存在更稳定的家族几何。

---

## 1. 数据协议

每个样本至少保留：

```text
code, task_id(optional), language, source_family, human_ai_label,
generator_id(optional), generation_condition(optional)
```

主训练对象：

- Human 代码；
- GPT、Claude、Gemini、DeepSeek、Qwen-Coder 等旗舰/强模型代码；
- 已有数据中的改写或对抗代码，只在其标签和生成条件明确时使用。

规则：

1. 不重新生成样本；
2. 不把 base/instruct 作为主训练配对；
3. 没有 `task_id` 时，不伪造同题正例；
4. 家族归因只在 `human_ai_label=AI` 的样本上训练；
5. 评测按题目或项目来源分组，不能让同一题跨 train/test；
6. 未见模型或未见家族只用于最终泛化测试，不参与阈值和温度选择。

---

## 2. 编码器输出

使用当前 CodeT5 上下文编码器。对有效 token 得到：

\[
H(x)=\{h_i\}_{i=1}^{L},
\qquad h_i\in\mathbb{R}^{768}.
\]

当前 `mean` 池化不能作为唯一主表示，因为少数关键 token 可能被大量普通 token 稀释。不同代码的 token 排列也不稳定，所以主方法不计算跨样本的逐位置差分：

\[
h_i(x)-h_i(x')
\]

只有在明确结构对齐后才允许作为附加实验。

---

## 3. 排列不变的代码集合表示

### 3.1 共享 token 变换

先对每个 token 做同一个上下文变换：

\[
u_i=\operatorname{LN}(h_i),
\qquad
\tilde h_i=W_2\operatorname{GELU}(W_1u_i)+u_i.
\]

其中 \(u_i,\tilde h_i\in\mathbb R^{768}\)，
\(W_1\in\mathbb R^{d_m\times768}\)，
\(W_2\in\mathbb R^{768\times d_m}\)；第一轮只要求残差输出回到
768 维，\(d_m\) 是中间宽度。

第一轮令输出维度仍为 768，不使用低秩瓶颈。

### 3.2 一阶统计

\[
\mu(x)=\frac{1}{L}\sum_{i=1}^{L}\tilde h_i.
\]

它作为传统 mean baseline 的组成部分，而不是全部表示。

### 3.3 二阶统计

\[
v(x)=\frac{1}{L}\sum_{i=1}^{L}
\left(\tilde h_i-\mu(x)\right)^{\odot2}.
\]

这里的 \(v\) 保留 token 表示在每个特征方向上的离散程度，能够表达“少数位置与普通位置差异较大”的情况。

### 3.4 显著位置池化

令检测 token logit 为：

\[
q_i=w_q^\top\tilde h_i+b_q.
\]

其中 \(w_q\in\mathbb R^{768}\)，\(b_q,q_i\in\mathbb R\)。

排序后记为：

\[
q_{(1)}\ge q_{(2)}\ge\cdots\ge q_{(L)}.
\]

取前 \(K\) 个显著位置：

\[
K=\min(L,\max(8,\lceil0.1L\rceil)),
\qquad
s_{top}(x)=\frac1K\sum_{j=1}^{K}q_{(j)}.
\]

训练时先用 `torch.topk`；入选位置有梯度，未入选位置不接收该项梯度。Human 样本也可能有高分位置，最终符号由标签监督校准。这个聚合与 mean 不同：普通位置不再等权决定来源分数。

### 3.5 最终样本表示

\[
z(x)=\left[\mu(x);\log(v(x)+\epsilon);s_{top}(x)\right].
\]

因此：

- token 顺序改变不会改变 \(z\)；
- 表示中同时有均值、离散程度和显著位置证据；
- 不需要假设不同代码的 token 位置一一对应；
- 不是低秩归因表示，维度由统计表示直接决定。

第一轮使用 `z=[μ; log(v+ε); stop]`（1537 维）。这里的排列不变仅指固定上下文 token 向量集合的聚合；原始代码重排后，CodeT5 的上下文向量会改变。第一轮使用对角收缩方差；不要直接求逆高维小样本经验协方差。

---

## 4. 训练折内的来源流形估计

训练、验证、测试严格分开：训练折可更新参数和统计量；验证与测试只能读取固定的中心、方差、类别先验与温度。

### 4.1 人类与 AI 家族的对角收缩统计

令类别 \(c\in\{H\}\cup\mathcal F\)，其中 \(\mathcal F\) 是训练期已见 AI 家族：

\[
\mu_c=\frac1{N_c}\sum_{i:y_i=c}z_i,
\qquad
v_c=\frac1{N_c}\sum_{i:y_i=c}(z_i-\mu_c)^{\odot2}.
\]

这里 \(\mu_c,v_c,\widetilde v_c\in\mathbb R^{1537}\)。

用训练折的合并方差 \(v_{pool}\) 收缩：

\[
\widetilde v_c=(1-\lambda)v_c+\lambda v_{pool}+\epsilon,
\quad 0\le\lambda\le1.
\]

主实验预先固定 \(\lambda\) 或用训练折内部交叉验证选择。若某家族样本少，直接设 \(\lambda=1\)。这避免了 1537 维样本协方差求逆。第一轮不做全维白化。

### 4.2 带体积项的能量

不同家族的方差可以不同，因此必须同时计算距离和分布体积：

\[
G_c(z)=\frac12\sum_{j=1}^{1537}
\left[\frac{(z_j-\mu_{c,j})^2}{\widetilde v_{c,j}}
+\log\widetilde v_{c,j}\right].
\]

若日后引入完整协方差，公式必须改为

\[
G_c(z)=\tfrac12\left[(z-\mu_c)^\top\Sigma_c^{-1}(z-\mu_c)
+\log|\Sigma_c|\right].
\]

只比较平方 Mahalanobis 距离会系统性偏向方差大的家族。

### 4.3 AI 混合负对数密度

主实验使用均匀家族先验 \(\pi_f=1/|\mathcal F|\)，经验先验作为消融：

\[
G_{AI}(z)=-\log\sum_{f\in\mathcal F}\pi_f\exp[-G_f(z)].
\]

用 `logsumexp` 实现。训练开始前，只用训练折和冻结的当前 d-det 编码器初始化所有统计量；训练中按当前训练 batch 特征用 `detach` 的指数移动平均更新统计量。当前样本的能量仍对其特征 \(z\) 回传梯度。每个 epoch 结束，只在训练折重算统计量，并将其固定用于验证。若统计量振荡，先冻结它们训练读出，再在 epoch 结束时更新。局部/全局混合协方差列入后续实验，不能直接称第一轮已实现 DMHM。

---

## 5. AI–Human 检测标量

定义检测标量：

\[
s_D(z)=G_H(z)-G_{AI}(z)+\log\frac{\pi_{AI}}{\pi_H}+b_D.
\]

其中 \(s_D,b_D\in\mathbb R\)，\(p_D\in(0,1)\)。

定义 AI 概率：

\[
p_D(x)=\sigma(s_D(z(x))).
\]

训练损失：

\[
L_D=
-y\log p_D-(1-y)\log(1-p_D).
\]

这个分数表达的是：样本离 Human 流形有多远，同时离最近 AI 家族流形有多近。

主实验令 \(b_D=0\)，先验 \(\pi_{AI},\pi_H\) 在训练折确定；需要偏置校准时只用训练内部验证。它比当前 `s1=w^T h` 多表达了分布形状，但最终仍然是一个标量。不同数据集的先验变化须预先定义协议，不能用测试标签或测试机器比例反推先验。

---

## 6. 从总体 AI 差异中分离家族差异

### 6.1 总体 AI–Human 方向

定义训练折内的总体来源位移：

\[
\delta=\mu_{AI}-\mu_H,
\qquad
\mu_{AI}=\sum_f\pi_f\mu_f.
\]

因此 \(\delta\in\mathbb R^{1537}\)，
\(D_H=\operatorname{diag}(\widetilde v_H)\in\mathbb R^{1537\times1537}\)。

使用训练折 Human 对角收缩方差定义投影系数。令 \(D_H=\operatorname{diag}(\widetilde v_H)\)：

\[
\alpha(z)=
\frac{\delta^\top D_H^{-1}(z-\mu_H)}
{\delta^\top D_H^{-1}\delta+\epsilon}.
\]

### 6.2 去除总体来源位移

\[
r_F(z)=z-\mu_H-\alpha(z)\delta.
\]

\(r_F(z)\in\mathbb R^{1537}\)，并且不构造额外的低秩瓶颈。

它保留相对于总体 AI–Human 方向的残差。这个残差不是“低秩归因空间”：它仍然位于完整表示空间，只去除了一个估计的公共检测方向。

### 6.3 条件家族能量

对每个 AI 家族，在训练折残差上估计中心和对角收缩方差：

\[
\bar\mu_f=\mathbb{E}_{train}[r_F(z)\mid f],
\]

\[
\bar G_f(z)=\frac12\sum_j\left[
\frac{(r_{F,j}(z)-\bar\mu_{f,j})^2}{\bar v_{f,j}}
+\log\bar v_{f,j}\right],
\]

家族后验：

\[
p_F(f\mid AI,x)=
\operatorname{softmax}_f\bigl(-\bar G_f(z)/\tau_F+\log\pi_f\bigr).
\]

其中 \(\bar\mu_f,\bar v_f\in\mathbb R^{1537}\)，
\(\tau_F\in\mathbb R_{>0}\)，\(p_F\in\mathbb R^{|\mathcal F|}\)。

这里 \(\bar v_f\) 用与第 4 节相同的收缩公式估计；\(\tau_F\) 默认 1，调整时只用训练内部验证。必须和**未去除公共方向的原始 \(z\) 家族能量**做同数据对照。若残差变差，说明公共检测方向也含有真实家族信息，应放弃该投影。

这一步明确完成：

- `s_D`：判断是否为 AI；
- `r_F`：去掉总体 AI–Human 方向；
- `p_F`：判断 AI 内部的模型家族。

### 6.4 家族损失

只对 AI 样本计算：

\[
L_F=-\log p_F(f_i\mid AI,x_i).
\]

Human 样本不伪造家族标签，也不把 Human 作为一个 AI 家族。

---

## 7. 对比学习目标

### 7.1 检测对比损失

检测对比使用完整样本表示 \(z\)。把所有 AI 当同类可能压平家族信息，因此此项先作为消融；主方法需要报告它对家族归因的影响。对每个样本 \(i\)：

- 正例集合 \(P_D(i)\)：同一 Human/AI 标签，优先选择不同任务、不同语言或不同生成器；
- 负例集合：标签相反的样本，优先选择同任务的困难负例。

归一化表示：

\[
\hat z_i=\frac{z_i}{\|z_i\|_2}.
\]

相似度：

\[
s_{ij}=\frac{\hat z_i^\top\hat z_j}{\tau_D}.
\]

监督对比损失：

\[
L_{SupCon}^{D}
=\frac1{|\mathcal I_D|}\sum_{i\in\mathcal I_D}\frac{-1}{|P_D(i)|}
\sum_{p\in P_D(i)}
\log\frac{\exp(s_{ip})}
{\sum_{a\ne i}\exp(s_{ia})}.
\]

### 7.2 家族残差对比损失

只对 AI 样本使用残差 \(r_F\)。

- 正例集合 \(P_F(i)\)：同一家族、尽量不同任务；
- 困难负例：不同家族、相同任务或相同语言。

\[
\hat r_i=\frac{r_F(z_i)}{\|r_F(z_i)\|_2},
\qquad
s^F_{ij}=\frac{\hat r_i^\top\hat r_j}{\tau_F^C}.
\]

\[
L_{SupCon}^{F}
=\frac1{|\mathcal I_F|}\sum_{i\in\mathcal I_F}\frac{-1}{|P_F(i)|}
\sum_{p\in P_F(i)}
\log\frac{\exp(s^F_{ip})}
{\sum_{a\ne i,y_a=AI}\exp(s^F_{ia})}.
\]

只有存在正例与负例的锚点才进入 \(\mathcal I_D,\mathcal I_F\)。采样器需保证跨题同类正例与题目匹配的异类负例；小 batch 不具备这些关系时，不能计算该项。

这两个对比目标承担不同任务：

- \(L_{SupCon}^{D}\) 学总体 AI–Human 边界；
- \(L_{SupCon}^{F}\) 学去除公共来源方向后的家族差异。

### 7.3 题目条件约束

如果存在同题 Human–AI 输出，不直接做 token 差分，而是在样本表示上构造困难关系。设 \(\mathcal P_{HA}=\{(a,h):t_a=t_h,y_a=AI,y_h=H\}\)：

\[
L_{task-hard}^{D}
=\frac1{|\mathcal P_{HA}|}\sum_{(a,h)\in\mathcal P_{HA}}
\operatorname{softplus}\left(m_D-s_D(a)+s_D(h)\right),
\]

它要求同题 AI 的检测 logit 高于 Human。若有同题异家族集合 \(\mathcal P_{FG}\)，定义 \(\ell_f(x)=\log p_F(f\mid AI,x)\)，再加入：

\[
L_{task-hard}^{F}=
\frac1{|\mathcal P_{FG}|}\sum_{(a_f,a_g)\in\mathcal P_{FG}}
\operatorname{softplus}\left(m_F-\ell_f(a_f)+\ell_f(a_g)\right).
\]

主实验设 \(L_{task-hard}=L_{task-hard}^{D}+L_{task-hard}^{F}\)。任一配对集合为空时，该项为 0；没有可靠 `task_id` 时两项都关闭，记录有效配对数。

---

## 8. token 级监督与损失平均

### 8.1 token 级分数

\[
q_i=w_q^\top\tilde h_i+b_q.
\]

对于有真实 token/行标签的 hybrid 数据，使用：

\[
\ell_i=\operatorname{BCEWithLogits}(q_i,y_i).
\]

### 8.2 困难位置聚合

不要默认用所有 token 的均值损失：

\[
\frac1L\sum_i\ell_i.
\]

top-k 损失作为独立消融。它会放大噪声标签的影响，必须和 uniform mean、按类平衡的 mean 比较：

\[
L_{tok-topk}
=\frac1K\sum_{i\in TopK(\ell)}\ell_i,
\qquad K=\min(L,\max(8,\lceil0.1L\rceil)).
\]

同时保留 uniform mean 作为严格对照。表示池化的 mean 与损失的 mean 必须分别报告，不能合并为一个“平均改进”。

对于只有样本级 Human/AI 标签的旗舰模型数据，不伪造 token 标签，不使用 \(L_{tok-topk}\)。样本级监督通过 \(L_D\) 和显著位置池化传播。

---

## 9. 总损失

主方法的总损失为：

\[
\boxed{
L=
\lambda_D L_D
+\lambda_F L_F
+\lambda_{CD}L_{SupCon}^{D}
+\lambda_{CF}L_{SupCon}^{F}
+\lambda_TL_{task-hard}
+\lambda_{tok}L_{tok-topk}
}
\]

第一轮建议：

\[
\lambda_D=1,
\quad
\lambda_F=1,
\quad
\lambda_{CD}=0.1,
\quad
\lambda_{CF}=0.1,
\quad
\lambda_T=0.1,
\quad
\lambda_{tok}=1\text{（仅 hybrid 数据）}.
\]

这些系数只作为起始配置。不能在测试集上调权重；最多使用训练折和固定验证协议。

### 9.1 防止表示坍缩的可选正则

如果训练中发现所有类别的 \(z\) 都向各自中心同时收缩、类间能量差反而消失，加入仅在训练 batch 计算的方差下限：

\[
L_{var}=\frac1D\sum_{j=1}^{D}
\max\left(0,\gamma-\operatorname{Std}_{batch}(z_{:,j})\right)^2.
\]

只在诊断确认坍缩后启用，初始 \(\gamma=0.5\)（对 LayerNorm 后的 \(z\)）。此项不用于制造额外来源信息。

最终可选为：

\[
L\leftarrow L+\lambda_{var}L_{var},
\qquad \lambda_{var}\in\{0,10^{-3}\}.
\]

---

## 10. 与当前 d-det 代码的对应关系

| 新方法组件 | 当前项目对应 | 改动建议 |
|---|---|---|
| `H` | `CodeT5Encoder` | 保留 |
| `z(x)` | `DualScoreModel.features()` 后的池化 | 新增分布统计和显著位置聚合 |
| `s_D` | `w1→s1` 的替代检测分数 | 保留 `s1` 作为 M0，不直接删除 |
| `p_F` | `DiscHead` | 用于对照；新方法使用残差能量或作为后端判别器 |
| `q_i` | `tok_head` | 保留，比较 mean loss 与 top-k loss |
| `s2` | base/instruct hinge | 不作为主方法输入；仅作历史诊断臂 |

第一版代码可以新增：

```text
models/set_pool.py       # μ、variance、attention/top-k pooling
models/manifold.py       # shrinkage covariance and energy
losses/contrastive.py    # detection/family supervised contrastive losses
scripts/eval_flagship.py # family-held-out and generator-held-out evaluation
```

### 10.1 两 epoch 训练协议

为避免流形统计量与编码器同时冷启动，按以下顺序执行：

1. **初始化**：从当前 d-det 检查点提取训练折 \(z\)，只用训练折估计 \(\mu_c,\widetilde v_c,\pi_c\)。
2. **第 1 epoch**：冻结 CodeT5，只训练 `set_pool`、\(w_q\)、能量偏置和对比投影；统计量固定，验证 M1–M4 是否有梯度和类别分离。
3. **第 2 epoch**：默认仍冻结骨干，训练全部新模块；若第一轮显示新增模块有稳定收益，再做一个允许最后 1–2 个 CodeT5 层更新的续训臂。续训臂必须单独标记，不能和冻结结果混报。
4. **统计更新**：每个 epoch 结束后只用训练折重算统计量；验证和测试使用该 epoch 的冻结统计量。
5. **比较公平性**：M0–M5 使用相同起点、相同样本、相同 batch、相同 2 epoch；旧 `s1/s2` 结果另列，不与新流形头混用。

如果服务器端必须全参微调，先固定统计量完成一个 warm-up epoch，再仅在第二 epoch 使用训练折 EMA 更新；不得每个 batch 同时用当前 batch 估计类别统计量并评估同一 batch，否则会产生标签条件泄漏。

---

## 11. 严格对照实验

| 方法 | 表示 | 检测 | 家族 | 作用 |
|---|---|---|---|---|
| M0 | 当前 mean/ABMIL | `s1` BCE | 同数据的全维线性/收缩 LDA 单代码探针 | 公平单代码基线；旧配对 `DiscHead` 另表报告 |
| M1 | `μ` | shrinkage energy | family energy | 检验流形能量 |
| M2 | `μ+variance+stop` | shrinkage energy | family energy | 检验关键位置和分布统计 |
| M3 | M2 | + `SupCon^D` | family energy | 检验检测对比 |
| M4 | M3 | residualized family manifold | + `SupCon^F` | **候选主方法**；须验证残差相对原始表示有效 |
| M5 | M4 | top-k token loss | + token hybrid | 检验损失聚合影响 |

所有方法：

- 相同旗舰模型样本；
- 相同 train/val/test 题目分组；
- 每种方法最多 2 epoch；
- 不生成新样本；
- 不依赖 base/instruct；
- 保存 seed、协方差估计、温度、参数量和显存。

---

## 12. 必须报告的结果

### 检测

- Human/AI macro-F1、AUROC；
- 未见旗舰模型 AUROC/F1；
- 未见语言 F1；
- 长度桶和语言桶；
- Hybrid/Adversarial 的 macro-F1；
- top-k 证据覆盖率。

### 归因

- AI 家族 macro-F1、balanced accuracy；
- 已见家族未见模型；
- 留出整个家族时报告未知家族拒识；不要把新家族强制纳入已见家族 closed-set macro-F1；
- 相近家族的混淆矩阵；
- `p_F` 使用原始 `z` 与残差 `r_F` 的差异。

### 机制验证

- `s_D` 与 family energy 的相关性；
- 去除 \(\delta\) 前后家族准确率；
- M0–M5 的池化与损失平均独立消融；
- 对比正例/困难负例的替换实验；
- 固定编码器输出后置换 token 向量行，核对集合池化的数值不变性；重排原始代码会改变程序和上下文，不进行标签不变性测试；
- 结构单元打乱测试：模型是否依赖真正的代码结构；
- 训练折外估计的中心、协方差和温度检查。

---

## 13. 论文贡献的候选表述

如果 M4 在跨旗舰模型、跨题和未见生成器上成立，可以形成如下论文命题：

> AI 代码检测与模型家族归因共享代码语义，但来源差异具有不同的几何层次。我们提出一种来源流形与条件残差对比方法：先用密度感知能量刻画 Human–AI 分布差异，再从总体来源方向中提取家族残差，并用题目匹配的困难对比学习增强跨模型泛化。

这个命题的关键验证不是分布内分数，而是：

1. 旗舰模型留出泛化；
2. 家族残差相对于总体 AI–Human 方向是否真正提高归因；
3. top-k/分布统计是否减少关键位置被平均池化和平均损失削弱的问题；
4. 所有统计量是否在训练折内估计，避免题目和测试泄漏。

---

## 14. 明确不作为主方法的内容

- base/instruct 主配对训练；
- 单独扩大 `s2_rank`；
- 预设低秩归因瓶颈；
- token 逐位置跨样本差分；
- 无监督 KMeans 家族聚类；
- 只使用普通 mean pooling；
- 用测试集调协方差、温度或损失权重。
