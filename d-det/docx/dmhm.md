# DMHM 论文：核心假设与完整流程提取
> 来源：DMHM: Density-aware Manifold Learning and Hybrid Mahalanobis Energy for LLMs-generated Text Detection (ACL 2026)。本文档仅提取**假设体系**与**执行流程**两部分。
---
# 第一部分：核心假设体系
## 假设 1：任务范式假设（OOD 视角）
- 将 LGT 检测建模为 **OOD 检测问题**：**LGT = In-Distribution (ID) 样本，HWT = Out-of-Distribution (OOD) 样本**
- 依据：HWT 风格多样、自由度高 → 稀疏分布，无法建模为单一分布；二分类范式会过拟合已观察到的 HWT
- 依据：LGT 受训练数据与模型架构约束 → 形成更紧凑的分布，适合作为 ID
## 假设 2：LGT 密度不均匀假设（本文核心动机，挑战前人假设）
**被挑战的旧假设**：现有 OOD 检测方法假设 LGT 是单一紧凑同质分布。
**本文新假设**：LGT 是**复合分布，密度非均匀**：
| 类别 | 来源 | 分布特征 | 可检测性 |
|---|---|---|---|
| Easy-LGT | 弱 LLM | 统计规律性强、语义多样性低 → 占据有限语义区域，形成**密集区** | 易检测 |
| Hard-LGT | 强 LLM 或被攻击文本 | 多样性高 → 远离密集 LGT 模式，**落入低密度区域且与 HWT 重叠** | 难检测 |
**推论假设**：**密度是 LGT 检测的有效判别特征**——显式建模密度变化可分离“密集 easy-LGT”与“hard-LGT/HWT 共享的稀疏区域”。
## 假设 3：嵌入空间几何假设
- LGT 比 HWT 呈现更规律、更低噪声的统计分布，在嵌入空间呈结构化模式
- LGT 倾向集中于**紧凑平滑的低维流形**；HWT 分布更分散多样
- 因此不在原始嵌入空间判别，而应学习**流形感知嵌入**
## 假设 4：流形学习相关假设
- **局部保持原理**：原始空间中相邻样本应保持在同一局部邻域内（高亲和力/大权重 $w_{ij}$ 的样本对应保持接近）
- **度即密度**：邻域权重之和 $D_{ii}=\sum_j w_{ij}$ 代表样本 $z_i$ 周围的局部密度
- 权重 $w_{ij}$ 大小控制平滑强度：密集区强约束、稀疏区弱约束（避免过度正则）
## 假设 5：马氏距离度量假设
- 传统马氏距离用全局协方差 $\Sigma^{-1}$ 标准化，可捕捉各向异性嵌入空间的真实分布结构
- **但在非均匀密度流形上，单一全局协方差的估计被高密度区域主导** → 稀疏区域几何被抑制 → 距离估计有偏
- 修正假设：高密度区应侧重全局协方差（降低局部噪声），低密度区应侧重局部协方差（捕捉局部几何）
## 假设 6：能量学习假设（继承自 Liu et al. 2020; Ouyang et al. 2021）
- 能量学习给 ID 样本**低能量**、给 OOD 样本**高能量**，形成可分离的能量间隙，可作 OOD 评分函数
- 层级假设：easy-LGT 距 HWT 更远 → 能量边际更大（$m_1>m_2$）；对 hard-LGT 施加等大边际会破坏能量层级导致次优
## 假设 7：生成器身份作为代理变量的假设（标签使用合法性）
- 先验研究（Sadasivan et al., 2023; He et al., 2024）：检测难度存在与模型容量、对齐、规模相关的**一致模型级规律**
- 生成器身份被用作**不可直接观测的“可检测性隐因子”的工具变量**，而非语义标签
- **难度外生于生成器**：检测难度源于模型能力×提示×解码策略的交互，而非任一固定生成器的内在属性 → 因此所有对抗扰动样本无论来源一律标为 Hard-LGT
- **非泄露声明**：标签仅作为训练时结构正则（类似 curriculum learning / 类条件正则）注入弱分布级几何偏置；检测器从不学习区分生成器；推理时生成器身份不可用也不需要
## 假设 8：Mini-batch kNN 的理论假设（附录 B.2）
流形 $\mathcal{M}\subset\mathbb{R}^d$ 满足：
1. 紧凑、$C^2$ 光滑、曲率有界；内在维度 $m\ll d$
2. 密度 $p(z)$ 连续有界，存在常数 $c_1\le p(z)\le c_2$
3. 嵌入函数 $f\in C^2(\mathcal{M})$
4. 渐近条件：$B\to\infty$、$k\to\infty$ 且 $k/B\to 0$、$\sigma\to 0$ 且 $\sigma>\Theta\big((\log B/B)^{1/m}\big)$
在此条件下 mini-batch kNN 正则项收敛为**密度加权 Dirichlet 能量**；batch 图只能做**局部随机近似**而非全局拓扑恢复——但判别任务只需局部几何保持即可。
## 假设 9：工程可行性假设
- 中等 batch size（128）已足以捕捉稳定流形几何；超过后偏差项主导、收益递减
- batch 内 kNN 独立计算可替代全局 kNN 图（$N\le512$、$k\le10$ 开销可忽略）
- 小 batch 无法可靠估计局部结构时，安全回退到全局协方差
---
# 第二部分：完整执行流程
## 0. 总体流程图
```
【阶段0】数据准备与 Easy/Hard 标注（仅训练时）
      │
【阶段1】文本编码：x_i → z_i = φ(x_i)   [Unsup-SimCSE-RoBERTa]
      │
【阶段2】模块1：密度感知流形学习（3步）
      │   Step1 邻域权重矩阵 → Step2 Laplacian 平滑 → Step3 密度正则
      │   产出：流形感知嵌入空间
      │
【阶段3】模块2：密度自适应混合马氏度量（5步）
      │   局部协方差 → 局部密度 → 密度权重 α_i → 混合协方差 → 距离 D(z_i)
      │
【阶段4】模块3：分布分离
      │   能量边际学习 L_energy + SimCLR 对比学习 L_con
      │
【阶段5】联合训练：L = L_energy + L_con + L_d-manifold
      │
【阶段6】推理：min(D(z_q, μ_e), D(z_q, μ_h)) 与阈值比较
```
## 阶段 0：数据准备与标签流程（训练前）
1. 收集文本：LGT（含 easy/hard 来源）与 HWT
2. 按 Easy/Hard 判据打标（**仅训练时使用**）：
   - **Hard-LGT** = 参数量 > 30B 的 LLM 文本（如 text-davinci-003、LLaMA-65b/30b、OPT-30b、GLM-130B、GPT-4、Cohere、Mistral、MPT）∪ **所有对抗攻击样本**（无论来源）
   - **Easy-LGT** = 小型/早期模型文本（如 GPT-2），且在多个主流检测器上可被高置信检出
3. 三分类集合：$\mathcal{M}_{\text{easy}}$、$\mathcal{M}_{\text{hard}}$、$\mathcal{M}_{\text{human}}$
4. 用整个训练集（不含验证/测试）预初始化分布中心 $\mu_e,\mu_h$ 与全局协方差 $\Sigma$
## 阶段 1：编码
- 每个文本 $x_i$ 经编码器得到嵌入 $z_i=\phi(x_i)$
- 序列长度上限 512 tokens
## 阶段 2：模块一 —— 密度感知流形学习（§3.1）
**Step 1：邻域权重矩阵构建**
1. 对每个 $z_i$ 找 $r$-最近邻 $\mathcal{N}_r(z_i)$
2. 构建权重矩阵 $W\in\mathbb{R}^{N\times N}$：
$$
w_{ij}=\begin{cases}\exp\left(-\dfrac{\|z_i-z_j\|^2}{2\sigma^2}\right), & z_i\in N_k(z_j)\ \text{或}\ z_j\in N_k(z_i)\\[1mm] 0, & \text{否则}\end{cases}
$$
（$\sigma$ 为温度系数；实现上用余弦相似度 kNN 图，取互邻域并集对称化）
**Step 2：Laplacian 流形平滑**
1. 计算度矩阵 $D_{ii}=\sum_j w_{ij}$，图拉普拉斯 $\mathcal{L}=D-W$
2. 计算平滑损失：
$$
L_{\text{manifold}}=\text{Tr}(X^\top\mathcal{L}X)=\frac{1}{2}\sum_{i,j}w_{ij}\|z_i-z_j\|^2
$$
效果：大 $w_{ij}$（密集区）→ 强惩罚 → 邻域更紧凑；小 $w_{ij}$（稀疏区）→ 弱约束
**Step 3：局部密度感知正则化**
1. 定义局部密度：$\text{dens}(z_i)=D_{ii}=\sum_j w_{ij}$
2. 对三类样本分别施加：
   - Easy-LGT（增强密度）：$\mathcal{R}_{\text{easy}}=-\frac{1}{|\mathcal{M}_{\text{easy}}|}\sum_{i\in\mathcal{M}_{\text{easy}}}\log(\text{dens}(z_i))$
   - Hard-LGT（弱权重紧凑，权重 $\rho_h$ 防过度正则）：$\mathcal{R}_{\text{hard}}=-\frac{\rho_h}{|\mathcal{M}_{\text{hard}}|}\sum_{i\in\mathcal{M}_{\text{hard}}}\log(\text{dens}(z_i))$
   - HWT（反向正则、压入稀疏区）：$\mathcal{R}_{\text{human}}=\frac{1}{|\mathcal{M}_{\text{human}}|}\sum_{i\in\mathcal{M}_{\text{human}}}\log(1+\text{dens}(z_i))$
3. 合并：$L_{\text{dens}}=\mathcal{R}_{\text{easy}}+\mathcal{R}_{\text{hard}}+\mathcal{R}_{\text{human}}$
4. 模块目标：$L_{\text{con}}=L_{\text{manifold}}+L_{\text{dens}}$
## 阶段 3：模块二 —— 密度自适应混合马氏度量（§3.2）
按顺序执行五步：
| 步骤 | 操作 | 公式 |
|---|---|---|
| A. 局部统计 | 对 $z_i$ 找 $k$-NN，算局部均值与局部协方差 | $\mu_i=\frac{1}{k}\sum_{j\in\mathcal{N}_k}z_j$；$\Sigma_i^{\text{local}}=\frac{1}{k-1}\sum_{j\in\mathcal{N}_k}(z_j-\mu_i)(z_j-\mu_i)^\top$ |
| B. 局部密度 | KNN 平均距离倒数 | $\rho_i=\dfrac{k}{\sum_{j\in\mathcal{N}_k(z_i)}\|z_j-z_i\|}$ |
| C. 密度权重 | Min-max 归一化 | $\alpha_i=\dfrac{\rho_i-\rho_{\min}}{\rho_{\max}-\rho_{\min}+\epsilon}$（$\alpha_i\approx1$ 高密度，$\approx0$ 低密度） |
| D. 混合协方差 | 密度加权插值全局与局部协方差 | $\widetilde{\Sigma}_i=\alpha_i\,\Sigma+(1-\alpha_i)\,\Sigma_i^{\text{local}}$ |
| E. 距离计算 | 类中心马氏距离 | $\mathbb{D}(z_i)=\sqrt{(z_i-\mu_k)^\top\widetilde{\Sigma}_i^{-1}(z_i-\mu_k)}$，$\mu_k=\frac{1}{\|\mathcal{M}_k\|}\sum z_j$ |
## 阶段 4：模块三 —— 分布分离（§3.3）
**4a. 能量边际学习**
1. 计算各分布到中心的平均距离：$\mathcal{D}(\mathcal{M}_k,\mu)=\frac{1}{|\mathcal{M}_k|}\sum_{z_i\in\mathcal{M}_k}\mathbb{D}(z_i,\mu)$
2. 计算四个能量：$E_{\text{easy}}=\mathcal{D}(\mathcal{M}_{\text{easy}},\mu_e)$，$E_{\text{hard}}=\mathcal{D}(\mathcal{M}_{\text{hard}},\mu_h)$，$E_{H\text{-}e}=\mathcal{D}(\mathcal{M}_{\text{human}},\mu_e)$，$E_{H\text{-}h}=\mathcal{D}(\mathcal{M}_{\text{human}},\mu_h)$
3. 双阈值 softplus 边际损失：
$$
\mathcal{L}_{\text{energy}}=\text{softplus}[m_1-(E_{H\text{-}e}-E_e)]+\text{softplus}[m_2-(E_{H\text{-}h}-E_h)]
$$
约束：$m_1>m_2$（层级边际）；效果：HWT 推向高能量、LGT 保持低能量
**4b. 对比学习**
1. 正样本 = 同分布平均嵌入 $\bar{z}_k=\frac{1}{|\mathcal{M}_k|}\sum z_k$（分布内紧凑）
2. 负样本集 $\mathcal{J}$ = 相反分布的全部样本（实例级判别）
3. SimCLR 损失：
$$
\mathcal{L}_{\text{con}}=-\log\frac{\exp(z_i\cdot\bar{z}_k/\tau)}{\exp(z_i\cdot\bar{z}_k/\tau)+\sum_{j\in\mathcal{J}}\exp(z_i\cdot z_j/\tau)}
$$
## 阶段 5：训练流程
按以下顺序执行（每个 epoch 内联合优化）：
1. 密度感知流形学习优化 $L_{\text{d-manifold}}$ → 样本进入流形感知嵌入空间
2. 马氏度量模块计算样本到中心（$\mu_e,\mu_h$）的距离
3. 以该距离为能量执行能量学习 $\mathcal{L}_{\text{energy}}$
4. 联合对比损失 $\mathcal{L}_{\text{con}}$
5. **总目标**：
$$
L_{\text{our-OOD}}=L_{\text{energy}}+L_{\text{con}}+L_{\text{d-manifold}}\quad(\lambda_1=\lambda_2=\lambda_3=1)
$$
训练配置：AdamW + 余弦退火；lr = 2e-5（Deepfake/Raid）/ 5e-6（M4）；weight decay 1e-4；20 epochs；batch = 128；单卡 A800；跨 LLM/语言/领域联合估计协方差与邻域
## 阶段 6：推理流程
```
查询文本 x
   │ 编码
   ▼
嵌入 z_q
   │ 分别计算到两个 LGT 中心的马氏距离
   ▼
D(z_q, μ_e), D(z_q, μ_h)
   │ 取较小者作为决策分数
   ▼
score = min(两距离)
   │
   ├─ score > 阈值 E_τ  →  OOD 样本 → 判为 HWT
   └─ score ≤ 阈值 E_τ  →  ID 样本  → 判为 LGT
```
关键性质：**推理无需生成器身份**（easy/hard 标签不参与）；无 oracle 依赖；可直接泛化到未见/未来 LLM（只要输出落在相似可检测性区间）。
## 工程实现流程细节（协方差与 kNN 计算）
1. 所有 kNN 图、Laplacian 权重、局部密度、协方差均**在每个 mini-batch 内独立计算**，每次迭代重算、不缓存、不建全局 kNN 图
2. $\widetilde{\Sigma}_i$ 虽按样本定义，但局部协方差仅用于捕捉邻域二阶统计；**所有矩阵求逆在单批级协方差上执行**，$O(d^3)$ 求逆每批一次、与 batch size 无关
3. 防数值不稳：混合满秩全局协方差 + 对角正则 $\Sigma_{\text{mix}}\leftarrow\Sigma_{\text{mix}}+\epsilon I$
4. batch 过小无法可靠估计局部结构 → **安全回退到全局协方差**
5. 关键超参默认值：$\rho_h=0.5$，$m_1=2$，$m_2=1.5$（强制 $m_1>m_2$），$k=r=10$，batch=128（理论拐点：偏差项主导，之后收益递减）
---
*本提取覆盖论文正文 §3（Method）、§3.4（Training/Inference）、附录 A.3–A.4（实现与标签判据）、附录 B.1–B.2（理论假设）。*