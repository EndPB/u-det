# AutoDL 执行指导：核心假设、后训练差异与归因多实验

日期：2026-10-08  
适用对象：AutoDL 侧执行 AI  
主线状态：**立即恢复核心科研实验；停止把风格、空白、注释和主干规模探针当作当前主线。**

本文件是当前执行入口。它覆盖并替换此前“停止重复风格实验”之后的下一阶段计划；此前 A1/A2/A3、stage-1、stage-2 和 variant-transfer 产物均保留，不能覆盖、重命名或改写。旧 BigCodeBench 暴露集也不能因为本文件而重新读取。

## 1. 论文真正要验证的命题

论文不是要证明某个分类器在一个随机切分上分数更高，而是要回答一个有因果边界的可检验问题：

> 后训练或模型来源差异是否在同一任务条件下留下可读的表示关系；这种关系能否被归因读出，并在未见模型成员、未见任务或未见生成器上保持；同时不会以牺牲 Human/AI 检测为代价。

这句话必须拆成四个不同的问题，不能用一个总分代替：

| 任务 | 核心问题 | 主要统计单位 | 允许的结论 |
|---|---|---|---|
| R：后训练差异 | 同一模型或同一基座的不同训练/提示状态是否可读 | task cluster、model pair | protocol-conditioned difference；只有在严格 lineage 对照下才可讨论因果 |
| F：归因 | 代码来自哪个 generator、系列或模型成员 | task cluster、held-out generator/model | 来源可读性、系列内迁移、未知类拒识 |
| G：外推 | 读到的是来源关系还是闭集指纹 | unseen task / generator / model member | 迁移损失与泛化边界 |
| D：检测 | Human/AI 能否保持 | task cluster | 与 F 分开报告，不能把检测高分称作归因 |

当前的主实验顺序是 **R → F → G → D/H3**。只有 R/F 的受控证据成立后，才做复杂的任务分离机制。

## 2. 已有结果和硬约束

### 2.1 已经完成、无需重做的工作

1. AuthorBench、STACAD、Droid、AICD、CoDET-M4 和 BigCodeBench 的既有强基线、数据角色和暴露账本已经落盘。
2. E17–E21 已经说明：归因不是单一标量聚类问题；同题双侧联合表示比单独差分或单侧表示更有信息，任务中心化有价值但属于转导接口。
3. E31–E36、Stage B 和 Droid v2 已经说明：在当前 768 维表示、当前正对和当前预算下，盲目叠加 SupCon、低秩、正交、完整 DCAN/DMHM 没有稳定实用增益。
4. variant-transfer stage-2 已经证明，固定 BigCodeBench 协议下，11 个系列成员的静态融合读出可得到高迁移 AUROC；这只支持系列内尺寸迁移，不等于后训练因果，也不等于所有 generator 的不变归因。
5. A1/A2/A3 干预结果只作为开发诊断。A1 的 `rstrip(" \t")` 可能修改多行字符串内部，compile 通过不能证明语义等价；A2/A3 也不能被写成特征因果分解。

### 2.2 现在禁止的路线

- 不再扩展空白、注释、字面量、命名、换行的重复消融矩阵。
- 不因“主干表格还不完整”而下载一组 encoder/decoder/参数量权重。主干探针只有在核心读出定义稳定后才有价值。
- 不把 `complete`/`instruct` 名称直接写成纯后训练因果。它们首先是**协议/角色条件差异**；只有 lineage、相同任务、相同基座和训练来源均有证据时，才升级为后训练对照。
- 不把随机切分高分、同一 generator 闭集高分或 dev 选择结果写成 H2/H3 证据。
- 不重新读取已经暴露过的旧 test；任何新的确认性 test 必须是新任务集，先冻结 split 和预注册，再单次读取。

## 3. 统一数学对象和假设

对任务 (t)、模型/生成器 (m)、协议状态 (r)，令代码表示为

\[
h_{m,t}^{(r)}=E_\theta\bigl(x_{m,t}^{(r)}\bigr)\in\mathbb R^d,
\]

其中 (x) 是代码文本，(E_\theta) 必须由配置和权重哈希冻结。对于同一任务的两个状态 (+) 和 (-)，定义

\[
u_{m,t}=[h_{m,t}^{(+)};h_{m,t}^{(-)}],\qquad
S_{m,t}=\frac{h_{m,t}^{(+)}+h_{m,t}^{(-)}}2,\qquad
A_{m,t}=\frac{h_{m,t}^{(+)}-h_{m,t}^{(-)}}2.
\]

完整双侧 (u=[S;A]) 是归因主对象；(A\) 或 \(\Delta=2A\) 只是消融分量，不能代替完整联合结构。若使用同题统计量，定义

\[
\tilde h_{m,t}^{(r)}=h_{m,t}^{(r)}-\mu_{t,\mathrm{train}}^{(r)},
\]

其中 \(\mu_t\) 必须只由训练折中同题兄弟样本估计。用测试题的兄弟输出估计中心再评测，必须标成 transductive，不能当作单样本部署结果。

后训练动机采用 KL 正则化的闭式解。给定预训练分布 (p_0(y\mid x))、奖励 (R(y,x)) 和温度 \(\beta>0\)，

\[
p^*(y\mid x)=\frac{p_0(y\mid x)\exp(R(y,x)/\beta)}{Z(x)},\qquad
Z(x)=\sum_y p_0(y\mid x)\exp(R(y,x)/\beta).
\]

两种状态的对数比为

\[
\log\frac{p_a^*}{p_b^*}
=\log\frac{p_{0,a}}{p_{0,b}}
+\frac{R_a}{\beta_a}-\frac{R_b}{\beta_b}
-\log\frac{Z_a}{Z_b}.
\]

这只能说明后训练差异可能在输出分布中留下相对信号，不能推出共同基座、加性表示分解、正交子空间、低秩归因空间或跨生成器不变性。上述命题必须由实验识别。

论文的三个核心假设写成：

**H1（受控来源可读性）**：在 task-heldout 且控制长度、语言、仓库和协议后，(u) 对来源标签 (f) 的读出优于单侧 (h)、(A) 和强 P0。

**H2（关系迁移）**：同一来源族内，不同 generator/model member 的关系在 generator-heldout 或 member-heldout 上仍有正的、预注册的增量；随机闭集分类不算 H2。

**H3（任务分离）**：加入归因读出后，family 指标提升，检测 AUROC 不显著下降，且检测梯度与归因梯度干扰降低。

## 4. 当前最有价值的数据入口

公开 BigCodeBench 完整索引中，当前审计发现约 120 个 distinct full model，约 115 个 `model_id` 同时具有 `complete` 与 `instruct` 单元，且任务集合重合。执行前必须由脚本重新计算精确数字并写入 manifest，不能手填。

这给出一条无需立即下载新权重的核心入口：

**同一 model_id、同一 task、complete/instruct 的配对差异。**

它的正确标签是 `protocol_conditioned_difference`。它适合先检验 R0、双侧表示和归因接口；不能直接写“证明后训练因果”。严格 causal 子集只有在 `base_model`、版本、训练来源、提示协议和数据 provenance 均有独立证据时才可建立。

已有 11 系列尺寸迁移数据继续用于 F/G 外部验证，但不把系列名、参数量或 `complete/instruct` 自动当成 family truth。`family_is_confirmed=false` 的数据要保留原标签和证据等级。

## 5. 多实验总表

以下实验可以共享一次特征编码，但必须各自有 config、manifest、日志、输出目录和预注册主指标。E0–E4 可在 train/dev 阶段并行执行；任何需要新 test 的实验都等 train/dev 结果和注册文件冻结后再申请单读。

| 编号 | 实验 | 直接回答 | 训练/评测数据 | 首要结果 |
|---|---|---|---|---|
| E0 | lineage 与 pair 支持审计 | 哪些配对真的是同一模型/基座状态 | public full manifest + 官方文档 | pair 数、字段完整率、证据等级 |
| R0 | complete/instruct 协议差异 | 同模型协议差异是否可读 | 115 左右配对模型，task-heldout | AUROC/BA、(u/S/A/\Delta) 增量、model-heldout |
| R1 | 严格同基座后训练对照 | 是否有更接近因果的 post-training 信号 | 仅 lineage 全字段通过的子集 | paired delta、跨模型迁移、混杂敏感性 |
| F0 | exact unit attribution | 单个 generator/model unit 是否可读 | BCC full/instruct，task-heldout | macro-F1/BA、任务聚类 CI |
| F1 | family/series attribution | 系列归因是否超过模型身份捷径 | 11 系列 train/dev，heldout member | member-heldout AUROC/AP、迁移损失 |
| F2 | relational attribution | 双侧关系是否比单侧/差分更稳 | R0 配对 + 多 generator 配对 | (u) 相对 (h/A) 的 paired delta |
| F3 | open-set attribution | 未知 generator 能否被拒识 | 新鲜任务或未参与训练的公开 unit | AUROC、FPR@95、energy/entropy |
| H3 | detection-family joint | 归因是否损害检测 | H1/H2 通过后的固定表示 | family、detection、梯度余弦 |
| D0 | fresh task construction | 现有任务是否不足以支撑确认性证据 | 本机构建新任务池 | provenance、碰撞、split、支持矩阵 |

## 6. E0：先完成配对与 lineage 审计

### 6.1 输入和禁止事项

只读取已有 manifest、unit coverage、model card/官方文档、task id 和元数据字段。不要读取旧 test 代码正文，不下载模型权重，不生成新输出。

### 6.2 机器可读输出

建立独立目录 `artifacts/core_lineage_audit_2026-10-08/`，至少包含：

- `paired_unit_index.jsonl`：`model_id`、`mode`、`family_raw`、`base_model`、`revision`、`parameter_count`、`task_count`、`source_url`、`license`、字段证据等级；
- `lineage_adjudication.json`：`verified`、`partial`、`unknown` 三档，逐字段给出证据；
- `support_matrix.csv/json`：每个 unit 的 task 覆盖、split 支持、可否进入 R/F/G；
- `collision_audit.json`：exact、normalized whitespace、lexical skeleton 和 task id 跨 split 重复；
- `prereg_r0_r1.json`、`commands.txt`、环境、git head/status、SHA256SUMS。

若 `model_id` 相同但 base/version/协议字段不足，保留为 R0 `protocol_conditioned`，禁止升级为 R1 `causal_candidate`。

## 7. R0：同模型 complete/instruct 配对差异实验

### 7.1 训练协议

1. 重建固定 task split；旧 test 任务不进入拟合、特征选择或阈值选择。优先使用 train/dev，确认性评测等待新鲜 test。
2. 每个 pair 只在同一 `model_id`、同一 task 上组成；不可把不同 model_id 的 complete 和 instruct 拼成伪配对。
3. 先运行冻结 CodeT5-small/base 和 TF-IDF/P0；不用新 backbone。
4. 把 `model_id`、参数量、长度、语言、task difficulty、代码模板统计作为控制或分层变量，不能将 model_id 直接作为输入特征。

### 7.2 读出矩阵

对每个配对分别训练同一类线性读出，顺序固定为：

1. 单侧 (h^{(complete)})；
2. 单侧 (h^{(instruct)})；
3. (A/\Delta)；
4. (S)；
5. 联合 (u=[S;A])；
6. P0 lexical/style/metadata 控制；
7. `u` 加长度、语言和规模控制的 late-fusion。

主指标为 model-balanced AUROC、balanced accuracy、task-macro 和 task-cluster bootstrap 95% CI；主比较是 `u - best_single`、`u - P0`，差值必须同时保留点估计和 `delta_mean`。

### 7.3 三个泛化切分

- task-heldout：检验不是记忆题目；
- model-heldout：整组 model_id 留出，检验是否只是模型指纹；
- size-heldout：按参数量桶留出，检验是否只是规模轴。

`complete/instruct` 的 R0 结果只能写“协议条件差异可读性”。若通过 R0，才进入 R1；若只在随机 task split 高而在 model-heldout 归零，结论收窄为闭集模型指纹。

## 8. R1：严格后训练候选，不得从名称推断

R1 不是强行制造的实验。先从 E0 的 provenance 中筛选满足以下全部条件的 pair：

\[
\text{same base/version} \land \text{documented post-training relation}
\land \text{same task/protocol} \land \text{independent provenance}.
\]

缺一项就留在 R0。若没有任何 pair 通过，必须报告 `R1 unavailable`，不能用不同模型、不同规模、不同模板替代。

对 R1，主报告增加：

- paired representation delta：\(\delta_t=h_t^{(post)}-h_t^{(base)}\)；
- 若有 token log-prob，报告 \(r_t=\log p_{post}(y_t\mid x_t)-\log p_{base}(y_t\mid x_t)\)；没有概率就不伪造；
- 按 task、语言、长度、难度和 base scale 分层的效果；
- model-heldout 和新任务 heldout；
- 与 R0 协议差异的并列表格，禁止合并成一个“后训练分数”。

## 9. F0/F1：归因主实验

### 9.1 F0 exact unit attribution

标签先使用观测到的 exact unit，而不是未经核验的 family。严格 task-heldout，至少比较：

\[
\{h^+,h^-,A,S,[h^+;h^-],P0\}.
\]

闭集 exact unit 的高分只回答“有无来源指纹”。必须再做 model-heldout；未见类不报告未定义的 macro-F1，只报告拒识或 one-vs-rest 指标。

### 9.2 F1 series/member transfer

沿用 stage-2 的 11 系列成员，但把它明确标成 `series-transfer`。重算只能使用已保存分数；新训练只能使用 train/dev。主结论保留：P0 fusion 迁移 AUROC 约 0.945–0.964，内容融合高于 size/length 与 metadata 控制；这不是 R1 因果证据。

若要增加成员，必须先预注册 heldout 顺序和负集版本，再构造 train/dev。不得因为 dev 结果选择最有利的系列或尺寸桶。

## 10. F2：归因流形与后训练差异联合实验

F2 是当前最接近论文方法贡献的实验，但必须按增量顺序运行，不能同时堆叠机制。

### 10.1 基线和增量

先冻结一个不训练编码器的线性 baseline：

\[
\ell_0(u)=W_0u+b_0.
\]

然后只允许一个受控残差：

\[
\ell_F(u)=\ell_0(u)+\gamma W_R r_\phi(u),
\qquad \gamma=0\text{ 初始化}.
\]

残差训练目标为

\[
\mathcal L=\mathcal L_F
+\lambda_{rel}\mathcal L_{rel}
+\lambda_{reg}\lVert W_R\rVert_2^2,
\]

其中 

\[
\mathcal L_F=-\frac1n\sum_i\log p_\phi(f_i\mid u_i),
\]

\(\mathcal L_{rel}\) 只在同 family、不同 generator 的训练正对上使用；未见 generator 不得参与拟合。可以先使用 cosine/ranking 关系损失，随后才选择一种局部邻域机制。不要并行加入 SupCon、QDA、低秩、正交和 DCAN。

### 10.2 必须比较的消融

固定同一 split、同一分类器和正则选择：

1. `single h`；
2. `A/Delta`；
3. `S`；
4. `u=[S;A]`；
5. `u + task-centered`（单列 transductive）；
6. `u + one relational mechanism`；
7. P0 和随机错误配对控制。

只有 `u + one mechanism` 在至少两个 heldout 折中相对 `u` 和 P0 都方向一致，且 task-cluster CI 不被零覆盖，才允许进入 H3。否则保留为“当前表示/数据条件下无稳定关系增益”。

## 11. F3：未知类和真正外推

闭集归因不能支撑 ACL 的泛化结论。至少准备一种未知类检验：

- 训练于若干 exact units/series，整组留出另一个 unit；或
- 训练于旧任务，评测新任务集上的公开输出；或
- 训练于已有系列，评测新公开 generator 的同任务输出。

报告 AUROC、AUPR、FPR@95、energy/entropy 分布和 task-cluster CI。不得报告“未见类 macro-F1”作为主指标，也不得把未知类拒识写成 family 分类成功。

## 12. H3：检测—归因任务分离（条件实验）

H3 只能在 H1/H2 至少有一个受控出口后执行。匹配同一编码器、batch、epoch、seed 和参数预算，比较：

- detection-only：\(\mathcal L_D\)；
- family-only：\(\mathcal L_F\)；
- joint：\(\mathcal L_D+\alpha\mathcal L_F\)；
- joint + gradient separation：
\[
\mathcal L_{sep}=\max(0,\cos(g_D,g_F)-m),
\]
其中 (g_D=\nabla_\theta\mathcal L_D)、(g_F=\nabla_\theta\mathcal L_F)，(m) 在 dev 预注册。

同时报告 detection AUROC、family BA/F1、迁移损失、梯度余弦和 Pareto 曲线。只要 detection 下降，就不能称 H3 成功；只报告联合分数是不合格的。

## 13. 本机构建更强数据：D0 立即启动，生成按闸门执行

现有数据若不足以支撑新任务/未知类确认性检验，由本机 CPU、RTX 3050 和大虚拟内存完成**任务池构建、许可核验、哈希和碰撞审计**。这一步不需要把全部 raw corpus 上传 AutoDL。

### 13.1 目标数据结构

建立 `fresh_same_task_v1`，目标不是轻量样例，而是可支撑多折统计的独立任务集：

- 300–500 个任务作为初始目标，最终数量由 provenance 和去重结果决定；
- 至少 3 个有独立版本/许可证据的 generator 或公开输出来源；
- 任务先切分 train/dev/test，再允许任何输出读取或生成；
- task、prompt、代码、规范化骨架三层 hash 跨 split 均为 0；
- 记录语言、仓库、难度、长度和许可证，不静默混拼 CodeContests、BigCodeBench、LCv2；
- 如果只有公开模型输出而没有可证实的 lineage，标签写 `observed_source`，不能写 `post_training_family`。

### 13.2 本机必须交付给 AutoDL 的最小材料

只上传索引和必要特征，不上传无关完整语料：

1. `tasks.jsonl`：task_id、prompt_hash、source、license、language、split；
2. `unit_manifest.jsonl`：unit_id、generator、model_id、mode、revision、provenance、license；
3. `pair_index.jsonl`：可配对 unit、task、pair_type、证据等级；
4. `split_index.csv`、`collision_audit.json`、`lineage_adjudication.json`；
5. 若已存在公开输出，先传压缩后的必要代码/特征及 SHA256；原始大文件留本机，按实验所需切片传输；
6. `fresh_preregistration.json`：主问题、主指标、heldout 顺序、一次 test 读时间和停止规则。

新 API 调用、付费生成、下载新权重或读取新 test 前，必须让配置中的 `generation_allowed`、`test_read_allowed` 和相应授权记录显式变为 true。未授权时只做 D0 索引和审计。

## 14. 执行顺序和资源调度

### 第 0 阶段：共同特征缓存

一次完成 E0 和所有 train/dev 的 CodeT5-small/base、TF-IDF、metadata、长度/规模控制特征；每个特征记录模型哈希、tokenizer、截断、池化和输入行顺序哈希。

### 第 1 阶段：并行 train/dev

可并行运行 R0、F0、F1、F2 的 baseline 和消融；每个实验最多使用 3 seeds，task-cluster bootstrap 抽样索引共享但结果文件独立。不要重复计算每个读出的 bootstrap，按 task 缓存。

### 第 2 阶段：根据出口决定

- R0 只在 model-heldout 仍有信号时启动 R1；
- F0 只有跨 task 有信号时进入 F2；
- F2 只有双侧联合超过强 P0 且至少两个 heldout 折同向时进入 H3；
- 任一路径失败，保留负结果并转向数据/标签识别性，不靠扩大 backbone 数量补救。

### 必须写入的执行开关

```json
{
  "training_allowed": true,
  "generation_allowed": false,
  "test_read_allowed": false,
  "old_test_reuse": false,
  "new_fresh_test_read": false,
  "weights_downloaded": false,
  "source_status": "server_reconstruction_only"
}
```

如果原件仍未补齐，所有报告继续保留 `original_bundle_verified=false`，不得声称与本机原件字节一致。

## 15. 停止规则、判读规则和回传格式

### 15.1 停止规则

- R0 在 model-heldout 归零：停止“后训练差异可迁移”表述，只保留协议指纹结果。
- F0 高、F2 在 heldout generator 归零：归因主要是闭集 generator fingerprint，停止堆叠几何损失。
- 双侧 (u) 不超过 P0，或提升只在一个折：H1/H2 不过，不启动 H3。
- 只有 metadata/size/length 读出高：记录混杂，不能称语义归因。
- R1 无完整 lineage：写 `unavailable`，不拿 R0 冒充因果。
- fresh task pool 三层碰撞不为零或许可不清：该任务集不能进入主表。

### 15.2 每个实验必须回传

- `hypothesis.json`：假设、主/次指标、允许的结论和禁止表述；
- `config.json`、`execution_switches.json`、`data_role_matrix.json`；
- `metrics.json`：点估计、task-macro、paired `delta_mean`、500 次 task-cluster CI；
- `predictions.npz/csv.gz`：只保留必要 id、label、score、fold，不上传无关 raw code；
- `report.md`：数据来源、split、feature roles、混杂、判读和失败出口；
- `commands.txt`、完整日志、`git_head.txt`、`git_status.txt`、环境包版本和 `SHA256SUMS.txt`。

回传摘要必须回答六句话：

1. 本实验检验哪个 H/R/F 命题？
2. 训练、dev、test 各读了什么，是否读过旧 test？
3. 主比较相对哪个 P0，delta 是点差还是 bootstrap `delta_mean`？
4. 信号在 task-heldout、model-heldout、generator-heldout 中是否保持？
5. 哪些混杂仍无法排除？
6. 结果是否满足下一阶段闸门；若不满足，停止什么？

## 16. ACL 叙事出口

如果 R0/R1、F2 和 G 都通过，论文主线可以写成：

\[
\text{post-training/protocol relation}
\rightarrow
\text{paired attribution manifold}
\rightarrow
\text{held-out source transfer}
\rightarrow
\text{detection--attribution separation}.
\]

如果只有 R0/F0 成立，论文仍然有价值，但只能写“来源可读性和协议条件差异”，不能写后训练因果或跨 generator 不变性。如果 R0、F2 都失败，最诚实的结论是：当前公开任务、表示和标签结构不足以识别所声称的后训练来源几何；下一步应改进 provenance/task design，而不是继续堆叠网络。

本文件的目标是让 AutoDL 立即产生能进入 ACL 证据链的多实验结果，而不是再产生一组无法回答核心假设的新分数。
