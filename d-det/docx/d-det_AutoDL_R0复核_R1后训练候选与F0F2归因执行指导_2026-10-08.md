# AutoDL 下一轮执行指导：R0 复核、R1 候选与 F0/F2 归因

日期：2026-10-08  
前置提交：`1c9db16`  
入口文档：[`d-det_AutoDL_核心假设后训练差异与归因多实验指导_2026-10-08.md`](d-det_AutoDL_核心假设后训练差异与归因多实验指导_2026-10-08.md)

本轮 R0 结果足以证明：在同一 `model_id`、同一任务的 `complete/instruct` 配对中，存在很强的**协议条件差异可读性**，且 model-heldout 没有归零。它还不足以证明后训练因果，也不足以证明已经完成了模型来源归因。下一轮必须同时做三件事：

1. 修正 R0 的统计口径和机制对照；
2. 对已有 lineage 候选进行 R1 train/dev 配对评估；
3. 运行真正的 exact-unit/series attribution 与 F2 关系实验。

旧 test 不读取，新的 test 也不读取；不下载新权重、不生成新输出。所有工作限于已有 train/dev 文本、已有冻结特征和已保存分数。

## 1. R0 当前结论和必须收窄的表述

### 1.1 可以保留的结论

- 同一 `model_id` 的 complete/instruct 配对方向可被 CodeT5-base 双侧读出识别。
- task-heldout 和 model-heldout 仍有明显信号，说明结果不能简单归因于“只记住了同一模型的训练行”。
- `S_sanity=0.50` 和 `u_linear≡delta_pair` 是实现/代数核对，不是机制证据。
- style、长度和协议差异仍然很强，必须保留为竞争性解释。

### 1.2 禁止的结论

- 禁止写“已经证明后训练因果”。R0 标签固定为 `protocol_conditioned_difference`。
- 禁止写“非线性 S 条件化已经证明是来源机制”。当前 `S_sanity` 把 \(\Delta S\) 构造成恒等零，不能检验 S 是否有条件性贡献。
- 禁止把 pair-direction accuracy 直接写成单样本 family attribution AUROC。它的标签是“同一配对中哪一侧是 instruct”。
- 禁止把 `seen`、`task_dev`、`model-heldout` 的数字混在一个均值里；每个域必须有独立样本数、模型数、任务数和 bootstrap 单位。

## 2. R0 复核：不重读文本，只重算保存分数

建立 `artifacts/r0_protocol_diff_reaudit_2026-10-08/`。输入仅为 `r0_protocol_diff_2026-10-08/local/scores_*.npz`、对应 axis 元数据和配置；不得再次读取 `records.jsonl` 的旧 test 行。

### 2.1 修复 task-cluster bootstrap

当前实现对有放回抽到的任务只按原任务 ID 聚合一次，重复抽样的任务没有重复计权。因此不是标准的 cluster bootstrap。重写为：

\[
I_b=\mathop{\Vert}_{q\in Q_b} I(q),
\]

其中 \(Q_b\) 是从任务集合 \(Q\) 有放回抽取的长度为 \(|Q|\) 的序列，\(\Vert\) 表示按序列逐次拼接；同一任务出现两次就保留两份。task-macro 应定义为

\[
M_b=\frac1{|Q|}\sum_{j=1}^{|Q|}a(q_{b,j}),
\]

而不是对去重后的任务求一次平均。model-balanced pair accuracy 则在每个 bootstrap 样本内先对每个 model 计算，再在 model 上平均；如果某个 model 在该 bootstrap 样本没有有效行，按预注册规则跳过并记录有效 model 数。

同时增加 model-cluster bootstrap：

- task-heldout 主 CI：task-cluster；
- model-heldout 主 CI：model-cluster；
- 交叉敏感性：同时给出另一种 cluster 的区间，不把两者当独立证据。

输出必须包含：`delta_point`、`delta_mean`、`ci95`、`frac_le_0`、`n_clusters`、`effective_clusters`。500 次重采样不能写成后验概率。

### 2.2 修复汇总键和 representation 标签

重新生成 `summary_paired_reaudit.json`，每个键明确写成：

```text
{representation=base|small, domain=task_dev|seen|model_dev|size_dev,
 readout=u_mlp|u_linear|delta_pair|..., metric=pair_acc_mb|task_macro}
```

不得让 `task_-1_base` 同时承载 `task_dev` 和 `seen`，不得让 small 结果复用 base 的汇总名。报告中每个表格明确列出：

- fit models/tasks；
- eval models/tasks；
- readout representation；
- 评测域；
- bootstrap cluster；
- 是否使用文本重新拟合。

### 2.3 复核成对方向的定义

R0 的 pair score 应固定为

\[
d_{m,t}=g(h_{m,t}^{(instruct)},h_{m,t}^{(complete)})
-g(h_{m,t}^{(complete)},h_{m,t}^{(instruct)}).
\]

正类是 instruct，complete/instruct 交换时分数必须变号。对随机交换 50% 的配对方向，结果必须回到 0.50 附近。这个 permutation check 是 R0 必须补的实现核对。

## 3. S-A 条件性：用真正的负对照替代 S_sanity

### 3.1 必须运行的四个模型

在相同 fit/eval split、相同 hidden width、相同 epoch、相同 seed 下比较：

1. `A_only_mlp`：输入 \(A\)，不输入 \(S\)；
2. `S_only_pair`：输入 \(S\) 的两个方向，作为结构性常数对照，预期 0.50；
3. `SA_full_mlp`：输入 \([S;A]\)，即当前 `u_mlp`；
4. `SA_S_shuffle_mlp`：在每个 fit split 内按 task 分层随机置换 S，保留 A、标签和 task 频数，打破同一配对的 S-A 关系。

这里的关键比较是

\[
\Delta_{SA}=M(S,A)-M(A),\qquad
\Delta_{shuffle}=M(S_{perm},A)-M(A).
\]

只有当 `SA_full − A_only` 在 task-heldout 和 model-heldout 同向，而 `SA_S_shuffle − A_only` 消失或显著更小，才可以写“联合非线性交互具有条件性证据”。仍然不能把它写成后训练因果。

### 3.2 混杂残差对照

再运行一个 `S_resid_mlp`：用 fit split 的 style、length、metadata、size 预测 S，取残差 \(S_\perp\)，只允许在训练折估计回归器。评测折不能用自身统计量拟合。报告：

\[
M(S,A)-M(A),\quad M(S_\perp,A)-M(A),\quad
M(S_{perm},A)-M(A).
\]

如果只在原始 S 上有效，而在残差 S 或置换 S 上消失，结论是 S 中仍含协议/表面混杂，不能称来源几何。

### 3.3 资源与输出

这些实验只需复用 `emb_base.npz`、axis 和已有样本；不重编码、不读旧 test。输出到 `r0_mechanism_controls_2026-10-08/`，包含模型权重、每对 score、bootstrap 结果、permutation seed 和 `mechanism_claim.json`。若四个读出方向不一致，直接写 `mechanism_unresolved`。

## 4. R1：六条 lineage 候选的严格 train/dev 评估

E0 发现的六条 HF 文档链暂定为 `partial_candidates`。先做候选级评估，不能升级为 causal。

### 4.1 进入 R1 的硬条件

每一条 child/base 边必须同时满足：

\[
\text{same task set}\land\text{same generation mode}\land
\text{same evaluation protocol}\land\text{documented base relation}\land
\text{independent provenance}.
\]

版本不一致、base 只在模型树而非模型卡正文出现、或者 child/base 使用不同任务集时，保留为 `partial`，不进入 R1 主结果。

候选边至少包括：

- Athene-70B ↔ Meta-Llama-3-70B-Instruct；
- Athene-V2-Agent / Athene-V2-Chat ↔ Qwen2.5-72B-Instruct；
- Sky-T1-32B-Flash / Sky-T1-32B-Preview ↔ Qwen2.5-32B-Instruct；
- QwQ-32B-Preview ↔ Qwen2.5-32B-Instruct。

具体 edge 由 E0 的 `lineage_adjudication.json` 冻结，不能按结果删选。

### 4.2 R1 的配对对象

对同一 task 的 child/base 输出，构造：

\[
u^{R1}_t=[h_t^{child};h_t^{base}],\quad
S^{R1}_t=\frac{h_t^{child}+h_t^{base}}2,\quad
\Delta^{R1}_t=h_t^{child}-h_t^{base}.
\]

先做方向判别：child 是否可由配对表示识别；再做候选边识别：同一 task 上，模型输出属于哪条 child/base edge。前者是 R1 pair readout，后者才接近归因。

必须并列：单侧 child/base、\(\Delta\)、联合 \([S;\Delta]\)、style/length/metadata 控制。所有指标只用 train/dev；主 CI 用 task-cluster，edge 数不足时只做描述性分层。

### 4.3 R1 结果的合法表述

- `verified`：可写“documented post-training candidate relation”，仍需说明 observational、非随机化；
- `partial`：只能写“模型卡记录的 base relation 候选”；
- `unknown`：不得进入 R1 主表。

若所有 edge 都是 partial，R1 结果不得进入论文标题或摘要的“post-training effect”，只能作为候选机制分析。

## 5. F0：真正的 exact-unit attribution

R0 的标签是 complete/instruct，F0 的标签必须是来源 unit。两者不能混为一个结果。

### 5.1 F0-A：115-way observed-unit attribution

在 task-heldout 上训练 115-way observed `model_id` 分类器，标签为 `model_id`，不使用 model_id 字符串、参数量或 family 名作为输入。输入视图固定为：

1. `h_complete`；
2. `h_instruct`；
3. `delta`；
4. `u=[S;A]`；
5. P0 style/metadata/size-length。

报告 macro-F1、balanced accuracy、top-1/top-5、task-macro 和机会水平。未见 model_id 不报告未定义的 macro-F1；model-heldout 只做 open-set score/rejection。

### 5.2 F0-B：官方系列 observed-member attribution

对 CodeLlama、Qwen Coder、DeepSeek Coder 三个官方系列，标签是系列内 member；旋转留出一个 member。这个实验继承 stage-2 的系列定义，但必须使用同一 task split、同一模型容量和同一 P0。主指标是 heldout-member AUROC/AP 与 task-cluster CI。

若 `family_is_confirmed=false`，报告名必须使用 `observed_series/member transfer`，不能写“普遍 family attribution”。

## 6. F2：后训练差异辅助的归因关系

F2 不再从一个 `u_mlp` 分数直接跳到“归因流形”。使用以下最小关系协议：

### 6.1 关系样本

对同一 task 的两个 observed units (i,j)，构造：

\[
q_{ij}=\bigl[u_i;u_j;|u_i-u_j|;u_i\odot u_j\bigr].
\]

标签分两种，必须分开：

- `same_observed_series`：只用于官方系列成员；
- `same_exact_unit`：用于重复输出/同 unit 一致性诊断，不能冒充 family。

严禁把不同 dataset、不同 task 的代码拼成正对；正负对必须同 task、同语言或显式分层。负样本按 model size、length、style 分层匹配，版本在训练前冻结。

### 6.2 关系读出

依次运行：

1. cosine / Euclidean 线性关系基线；
2. logistic on (q_{ij})；
3. 一个 warm-start residual head，相对 P0 只学习关系残差。

暂不加入 SupCon、QDA、低秩、正交和完整 DCAN。每个机制必须有单独的 `method_id`、参数、seed 和停止判断。

### 6.3 通过条件

F2 只有在以下条件全部满足时才进入 H3：

- series/member heldout 至少两个折同向；
- 相对最佳单侧和 P0 的 `delta_mean` 为正；
- task-cluster CI 不覆盖 0；
- size/length/style 匹配控制仍保留增量；
- permutation negative 和错误配对不复现同样增益。

否则结论为“当前数据和表示下未观察到稳定关系增益”，停止增加模型复杂度。

## 7. 新鲜数据构建只做索引，不先生成

本轮先在本机完成 `fresh_same_task_v1` 的任务池、许可证、版本、三层 hash、split 和 generator 支持矩阵。目标是 300–500 个任务、至少三个独立公开 generator/source unit；先切 train/dev/test，再考虑输出生成。AutoDL 只接收必要的 `tasks.jsonl`、`unit_manifest.jsonl`、`pair_index.jsonl`、split/hash/provenance 和必要特征切片。

未得到新 generation 授权前，执行开关保持：

```json
{
  "training_allowed": true,
  "generation_allowed": false,
  "test_read_allowed": false,
  "old_test_reuse": false,
  "original_bundle_verified": false
}
```

## 8. 回传要求

分别回传以下目录，不能把 R0 复核和 R1/F0/F2 合并覆盖：

- `artifacts/r0_protocol_diff_reaudit_2026-10-08/`
- `artifacts/r0_mechanism_controls_2026-10-08/`
- `artifacts/r1_partial_lineage_train_dev_2026-10-08/`
- `artifacts/f0_exact_and_series_attribution_2026-10-08/`
- `artifacts/f2_relation_attribution_2026-10-08/`
- `artifacts/fresh_same_task_index_2026-10-08/`

每个目录必须含 `hypothesis/config/execution_switches/data_role_matrix/metrics/report/commands/logs/git_head/git_status/SHA256SUMS`。预测文件只保留必要的 id、label、score、fold；raw code 不上传。

回传摘要仍按六句话，但增加三项：

1. pair-direction、exact-unit attribution、series/member transfer 是否被分开；
2. S-A 条件性是否通过真实 A-only、S-shuffle 和 S-residual 对照；
3. R1 每条 edge 的证据等级和是否仍为 partial。

## 9. 下一阶段闸门

- R0 复核通过：只说明统计可复现，不能自动升级论文主张；
- R0 机制对照通过：允许把 S-A 交互写成表示机制候选；
- R1 至少一条 edge 达到 verified：允许单独讨论 observational post-training candidate；
- F0/F2 heldout 通过：才允许启动 H3 detection-family joint；
- 任一条件失败：保留负结果并改进数据/provenance，不通过下载更多 backbone 来规避问题。

本轮重点是把“协议差异可读性”推进为可审计的后训练候选和真实来源归因证据，仍然不把相关性写成因果性。
