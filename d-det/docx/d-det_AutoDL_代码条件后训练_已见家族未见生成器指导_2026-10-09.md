# d-det AutoDL 指导：代码条件后训练与已见系列未见生成器

日期：2026-10-09  
执行状态：`awaiting_server_boot`；服务器开机前只准备文件，不执行远程任务。  
主研究依据：[ACL 总研究总结](<D:/data/vscode/u-det/d-det/docx/d-det_ACL总研究总结与可迭代路线_2026-10-06.md>) §49，以及[代码条件后训练设计](<D:/data/vscode/u-det/d-det/docx/d-det_代码条件后训练_公式与论文核对_2026-10-09.md>)。  
参考论文：[UIT-AMMC at SemEval-2026 Task 13](https://aclanthology.org/2026.semeval-1.60/)。

## 0. 本轮要回答的核心问题

本轮只回答一个主问题：

> 在一个已经见过的观测系列内，留出一个未见的生成器/尺寸成员时，加入代码任务条件和代码约束的训练目标，能否在任务宏指标上超过完整强 P0？

当前公开数据的正式标签是 `observed_series` 和 `model_id`。三个系列的 `family_is_confirmed=false`，因此报告中使用“已见观测系列、未见成员/操作性生成器”，不能写成已证实的独立 family 泛化。11 个成员为 CodeLlama 4、Qwen 4、DeepSeek 3；每折留出一个成员，训练折中保留该系列其余成员和其它系列的负样本。

当前主闸门：相对同折完整强 P0，task-macro AUROC 平均提升至少 1.0pt，至少 8/11 折为正，任务簇配对 95% CI 下界大于 0；train-only 内部任务切分方向一致；置换和变换安全审计通过。任一条件失败，标记 `candidate_failed_or_unresolved`，不换 backbone、温度、权重、seed 或折继续追分。

执行开关：

```json
{
  "training_allowed": true,
  "generation_allowed": false,
  "test_read_allowed": false,
  "weights_downloaded": false,
  "code_execution": false,
  "source_status": "server_reconstruction_only",
  "original_bundle_verified": false
}
```

本轮所有实验均为 train/dev。旧 test 行不得进入内存；不得生成新模型输出；不得下载权重。用户另行授权后，才能进入受控生成或 test 单读。

## 1. 先做数据和角色审计

使用现有服务器文件；不要上传全量历史包、权重或预测数组。必要文件只有：

- `records_train_dev.jsonl`：当前 10,659 行、969 个 task、11 个成员；
- `family_series_admission.json`：系列与成员映射；
- BigCodeBench 任务元数据中与 train/dev task 白名单对应的 `task_id`、`instruct_prompt`、`code_prompt`、`entry_point`、`libs`；
- 若未来获准执行测试，再单独提供白名单 task 的 `test` 字段；当前阶段不读取它。

本机已运行：

```text
scripts/prepare_code_conditioned_pilot.py
```

本机预检产物：`d-det/artifacts/code_conditioned_design_2026-10-09/static_preflight/`。预检结果为 10,659/10,659 代码可 AST 解析，10,602/10,659 有顶层同名入口；parse 成功不是 correctness 标签，入口缺失也不是失败。服务器端重跑时必须保存：输入哈希、行序哈希、task split、成员映射、代码 SHA-256 和静态字段哈希。

审计必须确认：

1. model-task 无重复；task 不跨 train/dev；
2. heldout member 的训练行数为 0；
3. 每个 heldout 系列在训练折仍有至少两个 seen members；
4. train/dev 的任务集合由预注册 split 决定，不能按模型分数挑任务；
5. prompt、canonical solution、test、模型 ID 和 fold ID 不进入静态作者特征；
6. `family_is_confirmed=false`、checkpoint 未逐字节核验等边界写入每份报告。

## 2. 统一数据表示

对每一条 `(t,y,m,s)`：题目条件 t、代码 y、成员 m、观测系列 s，构造：

\[
z(t,y)=[h_t;h_y;h_t\odot h_y;|h_t-h_y|;\psi(t,y)].
\]

`h_t` 为固定 CodeT5-base/small 的题面或接口编码，`h_y` 为代码编码；长代码的截断、分块池化和 tokenizer 必须在配置中固定。`ψ` 只含代码条件可解释的静态量：AST 节点/调用/分支/返回数、导入根、入口存在、题目声明库与导入根的重叠代理、长度和安全解析状态。`ψ` 不是 correctness，也不是后训练奖励。

所有标准化参数、投影、分类头、阈值只在该折 train 拟合。任务采样先按 task，再在 task 内按 series/member 均衡；不能让 4 个成员的系列获得 3 个成员系列的四倍权重。

## 3. 代码约束的数学定义

任务条件写作 `t=(p,a,U)`，其中 p 是题面，a 是函数/API/库约束，U 是离线测试集合。部署评分默认只使用 p、a 和代码 y；U 只在以后获得授权时用于监督。

定义约束向量：

\[
v(t,y)=(v_{parse},v_{interface},v_{tests},v_{resource},v_{safety}),
\qquad r(t,y)=w^\top v(t,y)+b(t,y).
\]

`v_tests` 只有实际执行测试后才可填写；静态 parse、入口和 import 只能作为代理，必须带 `proxy` 后缀。缺失值使用 mask，不能补零。

若固定参考策略 `π_ref`，代码条件后训练的 KL 正则目标为：

\[
\max_{\pi}\;\mathbb E_{y\sim\pi(\cdot|t)}[r(t,y)]
-\beta\,KL(\pi(\cdot|t)\|\pi_{ref}(\cdot|t)).
\]

当 \(\beta>0\)、参考策略对候选输出给出正概率且指数矩有限时，拉格朗日驻点给出：

\[
\pi^*(y|t)=\frac{\pi_{ref}(y|t)e^{r(t,y)/\beta}}{Z(t)},
\qquad
Z(t)=\sum_y\pi_{ref}(y|t)e^{r(t,y)/\beta}.
\]

因此：

\[
\rho(t,y)=\log\frac{\pi^*(y|t)}{\pi_{ref}(y|t)}
=\frac{r(t,y)}{\beta}-\log Z(t).
\]

同一任务两份代码的差分消去 `Z(t)`：

\[
\rho(t,y_1)-\rho(t,y_2)=\frac{r(t,y_1)-r(t,y_2)}{\beta}.
\]

这个推导是可检验模型的结果，不能直接当作真实 SFT/RLHF 的因果事实。不同参考模型本身也会产生来源信号；正确性提升不能自动等同于系列指纹。

只有存在真实参考模型的 token log-prob，并且偏好 pair 由可执行约束产生时，才允许使用 code-DPO：

\[
\mathcal L_{code-DPO}=-\log\sigma\left(\beta\left[
\log\frac{\pi_\theta(y_w|t)}{\pi_{ref}(y_w|t)}-
\log\frac{\pi_\theta(y_l|t)}{\pi_{ref}(y_l|t)}\right]\right).
\]

完整 response token log-prob、相同 tokenizer、相同 response mask、EOS 和截断规则必须一致。当前公开输出没有参考策略 log-prob，因此不得把 CodeT5 分数冒充 DPO 密度比；先执行第 4 节的代理轨道。

## 4. 本轮实验矩阵

### C0：强 P0 复现

逐折重建四个 fit-only 组件：CodeT5 semantic linear、char TF-IDF、word TF-IDF、style/meta linear，并按冻结规则等权 z-score。输出 pooled、task-macro、member-macro、500 次 task-cluster paired CI。与本机强 P0 的任一点值差异超过 `1e-3`，立即停在 C0 修复。

### C1：题面条件化代码归因

输入 `[h_t;h_y;h_t⊙h_y;|h_t-h_y|;ψ]`，训练小型 MLP/线性头预测 heldout 系列。对照为：代码-only、题面-only、`[h_t;h_y]`、完整交互、静态 ψ-only。所有对照共用折、标准化、seed 和容量。C1 回答“任务条件是否改变任务内排序”，不能称后训练因果。

### C2：代码约束代理头

在没有测试执行授权时，只训练 `static_proxy_conditioning`：联合 family loss 与带 mask 的静态代理辅助 loss：

\[
\mathcal L=\mathcal L_{series}+\lambda_v\mathcal L_{proxy}
+\lambda_{inv}\mathcal L_{inv}.
\]

`parse_ok=1` 在当前数据零方差，不得作为训练标签；入口、AST、调用和库重叠仅做描述或辅助特征。报告各代理的方差、缺失率和与长度的相关系数，防止把规模重新命名为代码约束。

### C3：论文增强的一致性训练

参考 UIT-AMMC 原文 §4.2 的概率：comment stripping `p=.2`、whitespace perturbation `p=.3`、identifier renaming `p=.4`；原文还要求把增强样本与原始样本混合。论文的任务是 Human/AI 检测，当前借用的是增强思想，不借用其 F1 或因果解释。[论文原文](https://aclanthology.org/2026.semeval-1.60/)

只在语义安全集合 `G_valid(t,y)` 中使用变换，并将原样本与变换样本同折配对：

\[
\mathcal L_{inv}=\mathbb E_{g\in G_{valid}}
JS\big(p_\theta(\cdot|t,y),p_\theta(\cdot|t,g(y))\big).
\]

代码安全边界：

- 注释：只删除明确 comment token；Python docstring、编码声明、pragma、工具指令默认保留；
- 空白：只改 token 间安全空行/尾空格，保留 INDENT/DEDENT 和字符串内容，变换后 AST 必须一致；
- 标识符：必须按 AST 作用域改名；保留入口、公开函数名、关键字参数、导入别名、属性和魔法方法；遇 `eval/exec/getattr/locals/globals` 或反射敏感路径跳过；
- 任何变换若改变 AST、接口、静态约束或已授权测试结果，剔除并记录拒绝原因；不能用全局正则替代绑定安全改名。

必须有 `transform_acceptance.jsonl`：原代码哈希、变换代码哈希、变换类型、随机 seed、AST hash、接口 hash、拒绝原因。C3 的关键比较是原始/变换配对分数是否保持归因，不能只报告增强后的总体 AUROC。

### C4：代码正确性监督的条件实验（等待单独授权）

只有用户明确授权执行测试后，才读取白名单 `test` 字段并在隔离沙箱运行。每个任务报告通过率、超时、异常、资源和拒绝数；禁止执行未白名单代码。使用同题通过/失败响应构造 preference pair，执行 code-DPO 或约束辅助训练。

C4 的训练对照固定为：同一父 checkpoint、同一任务、同一响应池、同一预算；变化只允许是 correctness preference。另设 format-preference control 和 SFT control。至少 3 个独立 adapter seed；seed 不是独立 generator。记录实际 KL、训练 token、优化步、长度、通过率和拒绝率。

## 5. 评测与报告

每个 C 目录必须有 `hypothesis.json`、`config.json`、`data_role_matrix.json`、`metrics.json`、`report.md`、逐折分数、`commands.txt`、完整日志、`git_head.txt`、`git_status.txt`、`SHA256SUMS.txt`。每个结果同时报告：

- row pooled AUROC；
- task-macro AUROC；
- heldout member-macro AUROC；
- paired task-cluster delta_mean 和 CI；
- 每折结果、正负成员数、任务数和样本哈希；
- C0、代码-only、题面-only、长度/风格控制和置换 null；
- 原始/变换配对的一致率、AST/接口保持率和拒绝率。

所有指标使用同一方向、同一 task-cluster bootstrap 采样索引；不能把 500 次重采样频率称后验概率。不得只汇报 pooled AUROC。

## 6. 停止规则和边界

以下任一情况都停止该候选：

1. 相对 C0 的 task-macro 平均增量小于 1pt；
2. 少于 8/11 折为正；
3. paired CI 下界不大于 0；
4. 增量只在 pooled、单系列、单尺寸桶或单 seed 出现；
5. 变换拒绝率、AST 破坏率或接口变化率不为 0；
6. 增量在代码-only 或长度/style 控制后消失；
7. 训练数据中出现 heldout member、旧 test、目标标签筛选或输出顺序泄漏。

若 C1–C3 均失败，归档为 `code_conditioned_increment_not_observed`，转向新的同题 Human/AI 任务集或受控后训练数据构建；不得继续堆叠 encoder、decoder、参数量、温度或融合权重。若 C4 获得授权后失败，只能报告“代码正确性偏好没有可识别系列增量”，不得改写为后训练因果成功。

## 7. 服务器回传模板

回传首行写：

```text
本轮：代码条件后训练 / 已见观测系列未见成员 / train-dev only
test_read=false; generation=false; weights_downloaded=false; code_execution=false
source_status=server_reconstruction_only; original_bundle_verified=false
```

然后按七项回传：执行开关、数据和角色审计、C0、C1–C3、变换安全、闸门判定、产物与哈希。C4 若未获授权必须写 `not_executed`，不能生成空的执行结果文件。

## 8. 2968ff9 后的唯一补充动作：C0 逐点对账

服务器 C0 的 `.9329/.9301` 与本机旧近似值 `.9344/.9324` 差异超过 `1e-3`。这不是可直接接受的浮点误差。下一轮只允许做一次有限对账：

1. 接收 `d-det/artifacts/code_conditioned_p0_reference_2026-10-09/`；其中每折 `p0_scores.npz` 是本机已保存的 fit-only 逐行分数、组件分数、train 分数和 task 顺序，约 4.2 MB，不含代码、权重或 test；
2. 按 `reference_manifest.json` 逐折核对 heldout member、`y/task/train_task`、行数和哈希；
3. 明确统一 semantic 输入块。服务器当前只用 `h_y`；本机旧强 P0 曾有 embedding、style、meta、size 拼接路径，名称相同不代表规格相同；
4. 明确统一 lexical 规格：词表、IDF、`min_df`、ngram、solver/SGD seed 和训练范围必须 fold-train-only；
5. 输出 `p0_alignment.json` 和逐折最大绝对分数差。只有所有折的逐行分数与指标达到预注册容差，C0 才标记 `aligned`。

在 C0 标记 `aligned` 之前，C1–C3 结果只能写为“相对服务器当前 C0 的开发诊断”，不能写成相对本机完整强 P0 的确认性负结果。对账完成后若候选仍低于统一 P0，正式状态才是 `code_conditioned_increment_not_observed`；若基线规格改变了比较，必须按统一 C0 重新跑一次 C1–C3，不能把旧差值直接沿用。


## 9. edf3813 后的 provenance 裁定

`edf3813` 的字符分数块审计发现了一个 7-cycle 和不同的分区。这是重要的结构异常信号，但它是从连续分数相似性反推的，不能单独证明历史训练使用了另一张 series map。

本机已经核对注册命令与源码：`attribution_family_member_ho.py` 把当前 `family_series_admission.json` 传入每个 heldout 折；`attribution_family_metric.py` 从该文件计算 heldout series；`audit_public_full_p0_local.py` 用同一规则构造 eval 行。按当前 admission 重建后，11 个历史 `p0_scores.npz` 的 `y` 和 task 顺序全部逐位一致。因此当前官方 map 由命令、源码、标签和 task 顺序共同支持；7-cycle 暂记为 `score_block_alignment_hypothesis`。

C0 仍为 `not_aligned`，原因是连续分数没有逐点达到容差，旧产物也没有记录每折 effective map hash、fit/eval member IDs 和完整 semantic/lexical 规格。旧 P0 包降级为 `map_consistent_but_spec_and_per_fold_provenance_incomplete`，不能作为精确对齐参考。不要重命名目录、改标签或再次从连续分数猜 map。

下一轮若服务器重新开机，只允许做一次 train/dev fresh C0 重建。每折必须在拟合前写入并哈希：`effective_series_map`、`heldout_member`、`heldout_series`、`fit_member_ids`、`eval_member_ids`、train/eval 行序与 task 哈希、组件规格和代码提交号。test、生成、权重下载、代码执行均保持关闭。只有 fresh C0 对齐，才允许重新跑 C1–C3；否则不增加 backbone、参数量、温度、融合权重或新的解释。

当前 C1–C3 结果的写法限定为“服务器内部开发诊断”；共同成员块上的结果只能作为待核实对照，不得写成相对历史本机强 P0 的确认性比较。



## 10. fresh C0 规格修正

本机审计发现 `91ea14f` 的 `cc_fresh_c0.py` 在折循环之前执行全体 train/dev 文本的 `fit_transform(texts)`，再按折切片。这与 `spec_sheet.json` 声明的 fold-fit-only vocabulary/IDF 不一致，裁定为 `spec_violation_global_lexical_fit`。与旧 C0 逐位一致不等于协议正确。

下一轮只允许修正 C0：将 char 和 word vectorizer 的 fit 移入每个 fold，只使用该折 `fit_rows` 的文本；eval 行只调用 transform。每折 manifest 追加 vectorizer 参数、fit-row hash、词表大小和 IDF 摘要。重新生成 score digest、metrics 和审计报告。test、生成、权重下载、代码执行继续关闭。

在修正版 C0 通过 lexical fit-scope 审计并满足 `row_score_max_abs ≤ 1e-3`、`metric_abs ≤ 1e-3` 后，才允许考虑 C1–C3。此前所有 C1–C3 数字继续标为服务器内部开发诊断。

## 11. 修正版 fresh C0 回传后的执行闸门（2026-10-10）

`b57b8be` 的修正版 fresh C0 已解决 §10 的 global lexical fit：每折 char/word TF-IDF 只在 `fit_rows` 拟合，eval 只 transform；11 折 manifest 必须保留 fit-row/text hash、vocabulary/IDF attestation 和 pre/post manifest hash。修正版 C0 的协议审计通过，报告数字只作为修正版 train/dev C0 结果保存。

执行端仍不得把 C0 标为 `aligned`。唯一剩余闸门是本机按 `spec_sheet.json` 完成跨侧逐点对账：每折 fused 和四组件 score digest，以及 pooled/row/task 指标，必须同时满足 `row_score_max_abs ≤ 1e-3` 与 `metric_abs ≤ 1e-3`。服务器报告中的 execution head 为 `be8df507…`，交付提交为 `b57b8be…`，执行时工作树曾有未提交文件；应在对账前把修正版脚本 SHA 写入 manifest/metrics，或提供等价的干净来源证明。

在闸门闭合之前：

- `training_allowed` 仅保持现有 train/dev 结果的可追溯性，不启动新训练；
- `test_read_allowed=false`、`generation_allowed=false`、`weights_downloaded=false`、`code_execution=false`；
- C1–C3 不重跑、不升级为统一强 P0 的确认性负结果；
- 不增加 encoder/decoder、参数量、温度、融合权重或新的变换；
- 若本机重算受环境版本或特征缓存限制，只交换逐折 score digest/必要 feature attestation，不交换 test、权重或无关数据。

对账完成后只有两条合法路径：C0 `aligned` 且候选需要统一规格下重跑，或对账失败并继续保留 `cross_side_alignment_pending`。不得用旧 global-fit 结果、历史折猜测或近似指标提前关闭闸门。

## 12. 本机复现后的新增阻断：feature member order 必须先规范化（2026-10-10）

本机已在 AutoDL 关机期间完成修正版 C0 的 train/dev 复现准备与运行。对账没有通过，原因不是 lexical fit，而是 feature bundle 的成员顺序：服务器 `features_manifest.json` 使用 CodeLlama → DeepSeek → Qwen，admission map 与 `cc_common.fold_setup()` 使用 CodeLlama → Qwen → DeepSeek。`member_idx` 若按前者写入、按后者解释，会把 Qwen/DeepSeek 的特征行循环错配到标签。

因此下一次服务器执行前必须增加以下断言：

```text
features.members_order == admission.members_order
member_idx[i] == admission.members_order.index(rows[i].model_id)
row_mapping_sha256 = sha256(model_id + "|" + task_id + "|" + split + "|" + solution_sha256)
```

该断言与哈希必须在 bundle 写入前落盘；失败时停止，不进入任何折拟合。修正后只运行一次 train/dev corrected C0，补写 `members_order_sha256`、逐行 mapping hash 和脚本 SHA。test、生成、权重下载、代码执行继续关闭。

本机结果只能作为结构诊断：fit-text hash 0/11 对齐、词表同时对齐 4/11、fused digest 0/11；C0 状态为 `blocked_feature_member_order`。在该问题修复并完成跨侧 `row_score_max_abs ≤ 1e-3`、`metric_abs ≤ 1e-3` 前，不得重跑 C1–C3，也不得增加 backbone 或参数量。
