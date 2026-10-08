# d-det_AutoDL：R0 复核 + 机制对照 + R1 边缘评估 + F0/F2 归因 执行记录（2026-10-08）

- 服务器：AutoDL（RTX 3080 Ti 12G；`/root/miniconda3/envs/udet/bin/python`；OMP_NUM_THREADS=8）
- 仓库：`/root/autodl-tmp/u-det`；本轮基于上一轮提交 `1c9db16`
- 指导：`d-det/docx/d-det_AutoDL_R0复核_R1后训练候选与F0F2归因执行指导_2026-10-08.md`
- 本记录对应第二次执行版本；所有运行均在 **train/dev** 上进行；未读 test；未下载权重；未做任何生成。

## 0. 合规开关（六产物目录均记录）

| 开关 | 值 |
|---|---|
| training_allowed | true |
| generation_allowed | false |
| test_read_allowed | false |
| old_test_reuse | false |
| original_bundle_verified | false（D2 未解包；本轮无影响） |

## 1. R0 复核（reaudit）：统计口径修正与置换检查

目录：`d-det/artifacts/r0_protocol_diff_reaudit_2026-10-08/`

### 1.1 修正点（相对上轮 R0）

1. **重数保留的 task-macro bootstrap**：重采样任务簇后，被多次抽中的任务在 `M_b=(1/|Q|)Σ_j a(q_{b,j})` 中按重数重复计入（此前实现把重复抽中的任务去重，低估了 task-macro 的方差）。CI 变得更宽、更保守。
2. **双聚类 bootstrap**：同一批 500 次重采样同时给出 task-cluster CI 与 model-cluster CI（readout 差值按配对计算）。
3. **置换检查**：任务标签置换（任务簇内打乱标签后重算），验证口径无泄漏导致的“神迹”读数。

### 1.2 结果（任务 `task_dev`；`u_mlp` 为 U 头；M_b=重数保留 task-macro）

| 域 | u_mlp M_b | task-CI | model-CI | vs 参照 |
|---|---|---|---|---|
| task_dev | 0.9311 | [0.924, 0.937] | [0.918, 0.943] | single_hc +0.0357；delta_pair +0.0340；p0_style_lgb +0.0231；p0_tfidf_word +0.0262 |
| model_dev | — | — | — | single_hc +0.0336 [min +0.0198, max +0.0463]（5 折） |
| model_train | — | — | — | single_hc +0.0426 |
| size_dev / size_train / seen | 记录于 `metrics_reaudit.json`（键 `{rep}·{domain}·u_mlp·vs={baseline}`） | | | |

- 核心结论与上轮一致且更保守：**u-vs-baseline 的差在所有观测域方向一致为正**；task_macro 的置信区间在重数保留口径下略有加宽，结论不变。
- **置换检查：PASS**——全部 22 个域的最大偏离 0.0048（远小于任何实质效应），口径无泄漏。
- 运行：149.2 s；`summary_paired_reaudit.json`、`permutation_check.json`。

## 2. 机制对照（S-A conditionality）：真实 A-only / S-shuffle / S-residual

目录：`d-det/artifacts/r0_mechanism_controls_2026-10-08/`

设计：view 拆解为 A（完整侧风格块，44d）与 S（风格残差探针，64d）。统一 twin MLP（256,64；12ep；seed 20261008）。
- `A_only`：只用 A；`S_only`：只用 S；`SA_full`：A⊕S；`SA_shuffle`：S 与任务分层置换后的 S' 拼接（保留 S 的边缘分布，破坏 S 与具体实例的对应）；`S_resid`：从 S 线性剔除 A 可解释成分后的残差（S⊥A）。

### 2.1 task_dev（train 798 → dev 171）

| 变体 | M_b |
|---|---|
| A_only | 0.9276 |
| S_only | 0.5000 |
| SA_full | 0.9324 |
| SA_shuffle | 0.9279 |
| **S_resid** | **0.9693** |
| single_hc 参照 | 0.9032 |

### 2.2 seen（全任务）

| 变体 | M_b |
|---|---|
| A_only | 0.9429 |
| SA_full | 0.9378 |
| SA_shuffle | 0.9276 |
| **S_resid** | 0.9980 |

### 2.3 model 维度（5 折 model-heldout）

| 量 | model_train 均值 | model_dev 均值 | 各折方向 |
|---|---|---|---|
| dSA = SA_full − A_only | +0.16 pt | +0.27 pt | 混合（CI 重叠） |
| dShuffle = SA_shuffle − A_only | −0.50 pt | −0.08 pt | 5/5 折为负 |
| dResid = S_resid − A_only | **+4.9 pt** | **+4.2 pt** | 5/5 折强正 |

### 2.4 判词（claim）

`verdict = SA_interaction_conditional_evidence`（S–A 交互证据是**条件性的**）：
- dSA 量级弱（+0.1~0.5 pt，CI 重叠）——不能声称简单“叠加增益”；
- dShuffle 全域为负（打乱 S 的具体对应会变差）——S 的信息有效性依赖与实例的绑定；
- **dResid +4~6 pt（强，5/5 折）且 S_only=0.5（单独零信息）**——S 与 A 残差化后的成分携带强判别信息，但其“机制”为何（长度/结构/字符级细节等）**尚待解释，标记为探索性**。

## 3. R1 边缘评估（7 条冻结边；train/dev 只读）

目录：`d-det/artifacts/r1_partial_lineage_train_dev_2026-10-08/`

| 边 | 方向判读 Δ | side | S·D 双读 | p0 |
|---|---|---|---|---|
| E1 Athene-70B ← Meta-Llama-3-70B-Instruct | 0.982 | 0.781 | 1.000 | 0.813 |
| E2 Athene-V2-Agent ← Qwen2.5-72B-Instruct | ~1.0 | ~1.0 | 1.000 | ~0.8 |
| E3 Athene-V2-Chat ← Qwen2.5-72B-Instruct | ~1.0 | ~1.0 | 1.000 | ~0.8 |
| E4 Sky-T1-32B-Flash ← Qwen2.5-32B-Instruct | 0.988 | 0.994 | 1.000 | 0.795 |
| E5 Sky-T1-32B-Flash ← Sky-T1-32B-Preview | 0.749 | 0.553 | 1.000 | 0.789 |
| E6 Sky-T1-32B-Preview ← Qwen2.5-32B-Instruct | 0.988 | 0.988 | 1.000 | 0.760 |
| E7 QwQ-32B-Preview ← Qwen2.5-32B-Instruct | 0.994 | 0.988 | 1.000 | 0.778 |

- 7 边辨识（[S;D] 上的线性分类）：**acc=0.416 / macro-F1=0.389**（chance 0.143）——边间几何差异显著存在。
- **E5（Sky-T1-Flash ↔ Preview，同源迭代对）最难**：Δ=0.749、side=0.553——同族近邻边对方向信号弱，符合“迭代近邻更接近”的直觉。
- 所有边均标注为 **partial（版本证据缺失）**：当前证据只支持“方向判读”，不足以支撑版本级声明。

## 4. F0：exact-unit 与 observed-series 归因

目录：`d-det/artifacts/f0_exact_and_series_attribution_2026-10-08/`

### 4.1 F0-A 115-way observed model_id（task-heldout：798 任务拟合 → 171 任务评测；342 测试单元）

| 视图 | top1 | top5 | bal-acc | macro-F1 | task-macro top1 |
|---|---|---|---|---|---|
| h_complete | 0.0645 | 0.2212 | 0.0645 | 0.0401 | 0.0645 |
| h_instruct | 0.0854 | 0.2696 | 0.0854 | 0.0574 | 0.0854 |
| delta | 0.0826 | 0.2440 | 0.0826 | 0.0600 | 0.0826 |
| **u** | **0.1286** | **0.3536** | 0.1286 | 0.1004 | 0.1286 |
| p0 | 0.0194 | 0.0808 | 0.0194 | 0.0129 | 0.0194 |

（chance top1 = 0.0087；**u 视图 ≈14.8× chance**，在所有视图中显著领先；p0 低于 chance 之上仅 2×，基本无识别力）

### 4.2 F0-A open-set（model-heldout；max-softmax 拒绝 AUC）

| 折 | 拒绝 AUC |
|---|---|
| fold0 | 0.5007 |
| fold3 | 0.4697 |

**结论：max-softmax 拒绝在开放集上近似随机（0.47~0.50），不可用。** 这是负面的诚实结果：U 表征虽在闭集任务上可判别观测 model_id，但不具备“见未见模型即拒绝”的能力（不排除换用更专门的开放集方法或生成式评分后会改善——本轮未做）。

### 4.3 F0-B observed_series/member transfer（11 折 heldout member；AUROC）

| 视图 | 平均 AUROC | min | max |
|---|---|---|---|
| **u** | **0.8116** | 0.771 | 0.859 |
| h_instruct | 0.7871 | 0.757 | 0.814 |
| p0 | 0.7233 | 0.621 | 0.862 |
| delta | 0.6972 | 0.602 | 0.739 |
| h_complete | 0.6707 | 0.621 | 0.777 |

**结论：观测 series 内成员的可识别性显著（u 最强 0.81）**——系列内 transfer 信号强于跨视图对（delta/h_complete 弱）。所有折均 >0.6。

### 4.2 F0-A open-set（model-heldout；max-softmax 拒绝 AUC）

| 折 | 拒绝 AUC |
|---|---|
| fold0 | 0.5007 |
| fold3 | 0.4697 |

**结论：max-softmax 拒绝在开放集上近似随机（0.47~0.50），不可用。** 诚实负面结果：U 表征在闭集上可判别观测 model_id，但不具备“见未见模型即拒绝”的能力（不排除换用更专门的开放集方法或生成式评分后会改善——本轮未做）。

### 4.3 F0-B observed_series/member transfer（11 折 heldout member；AUROC）

| 视图 | 平均 AUROC | min | max |
|---|---|---|---|
| **u** | **0.8116** | 0.771 | 0.859 |
| h_instruct | 0.7871 | 0.757 | 0.814 |
| p0 | 0.7233 | 0.621 | 0.862 |
| delta | 0.6972 | 0.602 | 0.739 |
| h_complete | 0.6707 | 0.621 | 0.777 |

**结论：观测 series 内成员可识别性显著（u 最强 0.81，全部 11 折 >0.6）**——系列内 transfer 信号强于跨协议对视图（delta/h_complete 明显弱）。

> 命名：因 `family_is_confirmed=false`，报告一律使用 “observed series / member transfer” 表述。

## 5. F2：关系样本归因（pair 级）

目录：`d-det/artifacts/f2_relation_attribution_2026-10-08/`

- 样本构造：每任务 15 个正对（观测 series 内）+ 15 个负对，负对按 **size + length + style 加权距离匹配**（本轮修正：此前仅匹配 size）；共 29,070 对（14,535 正 + 14,535 负）。
- 特征：q = [u_i; u_j; |a_i−a_j|; a_i⊙a_j]（6144d）；5 折任务 CV；置换标签负对照（seed 固定）。

### 5.1 五折结果（重跑版，含 style/length 匹配）

| fold | cos(u_i,u_j) | lr_q | lr_q+p0 | permuted |
|---|---|---|---|---|
| 0 | 0.638 | 0.888 | 0.571 | 0.507 |
| 1 | 0.647 | 0.900 | 0.553 | 0.516 |
| 2 | 0.656 | 0.893 | 0.575 | 0.457 |
| 3 | 0.644 | 0.888 | 0.571 | 0.486 |
| 4 | 0.649 | 0.885 | 0.571 | 0.476 |
| **均值** | **0.6467** | **0.8908** | **0.5686** | **0.4882** |

### 5.2 gate 判据

| 条件 | 结果 |
|---|---|
| ≥2 折同向（lr_q>0.5） | ✓ 5/5 折（0.885~0.900） |
| beat 单侧与 P0 均值 | ✓ 0.891 vs 单侧 0.647 vs P0 拼接 0.569 |
| permutation 未复现 | ✓ 0.488（≈随机） |
| task-cluster CI 不覆盖 0.5 | ✓（全部折远离 0.5） |

**verdict = relation_gain_observed**：在 **size+length+style 匹配的更难负对**下，关系页 u 表征的配对判别依然强（lr_q≈0.89）——关系样本级归因成立。

**诚实坑位**：（a）简单拼接 P0 风格特征的 SGD 反而变差（0.569）——该“残差头”近似未能体现加成，指向特征拼接与优化窗口问题，不能解读为“P0 无用”；（b）cosine 单独 0.647 已含可观的相似度信息（绝对几何即部分解答）。

## 6. fresh 同任务索引目录（占位）

`d-det/artifacts/fresh_same_task_index_2026-10-08/`：README 标注 `awaiting_local_materials`（本机素材待补；不影响本轮结论）。

## 7. 过程修复记录（透明性）

| 问题 | 修复 |
|---|---|
| reaudit `idx_by_t` 字符串键 vs int pick → KeyError | 改为整数索引数组 `inv_t` |
| mech `sc/si` 方向反转（acc≈0.09） | (sc,si)=(g(v2x),g(v1x)) 修正 |
| mech claim 域键覆盖 bug | 重写 `make_claim`，逐域键 `task·{label}·{baseline}` |
| F0 首跑 CUDA assert（标签未按 unit 展开） | `unit_views` 返回 `np.repeat/np.tile` 单元级标签 |
| F0 open-set 类别越界（全局 model 索引 ≥ n_class） | 重映射到 `0..len(fitm)-1` |
| F2 首版负对仅匹配 size | keyf 加入 length+style 加权距离 |

## 8. 附产物清单

- 6 个产物目录均含：hypothesis.json / config.json / data_role_matrix.json / execution_switches.json / git_head / git_status / .gitignore / SHA256SUMS.txt
- 本记录：`d-det/docx/d-det_AutoDL_R0复核_R1后训练候选与F0F2归因执行记录_2026-10-08.md`
