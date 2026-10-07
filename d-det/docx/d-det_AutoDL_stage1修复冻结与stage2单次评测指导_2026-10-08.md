# AutoDL：stage-1 修复冻结 → stage-2 单次评测

日期：2026-10-08；审阅基准：`582df7a1dec6edccdf451b1bfa83b8d2ff42676a`。

本指导接续 train/dev 实验，允许完成下述修复和冻结；全部门槛通过后，才具备申请一次性 test 评分的条件。test 仍需单独授权，原件也不因本指导而必须补传。任何门槛失败时只停止依赖该门槛的评测，保存失败证据，不能以反复调参达标。

## 1. 执行范围和来源

- `source_status=server_reconstruction_only`；`original_bundle_verified=false`；`claims_of_byte_identity=forbidden`。这三个字段继续保留，16 件原件不再是执行依赖。
- 允许 stage-1b 在既定 train/dev 上恢复模型对象、修正统计实现、导出可验证评分包；通过 §3 后可提交 stage-2 test 单读授权申请。
- 新生成、下载权重、读旧 LCv2/AuthorBench/D1/N2 test 均禁止。
- 任务划分、11 个 heldout、两个既定负集、五个 P0 成员、C 网格、3 seeds、500 次 bootstrap 都保持原实验定义；不增添特征、网格、epoch 预算或新模型。
- stage-1 原报告、预测、代码和 hashes 保留。新增文件分别写入 `variant_transfer_stage1b_2026-10-08/`、`variant_transfer_stage2_2026-10-08/`，不覆盖 stage-1。

## 2. 本次代码审阅的发现与裁定

审阅对象：`scripts/variant_transfer_stage1_execute.py`（582df7a）；本机未运行该模型、未读代码样本或 test 预测。

### 2.1 模型对象未在脚本中保存

`lr_select_auroc`、`sgd_seeds`、`lgb_fit`、`fusion_lr` 仅返回概率或 C；TF-IDF 词表、StandardScaler、分类器、SGD 最佳 epoch 状态及融合头没有序列化。已有 dev 预测和特征缓存不能独立给新样本评分。

先检查服务器是否有独立保存的对象；存在则直接校验后导出。不存在则允许在原环境、原顺序、原 seed 和原预算上确定性重建，目的是恢复原实验对象，不是寻找新高分：

1. 保存每折词表/IDF、scaler、各 SGD seed 原最佳 epoch 系数、LR、LGBM、融合头和 R2 所需统计量。
2. LR 使用原 `C_selected`；SGD 原最佳 epoch 未记录时，只允许重放原 5 epoch、原 best-dev 规则，记录最佳 epoch 并核对原预测。不得增加候选或重复随机尝试。
3. 拟合继续仅使用 train。不得把 dev 合并进 train，也不得利用 heldout 的 train/dev 输出。
4. 从保存包重新加载并预测原 dev；用 `(fold, arm, readout, unit, task)` 对齐原 csv.gz。六位小数预测的绝对误差 ≤ `5.1e-7`；若确有序列化浮点边界差异，逐项给出解释，不能凭肉眼近似放行。重新计算的 AUROC/AP 点估计须与未舍入原 metrics 在 `1e-10` 内一致（task-macro 点估计也应保持）。失败就停在 stage-1b，不读 test。
5. 修正 `P0_equal` 的 train 返回值为 `Ptr5.mean(1)`，原 dev 返回值继续 `Pdv5.mean(1)`；这个修正不得改变原 dev 输出。

模型评分包及 train/dev 缓存保存在服务器本地并 gitignore，Git 只提交小型清单、对象 hashes、参数/版本、指标和报告。不要求把权重或大缓存推送 Git。

### 2.2 R2 cosine 中心退化

当前实现对正集 fit StandardScaler，然后在同一变换下取正集平均中心。设正集均值为 \(\mu_+\)、坐标尺度为 \(s\)，则

\[
 z_i=(x_i-\mu_+)/s,\quad
 c=\frac1{n_+}\sum_{i:y_i=1}z_i=0.
\]

因此 `cos(z,c)` 的方向不存在；代码的 `+1e-12` 只把数值残差变成分数，不能赋予它系列方向含义。

- 原 R2 cosine 标记 `invalid_zero_center`，不进入 stage-2，不把近 .5 解释成“无系列信号”。
- 原 Euclidean `-||z-c||` 可保留，名称明确为 **positive-standardized radial score**，只描述正集中心距离。它不是同一个退化问题。
- 本轮不引入新的未中心化余弦、白化、回归或对比损失；它们需以后单独注册。
- 完整 metrics 显示 base Euclidean 在 22 臂的 dev AUROC 范围为约 `.4486–.6393`，回传中的 `.536–.65` 不是全量范围；改展示表并保留原表。

### 2.3 task-macro bootstrap 丢失抽样重数

原代码抽 task 后复制对应行，但重新按原 task_id 求宏平均，会把重复抽中的 task 合并。整体 AUROC/AP 仍保留行重数，task-macro 的 CI 必须修正。

设每 task 的 AUROC 为 \(a_t\)，一次抽样 task 序列为 \(t_1^*,\ldots,t_T^*\)，正确统计量是

\[
 M^*=T^{-1}\sum_{j=1}^T a_{t_j^*},
\]

不能改成“抽中不同 task 的去重平均”。用预存 dev 预测修正 CI即可，不重新拟合。数值验证至少包含 `[a_1,a_1,a_2]` 被抽中时应为 `(2*a_1+a_2)/3`，并与保存独立 draw-id 的实现对照。整体 AUROC/AP 继续按每个抽中 task 的全部行计算。

### 2.4 长度加权未实现

amendment 注册了长度五分位加权，但脚本只实现了负成员尺寸选择，没有使用该样本权重。

- 已有 `size_matched` 结果只命名为 **member-size-matched / length-unweighted**，原 key 保留，并新增 `length_weighting_applied=false`。
- 长度加权实验记为 `registered_but_not_executed`，本轮不临时追加到已拟合主实验，不伪称已控制全部长度效应。
- 因为该公式已在 train/dev 执行前冻结，stage-1b 必须二选一并在 test 前写死：A）按原公式补跑一个 train/dev 敏感性臂，使用同一 seed/C/模型预算；或 B）生成带时间和理由的 prereg amendment，把该臂明确标成 deferred/omitted。不能看完 test 后决定是否补跑，也不能通过更换主结果掩盖本轮偏差。

### 2.5 控制特征和数据读取口径

`metadata_only` 实际为代码长度、行数、空行、缩进/布局等代码派生量，没有直接使用 series/model_id 作特征。显示名改为 `code_layout_control`，保留旧 key 以对账。`size_length_only` 包含外部已知参数规模，称 `oracle_size_length_control`；它不是未知来源单样本的可部署读出。

所以“内容高于规模/长度控制”仅说明优于这一个有限控制器；不能证明排除了规模、布局、清洗和提示风格等全部混杂。必须做同样本 paired CI 后再写比较结果。

原 loader 在 split 排除前 `json.loads(line)`，test 代码字符串会短暂进入字典；可依据代码称“未进入训练/特征/评分样本”，不能再称“test 正文从未进入内存”。在暴露账本区分 `raw_scan/transient_parse`、`feature_use`、`human_inspection` 和 `metric_evaluation`；已有整文件哈希/流式扫描同样登记。若发现 stage-1 实际用 test 计算特征或指标，停止新评测并报告。

不改变原 stylometry 实现来追高分；如发现缩进统计的旧实现问题，登记后保留可复现原特征，留到独立后续协议处理。

## 3. stage-1b 自动检查门槛

test 评分前生成 `freeze_validation.json`，全部通过才能继续：

1. stage-1 文件 hashes 和两份配置（`6aba9c4d…`、`4d0d1631…`）完整对账；源数据 records hash 为 `d786667a72f3f3864ea38115fd6ccfe8ac393141146d522673668f3730523210`。
2. task split hashes 与 e38715d 一致；11 折训练和 dev 均排除该折 heldout；没有 test 用于 fit/选择的证据。
3. 22 个评分包可加载；全部 R1 原 dev 概率和点指标通过 §2.1 的复现核对。记录对象 hash、scikit-learn/LightGBM/torch/transformers/numpy 版本和代码 commit。
4. 统计 multiplicity 数值验证通过；R2 cosine 被排除；长度加权偏差显式登记。
5. stage-2 评分代码静态审阅通过：只能 `transform/predict`，不调用 `fit/partial_fit` 或调参；评分失败不得触发自动重训。
6. 测试 manifest、允许输出字段、指标和 readout 清单均在读取前提交冻结；不能以 test 表现决定只报告某些折。

此处不要求16件原始 followup 资产，不要求用户再确认一次。对象恢复失败是真正的技术阻塞；原件缺失继续是 provenance 限制。

## 4. stage-2 test 单次评分范围

### 4.1 数据和时点

- 数据限定为当前完整 records 中 **11 个指定成员 × 171 个 TEST task = 1,881 个唯一 unit-task** 的 `full/instruct` 输出，不读其他项目测试数据。
- 同一批 1,881 行一次编码、共用冻结特征，供全部既定模型评分；不能每折独立重读并调试评测。
- 时间不填写虚构预定值。读取前写入 `test_access_ledger.json` 的实际 `started_at_utc`、本地时区、数据 hash、split hash、评分代码 commit、评分包清单 hash，使用当前系统 UTC；记录结束时 `finished_at_utc`、样本数、特征/预测 hash。
- 若哈希不一致、输出或模型包缺失，止于读取前；若中断后需恢复，仅允许复用已冻结特征或存档预测，登记重试，不产生新训练方案。
- 预测只保存在独立 stage-2 目录；test 读后关闭 `test_read_allowed`。之后可对保存分数重算预声明统计，不能再选模型。

### 4.2 正负域

每折主迁移表 **positive 仅为该 heldout variant**（171 行）；negative 按原 size_mix 或已冻结 member-size-matched 负集，每负成员171行。不能把目标系列已见成员混入 positive 来抬高迁移分数。

另附一个预声明的 seen-reference：同一171个test task上，目标系列已见成员作 positive、同样负成员作 negative。它使用同一评分包、同一编码批次、不再拟合。分别报告 `seen_test`、`heldout_test`、`heldout_test−seen_test` 的 paired task CI；这是在固定任务下的迁移损失参照，不是证明纯尺寸因果。

主结果仍为 size_mix；member-size-matched 为敏感性分析。匹配负集是基于已见正成员而选的，不一定与 heldout size 匹配；特别是70B等缺少同规模负成员的折须报告 support_gap，不能声称已严格控制测试尺度。

### 4.3 读出、统计和允许字段

全部10个既定R1读出都评分，双P0分别报告，不选 test 最佳者作主方法。R2只评分已恢复的base/small radial score；退化cosine记无效，不新加估计器。

允许输出：`fold/arm/readout/unit_id/task_id/split/label/score`、input code hash、预测/特征/model hashes；控制表可含尺寸、代码字符/行数、冻结tokenizer token数、既定hard-task标志及数据版本。不要导出代码正文、题面、测试用例或凭据；unit/task元数据只用于对账和分层，不能输入来源分类器。预测存足够精度（至少float64 round-trip），保留3个SGD seed单独分数用于审计，不在test选seed。

每折报告 pooled AUROC/AP、task-macro AUROC、171 task cluster的500次bootstrap CI（seed20261008）。AP同时报告正类率：size_mix不同系列/匹配版负集规模改变，AP不能脱离正类率直接比较。

比较P0-fusion/P0-equal、P0与两个控制器，均用同一fold/arm同一行索引与同一task抽样，分别给点差和bootstrap delta_mean/CI。arm差异是不同负域下的敏感性，不冒充同样本读出对比。

系列汇总对heldout variant等权；总汇总先对系列等权，避免4/4/3成员数决定权重。共用171个test task，汇总bootstrap每次使用同一个task抽样序列贯穿所有折/臂，不能把22次实验当作22套独立test。CI反映固定这些模型下的任务变异，不推出新家族总体结论。不挑显著折，不将bootstrap频率写成后验概率。

大小/长度/难度分层表只用于解释；单层若缺正或负标签则写`not_estimable`，不能强行给AUROC。分层边界只用train或固定元数据，不能按test效果选择。

## 5. 交付与完成条件

一次连续执行 stage-1b → stage-2 → 证据收尾：

- stage-1b：`implementation_deviation.json`、`freeze_validation.json`、对象hash清单、原dev重放对账、修正task-macro CI、命令/日志/SHA256SUMS、实现修正commit。
- stage-2：读取前配置提交、`test_access_ledger.json`、行索引/标签支持、预测、所有既定读出指标、paired差值、seen-reference、失败/无效项、源状态、日志/SHA256SUMS。
- 报告中严格分开 dev 选择内估计、test heldout迁移、实现偏差和provenance限制；补充stage-1 R2完整范围纠正。
- 保存小型预测和证据进Git；模型对象、向量缓存留服务器不强行推送，登记恢复方式和大小。
- 全部结束后关闭 test 开关，推送报告和代码，回传commit与六项摘要：冻结复现、偏差处理、test读取、逐系列迁移、控制差值、论文允许结论。

即使所有系列test分数很高，也只能写“固定公开BigCodeBench协议下的官方模型系列内尺寸变体迁移”，且保留模板/清洗/污染未知等限制。不能把本轮自动升级成H2方法胜利、unseen-independent-family或后训练因果。若heldout明显失效，完整归档迁移失败，不重开该test调损失。
