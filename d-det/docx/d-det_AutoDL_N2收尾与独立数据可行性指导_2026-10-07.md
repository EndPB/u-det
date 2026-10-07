# AutoDL：N2 收尾与独立数据可行性指导

日期：2026-10-07
接续提交：`7a0d51a` → `466d86d`（N2 协议预提交）→ `9936df1`
状态：按用户回传登记；本地尚未取得这些提交的原始文件，服务器 AI 须从仓库原始产物核对数值和协议。

## 1. 本轮真正解决了什么

N0/N1 已把数据卫生与来源支持的问题显式化：原 9,498 行中发现 25 个 norm_lex 跨 split 骨架组，涉及 51 task / 66 行；按预声明规则整组剔除后为 9,432 行，train/dev/test = 6,521/1,467/1,444，exact/ws/lex 跨 split 组均为 0。只有 AuthorBench 的 OpenAI 三 generator 支持 admitted heldout；其余五族仍为单 generator。LLM-CodeGen 的 Meta 三 generator 是另一数据域的附属单元，应独立报告。

N2 在六族齐全的 280 task 候选上运行，test 为 51 task / 404 行；等权集成优于 dev 拟合 fusion，内容视图相对旧 fusion 的提升不稳定。H2/H3 继续暂停。

三条边界必须保留：

- 单 generator 的 family 无法在该数据中区分“family 规律”与“这个 generator 的规律”；不能据此声称已建立跨 generator 家族归因。
- 候选与 D1 全量的样本组成、task 数、split 和基线拟合条件不同，分数下降说明结果对评测设计敏感，尚不能单独归因于“组成贡献”。
- 本轮支持“当前方案未达到进入 H2 的工程门槛”；并没有否定所有 H1、所有编码器或 H2 假设。TF-IDF 内容本来就有可读信号，未超越强 P0 与完全不可读是不同结论。

**下一步只有两项：E0 证据收尾；E1 独立数据可行性表。完成后停止，不自动启动新实验。**

## 2. E0：仅从现有产物关闭证据问题

输出目录建议：`d-det/artifacts/stage_e_evidence_feasibility_2026-10-07/e0/`。使用现有 JSON、配置、日志、逐样本预测和 manifest；不重新加载原始 test 文本，不重新调用模型，不改变超参数。已有 test 预测可用于下面预指定的口径核对，必须登记为 post-hoc audit。

### 2.1 数值和差值对账

用户回传表为：

| 视图 | macro-F1（回传） | 相对 fusion 的差值（回传） | 直接按所列值相减 |
|---|---:|---:|---:|
| tfidf_word | .6429 | +.0148 | +.0165 |
| codet5_small_centered | .6538 | +.0278 | +.0274 |
| fusion_lr | .6264 | 0 | 0 |
| mean_ensemble | .6940 | +.0663 | +.0676 |

这些偏差不能统一用四位小数舍入解释。不得任选一列作为正确值。先检查它们是否来自不同 seed、不同 test 集、bootstrap 均值、macro-F1 定义、有效类别集合或指标配对方式。D1 `.8200−.8388=−.0188` 与历史回传 `−.0191` 也一并对账。

生成 `metric_reconciliation.json`，每行至少保存：view、seed、metric definition、label set、n_samples、n_tasks、ordered example-id hash、split hash、point estimate、comparator、difference、CI method、CI endpoints、source file、verification status。

必须满足：同一 test、同一 seed、同一指标的差值等于两个点估计之差；bootstrap 均值另列，不能替代点差值。原表永久保留，修正采用“原值／修正值／原因”，不静默覆盖。不能解析时标记 unresolved，并阻止该项进入论文表格。

### 2.2 P0 对照与选择时间

核对 `mean_ensemble` 究竟平均什么：成员、各成员输出的类别顺序、概率是否归一化、校准方式、权重、是否某个成员缺少类别、融合层训练 task 数及 dev 数。

对概率等权集成，明确定义：

\[
p_{\mathrm{eq}}(f\mid x)=\frac1M\sum_{m=1}^{M}p_m(f\mid x),\qquad
\hat f(x)=\arg\max_f p_{\mathrm{eq}}(f\mid x).
\]

如果实际上平均 logits、vote 或 family-specific 值，须写出真实公式，不能沿用上式名称。对 dev 拟合融合，记录哪些参数只在 dev 拟合、是否存在双重 dev 选择，以及类别/样本权重。

核对 `466d86d` 是否在读取 N2 test 前已声明等权集成及全部视图。未声明的结果是探索性 P0 修正发现；已声明的结果仍只支持其对应协议。即使 paired CI 不含 0，也不是 H2、跨 generator 泛化或“内容超越最强 P0”的证据。

未来强 P0 候选应包括等权集成和 fusion_lr，在未来数据的 train/dev 上选择或同时预注册报告。不能用当前 test 结果为同一 test 选择新赢家，再称作独立确认。

### 2.3 重新表达 N2 闸门，不重跑实验

分别记录以下结论：

| 问题 | 可接受的证据 | 本轮表达 |
|---|---|---|
| H1 内容可读性 | 内容视图高于预声明的 chance/metadata 控制，含配对 CI | 按已有产物填，不由“未超越 P0”自动判无信号 |
| 普通单样本预测增量 | inductive 内容视图在同协议下超越预声明强 P0、重复方向稳定 | 当前未通过进入方法扩展的门槛 |
| 转导诊断增量 | 使用 test task 内无标签输出中心的独立结果 | centered 单独列，不能替代 inductive 条件 |
| 跨 generator 迁移 | 足够的同族多 generator 支持、严格 heldout 协议 | 现有六族候选不满足普遍性；不启动 H2 |

`word` 的三 seed 差值 −1.36/+2.18/−0.18pt 记录为不稳定，不只挑正 seed。cond2 若仅靠转导视图，标记“原执行器 cond2=true；inductive cond2 不通过/未确立”。这修正的是门槛解释，原配置和原报告保留。

500 次 task bootstrap 的经验频率是重采样频率，不是模型优于对照的贝叶斯后验概率。对 test task 只有 51 个的结果报告较小有效样本量；不能将 404 行当作 404 个独立 task。

### 2.4 数据卫生与集合选择边界

区分 exact/ws 重复与 norm_lex 骨架碰撞。去掉字符串可能把不同常量、不同 IO 规格和不同任务抹成同形；剔除是保守协议，不自动证明原始评测泄漏。

记录 norm_lex 实现 hash、是否词法分析或正则、注释中的引号/字符串中的注释符处理、整组剔除影响的 family/generator/task 数，以及是否保留人类/机器标签比例。原 test 行剔除后的新子集仍属于已经暴露的评测集合。

“六族齐全 task”选择会改变目标总体；公开 280 task 的纳入规则、各 split task 数、每族每 task 的 generator/行数，并说明规则在 test 读取前是否预声明。若审计需要查看 test 文本的新信息，停在 unknown；不要借审计继续根据错误样本调整数据清洗。

E0 交付：`metric_reconciliation.json`、`p0_definition.json`、`gate_reconciliation.json`、`test_exposure_ledger.json`、`evidence_closeout.md`、日志、命令、SHA256SUMS。暴露账本至少记录 D1、N2 test task 集及时间、各自用途；不能把已看过的 task 换一个 split 名称后当作 fresh test。

## 3. E1：独立数据可行性，先做元数据索引

输出目录建议：`.../e1/`。扫描服务器已有 manifest/summary/元数据，生成索引；不复制代码全文、不下载数据或权重、不生成模型输出、不运行训练。找不到字段就标 missing；不能通过模型名猜造 task_id、family lineage 或训练阶段。

### 3.1 先定义论文目标

主任务维持模型来源/家族归因；另表报告 detection 和未知来源拒识。必须声明 family 是哪个可检验概念：vendor group、已知共享基座的 lineage，还是生成器集合。闭源 GPT 版本属于同 vendor 不等于已知共享权重谱系；CodeLlama/Llama2/Llama3 也不能凭名字视作唯一同基座后训练对。H2 的泛化定义必须和标签语义一致。

若论文拟主张后训练差异分离，应另列 base/instruct 的可靠关系、版本与可得性；本轮不把任务中心化当作后训练因果证据。

### 3.2 现有数据的可识别性表

每个来源分别输出：dataset、license/provenance、family definition、family、generator、task namespace、language、可对齐 prompt/task hash、n_tasks、n_rows、seen/heldout role、各 split 支持、exact/ws/lex audit status、是否已被评估、是否可以独立留出。

AuthorBench OpenAI 与 LLM-CodeGen Meta 分别来自不同来源时，直接拼接会使 dataset 与 family 混杂。不得将二者合成一个“完成多族支持”的主赛道；即使均为 C，也未消除 task/domain/prompt/process 混杂。单独作为两个迁移单元可以，不能借此证明所有 family 的统一几何。

给出 task×family×generator 的覆盖矩阵、同任务交集和缺失率；至少输出每个 heldout 折的 train/dev 正支持和 test task 数。真实的 empty/missing 允许成为最终结论，不能为达到门槛重复采样来冒充新的 task 支持。

### 3.3 新独立验证集的最小蓝图（只规划）

作为后续 pilot 的设计目标，可用至少 3 个清晰定义的 family、每族至少 3 个 generator，在相同新任务集合上采样。这个数是本项目的工程设计门槛，不是 H2 可识别性的数学定理，也不保证显著性。

所有 generator 尽量使用相同 task/prompt 模板、语言和生成流程；generation settings、版本、失败重试和缺失输出完整登记。按 task 先划分 train/dev/test；根据任务近重复/来源组建立 split 约束，不能按输出行随机划分。heldout generator 的输出不进入该折 train/dev；test 同时要求未见 task。三族/三 generator 的实际数、预算和支持门槛应在后续协议中冻结。

输出 `pilot_design.json`：

```json
{
  "status": "ready | partial | unavailable",
  "family_semantics": "vendor_or_lineage_explicit",
  "target_families": [],
  "generators_per_family": {},
  "task_source": [],
  "unexposed_task_evidence": [],
  "split_grouping": "task_or_provenance_component",
  "heldout_generator_plan": [],
  "positive_definition": "same_task_same_family_different_generator",
  "hard_negative_definition": "same_task_different_family",
  "compute_and_disk_estimate": {},
  "cost_and_generation_authorization": "not_requested_here",
  "missing_assets": [],
  "training_allowed": false
}
```

估计磁盘只算增量：原始输出、轻量索引、可选一次性特征缓存和必要日志，列清文件数/字节估计依据及可复用位置。资源不够时输出部分可行矩阵；不自行删除历史资产。生成/购买/外部 API 调用仅写估算和缺口，等待用户明确授权后才执行。

pilot task 数不凭经验拍板为“论文足够”。先给每族每 generator 的 task 覆盖、拟比较的最小差值（如 1pt）、效应相关性假设、task bootstrap 或模拟检验计划。旧数据估计只用于规划，最终统计功效不承诺。无法估计时写 power_unknown。

E1 交付：`dataset_feasibility.json`、`coverage_matrix.csv`（UTF-8）、`pilot_design.json`、`feasibility_report.md`、命令、日志、SHA256SUMS。CSV 是审计导出，不需要额外电子表格格式。

## 4. 后续唯一允许的出口

- E0 完成而 E1 partial/unavailable：停止计算，回传精确缺口、可复用资产、预计增量磁盘与预算；不重复 N2，不加损失。
- E1 ready：只交付独立数据的预注册方案；这份指导不授权进入 H2/H3 或自动重读 test。后续执行指导须明确数据冻结、强 P0、inductive 主结果、转导附属结果、seed、CI、test 暴露策略和停止条件。
- E0 差值仍 unresolved：先以原始预测/配置解释 discrepancy，无法解释则将对应数字移出确认性结论，保留记录；不通过重跑掩盖问题。

已有 D1/N2 可用于论文的负结果与稳健性章节，表述为“所测协议未提供稳定内容增量；小 dev 的融合选择与 task 组成可能影响估计，尚需独立验证”。不要写成“无来源信息”“组成造成了全部高分”或“等权集成是新的 H2 方法”。

## 5. 回传模板

1. E0 四处差值对账：原值、原始文件、统一口径、修正值/unknown。
2. mean_ensemble 定义、预声明时间、strong P0 与转导/归纳门槛的分离。
3. 已暴露 test 账本；保守骨架去重与真实泄漏证据的区别。
4. E1 来源支持/同任务覆盖/混杂/许可/新任务可用性表。
5. pilot ready/partial/unavailable、缺口、预计增量磁盘和成本；训练状态必须仍为 false。
6. commit、命令、目录树、SHA256SUMS；不提交权重、代码全文、凭据或冗余缓存。

严格按 E0 → E1 顺序完成即可。该轮是证据闭合与数据路线决策，不是新一轮模型竞赛。
