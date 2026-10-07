# d-det AutoDL E0/E1 完成后 E2 独立数据预注册与干跑指导

日期：2026-10-07
服务器状态：E0/E1 已在提交 `881f300` 完成，工作树干净
当前结论：E1 = `partial`；`training_allowed=false`，新 generator 生成尚未授权、尚未执行

## 0. 这一步要解决的问题

E0 已关闭统计口径、P0 定义和 test 暴露账本；E1 已证明当前来源表不足以形成普遍的多族三 generator 主赛道。下一阶段不是重新调模型，而是先把**独立数据能否支持 H2 的条件**写成可检查的预注册方案，并在服务器上完成不调用外部生成器的干跑。

目标问题只有一个：

> 在一个未暴露的新 task 集上，是否能为一个语义明确的 family/source unit 提供至少三个可靠 generator，并在 task-heldout、generator-heldout 和同任务配对协议下产生可解释的 H1/H2 统计量？

本指导不授权外部 API、付费模型、云端生成或训练。它只授权读取现有 metadata、生成候选任务清单、碰撞审计、成本/磁盘估计和协议文件。

## 1. E2a：先冻结目标 source unit，不把不同数据集拼成主赛道

从 E1 的 `dataset_feasibility.json` 和 `coverage_matrix.csv` 选择一个 source unit，写入 `e2_preregistration.json`。选择规则：

1. family 语义必须明确，是 vendor group、可证实 lineage，还是生成器集合；三者不能混写；
2. 现有 generator、缺失的第三 generator、任务来源和语言必须逐项登记；
3. 现有三 generator 的 unit（如 E1 确认的 OpenAI/Meta）不要因为名称相近而和另一个 dataset 合并；若选择已有三 generator 的 unit，新任务集的作用是独立复核，不是制造第四个 generator；
4. Google/Mistral 目前各缺一个 generator 时，只有在 generator 版本/权限/输出协议真实可用的情况下才列为候选；不能用模型名推断 lineage；
5. 新 task 必须来自未在 P0、D1、N2 使用过的 task/provenance 命名空间，并且能够在 generation 前生成唯一的 `task_source_hash`。

`e2_preregistration.json` 至少包含：

```json
{
  "status": "draft | ready | blocked",
  "source_unit": "dataset_and_family_explicit",
  "family_semantics": "vendor_group | lineage | generator_set",
  "target_family": "",
  "existing_generators": [],
  "target_third_generator": "",
  "task_source": [],
  "language": "C",
  "task_namespace": "",
  "unexposed_task_rule": "",
  "positive_definition": "same task, same family, different generator",
  "negative_definition": "same task, different family, or pre-registered cross-family control",
  "split_rule": "task/provenance group before generation",
  "heldout_generator_rule": "test generator absent from train/dev",
  "generation_allowed": false,
  "training_allowed": false,
  "external_cost_authorized": false,
  "missing_fields": [],
  "blockers": []
}
```

如果现有 E1 表不能确定一个合法 target，状态必须为 `blocked`，回传缺口；不要为了让 JSON 变成 `ready` 而猜补字段。

## 2. E2b：新 task 清单和碰撞干跑

建立独立目录：

```bash
cd /root/autodl-tmp/u-det
mkdir -p d-det/artifacts/stage_e2_independent_pilot_2026-10-07/{prereg,task_dry_run,audit,logs}
git rev-parse HEAD > d-det/artifacts/stage_e2_independent_pilot_2026-10-07/git_head.txt
git status --short --branch > d-det/artifacts/stage_e2_independent_pilot_2026-10-07/git_status.txt
```

读取已有 manifest、task/source metadata 和 E0 test 暴露账本，只生成候选任务索引，不读入旧 test 原文做筛选。每个候选 task 记录：

```text
task_source_hash, task_id_candidate, provenance_group,
language, statement_hash, split_group,
existing_overlap_status, exact_hash, norm_ws_hash, norm_lex_hash,
source_dataset, license_status, generation_status
```

去重审计至少分三层：

- `exact_hash`：原始文本完全相同；
- `norm_ws_hash`：空白规整后相同；
- `norm_lex_hash`：按已登记的 C 注释/字符串处理规则得到的骨架。

三层碰撞都只能作为筛选证据，不能直接等同于语义泄漏。必须保存 norm_lex 实现版本、碰撞组、剔除数量和保留数量；任何组若跨 split 或命中 E0 暴露账本，整组标为 `excluded`，不能随机拆开。

如果 task 没有稳定的 prompt/task ID，使用 provenance group 和 `task_source_hash` 作为分组键；不要生成伪 task_id。新 task 预注册的 train/dev/test 划分先写入 manifest，再允许 generator 读取；同一 task 的不同 generator 输出只能在这个 task 的指定 split 中出现。

E2b 的干跑交付：`candidate_tasks.jsonl`、`excluded_tasks.jsonl`、`collision_groups.json`、`split_plan.json`、`license_and_provenance.json`、`dry_run_report.md`、命令、日志和 `SHA256SUMS.txt`。这一阶段不得出现生成代码全文、权重、预测或外部 API 响应。

## 3. E2c：第三 generator 的最小生成设计（只写计划）

只有 E2a 为 `ready` 且 E2b 的碰撞/许可/任务分组审计通过，才生成 `generation_plan.json`；仍不执行生成。计划至少说明：

- target generator 的确切版本、API/本地模型来源、许可证和可复现版本标识；
- 和已有 generator 完全相同的 task/prompt 模板、语言、temperature/top-p/max tokens、停止条件和失败重试；
- 每个 task 的预期输出数、失败/超时/拒答处理；
- 不允许根据输出质量回填 task 或删除难例；
- raw output、normalized output、metadata、异常和请求摘要的保存位置；
- 预计新增文件数、字节数、缓存复用、磁盘余量和清理策略；
- 生成后如何计算 exact/ws/lex hash，以及如何检查新输出与全部旧集合的碰撞。

推荐的最小结构是同一新 task 集由同一 source unit 的现有 generator 与目标 generator 共同生成，形成 `same_task/same_family/different_generator` 的正对；跨 family 控制必须在预注册时确定。若目标 generator 只产生 test-only 输出，不能用于 train/dev 监督，也不能称为完整三 generator 支持。

“至少三个 generator”是本项目的工程可识别性门槛，不是显著性定理。E2 只能先给出覆盖和成本；不能提前承诺 ACL 统计功效。当前 51 task 的 CI 半宽约 4.8pt 只作为规划参照，未来样本量需要按真实 task 相关性估计，字段不齐就写 `power_unknown`。

## 4. 生成授权边界

回传 `generation_plan.json` 后停止，等待明确授权。只有授权后才允许：

1. 调用外部模型/API 或下载目标 generator；
2. 写入新的 raw outputs；
3. 承担新增费用或显著磁盘占用；
4. 运行生成后的 H1/H2 训练或 test 读取。

在授权之前，AutoDL AI 可以完成所有 metadata、dry-run、collision、disk 和 cost 计算，但必须保持：

```text
generation_allowed = false
training_allowed = false
test_read_allowed = false
```

不要自动把“预计 15 MB”转成生成动作。若磁盘估计超过 E1 的量级，先只回传可缩减的任务数、字段或缓存方案；不要删除旧产物。

## 5. 授权后的冻结评测协议（先写入计划，不立即执行）

若后续获得授权，评测协议必须先冻结：

### 5.1 H1 主结果

在同一新 task-heldout split 上，以新数据 train/dev 冻结以下双 P0：

- P0-fusion：只在 dev 拟合 fusion；
- P0-eq：五成员概率逐元素等权平均：

\[
p_{\mathrm{eq}}(f\mid x)=rac{1}{M}\sum_{m=1}^{M}p_m(f\mid x),
\qquad \hat f(x)=rg\max_f p_{\mathrm{eq}}(f\mid x).
\]

内容视图、冻结 CodeT5-small、metadata 和 P0 必须同时列出；test 只读取一次。等权集成若与 fusion 比较，差值同时报告点估计差和 paired task-cluster bootstrap 的 `delta_mean`，不能混为一列。

### 5.2 H2 迁移结果

对每个 generator-heldout fold，train/dev 不包含 heldout generator，test 同时使用未见 task；同任务正对必须满足：

\[
P_i=\{j:\mathrm{task}_j=\mathrm{task}_i,\ f_j=f_i,\ g_j
e g_i\}.
\]

主结果按 task cluster 而非输出行进行 bootstrap。至少报告 BA、macro-F1、family recall、generator recall、AUROC/校准和每 fold CI；任何 family 只有一个 generator 时，该 family 不能进入 H2 主表。

### 5.3 转导诊断

task-centered/transductive 视图单独列为诊断。它不能替代普通 inductive H1，也不能用来满足“内容视图稳定超过强 P0”的门槛。若只有转导通过，结论仍是诊断性改善，H2/H3 不启动。

## 6. 停止规则

出现任一情况就停止 E2 扩展并回传：

- 找不到语义明确且可复现的第三 generator；
- 新 task 与已暴露 task/provenance 大量重叠；
- 正式 family 仍达不到三个可靠 generator；
- train/dev 无双标签支持或 heldout fold 只剩 test-only 支持；
- 生成计划无法在新增约 15 MB 量级内实现，且没有用户批准的资源方案；
- 只能通过随机复制、伪标签、跨 split 混合或人工挑容易样本来满足支持矩阵；
- 新 H1 仍未稳定超过双 P0，或提升只来自转导/单 seed；
- 需要新模型、新损失或 H3 才能继续解释结果。

停止不是失败；它把“当前数据无法检验的主张”变成可审计的限制。不要为跨过停止条件而改变标签、拆分或对照。

## 7. 回传格式

1. `e2_preregistration.json`：source unit、family 语义、第三 generator、任务命名空间、权限状态；
2. `candidate_tasks.jsonl`、`excluded_tasks.jsonl`、`collision_groups.json`、`split_plan.json`；
3. `generation_plan.json`：参数、版本、成本、磁盘、失败处理、生成后 hash；
4. `pilot_design.md`：为什么该 unit 能检验 H2，以及不能检验什么；
5. `feasibility_decision.md`：`ready/partial/blocked` 和唯一缺口；
6. commit、目录树、命令、日志、SHA256；确认没有生成 raw outputs、权重、预测或凭据。

E2 完成后只允许得到“可授权生成 / partial / blocked”三种状态。获得授权前不进入模型实验。
