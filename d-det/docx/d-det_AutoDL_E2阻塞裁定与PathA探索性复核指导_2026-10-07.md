# d-det AutoDL E2 阻塞裁定与 Path A 探索性复核指导

日期：2026-10-07
服务器基准：E2 提交 `78e52d3`，工作树干净
当前决定：**E2 对独立确认性 H2 仍为 blocked；Path A 仅可作为明确降级的探索性复核候选**

## 0. 裁定先写清楚

E2a 的 Path A 是 `h2_llm_codegen_v2::meta`，包含 llama2/llama3/codellama 三个 generator；E2b 从 LCv2 train+dev 形成 144 个候选 task，计划切为 train/dev/test = 102/20/22 task。它的数据卫生和 generator 支持比 B/C 更接近可用，但不能称为“新的独立验证集”：

- LCv2 v2 原始命名空间共 168 task，原 split 为 train/dev/test = 118/26/24；
- 这套 namespace 已参与历史 pair train/dev 开发；
- 原始 test 24 中已有 22 个被读取；其余历史 test 也不能自动成为全新 test；
- Path A 候选来自原 train+dev，新的 102/20/22 只是对已开发 namespace 的重新分组；其中新 test 可能包含历史训练/调参中已经使用过的 task 或相关 CWE group；
- 因此它可以回答“已有开发域内的未按本轮读取划分上的工程表现”，不能回答“未参与开发的新任务上的独立泛化”。

Path A 的正式标签应为：

```text
confirmatory_independent = false
exploratory_development_namespace_reuse = true
h2_main_table_allowed = false
h2_appendix_diagnostic_allowed = conditional
training_allowed = false
new_generation_allowed = false
```

这不是否定 Path A 的数据价值，而是防止把历史开发域重切分写成 ACL 的独立 H2 证据。

## 1. 三条路径的最终状态

| 路径 | 处理 | 允许的结论 |
|---|---|---|
| A：LCv2 Meta 三 generator，复用现有输出 | **exploratory only** | 族内同 task 跨 generator 的工程诊断；不能写独立泛化、不能进主 H2 表 |
| B：Google/Mistral 补第三 generator | **blocked** | 缺权重/API/版本/许可；不下载、不猜版本、不调用外部服务 |
| C：AuthorBench OpenAI 新 task | **blocked** | AB namespace 已多次暴露；新 task/API 未授权，不能作为独立确认 |

Path A 的 72 个 multi 组属于 group 内 simple/secure 设计对，因同 prompt、同 group、同 split 而不是泄漏；这一点可以保留。但它不能消除历史开发使用造成的评测依赖。

## 2. E3a：先做任务历史角色裁定，不读取新 test

创建：

```bash
cd /root/autodl-tmp/u-det
mkdir -p d-det/artifacts/stage_e3_patha_adjudication_2026-10-07/{audit,exploratory,logs}
git rev-parse HEAD > d-det/artifacts/stage_e3_patha_adjudication_2026-10-07/git_head.txt
git status --short --branch > d-det/artifacts/stage_e3_patha_adjudication_2026-10-07/git_status.txt
```

针对 Path A 的 144 task，按历史 `source_split`、历史 pair 训练/开发清单、task/group/CWE 标识和 E0 暴露账本输出：

```text
task_id, group_id, cwe, historical_role,
historical_train_or_dev_seen, historical_test_read,
current_e3_role, task_source_hash, endpoint_hashes,
related_group_seen, admissibility
```

`historical_role` 至少区分：

- `historical_train_seen`：task 或其同组数据参与过历史训练；
- `historical_dev_seen`：用于阈值/超参/模型选择；
- `historical_test_read`：已被读取或用于报告；
- `historical_unread_but_development_namespace`：未读但来自已开发 namespace；
- `fresh_unseen_namespace`：新的 provenance/task namespace。

Path A 的 102/20/22 新 split 不得覆盖历史角色。若候选 test 包含 `historical_train_seen` 或 `historical_dev_seen`，必须在报告中逐 task 标出；不能因为本轮 test 尚未读取就称为 fresh。

E3a 只读交付：`path_a_history_role.json`、`path_a_adjudication.md`、task/group 映射、hash、命令、日志、`SHA256SUMS.txt`。不生成预测，不重新读取 test 文本，不训练。

## 3. E3b：只有用户明确接受降级标签，才允许 Path A 探索性复核

服务器 AI 不自行把 Path A 从 blocked 改为 ready。若用户接受“development-namespace exploratory”标签，才可做一次只读复核，且先写入 `exploratory_authorization.json`：

```json
{
  "accepted_label": "exploratory_reuse_only",
  "confirmatory_independent": false,
  "main_h2_table": false,
  "appendix_or_diagnostic": true,
  "training_allowed": false,
  "new_generation_allowed": false,
  "new_test_read_allowed": true,
  "selection_after_read": false,
  "allowed_question": "performance on a held-out re-split of a historically developed namespace",
  "forbidden_claims": [
    "independent new-task generalization",
    "universal cross-family geometry",
    "post-training causal effect",
    "H3 separation"
  ]
}
```

即使授权，评测也只能使用**现有冻结输出/现有冻结读出**；不得在 Path A 上重新训练、调阈值、选 seed、选择任务、挑 generator、重做 P0 或改变 split。任何需要训练或重拟合 fusion 的方案都保持 blocked，因为它会在历史开发域上继续开发。

探索性表格最多报告：每 generator-heldout fold 的 BA、macro-F1、family/generator recall、task-cluster bootstrap CI、历史角色构成和 test exposure。主标题必须写“LCv2 development-namespace exploratory re-split”。结果只进 appendix/diagnostic ledger，不进 ACL 主 H2 结论。

## 4. E3c：真正的独立证据路线

如果目标是 ACL 主 H2 证据，必须构造 `fresh_unseen_namespace`：

1. 新 task 来源和 provenance 不在 LCv2/AB 已开发或已暴露集合内；
2. 先按 task/provenance group 固定 train/dev/test，再让 generator 生成；
3. 同一个新 task 至少由同一 source unit 的三个可靠 generator 生成；
4. 目标 generator 的版本、许可、请求参数和失败重试可复现；
5. generation 前冻结 positive/negative 定义、heldout generator 折、强 P0 双列、test 只读一次；
6. 生成后的 exact/ws/lex 与历史全量集合碰撞为 0，碰撞组保留并解释；
7. 未授权前只提交设计、成本和磁盘预算，不能调用 API 或下载模型。

新的三 generator 数据不必一次覆盖所有 family；可以先做一个语义明确的 pilot，报告它能检验什么、不能检验什么。一个 source unit 的 pilot 不能被写成全局 family geometry 证据。

## 5. 论文中如何写 Path A

可以写：

> We additionally report an exploratory generator-held-out re-split within the LLM-CodeGen v2 development namespace. This analysis reuses a historically developed task pool and is therefore not treated as an independent confirmation of H2 generalization.

不能写：

- “在全新任务上验证了 Meta 泛化”；
- “三 generator 证明了普遍 family 几何”；
- “未读本轮 test 等于未参与历史开发”；
- “Path A 结果足以启动 H2/H3”。

若不接受降级标签，E3 直接结束为 blocked，主论文保留 E0/E1 的数据限制和现有负结果；这比把开发域重切分包装成独立 test 更严谨。

## 6. 停止规则与回传

任一情况停止：

- Path A 历史角色无法逐 task 对齐；
- 需要重新训练、调参、挑 seed 或读取后选择 split；
- B/C 仍缺权重、API、许可或新 task 授权；
- 不能构造 fresh unseen namespace；
- 只能靠拼接不同 dataset 消除 generator 缺口。

回传：

1. `path_a_history_role.json` 和 `path_a_adjudication.md`；
2. 明确 Path A 是 `exploratory_reuse_only` 还是继续 `blocked`；
3. 若获降级授权，`exploratory_authorization.json`、冻结配置和不重拟合证明；
4. 若走真正独立路线，`fresh_namespace_plan.json`、第三 generator 缺口、成本/磁盘估计；
5. commit、目录树、命令、日志、SHA256；不提交 raw outputs、权重、预测或凭据，除非另有授权。

当前默认出口是 **E3a 只读历史角色裁定，Path A 仍不进入 ACL 主 H2 表**。在此之后再决定是否接受探索性附录，或投入资源构造 fresh namespace。
