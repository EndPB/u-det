# H2 pair benchmark v1 训练包指导（2026-10-04）

上传包：`d-det/data/h2_pair_benchmark_v1/h2_pair_benchmark_v1_upload.zip`。

这是从已有审计包抽出的紧凑训练/诊断包，不下载或生成新原始数据。它把 AuthorBench 和修正后的 LLM-CodeGen v2 分开保留，并在每个来源、每个 split 内严格配平：

- `pair_label=1`：同任务、同 family、不同 generator 的 H2 正对；
- `pair_label=0`：同任务、跨 family 的 hard negative；
- 每条记录含 `raw`、`ids_only`、`strings_only`、`comments_only`、`all` 五个视图和哈希。

主文件总计 4,692 条 pair（2,346 正、2,346 负）。服务器端先检查 `audit_index.json` 的所有 `checks`，再按 `source` 分开训练或报告。不要把两个来源的 family 编码合并，也不要把 pair 行当作独立样本；验证和 bootstrap 必须按 `task_id` 聚类。

包内另有 `task_balanced_pairs.jsonl` 任务级诊断视图和 `llm_family_balanced_pairs.jsonl` 家族级诊断视图：在每个 split 内将有正对的 Google、Meta、Mistral 三个家族等量抽样。AuthorBench 的正对全部来自 OpenAI，因此不伪装成多家族平衡证据。

`generator_fold_index.jsonl` 和 `generator_fold_summary.json` 提供 generator-held-out 索引。Meta 和 AuthorBench 的折有训练正对支持；Google 和 Mistral 只有两个 generator，留出一个后 train/dev 正对为零，摘要中已标记为 `trainable_positive_support=false`，只能做诊断，不能当作成功训练折。

`task_balanced_pairs.jsonl` 是更保守的任务级视图：只保留同时有正负 pair 的任务，并且每个来源/split/task/label 只取一条，覆盖所有存在对应正负 pair 的任务，适合检查任务数量偏斜是否影响结论。

`triplet_index.jsonl` 在这些任务上提供 anchor–positive–negative 三元组索引：anchor/positive 是同家族跨 generator，negative 是同任务跨家族。它只引用 pair ID 和端点哈希，服务器端通过 `pairs.jsonl` 连接五视图，不重复保存代码。

`lexical_hardneg_index.jsonl` 为每个有正对的任务选出 identifier/string 词法重叠最高的跨家族负例，用于检验模型是否依赖简单表面捷径；它只能作为负例诊断。

`task_multi_negative_index.jsonl` 按任务收集包内所有已选跨家族负例（1015 个任务、2007 条负例引用），只保存 pair ID、家族和 generator 元数据，不复制代码。它适合构造多负例对比 batch，也能避免把同一任务误当作独立样本。

`family_pair_balanced_pairs.jsonl` 是更严格的 family-pair 诊断子集：每个来源和 split 内，负例 family-pair 数量相等，同时每个保留任务有一条正例和一条负例。它共 358 条记录（179 正、179 负）；由于部分测试任务没有足够的 family-pair 交叉覆盖，LLM-CodeGen v2 测试 split 只保留可完成匹配的 9 个 family-pair，不能把该视图当作全量 benchmark。

该包适合直接做 H2 的 pair classification、对比损失或视图消融。STACAD 不在其中，因为它没有同家族跨 generator 正对。
