# fresh_same_task_index（等待本机材料）

- 状态：`awaiting_local_materials`（D0 任务池构建在本机执行；AutoDL 只接收索引与必要特征）。
- 期望输入：`tasks.jsonl`、`unit_manifest.jsonl`、`pair_index.jsonl`、`split_index.csv`、`collision_audit.json`、`lineage_adjudication.json`、`fresh_preregistration.json`。
- 收到后按指导 §7/§13 复核三层 hash、许可、split，再登记 `received_manifest.json`。
- 未授权生成：`generation_allowed=false`。
