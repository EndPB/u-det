# H3 定义恢复与 human 支持表

## 标签重命名（§4.2）

- 已完成四配置的“检测”分支实为 **protocol_direction**（complete/instruct 都是 AI）——
  改称 `protocol_direction_plus_relation_multitask`；D、F 与协议方向 R0 分开报告（记录收尾，不重跑补 epoch）。

## human 支持表

| 源 | human | 同题 human×AI | 依据 | 判定 |
|---|---|---|---|---|
| public_same_task_full_2026-10-07 (BCC) | 无 | ✗ | 全部 rows 为机器输出（complete/instruct 双协议）；used records 字段含 solution；无 human 行 | not_applicable_no_human |
| DroidCollection (h2_droid_full_selected) | 有 | ✗ | README 明示：DroidCollection does not expose a public task_id/prompt_id, so results | not_same_task |
| CoDET-M4 | 有 | ✗ | audit role=external structural/source-fingerprint control; not a same-task seman | not_same_task |
| STACAD-v2 alignment (h2_stacad_alignment_v1) | 有 | ✓ | task_index: 同 file 的 human_code + 7 个模型输出；pairs: 66780 same_task_cross_generator | support_candidate |

## 结论

- **H3_original_unavailable（当前同题全 machine 集无 human；不得把协议方向 D 当检测轴）**
- 支持候选：**STACAD-v2（同题 human×AI + 7 generator/7 family 归因轴）**
- 后续要求：检测轴（human vs AI）与归因轴（generator）在同一 STACAD 任务集上的分离设计验证（另行立项）；许可与使用条款复核；语言/难度分层；test 轴策略须预先冻结；不得在无同题支持的数据（Droid/CoDET-M4/BCC 拼接）上写“检测轴与归因轴分离”
