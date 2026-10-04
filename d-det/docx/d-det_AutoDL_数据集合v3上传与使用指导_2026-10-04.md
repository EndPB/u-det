# AutoDL 数据集合 v3 上传与使用指导（2026-10-04）

## 本轮新增上传包

按需上传以下压缩包及其 manifest：

0. `d-det/data/aicd_t2_numeric_balanced_v1/aicd_t2_numeric_balanced_v1_upload.zip`
   - 48,877 行；官方 T2 数字标签平衡原型；不得擅自命名 family。
1. `d-det/data/h2_droid_language_matched_v1/h2_droid_language_matched_v1_upload.zip`
   - 11,492 行；语言匹配诊断；不替代 Droid 主包。
2. `d-det/data/h2_droid_generator_balanced_v1/h2_droid_generator_balanced_v1_upload.zip`
   - 25,460 行；每个 split×family×generator 固定上限；用于 generator 供给敏感性分析。
3. `d-det/data/h2_droid_joint_balanced_v1/h2_droid_joint_balanced_v1_upload.zip`
   - 14,386 行；同时控制六种共同语言和 generator；用于严格混淆敏感性分析。
4. `d-det/data/h2_llm_codegen_v2_hardneg_pairs_v1/h2_llm_codegen_v2_hardneg_pairs_v1_upload.zip`
   - 4,332 个完整跨 family 控制对；1,461 个 task×family-pair 平衡对。
5. `d-det/data/h2_authorbench_dcan_hardneg_pairs_v1/h2_authorbench_dcan_hardneg_pairs_v1_upload.zip`
   - 15,232 个完整跨 family 控制对；10,902 个平衡对。
6. `d-det/data/codet_m4_balanced_control_v1/codet_m4_balanced_control_v1_upload.zip`
   - 18,870 行；CoDET-M4 外部来源/结构特征控制；不作为 H2 正对或 family 标签合并。
7. `d-det/data/agent_attribution_v1/agent_attribution_v1_upload.zip`
   - 3,000 条多语言平衡版 + 3,000 条 TypeScript 控制版；仓库留出来源归因控制；不作为 H2 正对。
8. `d-det/data/h2_stacad_alignment_v1/h2_stacad_alignment_v1_upload.zip`
   - 22,260 条核心记录、3,180 个任务、66,780 个同任务跨 generator 对；全部是跨家族 hard negative。
9. `d-det/data/stacad_crossprompt_ood_v1/stacad_crossprompt_ood_v1_upload.zip`
   - 9,933 条记录；P2 train/dev、P3 未见 prompt/file test；仅作 OOD 控制。
10. `d-det/data/h2_llm_codegen_v2_pairs/h2_llm_codegen_v2_pairs_upload.zip`
   - 695 个同任务、同 family、不同 generator 的正对；另含 447 个 family×task 平衡正对。
11. `d-det/data/h2_llm_codegen_view_consistency_v2/h2_llm_codegen_view_consistency_v2_upload.zip`
   - 1,336 条修正 v2 同输出多视图记录、765 条 task/split/family 平衡记录；只用于词法捷径诊断，不把视图当独立样本。
12. `d-det/data/h2_llm_codegen_code_only_pairs_v1/h2_llm_codegen_code_only_pairs_v1_upload.zip`
   - 618 个两端均通过保守代码抽取的正对；另含 334 个平衡正对。
13. `d-det/data/h2_authorbench_dcan_shortcuts_pairs_v1/h2_authorbench_dcan_shortcuts_pairs_v1_upload.zip`
   - 1,651 个同任务、同 family、不同 generator 的五视图正对；另含 863 个平衡正对。
14. `d-det/data/h2_alignment_v3/h2_alignment_v3_upload.zip`
   - 仅含 6,063 条任务索引、88,690 条 pair 元数据和 generator-held-out fold 计划，不含代码副本。
15. `d-det/data/h2_training_v2/h2_training_v2_upload.zip`
   - 源分离任务组便利总包；AuthorBench 2,715 任务、修正 LLM-CodeGen v2 168 任务、STACAD 3,180 任务；不得合并 family 标签。
16. `d-det/data/h2_pairs_v2/h2_pairs_v2_upload.zip`
   - 2,346 条统一格式正对（AuthorBench 1,651 + LLM-CodeGen v2 695），另含 1,310 条 source/task/family 平衡正对；不得合并来源标签。
17. `d-det/data/h2_droid_generator_fold_index_v1/h2_droid_generator_fold_index_v1_upload.zip`
   - 125,718 条 Droid 行级 fold 索引；只含 `source_row_sha1` 和元数据，不复制代码；固定两折 generator-held-out 的 train/dev/test 角色。
18. `d-det/data/acl_attribution_collection_v1/acl_attribution_collection_v1_metadata_upload.zip`
   - 仅含外部 provenance、AICD 审计和标签映射待核对信息；不含 3.2 GB 原始 parquet。

19. `d-det/data/h2_pair_benchmark_v1/h2_pair_benchmark_v1_upload.zip`
   - 4,692 条直接 pair 训练/诊断记录（2,346 正、2,346 hard negative）；每个来源和 split 内严格配平，含五视图；另有 LLM 三家族平衡诊断视图、覆盖 2,030 条任务级 pair 的平衡视图、1,015 条 triplet 索引、1,015 条词法高相似 hard-negative 索引、1,015 条任务多负例索引（2,007 条负例引用）、358 条严格 family-pair 平衡诊断记录和 generator-heldout fold 索引。

对应清单位于 `d-det/docx/*upload_manifest_2026-10-04.json`，总目录为 `d-det/data/dataset_collection_v3/upload_catalog.json`。解压后先核对总目录中的 archive SHA-256，再检查各包的 `SHA256SUMS.txt` 和 `audit_index.json`。

## 实验顺序

0. `aicd_t2_numeric_balanced_v1`：数字标签原型；官方数字到 family 的映射未核实前只能报告 numeric ID。
1. `codet_m4_balanced_control_v1`：外部来源/结构特征控制；只用于检查语言、模型标识和代码结构捷径。
2. `agent_attribution_v1`：仓库留出外部来源归因控制；先跑 TypeScript 控制版，再跑多语言平衡版。
3. `h2_stacad_alignment_v1`：多语言同任务跨 generator 控制；用于任务中心化、跨家族 hard-negative 和外部验证。
4. `h2_llm_codegen_v2_pairs`：H2 正对的直接 pair 输入；优先用于同 family 跨 generator 的最小验证。
5. `h2_authorbench_dcan_shortcuts_pairs_v1`：同 family 正对和五视图捷径诊断。
6. `h2_llm_codegen_view_consistency_v2`：同一输出的视图一致性和 code-only 抽取覆盖。
7. `h2_alignment_v3`：统一读取任务、pair 和 fold 元数据；必须按来源分开训练和报告。
8. `h2_training_v2`：服务器端直接读取三种任务组，但仍分源训练和报告。
9. `h2_pairs_v2`：读取统一正对格式，仍按 source 分开训练、评估和 bootstrap。
10. `h2_pair_benchmark_v1`：直接用于 pair classification/contrastive 训练；正负在 source/split 内配平，验证按 task_id 聚类。

11. `h2_droid_full_selected` + `h2_droid_generator_fold_index_v1`：主 generator-held-out family attribution；索引包固定折角色，主包提供代码。
12. `h2_droid_language_matched_v1`：仅做语言混淆敏感性分析。
13. `h2_authorbench_dcan_hardneg_pairs_v1`：同任务跨 family hard-negative 控制。
14. `h2_llm_codegen_v2_hardneg_pairs_v1`：同 CWE 任务跨 family hard-negative 控制。
15. `h2_stacad_hardneg_pairs_v1`：多语言跨 family 控制。

## 标签和报告边界

- `positive_for_h2=false` 的 pair 永远不能改成 H2 正例。
- Droid、AuthorBench、LLM-CodeGen、STACAD 的 family 标签空间必须分开统计。
- Droid 没有 task_id/prompt_id，不能声称同任务因果证据。
- STACAD 和各 hard-negative 包用于负例/控制；只有明确的同家族跨 generator pair 才能作为 H2 正对。
- AICD 的 numeric label 映射尚未核实，暂不并入任何 family 标签空间。

本机已完成全部重组和完整性审计；服务器端只需上传、解压、核对后运行实验，不需要重新下载原始大数据。


