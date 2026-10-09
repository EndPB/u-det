# code_conditioned_design_2026-10-09（输入重建与静态预检；server reconstruction）

性质：数据/角色审计与特征构建（非方法结果）。train/dev only；test_read=false；generation=false；code_execution=false。

## inputs/
- `records_train_dev.jsonl`：10,659 行 = 11 成员 × 969 task（instruct 模式），行序=语料文件序。
  模式判别证据：instruct 模式重建给出静态覆盖 10,659/10,659 可解析、10,602 顶层入口，与本机预检数字一致；complete 模式为 10,594。
- `family_series_admission.json`：3 已登记观测系列 → 11 成员（family_is_confirmed=false）。
- `input_hashes.json`：输入哈希/行序哈希/task split/成员映射/代码 SHA 列表哈希。
- `bigcodebench_task_columns.parquet`：任务元数据 5 列白名单（未选测试正文/canonical solution）。

## static_preflight/
服务器端重跑 `scripts/prepare_code_conditioned_pilot.py`（与随附脚本逐字节一致）：
- coverage: parse_ok=10,659 / entrypoint_present=10,602（与本机一致）
- 11 折 fold_plan（每折 train 7,980 / eval 1,368 / 正 171）
- `static_proxies.jsonl` 仅含代码静态代理；correctness=null。

## features/
- `bundle.npz`：hy_small(10659×512，行映射修复后与 texts 缓存逐行校验 10,659/10,659)、style/meta/sizelen/sizeB、psi(9)、member/task 索引。
- `emb_task_small.npz`：969×512 题面（instruct_prompt）嵌入；编码器协议与 r0 缓存一致（CodeT5-small 冻结 mean-pool，512=384+128，fp16）。
- 行映射修复记录：早期用本地成员索引乘 nT*2 导致抽错行；改为 115 模型全局索引后校验通过（见 features_manifest.json 的 row_mapping_check）。

> 本机 static_preflight 产物（audit.json/fold_plan.json/static_proxies.jsonl/SHA256SUMS）为本轮 manifest 列出的应传文件，但未出现在服务器；服务器按其规格重建并保存全部哈希。逐点 1e-3 审计需本机参考表。
