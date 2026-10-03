# 服务端数据审计（2026-10-03）

脚本 `scripts/audit_server_2026_10_03.py`；输出 JSON 同目录。
- 磁盘：{'total_gb': 50.0, 'used_gb': 47.34, 'avail_gb': 2.66, 'used_pct': 94.7}
- GPU：NVIDIA GeForce RTX 3080 Ti（11.6 GB）
- 运行时长（秒）：{'env': 2.4, 'aicd': 35.6, 'droid': 37.2, 'dcan': 37.4, 'llmcg': 37.4, 'stacad': 41.0, 'codet_m4': 42.9}

## AICD
| 配置 | 分片 | 行数 | train/val/test | 空代码 | 重复风险（哈希级） |
|---|---:|---:|---|---:|---|
| T1 | 6 | 1708207 | 500000/100000/1108207 | 2 | sampled-first-50000-per-shard；distinct 300000；跨 split 泄露哈希 0 |
| T2 | 5 | 1111199 | 502149/101176/507874 | 1 | full；distinct 1105413；跨 split 泄露哈希 1848 |
| T3 | 9 | 2100000 | 900000/200000/1000000 | 1 | sampled-first-50000-per-shard；distinct 449974；跨 split 泄露哈希 1 |

T2 逐 split 标签分布：
```
{"test": {"0": 243769, "1": 9674, "2": 26459, "3": 8471, "4": 3631, "5": 21913, "6": 20981, "7": 35411, "8": 11202, "9": 14185, "10": 104304, "11": 7874}, "train": {"0": 442096, "1": 4162, "2": 8993, "3": 3029, "4": 2227, "5": 1968, "6": 5783, "7": 8197, "8": 8127, "9": 4608, "10": 10810, "11": 2149}, "validation": {"0": 88490, "1": 847, "2": 1755, "3": 650, "4": 445, "5": 372, "6": 1118, "7": 1695, "8": 1579, "9": 895, "10": 2154, "11": 1176}}
```
标签映射：**未核验**（HF 数据集卡为空、GitHub 404）→ 一律用 numeric_id。

## Droid
- 行数 146718；家族 {'microsoft': 22998, 'human': 21000, 'meta-llama': 24560, 'qwen': 39046, '01-ai': 13545, 'codellama': 10669, 'deepseek-ai': 9168, 'ibm-granite': 5732}；generator 数 33（含 human 伪 generator，machine=32）；label {'MACHINE_GENERATED': 125718, 'HUMAN_GENERATED': 21000}；split_source {'train': 107599, 'dev': 19609, 'test': 19510}
- 折 anchor=0 家族：{'fold_0': ['codellama', 'ibm-granite'], 'fold_1': ['ibm-granite']}

## AuthorBench DCAN
- 行数 9498；任务 2715；split {'dev': 1484, 'train': 6557, 'test': 1457}；每任务家族数分布 {'2': 1413, '3': 564, '4': 252, '5': 201, '6': 285}；<2 家族任务 0
- 任务跨 split 数 0；重复(task,model) 0；空代码 0；围栏代码 0

## LLM-CodeGen
- 行数 1515；任务 168；split {'train': 1065, 'test': 225, 'dev': 225}；场景 {'secure': 759, 'simple': 756}；每任务家族数分布 {'5': 168}；<2 家族任务 0
- 任务跨 split 数 0；重复(task,model) 3；空代码 169；围栏代码 662

## STACAD
- 对 144958；文件 22053；文件级多 split 0；split_v1 {'train': 128811, 'val': 8055, 'test': 8092}；fold 分布 {'0': 28992, '1': 28992, '2': 28992, '3': 28991, '4': 28991}
- fold 对齐：{'mode': 'direct-stream-order', 'files_with_multi_fold': 0, 'multi_fold_examples': [], 'files_per_fold': {'0': 4411, '1': 4411, '2': 4411, '3': 4410, '4': 4410}, 'runs': 18981}

## CoDET-M4
- 行数 500552；target {'human': 246221, 'ai': 254331}；language {'python': 185163, 'java': 174169, 'cpp': 141220}；missing model 13587（null）+ 0（空串）

## 切分清单（split_manifest_2026-10-03.json 摘要）
- AICD T2：{'test': 507874, 'train': 502149, 'validation': 101176}（numeric labels；映射 pending）
- DCAN：任务/行 {'dev': 407, 'train': 1900, 'test': 408} / {'dev': 1484, 'train': 6557, 'test': 1457}（group=task_id）
- LLM-CodeGen：任务/行 {'train': 118, 'test': 25, 'dev': 25} / {'train': 1065, 'test': 225, 'dev': 225}（group=task_id）
- STACAD：文件级五折 0 跨折；文件/折 {'0': 4411, '1': 4411, '2': 4411, '3': 4410, '4': 4410}；folds sha256 55b59f60365418a0…
- Droid：generator-held-out 两折（仅外部压力测试）；CoDET-M4：仅结构化指纹控制

## 检查清单
- `aicd_shards_20`：**pass** — {"T1": 6, "T2": 5, "T3": 9}
- `aicd_rows_vs_manifest`：**pass** — {"T1": {"manifest_rows": 1708207, "computed_rows": 1708207, "match": true}, "T2": {"manifest_rows": 1111199, "computed_rows": 1111199, "match": true}, "T3": {"manifest_rows": 2100000, "computed_rows": 2100000, "match": true}}
- `aicd_t2_cross_split_hash_leak`：**info** — {"scan": "full", "rows_hashed": 1111199, "distinct_hashes": 1105413, "dup_rows_vs_distinct": 5786, "cross_split_leaked_hashes": 1848}
- `aicd_label_map`：**pending** — "官方映射未核验：HF 数据集卡为空、GitHub 404；保留 numeric_id（11 类），不出具具体家族命名"
- `droid_rows_146718`：**pass** — 146718
- `dcan_min2_families_per_task`：**pass** — {"2": 1413, "3": 564, "4": 252, "5": 201, "6": 285}
- `dcan_task_split_purity`：**pass** — {"1": 2715}
- `llmcg_min2_families_per_task`：**pass** — {"5": 168}
- `llmcg_task_split_purity`：**pass** — {"1": 168}
- `stacad_folds_len`：**pass** — {"folds_len": 144958, "pairs": 144958}
- `stacad_file_level_split_purity`：**pass** — 0
- `stacad_fold_alignment`：**pass** — {"mode": "direct-stream-order", "files_with_multi_fold": 0, "multi_fold_examples": [], "files_per_fold": {"0": 4411, "1": 4411, "2": 4411, "3": 4410, "4": 4410}, "runs": 18981}
- `codet_m4_model_missing`：**info** — {"null": 13587, "empty": 0}
