# 本机阶段 A：数据角色与留出支持审计

流式读取；未训练模型、未调参、未计算 test 分数。

| 数据 | 行数 | family | task | ≥3 generator 的 family | 跨 split task/hash |
|---|---:|---:|---:|---|---|
| authorbench_complete | 1912 | 6 | 239 | openai | 0/0 |
| authorbench_dcan | 9498 | 6 | 2715 | openai | 0/0 |
| llm_codegen_v2 | 690 | 5 | 138 | meta | 0/0 |
| stacad_alignment | 22260 | 7 | 3180 | none | 0/0 |
| droid_full | 146718 | 8 | 0 | 01-ai, codellama, deepseek-ai, meta-llama, microsoft, qwen | 0/0 |
| droid_subset | 18000 | 8 | 0 | 01-ai, codellama, deepseek-ai, meta-llama, microsoft, qwen | 0/0 |
| codet_m4_control | 18870 | 0 | 0 | none | 0/0 |
| aicd_numeric_control | 48877 | 0 | 0 | none | 0/0 |

Pair fold：10 折中 6 折具备 train/dev/test 双标签支持。
这些折只测 same-family pair verification，不能写成 K 类归因。

所有现有数据都不能仅凭 model name 推断共同 base / post-training 因果控制。重复 hash 和缺失字段见 JSON。

## 重跑

`python scripts/audit_acl_local_readiness.py`
