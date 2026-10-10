# H3 GPU 诊断批次：本机低资源实跑

日期：2026-10-11  
设备：NVIDIA GeForce RTX 3050 Laptop GPU，4 GiB  
批次性质：低资源诊断，不替代 H3 正式闸门，也不读取 test。

## 配置

- 数据：H3 revision v1 的 `train_balanced.jsonl`、`dev_clean.jsonl`、`train_variants.jsonl`；
- 编码器：本地 CodeT5-small，冻结 mean pooling；
- `max_length=128`，`max_source_chars=24000`；
- 2 个确定性 generator-heldout detection 折；source-only 使用任务留出；
- 4 个 job × 2 折 × 3 seed = 24 次运行；3 epoch；seed 为 `20261011,20261012,20261013`；
- test 正文、生成、权重下载均未使用。

编码 22,251 个唯一代码耗时约 55.7 秒。

## 聚合结果

| job | runs | detection AUROC mean | detection BA mean | source seen accuracy mean |
|---|---:|---:|---:|---:|
| detection_only | 6 | 0.7221 | 0.5708 | — |
| source_only | 6 | — | — | 0.1528 |
| joint | 6 | 0.7275 | 0.5794 | — |
| joint_invariance | 6 | 0.7280 | 0.5748 | — |

相对 `detection_only`，`joint` AUROC 增量约 +0.55pt，`joint_invariance` 约 +0.59pt；这是低资源诊断读数，不是 H3 正式主表结论。完整 512-token 版本已上传 AutoDL，使用同一脚本的默认配置。

证据：`metrics.json`、`run_manifest.json`、`run.log.txt`。
