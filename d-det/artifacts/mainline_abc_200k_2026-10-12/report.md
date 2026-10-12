# A/B/C 大数据快速本机基线

状态：完成；这是 200k 级 CPU 线路筛选，不是 CodeT5 主表。

## 数据

| 任务 | 配置 | train | validation | 标签 |
|---|---|---:|---:|---|
| A | AICD T1 | 200,000 | 12,000 | 2 个原始数字标签 |
| B | AICD T2 | 200,000 | 12,000 | 12 个原始数字标签，包含 Human 类集合 |
| C | AICD T3 | 200,000 | 12,000 | 4 个原始数字标签 |

AICD parquet 的公开字段只有 `code` 与 `label`，没有 task/project/generator 字段。训练和验证按官方 split 读取，test 没有打开。

## CPU 结果

| 任务 | lexical Macro-F1 | lexical+format Macro-F1 | format-only Macro-F1 |
|---|---:|---:|---:|
| A | 0.9003 | **0.9567** | 0.9218 |
| B | 0.2929 | **0.3502** | 0.2018 |
| C | 0.5926 | **0.6426** | 0.5254 |

模型是 streaming Hashing-TF-IDF + SGD logistic，目的只是快速确认大数据入口、格式通道和三任务标签管线能够产生可测读数。它不等于 CodeT5 微调，也不证明格式是后训练因果信号。

## 下一批

冻结包已上传 AutoDL。数据同时提供 6 个 `jsonl.gz` 标准库读取副本，因此 GPU 脚本无需 `pyarrow`。服务器无 GPU 时只完成 SHA256 与 test 边界检查；GPU 恢复后一次运行 `codet5_frozen` 与 `codet5_layout`，覆盖 A/B/C，再决定是否追加完整微调和 AMMC 视图批次。
