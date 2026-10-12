# AICD 数值外部控制：大数据快速本机基线

状态：完成；这是 AICD numeric external control，不是 d-det 的 A/B/C 主表，也不替代任务中心化、语义分离或 AMMC 数据。

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

冻结包已上传 AutoDL，但按纠偏裁定只做 SHA256 与 test 边界检查，禁止启动 GPU。数据同时提供 6 个 `jsonl.gz` 标准库读取副本，因此 GPU 脚本无需 `pyarrow`。这些读数只能作为 AICD 数值标签外部控制；真正 d-det ABC 批次必须重新使用任务中心化、语义/来源分离、AMMC train-only 视图和正确的 Human-inclusive 标签。
