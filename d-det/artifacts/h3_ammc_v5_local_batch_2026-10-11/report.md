# H3 v5 固定 parent 预算的 AMMC 类增强对照

探索性 train/dev；v4 曾使用 dev 选样；不读取 test，不声称 project-heldout。
所有 arm 每个 parent 每个 epoch 恰好曝光一次；失败回退原视图；dev 未增强。

| fold | eval | arm | raw lexical tm | detection tm | joint tm | invariance tm |
|---|---|---|---:|---:|---:|---:|
| fold_0 | clean | original | 0.9115 | 0.9137 | 0.9184 | 0.9184 |
| fold_0 | clean | single | 0.8978 | 0.9115 | 0.9192 | 0.9186 |
| fold_0 | clean | composed | 0.8902 | 0.9086 | 0.9179 | 0.9174 |
| fold_0 | hardneg | original | 0.8929 | 0.8997 | 0.9058 | 0.9058 |
| fold_0 | hardneg | single | 0.8955 | 0.8974 | 0.9076 | 0.9070 |
| fold_0 | hardneg | composed | 0.8914 | 0.8976 | 0.9044 | 0.9042 |
| fold_1 | clean | original | 0.8625 | 0.8563 | 0.8606 | 0.8606 |
| fold_1 | clean | single | 0.8568 | 0.8555 | 0.8624 | 0.8617 |
| fold_1 | clean | composed | 0.8526 | 0.8590 | 0.8613 | 0.8612 |
| fold_1 | hardneg | original | 0.8328 | 0.8397 | 0.8408 | 0.8408 |
| fold_1 | hardneg | single | 0.8375 | 0.8356 | 0.8436 | 0.8437 |
| fold_1 | hardneg | composed | 0.8333 | 0.8390 | 0.8431 | 0.8429 |

结论：`augmentation_increment_not_established`。完整 task 成对 bootstrap CI 见 metrics.json。
此实验只检验冻结表示上的增强效果；编码器未训练，不能据此排除 AMMC 在端到端微调上的作用。
未做 dev 变换压力测试，因此不能把原视图指标称为词法不变性证据。
词法控制使用全文，CodeT5 使用长度上限，结果只解释当前实例化；source-only 是已见 generator/task-heldout，不是未知 generator 归因。
