# H3 v4 本机完整对照实验

本轮为探索性诊断：v4 使用了 dev 信息选样，正式闸门仍为 revise_data；不声称项目留出或确认性主结果。

18 个检测运行 + 3 个独立 source-only 运行；训练集使用 label/transform-balanced 的 AMMC-like comment/whitespace/rename 视图，原始 clean-dev 与 v4 hard-negative 保持未增强。

| fold | view | raw lexical row / task | detection task | joint task | invariance task |
|---|---|---:|---:|---:|---:|
| fold_0 | hardneg | 0.7610 / 0.8838 | 0.9016 | 0.9072 | 0.9132 |
| fold_0 | clean | 0.7685 / 0.8993 | 0.9054 | 0.9108 | 0.9118 |
| fold_1 | hardneg | 0.6692 / 0.8336 | 0.8346 | 0.8424 | 0.8424 |
| fold_1 | clean | 0.7219 / 0.8560 | 0.8519 | 0.8608 | 0.8594 |

结论：`increment_not_observed_close_v4_frozen_candidate`。

所有增量在同折、同一行集合与同一 task-macro 指标下比较，CI 使用 task 成对 bootstrap。
AMMC-like 训练视图只来自 train parent，按 Human/AI 与 transform 等额采样，并在 AI generator 间轮转；没有把变体加入 dev/clean，也没有把 transform mask 作为输入。
掩码控制仅为次要诊断，不能替代 raw 词法强控制。source-only 只计 3 个独立 seed，human 不计作未知 generator。
逐行预测、输入审计、变体 accepted 证明、损失历史与环境均保存；test 未读取，未生成代码，未下载模型，未执行数据集代码。
