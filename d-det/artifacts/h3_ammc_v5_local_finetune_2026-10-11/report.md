# H3 v5 本机编码器部分微调可行性实验

单 seed，最后两层与 final norm 解冻；原始/组合增强两臂 × 两折 × 两 epoch，共四运行。
这是探索性可行性试验；配置在运行后整理，不称预注册。test 未读取。
FP32 master weights + autocast + GradScaler + clip norm 1.0；encoder dropout 关闭、head dropout 开启；两臂相同。

| fold | view | original task AUROC | composed task AUROC | delta / task CI95 |
|---|---|---:|---:|---|
| fold_0 | clean | 0.9570 | 0.9535 | -0.0035 [-0.0063, -0.0010] |
| fold_0 | hardneg | 0.9562 | 0.9510 | -0.0053 [-0.0105, -0.0009] |
| fold_1 | clean | 0.9317 | 0.9251 | -0.0066 [-0.0119, -0.0013] |
| fold_1 | hardneg | 0.9197 | 0.9139 | -0.0058 [-0.0140, +0.0022] |

保存分数复算 24/24；PyTorch peak allocated 369.8 MiB，不是进程/驱动总显存。
结论：本机可以训练该部分编码器；增强增益不一致，不能用一个 seed 支持或否定 AMMC/H3。
任务内 AUROC 与 pooled row AUROC 分别报告；BA 较低，不以 task AUROC 替代全局检测表现。
未保留 checkpoint、未做变换压力评估，也未使用 joint/source 目标。下一轮需先冻结配置、明确 encoder train/eval 行为、保留 checkpoint 后再运行成对多 seed 评测。
