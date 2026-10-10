# H3 GPU 诊断批次的同折词法控制复核

本文件是服务器 512-token GPU 批次的事后 CPU 控制复核，不是新的 GPU 批次。它只把已有 v1 lexical control 按服务器实际的两个 generator-heldout detection 折重新拟合，避免把旧的 `.9084`（全部 generator 训练、task-macro AUROC）与本批次的 generator-heldout row AUROC 直接比较。

## 同一折结果

| 折 | lexical row AUROC | lexical task-macro AUROC | detection-only GPU row AUROC | joint GPU row AUROC | joint-invariance GPU row AUROC |
|---|---:|---:|---:|---:|---:|
| fold_0 | .7849 | .9176 | .6916 | .6976 | .6983 |
| fold_1 | .7223 | .8378 | .6530 | .6546 | .6547 |
| 折均值 | **.7536** | **.8777** | **.6723** | **.6761** | **.6765** |

## 裁定

在相同 generator-heldout 折上，冻结 CodeT5-small 的 detection-only 低于 lexical row 控制约 8.13 个百分点；joint 和 joint-invariance 相对 detection-only 只有 +0.39/+0.42 个百分点，且没有保存 GPU 逐行分数，不能构造配对 CI。source-only 的两折结果实际上重复同一 task-heldout 三 seed，不应计作六个独立折。

因此本批次能支持的结论是：**在 H3 v1、冻结 CodeT5-small、3 epoch head、512-token mean-pooling 这个实例化中，没有观察到超过同折 lexical 控制的有效主线增量。** 它不能支持“所有编码器”或“所有 H3 形式都失败”。

后续动作是本机继续修订长度/词法捷径并重新做闸门；不重跑本批次，不增加容量、温度、epoch 或损失项。正式 H3 仍为 `revise_data`。

证据：`metrics.json`、`server_metrics_snapshot.json`、`fold_0_lexical_scores.npz`、`fold_1_lexical_scores.npz`、`config.json`、`SHA256SUMS.txt`。
