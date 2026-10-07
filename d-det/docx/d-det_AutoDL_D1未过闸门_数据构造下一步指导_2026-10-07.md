# d-det AutoDL D1 未过闸门后的数据构造指导

日期：2026-10-07
服务器基准：`6e329da` + `c2a2e93`，已推送 `origin/main`，工作树干净

## 0. 当前结论

D0 已通过：`h2_alignment_v3` 的 10 折中 6 折 admitted、4 折 diagnostic-only；task 不跨 split，端点哈希不跨 split，train/dev 双标签支持齐全。D0 的完整交付是 `d-det/artifacts/stage_d_support_2026-10-07/` 下的 `alignment_support.json`、`data_role_matrix.json`、`report.md`、`commands.txt`、日志、提交状态和 `SHA256SUMS.txt`。

D1 未通过。AuthorBench-DCan 的严格 task-heldout test（单次读取，`n=1457`，500 次 task-cluster bootstrap）得到：

| view | macro-F1 | CI95 | 相对 P0 fusion `.8388` |
|---|---:|---|---:|
| metadata_only | `.3744` | `[.3512,.3970]` | `-.4644` |
| tfidf_word | **`.8200`** | `[.7966,.8400]` | **`-.0191`** |
| tfidf_char | `.7692` | `[.7456,.7930]` | `-.0697` |
| codet5_meanpool | `.6440` | `[.6186,.6662]` | `-.1954` |
| codet5_centered | `.6487` | `[.6227,.6715]` | `-.1908` |
| p0_fusion_lr | `.8388` | `[.8189,.8588]` | `0` |

因此当前结论是：**task-aware 可读性/任务效应审计完成，但 H1 证据不足，H2/H3 不启动。** 不得把 D1 失败解释为某个损失函数失败，也不得据此声称 generator-invariant 表示已经被否定。

## 1. 下一阶段唯一目标

下一阶段只做**数据构造与支持性审计**，目标是回答：

> 当前内容视图低于 P0，究竟是因为任务/家族/generator 支持不均衡、task-size 和 generator 组成偏置，还是数据本身没有足够的跨 generator 可识别信号？

在这个问题没有被数据审计回答前，不做新 backbone、新 head、新损失、新权重下载和 H2/H3 训练。

## 2. N0：只读复盘 D1 失败来源

创建独立目录：

```bash
cd /root/autodl-tmp/u-det
mkdir -p d-det/artifacts/stage_d_data_construction_2026-10-07/{n0_diagnosis,n1_candidate_matrix,logs}
git rev-parse HEAD > d-det/artifacts/stage_d_data_construction_2026-10-07/git_head.txt
git status --short --branch > d-det/artifacts/stage_d_data_construction_2026-10-07/git_status.txt
```

只读解析 D1 已存在的 `metrics.json`、`predictions.npz`、`report.md` 和 P0 复算结果，按以下轴输出 macro-F1、BA、recall、样本数和 500 次 task-cluster bootstrap CI：

- family × generator；
- task-size 桶：`1`、`2–5`、`6+`；
- family × task-size；
- generator × task-size；
- train/dev/test 的 task 数、样本数和标签比例；
- 每个 family 的 generator 数及每个 generator 的 task 数；
- 预测错误中 exact duplicate、normalized code duplicate、同 task 跨 split 和同 generator 偏置的审计结果。

已知现象必须在报告中单独保留：task-size `6+` 桶 F1 `.7644`；`deepseek` 约 `.60`、`qwen` 约 `.65`，而 `gemini` 约 `.97`。这些数字只能作为分层诊断，不能直接当作模型因果结论。

N0 交付：`n0_diagnosis.md`、`n0_strata.json`、错误分层表、原始输入路径、命令、日志和 `SHA256SUMS.txt`。不产生训练权重。

## 3. N1：构造候选 task-aware 支持矩阵

N1 只生成轻量索引、manifest 和审计结果，优先复用 AutoDL 已有数据；没有必要上传或复制原始数据、CodeT5 权重和历史 predictions。

候选矩阵至少要显式记录：

```text
family, generator, task_id, split, example_id, label,
source_sha256, normalized_code_sha256, prompt_or_task_sha256,
task_size, language, role
```

硬约束：

1. 同一个 `task_id` 只能出现在一个 split；
2. `source_sha256` 和 `normalized_code_sha256` 不得跨 split；
3. train/dev 必须对每个正式 family 保持正负双标签支持；
4. 正式 H2 候选 family 至少需要 3 个有足够 task 支持的 generator；只有 1 个 generator 的 family 只能作为 H1 或诊断数据；
5. generator、family、task-size 的占比必须在 manifest 中公开，不能用总样本数掩盖 generator 不平衡；
6. 每个 test generator 必须在 train/dev 之外保持明确的 held-out 角色；
7. Google/Mistral 当前 train/dev 正支持为 0 的 4 折继续标为 diagnostic-only，不得伪造正例或把 test-only 支持用于训练；
8. 所有候选矩阵必须保存原始行数、去重行数、剔除原因和每一步 hash。

如果现有数据无法为某个 family 提供 3 个可靠 generator，必须保留该限制，不能通过随机复制、伪标签、跨 split 混合或把不同 task 冒充同 task 来补齐。

N1 交付：

- `n1_candidate_support_matrix.json`；
- `n1_data_role_matrix.json`；
- `n1_duplicate_and_split_audit.json`；
- `n1_balance_report.md`；
- `n1_manifest.jsonl` 或等价轻量索引；
- `commands.txt`、完整日志、`git_head/status`、`SHA256SUMS.txt`。

## 4. N2：候选矩阵通过后才做一次最小 H1 复核

只有 N1 满足全部硬约束，才允许在候选矩阵上做一次固定协议 H1 复核：

- 保持 task-heldout split；
- 先复算固定 P0，再比较 TF-IDF word、TF-IDF char 和冻结 CodeT5-small；
- 只在 train/dev 选择阈值和超参数，test 只读取一次；
- 使用至少 2 个 seed 或等价的预注册重复；
- 报告 macro-F1、BA、family recall、generator recall、task-cluster bootstrap CI、ECE/NLL 和完整 confusion matrix；
- 结果必须与原 D1 的 `.8388` P0 基线并列，而不是替换原结果。

N2 的作用是验证“数据构造是否修复了支持不足”，不是寻找更复杂模型。只有候选矩阵上的内容信号达到预注册门槛且方向稳定，才重新讨论 H2；否则把结果归档为数据限制和负结果。

## 5. 明确禁止事项

在 N0/N1/N2 完成前，不运行：

- SupCon、DMHM、低秩、正交、PCGrad 或任何多损失组合；
- 新 backbone、LoRA、可训练 CodeT5 或新 adapter；
- H2 pair training、跨 generator 伪正对、H3 detection/private-adapter；
- 重新跑已经完成且逐位一致的 B1/B2/P0 校准；
- 下载新数据或把整个数据集、权重、历史 predictions 再次上传。

## 6. 回传格式

回传必须包含：

1. N0 的分层诊断和失败来源判断；
2. 每个 family 的 generator/task 支持表；
3. 候选矩阵的 split、duplicate、label、hash 审计；
4. 是否满足“每正式 family 至少 3 个可靠 generator”的结论；
5. 是否允许 N2；若不允许，给出停止原因；
6. 目录树、命令、commit、完整日志和 `SHA256SUMS.txt`。

当前最正确的下一步是先完成 N0/N1 数据构造审计。没有新的支持矩阵证据，不启动任何 H2/H3 方法实验。
