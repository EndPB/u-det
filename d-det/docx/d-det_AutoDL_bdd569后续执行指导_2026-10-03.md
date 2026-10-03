# bdd569 之后的 AutoDL 执行指导

日期：2026-10-03

## 1. 这次提交解决了什么

远端最新提交为 `bdd569cf8f39be635f15d49897ccc3a9b718fd9d`。这是 ACL v1 round3 审计修正版，当前应以它及 `d-det/artifacts/acl_dcan_round3_audit/` 中的结果为准。

上一轮 MLP 评估没有可靠地关闭 BatchNorm 与 Dropout，导致验证/测试结果受 batch 组成影响。round3 已统一使用 `model.eval()`、`torch.inference_mode()`，保存完整 `state_dict`（包括 BatchNorm 缓冲区），并加入了模式断言、重复性检查和测试身份字段。旧报告里的 MLP 数字不能继续作为最终数字。

## 2. 当前可以写进论文的结果

- 纠正后的 late fusion：测试 Macro-F1 `0.7381 ± 0.0043`。
- 纠正后的 full disentangle：`0.7356 ± 0.0021`。
- SupCon 相对 `lf_nosupcon` 的方向趋势约为 `+1.07` 个百分点，但每个 seed 的配对 bootstrap 区间都跨 0，不能写成已证实的有效机制。
- 四类机制开关（SupCon、semantic-GRL、fingerprint-GRL、orthogonality）没有得到一致的正向因果证据；只能报告方向与不确定性。
- 在完全匹配的 task-held-out 条件下，LoRA 为 `0.7286 ± 0.0140`，head-only 为 `0.4163 ± 0.0112`，配对 bootstrap 差异约 `-31.2` 个百分点（head-only 减 LoRA，区间均不跨 0）。这说明适配器对当前任务划分有明显帮助，但还没有 file-held-out 或 generator-held-out 证据。
- TF-IDF 基线为 `0.7639`，仍高于当前神经模型；因此现在不能声称 SOTA。
- 捷径实验只能支持较窄的结论：联合匿名化（评论、字符串、空白等一起处理）会使性能从约 `0.7639` 降到约 `0.4572`，但尚未把标识符、评论、字符串、格式和任务混杂逐项分离。

round3 还重新核对了 LoRA checkpoint；旧 checkpoint 的概率可以复现，重算 F1 为 `0.7088`（seed0），所以不是 checkpoint 损坏问题。

## 3. AutoDL 现在不要做什么

1. 不要重新运行旧 round1/round2 MLP，也不要引用旧的约 `0.71` MLP 数字。
2. 不要继续调温度、门控、SupCon 权重、正则或堆叠新的分离损失来追求几个百分点。
3. 不要用测试集选择 epoch、超参或数据清洗规则。
4. 不要下载重复的 CodeT5、embedding 或完整 Droid 副本；先复用已有 checkpoint、tokenizer 和 manifest。
5. 不要把 task-held-out 结果写成跨文件或跨 generator 泛化结果。

## 4. 服务器端下一轮只做三件事

### A. 完成捷径单变量诊断

目标是回答：性能究竟来自哪一类可见线索，而不是再做一个综合清洗版本。保持现有 task split、训练集拟合预处理和配对 bootstrap，分别做：

- 仅去标识符；
- 仅去字符串/字符常量；
- 仅去注释；
- 仅去空白/格式；
- 可选的两两组合，只有在前四项结果明显后才运行。

每个版本必须保存：训练集拟合的变换、dev 曲线、best epoch、test predictions、seed、split manifest 和 408 个测试任务的身份字段。不要把 test 分布用于估计替换表、分位点或标准化参数。所有版本采用相同 epoch/early-stopping 规则；如果某个版本暂时只能固定 5 epoch，必须在报告中明确它与其他版本不完全同协议。

建议优先使用 `d-det/scripts/dcan_round3_shortcut_fix.py` 的数据读取和输出格式，另写最小的单变量入口，不要改动主模型。

### B. 只在必要时延长 LoRA 预算

如果 A 显示主要问题来自可解释的输入捷径，再使用已有 `best_lora.pt` 做预算扩展；否则先不要训练新模型。保持相同 encoder、tokenizer、head、采样和 split，只改变最大 epoch/early stopping，并保存每个 seed 的 dev 曲线。测试集只在模型冻结后使用一次。这个实验的目的只是确认是否存在明显欠拟合，不能用来制造新的方法主张。

### C. 方法冻结后做一次外部泛化审计

只有在 A/B 结束并冻结预处理、模型和超参后，才运行一组最小外部评估：

- STACAD：一次 file-held-out；
- Droid：一次 generator-held-out；
- 3 个 seed，完全相同的表示和评估脚本；
- 同时报告 TF-IDF、head-only、LoRA/最终模型。

Droid 只用于检验跨 generator 泛化，不要把其中不存在的语义正对解释成已有监督信号。外部结果必须单独标注为诊断性证据，不能和当前 408-task task-held-out 分数混成一个主表。

## 5. 资源和安全约束

- 只使用 AutoDL 服务器执行训练；本机不再生成大数据。
- 服务器上将 CPU 线程固定为 2（例如 `OMP_NUM_THREADS=2`、`MKL_NUM_THREADS=2`），按显存情况使用单卡和小 batch。
- 优先复用已有缓存与 checkpoint；每一步先检查剩余磁盘空间。
- 每个实验目录写入 `config.json`、`env.json`、`split_manifest.json`、`hashes.json`，并保存日志和结果摘要。
- 任何中断都保留已有产物，不覆盖旧结果；新实验使用新目录名。

## 6. 论文决策门槛

只有同时满足以下条件，才考虑把新方法写成主方法贡献：

1. 在当前 task-held-out 上超过 TF-IDF `0.7639`，而不是只超过 head-only；
2. 在 STACAD file-held-out 与 Droid generator-held-out 上没有明显崩溃；
3. 3 个 seed 的配对区间与方差支持结论；
4. 预处理、epoch 和 checkpoint 选择全部独立于测试集；
5. 能用单变量捷径诊断解释性能变化。

若不能满足，论文主线应保持为“后训练差异驱动的来源可读性分析、闭集归因与跨 generator 迁移审计”，而不是声称已经得到 SOTA 家族归因器。

## 7. 推荐的执行顺序

```text
先读 round3 报告和 artifacts
  -> 单变量捷径诊断
  -> 判断是否需要延长已有 LoRA
  -> 冻结方法和预处理
  -> STACAD file-held-out + Droid generator-held-out
  -> 统一表格、配对 bootstrap、论文结论
```

本指导只面向服务器端 AI 的执行；任何结果都必须回写到新的 round4 目录，并在报告开头注明使用的是 bdd569 修正版协议。
