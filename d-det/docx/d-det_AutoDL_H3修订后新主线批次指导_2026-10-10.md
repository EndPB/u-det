# AutoDL H3 修订后新主线批次指导

版本：2026-10-10  
适用状态：C0–C3 已关闭；H3 本机修订 v1 后仍为 revise_data；本文件是下一次大显存批次的预注册执行指导。
主线边界：本机完成数据修订和低资源准备；AutoDL 只执行通过数据闸门后的完整 GPU 批次。

## 1. 本批次要回答的问题

在同题 Human/AI、任务/项目隔离、generator-heldout 和 train-only 语义安全变体下，回答三件事：

1. detection head 是否能识别人类与 AI，而不是识别长度、词法或来源目录；
2. AI 子集的 observed source/generator head 是否保留可迁移信号；
3. 联合表示是否相对独立检测和来源基线产生稳定增量，并在变体上保持一致。

本批次不重跑已经关闭的 C0–C3，不通过增加容量、温度或损失项补救旧路线。

## 2. AutoDL 启动前必须由本机完成

本机直接完成以下工作并冻结输入包：

- 按 `label × generator × language × length_bin` 做长度匹配或分桶平衡；
- 完成七语言 parser、AST/接口安全变体和 parent-child 索引；
- 对两组跨任务骨架重复做排除或人工裁定；
- 重跑 metadata/source/transform/length/lexical/AST probes；
- 固定 task/project/generator/solution cluster split、行映射和所有输入 SHA256；
- 只在 train 生成变体，dev/test 保持原视图；test 只封存哈希，不读正文。

若 length、AST 或 transform probe 仍触发预注册停止规则，不生成 AutoDL GPU 批次。

## 3. 预注册批次矩阵

所有 job 使用同一数据包、encoder、batch、epoch、参数预算、折定义和 seed 集合。建议至少两个 generator-heldout 折、三个 seed；每个 job 单独目录，批次统一汇总。

| job | 训练目标 | 输入/约束 |
|---|---|---|
| `lexical_control` | char/word TF-IDF 强控制 | 只作控制，不宣称 invariant signal |
| `detection_only` | Human/AI detection | 同题 Human/AI；不使用 AI source 标签 |
| `source_only` | AI observed source/generator | 只在 AI 行计算，报告 generator-heldout |
| `joint` | detection + source | 共享 encoder，检测与来源 head 分开 |
| `joint_invariance` | joint + 变体一致性 | comment/whitespace/rename 只来自 train parent |
| `joint_adversary`（可选） | joint + nuisance adversary | 仅当 `lambda_adv`、nuisance 标签和出口规则预注册 |

`lexical_control` 必须作为强控制保留。H3 主模型只有在 heldout 上相对该控制有预注册增量，才允许进入主表。

## 4. 推荐损失与读出

令共享编码器为 (z_\theta(c))，检测 head 为 (D_\phi)，source head 为 (F_\psi)：

\[
L_D=\operatorname{BCE}(D_\phi(z),y_D),\qquad
L_F=\operatorname{CE}(F_\psi(z),y_F)\quad\text{仅对 AI 行计算}.
\]

联合配置使用：

\[
L_{joint}=L_D+\lambda_F L_F+\lambda_{inv}L_{inv}.
\]

一致性项只在同一 parent 的原视图/安全变体之间计算；不得把不同 task 的代码强行组成正对。若启用 nuisance adversary，必须在 manifest 中冻结 nuisance 字段和 λ，不得看 test 后调整。

## 5. 执行顺序

1. 读取 `batch_manifest.json`，核对 `data_sha256`、`code_commit`、split、job 数量和 `test_read=false`。
2. 做一次轻量环境和输入完整性检查；不重复本机已经完成的数据构建、parser、TF-IDF 或全量 probes。
3. 按 manifest 一次启动全部 job；允许并行，但每个 job 使用独立输出目录和固定 seed。
4. 每个 job 保存 train/dev 指标、task-cluster CI、generator-heldout 结果、变体一致性、梯度/训练日志和失败原因。
5. 全部 job 完成后生成统一 `metrics.json`、`report.md`、`predictions-or-score-digests/` 和 `SHA256SUMS.txt`。
6. 任一 job 失败只标记该 job；不得用临时调参替换预注册配置。需要改变假设时，结束本批次并开新批次。

## 6. 通过与停止规则

只有同时满足以下条件，H3 才能进入主表：

- detection 在 task-heldout 和 generator-heldout 均稳定；
- source head 在 AI 子集保留高于机会水平的迁移信号；
- joint 相对 `detection_only` 不显著下降，且相对 `source_only` 保留来源增量；
- joint/invariance 相对 `lexical_control` 有预注册增量；
- 原视图与变体的 detection/source 输出一致性通过；
- 三个 seed 和至少两个 generator-heldout 折方向一致；
- metadata/source/length/AST/transform probes 不再独立解释标签。

任一条件失败，判定为 `stop_or_revise_data` 或 `increment_not_observed`，不增加 backbone、温度、训练轮数或损失项来补救。

## 7. `batch_manifest.json` 最小字段

```json
{
  "data_sha256": "",
  "code_commit": "",
  "test_read": false,
  "generation": false,
  "weights_downloaded": false,
  "code_execution": false,
  "folds": [],
  "seeds": [],
  "jobs": [
    {"name": "lexical_control", "output_dir": "..."},
    {"name": "detection_only", "output_dir": "..."},
    {"name": "source_only", "output_dir": "..."},
    {"name": "joint", "output_dir": "..."},
    {"name": "joint_invariance", "output_dir": "..."}
  ]
}
```

## 8. 当前状态

本文件暂不触发 AutoDL 运行。本机 v1 的 train-balanced clean-dev length=.8231、lexical=.9084，尚未重新通过闸门；等本机 controlled-dev、view-probe 和 provenance 修订完成并明确通过后，把冻结数据包、manifest 和脚本一次性上传；AutoDL 按本文件完成整批执行并统一回传。服务器不接收零散小实验指令。

