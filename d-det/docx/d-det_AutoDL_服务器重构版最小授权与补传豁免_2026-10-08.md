# AutoDL 服务器重构版最小授权与补传豁免

日期：2026-10-08  
对应执行前闸门：`dccd1d2`  
适用范围：公开 BigCodeBench full/instruct 的系列内变体迁移 train/dev 阶段。

## 1. 结论

`dccd1d2` 的 C1–C8 全部通过，说明当前服务器上的数据、task split、heldout 排除、协议、P0、seed 和读取白名单已经满足执行前的科学条件。

16 个原件的逐文件核对属于 provenance 复现任务，不应继续阻塞当前 train/dev。服务器重构版必须明确标注：

```text
source_status = server_reconstruction_only
original_bundle_verified = false
claims_of_byte_identity = forbidden
```

这允许运行系列内迁移的 train/dev 阶段，但不允许把结果写成“逐字节复现本机交接包”。

## 2. 当前最小必要证据

### 执行必须保留

- `protocol_audit.json`：C1–C8 全部通过；
- `variant_transfer_registration.json`；
- `r1_negative_set_amendment.json`；
- `heldout_order_p0_seeds.json`；
- 11 个 full/instruct 成员的 task 支持索引；
- 4 份官方系列文档、`official_documentation_sources.json`；
- `split_index`、配置 hash、开关、命令、日志和 SHA256SUMS。

这些文件已经足以定义当前 train/dev 实验总体、样本、heldout 顺序、读出、P0 和统计方法。

### 暂不阻塞执行

以下文件对当前 train/dev 不是计算依赖，保留在 provenance 待补目录即可：

- `member_protocol_and_series.jsonl`：其成员与协议已由 C1/C2 和 11 折矩阵覆盖；
- `proposed_family_checkpoint_protocol.json`：当前不下载 checkpoint、不做 checkpoint 字节验证；
- `source_snapshot/*`：用于历史审计和复现，不参与特征或标签计算；
- 与已固定官方文档内容重复的副本。

原始 `family_series_admission.json` 若暂缺，可使用 `server_rebuild` 版本，但所有报告必须显示 `server_reconstruction_only`，并保存重构哈希 `c079abea...`；不得填成期望的 `cc4a7785...`。

## 3. 原件到达后的独立任务

原件到达后继续做 16/16 逐文件核对，不覆盖重构目录：

1. 计算 observed SHA-256；
2. 与 `expected_followup_manifest_via_user.txt` 对照；
3. 写入 `provenance_resolution.json`；
4. 若全部通过，将后续报告的 source 状态升级为 `provided_original_verified`；
5. 若仍有差异，保留两套目录并记录差异，不回写历史结果。

这项任务不会重写已经冻结的 split、seed、训练配置或 train/dev 预测。

## 4. 建议的执行边界

在得到新的明确执行授权后，可以运行：

- 3 个官方系列的 R1 train/dev；
- R2 相似度/回归 train/dev；
- size/length-only baseline；
- 原注册负集与已预声明 size-matched amendment 的 train/dev 敏感性分析；
- 双 P0：train-only fusion 和 equal ensemble。

仍然禁止：

- 读取 test；
- 读取项目旧 LCv2/AuthorBench/D1/N2 test；
- 选择性重写 heldout 顺序、负集或 seed；
- 下载权重或生成新样本；
- 将 `server_reconstruction_only` 结果写成原件复现或无污染确认性结论。

## 5. train/dev 产物要求

每个 series/fold 至少保存：

- execution config 与 amendment hash；
- train/dev 行数、task 数、heldout 排除证明；
- 预测、指标、task-cluster bootstrap 500 次 CI；
- size/length-only baseline；
- P0 fusion/equal；
- 命令、日志、代码版本和 SHA256SUMS；
- `source_status=server_reconstruction_only`。

只有这些产物冻结后，才重新申请一次性 test read。test 读取授权必须单独写明时间、数据 hash 和允许的输出字段。

## 6. 论文表述边界

允许：

> 在服务器重构并经过 C1–C8 审计的公开语料上，评估官方模型系列的未见尺寸变体迁移。

暂不允许：

- “完全复现本机原始交接包”；
- “unseen independent family”；
- “无污染确认性 benchmark”；
- “后训练因果效应”。
