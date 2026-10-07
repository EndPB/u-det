# 公开同任务完整语料：强数据路线与 AutoDL 执行指导

日期：2026-10-07  
状态：完整公共生成语料已在本机物化；竞赛人类控制原始文件已下载；尚未启动新训练或项目 test 评测。

## 1. 对上一版切片的纠正

上一版 36 MiB 切片确实不足以解决原问题。它只保留少数规模变体，无法支撑稳定的 generator-heldout、family-heldout 和人类控制。因此本轮不再以轻量为目标，而是保留完整公开成员，并把“精确生成单元”和“推定 family”分开。

## 2. 完整数据规模

目录：`d-det/data/public_same_task_full_2026-10-07/`  
压缩包：`d-det/data/public_same_task_full_2026-10-07.zip`  
压缩包 SHA-256：`b8511ee670ee99404f646840210b74284d20076d726ee57eb2a221d88cd2a417`

| 来源 | 内容 | 规模 |
|---|---|---:|
| EvalPlus v0.2.0 | 22 个公开模型包，MBPP 同题输出 | 8,778 行 / 399 task |
| BigCodeBench v0.2.4 | 全部公开 JSONL 成员，保留 full/hard、instruct/complete | 287,476 行 |
| BigCodeBench full/instruct | 同题多模型主协议 | 134,528 行 |
| BigCodeBench full/complete | 独立提示协议控制 | 140,220 行 |
| BigCodeBench hard/instruct | 难题子集控制 | 12,728 行 |
| CodeContests | validation/test 原始竞赛题与人类正确/错误提交容器 | 34.0 MB，尚未解析 |

完整 JSONL 约 699 MB；连同 CodeContests 原始控制文件约 749 MB。压缩包约 153 MB，已经不再是轻量演示包。

完整语料的 349 个精确观察单元定义为：

```text
source + model_id + generation_mode + subset + backend + temperature
```

这是 split 的最小 generator 单元。当前从模型名得到的 family 只是审核提示：79 个单元尚未解析出稳定 family，所有 `family_is_confirmed` 均为 `false`。这一步是刻意保守，避免再次把厂商、规模、协议或发布版本混成一个 family。

## 3. 为什么这条路线更强

它提供了三个层次的可检验对象：

1. **精确生成单元**：同一 task 由大量不同模型、后端和协议生成，可测闭集 generator attribution。
2. **谱系内迁移**：在 family 标签经人工/模型卡证实后，做规模、版本和后训练变体迁移；这些不能自动叫 unseen generator。
3. **人类/非 LLM 控制**：CodeContests validation/test 含竞赛题、人类正确与错误提交，CodeContests 官方说明也包含题目、测试和多语言正确/错误解；解析后可建立 Human/AI、正确/错误和语言控制。

因此它可以把原来的单一 H2 拆成：

- H1：同题多生成单元是否可读；
- H2a：精确 generator-heldout 是否保留关系；
- H2b：已证实 family 内的 variant transfer；
- H3：加入人类/错误提交后，family attribution 是否仍独立于 Human/AI 和 correctness。

## 4. 必须坚持的分层设计

### 4.1 不按输出行随机切分

所有相同 `task_id` 必须落在同一个 split。BigCodeBench 的 full/hard 同题重叠也必须按 task 组处理。统计单位是 task cluster，不是输出行。

### 4.2 不把 family 推定当标签真值

主表先使用精确 generator unit；family 只能在模型卡、官方仓库、checkpoint lineage 和生成文件证据齐全后进入确认性表。名称规则只用于候选分组，不用于证明 unseen。

### 4.3 协议轴必须独立

`complete` 与 `instruct` 不能混合；`full` 与 `hard` 不能混合；backend、temperature、sanitized/calibrated 状态必须保留。若只做一个主协议，预先冻结为 BigCodeBench `full/instruct`，其他协议作为控制。

### 4.4 外部数据和项目主赛道分开

公开 benchmark 可能已经被模型预训练看到。即使 task 对本项目没有历史使用，也不能直接写成无污染 confirmatory test。它首先是外部验证与方法诊断；只有完成污染/历史使用登记后，才决定是否进入主表。

## 5. AutoDL 侧下一步

### 第一步：接收完整包

```bash
unzip public_same_task_full_2026-10-07.zip -d d-det/data/
sha256sum public_same_task_full_2026-10-07.zip
sha256sum d-det/data/public_same_task_full_2026-10-07/SHA256SUMS.txt
```

### 第二步：只读核验

```bash
python scripts/adjudicate_public_model_units.py
python scripts/verify_public_task_slices.py
```

完整语料核验需要新增一个只读脚本，检查：

- `records.jsonl` 总行数和源包哈希；
- 每个精确 generator unit 的 task 支持；
- full/hard、instruct/complete 的交集；
- 任务级重复和成员级重复；
- family 候选的证据状态。

### 第三步：解析 CodeContests 控制

CodeContests 原始文件是官方 Riegeli 格式。AutoDL 侧使用官方仓库工具或 Linux 依赖解析 validation/test，提取：题目 ID、题面哈希、语言、正确/错误提交、测试存在性和来源站点。不要执行提交代码；先只做字段和哈希索引。

### 第四步：生成预注册 split

先冻结 task split，再指定 generator-heldout、family-variant-heldout 和 human-control 方案。每个方案都必须给出：目标总体、纳入/排除规则、任务数、generator 数、每 task 输出数、family 证据状态、污染风险和统计单位。

## 6. 允许的第一轮研究分析

只做冻结表示上的 task-cluster 诊断：字符/词法/AST/CodeT5 表示、family 候选和 exact generator unit。报告 macro-F1、balanced accuracy、per-task bootstrap CI、任务规模分层和 family/variant 支持。暂不重拟合项目旧 test 上的模型，暂不把任何外部结果回写为 LCv2 H2 主表。

如果 family 证据不足，论文结果应写成“公开生成单元归因”和“谱系候选迁移”，不能写成普遍后训练 family 几何。

## 7. 相关公开来源

- [EvalPlus v0.2.0 release assets](https://github.com/evalplus/evalplus/releases/expanded_assets/v0.2.0)
- [BigCodeBench 官方样本字段与协议](https://github.com/bigcode-project/bigcodebench/blob/main/ADVANCED_USAGE.md)
- [CodeContests 官方仓库与下载格式](https://github.com/google-deepmind/code_contests)
- [Project CodeNet 官方数据说明](https://github.com/IBM/Project_CodeNet)
- [APPS](https://github.com/hendrycks/apps)
- [LiveCodeBench](https://github.com/LiveCodeBench/LiveCodeBench)
