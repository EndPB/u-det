# AutoDL 下一阶段指导：公开完整语料的系列证据与变体迁移预注册

日期：2026-10-08  
对应接收提交：`e38715d`  
状态：只读审计已完成；本文件只授权预注册和 train/dev 方案准备，不自动授权训练、生成或读取 test。

## 1. 本轮接收结论

完整包以内部一致性接收。实际 ZIP SHA-256 为：

```text
e98f221a2b0b88a377fe4390222f30006e18eca00433312106514f675967df82
```

指导草稿中的 `b8511ee6…` 是旧版本标注，不是当前包的实际哈希。包内 `SHA256SUMS.txt` 10/10 通过，`records_sha256` 与 integrity 记录逐位一致，因此不重传、不替换当前包。

研究口径必须区分三个数量：

- **349 条 adjudication 行**：发布成员、协议或资产记录的审核行数；
- **306 个去重 exact unit**：按 `source + model_id + generation_mode + subset + backend + temperature` 去重后的研究单位；
- **21 个 unit-task 重复输出行**：统计和训练前按 `unit-task-sha` 去重，不能误写成 21 个额外模型。

官方脚本重跑得到 EvalPlus 8,778 行 / 399 task，BigCodeBench 287,476 行 / 1,140 task，均无 malformed 或重复 task 记录。主协议固定为 BigCodeBench `full/instruct`：118 个 unit、1,140 个 task、每 task 每 unit 一条输出；任务 split 已冻结为 train/dev/test = 798/171/171，seed `20261007`，split hash 见 `d-det/artifacts/public_full_receive_2026-10-08/prereg/split_plan.json`。

CodeContests 已解析为 valid 117 题、test 165 题；人类提交统计与 BigCodeBench 不同题，必须作为独立外部控制，不能直接拼成同题 Human/AI 主表。

## 2. 家族标签的正确升级方式

原始全局状态继续保留：`family_is_confirmed=false`。名称推定、厂商名称或文件夹路径不能单独证明 family，也不能证明独立 generator、后训练因果或无污染测试。

本轮新增的是一个更窄的标签：**model-series membership**。根据固定 commit 的官方资料，当前可进入“系列内变体迁移 pilot”的 3 个系列共 11 个 full/instruct 成员：

| series | 成员规模 | 允许的研究问题 |
|---|---:|---|
| CodeLlama-Instruct | 7B/13B/34B/70B | 同一官方系列内的 size/variant transfer |
| Qwen2.5-Coder-Instruct | 1.5B/7B/14B/32B | 同一官方系列内的 size/variant transfer |
| DeepSeek-Coder-v1-Instruct | 1.3B/6.7B/33B | 同一官方系列内的 size/variant transfer |

证据文件为 `d-det/artifacts/public_full_followup_2026-10-08/family_series_admission.json` 及其 `official_docs/`。这些官方资料支持系列成员关系；它们不支持“unseen independent family”“post-training causal effect”或“contamination-free confirmatory benchmark”。checkpoint 字节、历史生成参数和污染状态仍未验证。

对应官方资料：[CodeLlama model card](https://github.com/meta-llama/codellama/blob/main/MODEL_CARD.md)、[Qwen2.5-Coder repository](https://github.com/huggingface/Qwen2.5-Coder)、[DeepSeek-Coder repository](https://github.com/deepseek-ai/DeepSeek-Coder)。

## 3. AutoDL 只读复核

先在 AutoDL 项目根目录运行，不修改完整记录：

```bash
python scripts/verify_public_task_full.py
python scripts/adjudicate_public_model_units.py
python scripts/build_family_series_admission.py
```

`build_public_lineage_evidence.py` 只在本机需要刷新官方资料时运行；AutoDL 不必联网重抓，直接使用随交接包提供的 `public_full_followup_2026-10-08/official_docs/` 与来源 JSON。

需要回传：

1. 完整核验的行数、task 数、源包和 `records.jsonl` hash；
2. 306 distinct unit、349 adjudication 行、21 unit-task 重复的复核结果；
3. 三个系列各自的成员数、full/instruct 路径一致性、每 task 支持数；
4. `family_series_admission.json` 的 SHA-256 和 `git_head/status`；
5. 不读取任何项目旧 test，不生成新输出，不下载大模型权重。

## 4. 允许注册的 pilot：系列内留一尺寸变体

### 4.1 研究命题

只检验以下命题：

> 在固定 BigCodeBench `full/instruct` 任务切分和固定表示读出下，训练于某一官方 model-series 的若干已见尺寸后，能否在同系列的一个未见尺寸变体上保留可读的系列/来源信号。

这不是 unseen family，也不是后训练因果实验。输出标题建议固定为：`Official model-series variant transfer on task-heldout BigCodeBench`。

### 4.2 折设计

任务 split 不得重新抽样，继续使用 798/171/171。每个 series 预声明旋转的 heldout variant：

- 4 成员系列：四折，依次留出 7B、13B、34B、70B；Qwen 依次留出 1.5B、7B、14B、32B；
- 3 成员系列：三折，依次留出 1.3B、6.7B、33B；
- 每折的训练标签是“series 内已见 variant 的输出”，留出 variant 只在冻结后评估；
- 不把未见 variant 的 `model_id` 当作训练中不存在的闭集类别；普通 generator-ID 分类器不能用于这种折。

建议的可识别读出有两种，必须在 train/dev 之前冻结：

1. **同一 series 的二分类/排序读出**：训练目标是“该样本是否来自目标 series”，heldout variant 作为目标域；
2. **表示相似度或回归迁移**：用 train/dev 估计系列中心、尺度或成对关系，测试只报告 heldout variant 的距离/相关性。

不得在 test 后选择特征、阈值、seed、heldout variant 或折。若要报告 macro-F1，类别必须在训练折中存在；“未见 generator 作为新类别”的 macro-F1 不具备闭集定义，应改为 transfer score、AUROC、平均精度或预先定义的距离指标。

### 4.3 双 P0 与统计

在新数据上重新冻结两个 P0：

- `P0-fusion`：只用 train 拟合融合头，dev 只用于预声明选择；
- `P0-equal`：成员概率逐元素等权平均，类别顺序固定，无校准。

所有结果以 task cluster 为 bootstrap 单位，报告 heldout variant 的 AUROC/平均精度、任务宏平均、95% CI、每一折支持数和 train→heldout 的迁移差。不要把 500 次重采样频率写成后验概率。

## 5. 明确禁止的错误路线

- 不把 349 行写成 349 个独立模型；
- 不把 `family_is_confirmed=false` 改成 true；系列证据和 family 因果证据分栏保存；
- 不用 unit-heldout 折直接训练“未见模型类别”分类器；
- 不把 CodeContests 人类提交和 BigCodeBench AI 输出按行合并；没有同题对应关系时只能做外部控制；
- 不为本 pilot 下载权重或调用 API；生成和 training 仍为 `false`，除非后续单独授权；
- 不读取已有 LCv2、AuthorBench、D1/N2 或其他项目 test；旧 Path A 继续是 `exploratory_reuse_only`，不进 H2 主表；
- 不把公开 benchmark 的预训练污染风险写成已解决。污染状态只能登记为 `unknown`，除非有直接证据。

## 6. 执行闸门

当前状态：`read_only_ready / training_allowed=false / generation_allowed=false / test_read_allowed=false`。

AutoDL 只读复核通过后，下一份执行授权才可开始 train/dev 方案。该授权至少要固定：series、heldout variant 顺序、读出类型、seed、P0 定义、指标、task-cluster CI 和 test 单读时间。只有 train/dev 产物、配置、哈希和报告全部冻结后，才允许一次性读取 test。

## 7. 交付物

- 接收与预注册：`d-det/artifacts/public_full_receive_2026-10-08/`
- 系列证据与候选成员：`d-det/artifacts/public_full_followup_2026-10-08/`
- 完整核验脚本：`scripts/verify_public_task_full.py`
- 系列证据脚本：`scripts/build_public_lineage_evidence.py`、`scripts/build_family_series_admission.py`
- 本指导：`d-det/docx/d-det_AutoDL_公开语料系列证据与变体迁移指导_2026-10-08.md`

本文件可以在每个闸门完成后追加版本号、执行提交、配置 hash、split hash、test 读取时间和允许/禁止结论；不覆盖旧记录。
