# d-det AutoDL 执行记录：公开同任务完整语料接收 / 核验 / CodeContests 解析 / 预注册 split（2026-10-08）

依据《`d-det/docx/d-det_公开同任务完整语料_强数据路线与AutoDL执行指导_2026-10-07.md`》§5 四步；
产物 `d-det/artifacts/public_full_receive_2026-10-08/{audit,prereg,logs}` + `commands.txt` + `SHA256SUMS.txt`。
**未训练、未生成、未执行任何提交代码、未评测**（`code_executed=false` 全链保持）。

## 0 结论速览

- **接收 ✅**：`public_same_task_full_2026-10-07.zip`（153 MiB）已在服务器；**实测 zip sha256 `e98f221a…` 与指导标注的 `b8511ee6…` 不一致**——包内部 `SHA256SUMS.txt` **10/10 通过**、`records_sha256` 与包内 `full_integrity_check.json` **逐位一致**，按内部一致性接收，差异如实登记（`audit/received_files_sha256.json`）。若指导端希望以 `b8511ee6` 为准，请重发对应版本并说明。
- **核验 ✅**：官方 `scripts/verify_public_task_full.py` 重跑 6.7s：evalplus 8,778 / bigcodebench 287,476 行；399 / 1,140 task；**0 duplicate/malformed**；records sha `d786667a…` 与包内一致。
- **unit 口径澄清 ⚠️**：adjudication 文件是 **349 行 → 去重后 306 个精确 generator unit**（43 行为同 unit 的 asset/member 行）；records 原样字段计数同为 306，逐一匹配 349/349。**研究口径 = 306**（指导中“349 个精确观察单元”应按 306 更正记账）。
- **CodeContests ✅（自建官方规范解析器）**：PyPI 无 `riegeli` 包、`tf.data` 无 Riegeli 接口、bazel 工具链过重 → 按官方 *riegeli records file format* 规范写了最小解析器（64KiB 块头 + 0x7a zstd 块 + proto 字段解析），**一次跑通**：valid 117 题 / test 165 题；字段与三文件哈希索引已落盘；**未执行任何代码**。
- **预注册 split ✅（冻结、未执行）**：主协议 BigCodeBench full/instruct = **118 units × 1140 task = 134,528 行**；task 级 70/15/15 = **798/171/171**；unit-heldout 5 折（31/23/23/21/20）；家族层仍 `family_is_confirmed=0/306` → family-variant 折 **blocked**。

## 1 接收与核验（第一步/第二步）

| 项 | 结果 |
|---|---|
| zip sha256 | `e98f221a2b0b88a377fe4390222f30006e18eca00433312106514f675967df82`（≠ 指导参考值，已登记） |
| 内部 SHA256SUMS | **10/10 OK**（Python 逐行复核；用 Python zipfile 解压） |
| records 行数 | evalplus 8,778；bigcodebench 287,476（= manifest） |
| task 数 | evalplus 399；bigcodebench 1,140 |
| duplicate/malformed | 0（官方脚本口径）；另发现 **21 行 unit-task 内重复输出**（统计时按 unit-task-sha 去重，已登记） |
| 单位 | 306 精确 unit（adj 349 行去重）；差异仅 evalplus 22 单元缺 backend/temperature 字段 |
| 协议轴 | full 1140 / hard 148（全 ⊂ full）；instruct 1140 / complete 1140（同 task 集） |
| 主协议支持度 | 每 unit 覆盖全部 1140 task（min=median=max=1140）→ 折设计充分 |
| family 证据 | name_inferred 233 / provider_or_model 37 / unresolved 79；**confirmed = 0** |

## 2 CodeContests 解析（第三步，自建最小读取器）

- 工具链判定：`pip install riegeli`（aliyun/PyPI 均无此包）；`tf.data.experimental.RiegeliDataset` 404；官方工具需 bazel（重）→ 按规范自写：
  `scripts/cc_riegeli_extract.py`（块头 24B / chunk 头 40B / gather 跳过 64KiB 块头 / 0x72 简单块 + zstd 解压 / 字段按 `contest_problem.proto`（name=1, description=2, tests=4/5/18, source=6, difficulty=7, solutions=8, cf_*=10/12/14/15, incorrect_solutions=19））
- 提取结果（索引：`audit/cc_index_{valid,test}.jsonl` + `audit/code_contests_index.json`）：
  - valid 117 题 / test 165 题；**valid∩test 题面描述重叠 = 0**；
  - 来源：Codeforces 为主（AIZU/AtCoder 字段在枚举表中，实际分布见 JSON）；
  - 人类提交：valid 正确 29,863（CPP 12,734 / JAVA 8,251 / PYTHON3 8,396 / PYTHON2 482）、错误 28,625；test 计数同表；
  - 测试存在性、cf_rating、cf_tags 计数已入索引。
- 说明：仅解析字段与哈希，**未执行任何提交代码**。

## 3 预注册 split（第四步，`prereg/`）

| 方案 | 目标总体 | generator 数 | task 数 | 每 task 输出 | family 证据 | 污染 | 统计单位 |
|---|---|---|---|---|---|---|---|
| **主协议 H1/H2a**：BCC full/instruct | 全量成员 | **118 unit** | **1,140** | 1 | 未确认（0/306） | unknown | task cluster |
| 协议控制：BCC full/complete | 同 task | 123 | 1,140 | 1 | 未确认 | unknown | task |
| 难题控制：BCC hard/instruct | hard 子集 | 86 | 148 | 1 | 未确认 | unknown | task |
| 次级切片：EvalPlus MBPP+ | 22 模型包 | 22 | 399 | 1 | 未确认 | unknown | task |
| 人类/正确性控制：CodeContests | 竞赛题 | —（人提交） | 117+165 | 多语言 | — | unknown | 题目 |

- task split（seed 20261007，按 `sha256(seed|task)`）：**798/171/171**；每 split task 列表 sha256 + 逐 task 索引（`split_index.csv`）已冻结；
- unit-heldout：5 折（family_label 分组轮转）31/23/23/21/20（`unit_folds.json`）；每折 test = 折内 unit × 全部 171 test task；
- family-variant-heldout：**blocked**（需先有 family 证据：模型卡 / lineage / 官方仓库）；
- 8 个 unit-task 为多行输出（134,528 = 118×1140 + 8，已登记）；
- 污染登记：公开 benchmark 可能进预训练 → 定位 = **外部验证/方法诊断**，进项目主表前须完成污染与历史使用登记。

## 4 边界与下一步

- 仍保持：`training_allowed=false`、`generation_allowed=false`；本轮未做任何模型调用与训练。
- 允许的下一步（待指导端确认后执行）：在**冻结表示**上做 task-cluster 诊断（字符/词法/AST/CodeT5 表示 × 306 精确 unit + family 候选）；不重拟合项目旧 test 模型、不把外部结果回写 LCv2 H2 主表。
- 待办（阻塞项）：family 证据（模型卡/lineage 登记）、污染/历史使用登记、双 P0（fusion+eq）在新 train/dev 冻结。

## 5 交付与复现

- `audit/`：`received_files_sha256.json`（11 文件哈希 + zip 差异登记）、`full_integrity_check_from_pack.json`、`full_adjudication_audit.json`、`unit_coverage_matrix.csv`、`full_adjudication.md`、`cc_index_{valid,test}.jsonl`、`code_contests_index.json`；
- `prereg/`：`split_plan.json`、`split_index.csv`（1140 行）、`unit_folds.json`、`public_prereg.md`；
- `logs/`、`commands.txt`、`git_head/status`、`SHA256SUMS.txt`（18 文件）；脚本：`scripts/{verify_public_task_full,public_full_adjudication_audit,cc_riegeli_extract,public_prereg_split}.py`；
- 大文件（zip/records.jsonl/riegeli）已在 `.gitignore`，哈希登记在上表。

---
*生成时间：2026-10-08；记录人：服务器执行 AI。*
