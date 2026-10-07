# N1：候选 task-aware 支持矩阵（2026-10-07）

主矩阵 = `h2_authorbench_dcan`（原始 9498 行 → 剔除 66 行 → **保留 9432 行**/2714 tasks / 6 families / 8 generators / C）；附属 = `h2_llm_codegen_v2`（不合并：标签空间与任务域不同）。

## 0 行级剔除（预声明规则）

- 规则：任一归一化层级下**跨 split** 的重复组，整组行剔除（不做任意“保留一侧”）；同 split 重复组保留。
- 实测：exact 跨 split 0 组、norm_ws 0 组、**norm_lex（去 C 注释/字符串）25 组**→ 共剔除 66 行（涉及 51 个 task；剔除行明细见 `n1_manifest_excluded.jsonl` 与审计 JSON 的 `cross_group_detail`）。
- `source_sha256`：9498 行全部唯一、跨 split 0 组，无需处理。

## 1 硬约束核验（剔除后）

| 约束 | 通过 | 证据 |
|---|---|---|
| 1 task 单 split | True | crossing tasks = 0 |
| 2 hash 不跨 split | True | 剔除后 exact/ws/lex 跨 split = 0/0/0 |
| 3 train/dev 每族支持 | True | train ≥100 行/≥30 task；dev ≥20 行/≥10 task（§4 表） |
| 4 正式 H2 family ≥3 generator | True | 仅 openai；其余 5 族单 generator → H1/诊断 |
| 5 占比公开 | True | §2/§3 与本文件 |
| 6 test generator held-out 角色 | True | openai 3 折（D0 admitted）；单 generator 族不可定义 |
| 7 google/mistral llm 折 diagnostic-only | True | D0 记录；未伪造正例 |
| 8 原始/去重计数与 hash | True | matrix.exclusions/dedup_counts + 审计 JSON |

## 2 family×generator（行数 / task 数，剔除后）

| family | generator | role | rows | tasks | rows tr/dv/te | tasks tr/dv/te |
|---|---|---|---|---|---|---|
| claude | claude-3.5-haiku | h1_only | 1176 | 1176 | train:812/dev:190/test:174 | train:812/dev:190/test:174 |
| deepseek | deepseek-chat | h1_only | 1223 | 1223 | train:850/dev:189/test:184 | train:850/dev:189/test:184 |
| gemini | gemini-2.5-flash-preview-05-20 | h1_only | 1176 | 1176 | train:801/dev:182/test:193 | train:801/dev:182/test:193 |
| llama | llama-3.3-70b-instruct | h1_only | 1284 | 1284 | train:892/dev:192/test:200 | train:892/dev:192/test:200 |
| openai | gpt-4.1 | h2_candidate | 1127 | 1127 | train:782/dev:165/test:180 | train:782/dev:165/test:180 |
| openai | gpt-4o | h2_candidate | 1060 | 1060 | train:727/dev:172/test:161 | train:727/dev:172/test:161 |
| openai | gpt-4o-mini | h2_candidate | 1125 | 1125 | train:783/dev:181/test:161 | train:783/dev:181/test:161 |
| qwen | qwen-2.5-72b-instruct | h1_only | 1261 | 1261 | train:874/dev:196/test:191 | train:874/dev:196/test:191 |

## 3 任务组成（剔除后）

- family coverage（每 task 含族数）：{1:7, 2:1416, 3:573, 4:249, 5:189, 6:280}
- task-size（每 task 行数）：{1:6, 2:1181, 3:654, 4:280, 5:150, 6:104, 7:104, 8:235}
- 六族齐全子集（N2 候选）：280 tasks；行 1424/348/404（tr/dv/te）；tasks 184/45/51

## 4 每正式 family 的 generator/task 支持（回传 §6.4）

| family | #gen | tasks per generator | role |
|---|---|---|---|
| claude | 1 | {"claude-3.5-haiku": 1176} | h1_only |
| deepseek | 1 | {"deepseek-chat": 1223} | h1_only |
| gemini | 1 | {"gemini-2.5-flash-preview-05-20": 1176} | h1_only |
| llama | 1 | {"llama-3.3-70b-instruct": 1284} | h1_only |
| openai | 3 | {"gpt-4.1": 1127, "gpt-4o": 1060, "gpt-4o-mini": 1125} | h2_candidate |
| qwen | 1 | {"qwen-2.5-72b-instruct": 1261} | h1_only |

→ 结论：**矩阵内仅 `openai` 满足「每正式 family ≥3 个可靠 generator + admitted heldout 折」，可作 H2 候选单元（holdout=gpt-4.1 / gpt-4o / gpt-4o-mini 三折）；claude/deepseek/gemini/llama/qwen 为单 generator，只能作 H1/诊断数据。**

## 5 llm_codegen_v2 附属支持（不合并）

| family | #gen | generators(tasks) | admitted folds |
|---|---|---|---|
| google | 2 | Gemni-1.5-pro(168t), codegemma(168t) | — |
| ibm | 1 | granite-code:3B(168t) | — |
| meta | 3 | codellama(168t), llama2(168t), llama3(168t) | llm_codegen_v2:meta:holdout=codellama, llm_codegen_v2:meta:holdout=llama2, llm_codegen_v2:meta:holdout=llama3 |
| microsoft | 1 | phi3(168t) | — |
| mistral | 2 | codestral(168t), mistral(168t) | — |

（其 `meta` 族 3 generator 且 D0 有 3 个 admitted 折，可作为第二个 H2 候选单元评估；跨源合并属于新的构造决策，本轮到 N1 为止。）

## 6 N2 决策

- 硬约束 c1–c4 全部满足；候选子集 = 六族齐全 280 tasks（test 51 tasks / 404 行，≥协议下限）→ **允许且已预注册 N2**（`n2_protocol.json`）。
- N2 只作“数据构造是否修复支持不足”的最小复核；不启动任何 H2/H3 训练。

