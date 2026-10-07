# d-det AutoDL 执行记录：N0/N1/N2（D1 未过闸门后的数据构造审计）

2026-10-07 · 服务器端执行 · 依据《`d-det/docx/d-det_AutoDL_D1未过闸门_数据构造下一步指导_2026-10-07.md`》
运行锚点：HEAD `c2a2e93`（N0/N1/N2 运行前）；产物目录：`d-det/artifacts/stage_d_data_construction_2026-10-07/`

## 0 结论速览

- **N0 ✅（只读复盘）**：D1 失败来源分层诊断完成；无 exact/norm_ws 跨 split 重复，但发现
  **norm_lex（去注释/字符串）25 组跨 split 骨架重复**（真实数据卫生问题）。
- **N1 ✅（候选支持矩阵）**：行级预声明剔除 66 行（跨 split 重复组整组剔除）→ 保留 9,432 行；
  剔除后 exact/ws/lex 跨 split = 0/0/0；硬约束 c1–c4 全过。**≥3 generator 的正式 H2 候选
  仅有 `openai`（3 个 generator + 3 个 admitted heldout 折）**，其余 5 族单 generator → H1/诊断；
  `llm_codegen_v2` 的 `meta` 族（3 generator + 3 admitted 折）为第二候选单元（附属表，未合并）。
- **N2 ❌（预注册门未通过）**：候选 = 六族齐全 280 tasks（test 51 tasks / 404 行）上整体复算 P0 与内容视图：
  cond1 ✅、cond2 ✅（但仅由**转导视图** codet5_small_centered .6538 达成）、**cond3 方向稳定 ❌**
  （tfidf_word 3 seeds Δ = [−1.36, +2.18, −0.18]pt 符号不一致）→ **判定未通过，归档为数据限制/负结果；
  不重议 H2、不启动 H2/H3。**

## 1 N0：D1 失败来源复盘（`n0_diagnosis/`，46.8s，只读）

输入：D1 `metrics.json`/`predictions.npz` + `core.jsonl`（sha256 已记录）。

- **分层诊断**（test；tfidf_word / p0_fusion_lr / codet5_centered；500× task-cluster CI）：
  - family×generator（8 行）：openai 内 3 个 generator 跨度大（如 fusion 行 gpt-4o ≈.60 vs gpt-4.1 ≈.94）；
  - task-size 桶（1/2–5/6+）：`6+` 桶 tfidf_word F1 `.7644`（对照 `2-5`≈`.86`）——保留为分层诊断；
  - family×task-size（18 格）、generator×task-size（24 格）全量见 `n0_strata.json`。
- **结构**：每 family 的 generator 数：openai=3，其余=1；每 generator 的 task 数 1,062–1,288；
  任务组成：仅 285/2,715（10.5%）task 含全部 6 族。
- **错误审计**：全库 exact 跨 split 0 组、norm_ws 0 组、**norm_lex 25 组（涉及 51 task/66 行）**；
  同 task 跨 split = 0；错误样本中 exact/norm_ws 与其它 split 重复 = 0。
- **失败来源判断**：① 组成/支持不均衡证据成立（竞争集随 task 变化、`6+` 更难）；
  ② generator 异质（5 族单 generator，family 数字绑定单个指纹）；③ 无 exact/ws 级泄漏伪影；
  ④ “数据是否缺跨 generator 信号”需 N1/N2 回答。

## 2 N1：候选支持矩阵（`n1_candidate_matrix/`，8.0s）

### 2.1 行级剔除（预声明规则）

- 规则：任一归一化层级下**跨 split** 的重复组**整组剔除**（不做任意“保留一侧”）；同 split 重复组保留。
- 结果：exact 0 组 / norm_ws 0 组 / **norm_lex 25 组** → 剔除 **66 行**（涉及 51 task；
  明细 `n1_manifest_excluded.jsonl` + 审计 JSON `cross_group_detail`）。
- 剔除后：`exact/ws/lex` 跨 split = **0/0/0**；`source_sha256` 全库唯一（0 组跨 split）。
- 数据：9,498 → **9,432 行**（train 6,521 / dev 1,467 / test 1,444）。

### 2.2 硬约束核验（指导 §3，全过）

| 约束 | 结果 | 证据 |
|---|---|---|
| 1 task 单 split | ✅ | crossing tasks = 0 |
| 2 hash 不跨 split | ✅ | 剔除后 0/0/0（原始 norm_lex 25 组已剔除） |
| 3 train/dev 每族支持 | ✅ | 单 generator 族：train ≥801 行/≥801 task，dev ≥182 行/≥182 task；openai：train 2,292 行/1,451 task，dev 518 行/315 task（各族全过） |
| 4 正式 H2 family ≥3 generator | ✅（有人满足） | 仅 openai；其余 5 族单 generator |
| 5 占比公开 | ✅ | balance report §2/§3 |
| 6 test generator held-out 角色 | ✅ | openai 3 折（D0 admitted）；单 generator 族不可定义 |
| 7 google/mistral llm 折 diagnostic-only | ✅ | 记录在案，未伪造正例 |
| 8 原始/去重计数与 hash | ✅ | exclusions/dedup_counts/审计 JSON |

### 2.3 每 family 的 generator/task 支持（回传 §6.4）

| family | #gen | tasks per generator | role |
|---|---|---|---|
| claude | 1 | claude-3.5-haiku: 1,176 | h1_only |
| deepseek | 1 | deepseek-chat: 1,223 | h1_only |
| gemini | 1 | gemini-2.5-flash-preview: 1,176 | h1_only |
| llama | 1 | llama-3.3-70b-instruct: 1,284 | h1_only |
| **openai** | **3** | gpt-4.1: 1,127；gpt-4o: 1,060；gpt-4o-mini: 1,125 | **h2_candidate** |
| qwen | 1 | qwen-2.5-72b-instruct: 1,261 | h1_only |

→ **结论（回传 §6.4）：矩阵内仅 `openai` 满足“每正式 family ≥3 个可靠 generator + admitted
heldout 折”**（holdout=gpt-4.1 / gpt-4o / gpt-4o-mini 三折）；其余 5 族只能作 H1/诊断数据。
附属：`llm_codegen_v2` 的 `meta` 族 3 generator + 3 admitted 折（第二候选单元；跨源合并属新构造决策）。

### 2.4 N2 预注册

`n2_protocol.json`（在 N2 读取 test 之前写出并单独提交）：候选 = 六族齐全子集
（280 tasks；train/dev/test = 1,424/348/404 行；184/45/51 tasks）；P0 在候选矩阵上**整体复算**；
门 = 原 §3.2 三条（content>chance+metadata；≥P0+1pt；3 seed 方向稳定）。

## 3 N2：候选矩阵上的最小 H1 复核（`n2_h1_recheck/`，101.4s）

test 读取（UTC）：**2026-10-07T09:05:32Z**（本协议下单次）；500× task-cluster bootstrap。

| 视图 | macro-F1 | CI95 | Δ vs P0(候选) |
|---|---|---|---|
| metadata_only | .3689 | [.3325,.4059] | −.2596 |
| tfidf_word | .6429 | [.5970,.6923] | +.0148（CI [−.0394,+.0739]） |
| tfidf_char | .6207 | [.5746,.6660] | −.0064 |
| codet5_small_meanpool（冻结） | .5864 | [.5334,.6398] | −.0404 |
| **codet5_small_centered（转导）** | **.6538** | [.6046,.7006] | +.0278（CI [−.0191,+.0778]） |
| codet5_small_centered_unscaled（转导） | .6159 | [.5769,.6558] | −.0113 |
| fusion_lr（P0 候选复算） | .6264 | [.5756,.6792] | 0 |
| **mean_ensemble（P0 家族，无 dev 选择）** | **.6940** | [.6535,.7307] | **+.0663（CI [.0296,.1064]）** |
| style_lgb | .5603 | [.5133,.6088] | −.0660 |
| style_lr | .4898 | [.4426,.5393] | −.1368 |
| sem_lr | .5764 | [.5351,.6189] | −.0515 |

- **门**：cond1 ✅（.6538 > metadata .3689 > chance .1667）；cond2 ✅（best=转导视图 .6538 ≥ P0 .6264+1pt）；
  **cond3 ❌**（tfidf_word 3 seeds 的 Δ = −1.36/+2.18/−0.18pt，符号不一致）。
- **判定：N2 未通过**（预注册出口）→ 归档为**数据限制/负结果**；不重议 H2。
- 并列参照（D1 全量）：tfidf_word `.8200` / P0 fusion `.8388`。

### 3.1 附加观察（不作门，全部登记）

1. **绝对水平大幅下降**：候选矩阵上全部非 ensemble 视图 ∈ [.49,.65]，远低于 D1 全量（.77–.84）
   → 组成/支持效应对 D1 头部数字贡献大；六族齐全（竞争集均匀）的任务本身更难。
2. **dev 拟合融合在小 dev 上不稳**：候选上 `mean_ensemble` .6940 ≫ `fusion_lr` .6264
   （paired Δ +6.63pt，CI 不含 0）——这是 P0 实现层观察（等权集成 vs dev 选择），不是内容超越 P0 的证据。
3. 转导中心化方向为正但不显著（+.0278，CI 含 0；frac≤0=0.13）——与 D1（+.005）方向一致、幅度略大；
   仍按指导标注为转导诊断。
4. family/generator 层面：gemini recall .98、claude .69 vs deepseek .31、qwen .43——generator
   异质性在均匀组成下依旧存在。

## 4 交付与复现

- 目录树：`stage_d_data_construction_2026-10-07/{git_head.txt,git_status.txt,commands.txt,SHA256SUMS.txt,
  logs/{n0_diagnosis,n1_candidate_support,n2_h1_recheck}.log, n0_diagnosis/{n0_diagnosis.md,n0_strata.json,SHA256SUMS.txt},
  n1_candidate_matrix/{n1_candidate_support_matrix.json,n1_data_role_matrix.json,n1_duplicate_and_split_audit.json,
  n1_balance_report.md,n1_manifest.jsonl,n1_manifest_excluded.jsonl,n2_protocol.json,SHA256SUMS.txt},
  n2_h1_recheck/{config.json,metrics.json,predictions.npz,report.md,logs/run.log,SHA256SUMS.txt}}`
- 脚本：`scripts/stage_d_n0_diagnosis.py`、`scripts/stage_d_n1_candidate_support.py`、`scripts/stage_d_n2_h1_recheck.py`
- 命令、日志、SHA256 均在上表目录内（`commands.txt`、`logs/`、两级 `SHA256SUMS.txt`）。
- 小瑕疵（如实记录）：行级剔除使 7 个 task 的家族覆盖降为 1（其行仍单 split、合法；见 balance report §3）。

## 5 回传清单（指导 §6 对照）

1. N0 分层诊断与失败来源判断 → §1；
2. 每 family 的 generator/task 支持表 → §2.3；
3. 候选矩阵 split/duplicate/label/hash 审计 → §2.1/§2.2 + `n1_duplicate_and_split_audit.json`；
4. “每正式 family ≥3 可靠 generator”结论 → **仅 openai 满足**（llm_codegen_v2.meta 为第二单元）；
5. N2 是否允许与结果 → 允许（硬约束全过、预注册）且**已执行，门未通过（cond3）→ 归档数据限制/负结果**；
   不重议 H2、不启动 H2/H3；
6. 目录、命令、commit、日志、SHA256 → §4 与产物目录。

## 6 唯一下一步建议

按指导 §4/§5：本轮已把“数据构造能否修复支持不足”的问题回答为**否（在均匀组成下无稳定的
非转导内容增益；唯一稳定正项是 P0 家族内的等权集成实现差异）**。建议：**维持冻结**，将本记录
回传指导方；下一步（若继续）应围绕“数据来源/构造本身”的新指令展开（如跨源合并方案、或接受
数据限制重写论文主张），在此之前不启动任何 H2/H3 方法与训练。

---
*生成时间：2026-10-07；记录人：服务器执行 AI；提交号见本文件所在提交与后续补记。*
