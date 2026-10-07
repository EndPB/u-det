# d-det AutoDL 执行记录：E2 独立数据预注册与干跑（2026-10-07）

依据《`d-det/docx/d-det_AutoDL_E0E1完成后_E2独立数据预注册与干跑指导_2026-10-07.md`》；
运行锚点 `881f300`；产物 `d-det/artifacts/stage_e2_independent_pilot_2026-10-07/`
（`prereg/`、`task_dry_run/`、`audit/`、`logs/` + `git_head/git_status/commands/SHA256SUMS`）。
**未生成任何数据/输出/权重/预测；未运行训练；未读取任何旧 test 原文（未新增 test 读取）。**

## 0 结论速览

- **E2a = `blocked`**：现有材料无法在不猜补字段的前提下确定一个合法 target（三条路径均卡在可证实的输入上，见 §1）。
- **E2b 干跑已交付**：LCv2 命名空间候选池 **144 task**（train+dev；三层 hash 建档）通过碰撞审计：
  跨 split 组 0、对 AB prompt 命中 0、对 LCv2 原 test 命中 0；72 个“multi 组”全部是
  **group 内 simple/secure 设计对**（同 prompt、同 group、同 split，`intra_group_only=True`），不构成泄漏。
- **E2c `generation_plan.json` 未生成**（按指导 §3：E2a≠ready 时不产出行动文件；原因见 `prereg/feasibility_decision.md`）。
- 全程保持 `generation_allowed=false / training_allowed=false / external_cost_authorized=false / test_read_allowed=false`。

## 1 E2a 预注册（`prereg/e2_preregistration.json`，status=blocked）

三条候选路径与各自 blocker：

| 路径 | source unit | 现状 | blocker | 转 ready 的最小输入 |
|---|---|---|---|---|
| **A（最接近）** | `h2_llm_codegen_v2::meta`（vendor_group；llama2/llama3/codellama） | 三 generator 齐备（非空 code 覆盖 134/152/164 of 168）；D0 3 折 admitted；**复用无需生成** | 命名空间已在 pair 轮 train/dev 开发使用；原 test 24 个已读 22 个 → “新任务集”口径需裁定 | 指导端书面接受该口径 + 授权一次评测读取 |
| **B** | `…::google` 或 `…::mistral` | 各 2 generator，缺第三 | **第三 generator 不可证实**：本机无任何候选权重/缓存（仅 codet5 系）、无 API 授权、版本/许可未定 | 指定确切版本 + 许可 + 调用可行性 |
| **C** | `h2_authorbench_dcan::openai` | 3 generator，但命名空间已 3 次暴露 | 新任务需外部 API（未授权）且 AB 命名空间不可复用 | 全新任务集 + API 授权 |

预注册字段（守规）：`generation_allowed=false`、`training_allowed=false`、`external_cost_authorized=false`、
`test_read_allowed=false`；`missing_fields` / `blockers` 已按 §1 模板登记，未猜补。

## 2 E2b 干跑交付（`task_dry_run/`、`audit/`、`prereg/split_plan.json`）

- **候选**（`candidate_tasks.jsonl`，144 行）：task_source_hash / task_id_candidate（真实 task_id，未造伪）/
  provenance_group / language=C / statement_hash / split_group / existing_overlap_status /
  exact+norm_ws+norm_lex hash / source_dataset / license_status=missing /
  generation_status=existing_outputs_present(reuse_req_authorization) / per-generator 非空 code 标志。
- **排除**（`excluded_tasks.jsonl`，432 行 = LCv2 原 test 24（保守整组，含 pair 轮已读 22）+ AB test 408（E0 账本）+ 交叉命中 0）。
- **碰撞三层**（`audit/collision_groups.json`）：候选内 multi 组 72/72/72（全部 group 内设计对，`intra_group_only=True`）；
  **跨 split 组 0/0/0**；对 AB prompt 0；对 LCv2 原 test 0；norm_lex 实现 sha256 已登记。
- **split_plan.json**（只计划不执行）：group（CWE）级、seed 20261007、目标 70/15/15 →
  实际 **train/dev/test = 102/20/22 task（51/10/11 group）**；各 split 的 task 列表 sha256 已记录；
  **meta 三折支持预览**：heldout=llama2/llama3/codellama 时，
  train(≥2 seen gen 非空) = 90/80/73，dev = 19/15/16，test(heldout 非空) = 18/21/20。
- **license_and_provenance.json**：LCv2 provenance（CSV 列表+hash）在案；**license=missing**；
  family 映射=release-folder 推导；无 base/SFT/DPO 谱系；第三 generator 不可用证据（本机核查）记录。

## 3 E2c：生成计划（**未生成**，按门槛）

按指导 §3：“只有 E2a 为 ready 且 E2b 审计通过，才生成 generation_plan.json”。
E2a=blocked ⇒ 不产生 `generation_plan.json`（避免出现无门槛授权的行动文件）。若 Path A 被裁定接受，
其“生成计划”实质为**无需生成**（复用现有输出）+ 一次评测运行授权；若走 Path B/C 需要外部生成，
所需参数模板（版本/temperature/重试/保存位置/生成后 hash 检查）已在 `pilot_design.md` 与
`feasibility_decision.md` 中列出待落位。

## 4 pilot_design 摘要（能检验什么 / 不能检验什么）

- **Path A 能检验**：Meta 单元在同任务、同族、不同 generator 正对下的 H2 工程形态，
  以及族内 H1 可读性（干跑池 144 task，其中 meta 三折 train/dev/test 支持预览充足）；
- **不能检验**：跨 dataset 拼接的“普遍 family 几何”、后训练因果（无 base/SFT/DPO）、H3；
- **Path B/C**：只有补齐第三 generator（B）或全新任务集+API（C）后才可重复上述检验。

## 5 可行性决定（`prereg/feasibility_decision.md`）

- **status = blocked**；唯一缺口（两问）：
  1. **任务独立性口径**：Path A 是否算“新任务集/独立复核”（命名空间曾用于 pair 轮开发；需指导端裁定）；
  2. **第三 generator 不可证实**（Path B）：版本/许可/权重/API 全缺。
- 停止规则对照：仅 §6 第 1 条（找不到可复现第三 generator）触发；其余（重叠、双标签、磁盘、随机复制）未触发。
- 转 ready 的最小输入：二选一（(a) 接受 Path A 口径 + 授权评测读取；或 (b) 提供全新 C 任务集 + 生成授权），
  以及（若走 B）第三 generator 的确切版本与许可。

## 6 交付、复现与边界声明（回传 §7 第 6 项）

- 目录：`d-det/artifacts/stage_e2_independent_pilot_2026-10-07/`（`git_head.txt`/`git_status.txt`/`commands.txt`/
  `SHA256SUMS.txt` + 上述 9 个产物 + 控制台日志）；
- 命令：`python scripts/stage_e2_pilot_prereg.py`（0.1s，只读元数据）；
- **无 raw outputs、无权重、无预测、无凭据入产物**；未新增 test 读取；旧资产未删除；
- 提交号：见本文件所在提交（运行锚点 `881f300`）。

## 7 唯一下一步

将本记录与 `prereg/feasibility_decision.md` 回传：**等待对 Path A 口径的裁定与（如选 B/C）资源授权**；
在获得书面授权前不生成数据、不调用外部模型、不运行评测。

---
*生成时间：2026-10-07；记录人：服务器执行 AI。*
