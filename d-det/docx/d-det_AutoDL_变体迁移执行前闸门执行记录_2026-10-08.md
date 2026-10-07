# d-det AutoDL：变体迁移执行前闸门执行记录（2026-10-08）

- 上游指导：《AutoDL 变体迁移执行前闸门指导》（2026-10-08）；上游预注册提交 `811ecfe`。
- 本轮性质：**只读**。不训练、不生成、不读 test 正文、不下载权重。
- 产出目录：
  - `d-det/artifacts/variant_transfer_preflight_2026-10-08/`（execution_config_frozen / r1_negative_set_amendment / execution_switches / protocol_audit / fold_matrix / heldout_order_p0_seeds / commands / logs / SHA256SUMS）
  - `d-det/artifacts/public_full_followup_2026-10-08/`（server_rebuild/、provided_original/、provenance_resolution.json；SHA256SUMS 更新）
- 脚本：`scripts/variant_transfer_preflight_provenance.py`、`scripts/variant_transfer_execution_freeze.py`、`scripts/variant_transfer_protocol_audit.py`。

## 1 交接包身份（§1）

结构（followup 目录下）：

```
server_rebuild/                      # 服务器重构版快照（临时复核材料）
  family_series_admission.server_rebuild.json（sha c079abea…，非原件）
  official_docs_server_fetch/（4 文档 + sources.json；commit pins e81b597e/5948e971/2f9fd859/fa7a4f81）
provided_original/                   # 原件到达位置（当前为空，含 README 说明核对流程）
expected_followup_manifest_via_user.txt   # 用户提供 16 项期望哈希
provenance_resolution.json           # 逐文件核对记录
```

核对结果（`provenance_resolution.json`）：**16/16 awaited**（provided_original 为空）——
- `decision=awaiting_original` ×16；`server_reconstruction_only=true`；
- 旁证（不构成原件核验）：11/16 项存在服务器侧对应物且 sha 与期望清单**相同**（4 份 official_docs ↔ 官方拉取文档、2 个脚本、5 个接收阶段产物）；3 项不同（`family_series_admission.json` 重构版 `c079abea…` ≠ 期望 `cc4a7785…`；`official_documentation_sources.json`；`full_adjudication_audit.json` 系 811ecfe 重跑更新）；2 项无对应物（`member_protocol_and_series.jsonl`、`proposed_family_checkpoint_protocol.json`）。
- 政策已写入：16/16 通过前不删/不覆盖 server_rebuild/；不得把 `c079abea…` 写成原件哈希；长期不可达时定性 `server_reconstruction_only`。

## 2 执行前协议审计（§2，`protocol_audit.json` + `fold_matrix.csv`）

单遍扫描 records.jsonl（296,254 行→提取白名单元数据字段，零正文落盘）：**C1-C8 全 PASS**。

| 检查 | 结果 |
|---|---|
| C1 11 成员均来自 BCC full\|instruct | PASS（118 主协议 unit 内 11 成员；其他切片行数单独登记、不混用） |
| C2 每成员 1,140 task / 1,140 行 | PASS（rows=1140 且 tasks=1140；单 asset/backend/temperature） |
| C3 task split 重建 | PASS：798/171/171；train/dev/test 三个 task_list_sha256 与 e38715d 预注册**全部匹配**；split_index.csv 逐行 0 不一致 |
| C4 heldout 顺序 | PASS：CodeLlama 7b/13b/34b/70b；Qwen 1.5B/7B/14B/32B；DeepSeek 1.3b/6.7b/33b（与注册/指导一致） |
| C5 配置先于 train/dev 读取冻结 | PASS：`execution_config_frozen.json` sha `6aba9c4d…`、`r1_negative_set_amendment.json` sha `4d0d1631…`、`execution_switches.json` sha `1552b1c1…`（记录于 audit） |
| C6 折矩阵不含 heldout | PASS：11 折 train_units 均不含 heldout member；heldout 行仅在 TEST tasks 评分（train/dev 中的 969 行由 ingestion 白名单排除） |
| C7 test 正文未读 | PASS：读字段白名单 13 项（manifest/task_id/行哈希等）；代码正文零读取零落盘；未下载权重 |
| C8 重复口径复核 | PASS：全语料 unit-task-sha 重复行 **21**（期望 21）；主协议多行 unit-task **8**（期望 8：claude-3-5-sonnet-20241022 × BigCodeBench/18–25） |

折矩阵摘要（每折）：train 10 units × 798 task = 7,980 行；dev 1,710 行（选择用，不含 heldout）；test = heldout 171 行 + 负集 1,197/1,368 行（CodeLlama/Qwen 折 7 负成员；DeepSeek 折 8 负成员）。

## 3 R1 负集 amendment（§3，`r1_negative_set_amendment.json`）

- **原注册版保留**：负集=另外两系列全体 → 结果名 `series-transfer-with-size-mix`（默认主结果；不得静默改写）。
- **预声明匹配控制版**（amendment，敏感性分析）→ 结果名 `series-transfer-size-matched`：
  - 成员级匹配：穷举最优、一一配对，最小化 Σ|log10(size_pos)−log10(size_neg)|；平局按 (总距离, 成员名升序)；|正集|≥2 时要求每个负系列至少 1 名成员；**纯元数据**（不读任何数据内容）；
  - 11 折选择已冻结，例：CL heldout 7b → {Qwen-14B, Qwen-32B, DS-33b}（总距离 0.3851）；CL heldout 70b → {Qwen-7B, Qwen-14B, DS-33b}（0.0451）；DS heldout 1.3b → {Qwen-7B, CL-34b}（0.0320）；
  - 长度分位加权（执行阶段公式冻结）：task cluster 内按输出字符长度五分位桶，w= p_pos/p_neg 截断 [0.1,10]，train 估计一次、dev/test 复用；
  - 升级为主结果需新 prereg 版本并保留原注册结果；不得看完 dev/test 后决定采用哪版。
- **必备对照**：size/length-only baseline（log10 参数规模 + 输出长度 + 结构特征）；原版结果按 size bucket / 长度 / 难度（in_hard）分层报告。
- 标签纪律：heldout variant=正域迁移样本；禁止把未见 variant 当新类别报 macro-F1。

## 4 R2 与执行配置冻结（§4-§5，`execution_config_frozen.json`）

- R2 对照：长度/词法 baseline、CodeT5-small/base 冻结表示、`P0-fusion`（仅 train 拟合）、`P0-equal`（等权、无校准）；中心/回归/阈值仅 train 拟合，heldout 仅冻结后评分；不得作因果解释。
- seeds：特征 [0,1,2]；task-cluster bootstrap 500（seed 20261008）。
- R1 分类器：LogReg(balanced)，C 网格 [0.03,0.1,0.3,1.0]，dev 仅一次预声明选择；阈值 0.5。
- dev 冻结解读：dev=已见尺寸+负集（**不含** heldout variant），一次预声明选择；heldout 仅 test 单读评分。
- 停止规则与论文允许/禁止表述原文登记（§5-§6）。

## 5 回传清单（§7）与开关

| 回传项 | 位置 |
|---|---|
| provenance_resolution.json | `public_full_followup_2026-10-08/` |
| protocol_audit.json（+fold_matrix.csv） | `variant_transfer_preflight_2026-10-08/` |
| R1 负集/匹配配置 | 同上 `r1_negative_set_amendment.json` |
| heldout 顺序 / P0 / seeds | 同上 `heldout_order_p0_seeds.json`（+execution_config_frozen.json） |
| git_head / git_status | 同上（HEAD=811ecfe） |
| 开关 | 同上 `execution_switches.json`：**training_allowed=false / generation_allowed=false / test_read_allowed=false**（authority=awaiting_new_execution_authorization） |

校验和：preflight 目录 13 件、followup 目录 25 件（SHA256SUMS.txt）。本轮 commit 后 push。

## 6 大文件与边界

- 未读：项目旧 test（LCv2/AuthorBench/D1/N2）、任何代码正文写入、模型权重；Path A 维持 exploratory_reuse_only。
- 未执行：训练/拟合/生成/任何 test 评分。
- 唯一下一步：等待指导端（a）补传 followup 原件 16 件以便逐文件核对，且（b）出具新的执行授权（含 test 单读时间与授权记录），之后方可进入 train/dev 阶段。
