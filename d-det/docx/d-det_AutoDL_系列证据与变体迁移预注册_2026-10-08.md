# d-det AutoDL 执行记录：系列证据复核与变体迁移预注册（2026-10-08）

依据《`d-det/docx/d-det_AutoDL_公开语料系列证据与变体迁移指导_2026-10-08.md`》；
运行锚点 `e38715d`；产物 `d-det/artifacts/public_full_followup_2026-10-08/`。
**本轮 = 只读复核 + 预注册**：`training_allowed=false / generation_allowed=false / test_read_allowed=false`
全程保持；未读项目旧 test、未生成输出、未下载模型权重。

## 0 重要状态：交接包缺失 → 服务器侧重构

- 指导方应随包提供的 `public_full_followup_2026-10-08/` 资产**未上传**：用户提供的**期望 SHA256 清单 16/16 文件全部缺失**（清单已保存为 `expected_followup_manifest_via_user.txt`，核对结果 `expected_manifest_check.json`）。
- 缺失清单：`family_series_admission.json`（期望 sha `cc4a7785…`）、`member_protocol_and_series.jsonl`、
  `official_docs/{CodeLlama-Instruct,Qwen2.5-Coder-Instruct,DeepSeek-Coder-v1-Instruct,CodeContests-schema}.txt`、
  `official_documentation_sources.json`、`proposed_family_checkpoint_protocol.json`、`source_snapshot/*`（8 件）。
- 处理：按指导 §2 的三个官方 URL 做**服务器侧等价重构**并全部登记来源与 commit pin；原件到达后逐文件核对替换。

## 1 完整核验复核（回传 item 1）

官方 `scripts/verify_public_task_full.py` 重跑（6.7s，日志 `logs/verify_rerun.log`）：
EvalPlus **8,778 行 / 399 task**；BigCodeBench **287,476 行 / 1,140 task**；0 duplicate/malformed；
`records_sha256 = d786667a…`（与包内 integrity 逐位一致）；zip 实测 sha `e98f221a…`（2026-10-08 指导已更正旧标注 `b8511ee6…`，不重传）。

## 2 单位口径复核（回传 item 2）

`public_full_adjudication_audit.py` 重跑（日志 `logs/audit_rerun.log`）：
**349 adjudication 行 = 306 去重 exact unit + 43 同 unit 的 asset/member 行**；
records 原样字段计数同为 306（全部匹配）；**21 行 unit-task 重复输出**（统计/训练按 `unit-task-sha` 去重，非 21 个额外模型）。

## 3 三系列成员与协议一致性（回传 item 3）

服务器侧重构 `family_series_admission.server_rebuild.json`（脚本 `scripts/build_family_series_admission.py`）：

| series | 成员（size 序） | full/instruct 支持 | 一致性 |
|---|---|---|---|
| CodeLlama-Instruct | 7b / 13b / 34b / 70b（`codellama--CodeLlama-*-Instruct-hf`） | 每成员 1,140 行 / 1,140 task | ✅（单 asset、backend 归一后一致） |
| Qwen2.5-Coder-Instruct | 1.5B / 7B / 14B / 32B | 每成员 1,140 / 1,140 | ✅（另有 hard/instruct 切片，Qwen-1.5B 296 行记录在案） |
| DeepSeek-Coder-v1-Instruct | 1.3b / 6.7b / 33b | 每成员 1,140 / 1,140 | ✅ |

- **共 11 成员**（= 指导 §2 的系列表）；`family_is_confirmed` 全局保持 `false`（本表只登记 model-series membership）。
- 官方资料（服务器侧拉取 + commit pin，`official_docs_server_fetch/sources.json`）：
  CodeLlama `e81b597e…`（MODEL_CARD.md）、Qwen2.5-Coder `5948e971…`、DeepSeek-Coder `2f9fd859…`、CodeContests proto `fa7a4f81…`。

## 4 回传 item 4/5

- `family_series_admission.server_rebuild.json` sha256 = `c079abea65807ce2…`（**原件期望值 `cc4a7785…` 待上传后核对**）；
  `readback_verification.json` 含全部条目与 git HEAD（运行锚点 `e38715d`，见 `git_head.txt` 类字段）。
- 政策核对：不读项目旧 test ✅；不生成新输出 ✅；不下载模型权重 ✅（仅拉取官方**文档文本** 4 件）。

## 5 变体迁移预注册（指导 §4，已冻结）

`variant_transfer_registration.json`（不执行）：
- 标题固定 `Official model-series variant transfer on task-heldout BigCodeBench`；
  折设计 = CodeLlama 4 折（7b→13b→34b→70b 旋转留出）、Qwen 4 折（1.5B/7B/14B/32B）、DeepSeek 3 折（1.3b/6.7b/33b）；
- 读出（train/dev 前冻结）：**R1 系列成员二分类**（见尺寸=正 vs 预声明负集=另两系列全部成员）、**R2 相似度/回归迁移**（系列中心/尺度/距离分数）；
- **双 P0**：P0-fusion（五成员 log-prob LR，仅 train 拟合，dev 只做预声明选择）+ P0-equal（等权）；
- 指标：heldout-variant AUROC / AP / task-macro / 95% CI（task-cluster bootstrap 500）/ 每折支持 / 迁移差；
  **不使用未见类的 macro-F1**；统计单位 = task cluster；seeds 0/1/2；
- 执行前闸门：注册文件冻结 → 原件包上传核对 → `proposed_family_checkpoint_protocol.json` 对齐 → 双 P0/特征确认 → 授权 test 单读时间；
- 禁止项（§5）全部写入 `forbidden` 字段；污染状态 `unknown`。

## 6 交付与复现

- 目录：`d-det/artifacts/public_full_followup_2026-10-08/`（`expected_followup_manifest_via_user.txt`、`expected_manifest_check.json`、`readback_verification.json`、`family_series_admission.server_rebuild.json`、`variant_transfer_registration.json`、`official_docs_server_fetch/`（4 文档 + `sources.json`）、`logs/`、`commands.txt`、`SHA256SUMS.txt`（15 件））；
- receive 目录 `SHA256SUMS.txt` 已恢复为其自身 18 件内容（用户注入的期望清单已另存于 followup 目录，不覆盖不丢失）；
- 脚本：`scripts/build_public_lineage_evidence.py`、`scripts/build_family_series_admission.py`（均标注服务器侧重构）；
- 提交：见本文件所在提交；**下一行建议**：上传缺失的 followup pack（16 件）以便逐文件核对（尤其 `cc4a7785…` 与 `proposed_family_checkpoint_protocol.json`），之后由指导端出执行授权。

---
*生成时间：2026-10-08；记录人：服务器执行 AI。*
