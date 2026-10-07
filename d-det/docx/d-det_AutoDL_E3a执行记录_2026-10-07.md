# d-det AutoDL 执行记录：E3a Path A 历史角色裁定（2026-10-07）

依据《`d-det/docx/d-det_AutoDL_E2阻塞裁定与PathA探索性复核指导_2026-10-07.md`》§2；
运行锚点 `78e52d3`；产物 `d-det/artifacts/stage_e3_patha_adjudication_2026-10-07/`
（`audit/`、`exploratory/`、`logs/` + `git_head/git_status/commands/SHA256SUMS`）。
**只读：未读任何 test 原文、未训练、未生成、未评测、未新增 test 读取。**

## 1 E3a 裁定结果（`audit/path_a_history_role.json`、`path_a_adjudication.md`）

- 候选 = Path A 的 **144 task**（E2 干跑池）；历史角色：
  - `historical_train_seen` **118**、`historical_dev_seen` **26**；
  - 有 pair 使用史（train 或 dev）的 task：**138/144**（依据 `positive_pair_index.jsonl`，695 行/159 task，
    README 声明 train 正对用于训练、dev 用于开发）；
  - `fresh_unseen_namespace` task 数：**0**；`fresh_test_eligible` task 数：**0**。
- **新的 102/20/22 重切分中，22 个候选 test 的历史构成 = 18 个 `historical_train_seen` + 4 个 `historical_dev_seen`**——
  即使本轮未读，也**不能称为 fresh**（逐 task 已标出，见 JSON `tasks[*].historical_role`）。
- 原 test 24（已读 22 + 保守 2）作为附录单列，整体 excluded（E2 已完成）。
- 与 E0 账本核对：144 task 零命中（AB/LCv2 命名空间互异）。

## 2 正式标签（与指导 §0 一致）

```text
confirmatory_independent = false
exploratory_development_namespace_reuse = true
h2_main_table_allowed = false
h2_appendix_diagnostic_allowed = conditional（需用户明确接受降级标签后才可进行 E3b 只读复核）
training_allowed = false
new_generation_allowed = false
```

Path A 正式状态 = **exploratory_reuse_only**（不得进入 ACL 主 H2 表；不得写“独立新任务泛化 /
普遍 family 几何 / 后训练因果 / H3 分离”）。

## 3 E3b 状态：**未执行，等待用户明确接受降级标签**

`exploratory/status.json`（`awaiting_user_acceptance`）：
- 服务器**不自行**把 Path A 从 blocked 改为 ready；
- 授权模板（accepted_label=`exploratory_reuse_only`、`new_test_read_allowed=true`、`selection_after_read=false`、
  forbidden_claims 四项）已随附，待用户确认后落为 `exploratory_authorization.json`；
- 若获授权，仅允许**冻结输出/冻结读出**（如：复用 h2_pair_round2 既有冻结模型对 meta 正对打分；冻结 CodeT5-base
  余弦），**禁止任何训练/重拟合（含 P0 fusion）、禁止阈值/seed/任务/generator 的选择**；
- 探索性报告标题固定为 “LCv2 development-namespace exploratory re-split”，只进 appendix/diagnostic ledger。

## 4 E3c：fresh unseen namespace 蓝图（`audit/fresh_namespace_plan.json`，只规划）

- 7 条硬要求（新 provenance、先切 split 再生成、同 task 三 generator、版本/许可/重试可复现、
  预注册 positive/negative/heldout/双 P0/test 单次、生成后三层碰撞=0、未授权不调用）；
- 缺口：fresh 任务集（来源+许可）、生成授权、（若走 google/mistral）第三 generator 权重/API（本机无）；
- 成本/磁盘：≈150–300 task × 3 family × 3 gen ≈ 1–2MB 原文 + <5MB 索引（≈15MB 量级上限同 E1/E2）；
- `training_allowed=false`、`new_generation_allowed=false`；power_unknown（需在协议中模拟）。

## 5 回传清单（指导 §6）

1. `path_a_history_role.json` + `path_a_adjudication.md`（本提交）；
2. Path A 明确为 **`exploratory_reuse_only`**（执行层面仍未启动 E3b；主 H2 表不允许）；
3. E3b 未获授权 → `exploratory/status.json` 待用户裁定（模板随附）；
4. 独立路线蓝图 `fresh_namespace_plan.json` + 缺口/成本/磁盘估计；
5. commit、目录树、命令、日志、SHA256（见产物目录）；**未提交 raw outputs/权重/预测/凭据**。

## 6 唯一下一步

等待用户回答一个二选一问题：
(a) **接受降级标签** → 落 `exploratory_authorization.json` 后执行 E3b 只读复核（仅冻结读出）；或
(b) **不接受** → E3 结束为 blocked，主论文保留 E0/E1 的数据限制与现有负结果；
无论 (a)/(b)，若要以 ACL 主表为目标，均需按 E3c 蓝图投入 fresh namespace（待授权）。

---
*生成时间：2026-10-07；记录人：服务器执行 AI。*
