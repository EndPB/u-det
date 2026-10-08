# d-det_AutoDL：`1c5b802` 裁定修正与 adjudication（2026-10-08）

来源：指导《`1c5b802后_关系增量与角色未知残差实验指导`》§1.1 五项文档修正。
本文件为**带来源的 adjudication**：不覆盖任何历史产物；历史 gate 原文件保留。

## 1. F2 M-HO 裁定收窄

- 历史：`f2_member_relation_protocol_fix_2026-10-08/metrics_f2_fix.json` 中
  `gate.verdict = member_holdout_relation_candidate`（字面 §6 条件全过，**原文件保留**）；
- **正式裁定收窄为：`relation_increment_over_endpoint_unresolved`** ——
  来源：`partner_only_probe=0.8667 > q_only=0.8139`（11/11 折 q 低于 partner 探针）——
  **现有数据未证明“关系增量”超过端点（partner 成员身份）组成**；
- 后续（实验 A，本轮执行）：端点边际平衡 + independent-source composition 基线的 48 折实验
  提供该问题的正式检验入口（`relation_endpoint_balanced_2026-10-08/`）。

## 2. H3 标签修正

- 已完成四配置的“检测”分支：得分用 `[S;A]` vs `[S;−A]`，**无 human 标签**（complete/instruct 均为 AI）；
- 正式改称：**`protocol_direction_plus_relation_multitask`**；D（协议方向）、F（关系）与协议方向 R0 分开报告；
- 相关：`h3_definition_and_human_support_2026-10-08/`（标签重命名记录 + human 支持表 + c3 激活统计）。

## 3. c3 梯度余弦证据修正

- c2/c3 输出相同**不能证明梯度分离有效**；
- 本轮补测（`metrics_h3_cos_stats.json`，14 折逐批记录）：
  **激活率（cos>0.3）= 0.0000**（所有折全未激活；cos 均值 0.04~0.13；部分折负余弦率 25~50%）——
  **`relu(cos−0.3)` 惩罚从未触发**，因此 c2≡c3 是该设计下预期的平凡结果，不构成任何“梯度分离”证据；
- 正对齐惩罚的目的与效果需在正式协议前重新论证（指导 §4.2 要求）。

## 4. R0 复核记录精确化

- 精确表述（修正此前过宽的说法）：
  - 早期 **`_tmacc`（task-macro）确实去掉了抽样重数**（被 81c68d2 轮的 reaudit 修正为重数保留口径）；
  - **model-balanced `_acc` 与 `paired_delta` 原先已保留重数**；
  - **不得写“全部 pair CI 原来均错误”**；
- 来源：`r0_protocol_diff_reaudit_2026-10-08/`（修正后口径）与 81c68d2 轮记录文档的差异点。

## 5. 差值口径修正（撤销“98%”表述）

- **所有差值必须同指标、同 split、同输入接口**；
- 撤销此前“联合训练几乎免费/归因保留 98%”的表述：c2（rel 0.6922）与 c1（rel 0.7068）是
  **不同损失配置下 relation 分支的同 split 点差**，无配对 CI/容忍界限，**不得写免费或无损**；
- 来源：`h3_train_dev_registered_2026-10-08/metrics_h3_registered.json`（点差保留为开发诊断）。

## 6. 与历史文档的关系

- `d-det_AutoDL_67b4ccd修复轮执行记录_2026-10-08.md` 与对应回传中的
  “关系增量”“98%”“梯度分离”表述以本文件为准收窄/撤销；
- 其余数字（F0-B 0.7967 等）不变；所有裁定均为 train/dev 开发估计，旧 171 dev 不构成独立确认。
