# d-det AutoDL 执行记录：H3 修订后新主线批次指导接收与待命（2026-10-10 夜）

## 0. 输入

- 新指导：`d-det/docx/d-det_AutoDL_H3修订后新主线批次指导_2026-10-10.md`
  （sha256 `70ea6d81cab0137f85599d1032f40851fe293ace51b918dc377abe9e7c844e85`）。
- 定位：下一次大显存批次的**预注册执行指导**；明确“**本文件暂不触发 AutoDL 运行**”。

## 1. 指导要点（服务器侧接收口径）

- 批次矩阵：`lexical_control`（强控制，必须保留）、`detection_only`、`source_only`、
  `joint`、`joint_invariance`（可选 `joint_adversary`）；≥2 个 generator-heldout 折、3 seed。
- 约束：不重跑已关闭的 C0–C3；不通过增加容量/温度/损失项补救；变体只来自 train parent；
  `test_read=false`、`generation=false`、`weights_downloaded=false`、`code_execution=false`。
- 触发条件：本机完成 H3 数据修订并重新通过闸门后，**冻结数据包 + `batch_manifest.json`
  + 脚本一次性上传**；AutoDL 整批执行、统一回传；**不接收零散小实验指令**。
- 通过规则：相对 `lexical_control` 需有预注册增量；任一条件失败 → `stop_or_revise_data` /
  `increment_not_observed`，不加 backbone/温度/轮数/损失项。

## 2. 本轮动作（无计算）

1. 接收并核验指导文件：实测 sha256 `70ea6d81cab0137f85599d1032f40851fe293ace51b918dc377abe9e7c844e85` 与记录一致；指南 §7 指针、H3 入口指针、总账两条版本日志行核对一致。
2. 入库：新指导、指南 §7 指针、H3 入口指针、总账版本日志随提交 `0281f36` 入库；本记录经服务器侧实测复核（哈希/指针/约束逐项）后随本次提交确认。
3. 服务器保持**待命**：开关全 false，不预设 job；等待冻结数据包。

## 3. 交付物

- 本记录（服务器侧复核版）；指导、指针与版本日志随提交 `0281f36` 入库；本记录随本次提交推送。
