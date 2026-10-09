# d-det AutoDL 执行记录：v3 canonical 特征包最小补交（2026-10-10）

## 0. 上游与授权范围

- 指导文档 §13（`d-det/docx/d-det_AutoDL_代码条件后训练_已见家族未见生成器指导_2026-10-09.md`）：
  本机已按 v3 canonical member order 与服务器原始行序完成复现，**11/11 折的 fit-text、
  词表、heldout/train/eval 行哈希一致**（映射与词法协议已对齐）；数值对账仍待 canonical
  bundle —— 下一次开机**仅需补交三个 train/dev 文件**，不补传 test、权重、生成资产或原始语料。
- ACL 总路线 §55 同述：本机 dev row/task 均值 `.943838/.958068`，服务器 `.944192/.958542`；
  最大折级差 row `.001383`、task-macro `.003342`；`1e-3` 闸门未关，C0 =
  `cross_side_score_pending`。
- 开关：train/dev only；`test_read=false`；`generation=false`；`weights_downloaded=false`；
  `code_execution=false`（本轮未重跑任何 C0，未读取 test、未生成、未下载权重、未执行代码）。

## 1. 本轮动作（唯一）

1. **核验三文件**（工作树 + git blob 双重比对）：

| 文件 | SHA-256（期望=实测） |
|---|---|
| `features/bundle.npz` | `d05f8c899d771e0572fb3e1756e06810ef5919cd9a2aac5d598e2bbc7caac19a` |
| `features/features_manifest.json` | `aaef7b4716c1821780719ba18415351abcdb7abbb793ba3c619bfc8a18fae507` |
| `features/feature_member_order_check.json` | `11428bf8f88dea0e44fbdf434f60a074138af70685d17dab8b105ade66d422e2` |

   三者与 canonical C0 交付 `metrics.json → feature_mapping` 记录一致、与 §13/§55
   期望值一致、与 commit `53524e7` 的 git blob 逐字节一致。

2. **制作最小补交包**：新目录
   `d-det/artifacts/code_conditioned_v3_bundle_handoff_2026-10-10/`，仅含：
   - `bundle.npz`（21,477,099 B）、`features_manifest.json`（3,348 B）、
     `feature_member_order_check.json`（2,445 B）；
   - `README.md`（替换步骤 + 1e-3 对账要求 + runtime 说明）、
     `handoff_manifest.json`（来源 commit、来源路径、逐文件 SHA 与字节数、核验声明）、
     `SHA256SUMS.txt`。
   - 不含任何 test/权重/生成资产/原始语料；bundle 行序 = `records_train_dev.jsonl`
     原序，成员序 = admission canonical，`member_idx` 为修复后版本（其余数组与旧
     bundle bitwise 相同）。

## 2. 下一步（照 §13，等待本机执行）

本机用补交包覆盖本地 features 三文件后，**原样**重跑 v3 C0，并同时检查：
`row_score_max_abs ≤ 1e-3` 且 `metric_abs ≤ 1e-3` —— 两者均满足才关闭闸门
（标记 `aligned`），之后才讨论是否统一规格重跑 C1–C3；在此之前 C1–C3 继续停止，
不增加 backbone/参数量/温度/融合权重或新变换。

## 3. 交付物

- 补交包目录（见上）；本记录；两份文档（指导 §13 / ACL §55，随本提交同步）。
- 推送：本记录随下一次 main 提交推送。
