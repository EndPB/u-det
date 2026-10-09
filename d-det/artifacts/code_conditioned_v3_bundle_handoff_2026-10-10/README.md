# v3 canonical feature handoff（2026-10-10）

按指导文档 §13 / ACL 总路线 §55：**最小 train/dev 补交包** —— 仅含本机要求的三个
文件。替换本机同名文件后，按 v3 `spec_sheet.json` 原样重跑 C0 即可做最终数值对账。

## 文件与 SHA-256（均已核验匹配）

| 文件 | SHA-256 | 字节 |
|---|---|---|
| `bundle.npz` | `d05f8c899d771e0572fb3e1756e06810ef5919cd9a2aac5d598e2bbc7caac19a` | 21,477,099 |
| `features_manifest.json` | `aaef7b4716c1821780719ba18415351abcdb7abbb793ba3c619bfc8a18fae507` | 3,348 |
| `feature_member_order_check.json` | `11428bf8f88dea0e44fbdf434f60a074138af70685d17dab8b105ade66d422e2` | 2,445 |

三个 SHA 同时满足：① 与 canonical C0 交付中
`metrics.json → feature_mapping` 记录一致；② 与 ACL §55 所列期望值一致；
③ 服务器端工作树文件与 commit `53524e7` 的 git blob 逐字节一致（已复核）。

## 内容说明（仅 train/dev）

- 行序 = `inputs/records_train_dev.jsonl` 原序（10,659 行，服务器原始行序）；
- 成员序 = admission canonical 序 [CodeLlama×4, Qwen×4, DeepSeek×3]（`member_ids`
  数组内嵌于 bundle；`members_order_sha256 = 1afd29b6…`）；
- 与修复前 bundle 相比**仅 `member_idx` 变化**（置换 [0,1,2,3,8,9,10,4,5,6,7]），
  其余全部数组 bitwise 相同（详见 canonical C0 目录内 `feature_order_fix_summary.json`）；
- 不含 test 数据、权重、生成资产或原始语料；开关：train/dev only，
  test_read / generation / weights_downloaded / code_execution 全 false。

## 使用方法

1. 用本目录三个文件覆盖本机 `…/code_conditioned_design_2026-10-09/features/`
   下的同名文件（保持相对路径一致）；
2. **不改动任何其它内容**，按 v3 `spec_sheet.json` 重跑 C0；
3. 逐折/逐行比较 fused 与四组件分数及指标：`row_score_max_abs ≤ 1e-3` 且
   `metric_abs ≤ 1e-3` 才关闭闸门（指导 §13）。

服务器运行环境（解释残余浮点差异用）：Python 3.12.14 / NumPy 2.2.6 /
scikit-learn 1.9.1 / OMP_NUM_THREADS=8（见 canonical C0 目录
`runtime_attestation.json`）。

## 校验

```bash
sha256sum -c SHA256SUMS.txt
```
