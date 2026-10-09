# Fresh C0 规范成员序报告（member-order 修复 + 折内词法；2026-10-10）

上游：跨侧复现裁定 `code_conditioned_fresh_c0_cross_side_local_reproduction_2026-10-10/`
（阻断 = feature member order 错位）与指导文档 §12。**本目录取代
`code_conditioned_fresh_c0_corrected_2026-10-10`（成员序仍错位）成为跨侧对账用的
fresh C0 记录。** 开关：train/dev only；test_read / generation / weights_downloaded /
code_execution 全 false（逐折 manifest v3 记录）。

## 1. 修复内容（指导 §12）

- **features 重建**：`cc_build_features.py` 的 `member_idx` 从 endpoint-plan 顺序
  [CodeLlama, DeepSeek, Qwen] 改为 admission 顺序 [CodeLlama, Qwen, DeepSeek]；
  逐行断言 `member_idx[i] == admission.members_order.index(rows[i].model_id)`；
  断言与哈希在 **bundle 写入前**落盘到 `features/feature_member_order_check.json`，
  失败即停止（不进入任何折拟合）。
- **成员序哈希**：`members_order_sha256 = sha256("\n".join(members)+"\n") = 1afd29b6…`，
  与本机 audit 的 admission 值一致；旧错误序哈希 `e6d7852c…`；旧 bundle `be5cc9d6…`。
- **仅 member_idx 变化**（服务器逐数组 bitwise 比对）：除 `member_idx`（按置换
  [0,1,2,3,8,9,10,4,5,6,7]）与新增 `member_ids` 外全部相同；`emb_task_small.npz`
  bitwise 不变（f1e75299…）。新 bundle `d05f8c89…`。
- **行级映射哈希**（§12 公式，约定见 manifest）：`row_mapping_sha256 = b38a41e4…`；
  另记 `nl_join_trailing_newline`、`concat_no_separator` 两个变体备核。
- **C0 manifest v3（每折）**：`feature_mapping`（members_order_sha256、
  row_mapping_sha256、bundle/manifest/check 文件 SHA、bundle member_ids）、
  `script_sha256`（本脚本 `a3911260…`、`cc_common.py` `851b62e9…`）、
  `runtime_attestation`（Py 3.12.14 / NumPy 2.2.6 / sklearn 1.9.1 / OMP=8）。
- 词法仍为折内拟合（沿用修正版 v2：char/word 只在 `fit_rows` fit，eval 只 transform）；
  每折额外断言正类行 = heldout 成员、负类行 = 跨家族成员（canonical 映射下）。

## 2. 规范序结果

- **dev**：row 均值 **.9442**、task-macro **.9585**、pooled .9427
  （bootstrap CI：row [.9346, .9531]；tm [.9491, .9672]）；
- **inner**：row .9491、tm .9666。
- 逐折 dev（row/tm）：CL-13b .9556/.9574、CL-34b .9485/.9724、CL-70b .9381/.9624、
  CL-7b .9773/.9783、Q-1.5B .8744/.9165、Q-14B .9838/.9916、Q-32B .9851/.9858、
  Q-7B .9841/.9850、DS-1.3b .9285/.9488、DS-33b .8866/.8991、DS-6.7b .9242/.9466。
- 组件 row 均值（dev）：semantic .9131、char .8660、word .8799、style_meta .9164。
- 确定性自检：复拟合 max|Δ|=0.0（semantic、char），词表哈希一致。

## 3. 成员序修复的受控对比（vs 上一版 corrected/permuted run）

- **不变性（4 个 CodeLlama 折）**：fit/eval 行集合与 y 完全相同，
  fused 分数 **max|Δ| = 0.0、r = 1.0** —— 本次修改只改变映射、不改变计算。
- **重定义（7 个 Qwen/DeepSeek 折）**：上一版“折标签”与实际留出成员循环错位
  （服务器侧复原证据，`member_order_impact_vs_permuted_run`）：

| 折（标签） | 上一版实际留出成员 | 上一版 row/tm | 规范版 row/tm |
|---|---|---|---|
| Q-1.5B | DS-1.3b | .9685/.9833 | .8744/.9165 |
| Q-14B | DS-33b | .8488/.8697 | .9838/.9916 |
| Q-32B | DS-6.7b | .9361/.9632 | .9851/.9858 |
| Q-7B | Q-1.5B | .7688/.7769 | .9841/.9850 |
| DS-1.3b | Q-14B | .9707/.9766 | .9285/.9488 |
| DS-33b | Q-32B | .9873/.9912 | .8866/.8991 |
| DS-6.7b | Q-7B | .9605/.9642 | .9242/.9466 |

  聚合口径上 dev row .9328 → .9442、tm .9451 → .9585，全部来自这 7 折的语义修正。

## 4. 历史包诊断（canonical 映射重算；非闸门）

r = .8946–.9418，max|Δ| = 1.02–2.01，row/tm 双向接近但未达 1e-3：
历史包仍为 `provenance_incomplete`，只作诊断，不构成闸门结论。

## 5. 闸门状态与下一步

- 协议审计：通过（折内词法 + §12 映射断言/哈希先行落盘 + 逐折映射断言）。
- C0 状态：`prepared (canonical member order); cross-side score comparison pending`。
- 下一步（裁定 required_fix #4）：本机按 v3 `spec_sheet.json` 复现并逐折比较——
  词表预期 11/11 相等；fused/组件 digest 需在匹配 runtime 下比较，或采用
  服务器侧 digest + 本 manifest/metrics 内的 runtime attestation 进行对账；
  仍要求 `row_score_max_abs ≤ 1e-3`、`metric_abs ≤ 1e-3` 方可标记 `aligned`。
- 在此之前：C1–C3 不重跑；test、生成、权重下载、代码执行继续关闭；不增加
  backbone/参数量/温度/融合权重或新变换。
