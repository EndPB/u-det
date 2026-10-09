# d-det AutoDL 执行记录：跨侧成员序修正与 C0 规范序重跑（2026-10-10）

## 0. 上游与授权范围

- 上游裁定：`d-det/artifacts/code_conditioned_fresh_c0_cross_side_local_reproduction_2026-10-10/`
  （跨侧 `1e-3` 闸门仍 `blocked`；阻断原因为 **feature member order 错位**；
  `required_fix` 四项：①规范化成员序 ②成员序+行映射哈希写入 features manifest 与每个 C0
  manifest ③修正后只重跑一次 corrected train/dev C0 ④之后才允许跨侧 digest 对账）。
- 指导文档：`d-det/docx/d-det_AutoDL_代码条件后训练_已见家族未见生成器指导_2026-10-09.md` §12
  （断言公式与“断言+哈希必须在 bundle 写入前落盘；失败即停止”）。
- 开关：train/dev only；`test_read=false`；`generation=false`；`weights_downloaded=false`；
  `code_execution=false`（未读取 test、未生成、未下载权重、未执行任何被评估代码）。
- 执行 head：`be8df5075a79e3b5425ec6c79eaafa2bfca19bff`（执行时工作树含本次提交的全部内容）；
  交付提交：随本记录一并提交的下一次 main 提交（内容即本次执行产物）。

## 1. 修复实施（逐项对应 required_fix）

1. **规范化成员序**（`scripts/cc_build_features.py`）：
   - `members` 改为从 `inputs/family_series_admission.json` 展开（[CL, Q, DS]），
     endpoint plan 只做集合一致性交叉校验；
   - 逐行断言 `member_idx[i] == admission.members_order.index(rows[i].model_id)`；
   - `members_order_sha256 = sha256("\n".join(members)+"\n") = 1afd29b6…`（与本机 audit
     的 admission 值一致；旧错误序 `e6d7852c…`）；
   - 断言与哈希写入 `features/feature_member_order_check.json`，**先于** bundle 落盘。
2. **哈希写入 manifest**：`features_manifest.json` 增 `members_order_sha256`、
   `row_mapping_sha256`（§12 公式；约定：全部行 `model_id|task_id|split|solution_sha256`
   以 `\n` 连接、无尾换行；另存两个变体）、`canonicalization`、`assertions`、
   `script_sha256`；bundle 增 `member_ids` 数组。
3. **防回归断言**（`scripts/cc_common.py`）：`load_design()` 现校验 bundle 的
   `member_ids == admission members_order` 且 `member_idx` 与逐行重算完全一致，失败即停止。
4. **重建与验证**：新旧 bundle 逐数组 bitwise 比对——除 `member_idx`（按置换
   `[0,1,2,3,8,9,10,4,5,6,7]`）与新增 `member_ids` 外，全部数组相同；
   `emb_task_small.npz` bitwise 不变（`f1e75299…`）。新 bundle `d05f8c89…`（旧 `be5cc9d6…`）。
5. **规范序 C0**（新脚本 `scripts/cc_fresh_c0_canonical.py`，保留折内词法拟合）：
   - 每折 manifest v3：`feature_mapping`（members_order/row_mapping/check/bundle/manifest
     SHA + member_ids）、`script_sha256`、`runtime_attestation`；
   - 折内新增 canonical 断言：正类行 = heldout 成员、负类行 = 跨家族成员；
   - 受控对比段：CodeLlama 折不变性 vs 上一版 permuted run；Q/DS 七折语义重定义表；
   - score digests v3 附 fused 统计；`runtime_attestation.json` 单独落盘。

## 2. 结果（dev，服务器内部 train/dev 口径）

- row 均值 **.9442**、task-macro **.9585**、pooled .9427
  （bootstrap CI：row [.9346, .9531]；tm [.9491, .9672]）；inner row .9491 / tm .9666。
- **不变性证据**：4 个 CodeLlama 折 fit/eval 行集合与 y 相同，fused `max|Δ| = 0.0`、
  `r = 1.0` —— 本次修复只改映射、不改任何计算。
- **重定义证据**（旧折标签实际留出的成员）：旧“Q-1.5B 折”实为 DS-1.3b、旧“Q-14B”实为
  DS-33b、旧“Q-32B”实为 DS-6.7b、旧“Q-7B”实为 Q-1.5B、旧“DS-1.3b”实为 Q-14B、
  旧“DS-33b”实为 Q-32B、旧“DS-6.7b”实为 Q-7B —— 与本机报告的循环错位完全吻合。
- 确定性自检：复拟合 `max|Δ| = 0.0`（semantic、char），char 词表哈希一致。
- 历史包诊断（canonical 映射重算；非闸门）：r .8946–.9418、max|Δ| 1.02–2.01。
- 关键 SHA：members_order `1afd29b6…`；row_mapping `b38a41e4…`；bundle `d05f8c89…`；
  脚本 `cc_fresh_c0_canonical.py a3911260…`、`cc_common.py 851b62e9…`、
  `cc_build_features.py 15304671…`（均已写入 metrics/manifest，闭合 §11 的 provenance 补强项）。

## 3. 闸门状态与下一步

- 协议审计（词法折内拟合 + §12 映射断言/哈希）通过；C0 状态
  `prepared (canonical member order); cross-side score comparison pending`。
- 下一步仅一项（裁定 #4）：本机用 v3 `spec_sheet.json` 复现并逐折对账——
  词表预期 **11/11**；fused/组件 digest 需在匹配 runtime 下比较，或采用服务器侧
  digest + `runtime_attestation.json`（Py 3.12.14 / NumPy 2.2.6 / sklearn 1.9.1）对账；
  仍要求 `row_score_max_abs ≤ 1e-3` 且 `metric_abs ≤ 1e-3` 才可标记 `aligned`。
- 在此之前：C1–C3 不重跑；test、生成、权重下载、代码执行继续关闭；不增加 backbone、
  参数量、温度、融合权重或新的变换。

## 4. 交付物清单

- 新目录 `d-det/artifacts/code_conditioned_fresh_c0_canonical_2026-10-10/`：
  README / report.md / metrics.json / score_digests.json(v3) / spec_sheet.json(v3) /
  runtime_attestation.json / feature_order_fix_summary.json / manifests v3 ×11 +
  manifests_index.json / scripts 冻结副本 / commands.sh / logs / SHA256SUMS.txt
  （`local/` 分数缓存按惯例不入库）。
- features：`bundle.npz`（更新）、`features_manifest.json`（更新）、
  `feature_member_order_check.json`（新增）。
- 脚本：`cc_build_features.py`、`cc_common.py`、`cc_fresh_c0_canonical.py`。
- 文档：本记录 + 上游指导下发文档与 ACL 路线的相应更新（随仓库 main 同步）。
