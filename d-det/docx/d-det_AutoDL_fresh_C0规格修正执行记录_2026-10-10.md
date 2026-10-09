# d-det AutoDL 执行记录：fresh C0 规格修正（2026-10-10）

执行方：AutoDL 服务器（/root/autodl-tmp/u-det）。
上游：`code_conditioned_fresh_c0_spec_adjudication_2026-10-09/`（裁定
`spec_violation_global_lexical_fit`）、指导文档 §10。
开关：train/dev only；test_read=false；generation=false；weights_downloaded=false；
code_execution=false（逐折 manifest 记录）。

---

## 1. 状态

**修正版 fresh C0 完成**（343 s），取代 `code_conditioned_fresh_c0_2026-10-09`
（blocked_before_gate）。修正点：char/word `TfidfVectorizer` 从“折外对全部
train/dev 文本 fit”改为“**每折仅对 fit_rows 文本 fit，eval 行只 transform**”。
C1–C3 未重跑（保持停止）；旧值继续仅作服务器内部开发诊断。

- dev：row **.9328** / tm **.9451** / pooled .9320（CI row [.9228,.9421]、
  tm [.9350,.9540]）；
- inner：row .9296 / tm .9520；
- 确定性：折 Q-1.5B 复拟合 max|Δ|=0.0，词表 sha 一致；泄漏修复影响很小
  （与阻断版 r=0.997–0.998，Δtm 均值 +0.09pt，max|Δ|≤0.42）——量级无变化，
  但按协议修正版为准。

## 2. 按裁决要求落实的项

1. **两个 vectorizer 移入每折**，只用 `fit_rows` 文本 fit、`ev_rows` 仅 transform；
2. **每折 manifest 追加**（v2，pre/post 双哈希）：
   - 拟合前 `lexical_fit_spec`：vectorizer 参数、`fit_rows_key_sha256`、
     `fit_texts_sha256`（fit 文本按 fit 行索引序 + 0x00 分隔）、行数；
   - 拟合后 `lexical_fit_attestation`：char/word 的 `n_features`、`vocab_sha256`、
     `idf_sha256`、IDF 极值；`manifest_sha256_pre_fit` / `manifest_sha256_post_fit`；
3. **重算** C0 指标与 `score_digests.json`（fused + 四组件、dev/inner、含 train 侧）；
4. 审计报告：`report.md`（含逐折影响表与历史包诊断）；`spec_sheet.json` 更新为 v2。

## 3. 逐折结果（dev，修正版）

CL-13b .9556/.9574｜CL-34b .9485/.9724｜CL-70b .9381/.9624｜CL-7b .9773/.9783｜
Q-1.5B .9685/.9833｜Q-14B .8488/.8697｜Q-32B .9361/.9632｜Q-7B .7688/.7769｜
DS-1.3b .9707/.9766｜DS-33b .9873/.9912｜DS-6.7b .9605/.9642（row-level / task-macro）。

## 4. 闸门与后续

- 协议审计层面：满足（折内词法拟合 + 全折 attestation + 双哈希清单 + 分数摘要）；
- **`1e-3` 跨侧闸门待对侧**：本机按 `spec_sheet.json` 在该数据上复现修正版，
  以 `score_digests.json` / 逐行分数比对；闸门闭合前 C1–C3 保持停止。

## 5. 产物与提交

- 产物：`d-det/artifacts/code_conditioned_fresh_c0_corrected_2026-10-10/`
  （manifests/ 11 折 v2、manifests_index、metrics、score_digests、spec_sheet、
  report、README、commands、logs、git_*、SHA256SUMS；分数 npz 在 local/ 不入库）。
- 脚本：`scripts/cc_fresh_c0_corrected.py`。
- 提交：本轮一并提交推送（含规格裁决目录与更新后的指导/ACL 文档）。
