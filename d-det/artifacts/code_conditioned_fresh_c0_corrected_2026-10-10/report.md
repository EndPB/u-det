# Fresh C0 修正版报告（fold-fit-only 词法；2026-10-10）

上游：`code_conditioned_fresh_c0_spec_adjudication_2026-10-09/`（裁决
`spec_violation_global_lexical_fit`）与指导文档 §10。**本目录取代
`code_conditioned_fresh_c0_2026-10-09`（blocked_before_gate）成为修正后的 fresh
C0 记录。** 开关：train/dev only；test_read / generation / weights_downloaded /
code_execution 全 false（逐折 manifest 记录）。

## 1. 修复内容

- **词法拟合移入每折**：char 与 word `TfidfVectorizer` 现在在每个折内仅用该折
  `fit_rows` 文本 `fit_transform`，eval 行只做 `transform`（词表/IDF 不再接触 dev
  行与 heldout 成员）。semantic / style-meta 的 scaler 与分类器原本即折内拟合，未变。
- **manifest v2（每折）**：
  - 拟合前写入 `lexical_fit_spec`：vectorizer 参数、`fit_rows_key_sha256`、
    `fit_texts_sha256`（fit 行文本按 fit 行索引序 + 0x00 分隔的 sha256）、行数；
  - 拟合后追加 `lexical_fit_attestation`：char/word 的 `n_features`、`vocab_sha256`
    （排序词表）、`idf_sha256`（float64 字节）、IDF 极小/极大；并记录
    `manifest_sha256_pre_fit` 与 `manifest_sha256_post_fit`（`manifests_index.json`
    含两组哈希）。
- **分数摘要重算**：`score_digests.json`（fused + 四组件、dev/inner、含 train 侧）。
- **确定性自检**（折 Q-1.5B 复拟合）：`max|Δ| = 0.0`（semantic、char），
  `vocab_sha256_match = True`。

## 2. 修正后结果

- **dev**：row-level 均值 **.9328**、task-macro **.9451**、pooled .9320
  （bootstrap CI：row [.9228, .9421]；tm [.9350, .9540]）；
- **inner**：row .9296、tm .9520、pooled .9300。
- 逐折（dev）：CL-13b .9556/.9574、CL-34b .9485/.9724、CL-70b .9381/.9624、
  CL-7b .9773/.9783、Q-1.5B .9685/.9833、Q-14B .8488/.8697、Q-32B .9361/.9632、
  Q-7B .7688/.7769、DS-1.3b .9707/.9766、DS-33b .9873/.9912、DS-6.7b .9605/.9642。

## 3. 泄漏修复的影响（与 blocked 全局拟合版对比）

| 折 | r | max\|Δ\| | row 修正/阻断 | tm 修正/阻断 |
|---|---|---|---|---|
| CL-13b | .998 | 0.290 | .9556/.9557 | .9574/.9607 |
| CL-34b | .998 | 0.278 | .9485/.9500 | .9724/.9716 |
| CL-70b | .998 | 0.355 | .9381/.9379 | .9624/.9632 |
| CL-7b | .998 | 0.418 | .9773/.9767 | .9783/.9774 |
| Q-1.5B | .998 | 0.333 | .9685/.9709 | .9833/.9825 |
| Q-14B | .997 | 0.322 | .8488/.8453 | .8697/.8663 |
| Q-32B | .998 | 0.302 | .9361/.9380 | .9632/.9657 |
| Q-7B | .998 | 0.409 | .7688/.7685 | .7769/.7736 |
| DS-1.3b | .998 | 0.293 | .9707/.9701 | .9766/.9766 |
| DS-33b | .998 | 0.299 | .9873/.9880 | .9912/.9912 |
| DS-6.7b | .997 | 0.303 | .9605/.9605 | .9642/.9569 |

全体：r = 0.997–0.998；Δrow 均值 −0.01pt、Δtm 均值 **+0.09pt**；max|Δ| ≤ 0.42。
即：词法泄漏对结果量级影响很小，但按协议必须修复；**修正版为准**。

## 4. 与历史包的诊断（官方读法；非闸门）

| 折 | r | max\|Δ\| | row 修正/历史 | tm 修正/历史 |
|---|---|---|---|---|
| CL-13b | +.606 | 2.389 | .9556/.9281 | .9574/.9348 |
| CL-34b | +.581 | 2.131 | .9485/.9112 | .9724/.9332 |
| CL-70b | +.580 | 2.614 | .9381/.9038 | .9624/.9273 |
| CL-7b | +.615 | 2.181 | .9773/.9680 | .9783/.9724 |
| Q-1.5B | +.411 | 3.065 | .9685/.8812 | .9833/.9231 |
| Q-14B | +.490 | 2.824 | .8488/.9801 | .8697/.9841 |
| Q-32B | +.585 | 2.305 | .9361/.9851 | .9632/.9883 |
| Q-7B | +.396 | 2.759 | .7688/.9833 | .7769/.9808 |
| DS-1.3b | +.282 | 3.395 | .9707/.9192 | .9766/.9284 |
| DS-33b | +.325 | 4.316 | .9873/.9013 | .9912/.9137 |
| DS-6.7b | +.233 | 5.473 | .9605/.9175 | .9642/.9423 |

历史包仍为 `map_consistent_but_spec_and_per_fold_provenance_incomplete`；以上仅为
记录，不构成闸门结果。

## 5. 闸门与后续

- 修正版 fresh C0：协议审计层面已满足（词法折内拟合 + 全折 attestation +
  map/行/任务/manifest 双哈希 + 分数摘要）；
- **`1e-3` 跨侧分数/指标闸门仍待对侧**：本机按 `spec_sheet.json` 在该数据上复现
  修正版 fresh C0，用 `score_digests.json` 与逐行分数比对（阈值 1e-3）；
- C1–C3：**保持停止**；旧值继续仅作“服务器内部开发诊断”，闸门闭合前不重跑、不升级。

## 6. 产物

`manifests/`（11 折 v2：pre/post 双哈希 + 词法 attestation）、`manifests_index.json`、
`metrics.json`、`score_digests.json`、`spec_sheet.json`、`report.md`（本文件）、
`README.md`、`commands.sh`、`logs/nohup.log` + `logs/cc_fresh_c0_corrected.log`、
`git_head.txt`、`git_status.txt`、`SHA256SUMS.txt`；分数 npz 在 `local/`（不入库）。
