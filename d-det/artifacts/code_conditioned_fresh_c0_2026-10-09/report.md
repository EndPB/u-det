# Fresh C0 重建报告（train/dev，2026-10-09）

上游：`code_conditioned_p0_provenance_adjudication_2026-10-09/`（裁决 + `fresh_c0_protocol.json`）
与指导文档 §9。**唯一允许动作：一次 fresh train/dev C0 重建**；C1–C3 重跑需 fresh C0
先对齐。开关：train/dev only；test_read / generation / weights_downloaded /
code_execution 全 false（manifest 中逐折记录）。

## 1. 结果摘要

- **执行完成**（114 s；`scripts/cc_fresh_c0.py`）：
  - dev：row-level 均值 **.9329**、task-macro 均值 **.9442**、pooled .9323
    （bootstrap CI：row [.9228, .9424]；tm [.9338, .9530]）；
  - inner（train-only 内部切分，次要诊断）：row .9301、tm .9515、pooled .9303。
- **确定性**：折 5（Qwen2.5-Coder-1.5B）复拟合 max|Δ| = 0.0（semantic、char）；
  本次 fresh 与此前 `2968ff9` C0 逐折 **max|Δfused| = 0.0、ev_rows 全同**（跨会话逐位复现）。
- **Manifest**：11 折 manifest 在**拟合开始前**写入并哈希（`manifests/` +
  `manifests_index.json`，含协议要求的全部字段：effective_series_map_sha256、
  heldout_member/heldout_series、fit/eval_member_ids、positive/negative/train 行哈希、
  eval_task 哈希、component_specs、code_commit）。
- **分数摘要**（`score_digests.json`）：逐折对 fused 与四组件（含 train 侧）给出
  canonical float64 数组 sha256，供本机按 `spec_sheet.json` 复现后逐位比对。
- **对齐闸门**（预注册字段）：`row_score_max_abs ≤ 1e-3`、`metric_abs ≤ 1e-3`、
  全折 effective map 已记录 —— 前两项等待**对侧 fresh C0**（本机按同规格复现/重建）
  才能判定；本次记录 `status=prepared`。

## 2. 与历史包的关系（诊断，非闸门）

历史包按裁决降级为 `map_consistent_but_spec_and_per_fold_provenance_incomplete`。
在“官方读法”（历史包块序 = ASCII 排序 eval 成员 × 排序任务；用 (member, task) 恒等
逐行对齐）下与本次 fresh C0 的逐行对比（诊断）：

| 折（目录名） | n | y 一致 | r(fused) | max\|Δ\| | row 本次/历史 | tm 本次/历史 |
|---|---|---|---|---|---|---|
| CL-13b | 1368 | ✓ | +0.599 | 2.446 | .9557/.9281 | .9607/.9348 |
| CL-34b | 1368 | ✓ | +0.571 | 2.148 | .9500/.9112 | .9716/.9332 |
| CL-70b | 1368 | ✓ | +0.570 | 2.641 | .9379/.9038 | .9632/.9273 |
| CL-7b | 1368 | ✓ | +0.605 | 2.225 | .9767/.9680 | .9774/.9724 |
| Q-1.5B | 1368 | ✓ | +0.420 | 3.048 | .9709/.8812 | .9825/.9231 |
| Q-14B | 1368 | ✓ | +0.492 | 2.816 | .8453/.9801 | .8663/.9841 |
| Q-32B | 1368 | ✓ | +0.588 | 2.252 | .9380/.9851 | .9657/.9883 |
| Q-7B | 1368 | ✓ | +0.399 | 2.652 | .7685/.9833 | .7736/.9808 |
| DS-1.3b | 1539 | ✓ | +0.283 | 3.388 | .9701/.9192 | .9766/.9284 |
| DS-33b | 1539 | ✓ | +0.328 | 4.310 | .9880/.9013 | .9912/.9137 |
| DS-6.7b | 1539 | ✓ | +0.231 | 5.468 | .9605/.9175 | .9569/.9423 |

结论（与 §9 一致）：l**abels 与 task 顺序完全一致**；连续分数差距**无法由已保留的
每折规格解释**（即处于 `not_aligned`），且不属于本次 fresh C0 能单独闭合的部分——需要
本机按 `spec_sheet.json` 复现/新建对侧 fresh C0 后再判定闸门。`edf3813` 的 7-cycle/
分区推断按裁决保留为 `score_block_alignment_hypothesis`（见对齐目录 v1/v2 记录），
本目录不做目录改名或标签重写。

## 3. 规格清单（供本机复现；完整版见 `spec_sheet.json`）

- 语义：`h_y` = CodeT5-small 冻结 mean-pool 嵌入（512 维；r0 语料缓存 instruct 行）；
  StandardScaler（折 fit 行）+ LogisticRegression(max_iter=2000, C=1.0)，decision_function。
- char：TfidfVectorizer(char_wb, (2,5), min_df=1, sublinear_tf=True, lowercase=False)，
  **词表/IDF 仅折 fit 行**；SGDClassifier(log_loss, alpha=1e-6) × seeds(0,1,2)，5 epoch
  shuffled partial_fit、batch=4096（seed 局部 RNG），三 seed 平均 decision_function。
- word：同上，analyzer=word，token_pattern=[A-Za-z_][A-Za-z0-9_]*，(1,3)。
- style/meta：concat(style 92, meta 10, sizelen 3) + StandardScaler + LR(C=1)。
- 融合：逐组件用折 fit 行分数的 mean/std（population）标准化后等权平均（keys 排序）。
- 目标：`y_fit = 1 iff 行成员系列 == heldout 系列`。
- eval 布局：ASCII 排序的 eval 成员块 × 排序任务；任务切分沿用 798/171；
  inner 切分规则 `sha256('cc_inner_split_v1|'+task_id) % 5 == 0`。
- 指标：AUROC ties=0.5；task-macro = 逐任务 AUROC 均值；500 次 task-cluster
  bootstrap（seed 20261009，共享重采样索引）。
- 环境：python {sys.version}、numpy/sklearn 版本见 `spec_sheet.json`（复现时优先对齐）。
- 摘要约定：`score_digests.json` = 对 canonical float64 数组字节的 sha256
  （eval 数组按 taskpos 序；train 数组按 fit-row 序）。

## 4. 判定与下一步

- Fresh C0：已重建、可逐位复现、manifest 完整、全折 map 已记录；
  闸门 `row_score_max_abs/metric_abs ≤ 1e-3` **待对侧对照**后判定。
- C1–C3：**未重跑**（按 §9 需 fresh C0 对齐后）；现有 C1–C3 数字维持
  “服务器内部开发诊断”表述。
- 待本机：按 `spec_sheet.json` 在同一数据上复现 fresh C0（或提供其 fresh 规格），
  用 `score_digests.json` 逐折比对，达到 1e-3 后闸门闭合。

## 5. 产物

本目录：`manifests/`（11 折 + index）、`metrics.json`、`score_digests.json`、
`spec_sheet.json`、`report.md`（本文件）、`README.md`、`commands.sh`、
`logs/nohup.log`、`logs/cc_fresh_c0.log`、`git_head.txt`、`git_status.txt`、
`SHA256SUMS.txt`；分数 npz 在 `local/`（不入库）。
