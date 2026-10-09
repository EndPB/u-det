# d-det AutoDL 执行记录：fresh C0 重建（2026-10-09）

执行方：AutoDL 服务器（/root/autodl-tmp/u-det）。
上游：`code_conditioned_p0_provenance_adjudication_2026-10-09/`（provenance 裁决 +
`fresh_c0_protocol.json`）、指导文档 §9。
开关：train/dev only；test_read=false；generation=false；weights_downloaded=false；
code_execution=false（逐折写入 manifest）。

---

## 1. 状态

**§9 唯一动作（fresh train/dev C0 重建）已完成**；C1–C3 未重跑（按 §9 需 fresh C0
对齐后）。提交：见文末。

- dev：row-level **.9329**、task-macro **.9442**、pooled .9323
  （bootstrap CI：row [.9228, .9424] / tm [.9338, .9530]）；
- inner（train-only 内部切分）：row .9301、tm .9515；
- 确定性：折 Q-1.5B 复拟合 max|Δ|=0.0；与本轮前 C0（2968ff9）逐折
  **max|Δfused|=0.0、ev_rows 全同**（跨会话逐位复现）；
- 全流程 114 s（`scripts/cc_fresh_c0.py`）。

## 2. 每折 manifest（拟合前写入并哈希）

`manifests/`（11 折 JSON）+ `manifests_index.json`，全部字段按协议：
`effective_series_map_sha256`（canonical map 内容哈希；另附官方期望 admission sha
与服务器 admission 文件 sha）、`heldout_member`、`heldout_series`、`fit_member_ids`、
`eval_member_ids`、`positive_row_sha256`、`negative_row_sha256`、`train_row_sha256`、
`eval_task_sha256`、`component_specs`（+sha）、`code_commit`；含行数计数与生成时间。

## 3. 分数摘要与复现接口

- `score_digests.json`：逐折 fused 与四组件（含 train 侧）的 canonical float64 数组
  sha256（eval 数组 = taskpos 序；train 数组 = fit-row 序）；
- `spec_sheet.json`：完整复现规格（四组件/融合/指标/环境版本）；
- 对齐闸门字段已记录：`row_score_max_abs ≤ 1e-3`、`metric_abs ≤ 1e-3`、
  `all_fold_effective_map_logged=true` —— **前两项待对侧 fresh C0（本机按同规格复现）
  后判定**；本目录 `status=prepared`。

## 4. 与历史包的诊断对照（官方读法；非闸门）

以 (member, task) 恒等逐行对齐（本新 npz 为任务序、历史包为成员块序）：

| 折 | n | y 一致 | r | max\|Δ\| | row 本次/历史 | tm 本次/历史 |
|---|---|---|---|---|---|---|
| CL-13b | 1368 | ✓ | +.599 | 2.446 | .9557/.9281 | .9607/.9348 |
| CL-34b | 1368 | ✓ | +.571 | 2.148 | .9500/.9112 | .9716/.9332 |
| CL-70b | 1368 | ✓ | +.570 | 2.641 | .9379/.9038 | .9632/.9273 |
| CL-7b | 1368 | ✓ | +.605 | 2.225 | .9767/.9680 | .9774/.9724 |
| Q-1.5B | 1368 | ✓ | +.420 | 3.048 | .9709/.8812 | .9825/.9231 |
| Q-14B | 1368 | ✓ | +.492 | 2.816 | .8453/.9801 | .8663/.9841 |
| Q-32B | 1368 | ✓ | +.588 | 2.252 | .9380/.9851 | .9657/.9883 |
| Q-7B | 1368 | ✓ | +.399 | 2.652 | .7685/.9833 | .7736/.9808 |
| DS-1.3b | 1539 | ✓ | +.283 | 3.388 | .9701/.9192 | .9766/.9284 |
| DS-33b | 1539 | ✓ | +.328 | 4.310 | .9880/.9013 | .9912/.9137 |
| DS-6.7b | 1539 | ✓ | +.231 | 5.468 | .9605/.9175 | .9569/.9423 |

labels/task 顺序完全一致；连续分数差距未被已保留规格解释（`not_aligned` 维持）。
`edf3813` 的 7-cycle/分区推断保留为 `score_block_alignment_hypothesis`（对齐目录
v1/v2 记录）；未做任何目录改名或标签重写。

## 5. 下一步（待本机）

1. 按 `spec_sheet.json` 在同一数据上用同规格复现 fresh C0（或给出本机 fresh 规格）；
2. 用 `score_digests.json` 逐折比对（或直接对分数数组），达到 1e-3 后闸门闭合；
3. 闸门闭合前，C1–C3 维持“服务器内部开发诊断”表述，不重跑、不新增解释。

## 6. 产物与提交

- 产物：`d-det/artifacts/code_conditioned_fresh_c0_2026-10-09/`（manifests/、
  metrics.json、score_digests.json、spec_sheet.json、report.md、README、commands、
  logs、git_*、SHA256SUMS；分数 npz 在 local/ 不入库）。
- 脚本：`scripts/cc_fresh_c0.py`（仓库与本目录 scripts/ 各一份）。
- 提交：本次一并提交推送（含裁决目录与本记录）。
