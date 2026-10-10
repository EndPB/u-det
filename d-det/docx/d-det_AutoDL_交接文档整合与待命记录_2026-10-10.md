# d-det AutoDL 执行记录：交接文档整合与待命（2026-10-10 晚）

## 0. 背景

本机更新了 H3 入口裁定（**本机直接完成数据侧修复**：长度分桶平衡、七语言 parser/
变体、两组骨架重复裁定、probes 重跑；**AutoDL 在闸门通过前不启动 H3**），发布更新版
交接指南（含闸门结果 §8.1 与“完整批次”执行模型 §7），并在仓库根遗留两份临时指南
文件（旧名重定向 stub + 与 docx 相同的副本）。

## 1. 本轮动作（文档整合，无计算）

1. **指南定稿入库**：`d-det/docx/AUTODL_AI_HANDOFF_GUIDE_2026-10-10.md`
   （sha `263f42b8…`；写明当前 HEAD `e53293e…`、闸门运行基准 `a65158f…`）。
2. **旧版归档**：`d-det/docx/AUTODL_AI_HANDOFF_GUIDE_2026-10-07.md` →（git mv）
   `d-det/docx/archive/`，并新增 `archive/README.md`（与 ACL/stub 所述归档约定一致）。
3. **根目录清理**：删除未跟踪的根副本 `AUTODL_AI_HANDOFF_GUIDE_2026-10-10.md` 与旧名
   stub `AUTODL_AI_HANDOFF_GUIDE_2026-10-07.md`（内容等价物已在 docx/archive）。
4. **链接修正**：H3 入口文档内指向根目录的指南链接改为同级链接
   （`../../AUTODL…` → `AUTODL…`）。
5. **回执更新**：`d-det/artifacts/handoff_verify_2026-10-10/README.md` 增记“后续更新”
   （封存交付版 `999ec792…` / 现行 `263f42b8…`）。
6. **ACL 版本日志**：追加文档整合与待命条目。

## 2. 服务器状态（待命确认）

- 闸门：H3 = **`revise_data`**（length-only `.8459`、lexical-only `.9261`；证据
  `d-det/artifacts/h3_data_gate_stacad_2026-10-10/`）。
- AutoDL **不启动任何 H3 训练/审计**；未来仅在收到“完整批次指导”（含
  `batch_manifest.json`：data_sha256 / code_commit / fold / seed / model_variant /
  resource_request / output_dir / test_read=false）时**一次性**执行并统一回传。
- 这条待命仅针对 H3；C0–C3 主线按已接受的跨平台容差恢复，不再等待逐行微小差异闭合。
- 开关保持：`test_read(modeling)=false`、`generation=false`、`weights_downloaded=false`、
  `code_execution=false`；旧 C0–C3 不重跑。

## 3. 交付物

- 本记录；更新后的指南/入口/ACL；`docx/archive/`；回执更新；随本提交推送。

## 4. 主线恢复（2026-10-10）

后续 AutoDL 批次优先执行 C0 baseline、C1 prompt-conditioned、C2 static-proxy auxiliary、C3 invariance，统一使用冻结输入、既定折和 seed，并一次性回传相对 C0 的比较结果。H3 继续单独遵守 `revise_data` 闸门。
