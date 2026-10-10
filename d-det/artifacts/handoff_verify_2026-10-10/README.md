# 交接验收回执：handoff_2026-10-10（2026-10-10）

按本机指示：`handoff_2026-10-10/` 不在其中工作；内部文件已全部迁移到规范位置，
壳目录已移除。本目录保存交接包的封存材料与校验日志。

## 校验结果

- `unpacked/SHA256SUMS.txt`（CRLF 剥离后）：**10/10 OK**（含 2 docx、1 指南、upload zip、
  upload_hashes、README、split_manifest、summary、2 scripts）。
- 外层包 `autodl_handoff_2026-10-10.zip`：与 `.zip.sha256` 一致 **OK**。
- 逐文件复核（复制到目标位置后）：与 `HANDOFF_MANIFEST.json` 中 sha/bytes 逐一相符：
  - `d-det/docx/d-det_ACL总研究总结与可迭代路线_2026-10-10.md` — 63f5cae9…
  - `d-det/docx/d-det_AutoDL_H3数据构建与UIT-AMMC反捷径指导_2026-10-10.md` — 54181671…
  - `scripts/audit_stacad_alignment_v1.py` — 0787e39f…
  - `scripts/build_stacad_task_alignment_v1.py` — 18cd1f60…
  - `d-det/data/h2_stacad_alignment_v1/h2_stacad_alignment_v1_upload.zip` — fec137fe…

## 保存内容

- `AUTODL_AI_HANDOFF_GUIDE_2026-10-10.md`、`HANDOFF_MANIFEST.json`、
  `handoff_SHA256SUMS.txt`（原交接 SHA256SUMS）、`autodl_handoff_2026-10-10.zip`
  与其 `.sha256`、`logs/verify.log`。

## 迁移去向

- 文档 → `d-det/docx/`；脚本 → `scripts/`；数据（8 文件 + upload zip）→
  `d-det/data/h2_stacad_alignment_v1/`（该目录按既有约定被 .gitignore，数据不入库；
  解包验证副本在 `d-det/data/h2_stacad_alignment_v1_received/`，同样不入库）。
- 接收审计与 probes 结果：`d-det/artifacts/h3_data_gate_stacad_2026-10-10/`。

## 后续更新（2026-10-10 晚）

- 交接指南已更新：本目录封存的 `AUTODL_AI_HANDOFF_GUIDE_2026-10-10.md`（`999ec792…`）
  为**交付时点版本**；现行为 `d-det/docx/AUTODL_AI_HANDOFF_GUIDE_2026-10-10.md`
  （`263f42b8…`，含 H3 闸门结果 §8.1 与“完整批次”执行模型 §7）。旧 10-07 指南归档到
  `d-det/docx/archive/`。
- 职责边界修订：CPU/小显存任务本机直接完成；AutoDL 待命，闸门通过后按
  `batch_manifest.json` 一次性执行完整 GPU 批次。
