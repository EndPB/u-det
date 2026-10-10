# d-det AutoDL 执行记录：H3 修订 v1 包接收核验与待命（2026-10-11）

## 0. 输入

- 提交：`1704348`（本机侧，"Record H3 local revision v1 gate"）。
- 产物：`d-det/artifacts/h3_local_revision_v1_2026-10-10/`（19 件 + `SHA256SUMS.txt`）；
  派生数据 `.gitignore` 于 `d-det/data/h3_stacad_revision_v1/`；新文档
  `d-det_H3本机修订v1结果与下一轮数据指导_2026-10-11.md`；脚本 `scripts/h3_local_revision_v1.py`。
- 本机运行侧：win32 / Python 3.12.3 / tree-sitter 0.26.0 + 七语言 parser；生成时本机 HEAD
  `e866c82a`（2026-10-01，main 历史内）；`test_read_modeling=false`、`weights=false`、
  `code_execution=false`、`generation=false`。

## 1. 文件完整性（服务器侧，无 GPU 计算）

- 原始口径直接匹配 4/20：`all_train_scores.npz`、`balanced_train_scores.npz`（二进制）、
  `hypothesis.md`、`variant_records.jsonl`。
- LF→CRLF 重建口径匹配 15/15：其余文本文件（win32 源文件为 CRLF，仓库存储规范化为 LF，
  故列表摘要在原始口径下不匹配）。
- `run.log`：列于 `SHA256SUMS.txt` 但未随提交（命中 `d-det/.gitignore` 与根 `.gitignore`
  的 `*.log`，未 `-f` 添加）→ 1 项不可读，属遗留项。

## 2. 独立数值复算（服务器侧重算，非引用）

从两个 npz 的逐行分数重算 task-macro AUROC（primary 子集 = dev_clean 659 task × 8 行 = 5,272 行）：

| 读数 | 声称（文档/metrics） | 复算 |
|---|---:|---:|
| balanced length（log1p） | .8231 | .8231 |
| balanced length（旧 raw） | .8042 | .8042 |
| balanced lexical | .9084 | .9084 |
| balanced AST-shape | .6591 | .6591 |
| balanced metadata / source | .5000 / .5000 | .5000 / .5000 |
| all-train length（log1p） | .8628 | .8628 |
| all-train lexical | .9264 | .9264 |

- 8/8 点估计与产物记录一致（显示精度）。
- bootstrap CI 复算近似一致（例：balanced length 复算 ≈ [.807, .840]，记录 [.8072, .8390]；
  RNG/百分位口径未完全复现）→ CI 以 `metrics.json` 为准。
- `transform-only` = .5670 [.5293, .5977]，与文档一致。

## 3. 溯源与结构核对

- `hypothesis.md` sha256 = `2d04a1d2…` = env.json `hypothesis_sha256` ✓；
  `scripts/h3_local_revision_v1.py` sha256 = `22fe1523…` = env.json `script_sha256` ✓。
- `parser_audit.json`：rows_audited = 22,048、invalid_rows_retained_for_modeling = true、
  test_deserialized = false ✓。
- `balance_manifest.json`：train 2,096 task、dev 659 task、matched dev_rows 2,636；
  8,384 = 2,096×4、5,272 = 659×8 ✓。
- 变体：15,191 抽样 = 8,595 接受（1,430 comment + 2,380 rename + 4,785 whitespace）+ 6,596 拒绝
  （parent_parse_error / no_scope_safe_candidate / unsafe_dynamic_or_macro / ast_mismatch /
  token_mismatch 类目齐全）✓。
- 排除裁定：4 py + 1 go 任务；22,080 − 40 = 22,040 = 16,768 + 5,272 ✓。
- 派生大数据 5 个 `.jsonl` 的摘要仅在 `data_manifest.json` 记录（未随提交）——服务器侧不可复核，
  属预期（冻结包上传时再核对）。

## 4. 结论

- **接收通过**（含两项遗留注记）：完整性、数值、溯源、结构四类核对完成，无阻断项。
- 闸门仍 **`revise_data`**（balanced length .8231 > .70 预注册停止阈值；lexical 控制 .9084 保留）
  → AutoDL **不启动 GPU 批次**；开关保持全 false；C0–C3 关闭结论不变。
- 遗留（建议本机下一轮处理）：
  1) `run.log` 如需入库：`git add -f` 或改用非 `*.log` 文件名；
  2) SHA256SUMS 行尾口径（win32 CRLF 计算 vs 仓库 LF 存储）建议统一（`-text` 保留行尾或按 LF
     字节出具摘要），便于跨平台一键校验。

## 5. 交付物

- 本记录；ACL 版本日志行；随本次提交推送。
