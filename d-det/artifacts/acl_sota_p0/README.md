# artifacts/acl_sota_p0 — ACL 家族归因 SOTA 主线 P0（强基线复现）

本轮（2026-10-04，指导：`docx/d-det_AutoDL_1f3d547_ACL家族归因SOTA执行指导_2026-10-04.md`）
两项工作：

## 1) stacad_official_repro/ — STACAD 官方协议复现（纯 CPU）

- `config.json`：复现参数、依赖版本、features_v2.npz/protocol.json 的 SHA-256；
- `metrics_repro.json`：classic / learners / stack 的本机复现指标（5 折 OOF macro-F1 mean±std、ECE/NLL/Brier）；
- `compare_vs_official.json`：与随包发布结果（`cache/results/journal_v2/*.json`）的逐项对比；
- OOF 概率矩阵（`*_oof_repro.npz`、`stack_probs_*.npz`）为 26MB 级缓存，**不入库**（可由
  `scripts/acl_sota_p0_stacad_repro.py --stages classic learners stack` 重算）。

复现脚本：`scripts/acl_sota_p0_stacad_repro.py`；日志：`logs/`。
CatBoost 未安装（官方 cb 行以引用形式给出）；xgboost 使用 CPU wheel（xgboost-cpu 3.4.1）。

## 2) server_tracks/ — 三赛道 P0 基线套件

- `authorbench_dcan/`：主赛道（task 留出；6 families）
- `stacad_fold0/`：STACAD round4-C 协议（file 留出；官方 105 特征与行空间逐行对齐）
- `droid_fold0/`：Droid generator 留出（7 families）

每目录含：`config.json` / `env.json` / `metrics.json`（全部行：macro-F1+CI95+balanced acc+ECE+逐类召回）/
`predictions_all.npz`（dev/test 概率，含逐 seed 或均值）与逐模型 `*_probs.npz`。

复现脚本：`scripts/acl_sota_p0_baselines.py`；聚合：`scripts/acl_sota_p0_summary.py`
（生成上级目录 `summary_p0_v1.json/.md`）；日志：`logs/run.log`。

## 重要提示

- 复排行（codet5_* / tfidf_char_round4c）**只重算指标**，预测来自既有 round3/round4 产物；
- `fusion_lr` 为 dev 拟合的 late-fusion（test 只评估一次）；`dev_macro_f1` 字段对融合行非 out-of-sample；
- STACAD 官方包解压物在 `data/stacad_v2/extracted/`（不入库）。
