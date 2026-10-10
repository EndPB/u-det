# d-det AutoDL 执行记录：H3 GPU 诊断批次执行（2026-10-11）

## 0. 授权与输入

- 指导：`d-det/docx/d-det_AutoDL_H3_GPU诊断批次指导_2026-10-11.md`（一次性执行版；
  "GPU 诊断批次"，正式 H3 闸门仍为 `revise_data`）。
- 钉版清单：`d-det/artifacts/h3_gpu_diagnostic_batch_2026-10-11/batch_manifest.json`
  （schema `h3_gpu_diagnostic_batch_manifest_v1`；`code_commit_required=98a9f79`；24 runs 预注册）。
- 服务器对齐：快进 `1e584c2 → fcea025`（脚本与 `98a9f79` 相同，sha256 `6707e57f…`）；
  数据三件 sha256 与钉版逐一核对通过；模型 `d-det/models/codet5-small` 本地存在（未下载）。

## 1. 执行前核对（全过）

- CUDA 断言通过：NVIDIA GeForce RTX 3080 Ti 11.6 GiB，torch 2.9.1+cu128，cuda 12.8。
- 数据 sha256：`train_balanced=152602c3… / dev_clean=fe7955a5… / train_variants=e4d38e93…` ✓
- 开关：`test_read=false`、`generation=false`、`weights_downloaded=false`。

## 2. 执行

按指导 §3 原样执行（命令见 `commands.txt`；GPU 监控 10s 采样）。24/24 运行全部完成，
`BATCH_STATUS=0`。编码 22,251 个唯一代码（median tokens 512、68.6s）；整批约 2 分钟
（02:21 → 02:23:22 +08）。

GPU 监控稳定段：942 MiB / 58–67% 利用率 / ~280–290 W（峰值温度 57°C）——确认为真实
CUDA 训练，非文件核验。

## 3. 结果（服务器 512-token，详见 `metrics.json`）

| job | runs | detection AUROC mean | detection BA mean | source seen acc mean |
|---|---:|---:|---:|---:|
| detection_only | 6 | .6723 | .5486 | — |
| source_only | 6 | — | — | .1473 |
| joint | 6 | .6761 | .5489 | — |
| joint_invariance | 6 | .6765 | .5488 | — |

- 逐折：fold_0（heldout = claude_3_haiku、deepseek_v3_2、devstral_2512）det .686–.706；
  fold_1（heldout = gemini_3_flash、gpt5_nano、grok_4_fast、qwen3_coder_30b）det .649–.657。
- source：seen n=4,613（acc .1429–.1496；7 类机会 .1429），unseen 659 行显式标记 unsupported；
  generator-heldout 折内按设计报告 `no_seen_source_in_eval`。
- 增量：joint−detection_only = **+0.39pt**；joint_invariance−detection_only = **+0.42pt**
  （方向为正、幅度 <0.5pt，诊断读数）。
- 对照：本机 len128 低资源读数 .7221/.7275/.7280（设备与截断长度不同，仅参考）。

## 4. 判读

- generator-heldout 检测 ≈ .67，远低于 v1 lexical 强控制（.9084）；source 读出 ≈ 机会；
  joint/一致性增量 <0.5pt → **不构成 H3 主张量**；正式闸门维持 `revise_data`
  （length .8231 > .70 阈值）。符合"不加容量/温度/轮数/损失项"的停止规则。
- 开关保持 `test_read=false`、`generation=false`、`weights_downloaded=false`；
  `code_execution` 仅为本批次脚本自身执行（无样本代码执行）。

## 4.1 同折 lexical 控制复核

本机随后按服务器实际两个 detection 折重新拟合原 v1 lexical 控制，避免把全部 generator 拟合的 `.9084` task-macro AUROC 与 generator-heldout GPU row AUROC 直接比较。fold_0 / fold_1 的 lexical row AUROC 为 `.7849/.7223`，折均值 `.7536`；对应 detection-only GPU row AUROC 为 `.6916/.6530`，折均值 `.6723`。joint 与 joint-invariance 仅为 `.6761/.6765`，增量 `.39/.42pt`。复核不是新 GPU 实验，证据目录为 `artifacts/h3_matched_fold_closeout_2026-10-11/`。

另外，`source_only` 在两个 fold 名下使用同一完整 train/dev 与同一三枚 seed，实际是 3 个独立运行的重复展示；`unseen_rows=659` 是 human 行，不代表 unseen generator AI 行。以上两点已纳入正式边界。

## 4.2 服务器侧独立复核（closeout 包，2026-10-11）

- **完整性**：`SHA256SUMS.txt` 7/7 原始口径通过；`server_metrics_snapshot.json` 与批次 `metrics.json`（`feeb93d`）字节一致（sha `72045d13…`，与 `config.json` 的 `source_metrics_sha256` 一致）。
- **npz 复算**：从 `fold_*_lexical_scores.npz` 重算 fold_0 row `.7848666` / tm `.9175518`、fold_1 row `.7222786` / tm `.8378225`（与 `metrics.json` 一致至 ~1e-7）；折成员核查通过（eval = 659 human + 该折 heldout generators；n=2636/3295），`membership_bad=0`。
- **GPU 行读数复算**：与快照一致——fold_0 det `.6916` / joint `.6976` / inv `.6983`；fold_1 `.6530/.6546/.6547`。
- **独立复拟合**（服务器 sklearn 1.9.1/numpy 2.2.6 vs 本机 1.7.1/1.26.4；同协议）：fold_0 row `.784867` / tm `.917552`（与记录一致）；fold_1 row `.722321` / tm `.837064`（差 +0.004pt / −0.08pt，伴随 liblinear 迭代 64/60 vs 65/61，属 sklearn 版本差）；逐行分数 `corr ≥ 0.9999`、`max|Δs| ≤ 0.036`。
- **结论**：closeout 两点边界修正（`source_only` 两折=同一数据 3 seed 重复；`unseen_rows=659` 为 human 行）与裁定经服务器侧复核**成立**；同折 lexical row 控制高于 detection-only 约 8.1pt，本实例化无有效增量，诊断关闭，不重跑、不加容量/温度/轮数/损失项。

## 5. 产物与回传

- `metrics.json`（24 runs + 聚合 + 数据/模型哈希 + 设备）、`run_manifest.json`、
  `run.log.txt`、`gpu_utilization.csv`、`commands.txt`、`git_head.txt`；
  回传 tar：`/tmp/h3_gpu_diagnostic_batch_2026-10-11.tar.gz`
  sha256 `a4cbc1ff85c314ad8e6953b0d4eac025a65b8d9dd6dfea1ecb7e24200811c23b`（7 件；监控进程终止后重打包，csv 与仓库一致）。
- 仓库提交推送为正式回传；本地另存 tar 供直接下载。

## 6. 备注

- 执行前工作树中的三份未跟踪草稿与钉版内容一致（清单仅排版差异），已备份 `/tmp/h3_draft_backup/`；
  以仓库钉版为准。
- GPU 监控进程因复合命令后台化（`mkdir && nvidia-smi &`）延迟停止：kill 命中的是子壳、
  nvidia-smi 成为孤儿并继续追加 CSV；已按 PID 强制终止并重打包，`gpu_utilization.csv`
  冻结为 18 行（运行段 + 回归空闲）。
