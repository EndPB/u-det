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

## 5. 产物与回传

- `metrics.json`（24 runs + 聚合 + 数据/模型哈希 + 设备）、`run_manifest.json`、
  `run.log.txt`、`gpu_utilization.csv`、`commands.txt`、`git_head.txt`；
  回传 tar：`/tmp/h3_gpu_diagnostic_batch_2026-10-11.tar.gz`
  sha256 `56c5e944b0b9bef91b8306a1c0a79c000e90a2a134e02b6a21b1e3079290bd5a`（7 件）。
- 仓库提交推送为正式回传；本地另存 tar 供直接下载。

## 6. 备注

- 执行前工作树中的三份未跟踪草稿与钉版内容一致（清单仅排版差异），已备份 `/tmp/h3_draft_backup/`；
  以仓库钉版为准。
