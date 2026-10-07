# d-det AutoDL 执行记录：阶段 D（D0 支持矩阵 + D1 H1 控制矩阵）

2026-10-07 · 服务器端执行 · 依据《`d-det/docx/d-det_AutoDL_交接后阶段D主线执行指导_2026-10-07.md`》

## 0 结论速览

- **D0 ✅ 完成**：10 折中 **admitted = 6**（llm `meta` ×3、authorbench_dcan `openai` ×3），
  diagnostic = 4（llm `google`/`mistral`：train/dev 侧无正支持，pos=0）。
- **D1 ❌ 未通过闸门（cond2）**：最佳内容视图 `tfidf_word` **.8200** < 同切分 P0 `fusion_lr` **.8388**
  （paired task-cluster Δ = **−1.91pt**，95% CI [−3.55, −0.35]pt，P(Δ≤0)=0.99），距 +1pt 门槛差 **2.88pt**。
- 按指导 §3.2/§7：结论记为 **“task-aware 可读性/任务效应审计”**；**停止新架构探索，转入数据构造**；
  **H2/H3 未启动**（§7 停止规则第 1、4 条命中）。

## 1 锚点与环境

- 运行前 HEAD：`83e02c0674d0736f6c6e5f9048ab8a090876b2d5`（D0 与 D1 两个目录的 `git_head.txt` 同为该值）
- 环境：conda `udet`（Python 3.12.14 / torch 2.9.1+cu128 / transformers 5.17.0 / sklearn 1.9.1）；
  GPU RTX 3080 Ti 12G；CPU 阶段 `OMP_NUM_THREADS=8 MKL_NUM_THREADS=8`、`TOKENIZERS_PARALLELISM=false`
- 数据：`d-det/data/h2_authorbench_dcan/core.jsonl`
  （sha256 `2bd5ef8667b521bece97a26294cdbed6f5e9c3781b97ca1396dcf7548c244aca`；
  9,498 行；train 6,557 / dev 1,484 / test 1,457；2,715 tasks；6 families；全 C；task 不跨 split）
- 命令：见两产物目录内 `commands.txt`

## 2 D0：alignment 支持矩阵（`artifacts/stage_d_support_2026-10-07/`）

脚本 `scripts/stage_d_alignment_support_audit.py`（流式，1.0s）。成员规则 = benchmark fold 规则
（positive: pair family == fold.family；negative: fold.family ∈ 端点 families；train/dev：split 匹配且
held-out ∉ 端点；test：split==test 且 held-out ∈ 端点；source 隔离）。**该规则与 `fold_plan` 的记录级计数
不可混用**（已在脚本内注明）。

### admitted 折（6）

| fold | train pairs (pos/neg) | dev pairs (pos/neg) | test pairs (pos/neg) | task 交叉 | 端点哈希跨 split |
|---|---|---|---|---|---|
| llm meta: codellama | 1181 (83/1098) | 312 (24/288) | 160 (40/120) | 0 | 0 |
| llm meta: llama2 | 1313 (102/1211) | 324 (26/298) | 150 (38/112) | 0 | 0 |
| llm meta: llama3 | 1177 (88/1089) | 312 (24/288) | 158 (40/118) | 0 | 0 |
| AB openai: gpt-4.1 | 4358 (343/4015) | 1070 (93/977) | 690 (193/497) | 0 | 0 |
| AB openai: gpt-4o | 4525 (400/4125) | 1074 (102/972) | 645 (181/464) | 0 | 0 |
| AB openai: gpt-4o-mini | 4351 (357/3994) | 1031 (76/955) | 645 (186/459) | 0 | 0 |

### diagnostic 折（4，未 admitted）

llm `google`: Gemni-1.5-pro / codegemma、`mistral`: codestral / mistral —— 四折 train/dev 全为负样本
（pos=0），失败原因 `no_train_positive_support` / `no_dev_positive_support`，仅可作诊断。

- 交付：`alignment_support.json`、`data_role_matrix.json`、`report.md`、`git_head.txt`、`git_status.txt`、
  `commands.txt`、`logs/{audit_baseline,alignment_support}.log`、`SHA256SUMS.txt`（8/8）。
- **判定：D0 通过**（6 折 admitted 足以支撑 D1 的 task-aware 结论；4 折诊断单独记录）。

## 3 D1：H1 控制矩阵（`artifacts/stage_d_h1_authorbench_dcan_2026-10-07/`）

脚本 `scripts/stage_d_h1_authorbench_dcan.py`（145.6s）+ `scripts/stage_d_h1_build_report.py`（报告渲染）。
**test 读取（UTC）：2026-10-07T08:35:45Z，单次读取，未参与任何选择**；bootstrap = 500× task-cluster
（cluster=task_id，seed 20261007）。

### 3.1 视图与协议

- `metadata_only`：log1p(char_count)/num_lines/nloc/cyclomatic/token_size（不含 prompt；shortcut control）
- `tfidf_word` / `tfidf_char`：P0 同协议（train 词表、SGD log_loss 5ep、类逆频率权重、best-dev 快照），3 seeds（0/1/2）平均
- `codet5_meanpool`：冻结 CodeT5-small（包内已校准权重），512d mean-pool（截断 384+128、无特殊符、fp16 autocast），
  train 标准化 + balanced LR（C∈{.03,.1,.3,1}，仅由 dev 选择）
- `codet5_centered`（+`_unscaled`）：**转导诊断**——逐 task LOO 兄弟样本均值中心化（无标签），其余协议同上
- `p0_*` 行：读既有 `artifacts/acl_sota_p0/.../predictions_all.npz`（fusion_lr/tfidf_word/tfidf_char/style_lgb/style_lr/sem_lr/mean_ensemble），**只重算指标**

### 3.2 test 指标（n=1457）

| 视图 | macro-F1 | BA | CI95（task-cluster） | ECE15 | NLL | Δ vs P0 fusion | P(Δ≤0) |
|---|---|---|---|---|---|---|---|
| metadata_only | .3744 | .3962 | [.3512, .3970] | .1126 | 1.527 | −.4644 | 1.00 |
| **tfidf_word（最佳内容视图）** | **.8200** | .8125 | [.7966, .8400] | .0272 | .611 | **−.0191 [−.0355, −.0035]** | 0.99 |
| tfidf_char | .7692 | .7767 | [.7456, .7930] | .0539 | .753 | −.0697 | 1.00 |
| codet5_meanpool | .6440 | .6580 | [.6186, .6662] | .1011 | .963 | −.1954 | 1.00 |
| codet5_centered（转导） | .6487 | .6660 | [.6227, .6715] | .0531 | .916 | −.1908 | 1.00 |
| codet5_centered_unscaled（转导） | .6380 | .6607 | [.6107, .6625] | .0791 | .968 | −.2015 | 1.00 |
| p0_fusion_lr | .8388 | .8319 | [.8189, .8588] | .0388 | .485 | 0 | — |
| p0_mean_ensemble | .8282 | .8159 | [.8056, .8494] | .1319 | .555 | −.0109 | 0.92 |
| p0_style_lgb | .7227 | .7163 | [.6978, .7441] | .1059 | .815 | −.1169 | 1.00 |
| p0_style_lr | .5586 | .5673 | [.5328, .5844] | .0260 | 1.056 | −.2803 | 1.00 |
| p0_sem_lr | .6535 | .6543 | [.6274, .6760] | .1785 | 1.305 | −.1864 | 1.00 |

复现性核对（独立实现 vs P0 复用预测）：`tfidf_word` **Δ=+0.0000**、`tfidf_char` **Δ=+0.0000**、
`p0/style_lgb`、`p0/fusion_lr` 复算同值 —— 4/4 完全一致。

### 3.3 分桶与观测（test）

- **task-size**（同 task_id 样本数）：tfidf_word 在 `6+` 桶 F1 .7644，低于 `3`(.8664)/`4-5`(.8781)；
  codet5_centered 在 `2` 桶仅 .5343 —— 大 task 的同族多版本子集是更难区域。
- **generator**（accuracy）：`gemini-2.5-flash-preview` .9741、`gpt-4.1` .9396、`claude-3.5-haiku` .9209
  ≫ `qwen-2.5-72b` .6458、`deepseek-chat` .6011 —— family 内 generator 异质性是主要误差源之一。
- **长度四分位**（边界= train 分位）：tfidf_word 从 `[0,1264)` .7367 单调升至 `[3049,∞)` .8448。
- metadata-only 显著高于 chance（.3744 vs .1667）：长度/结构存在强 shortcut，但远不足以替代内容情报。

### 3.4 闸门（指导 §3）逐条

1. 内容视图高于 chance/metadata-only：✅（.8200 > .3744，chance .1667）
2. 相对同切分 P0 ≥ +1pt：❌（差 2.88pt）
3. 多 seed/多折方向一致：TF-IDF 3 seeds dev F1 word [.8442,.8339,.8429]、char [.7902,.7896,.7979]（CodeT5 视图为确定性前向）
4. paired CI：Δ vs fusion 95% CI [−.0355,−.0035]（不含 0、方向为负）
5. 转导项单独标注：✅（仅 `codet5_centered*`，不与其他视图混报）
6. 未混入检测轴：✅

**判定：D1 未通过** → 停止规则第 4 条（增益 <1pt）与第 1 条（10 折仅 6 折 admitted）命中。

## 4 交付清单

- `artifacts/stage_d_support_2026-10-07/`：8 个文件 + `SHA256SUMS.txt`（见 §2）
- `artifacts/stage_d_h1_authorbench_dcan_2026-10-07/`：`config.json`、`metrics.json`（72KB，含全量分桶/CI/paired）、
  `predictions.npz`（215KB，13 视图 test 概率 + y + task_id）、`report.md`、`SHA256SUMS.txt`（9/9）、
  `logs/{run.log, run_console.log}`
- 新建脚本 3 个（repo `scripts/`）：`stage_d_alignment_support_audit.py`、`stage_d_h1_authorbench_dcan.py`、
  `stage_d_h1_build_report.py`

## 5 偏差与失败项（如实记录）

- 显存/内存峰值未采样（脚本内无采样逻辑）——如需要可重跑单加采样。
- `length_quartiles` 输出含一个恒空的 `>=inf` 边界桶（显示为 n=0），无信息损失。
- 交接本记录的最终提交号见 git log（本文件所在提交）。

## 6 唯一下一步建议

按 §7：不启动 H2/H3、不再尝试新架构。将本审计结论（**task-aware 可读性/任务效应**：在此数据/切分上，
冻结 CodeT5-small 及其转导中心化变体均显著低于 P0 融合 .8388，且 10 折支持矩阵仅 6 折可用）回传指导方；
下一步实验（数据构造方向）应由指导方书面指令后再执行；在此之前仓库保持冻结与可复现状态。

---
*生成时间：2026-10-07；记录人：服务器执行 AI。*
