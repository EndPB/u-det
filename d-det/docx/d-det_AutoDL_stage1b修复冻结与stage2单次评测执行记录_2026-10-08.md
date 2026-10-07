# d-det 变体迁移 stage-1b 修复冻结与 stage-2 单次评测执行记录（2026-10-08）

> 对象：`d-det_AutoDL_stage1修复冻结与stage2单次评测指导_2026-10-08.md`（第 4 轮）。
> 前序：stage-1 执行（commit `582df7a`）。本轮一次连续执行：stage-1b 修复冻结（commit `5bba65b`）→ stage-2 test 单读 → 证据收尾（本 commit）。
> 环境：udet conda（sklearn 1.9.1 / scipy / lightgbm 4.7.0 / torch 2.9.1+cu128 / RTX 3080 Ti / RTX 3080 Ti 单卡）。
> 语料：`d-det/data/public_same_task_full_2026-10-07/records.jsonl`（sha256 `d786667a…`，20,370 行，11 成员 × 798/171/171 任务）。

## 1 摘要（六项对照）

| 项 | 结果 |
|---|---|
| ① 冻结复现 | 22 折对象（TF-IDF/scaler/SGD 逐 epoch 最优/LR/LGBM/融合/R2 统计）确定性重建；dev 重放对齐 |Δ|≤5.0e-7（csv↔new）、≤4.4e-16（reload↔new）、点指标 ≤1e-10；freeze_validation 6/6 PASS |
| ② 偏差处理 | D1–D5 全部处置（对象重建 / R2 cosine 弃用 / task-macro 重数修正 / 长度加权 deferred amendment / 控制读出改名与暴露账本 v2） |
| ③ test 读取 | 单次读取 1,881 行（11×171）；ledger 2026-10-07T19:52:15Z 开始 → 20:11:25Z 结束；commit `5bba65b` 锚定；读后 `test_read_allowed=false` |
| ④ 逐系列迁移 | heldout_test AUROC（P0_fusion，系列均值）：CodeLlama 0.950 / Qwen2.5-Coder 0.964 / DeepSeek-Coder-v1 0.922 |
| ⑤ 控制差值 | fusion−P0_equal Δ≈0（+0.0006）；fusion−size_length_only **+0.341**；fusion−metadata_only **+0.058** |
| ⑥ 论文允许结论 | "固定公开 BigCodeBench 协议下，官方代码模型系列内尺寸变体迁移可被静态作者信号（风格+语义+布局融合）高精度识别；规模/长度与元数据控制不能解释"；保留模板/清洗/污染 unknown 限制 |

## 2 执行时间线（UTC）

| 时刻 | 事件 | 产物/commit |
|---|---|---|
| 19:35 | stage-1b 对象重建开始 | `variant_transfer_stage1b_2026-10-08/` |
| 19:42 | 重建完成（444.1s；22 对象） | `objects_manifest.json` |
| 19:44 | 对齐校验完成（81s）| `replay_check.json`（三项全过） |
| 19:51 | 修正与账本（fixers） | `task_macro_ci_fix.json`、`r2_correction.json`、`implementation_deviation.json`、`prereg_amendment_length_weighting_deferred_2026-10-08.json`、`test_exposure_ledger_v2.json`、`readout_display_names.json`、`dev_paired_controls.json` |
| 19:51:34 | 冻结校验 6 门全过 | `freeze_validation.json` |
| — | 冻结 config 先行写入 | `stage2_config_frozen.json`（`created_before_any_test_read=true`，commit `5bba65b`） |
| 19:52:15.637 | **test 读取开始**（ledger 先于任何 test 访问） | `test_access_ledger.json` |
| 20:11:25.968 | **test 读取结束**（评分完成，1150.3s） | `stage2_metrics.json`、`test_scores.npz`（sha256 `a3ae6ee3…`） |
| 20:12 | 报告与校验和 | `stage2_report.md`、`SHA256SUMS.txt`（12 件） |

## 3 stage-1b 修复与冻结（commit `5bba65b`）

### 3.1 对象重建（D1）
- `variant_transfer_stage1b_rebuild.py`：同环境、同顺序、同 seed（SGD 6 seeds；C/LGBM 网格）重跑 22 折训练管线，序列化全部推理对象：`char/word TfidfVectorizer`、`StandardScaler`（按读出）、`SGD` 每 seed 最优 epoch 的 `coef_/intercept_`、`LR` 每 C、`LGBM`、`P0 融合头`、`R2 scaler 统计量（正集均值/标准差）`。
- P0_equal 训练特征固定为 `Ptr5.mean(1)`（与 stage-1 执行一致；写入对象清单自证）。
- dev 分数矩阵 `dev_scores_new.npz`（本地保留，sha256 记录于 `objects_manifest.json`）。

### 3.2 对齐校验（replay 三项）
- csv↔new（stage-1 预测 CSV 对重建分数）：max|Δ|≤5.0e-7 ✓
- reload↔new（磁盘 npz 重载对重建内存对象）：max|Δ|≤4.4e-16 ✓（bitwise 级一致）
- points↔metrics（重算点指标对 stage-1 metrics JSON）：≤1e-10 ✓
- **dtype 说明**：sem 读出为 float32 路径（rankdata 在 float32 上的排名结果），重放时须保持 `astype(np.float32)`；TF-IDF 为 float64。同 dtype 下分数 bitwise 一致。已写入 `replay_check.json` 的 `dtype_path_note`。

### 3.3 修正（D2–D5）
- **D3 task-macro bootstrap 重数**：`M* = T⁻¹ Σ a_{t_j*}`（保留重数），对 stage-1 CI 作出修正文件 `task_macro_ci_fix.json`；用 `[a1,a1,a2]` 合成例验证修正公式；整体 AUROC/AP 不受影响。
- **D2 R2 cosine 退化**：正集标准化后取正集均值 → 中心恒 0，标记 `invalid_zero_center`，stage-2 排除；euclid 改名 `positive_standardized_radial_score`（正向标准化径向分）。每表示范围（base `[0.4486,0.6393]`）与指导预期一致。
- **D4 长度加权**：未实现；test 前写死 `prereg_amendment_length_weighting_deferred_2026-10-08.json`（决策=deferred），臂命名 `member-size-matched / length-unweighted`。
- **D5 控制命名与暴露**：显示名 `metadata_only→code_layout_control`、`size_length_only→oracle_size_length_control`（旧 key 保留于产物）；暴露账本 v2 更正此前"test 字符串从未进入内存"的不准确陈述（实际为 loader 在 split 排除前 `json.loads`，test 代码字符串曾短暂进入字典，随后立即删除，未参与任何特征/统计）。

### 3.4 冻结校验（6/6 PASS，`freeze_validation.json`）
1. **哈希**：stage-1 34 件产物哈希全对齐；config/records sha 复核通过。
2. **split/heldout**：split sha 一致；22 折 heldout 单元均确认不在训练单元内。
3. **对象重载**：22 对象从磁盘重载，与重建分数核对一致。
4. **统计/注册**：bootstrap 500（seed 20261008）、修正文件存在、amendment 先于 test。
5. **静态审查**：stage-2 脚本无任何 fit 调用（transform/predict only）。
6. **stage-2 config**：冻结文件先于任何 test 读取写入（`created_before_any_test_read=true`）。

## 4 stage-2 test 单次评测（本 commit）

### 4.1 读取与开关
- ledger：`started 2026-10-07T19:52:15.637573Z`（记录先写入）→ `finished 2026-10-07T20:11:25.968222Z`；`n_test_rows_read=1881`；`scoring_code_commit=5bba65b330dd…`。
- 读取后 `execution_switches_stage2.json`：`test_read_allowed=false`（开关关闭，此后只能对保存分数重算预声明统计）。
- 特征编码仅一次（style/meta sha `e6576a84…`；emb small `a8c869a6…`；emb base `2252f5ba…`）；22 对象仅 transform/predict。

### 4.2 汇总结果（heldout_test；mean [95% CI]；跨任务 bootstrap 500，系列内折等权→系列等权）

| 读出 | heldout AUROC | heldout task-macro | seen AUROC | Δ(heldout−seen) |
|---|---|---|---|---|
| tfidf_char | 0.854 [0.840,0.869] | 0.897 | 0.879 | −0.025 |
| tfidf_word | 0.879 [0.866,0.893] | 0.915 | 0.904 | −0.026 |
| sem_base | 0.910 [0.897,0.923] | 0.929 | 0.922 | −0.012 |
| sem_small | 0.909 [0.896,0.922] | 0.932 | 0.922 | −0.012 |
| style_lr | 0.918 [0.903,0.930] | 0.930 | 0.926 | −0.007 |
| style_lgb | 0.933 [0.921,0.943] | 0.935 | 0.943 | −0.009 |
| code_layout_control | 0.886 [0.870,0.901] | 0.900 | 0.891 | −0.005 |
| oracle_size_length_control | 0.606 [0.594,0.619] | 0.650 | 0.682 | −0.075 |
| **P0_fusion** | **0.945 [0.936,0.953]** | **0.954** | 0.958 | −0.013 |
| P0_equal | 0.945 [0.936,0.953] | 0.955 | 0.957 | −0.013 |

- heldout−seen 损失全体 ≤0.026，P0_fusion 仅 −0.013 [−0.015,−0.010]（小而显著）。
- R2 radial（heldout）：base 0.560 [0.457,0.646]（22 折），small 0.579 [0.502,0.698]（弱信号，仅供解释）。

### 4.3 逐系列（P0_fusion heldout 折均值）
- CodeLlama-Instruct：0.950（8 折，范围 0.915–0.983）
- Qwen2.5-Coder-Instruct：0.964（8 折，0.920–0.988）
- DeepSeek-Coder-v1-Instruct：0.922（6 折，0.904–0.938）

### 4.4 控制与配对差（k=22 折分布）
| 对照 | ΔAUROC 均值 [min,max] | Δtask-macro 均值 | frac≤0 |
|---|---|---|---|
| fusion−P0_equal | +0.001 [−0.014,+0.014] | −0.000 | 0.50 |
| fusion−oracle_size_length | **+0.341** [−0.015,+0.763] | +0.309 | 0.042 |
| fusion−code_layout | **+0.058** [−0.004,+0.104] | +0.053 | 0.051 |

（size_length 控制仅在负集最小尺寸距离很近的个别折接近融合水平：CL/70b size_mix 0.930、CL/34b 0.844——与"尺寸距离近时长度信号可部分近似"一致；配对均值差距仍大。）

### 4.5 分层与 support_gap（解释用）
- 硬层（in_hard，n=168/189，pos=21）AUROC 0.897–0.979；非硬层 0.917–0.988——未见硬层系统性崩塌。
- matched 臂 holdout 与负集最小 |Δlog10(size)| 0.013–0.731（方向性解释见报告 §6）。

## 5 产物清单

| 目录 | 内容 |
|---|---|
| `artifacts/variant_transfer_stage1b_2026-10-08/` | 17 件：rebuild/align/fixers 全部 JSON、`objects/`（22 pkl，本地）、`dev_scores_new.npz`（本地，sha 在 manifest）、`SHA256SUMS.txt` |
| `artifacts/variant_transfer_stage2_2026-10-08/` | 12 件：`stage2_metrics.json`、`test_scores.npz`、`test_access_ledger.json`、`execution_switches_stage2.json`、`test_rows_index.csv`、`failures_and_invalids.json`、`stage2_report.md`、`stage2_config_frozen.json`、`commands.txt`、`git_head/status`、`logs/`、`SHA256SUMS.txt` |
| 本记录 | `docx/`（commit 随附） |

`failures_and_invalids.json`：0 失败、0 无效（22 折 × 12 读出全部完成）。

## 6 解释边界（沿用并强化）

- `source_status=server_reconstruction_only`；original bundle 未到达；**不得表述"复现本机原件"**；checkpoint 字节未验证；污染状态 unknown。
- 22 折共享同一 test 集与同一 bootstrap 抽样序列：**不把 22 折当作 22 套独立 test**；不挑显著折扩写结论。
- test 单读已消费；后续任何统计只能基于 `test_scores.npz` 重算，且须预声明。
- R2 弱信号、support_gap、分层均为解释性，不作验证性结论。
- 论文表述上限：*固定公开 BigCodeBench 协议下的官方模型系列内尺寸变体迁移*；模板/清洗/污染限制保留。

## 7 状态

- stage-1b → stage-2 一次性连续执行完成；stage-2 单读消耗完毕；开关关闭。
- 待办：原件到达后按其独立任务补核对（不覆盖本轮产物）；指导端如需新增 test 统计，须走新授权并在 amendment 中预声明。
