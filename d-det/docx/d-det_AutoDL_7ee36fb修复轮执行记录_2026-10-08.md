# d-det AutoDL 7ee36fb 修复轮执行记录

日期：2026-10-08（执行跨至 10-09 凌晨，北京时间）；执行：服务器 AI（AutoDL，RTX 3080 Ti / 251GB 宿主内存，cgroup 上限 32GB）。
入口指导：`d-det/docx/d-det_AutoDL_7ee36fb修复裁定与下一轮执行指导_2026-10-08.md`。
基线 HEAD：`7ee36fb`（未覆盖；本轮全部为新增目录与新增脚本）。
首行性质：**修复轮**——指标口径修复（weighted_ap）、role-blind 标准化修复、强 P0 补强、R1 后训练比较；不改旧闸门、不产生新因果声明、不增加一轮换名称或调 gate。修复后负结果照实闭合。

---

## 0. 回传四点（§4 指定）

### 0.1 真实标准化 swap 断言（全部通过）

对 raw `[S,A]` 与 raw `[S,−A]` **分别**用同一 fit 参数标准化（A 块均值严格置 0、尺度为 √mean(A²)），在每次 run（smoke/task/model 5 折）中重验：

| 断言 | 结果 |
|---|---|
| `std_swap_A_strict_negate`（标准化后 A 块严格反号，max abs） | **true（0.0）** |
| `std_swap_main_identical`（主块逐元素相同） | **true（0.0）** |
| `flip_of_std_equals_std_of_flip`（先标准化再翻 A = 先翻 A 再标准化） | **0.0** |
| `raw_swap_Csym_identical`（原始层对称对特征交换不变） | true |
| `raw_swap_A_negated`（原始层 A 交换严格反号） | true |

记录：`d-det/artifacts/role_blind_residual_corrective_2026-10-08/interface_audit.json`（smoke 冻结版）；task 补跑与 model 5 折的完整断言同时保存在 `results/task_-1.json`/`results/model_*.json` 的 `audit` 字段及各自日志中（全部 true）。实际 `mu_A_raw`≈2.5e-4、`sigma_A_used_rms`≈3.06e-2（fit 域冻结值）。每 seed 的 init hash 与 final loss 存 `seed_meta`（例：A_only::s0 init_hash=804776dc5f48）。旧版对"标准化后的 A 块直接取负"造成的 2μ_A/σ_A 误差已消除。

### 0.2 role-blind 相对 matched baseline 的 paired 差值

**task_dev**：rb−zero_A = −0.26pt [−0.73,+0.26]；rb−A_only = −0.38pt [−0.84,+0.08]；rb−SA_full = −0.75pt [−1.20,−0.37]；rb−rb_shuffle = +0.24pt [−0.18,+0.70]。
**model 5 折均值**：rb−zero_A = −0.41pt [−1.20,+0.21]；rb−A_only = −0.36pt [−1.14,+0.41]；rb−SA_full = −0.69pt [−1.27,−0.33]；rb−rb_shuffle = +0.23pt [−0.40,+0.47]。
口径：model-balanced pair-direction accuracy；同一 task 抽样序列配对 bootstrap；无正增量、无机制证据。详见 §3。

### 0.3 relation 相对完整 P0 的差值

**residual − full_P0 = +0.14pt [−0.97, +1.29]（跨 0，增量被强 P0 吸收，照实归档）**；
residual − compose（canonical 弱基线）= +0.46pt [+0.18, +0.75] 并列保留；
full_P0 = 0.8486、residual = 0.8501、compose = 0.8455、fused3 = 0.8225。详见 §4。

### 0.4 R1 逐边/endpoint 隔离边界

见 §5（逐边读数 + 隔离重训读数 + 共享端点清单；E5 反向在隔离条件下依旧存在）。

---

## 1. 指导逐条执行对照

| 指导条目 | 执行情况 |
|---|---|
| §2.1 分别标准化 raw [S,A]/[S,−A]，补原始/标准化交换断言 | ✅ `core_role_blind_corrective.py`（新脚本，旧 `core_role_blind.py` 弃用） |
| §2.2 设 seed（Python/NumPy/Torch/CUDA）、保存每 seed 初始化/损失/分数哈希、重跑 task-dev 与 model 5-fold | ✅（链：task 折 + model 0–4 折） |
| §2.3 修复 weighted AP（tie 规则）、合成数据对照 sklearn | ✅ `core_fix_metric_audit.py`（见 §2） |
| §2.4 同一 48 折补齐三族 composition + symmetric pair P0（fit-only、系列等权、内层选择） | ✅ `core_strong_p0.py`（见 §4） |
| §3 R1 后训练比较与 checkpoint 隔离表 | ✅ `core_r1_posttraining.py`（见 §5） |
| §4 三独立目录 + 全部元数据 | ✅ 三目录均含 README/hypothesis/config/开关/data-role/git/SHA/命令 |
| "先修复测试再训练"；断言失败停止对应 branch | ✅ 修复-测试（smoke）全部断言通过后才跑全量；无 branch 需停止 |
| 不读 test、不生成、不下载权重 | ✅ 全部 switch=false（见各目录 `execution_switches.json`） |

---

## 2. 指标口径修复（§2.3）

### 2.1 合成审计（`scripts/core_fix_metric_audit.py`，3.8s）

| 合成用例 | weighted_auc | weighted_ap | 期望 | 结论 |
|---|---|---|---|---|
| 全 tie | 0.5 | 0.5 | 0.5 / 0.5 | ✅（旧向量版 AP=0.8333，已修复） |
| 全正 | null | null | null（无效输入警告） | ✅ |
| 全负 | null | null | null | ✅ |
| balanced anchor ties | 0.5 | 0.5 | 0.5 / 0.5 | ✅ |

AP 实现改用 `sklearn.metrics.average_precision_score(y, s, sample_weight=w)`（逐 tie 组累计口径），不再对 tie 组内部顺序逐行积分。

### 2.2 48 折重算（保存分数，不重训）

| 读出 | AUROC | AP（修复后） |
|---|---|---|
| independent-source composition | 0.8455 | 0.8458 |
| symmetric pair P0（P0_pair） | 0.7391 | 0.6917 |
| P0 compose | 0.6419 | 0.6371 |
| **relation residual** | **0.8501** | **0.8502** |
| q_sym（对称对 LR） | 0.6598 | 0.6371 |
| endpoint 锚 | 0.5000 | — |

记录：`d-det/artifacts/relation_endpoint_balanced_strong_p0_2026-10-08/metrics_metric_audit.json`。
（weighted_auc 向量化版与逐段循环版最大差 <1.1e-15，沿用通过；旧 AP 数字不进入论文的口径已落实。）

---

## 3. role-blind 标准化修复（§2.1–2.2）

### 3.1 model 5 折（member-heldout，3 seeds ensemble；model-balanced pair-direction accuracy）

| fold | roleblind | zero_A | A_only | SA_full | oracle | rb_shuffle |
|---|---|---|---|---|---|---|
| model_0 | 0.9067 | 0.9067 | 0.9067 | 0.9146 | 0.9573 | 0.9021 |
| model_1 | 0.9195 | 0.9246 | 0.9218 | 0.9231 | 0.9701 | 0.9167 |
| model_2 | 0.9279 | 0.9340 | 0.9368 | 0.9327 | 0.9661 | 0.9233 |
| model_3 | 0.9378 | 0.9358 | 0.9338 | 0.9437 | 0.9760 | 0.9345 |
| model_4 | 0.9045 | 0.9165 | 0.9160 | 0.9175 | 0.9676 | 0.9086 |

**paired Δ（同一 task 抽样序列，折间 mean [min,max]）**：

| 对比 | mean | [min, max] | 结论 |
|---|---|---|---|
| roleblind − zero_A | **−0.0041** | [−0.0120, +0.0021] | 跨 0，无显著差 |
| roleblind − A_only | **−0.0036** | [−0.0114, +0.0041] | 跨 0 |
| roleblind − SA_full | −0.0069 | [−0.0127, −0.0033] | 不跨 0但幅度 <0.7pt |
| roleblind − oracle | −0.0482 | [−0.0632, −0.0382] | oracle（受辅助信息上界）高 +4.8pt；不作机制证据 |
| roleblind − rb_shuffle | +0.0023 | [−0.0040, +0.0047] | 跨 0（与乱序残差同水平） |

**解读**：修正标准化后，**role-blind 残差读数与 matched baselines（zero_A / A_only / 乱序残差）无实质差别**（全部 |Δ| < 0.7pt，CI 基本跨 0）——"角色未知残差"在 member-heldout 上没有可识别的额外成对方向信息；oracle 的 +4.8pt 仅反映已知 instruct 侧的辅助信息，不能当机制证据。旧 `.9742` 等数字已降级（sign-construction diagnostic）。

### 3.2 task_dev（task-heldout，全成员）

| 臂 | pair_dir_acc_mb |
|---|---|
| roleblind | 0.9246 |
| zero_A | 0.9273 |
| A_only | 0.9286 |
| SA_full | 0.9322 |
| oracle（上界，辅助信息） | 0.9709 |
| rb_shuffle | 0.9223 |

**paired Δ（同一 task 抽样序列，500 bootstrap）**：

| 对比 | mean | 95% CI |
|---|---|---|
| roleblind − zero_A | −0.0026 | [−0.0073, +0.0026] |
| roleblind − A_only | −0.0038 | [−0.0084, +0.0008] |
| roleblind − SA_full | −0.0075 | [−0.0120, −0.0037] |
| roleblind − oracle | −0.0463 | [−0.0528, −0.0402] |
| roleblind − rb_shuffle | +0.0024 | [−0.0018, +0.0070] |

`order_swap_identity_max_abs = 0.0`（随机输入顺序后分数严格变号审计）；seq 分数 sha1=`ad0682369123`。
结论与 model 折一致：**role-blind 无可识别正增量**（对 matched baseline 全部 ≤0.8pt 小负或跨 0）。

其余说明（已定稿）：

- 方案 A（采纳并冻结）：A 块均值严格置零、尺度 √mean(A²)；S/R 块用 fit 均值/标准差。真实交换后 A 的标准化块严格反号（断言 0.0）。
- 内层 cross-fit 按 **task 分组**（同 task 的 model 行同行），不再用行号 mod。
- 每个 seed 记录 init hash 与 final loss；分数逐 seed 保存（npz），报告 ensemble。
- 旧 `.9742` 等数字降级为 sign-construction diagnostic，不再作为 AUROC 或机制证据引用。
- oracle 为受辅助信息上界（仅在已知 instruct 侧），不作机制证据。

---

## 4. 强 P0 补强（§2.4）

### 4.1 48 折结果（共享 task-cluster bootstrap 500）

| 读出 | 点值 | 95% CI |
|---|---|---|
| s_fullP0（四成分等权） | **0.8486** | [0.8322, 0.8618] |
| s_fused3（三族规则融合） | 0.8225 | [0.8057, 0.8374] |
| s_pairLR（对称对 SGD） | 0.7673 | [0.7522, 0.7826] |
| s_lexical | 0.6657 | [0.6476, 0.6815] |
| s_style | 0.7677 | [0.7523, 0.7814] |
| s_meta | 0.8350 | [0.8147, 0.8526] |
| s_residual（canonical） | 0.8501 | [0.8323, 0.8650] |
| s_compose（canonical） | 0.8455 | [0.8272, 0.8611] |

**配对 Δ（同一 bootstrap）**：

| 对比 | mean | 95% CI | 结论 |
|---|---|---|---|
| **residual − full_P0** | **+0.0014** | **[−0.0097, +0.0129]** | **跨 0：增量被强 P0 吸收（照实归档）** |
| residual − compose | +0.0046 | [+0.0018, +0.0075] | canonical +0.46pt 复现（小基线） |
| full_P0 − compose | +0.0033 | [−0.0086, +0.0150] | 跨 0 |
| full_P0 − fused3 | +0.0260 | [+0.0214, +0.0302] | 四成分显著优于三族融合 |

**结论（照实闭合）**：
- relation residual 相对**弱基线**（independent-source composition，0.8455）仍有 +0.46pt 小幅增量（CI 不跨 0）——与 7ee36fb 裁定一致；
- 但相对**本轮强 P0**（full_P0=0.8486）**增量不再显著**（+0.14pt，CI 跨 0）——“若新基线抹去增量照实归档”落实：**强基线已吸收残差增量**；
- full_P0 相对 fused3 显著 +2.6pt：对称对 P0 成分本身有效，但主要贡献可由对称对特征解释，不需额外的残差化表示。

### 4.2 协议（已定稿）
- 三族（lexical=char_wb 2–4 哈希 1024；style=style+sizelen(92+4)；meta）双侧 `[c;i]` 拼接，fit-only 标准化；
- 来源分类器按 **3 系列等权**训练（LR），内层 5 折（按 task 分组）OOF 概率 → 族 pair 分数 `s_g=Σ_f p(i,f)p(j,f)`；
- 完整 symmetric pair P0：`[|Δlex|; lex和; |Δlex|… |Δstyle|; style和; |Δmeta|; meta和]` 的 SGD LR（同 q_sym 协议）；
- 融合规则 equal vs LR-stack 在每折 train 内层 CV 选择（平手→equal）；full_P0=4 成分等权；
- **修正记录（照实归档）**：首版直接用原始分数等权 mean，SGD pairLR 分数尺度 std≈5.7e4 而三族概率积类分数尺度 std≈0.3，导致 full_P0 实际被 pairLR 独裁（与单列 AUC 完全相同，fold0/fold1 复现）。已修正为**四成分各自 z 化（fit-only 尺度参数：三族用 train OOF，pairLR 用 train in-sample）后再等权**，修正后 fold0 full_P0=0.8682 ≠ pairLR=0.7355，量纲独裁消除。第一版数字不进入任何回传。
- 主比较：residual−full_P0、residual−compose；保留 +0.46pt（+0.00465，CI [+0.00176,+0.00749]）canonical 并列。若新基线抹去增量，照实归档。

---

## 5. R1 后训练比较（§3）

脚本：`scripts/core_r1_posttraining.py`；产物：`d-det/artifacts/r1_posttraining_comparison_2026-10-08/`。
读出定义：`delta_only`=前训练基线（LR on [D,−D]）；`raw_joint_lr`=[S,D] vs [S,−D] swap LR；`mlp_swap`=256-64 MLP（3 seeds）；`side_combo`=单侧 child vs partner 判别；`full_P0`=对称对特征 P0。全部读出在 **7 边 × 2 模式全量**上拟合（含被评边自身），因此是"给定训练视角下的可分离性"读数，不是泛化读数。

### 5.1 逐边读数（instruct 模式为准；complete 侧见 metrics）

| edge | delta_only | raw_joint_lr [CI] | mlp_swap | side_combo | full_P0 |
|---|---|---|---|---|---|
| E1 Athene70B_Llama3 | 0.983 | 0.965 [0.94,0.99] | 0.924 | 0.765 | 0.826 |
| E2 AtheneV2Agent_Qwen72B | 1.000 | 0.994 | 1.000 | 1.000 | 1.000 |
| E3 AtheneV2Chat_Qwen72B | 1.000 | 0.994 | 0.988 | 1.000 | 1.000 |
| E4 SkyT1Flash_Qwen32B | 0.987 | 0.994 | 0.988 | 0.994 | 0.850 |
| **E5 SkyT1Flash_SkyT1Preview** | 0.751 | **0.373 [0.30,0.45]** | **0.389 [0.32,0.46]** | **0.367** | **0.313** |
| E6 SkyT1Preview_Qwen32B | 0.987 | 0.994 | 0.988 | 0.988 | 0.809 |
| E7 QwQ32B_Qwen32B | 0.993 | 0.993 | 1.000 | 0.988 | 0.812 |

（E5 complete 侧 raw_joint_lr=0.376 [0.31,0.44]、mlp=0.386、side=0.351、full_P0=0.239。）

### 5.2 数学边界（写入 metrics `math_note`）

线性 swap 分数差 `f([S,D])−f([S,−D])=2·w_D·D`（S 项与截距消去）——**线性 swap 读数不是 S-A 非线性交互证据**；小 MLP 为非线性对照。方向不依 dev 翻转、不取 max(p,1−p)、全部边保留（含 E1/E5 低值）。

### 5.3 endpoint 隔离表（移除 incident 后重训评分——隔离泛化读数）

| edge | 共享端点 | 移除 incident 边后训练边数 | isolated_rawLR（comp/inst） |
|---|---|---|---|
| E1 Athene70B_Llama3 | — | 6 | 0.550 / 0.526 |
| E2 AtheneV2Agent_Qwen72B | Qwen2.5-72B（E3） | 5 | 1.000 / 0.988 |
| E3 AtheneV2Chat_Qwen72B | Qwen2.5-72B（E2） | 5 | 0.994 / 0.982 |
| E4 SkyT1Flash_Qwen32B | SkyT1Flash(E5), Qwen2.5-32B(E6,E7) | 3 | 0.930 / 0.947 |
| **E5 SkyT1Flash_SkyT1Preview** | **SkyT1Flash(E4), SkyT1Preview(E6)** | **4** | **0.339 / 0.374** |
| E6 SkyT1Preview_Qwen32B | SkyT1Preview(E5), Qwen2.5-32B(E4,E7) | 3 | 0.912 / 0.930 |
| E7 QwQ32B_Qwen32B | Qwen2.5-32B(E4,E6) | 4 | 0.860 / 0.959 |

**边界结论**：
- **E5 的反转不是共享端点污染**——把 E5 的两个共享端点关联边（E4、E6）拿掉、只用其余 4 条不相干边训练，E5 仍为 0.339/0.374（<0.5）；
- E1 的读数高度边特异：全数据 0.983 → 隔离后 0.526（跨边不可迁移）；
- 7 条边全部 `evaluable_descriptive`（无 insufficient），但均保持 partial 口径；**不得**写 verified post-training effect 或 checkpoint-heldout 泛化。

---

## 6. 合规与开关

- 全部轮次：`train/dev 训练=true；test_read=false；generation=false；weights_downloaded=false`（各目录 `execution_switches.json`）。
- 未覆盖 7ee36fb：本轮仅新增 3 个 artifact 目录 + 新增 4 个脚本 + 1 个执行记录（本文）。
- 未重新编码/采集；复用既有缓存（H 嵌入、texts.jsonl.gz、plan/48 折）。
- 环境：`/root/miniconda3/envs/udet/bin/python`；sklearn 1.9.1、torch 2.9.1+cu128。

## 7. 产物与哈希

三独立目录（均含 README/hypothesis/config/execution_switches/data_role_matrix/git_head/git_status/commands.txt/SHA256SUMS.txt/逐 seed 分数 npz(local，不入库)）：

| 目录 | SHA256SUMS 行数 | 关键文件 |
|---|---|---|
| `d-det/artifacts/role_blind_residual_corrective_2026-10-08/` | 24 | interface_audit.json、metrics_role_blind_corrective.json、results/(task_-1 + model_0..4).json、scores_*.npz |
| `d-det/artifacts/relation_endpoint_balanced_strong_p0_2026-10-08/` | 14 | metrics_strong_p0.json、metrics_strong_p0_folds.json、metrics_metric_audit.json、report.md |
| `d-det/artifacts/r1_posttraining_comparison_2026-10-08/` | 11 | metrics_r1_posttraining.json、report.md |

脚本（新增 4 个，未改动旧脚本）：`core_fix_metric_audit.py`、`core_role_blind_corrective.py`、`core_strong_p0.py`、`core_r1_posttraining.py`。
本轮提交为 7ee36fb 之后的唯一后续提交（哈希见 git log / 回传）。

---

## 8. 已知限制（照实）

1. E5（SkyT1Flash vs SkyT1Preview）在 swap 与 P0 读出上系统性反向（均 <0.5 且 CI 不跨 0.5），本轮只登记为"该边在现有特征/协议下的方向性问题"，不作任何排除或翻转处理；若未来使用必须预注册处理方式。
2. role-blind/strong-P0 全部为 train/dev 开发协议内数字；不得升级为 test 或论文主结果。
3. 强 P0 初版量纲问题照实归档（§4），修正版本为准。
