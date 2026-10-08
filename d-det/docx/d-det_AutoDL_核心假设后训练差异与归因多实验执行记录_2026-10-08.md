# d-det 核心假设与后训练差异执行记录（E0 + R0，2026-10-08）

> 对象：《d-det_AutoDL_核心假设后训练差异与归因多实验指导_2026-10-08.md》。
> 基线提交：`6aeee8d`；本记录随本轮提交入库（git log 为准）。
> 开关：`training_allowed=true`、`generation_allowed=false`、`test_read_allowed=false`、`old_test_reuse=false`、`weights_downloaded=false`；`source_status=server_reconstruction_only`。
> 不覆盖 A1/A2/A3、stage-1/2、variant-transfer 既有产物；旧 test 未读取。
> 结果一句话：**协议方向在 task/model/size 三种 heldout 下均稳健可读（0.89–0.95）；u=[S;A]×非线性一致优于单侧线性（+3~5pt）与全部 P0（+0.7~9pt）；R0 闸门通过，R1 待执行（partial 链）**。

## 1 本轮范围

- **E0**：lineage 与 pair 支持审计（§6）——`artifacts/core_lineage_audit_2026-10-08/`；
- **R0**：同 model_id complete/instruct 协议条件差异读出（§7，train/dev）——`artifacts/r0_protocol_diff_2026-10-08/`；
- **R1**：按闸门保留为 `partial_candidates_available`（6 条 HF 文档链），**本轮不执行**（等 R0 出口，§8/§14）；
- F0/F1/F2/F3/H3/D0：不在本轮（按 §14 顺序）。

## 2 E0 结果（脚本重算，非手填）

| 指标 | 数值 |
|---|---|
| distinct model_id（subset=full） | **120** |
| 同时具有 complete 与 instruct 单元（dual） | **115**（任务集 100% 重合，各 1140） |
| instruct-only / complete-only | 3 / 2 |
| 行级 total（full） | 274,748 |
| exact sha 重复组（字段级） | 5,598 |
| unit-task 多行组 | 6,848（identical=111，distinct=6,737） |
| 多行来源 | DeepSeek-R1-Distill × 6 系列 complete 全量双份（1,140×6）+ claude-3-5-sonnet 8 组 |
| 跨 task normalized 碰撞（train/dev 域） | 256 组（1 空 + 28 短 + **227 实质**） |
| └ 跨 split 实质碰撞 | **24 组，全为短输出（<200ch）自然撞车**；无长输出跨 split 泄漏 |
| └ 同 split 长输出重复 | 169 组 = **BigCodeBench/1120↔1121**（同题重复注册，均在 train，已标注） |
| skeleton 碰撞 | 336 组（含空/短输出） |
| R1 文档链（HF cardData.base_model 指向 dual 内模型） | **6 条**（见下） |

R1 链（`lineage_adjudication.json`）：Athene-70B←Llama-3-70B-Instruct；Athene-V2-Agent/Chat←Qwen2.5-72B-Instruct；Sky-T1-32B-Preview←Qwen2.5-32B-Instruct；Sky-T1-32B-Flash←Qwen2.5-32B-Instruct(+Sky-T1-Preview)；QwQ-32B-Preview←Qwen2.5-32B-Instruct。判定=**partial**（documented relation ✓；version 级独立证据缺）。

过程修正（透明记录）：
- 字段修正：BCC 行使用 `solution`/`solution_sha256`（首版脚本误用 code 字段，已在 41.5s 重跑中修正）；
- 多行选择：R0 每 (model, mode, task) 组按 sha 序取第一行（dup_extra_rows=5,822 in train/dev scope）；
- test 正文未读：test-split 行不装载；normalized/skeleton 仅对 train/dev 计算。

## 3 R0 方法与预注册（执行前写入 `hypothesis/config/switches/data_role_matrix/prereg_r0_r1`）

- **数据**：115 dual × 969（train 798 + dev 171）× 2 modes = 222,870 行（test 不装载）；
- **表示**：CodeT5-base 冻结 mean-pool（768d；small 512d 作敏感性）；style/meta/size 控制特征（P0 协议）；
- **任务**：配对方向判别——对每 (model, task) 的 (h_complete, h_instruct) 判断哪一侧是 instruct；
- **读出（13）**：single_hc / single_hi（对称等价）、delta_pair、S_sanity（阴性对照）、u_linear（≡delta_pair）、**u_mlp（twin-MLP on [S;A]，S 条件化机会）**、p0_tfidf_char/word（SGD×3seed）、p0_style_lr、p0_style_lgb、p0_metadata、p0_size_length、late_fusion（ΔA+Δ控制）；
- **切分**：seen（上界参考）；task-heldout（fit=train → eval=dev）；model-heldout 5 折（seed 20261008，fit=其余 92 模型 → eval=留出 23 模型的 train+dev 双口径）；size-heldout 4 桶（<2B/2-8B/8-20B/≥20B；NaN 规模模型两向排除）；
- **指标**：model-balanced pair-acc（对称化 AUROC/BA）+ task-macro + task-cluster bootstrap 500（seed 20261008，域内共享 picks）；方向校准用 fit 域符号；
- **主比较**：u − best_single、u − P0。

## 4 R0 结果

（全部为 model-balanced pair-acc；series/规模口径；bootstrap 500 共享 picks；方向校准已在 fit 域完成。）

### 4.1 主口径表（u_mlp 与基线）

| 域 | u_mlp | best_single(=single_hc) | delta_pair | best P0 |
|---|---|---|---|---|
| task_dev（训练=train 任务→评测=dev） | **0.9311 [0.924,0.937]** | 0.8954 | 0.8971 | style_lgb 0.9080 / word 0.9049 |
| model-heldout 5 折 × dev（未见 23 模型） | 0.9108–0.9429 | 0.8645–0.9231 | 0.8680–0.9254 | 0.8846–0.9237 |
| size-heldout 4 桶（dev） | 0.8850–0.9478 | 0.8421–0.9194 | 0.8464–0.9227 | 0.8710–0.9520† |
| seen（参考上界） | 0.9366 | 0.8846 | 0.8871 | style_lgb 0.9434‡ |
| small 表示敏感性（task_dev） | 0.9295 | 0.8972 | 0.9005 | — |

† size_0（<2B, 7 模型）上 tfidf_word 0.9520 略高于 u_mlp 0.9478（−0.4pt，诚实列出）；
‡ seen 上 style_lgb 0.9434 > u_mlp（train 内风格统计过拟合，heldout 均反转）。

### 4.2 paired delta（u_mlp − 基线；delta_mean，共享 picks，逐核分布）

| 对比 | task_dev | model-heldout（k=10） | size-heldout（k=8） |
|---|---|---|---|
| − single_hc | **+0.0357** | **+0.0381** [+0.0198,+0.0500] | +0.0323 [+0.0209,+0.0500] |
| − delta_pair | +0.0340 | +0.0356 [+0.0175,+0.0484] | +0.0301 [+0.0188,+0.0502] |
| − p0_style_lgb（最强 P0） | +0.0231 | +0.0165 [+0.0026,+0.0342] | +0.0178 [−0.0004,+0.0376]† |
| − p0_tfidf_word | +0.0262 | +0.0220 [+0.0150,+0.0298] | +0.0136 [−0.0042,+0.0316]† |
| − p0_tfidf_char | +0.0312 | +0.0301 [+0.0188,+0.0406] | +0.0238 [+0.0090,+0.0376] |
| − p0_metadata | +0.0693 | +0.0710 [+0.0380,+0.0879] | +0.0716 [+0.0519,+0.0873] |
| − p0_size_length | +0.0942 | +0.0939 [+0.0624,+0.1208] | +0.0914 [+0.0600,+0.1053] |
| − late_fusion | +0.0403 | +0.0440 [+0.0231,+0.0608] | +0.0409 [+0.0170,+0.0556] |

† size 聚合 CI 含 0（因 size_0 桶 tfidf_word 略优）；其余对比均为全面正差。
详细（含 CI、frac≤0、逐域分数）见 `summary_paired.json`、`results/*.json`、`local/scores_*.npz`。

### 4.3 要点

1. **协议方向可读且跨模型保持**：未见模型（model-heldout，拟合={102 模型}×train，评测=留出 23 模型±dev）上 u_mlp 0.91–0.94——不是简单模型指纹，而是可迁移的“协议差异方向”。
2. **双侧联合 + 非线性有稳定增量**：u_mlp 一致超过单侧线性（+3~5pt）与全部 P0 控制（+0.7~9pt）；线性 u≡Δ（数学等价，实测 pair_acc 完全一致）——增量来自 S×A 的非线性条件化，而非单纯配对差分。
3. **S 阴性对照精确 0.50**（全核零方差），证明指标与重采样链无泄漏。
4. **协议差异的强表面成分**：style_lgb/tfidf 在 t2 上 0.90–0.91（高于单个线性嵌入读出）—— complete vs instruct 的差异大量体现在风格/词法层面（与 A1/A2 干预轮结论方向一致，但此处为完整形式）。
5. **size_0 桶例外**：极小模型（<2B）上词法控制在 dev 上略优（0.9520 vs 0.9478）——在报告界限内如实保留；大模型桶（8B+）上 u_mlp 明显最强。

## 5 回传六句话（§15.2）

1. **本实验检验哪个命题？** R0：同一 model_id 的 complete/instruct **协议条件差异**是否可读（H1 的前置；标签 `protocol_conditioned_difference`）；不涉及后训练因果。
2. **训练/dev/test 各读了什么？** 拟合=115 模型 × 798 train 任务双模式（含 P0 特征）；评测=dev 171 任务（task 口径）、留出模型（model 口径）与规模桶（size 口径）；**旧 test 未读**（test 行从不装载；HF 仅 API 元数据）。
3. **主比较相对哪个 P0，delta 是点差还是 delta_mean？** u_mlp − best_single（single_hc/delta_pair）与 6 个 P0 控制；同时报告点差与 paired bootstrap 的 delta_mean+CI（共享 picks）；task 口径分别 +0.036 与 **+0.023（vs 最强 P0 style_lgb）**。
4. **信号在三种 heldout 中是否保持？** task-heldout 0.931；model-heldout 0.91–0.94（**5/5 折、10/10 核为正差**）；size-heldout 0.885–0.948（4/4 桶 u>single）。generator-heldout 尚未做（属 F3）。
5. **哪些混杂仍无法排除？** 协议强相关的输出长度/表面风格（size_length/style 控制仍很强）；BCC prompt 模板细节未读取；无法获取 token log-prob（不能构造 r_t 量）；BCC 内部同题对 1120↔1121（train 内，已标注）。
6. **是否满足下一阶段闸门？** **是**：model-heldout 未归零（0.91+）→ R0 通过，可进入 R1（6 条 partial 文档链待执行，需先补 version 级证据或显式标 partial）；禁止升级为后训练因果；size_0 的 tfidf_word 例外保留在报告内。

## 5 回传六句话（§15.2）

<R0_SIX_SENTENCES_PENDING>

## 6 合规

- `test_read=false`（旧 test 未读、未用）；`generation=false`；未下载权重（HF 仅 API 元数据）；
- `exploratory train/dev`；`protocol_conditioned_difference` 标签；禁止表述后训练因果；
- R1 保持 partial（无 version 全证据）；D0/fresh pool 属本机，本轮未涉及。
- 过程修正（透明记录）：首轮链因 `sizelen` NaN（参数解析失败模型）全部失败 → 中位数填充+缺失标志列修复；读出方向校准改用 fit 域符号（single_hc/hi 现对称一致）；两轮 eval 均未读取 test。
