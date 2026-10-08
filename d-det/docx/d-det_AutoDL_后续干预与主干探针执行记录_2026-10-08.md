# d-det 后续干预与主干探针执行记录（2026-10-08）

> 对象：《d-det_AutoDL_后续干预与主干探针指导_2026-10-08.md》。
> 基线提交：`958eeb4`（本记录随本轮提交入库，git log 为准）。
> 权限边界：只新增 train/dev 探索；`test_read=false`、`generation=false`、`weights_downloaded=false`；`exploratory_train_dev_only=true`。
> 不覆盖 `variant_transfer_stage1b_2026-10-08/`、`variant_transfer_stage2_2026-10-08/` 或原始分数。

## 1 范围与产物

- 新产物目录：`d-det/artifacts/post_stage2_intervention_2026-10-08/`。
- 变换：A1 格式规范化 / A2 注释屏蔽 / A3 字面量屏蔽（敏感性）；**A4 标识符屏蔽 = `not_executed`**（指导 §3：多语言覆盖率与 API/变量区分不满足，不允许强行执行）。
- 读出：r1 = stage-1b 冻结对象仅 transform/predict（回答"原分类器对表面干预的脆弱性"）；r2 = 变换文本上按原规则重训（回答"干预后重新训练仍可读多少"）。
- 统计：500 次 task-cluster bootstrap（seed `20261008`；同一折内跨变换/读出共享同一 picks 序列）。

## 2 变换审计（全量 10,659 样本 = 11 成员 × (798+171)；test 未读取）

| 变换 | 失败 | compile 破坏 | token 变化 | 字符变化 | 关键计数 |
|---|---|---|---|---|---|
| A1_format_norm | 0 | 0 | −0.20% | −0.64% | 全部样本补尾换行；16,473 行去尾空白 |
| A2_comments_masked | 0 | 0 | −12.73% | −17.94% | 41,649 条行注释删除（无块注释）；删除 170.7 万字符 |
| A3_literals_masked | 0 | 0 | −13.34% | −18.42% | 字符串 72,140 / 数字 42,820；2,917 样本行数变化（多行字符串折叠） |

- 状态机：Python / C-family 双模式（语料 10,659/10,659 判定为 Python）；字符串/字符/三引号内部不误判；编译校验安全闸（原可编译而变换后不可编译 → 该样本保留原文）。
- hash：每变换保存原始↔变换逐样本 sha256 映射（联合 hash 见 `transforms_audit.json`；逐样本映射 `local/hash_maps/*.jsonl.gz` 本地）。代码正文不写入 Git。

## 3 A0 对照

- 22 折（11 heldout × 2 负集版本）重放：`max|Δ| = 0.000e+00`（**逐位一致**），meta 校验全 True（`a0_replay_check.json`）。
- 说明：dev_scores_new 为 float64 style 特征路径；本重放同式重算（float32 特征缓存会引入 ~2e-7 量化差，已避免）。

## 4 Phase A 结果

### 4.1 dev 汇总（系列内折等权→系列等权；原始 vs 变换）

P0_fusion 及关键读出（全表见 `intervention_report.md` §2）：

| 读出 | 原始 | A1·r1 | A1·r2 | A2·r1 | A2·r2 | A3·r1 | A3·r2 |
|---|---|---|---|---|---|---|---|
| tfidf_char | 0.8834 | 0.8834 | 0.8834 | 0.8519 | 0.8636 | 0.7544 | 0.8263 |
| sem_base | 0.9240 | 0.8060 | 0.8514 | 0.8371 | 0.9021 | 0.8464 | 0.9038 |
| style_lgb | 0.9401 | 0.7663 | 0.8412 | 0.6491 | 0.9326 | 0.8637 | 0.8910 |
| metadata_only(ctrl) | 0.8815 | 0.6443 | 0.7043 | 0.8440 | 0.8732 | 0.8236 | 0.8327 |
| size_length_only(ctrl) | 0.6800 | 0.6851 | 0.6864 | 0.6857 | 0.6974 | 0.6096 | 0.6648 |
| **P0_fusion** | **0.9566** | **0.8774** | **0.8966** | **0.8305** | **0.9460** | **0.8920** | **0.9355** |

### 4.2 paired delta（变换 − 原始，P0_fusion；AUROC）

| 变换 | r1（冻结对象） | r2（重训） | 分臂（size_mix / matched，r2） |
|---|---|---|---|
| A1 格式规范化 | **−0.0792** | **−0.0600** | −0.0603 / −0.0597 |
| A2 注释屏蔽 | **−0.1261** | **−0.0107** | −0.0125 / −0.0088 |
| A3 字面量屏蔽 | −0.0646 | −0.0211 | −0.0238 / −0.0185 |

（r1=冻结对象仅 transform/predict；r2=重训。全读出×逐折 CI 见 `paired_delta.json`；task-macro 为与 stage-2 一致的重数保留口径。）

### 4.3 判读（对照指导 §5 四种情形）

1. **A2 注释屏蔽 = 第一情形**：原对象大幅下跌（fusion −0.126；style_lgb −0.291）而**重训几乎完全恢复**（fusion −0.011；style_lgb −0.008）⇒ 主要来自**表示/词表不匹配**（原分类器依赖注释表面统计量），**不能说信号消失**；注释文本本身不是系列信号的主要载体。
2. **A1 格式规范化**：r1 −0.079 / r2 −0.060（显著非零）⇒ 格式/空白细节存在**真实的表面信号贡献**（A1 下 TF-IDF char/word Δ=0.0000 精确为零，验证该变换的语义安全性；style/metadata 控制因直接统计格式量而大幅改变）。
3. **A3 字面量（敏感性）**：r1 −0.065 / r2 −0.021 ⇒ 字面量有中等贡献，重训大部分恢复；仅作敏感性解读。
4. **综合**：干预后重训的融合读出在注释屏蔽下仍 0.946（≈原始 0.957）⇒ "两者都保持高分"：支持**干预后仍存在稳定可读成分**（但不得称因果后训练信号）；控制读出未接管（无 "只剩控制" 情形）⇒ 不在杂讯诊断路径上停滞，H-content 方向保留继续探查价值。
5. 两个负集版本（size_mix / size_matched）的 Δ 基本一致（差异 ≤0.006）⇒ 干预效应不随负集版本变化。

### 4.4 过程修正（透明记录）

- 首版 paired 实现低效（每折每读出重复计算 task-macro bootstrap，预计 1.8h），已终止并重写：per-task a_t 预计算 + 共享 idx_lists/yy_lists，实测 **470.2s（约 14x 提速）**。
- task-macro 口径对齐：将 r1/r2 metrics 的 task_macro CI 修正为与 stage-2 相同的**重数保留口径** M*=T⁻¹Σa（`fix_tm`，41.6s；point 值不变；行级 AUROC delta 与去重版逐位一致已验证）。

## 5 Phase B：主干可用性

- encoder-decoder：CodeT5-small（weights sha256 `968fb0f4…`，242MB）与 CodeT5-base（`053fbafd…`，892MB）——现有基线。
- encoder-only / decoder-only：**not_executed** —— 本地 HF 缓存已清理（Qwen/DeepSeek/Yi/SmolLM2 等均不存在），未获下载授权；最小 2×3 设计已登记于 `backbone_availability.json` 待授权。

## 6 合规声明

- `test_read=false`：未读取旧 test 行、未使用 `test_scores.npz` 做任何选择。
- `generation=false`：无新模型输出生成。
- 干预样本与变换均只作用于 train/dev；所有结论为探索性 train/dev 证据，不得写作迁移或 test 证据。
- `source_status=server_reconstruction_only`、`original_bundle_verified=false`、`claims_of_byte_identity=forbidden` 继续保留。

## 7 回传清单对照（指导 §8）

| # | 要求 | 产物 |
|---|---|---|
| 1 | intervention_manifest.json | ✓ |
| 2 | 每变换规则/失败率/hash/token 统计 | ✓（transforms_rules.json + transforms_audit.json + hash_maps） |
| 3 | 原对象 vs 重训评分对照 | ✓（`intervention_report.md` §2-§3；r1/r2 全出） |
| 4 | 两负集版本 paired delta | ✓（`paired_delta.json` §3.2；两臂差异 ≤0.006） |
| 5 | Phase B 可用性与权重 hash | ✓（backbone_availability.json） |
| 6 | commands/logs/env/git/SHA256SUMS | ✓ |
| 7 | 声明 test_read=false 等 | ✓（execution_switches.json + 报告头） |
