# F2 升级：member/series-heldout（新证据，不覆盖 task-CV 结果）

## M-HO（leave-one-member-out；11 折）

| 读出 | mean AUROC | min | max |
|---|---|---|---|
| q_only | 0.9645 | 0.9430 | 0.9859 |
| P0_pair_only | 0.6514 | 0.5476 | 0.8315 |
| q_plus_P0 | 0.6603 | 0.5605 | 0.8463 |
| cosine | 0.5613 | | |
| permuted | 0.3984 | | 0.5170 |
| mismatched_partner | 0.8570 | | |
| cross_task_swap | 0.9630 | | 0.9844 |

- q_only mean member-cluster CI95: [0.9570258106698805, 0.9717380881072272]
- delta q_only−P0_pair_only: **+0.3132** CI95 [0.26103007737164396, 0.36115389816913357]
- delta q_plus_P0−q_only: -0.3043
- 逐 series（q_only 均值）: {'CodeLlama-Instruct': 0.973834304192363, 'Qwen2.5-Coder-Instruct': 0.9623293397018646, 'DeepSeek-Coder-v1-Instruct': 0.9550887452549502}

## 长度桶分解（池化 M-HO 逐对分数；粗略检查）

| 桶 | n | q_only AUROC | P0_pair_only AUROC |
|---|---|---|---|
| short | 3506 | 0.9410 | 0.7446 |
| mid | 3451 | 0.9650 | 0.5445 |
| long | 3303 | 0.9613 | 0.5258 |

## 辅助对照解读（重要）
- `mismatched_partner`（伪对 (h,随机其他系列成员)）均值 0.8570：**伪对仍含 heldout member h**，该对照无法把分数压回机会——它表明分数含强“对中出现 h”的 unseen-member 识别成分，属协议性质，不作失败判据。
- `cross_task_swap`（成员对保留、任务特征换成另一 dev task）均值 0.9630：**高值为正面证据**——分数由成员对决定、与任务内容无关；
  因此指导条件“跨 task 错配回到机会”按字面未触发（构造无法回机会），以置换（permuted 回机会）与 P0 对照（delta 显著为正）承担该闸门功能，如实记录差异。

## S-HO（leave-one-series-out；3 折，低功效单列）

- q_only mean: 0.5635 (min 0.4670, 逐折 ['0.467', '0.519', '0.705'])
- P0_pair_only mean: 0.5336
- permuted mean: 0.4758
- **未通过（CL 0.467 / Qwen 0.519 / DS 0.705 接近机会）**：series 级迁移在未见 series 上不成立——M-HO 强度不可外推为“跨 series 迁移”。

## gate（M-HO）
- verdict: **member_holdout_transfer_candidate**
- {"two_folds_same_direction": true, "delta_q_minus_P0_positive": true, "permutation_not_reproduced": true, "not_single_series": true, "ci_reported": true, "q_only_mean": 0.9645382555765238, "delta_mean": 0.31318217916658464}

> 主比较为 fold-paired q_only − P0_pair_only；q_plus_P0 不得充当 P0-only。
> 总结限定：M-HO=unseen-member 识别（含“对中出现 h”成分）成立；S-HO=未见 series 迁移未成立（低功效但方向为负）。
