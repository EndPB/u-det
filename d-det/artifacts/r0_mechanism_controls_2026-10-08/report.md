# R0 机制对照（S-A 条件性）

- verdict: **SA_interaction_conditional_evidence**
- judge_domains: task_dev + model（5 折）; all_dSA_positive=True shuffle_weaker=True resid_strong=True

| 域 | dSA | dShuffle | dResid |
|---|---|---|---|
| model·model_dev | +0.0027 | -0.0008 | +0.0421 |
| model·model_train | +0.0016 | -0.0050 | +0.0489 |
| task·seen | -0.0051 | -0.0153 | +0.0551 |
| task·task_dev | +0.0047 | +0.0003 | +0.0416 |

## 判读
- S 原始形式的条件性增量极弱；置换对照不比它差 → 按指导 §3 判据 **mechanism_unresolved**（不得写 S-A 交互机制成立）。
- S⊥（表面统计正交残差）联合 A 的增益强且全域稳定——探索性观察，机制待解释，不得写成后训练因果。
