# F2：关系归因（same_observed_series；task-CV 5 折）

- pairs: pos=14535, neg=14535（负对按 size+length+style 加权距离匹配）

| 读出 | 平均 AUROC | min | max | 折>0.5 |
|---|---|---|---|---|
| cosine_u | 0.6467 | 0.6379 | 0.6557 | 5/5 |
| lr_q | 0.8908 | 0.8850 | 0.9003 | 5/5 |
| lr_q_p0 | 0.5686 | 0.5535 | 0.5754 | 5/5 |
| permuted_labels | 0.4882 | 0.4565 | 0.5163 | 2/5 |

## 判据（§6.3）
- verdict: **relation_gain_observed**
- {"two_folds_same_direction": true, "beat_best_single_and_p0_mean": true, "permutation_not_reproduced": true, "mean_lr_auroc": 0.8907805469710283}
