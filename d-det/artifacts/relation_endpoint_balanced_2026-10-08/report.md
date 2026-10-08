# 实验 A：端点平衡关系增量（48 折，train/dev）

- 折数=48（4x4x3）；共享 task-cluster bootstrap（500，seed 20261008）

| 读出 | 加权 mean | 95% CI | unweighted |
|---|---|---|---|
| compose | 0.8455 | [0.8272, 0.8611] | 0.8533 |
| q_sym | 0.6598 | [0.6454, 0.6739] | 0.6585 |
| P0_compose | 0.6419 | [0.6296, 0.6537] | 0.6610 |
| P0_pair | 0.7391 | [0.7195, 0.7566] | 0.7503 |
| residual | 0.8501 | [0.8323, 0.8650] | 0.8565 |
| endpoint_anchor_size | 0.5000 | [0.5000, 0.5000] | 0.5260 |
| endpoint_partner_size | 0.5000 | [0.5000, 0.5000] | 0.5000 |
| endpoint_unorm_diff | 0.5002 | [0.5000, 0.5005] | 0.5131 |

## 配对 Δ（同一 bootstrap 内）

- residual_minus_compose: mean=+0.0046 CI95 [+0.0018, +0.0075]
- residual_minus_P0_compose: mean=+0.2080 CI95 [+0.1953, +0.2202]
- residual_minus_P0_pair: mean=+0.1107 CI95 [+0.0951, +0.1266]
- q_minus_compose: mean=-0.1854 CI95 [-0.2028, -0.1677]

## 判词
- **verdict = `relation_increment_dev_evidence (当前三个已知系列)`**
- endpoint sanity：anchor=0.500 / partner=0.500 （应≈0.5，passed=True）

> weighted=系列平衡的加权总体（pairwise concordance）；unweighted 另列。
> 主比较为配对 Δ（residual−compose 等）；48 折共享 tasks，CI 用共享 task 序列。
