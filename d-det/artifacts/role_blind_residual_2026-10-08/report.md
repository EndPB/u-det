# 实验 B：角色未知残差（role-blind）

## task_dev（M_b）

| 变体 | M_b | CI95 |
|---|---|---|
| A_only | 0.9767 | [0.971694635138571, 0.980858123569794] |
| zero_A | 0.9750 | [0.9693338418510044, 0.97976353928299] |
| SA_full | 0.9990 | [0.9984744469870328, 0.99941647597254] |
| roleblind | 0.9742 | [0.9666653953724891, 0.9804754640223747] |
| oracle | 0.9939 | [0.9909216882786677, 0.9961886600559368] |
| rb_shuffle | 0.9569 | [0.947673531655225, 0.9650406814136792] |

## Δ（roleblind − 基）
- roleblind_minus_A_only: -0.0025
- roleblind_minus_zero_A: -0.0009
- roleblind_minus_SA_full: -0.0248
- roleblind_minus_oracle: -0.0197
- roleblind_minus_rb_shuffle: +0.0173

## model 5 折均值 Δ
- roleblind_minus_A_only: -0.0036
- roleblind_minus_zero_A: -0.0028
- roleblind_minus_SA_full: -0.0319
- roleblind_minus_roleblind: +0.0000
- roleblind_minus_oracle: -0.0257
- roleblind_minus_rb_shuffle: +0.0242
