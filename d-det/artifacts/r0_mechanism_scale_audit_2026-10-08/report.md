# R0 机制裁决（canonical）与 S-resid 尺度控制

- **raw_SA_mechanism = unresolved**（指导 §4.1；旧 claim 字符串不采用）
- **s_resid_status = residualized_representation_candidate**

## 尺度控制（task_dev）

| 变体 | M_b | CI95 |
|---|---|---|
| A_only | 0.9279 | [0.921, 0.934] |
| S_resid | 0.9683 | [0.964, 0.972] |
| S_resid_std | 0.9650 | [0.961, 0.969] |
| S_resid_shuffle | 0.9168 | [0.907, 0.924] |

- dResid=+0.0403 / dResidStd=+0.0371 / dShuffle=-0.0112

## model 5 折（model_dev 域）

| fold | dResid | dResidStd | dShuffle |
|---|---|---|---|
| model_0 | +0.0463 | +0.0430 | -0.0041 |
| model_1 | +0.0440 | +0.0386 | -0.0058 |
| model_2 | +0.0361 | +0.0328 | -0.0107 |
| model_3 | +0.0389 | +0.0351 | +0.0000 |
| model_4 | +0.0453 | +0.0384 | -0.0051 |

- model dResid 均值=+0.0421；全折为正=True

## 审计量（fit fold 摘要）
- S_resid 每维 std: min=0.00716 median=0.0412 max=0.06
- 范数: ‖S⊥‖ mean=1.12 vs ‖S‖ mean=5.31 vs ‖A‖ mean=0.78
- cov top10 能量占比: S=0.445 / S⊥=0.354

> 结论引用方式：一切报告统一引用 `mechanism_verdict.json`（raw_SA=unresolved；S⊥ 状态见上）。
