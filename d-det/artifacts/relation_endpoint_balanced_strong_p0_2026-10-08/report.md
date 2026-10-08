# 强 P0 补强（48 折；train/dev）

| 读出 | 点值 | 95% CI |
|---|---|---|
| s_fullP0 | 0.8486 | [0.8322, 0.8618] |
| s_fused3 | 0.8225 | [0.8057, 0.8374] |
| s_pairLR | 0.7673 | [0.7522, 0.7826] |
| s_lexical | 0.6657 | [0.6476, 0.6815] |
| s_style | 0.7677 | [0.7523, 0.7814] |
| s_meta | 0.8350 | [0.8147, 0.8526] |
| s_residual | 0.8501 | [0.8323, 0.8650] |
| s_compose | 0.8455 | [0.8272, 0.8611] |

## 配对 Δ（同一 bootstrap）
- fullP0_minus_fused3: +0.0260 [+0.0214, +0.0302]
- residual_minus_fullP0: +0.0014 [-0.0097, +0.0129]
- residual_minus_compose: +0.0046 [+0.0018, +0.0075]
- fullP0_minus_compose: +0.0033 [-0.0086, +0.0150]

> 全部 P0 成分只在每折 train 上拟合（fit-only scaler、系列等权、task 分组 cross-fit）；
> 融合选择=内层 5 折 CV（equal vs LR-stack）；7ee36fb 的 +0.46pt（residual−compose_canonical）保留并列。
