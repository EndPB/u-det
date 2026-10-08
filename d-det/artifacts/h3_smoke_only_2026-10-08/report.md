# H3 代码 smoke（smoke only；不产出论文主结果）

- 子集：300 train tasks × 10 members；holdout member=deepseek-ai--deepseek-coder-33b-instruct
- 参数量一致性（4 配置）：PASS

| 配置 | det AUROC(smoke) | holdout rel AUROC(smoke) | grad cos |
|---|---|---|---|
| c0_L_D | 0.9911 | 0.5494 | — |
| c1_L_F | 0.0751 | 0.3689 | — |
| c2_L_D+L_F | 0.9896 | 0.3673 | — |
| c3_L_D+L_F+cos | 0.9896 | 0.3673 | 0.028 |

> smoke_only=true；数据为固定种子子集；一切数字仅供管线验证；H3 主实验须待 member-heldout 归因闸门通过。
