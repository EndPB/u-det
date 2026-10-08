# H3 registered（train/dev；不读 test）

| 配置 | det AUROC（mean） | rel AUROC（mean [min,max]） | grad cos |
|---|---|---|---|
| c0_L_D | 0.9717 | 0.5031 [0.4494,0.5374] | — |
| c1_L_F | 0.6556 | 0.7068 [0.5833,0.7502] | — |
| c2_L_D+L_F | 0.9715 | 0.6922 [0.5564,0.7486] | — |
| c3_L_D+L_F+cos | 0.9715 | 0.6922 [0.5564,0.7486] | 0.067 |

## S-HO 外推（3 折，低功效）

- c0_L_D: rel AUROC mean=0.4934
- c1_L_F: rel AUROC mean=0.5725
- c2_L_D+L_F: rel AUROC mean=0.5577
- c3_L_D+L_F+cos: rel AUROC mean=0.5577

> registered train/dev；det=held-out member 的协议检测；rel=v2 同 h 协议（partner 身份警告见 f2 fix 报告）；
> Pareto 点见 metrics；P0 参照引用 f2 v2（不重训）。
