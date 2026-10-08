# F2 M-HO 关系协议再修复 v2（corrective rerun；不覆盖旧结果）

- 协议：P+=(h,m_seen) / P-=(h,m_other)（**都含同一 h**）；partner 按 size/length/style 贪心匹配；
  **pair 顺序：双向平均（§3.3 选项 2）**——canonical 选项经检查在本数据存在位置-标签相关（部分折 gap=1.0），已弃用；
  direction_gap_q 均值（逐折）max=76.3652
- 折数=11（4+4+3）

| 读出 | 均值 | min | max |
|---|---|---|---|
| q_only | 0.8139 | 0.6681 | 0.8693 |
| P0_pair_only | 0.5648 | 0.5165 | 0.6398 |
| q_plus_P0 | 0.5659 | | |
| cosine | 0.6227 | | |
| h_only_probe（构造性≈0.5） | 0.5000 | 0.4999 | 0.5001 |
| partner_only_probe（⚠ 警告） | 0.8667 | | |
| task_cross | 0.8242 | | |

- **Δ(q_only−P0_pair_only) = +0.2491** CI95 [0.22350982595352664, 0.27084194774253256]
- q_only member-cluster CI95: [0.7772368079930663, 0.8422408397092915]
- 置换分离折数: 11/11（真值 > null 97.5 分位）

## ⚠ partner 身份警告
partner_only_probe 均值 0.8667 **高于** q_only 0.8139（逐折 q−partner 多数为负）——
即『只在特征的 partner 成员身份』本身可解该任务；q 的读数在『同系列关系』与『partner 个体识别』之间**尚未分解**。
本协议已排除 h 身份（h_only=0.5）与表面 shortcut（P0≈0.5），但 partner 身份短径仍然可及，需下一轮用成员-平衡或关系残差设计分离。

## gate（§6；含条件 6 的构造性判据）
- verdict: **member_holdout_relation_candidate**
- {"both_classes_contain_same_h": true, "order_leak_removed": "bidirectional_averaging (option 2; canonical 选项在本数据存在位置-标签相关 gap=1.0，已弃用并记录)", "direction_gap_q_max": 76.36519585250647, "delta_q_minus_P0_positive": true, "two_series_two_members_same_direction": true, "permutation_separated_folds": 11, "permutation_separated_ok": true, "not_h_identity_only": true, "h_only_probe_mean": 0.5000023317197834, "task_cross_mean": 0.8242101774214219, "partner_only_probe_mean": 0.8667478277871219, "partner_identity_warning": "q_only 均值低于 partner_only_probe（身份可单独识别）——『关系 vs partner 身份』未分解，已列入警告"}

> partner_swap = 匹配负类本身（协议别名）；task_cross 与同 h 负类联合解读；pair_order_swap 被双向平均吸收（记录 direction_gap 替代）。
