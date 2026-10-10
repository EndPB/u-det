# C3 一致性训练（2026-10-09）

性质：增强一致性（λinv=0.1）；非论文复现；train/dev only。

- C3 row-level = 0.9019；task-macro = 0.9253
- C0 row-level = 0.9442；task-macro = 0.9585
- C3−C0（task-macro）: -0.0331 [-0.0402,-0.0261]，正折 1/11

## 变换统计

| type | attempted | accepted | reject_rate | ast_broken | iface_changed |
|---|---|---|---|---|---|
| comment_strip | 2103 | 1677 | 0.203 | 0 | 0 |
| whitespace | 3161 | 1076 | 0.660 | 0 | 0 |
| rename | 4271 | 3666 | 0.142 | 0 | 0 |

## 变换后归因保持（transformed eval）

| type | n | AUROC_folds | AUROC_pooled | mean|Δp| | sign agreement |
|---|---|---|---|---|---|
| comment_strip | 2269 | 0.8706 | 0.8686 | 0.0840 | 0.9200 |
| whitespace | 1587 | 0.8504 | 0.6785 | 0.2425 | 0.7630 |
| rename | 5516 | 0.8954 | 0.8917 | 0.0445 | 0.9558 |
（C3 原码 row-level 参考 = 0.9019；|Δp|=|p_orig−p_trans|）

> 拒绝原因见 transform_acceptance.jsonl；接受样本 AST/接口零破坏。
