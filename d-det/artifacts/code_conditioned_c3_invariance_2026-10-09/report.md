# C3 一致性训练（2026-10-09）

性质：增强一致性（λinv=0.1）；非论文复现；train/dev only。

- C3 row-level = 0.8748；task-macro = 0.8972
- C0 row-level = 0.9329；task-macro = 0.9442
- C3−C0（task-macro）: -0.0473 [-0.0577,-0.0378]，正折 0/11

## 变换统计

| type | attempted | accepted | reject_rate | ast_broken | iface_changed |
|---|---|---|---|---|---|
| comment_strip | 2103 | 1677 | 0.203 | 0 | 0 |
| whitespace | 3161 | 1076 | 0.660 | 0 | 0 |
| rename | 4271 | 3657 | 0.144 | 0 | 0 |

## 变换后归因保持（transformed eval）

| type | n | AUROC_folds | AUROC_pooled | mean|Δp| | sign agreement |
|---|---|---|---|---|---|
| comment_strip | 2290 | 0.8440 | 0.8398 | 0.0858 | 0.9157 |
| whitespace | 1710 | 0.7989 | 0.7372 | 0.1468 | 0.8644 |
| rename | 5512 | 0.8633 | 0.8600 | 0.0431 | 0.9542 |
（C3 原码 row-level 参考 = 0.8748；|Δp|=|p_orig−p_trans|）

> 拒绝原因见 transform_acceptance.jsonl；接受样本 AST/接口零破坏。
