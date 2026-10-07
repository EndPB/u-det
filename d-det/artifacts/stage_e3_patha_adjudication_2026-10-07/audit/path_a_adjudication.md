# E3a：Path A 历史角色裁定（2026-10-07，只读）

依据《d-det_AutoDL_E2阻塞裁定与PathA探索性复核指导_2026-10-07.md》§2。
**未读任何 test 原文；未训练；未生成；未评测。**

## 1 结论

Path A 的 144 个候选 task **全部来自已开发 namespace**：
- 历史角色：{"historical_train_seen": 118, "historical_dev_seen": 26}；
- 有新 pair 使用史（train 或 dev）的 task：138/144；
- 新 split 的 22 个候选 test task 的历史角色构成：{"historical_train_seen": 18, "historical_dev_seen": 4}；
- `fresh_unseen_namespace` task 数：0；`fresh_test_eligible` task 数：0。

**正式标签（与指导 §0 一致）**：
```text
confirmatory_independent = False
exploratory_development_namespace_reuse = True
h2_main_table_allowed = False
h2_appendix_diagnostic_allowed = conditional（需用户明确接受降级标签后才可进行 E3b 只读复核）
training_allowed = False
new_generation_allowed = False
```

## 2 规则与证据

- 历史角色规则：原 split=train → `historical_train_seen`（118）；dev → `historical_dev_seen`（26）；原 test → `historical_test_read`（24，整体 excluded，见附录）。
- pair 使用证据：`positive_pair_index.jsonl`（695 行/159 task；README 声明 train 正对用于训练、dev 用于开发）。
- E0 暴露账本核对：144 task 无命中（无 AB 交集）；原 test 24 中 22 已读（pair 轮）。
- 新的 102/20/22 只是对已开发 namespace 的重新分组；22 个新 test 中 
  train/dev 历史 task 占 22 个——不能因本轮未读而称为 fresh。

## 3 允许的结论与禁止的写法（与指导 §1/§5 对齐）

- 允许：`LCv2 development-namespace exploratory re-split` 的工程诊断（appendix/diagnostic；需用户接受降级标签并授权 E3b）。
- 禁止：独立新任务泛化 / 普遍 family 几何 / 后训练因果 / H3 分离 / 把重切分写成 ACL 独立 H2 证据。

## 4 出口

- 默认出口 = 本裁定（E3a 只读）；**Path A 仍不进入 ACL 主 H2 表**。
- E3b（仅冻结输出/冻结读出的探索性复核）**等待用户明确接受降级标签**后执行；服务器不自行把 Path A 改为 ready。
- E3c（fresh unseen namespace）蓝图见 `audit/fresh_namespace_plan.json`。

