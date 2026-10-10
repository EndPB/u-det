# H3 本机数据修订 v1 结果

日期：2026-10-10；test payload read = false。

## 数据

- 旧报告裁定的 4 个任务已从派生 train/dev 排除；新增跨 split 重复涉及的 1 个 train 任务整簇排除。
- train_all=16768，train_balanced=8384，clean-dev=5272；matched-dev 只作敏感性分析。
- 七语言 parser 已对 train/dev 派生行执行审计；变体只在 train 采样。

## 预注册判断

balanced-train 在 untouched clean-dev 上的 length-only task-macro AUROC = 0.8231。
固定规则给出的数据侧裁定：revise_data。这不是 H3 方法增量结论；GPU 批次仍需在数据闸门通过后一次性执行。

## 限制

STACAD 每个 observed source 只有一个 generator，因此本包不宣称同家族未见 generator 迁移；source 仅作 task-heldout observed-source 诊断。parser tree-shape 是安全代理，不等价于编译/运行正确性。
项目来源证据缺失，无法认证 project-heldout；语法无效原行保留建模以免引入解析选择偏差，变体对这些行全部拒绝。transform-only task-heldout probe = 0.5670（CI 0.5293..0.5977）。不能将单项长度下降称为全闸门通过。
