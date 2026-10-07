# 公开完整语料预注册 split（2026-10-08）

依据指导 §5 第四步；**先冻结 split 再允许任何生成/训练**；本轮不训练、不生成、不评测。

- 主协议：BigCodeBench full/instruct —— **118 units × 1140 task = 134528 行**
- task split（seed 20261007）：train/dev/test = 798/171/171；每 split task 列表 sha256 已记录（`split_plan.json`）
- 64KiB 含 hard 子集：148 task（全部 ∈ full），split 内分布 {'train': 106, 'dev': 21, 'test': 21}
- unit-heldout：5 折（family_label 分组轮转）→ {'fold_0': 31, 'fold_1': 23, 'fold_2': 23, 'fold_3': 21, 'fold_4': 20}；每折测试 = 折内 unit × 全 test task
- family-variant-heldout：**blocked**（family 证据为 0/306 确认）
- 人类控制：CodeContests valid 117 / test 165 题；valid∩test 描述重叠 0；语言层计数见 `split_plan.json.human_control`
- 统计单位：task cluster；污染：unknown（外部验证定位）

## 每条方案的必填字段（对照指导）

| 方案 | 目标总体 | 纳入规则 | generator 数 | task 数 | 每 task 输出 | family 证据 | 污染 | 统计单位 |
|---|---|---|---|---|---|---|---|---|
| 主协议 H1/H2a | BCC full/instruct | 全量成员 | 118 | 1140 | 1 | 未确认 | unknown | task |
| 协议控制 | BCC full/complete | 同 task | 123 | 1140 | 1 | 未确认 | unknown | task |
| 难题控制 | BCC hard/instruct | hard 子集 | 86 | 148 | 1 | 未确认 | unknown | task |
| 次级切片 | EvalPlus MBPP+ | 22 模型包 | 22 | 399 | 1 | 未确认 | unknown | task |
| 人类/正确性控制 | CodeContests valid/test | 竞赛题 | — | 117+165 | 人提交 | — | unknown | 题目 |

