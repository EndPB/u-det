# 公开完整语料只读审计（2026-10-08）

依据指导 §5 第二步；records 全量扫描一遍（未执行任何生成代码/模型）。

- 行数：evalplus 8,778 / bigcodebench 287,476（与 manifest 一致）；records_sha256 `d786667a72f3f3864ea38115fd6ccfe8ac393141146d522673668f3730523210`（与包内 integrity 逐位一致）。
- **unit 口径**：adjudication 文件 **349 行 → 去重后 306 个精确 generator unit**（多出来的 43 行为同 unit 的 asset/member 行）；records 原样字段计数同为 306，前缀匹配 349/0；差异仅 evalplus 22 单元缺 backend/temperature 字段。**研究口径 = 306。**
- 协议轴：BCC full 1140 task / hard 148（交集 148）；instruct 1140 / complete 1140（交集 1140）；evalplus 399 task。
- 主协议预览（BCC full/instruct）：{"units": 118, "rows_sum": 134528, "tasks_min": 1140, "tasks_median": 1140, "tasks_max": 1140}（每 unit 覆盖全部 1140 task → 折设计充分）
- 其他切片：{"bcc_full_complete": {"units": 123, "rows_sum": 140220, "tasks_min": 1140, "tasks_median": 1140, "tasks_max": 1140}, "bcc_hard_instruct": {"units": 86, "rows_sum": 12728, "tasks_min": 148, "tasks_median": 148, "tasks_max": 148}, "evalplus": {"units": 22, "rows_sum": 8778, "tasks_min": 399, "tasks_median": 399, "tasks_max": 399}}
- family 证据：{'name_inferred': 233, 'unresolved': 79, 'provider_or_model_name_inferred': 37}；family_is_confirmed=true = 0。
- 重复：主键/身份重复 0；**unit-task 内 code/solution sha 重复 21 行（统计时按 unit-task-sha 去重）**；official dup/malformed 0。

## 结论（供第三步/第四步）

- 研究口径 = **306 个精确观察单元**；family 仅作候选提示（全部未证实）。
- 主协议冻结候选 = BigCodeBench full/instruct（118 unit × 1140 task × 1 行）；complete 与 hard 作为独立控制，不混合。
- 污染不可测 → 定位为外部验证/诊断；进项目主表前须完成污染与历史使用登记。

