# H3 数据闸门轮（h2_stacad_alignment_v1 接收审计 + 反捷径 probes）——预注册

- 日期：2026-10-10
- 上游：`AUTODL_AI_HANDOFF_GUIDE_2026-10-10.md`（§7 无卡执行顺序）、`ACL 研究总账 v2.0`（§9 可执行下一步 1–4）
- 输入：`d-det/data/h2_stacad_alignment_v1/`（3,180 tasks × (1 human + 7 AI generators)，7 languages；file-level split train 2100 / dev 660 / test 420）及其上传 zip
- 开关：`test_read`（建模用途）= false；test 行仅用于**完整性/去重证书**（哈希级），不用于任何模型拟合、阈值、seed 或变换概率选择；`generation=false`；`weights_downloaded=false`；`code_execution=false`；本目录内存的全部计算为 CPU。

## 问题

1. 接收包是否完整、许可与数据角色是否明确、split 是否无跨 split 任务重复、去重（exact / whitespace-normalized / Python AST）能否得到明确裁定？
2. 变体协议（comment .2 / whitespace .3 / rename .4）在最小的 Python 子集 smoke 上是否 AST-safe、reject 规则是否有效？（完整 7 语言变体构建按职责边界在本地完成；本轮只做 smoke_only。）
3. 主标签 Human-vs-AI 是否存在“单一捷径族”独自解释标签的风险？逐族运行 metadata / source / length / lexical / AST-shape probes（train-only 拟合，dev-only 评估；test 封存）。

## 预测（跑前登记）

- metadata-only 与 source-only（file-name 桶）应接近机会（≈0.5）：同任务 8 行共享语言与文件名，任务级控制使两者不可分。
- length-only 预计中等（0.55–0.70）：人工文件与模型 paraphrase 的长度分布可能有系统差异。
- lexical-only 预计较高（0.70–0.90）：AI 代码的字符/词法风格与人工文件存在可读差异；这是 detection 轴的既有难点，需要变体 + 平衡缓解，不单独否决数据。
- AST-shape-only（仅 Python 子集）：预计 0.60–0.75；非 Python 语言因无 tree_sitter 解析器标记 `pending_parser`。
- transform-only：无变体样本，标记 `not_run_no_variants`（机制在变体 smoke 中验证 mask 不入模型）。

## 判定规则（预登记）

- 数据完整性任意一项失败 → `stop`。
- 变体 smoke 中任一 transform 的 AST 保真 < 100%（在 accepted 样本上）→ `revise_data` 并回传缺口。
- 若 metadata/source/length 任一族 task-macro AUROC ≥ 0.70，或 ≥ lexical-only − 0.05 → 记 `shortcut_risk_high`，decision 建议 `revise_data`（先修平衡/再加变体），不进入 H3。
- 其余情况 → `continue`（数据闸门通过，下一步按 §6 补齐 generator-heldout 折设计后交给 GPU 阶段）。

## 输出（§8 格式）

`d-det/artifacts/h3_data_gate_stacad_2026-10-10/`：hypothesis.md / config.json / data_role.json / metrics.json / report.md / commands.txt / env.json / git_head.txt / SHA256SUMS.txt / logs/ / probes/（dev 分数 digest 与 npz）/ variant_smoke 明细。
