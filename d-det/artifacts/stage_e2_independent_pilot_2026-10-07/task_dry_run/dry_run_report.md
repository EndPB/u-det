# E2b 干跑报告（只读；未生成、未训练、未读旧 test 原文）

- 候选：144（LCv2 train+dev；三层 hash 建档）；排除：LCv2 原 test 24 + AB test 408。
- 碰撞：候选内 multi 组 exact/ws/lex = 72/72/72，**全部为 group 内 simple/secure 设计对**（同 prompt、同 group、同 split；`multi_groups_intra_provenance_group_only=True`，见 `audit/collision_groups.json` 样本）；跨 split 组 {'exact': 0, 'norm_ws': 0, 'norm_lex': 0}；对 AB prompt 命中 0；对 LCv2 原 test 命中 0。
- split_plan（group 级 seed 20261007）：{'train': 102, 'dev': 20, 'test': 22}；meta 折支持预览见 `prereg/split_plan.json`。
- 磁盘：若走 Path A 复用（无需生成）：新增索引/预测量级 < 5MB；若走 Path B/C 生成 144~300 task × 1–3 gen：≈0.6–1.8MB 原文 + 索引 < 5MB（E1 的 ≈15MB 估计不变）。
- 生成计划：**未生成**（E2a=blocked；按 §3 门槛不产出 generation_plan.json）。

## 三层碰撞明细

| level | candidate 内 multi 组 | 跨 split 组 |
|---|---|---|
| exact | 72 | 0 |
| norm_ws | 72 | 0 |
| norm_lex | 72 | 0 |

（说明：LCv2 的 group = 同一 CWE 的 simple/secure 两个 task，prompt 文本相同；同名 multi 组均为该设计对，不构成跨任务泄漏；group 级切分保证其不跨 split。）

## meta 单元 H2 折支持预览（split_plan 下）

| heldout | train(≥2 seen gen 非空) | dev(同上) | test(heldout 非空) |
|---|---|---|---|
| llama2 | 90 | 19 | 18 |
| llama3 | 80 | 15 | 21 |
| codellama | 73 | 16 | 20 |

