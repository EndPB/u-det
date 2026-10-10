# H3 数据闸门报告：h2_stacad_alignment_v1 接收审计与反捷径 probes（2026-10-10）

- 判定：**`revise_data`**（预注册规则触发：length 族 task-macro ≥ 0.70；停止进入 H3，先回传缺口）
- 证据等级：**diagnostic**（train/dev only；test 仅哈希级完整性，不用于建模/阈值）
- 上游：`AUTODL_AI_HANDOFF_GUIDE_2026-10-10.md` §7、`ACL 总账 v2.0` §9（接收→审计→probes→无卡报告）

## 1. 接收与核对

- 交接包 `handoff_2026-10-10/`：`SHA256SUMS` **10/10 OK**（CRLF 剥离后）；外层 zip sha256 **OK**。
- 按指示完成迁移（全部内部文件 → 规范位置，壳目录已移除）：

| 交接内文件 | 现在位置 |
|---|---|
| `d-det/docx/d-det_ACL总研究总结与可迭代路线_2026-10-10.md` | `d-det/docx/`（sha 63f5cae9… ✓） |
| `d-det/docx/d-det_AutoDL_H3数据构建与UIT-AMMC反捷径指导_2026-10-10.md` | `d-det/docx/`（sha 54181671… ✓） |
| `d-det/data/h2_stacad_alignment_v1/*`（8 文件） | 已在位且逐位一致；补齐 upload zip（sha fec137fe… ✓） |
| `scripts/audit_stacad_alignment_v1.py`、`build_stacad_task_alignment_v1.py` | `scripts/`（sha 0787e39f… / 18cd1f60… ✓） |
| 指南 / manifest / SHA256SUMS / 外层 zip | `d-det/artifacts/handoff_verify_2026-10-10/`（验收回执） |

- `upload_hashes.json` **7/7 OK**；`unzip -t` OK；解包到 `d-det/data/h2_stacad_alignment_v1_received/`，与 canonical 5 个数据文件**逐位一致**。
- 交付审计脚本重跑：**7/7 checks 全 true**；`audit_index.json` 复现交接包新版 `4c658ae0…`（zip 与 upload_hashes 携带的旧版 `11f959e5…` 为封装时点差异，两者均已留档）。
- 环境：HEAD `a65158f`（与 ACL 基准一致）；Python 3.12.14 / NumPy 2.2.6 / sklearn 1.9.1 / SciPy 1.18.1；**GPU 实机存在（RTX 3080 Ti）但本轮按无卡流程 CPU-only**；`tree_sitter` 未安装（AST 仅 Python 标准库）；磁盘余量 19.1 GiB（≥2 GiB 闸门 ✓）。

## 2. 数据角色与许可

- 3,180 tasks（train 2,100 / dev 660 / test 420，文件级划分）× (1 human + 7 AI)；7 语言；7 个不同厂商 generator。
- 角色：H3 同题 Human/AI 候选 + 多语言任务控制；`same_task_cross_generator=66,780` 全为跨 family，`same_family_pairs=0` → **不能用作 H2 同家族正对**。
- 许可：CC BY 4.0（数据/文档）+ MIT（代码），`stacad_v2/LICENSE` sha 已登记；`source_ref` 全 25,440 行保留 ✓。

## 3. split 与重复裁定

- 无任务跨 split（交付审计 + 本机复核一致）；human 哈希与内容 3,180/3,180 逐一相符。
- **exact**：25,440 行全唯一；human∩AI = **0**；任务内重复 0。
- **whitespace-normalized**（行尾空白/空行折叠）：全唯一；human∩AI = **0**。
- **Python AST**（3,680 行 py 全部解析，0 解析失败）：
  - exact-AST 额外拷贝 23（均在任务内，paraphrase 巧合）；
  - 骨架（名称/常量归一）额外拷贝 807；**433/435 组为同任务内**；
  - **跨任务仅 2 组**：① 纯 train 内 2 任务 6 行；② **跨 train↔dev 2 任务 6 行** → 标记 `needs_manual_ruling`，建议 H3 前排除或人工核验（总影响 12 行）。
- 非 Python 语言的骨架/AST 去重：`pending_parser`（tree_sitter 缺失；按职责边界在本地完成）。

## 4. 变体安全 smoke（Python 子集：60 个 train 任务 × 8 行）

| transform | p | requested | accepted | 拒绝原因 | accepted 的 AST 保真 |
|---|---|---|---|---|---|
| comment_strip | 0.2 | 101 | 39 | no_change=62 | **1.0** |
| whitespace（±10% 空行密度） | 0.3 | 157 | 147 | ast_mismatch=10 | **1.0** |
| rename（作用域安全） | 0.4 | 192 | 153 | no_safe_candidate=22, unsafe_reflection=14, ast_mismatch=3 | **1.0** |

机制与拒绝规则工作正常（accepted 变体 100% AST/compile 保留；无变化不计样本）。全 7 语言全量变体构建归本地（parser 边界）。

## 5. 反捷径 probes（fit=train 16,800 行；eval=dev 5,280 行；test 封存）

| probe | task-macro AUROC | CI95（task-cluster） | row AUROC | 备注 |
|---|---|---|---|---|
| metadata_only | 0.5000 | [0.500, 0.500] | 0.5000 | 机会 ✓（语言/文件名特征无泄漏） |
| source_only | 0.5000 | [0.500, 0.500] | 0.5000 | 机会 ✓（file-name 桶无泄漏） |
| **length_only** | **0.8459** | [0.8287, 0.8617] | 0.6069 | ⚠ 长度族捷径 |
| **lexical_only** | **0.9261** | [0.9161, 0.9350] | 0.7886 | ⚠ 强控制上限（char+word TF-IDF+LinearSVC） |
| ast_shape_only（py） | 0.6486 | [0.6064, 0.6964] | 0.5671 | 中等；其余语言 `pending_parser` |
| transform_only | — | — | — | `not_run`（包内无变体；机制见 §4） |

逐语言 task-macro：length 0.78–0.90（py 最高 0.897）；lexical 0.90–0.95。

## 6. 长度诊断（train）与解释

- 每 generator 的 AI/human 字符长度中位比 ≈ **0.94–1.01**（整体无系统性长短）；human 比全部 7 个 AI 都短/长的任务各占 ~8–14%。
- 但 length_only 的 9 维特征（字符/行/空行/最大行/均行/词元/空格/制表）联合后，**任务内排名分离仍达 0.846** → 长度族信号来自多特征联合，而非单一比值；建议按长度分桶（含空行与行结构维度）做平衡，并将 length-only 基线保留为必须被训练缓解的控制。

## 7. 判定与最小下一步（回传缺口）

- 预注册规则：metadata/source/length 任一 ≥0.70 → `revise_data`。length 0.8459 触发；加 lexical 0.9261 → 判定 **`revise_data`**，不进入 H3，不通过加模型容量补救。
- 缺口清单（交给本地/下一轮）：
  1. **长度族平衡**：按（label×generator×language×长度桶）平衡或长度匹配采样；
  2. **lexical 强控制**：H3 模型必须相对 lexical-only 基线（.9261）仍有增量；训练配 UIT-AMMC 变体+平衡；
  3. 2 组跨任务 AST 骨架重复（各 6 行，含 1 组 train↔dev）→ 排除或人工裁定；
  4. 非 py 语言骨架/AST 去重 + 全量 7 语言变体构建（本地，parser 边界）；
  5. generator-heldout 折设计（≥2 折）+ 三 seed（进入 H3 前置，非本轮）。
- 复测建议：完成 1–4 后重跑本 probes（同脚本、同 seed），以 metadata/source/length/AST 全部 < 0.70 且相对 lexical 有明显缺口为目标。

## 8. 开关与证据边界

`test_read(modeling)=false`（test 仅用于哈希级完整性证书）；`generation=false`；`weights_downloaded=false`；`code_execution=false`。本报告全部数字为 **diagnostic**（train/dev）。

## 9. 交付物

本目录（`h3_data_gate_stacad_2026-10-10/`）：hypothesis.md（预注册）/ config.json / data_role.json / metrics.json / report.md / commands.txt / env.json / git_head.txt / SHA256SUMS.txt / logs/ / probes/（dev 分数 npz + digest）/ variant_smoke_records.jsonl / python_ast_dedup_detail.json / length_diagnostics.json。脚本：`scripts/h3_datagate_stacad_2026-10-10.py`；交接验收：`d-det/artifacts/handoff_verify_2026-10-10/`。
