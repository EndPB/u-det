# AutoDL 当前交接指南

版本：2026-10-10  
项目：`u-det` / `d-det`  
闸门运行代码：`a65158f83434521acaec3380e5b10a6f94b80b3a`；当前 GitHub/AutoDL HEAD：`e53293ed6a9ed67ad071d75f4e9430228f56d07d`  
当前服务器：SSH 可达，实机有 RTX 3080 Ti。本指南只在需要较大显存或长时间 GPU 批处理时交给 AutoDL；CPU 友好、小显存和数据准备任务由本机直接完成，不需要等待服务器或额外通知。

## 1. 当前目标

把本地已经准备好的 STACAD 同题 Human/AI 候选数据和可迭代研究文档交给服务器，为 H3 做数据闸门。H3 的 detection head 使用真实 `human_or_ai`；AI 子集才使用 observed source/generator head。BCC 的 complete/instruct、Droid 和 CoDET-M4 不得拼成同题人机检测。

在 GPU 批次尚未满足数据闸门前，AutoDL 不重跑旧 C0–C3，不下载权重，不读旧 test，不生成模型输出，不执行样本代码。Windows/Linux 重拟合差异已接受；服务器拟合对象若需复用，使用冻结系数包，不把它写成跨平台重新拟合一致。

## 2. 本地优先完成的工作

在本地直接完成：公开源下载和许可核对、流式解压、task/project/generator split、代码和任务哈希、exact/whitespace/lexical/AST 去重、STACAD 派生包、AST 安全变体、parent-child 索引、TF-IDF/线性基线、metadata/source/transform/length/AST probes、manifest 和压缩。简单、显存需求不高的任务不写成 AutoDL 待办，也不为此打断用户。

本轮交接包应放在：

```text
handoff_2026-10-10/
  d-det/docx/d-det_ACL总研究总结与可迭代路线_2026-10-10.md
  d-det/docx/d-det_AutoDL_H3数据构建与UIT-AMMC反捷径指导_2026-10-10.md
  d-det/data/h2_stacad_alignment_v1/h2_stacad_alignment_v1_upload.zip
  d-det/data/h2_stacad_alignment_v1/upload_hashes.json
  scripts/audit_stacad_package.py
  scripts/build_stacad_task_alignment_v1.py
  HANDOFF_MANIFEST.json
```

不包含凭据、`~/.codex/auth.json`、模型权重、旧 test 原文、原始 3.8GB STACAD 包、历史预测缓存或无关大目录。

## 3. AutoDL 批次启动前：一次性只读核对

进入项目目录后先记录机器、代码和磁盘：

```bash
cd /root/autodl-tmp/u-det
git rev-parse HEAD
git status --short --branch
python --version
python - <<'PY'
import importlib.util, shutil
for n in ('numpy','sklearn','torch','tree_sitter'):
    print(n, bool(importlib.util.find_spec(n)))
print('disk_free_bytes', shutil.disk_usage('.').free)
PY
```

确认代码 HEAD 是当前批次声明的提交（当前主线为 `e53293e...`）；闸门历史产物使用 `a65158f...`，不得混用。不要把服务器旧分支当本地最新。检查交接包：

```bash
python - <<'PY'
import hashlib, json
from pathlib import Path
root = Path('d-det/data/h2_stacad_alignment_v1')
spec = json.loads((root/'upload_hashes.json').read_text())
for name, item in spec['files'].items():
    h = hashlib.sha256((root/name).read_bytes()).hexdigest()
    assert h == item['sha256'], (name, h, item['sha256'])
    assert (root/name).stat().st_size == item['bytes'], name
print('upload manifest OK')
PY
unzip -t d-det/data/h2_stacad_alignment_v1/h2_stacad_alignment_v1_upload.zip
```

服务器磁盘剩余空间低于 2 GiB 时停止解包。解包到独立目录，例如 `/root/autodl-tmp/u-det/d-det/data/h2_stacad_alignment_v1_received/`，不要覆盖历史目录。

## 4. STACAD 数据角色和许可闸门

当前本地派生包应报告：3,180 tasks、22,260 AI rows、7 languages、7 observed generators；每 task 有 human anchor；`same_task_cross_generator=66,780`，且 `same_family_pairs=0`。因此它支持同题 Human/AI 与任务控制，不能直接作为同家族 H2 正对。

原 STACAD 数据/文档为 CC BY 4.0，代码为 MIT。保留 `source_ref`、human/code SHA256、原始 split、模型名、语言和过滤原因。任何缺失许可、行哈希、task 映射或 split 证据的文件标记 `blocked`，不训练。

## 5. 变体协议（只在 train）

对每条 train parent 独立抽样：

```text
comment stripping:      p = 0.2
whitespace perturbation: p = 0.3, ±10% empty-line density margin
identifier renaming:     p = 0.4, AST/scope safe
```

这些概率来自 UIT-AMMC 论文（Pham et al., SemEval 2026 Task 13）；项目只借用数据增强思想。原视图和接受的变体混合进 train，dev/test 只保留原视图。每个变体写 `parent_row_id`、`transform_mask`、before/after SHA256、AST hash、syntax/compile status、changed token/line count 和 reject reason。字符串、宏、反射、导出接口或 AST 不可确认时拒绝；无变化不补样本。

增强后必须按 label、generator、language、长度分桶平衡原/变体比例。`transform_mask` 不作为模型输入。若 transform-only probe 在 heldout 上明显高于机会，先修平衡，不进入 H3。

## 6. 训练前 probes 和 H3 闸门

只用 train/dev 运行：

| probe | 目标 |
|---|---|
| metadata-only | 识别长度、语言、来源字段捷径 |
| source-only | 检查 project/repository/generator 泄漏 |
| transform-only | 检查增强标记是否泄漏标签 |
| length-only | 检查 token/字符/AST 大小捷径 |
| lexical-only | 给出 char/word TF-IDF 强控制 |
| AST-shape-only | 检查模板和任务难度捷径 |

H3 正式训练前必须同时满足：task/project/generator/solution cluster 无跨 split 重复；至少一个 human 和三个 AI generator 的 task 支持；source unit 至少三个 generator；变体安全审计通过；probe 没有独自解释主标签；至少两个 generator-heldout 折可用；test 仍未读取。

## 7. AutoDL 大显存批次执行顺序

1. 接收一个完整批次包，一次性核对许可、哈希、split、schema、目录和运行环境。
2. 流式审计 `core.jsonl`、`task_index.jsonl`、`pair_index.jsonl`；本机已经完成的 CPU 准备不在 AutoDL 重复。
3. 在同一批次内完成必要的最终数据闸门复核；发现支持不足、泄漏、变体破坏或 probe 高时，整批停止，不通过增加模型容量补救。
4. 闸门通过后，一次启动预先写入 `batch_manifest.json` 的多项 GPU 任务，固定 encoder、batch、epoch、参数量和数据版本。
5. 同一批次至少覆盖两个 generator-heldout 折、三个 seed，以及 `detection-only`、`family-only`、`joint`、`joint+invariance`；可选 nuisance adversary 必须在批次开始前冻结。
6. 所有任务完成后只回传一个汇总目录，包含逐任务日志、指标、预测摘要、环境、commit、配置和 SHA256SUMS；不为每个小实验单独往返指导。

`batch_manifest.json` 至少列出 `data_sha256`、`code_commit`、`fold`、`seed`、`model_variant`、`resource_request`、`output_dir` 和 `test_read=false`。AutoDL 按清单执行完全部 job 后再统一回传，不临时增加未预注册配置。

当前 H3 数据仍为 `revise_data`，因此上述 GPU 批次暂不启动；长度平衡、七语言 parser/变体和重复裁定由本机直接完成。

## 8. 结果交付格式

每次输出到新目录：

```text
batch_manifest.json  hypothesis.md  config.json  data_role.json
metrics.json  report.md  commands.txt  env.json  git_head.txt
SHA256SUMS.txt  logs/       predictions-or-score-digests/
```

报告必须写清：数据源、许可证、行数、split、generator-heldout、是否读 test、是否生成、是否下载权重、是否执行代码、完整命令、结果和 task-cluster CI。小于 1 个百分点、折间方向混合、只在原视图提升、或 probe 失败时标记 `stop_or_revise_data`。

## 8.1 当前 H3 闸门结果（2026-10-10）

交接迁移和验收已经完成：交接 SHA256SUMS `10/10`、上传清单 `7/7`、外层 ZIP、解包逐位核对和 7 项 STACAD 审计断言全部通过。数据审计与 probes 产物见 `d-det/artifacts/h3_data_gate_stacad_2026-10-10/`，证据等级为 `diagnostic`；建模只使用 train/dev，test 只做哈希级封存。

当前裁定是 **`revise_data`**，不进入 H3 训练：

| probe | task-macro AUROC | CI95 | 裁定 |
|---|---:|---|---|
| metadata-only | `.5000` | `[.500,.500]` | 通过 |
| source-only | `.5000` | `[.500,.500]` | 通过 |
| length-only | `.8459` | `[.8287,.8617]` | 长度族捷径 |
| lexical-only | `.9261` | `[.9161,.9350]` | 强控制，H3 必须超越 |
| AST-shape-only（Python） | `.6486` | `[.6064,.6964]` | 中等；其它语言待 parser |

Python 60-task 变体 smoke 的 accepted 样本 AST 保真率为 `1.0`；跨任务仅有两组骨架重复、各 6 行，其中一组跨 train/dev，需先排除或人工裁定。非 Python 语言的 AST/骨架审计和七语言全量变体属于本机直接完成的准备任务，不应拆成 AutoDL 小任务。

下一步固定为：本地按 `label × generator × language × 长度桶` 平衡或匹配；完成七语言 parser/变体；处理两组跨任务重复；保留 `.9261` lexical-only 作为强控制；然后用同一 seed 重跑 probes。只有 length、AST 和其它高风险 probe 过闸，且 H3 在 lexical-only 之上有预注册增量，才恢复训练。

## 9. 明确禁止

- 不上传或展示 SSH 密码、私钥、`auth.json`。
- 不把 complete/instruct、协议方向、模型尺寸或目录名叫 Human/AI/family。
- 不把 STACAD 跨厂商 pair 当同家族正对。
- 不读旧 test，不用 test 选任务、阈值、seed 或变换概率。
- 不执行数据集中的代码；未获单独授权时 `code_execution=false`。
- 不下载大权重来做本地已经能完成的 split、审计、TF-IDF 或数据构建。
- 不覆盖历史产物；每轮新目录并保留 manifest/hash。

## 10. 回传模板

```text
server_head=
working_tree=
gpu_mode=
python/numpy/sklearn/torch=
disk_free_bytes=
received_manifest_sha256=
stacad_rows/tasks/languages/generators=
license_status=
duplicate_cross_split=
variant_ast_breakage=
probe_summary=
test_read=false
generation=false
weights_downloaded=false
code_execution=false
decision=continue | revise_data | stop
artifact_dir=
```

下次更新只需追加本轮日期、commit、输入和输出 hash、闸门结果及一个最小下一步；不要把旧长篇聊天重新复制进指南。
