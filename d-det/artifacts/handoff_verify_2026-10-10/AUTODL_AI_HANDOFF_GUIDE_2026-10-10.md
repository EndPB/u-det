# AutoDL 当前交接指南

版本：2026-10-10  
项目：`u-det` / `d-det`  
代码：`a65158f83434521acaec3380e5b10a6f94b80b3a`  
当前服务器：SSH 可达、无 GPU 模式；因此本轮只做文件接收、CPU 审计和轻量 smoke。

## 1. 当前目标

把本地已经准备好的 STACAD 同题 Human/AI 候选数据和可迭代研究文档交给服务器，为 H3 做数据闸门。H3 的 detection head 使用真实 `human_or_ai`；AI 子集才使用 observed source/generator head。BCC 的 complete/instruct、Droid 和 CoDET-M4 不得拼成同题人机检测。

服务器当前不重跑旧 C0–C3，不下载权重，不读旧 test，不生成模型输出，不执行样本代码。Windows/Linux 重拟合差异已接受；服务器拟合对象若需复用，使用冻结系数包，不把它写成跨平台重新拟合一致。

## 2. 本地优先完成的工作

在本地完成：公开源下载和许可核对、流式解压、task/project/generator split、代码和任务哈希、exact/whitespace/lexical/AST 去重、STACAD 派生包、AST 安全变体、parent-child 索引、TF-IDF/线性基线、metadata/source/transform/length/AST probes、manifest 和压缩。简单任务不占用 AutoDL，也不因为服务器无卡而等待。

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

## 3. 接收后第一阶段：只读核对

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

确认代码 HEAD 是 `a65158f...` 或明确记录实际 HEAD；不要把服务器旧分支当本地最新。检查交接包：

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

## 7. 无卡模式的执行顺序

1. 只读核对接收包、许可、哈希、split、schema 和目录。
2. 流式审计 `core.jsonl`、`task_index.jsonl`、`pair_index.jsonl`，不整表载入内存。
3. 运行变体安全审计和所有 CPU probes；保存新目录、命令、环境和 SHA256SUMS。
4. 可运行小规模 CPU pipeline smoke，但输出只能标 `smoke_only`，不能当 H3 结果。
5. 发现支持不足、泄漏、变体破坏或 probe 高时，停止并回传缺口；不要通过加模型容量补救。

GPU 恢复后，先复核无卡阶段产物，再按固定 encoder、batch、epoch、seed 和参数量比较 `detection-only`、`family-only`、`joint`、`joint+invariance`，可选 nuisance adversary 必须预注册。每轮只增加一个方法变量。

## 8. 结果交付格式

每次输出到新目录：

```text
hypothesis.md  config.json  data_role.json  metrics.json
report.md      commands.txt  env.json  git_head.txt
SHA256SUMS.txt  logs/       predictions-or-score-digests/
```

报告必须写清：数据源、许可证、行数、split、generator-heldout、是否读 test、是否生成、是否下载权重、是否执行代码、完整命令、结果和 task-cluster CI。小于 1 个百分点、折间方向混合、只在原视图提升、或 probe 失败时标记 `stop_or_revise_data`。

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
