# d-det AutoDL 端 AI 独立交接指导

版本：2026-10-12  
文档性质：独立交接文件；不替代、不改写 ACL 总账和历史实验记录  
适用端：`codex-gpu:/root/autodl-tmp/u-det`

## 0. 先读这一页

你是 AutoDL 端执行 AI。先按本文理解项目，再读取本文列出的证据文件。不要把聊天中出现的临时实验名称、AICD T1/T2/T3 或 SemEval 官方协议自动当成 d-det 的任务定义。

本项目只把 ABC 用作三个问题的简称：

* **A：二分类检测**；
* **B：含 Human 的家族归因**；
* **C：四分类检测**。

ABC 不指定数据集、类别名称、切分方式或损失函数。具体标签必须来自本项目冻结的 `data_role.json` 和 `batch_manifest.json`。AICD 的大规模数字标签包已经标记为 external control，禁止把它当作 d-det ABC 主线。

当前 AutoDL 已确认 RTX 3080 Ti 可见，但错误的 AICD 外部控制脚本已加撤回保护，运行会输出 `WITHDRAWN_EXTERNAL_CONTROL` 并退出。新的 d-det GPU 批次只有在本地冻结正确数据包并上传 manifest 后才启动。

## 1. 研究主张和边界

核心卖点是 **Code-Conditioned Post-Training Differences：基于代码条件的后训练差异**。旗舰模型的 base 差异按用户指定作为工作假设，在主线中近似忽略；这不是已证明的因果结论。AutoDL 端不得把最终来源分类结果改称“已识别的纯后训练因果”。

主线要回答：在同一代码任务条件、来源控制和反捷径视图下，d-det 的代码条件后训练差异表示是否相对同预算 CodeT5 微调基线，改善 A/B/C 的检测、归因、短长代码效果和格式/来源外推。

历史结论继续有效：

* C0–C3 已关闭，旧代码条件输入、静态代理和一致性项未观察到增量，不重跑；
* H3 多轮 lexical/length 闸门和 AMMC 本机诊断是数据约束，不是模型容量问题；
* DCAN/AuthorBench 的任务中心化、semantic/fingerprint 分离、late fusion 和 full disentangle 结果都必须保留；
* frozen CodeT5-small 不能替代完整 CodeT5 微调基线；
* 旧实验的负结果、预测摘要、manifest 和日志不能删除或覆盖。

## 2. 数据角色：只按角色使用，不拼标签空间

### 2.1 A：二分类

主数据优先使用已经审计的同题 Human/AI 数据，保留 `task_id`、`language`、`project/repository`、`human_or_ai`、`generator/source_unit`、`solution_sha256` 和 split。STACAD alignment 是 Human anchor、多生成器和多语言任务控制候选；CoDET-M4、CodeMirage 等只能单独作为外部检测/OOD 控制。

### 2.2 B：含 Human 的家族归因

B 的正式类别空间必须含真实 Human 类和已核实的 AI family/source 类。AI-only source/generator 头只能作为辅助诊断。AuthorBench/DCAN、LLM-CodeGen、Droid、STACAD 的来源标签不能直接拼成一个共同 family label space；每个源必须有自己的 `label_map` 和 provenance。

任务中心化优先使用 AuthorBench/DCAN 的同语义任务簇、LLM-CodeGen 的同 CWE/场景对齐，以及 STACAD 的同题 anchor。Droid 缺少公开 task/prompt ID 时，只作为 generator-rich 外部控制，不能写成同题因果证据。

### 2.3 C：四分类检测

C 的四个类别由本项目实际数据构造和标签协议决定。AutoDL 端不得自动套用 AICD 或 SemEval 的 Human/Machine/Hybrid/Adversarial 名称；只有 `data_role.json` 明确给出创建证据时才接收对应类别。普通注释删除、空白扰动、变量改名属于训练视图，不会产生新的 Hybrid 或 Adversarial 标签。

### 2.4 AICD external control

`d-det/data/mainline_abc_200k_2026-10-12/` 是 AICD T1/T2/T3 numeric external control：公开字段只有 `code` 和数字 `label`，没有 task/project/generator/family 字段。它可用于规模、读取、格式统计和工程链路 sanity check，不能用于 d-det ABC 主表、任务中心化、语义分离或后训练差异监督。看到 `ROLE_EXTERNAL_CONTROL.md` 时，不得运行其中的 GPU 训练脚本。

## 3. 语义分离与任务中心化必须保留

代码编码器得到语义/结构表示 `h_i=E(c_i)`。对同一任务簇的参考集合计算任务中心：

```text
mu_task = mean(h_j : task_cluster(j) = task_cluster(i))
r_sem_i = h_i - mu_task
```

`r_sem` 用于减少题目内容和模板差异对来源读出的干扰；它不是把 task ID 作为模型输入，也不是默认要求部署时提供参考代码。训练/转导式评估与单样本部署必须分别记录：是否有同题参考、中心估计是否排除待测样本、参考样本来自哪个 split、是否允许跨来源参考。

语义中心化与来源分离至少要有以下读出：

1. `semantic-only`：只看代码语义/结构表示；
2. `centered-semantic`：看 `r_sem` 或等价同题残差；
3. `fingerprint/layout-only`：只看可解释来源/格式统计；
4. `late-fusion`：语义与 fingerprint/layout 融合；
5. `full-disentangle`：历史候选，仅作为对照，不因名称预设成功。

报告中必须同时给出任务中心化前后、语义/来源分支和融合分支。不能用普通 pooled CodeT5 + 手工特征的一次结果替换这组语义实验。

## 4. 代码格式与 AMMC 训练视图

布局流保留 tab/space/混合缩进、空行率、行尾空格、行长均值/方差、换行风格和注释位置。必须有 `layout-only`、`lexical-only`、格式规范化和 tokenizer 可见性控制。布局可预测不等于纯后训练信号。

AMMC 类增强只在 train parent 上生成：

* comment/docstring stripping；
* 语法安全的轻度 whitespace perturbation；
* AST/CST 和作用域安全的 identifier rename。

每个变体保存 `parent_id`、`transform_mask`、before/after SHA256、AST/CST hash、syntax/compile status、changed token/line counts 和 reject reason。变体继承 parent 的 A/B/C 标签；不把变体数量当独立样本，不把变体改名为 Hybrid/Adversarial。

原视图与变体按 label、family/source、language、长度桶配平；`transform_mask` 不作为模型输入。dev/test 保留原始视图，并建立 formatter、去注释、改名和空白规范化 stress view。先在本机完成 parser、重复、view-probe 和闸门，AutoDL 不重复这些低资源工作。

## 5. 正确的模型批次

新 `batch_manifest.json` 冻结后，一次批次至少包括：

| 配置 | 作用 |
|---|---|
| `codet5_full` | 同数据、窗口、预算和 seed 的完整 CodeT5 微调基线 |
| `d-det_semantic` | d-det 语义/结构表示和任务中心化读出 |
| `d-det_semantic_layout` | 加布局流，保留 layout-only 控制 |
| `d-det_semantic_ammc` | 加 train-only AMMC 视图与选择性一致性 |
| `d-det_joint` | A/B/C 联合头，B 含 Human，C 使用项目真实四类 |
| `lexical/format/metadata` | 强控制与捷径诊断 |
| `label-shuffled` | 来源/任务标签打乱负控制 |

同一配置固定 encoder、最大长度/滑窗、batch、epoch、参数预算、fold、seed 和优化器。A、B、C 不共享标签含义；可共享底层 encoder，但必须分别报告。

最小损失结构为：

```text
L = L_A + lambda_B L_B + lambda_C L_C
  + lambda_sem L_semantic_center
  + lambda_view L_selective_AMMC
  + lambda_hard L_hard_negative
```

`L_semantic_center` 只作用于已冻结任务簇和合法配对；`L_selective_AMMC` 只约束预注册的检测/控制头，不强迫 B 抹去真实 family style；没有 parent 配对或 provenance 时，不构造伪造的 post-training delta loss。

## 6. GPU 启动闸门

AutoDL 端收到新包后，按以下顺序执行：

1. 读取本文和 `batch_manifest.json`、`data_role.json`、`hypothesis.md`；
2. 核对 commit、数据哈希、许可证、字段、split、task/project/solution cluster 和 test 边界；
3. 确认 A/B/C 标签分别符合本项目定义，且没有 AICD numeric label 混入；
4. 核对语义中心索引、parent-child 变体索引、AMMC 安全审计和 transform-only/view-probe；
5. 核对 lexical/length/format/metadata 控制和至少两个 heldout 评估单元；
6. 通过后才按 manifest 一次性启动所有 fold × seed × configuration；
7. 失败时整批停止并回传原因，不自行加轮数、容量、温度、损失项或新数据。

当前没有新的正确 ABC manifest，因此**现在不要启动任何训练**。不要自行把 AICD 包改名后继续运行。

## 7. 回传内容

一次性回传一个目录：

```text
batch_manifest.json
hypothesis.md
data_role.json
config.json
metrics.json
per_task_metrics.json
short_medium_long.json
format_stress.json
heldout_metrics.json
commands.txt
env.json
git_head.txt
logs/
SHA256SUMS.txt
```

必须报告 A/B/C 的整体、各类、短/中/长代码、原始/AMMC/formatter stress、task-heldout、project/source/family-heldout、校准和 task/project cluster bootstrap CI。统一写明 `test_read=false`、`generation=false`、`code_execution=false` 和是否下载新权重。

## 8. 禁止事项

* 不修改 ACL 总账、历史实验记录或旧结果；
* 不把 AICD T1/T2/T3 当作 ABC；
* 不把 AI-only source 结果写成 B；
* 不把普通格式变体写成 C 的新创建类别；
* 不把 task ID、transform mask、来源字段直接作为模型输入捷径；
* 不读旧 test，不执行数据集代码，不临时生成模型输出；
* 不拆成零散小实验，所有高 GPU 配置一次执行、统一回传；
* 不在结果不佳时通过加容量、温度、轮数或损失项挽救未经预注册的分支。

## 9. 当前交接结论

AutoDL 端当前只需保留环境和接收能力，等待本机生成新的正确 ABC 冻结包。现有 AICD 包已经完成接收核验并明确封存为 external control；已有任务中心化、语义分离、格式通道、AMMC 和历史负结果均继续作为下一批 d-det 主线的前置约束。

