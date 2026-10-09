# d-det AutoDL 代码条件后训练执行记录（C0–C3，已见观测系列未见成员；2026-10-09）

> 本轮：代码条件后训练 / 已见观测系列未见成员 / train-dev only
> test_read=false; generation=false; weights_downloaded=false; code_execution=false
> source_status=server_reconstruction_only; original_bundle_verified=false

入口指导：`d-det/docx/d-det_AutoDL_代码条件后训练_已见家族未见生成器指导_2026-10-09.md`（及公式核对文档、pilot 脚本、manifest）。
基线 HEAD：`fb6a822`（未覆盖；本轮为新增提交）。性质：**实验执行轮（C0–C3）**；负结果照实闭合，不换 backbone/温度/权重/seed/折追分。

---

## 0. 一句话结论

- **C0（强 P0 重建）**：dev row-level = **0.9329**、inner = **0.9301**；对 ACL §48 本机 approx 参考（.9344/.9324）差 **−0.15pt / −0.23pt**（严格 1e-3 逐点参考表未随本轮传输，见 §3.3）。
- **C1/C2/C3 全部未过闸门**：相对同折 C0 的 task-macro 增量均为**负**（最佳 C1-code_only-MLP −4.0pt；C2 −4.5pt；C3 −4.7pt），正折 ≤1/11，CI 全部 <0 ⇒ 判定 **`code_conditioned_increment_not_observed`**（§6）。
- **C4 = `not_executed`**（未获得测试执行授权；未生成任何空执行文件）。

---

## 1. 执行开关（§指导文件）

```json
{"training_allowed": true, "generation_allowed": false, "test_read_allowed": false,
 "weights_downloaded": false, "code_execution": false,
 "source_status": "server_reconstruction_only", "original_bundle_verified": false}
```

全程 train/dev；旧 test 行未进入内存；未生成新模型输出；未下载权重；未执行任何被测代码。

---

## 2. 数据和角色审计（§1）

### 2.1 输入重建（server reconstruction）

从服务器既有公开语料 `d-det/data/public_same_task_full_2026-10-07/records.jsonl`（296,254 行全语料）重建：

- `inputs/records_train_dev.jsonl`：**10,659 行 = 11 成员 × 969 task**（798 train / 171 dev），subset=full、generation_mode=instruct；每行含 model_id/series/task_id/split/code/solution_sha256（逐行哈希自校验通过）。
- **模式判别证据**：instruct 模式重建的静态覆盖 = 10,659/10,659 可 AST 解析、**10,602** 顶层同名入口，与本机预检数字（10,659/10,602）**完全一致**；complete 模式为 10,594 ⇒ 冻结 instruct。
- `inputs/family_series_admission.json`：3 观测系列 → 11 成员（CodeLlama-Instruct 4 / Qwen2.5-Coder-Instruct 4 / DeepSeek-Coder-v1-Instruct 3；family_is_confirmed=false）。
- `inputs/input_hashes.json`：语料/split/系列注册表的输入哈希、行序哈希（`e61c…`）、task split 哈希、成员映射哈希、代码 SHA 列表哈希、逐成员行数。
- 任务元数据：`bigcodebench_task_columns.parquet`（仅 task_id/entry_point/libs/instruct_prompt/code_prompt 5 列，1140 行白名单过滤；未选 test 正文/canonical solution）。

### 2.2 静态预检（服务器端重跑，脚本与随附版逐字节一致）

`static_preflight/`：`{"rows":10659,"tasks":969,"folds":11,"coverage":{"parse_ok":10659,"entrypoint_present":10602}}`。
覆盖数与本机完全一致；11 折 fold_plan：每折 train_rows=7,980 / eval_rows=1,368 / eval_positive_rows=171；含 ordered_train/eval key 哈希。

### 2.3 审计断言（全部通过）

1. model-task 无重复（10,659 全唯一）；task 不跨 train/dev；2. heldout 成员训练行数=0；3. 每折 heldout 系列在训练折仍有 ≥2 个 seen members；4. task 集合由预注册 split（`public_full_receive_2026-10-08/prereg/split_index.csv`）决定，未按模型分数挑任务；5. prompt/canonical solution/test/模型ID/foldID 未进入静态作者特征（pilot 的 static_view 仅含代码本身与任务接口）；6. 每份报告与 metrics 均写入 `family_is_confirmed=false`、checkpoint 未逐字节核验、原始包未验证等边界。

### 2.4 已知缺口（照实）

- 本机 `static_preflight/` 四个文件（manifest 应传项）**未出现在服务器**；服务器按其规格重建并保存全部哈希与覆盖数（一致）。
- **本机强 P0 逐点参考表未随本轮传输**（见 §3.3）。

---

## 3. C0：强 P0 复现（冻结规格）

### 3.1 规格（fit-only，全部只在折内 train 拟合）

| 组件 | 实现 |
|---|---|
| semantic | LR(C=1) on 折内标准化 CodeT5-small h_y (512d，冻结 mean-pool 512=384+128) |
| char TF-IDF | char_wb(2-5), min_df=1, sublinear, lowercase=False → **SGD log-loss ×3 seed 集成**（alpha=1e-6, 5ep） |
| word TF-IDF | word 标识符 (1-3), min_df=1, sublinear → SGD ×3 seed 集成 |
| style/meta | LR on 折内标准化 [style(92); meta(10); sizelen(3)] |
| 融合 | 四成分等权 z-score（以折内 train 分数组内均值/标准差标准化） |

### 3.2 结果（11 折）

- **dev**：row-level mean = **0.9329**（CI95 0.9228–0.9424）；task-macro = 0.9442（CI95 0.9338–0.9530）；pooled = 0.9323。
  逐折 row-level：Qwen-1.5B .971 / Qwen-7B .768 / Qwen-14B .845 / Qwen-32B .938 / CL-7b .977 / CL-13b .956 / CL-34b .950 / CL-70b .938 / DS-1.3b .970 / DS-6.7b .961 / DS-33b .988。
- **inner（train-only 任务切分，rule 冻结于 config）**：row-level = 0.9301；task-macro = 0.9515。

### 3.3 与本机参考的对照（残差与请求）

| 量 | 服务器 | 本机 approx（ACL §48） | Δ |
|---|---|---|---|
| dev row-level | 0.9329 | .9344 | **−0.0015** |
| inner row-level | 0.9301 | .9324 | **−0.0023** |

**严格 1e-3 逐点审计无法完成**：本机强 P0 的逐点参考表/产物（`attribution_method_design_2026-10-09`）未随本轮传输。为定位差异，执行了**预声明规格探针**（脚本与结果全部归档，见 `code_conditioned_c0_strong_p0_2026-10-09/probe_scripts/` 与 `metrics.json.reference_check.spec_probe_history`）：

| 探针 | dev row-level |
|---|---|
| S1+L R+in-sample z（初版） | 0.9245 |
| OOF z 统计 | 0.9236 |
| semantic=base / [small+base] | 0.8649 / 0.8553（单列） |
| 宽词表+LR(C=4) | 0.9303 |
| 宽词表+类别平衡 | 0.9294 |
| **宽词表+SGD×3seed（冻结）** | **0.9329** |

关键修复：**emb 行映射 bug**（早期误用本地成员索引，文本缓存逐行校验仅 17/10,659 匹配；改为 115 模型全局索引后 10,659/10,659 通过）。
若需要严格 ≤1e-3 对齐，请下一轮提供本机 P0 逐点参考分数或本机实现规格；当前所有下游 Δ 均相对此冻结 C0。

---

## 4. C1–C3（全部为负）

### 4.1 C1 题面条件化代码归因

5 视图 × 2 头（linear / MLP d→256→1、3 seeds、20ep），共用折/标准化/seed/容量；对照 C0 用**同一 task 抽样序列**配对 bootstrap 500。

| view | head | row-level | task-macro | ΔTM vs C0 [CI] | 正折 |
|---|---|---|---|---|---|
| code_only | mlp | 0.8921 | 0.9042 | **−0.0401** [−0.0485,−0.0317] | 0/11 |
| code_only | linear | 0.8694 | 0.8927 | −0.0514 | 0/11 |
| ht_hy | mlp | 0.8880 | 0.9022 | −0.0420 | 1/11 |
| full（含交互+ψ） | mlp | 0.8847 | 0.8998 | −0.0445 | 0/11 |
| full | linear | 0.8141 | 0.8727 | −0.0717 | 0/11 |
| prompt_only | linear/mlp | 0.5000 | 0.5000 | −0.4445 | 0/11 |
| psi_only | mlp | 0.5499 | 0.5800 | −0.3648 | 0/11 |

- **full − code_only（task-macro）**：MLP −0.0044、linear −0.0204 ⇒ **题面条件没有改变（改善）任务内排序，方向为负**。
- prompt_only = 0.5000 符合构造（h_t 任务内恒定，无任务内区分信息）。
- C1 结论：**回答否定**；不称后训练因果。

### 4.2 C2 静态代理条件（static_proxy_conditioning）

full 视图 + λv=0.1 的静态代理辅助 MSE（parse_ok 零方差排除；其余 8 项 mask-free）。
C2 row 0.8846 / task-macro 0.8998；**ΔTM vs C0 = −0.0446（0/11 为正）**；**vs C1 full-mlp = −0.0001（辅助损失零效应）**。
长度/style 残差化（label-free，OLS 于 [nchar, nlines, style 块]）后 C2−C0 = −0.069（row）/ −0.0796（TM）——增量本就不存在，控制后仍为负。
代理审计：n_ast_nodes 与长度相关 +0.70/+0.71（高），n_calls +0.58/+0.62，其余 0.19–0.41 ⇒ **静态代理高度混入规模/长度，不得重命名为"代码约束"**。

### 4.3 C3 论文增强的一致性训练

三种变换（comment .2 / whitespace .3 / rename .4，独立 Bernoulli，固定 seed）；训练 = 原+变换混合 BCE + λinv=0.1 · JS 一致性。
C3 row 0.8748 / task-macro 0.8972；**ΔTM vs C0 = −0.0473 [−0.0577,−0.0378]（0/11 为正）** ⇒ 闸门失败。

**变换安全审计（§5 要求）**：

| type | attempted | accepted | 拒绝率* | 接受样本 AST 破坏 | 接口变化 | 主要拒绝原因 |
|---|---|---|---|---|---|---|
| comment_strip | 2,103 | 1,677 | 0.203 | **0** | **0** | no_comment 426 |
| whitespace | 3,161 | 1,076 | 0.660 | **0** | **0** | no_change 2,085 |
| rename | 4,271 | 3,667 | 0.141 | **0**（mod-rename AST 等价） | **0** | ast_changed 333 / reflection_sensitive 121 / no_safe_candidate 150 |

\* 拒绝率 = 拒绝/尝试；拒绝均为**安全跳过或空操作**（"no_change/no_comment" 为本无可改；reflection/ast 为安全护栏），**接受样本 AST 破坏率与接口变化率严格为 0**。逐样本记录见 `transform_acceptance.jsonl`（含原始/变换哈希、AST 摘要、拒绝原因）。

**变换后归因保持（transformed eval）**：

| type | n | AUROC(折均值) | AUROC(pooled) | 平均\|Δp\| | 符号一致率 |
|---|---|---|---|---|---|
| comment_strip | 2,290 | 0.8440 | 0.8398 | 0.0858 | 91.6% |
| whitespace | 1,710 | 0.7989（8/11 折，3 折子集无正/负例） | 0.7372 | 0.1468 | 86.4% |
| rename | 5,512 | 0.8633 | 0.8600 | 0.0431 | 95.4% |
| （参考：C3 原码 row-level = 0.8748） | | | | | |

⇒ 变换后归因保持中高（rename 最好 95.4%），但没有产生任何相对 C0 的正增量。

---

## 5. 闸门判定（§6 停止规则）

对 C1/C2/C3 的**最佳候选**（C1-code_only-MLP，ΔTM −4.0pt）逐条检查：

1. 相对 C0 的 task-macro 平均增量 ≥1pt？**否（−4.0pt）**；2. ≥8/11 折为正？**否（0–1/11）**；3. 配对 CI 下界 >0？**否（CI 全负）**；4. 增量只在 pooled/单系列/单尺寸/单 seed？**无增量**；5. 变换拒绝率/AST 破坏/接口变化（C3）：接受样本 0/0，安全审计通过；6. 增量在代码-only 或长度/style 控制后消失？**不存在需控制的增量**；7. 训练数据泄漏（heldout/test/标签筛选/顺序泄漏）？**无**。

**裁定：`code_conditioned_increment_not_observed`**（全体候选）。按指导：停止此支线，不再堆叠 encoder/decoder/参数量/温度/融合权重；转向新同题 Human/AI 任务集或受控后训练数据构建的建议保留至后续轮次。
C4（测试执行授权）未获授权：`not_executed`，未生成空结果文件。

---

## 6. 产物与哈希

| 目录 | 内容 | SHA256SUMS 行数 |
|---|---|---|
| `d-det/artifacts/code_conditioned_design_2026-10-09/` | inputs（记录重建设+哈希）/ static_preflight（服务器重跑）/ features（bundle+题面嵌入+manifest）/ README | 11 |
| `d-det/artifacts/code_conditioned_c0_strong_p0_2026-10-09/` | metrics/report/逐折分数/探针脚本×4/日志/git | 14 |
| `d-det/artifacts/code_conditioned_c1_prompt_conditioned_2026-10-09/` | 5 视图×2 头结果/逐折分数/日志/git | 10 |
| `d-det/artifacts/code_conditioned_c2_static_proxy_2026-10-09/` | proxy 审计/长度控制/逐折分数/日志/git | 10 |
| `d-det/artifacts/code_conditioned_c3_invariance_2026-10-09/` | transform_acceptance.jsonl(31,977 尝试)/metrics/逐折分数/日志/git | 11 |

脚本（新增）：`cc_build_inputs.py`、`cc_build_features.py`、`cc_common.py`、`cc_c0_strong_p0.py`、`cc_c0_probe2.py`、`cc_c1_prompt_conditioned.py`、`cc_c2_static_proxy.py`、`cc_c3_invariance.py`（另收编 `prepare_code_conditioned_pilot.py` 为随附原件）。
本地分数 npz（逐折、逐 seed、逐类型）在各目录 `local/`（不入 git，SHA256SUMS 排除）。

---

## 7. 限制（照实）

1. **C0 对齐缺本机逐点参考**：dev −0.15pt / inner −0.23pt 于 approx 参考；规格探针与行映射修复全部归档；精确 1e-3 对齐需本机参考表/实现。
2. 本机 `static_preflight` 原件未传输，服务器重建一致（覆盖数逐位相同），但输入字节未做本机对拷校验（original_bundle_verified=false）。
3. C1 的 MLP 采用固定 20ep/3-seed 平均（共享于全部视图）；linear 头并列报告；未做任何按 dev 的选择。
4. whitespace 有 3 折子集无正/负例（AUROC 记缺失，pooled 供参考）。
5. 全部为 train/dev 开发协议；family_is_confirmed=false；不得写成后训练因果或确认性结论。
