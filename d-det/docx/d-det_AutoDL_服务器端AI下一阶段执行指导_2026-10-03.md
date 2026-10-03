# AutoDL 服务器端 AI 下一阶段执行指导

日期：2026-10-03  
适用目录：`/root/autodl-tmp/u-det`

这份文档可以直接交给服务器端 AI 执行。它描述的是从数据审计到 DCAN 风格语义/来源分离的完整工作流，不要求继续在旧的 768 维残差 + SupCon 路线上调参。

## 可以直接发送给服务器端 AI 的任务说明

```text
你现在位于 /root/autodl-tmp/u-det。用户已经授权你在这个项目目录内进行数据审计、实验代码实现、轻量基线和 DCAN 风格语义/来源分离实验。不要重新下载已经上传的数据，不要生成大规模新数据，不要覆盖旧实验结果，不要声称 SOTA，直到完成同协议基线和严格的 held-out 审计。

当前磁盘空间紧张，先执行 df -h /root/autodl-tmp。上传后约剩 2.7GB，禁止解压新的大包、复制 AICD parquet、把 JSONL 全量转换成另一份 JSONL、保存多个大 checkpoint。预处理最多使用 2 个 CPU 线程；GPU 训练使用混合精度和小 batch。

第一步只读检查：
1. 阅读 d-det/docx/d-det_AutoDL_服务器端AI下一阶段执行指导_2026-10-03.md、d-det/docx/d-det_ACL家族归因数据集合_v1_服务端执行指导_2026-10-03.md、d-det/data/acl_attribution_collection_v1/collection_manifest.json。
2. 确认 d-det/data/acl_attribution_collection_v1/raw/aicd/ 有 20 个 parquet 分片；确认 h2_authorbench_dcan、h2_llm_codegen、stacad_v2、codet_m4 和 h2_droid_full_selected 可读。
3. 用 parquet batch / JSONL 逐行方式审计，不允许 pandas.read_* 或 json.load 整个代码集合。
4. 输出 audit_server_2026-10-03.json 和 audit_server_2026-10-03.md，记录行数、schema、标签分布、缺失字段、重复风险、磁盘空间和可用 GPU。

第二步完成标签和切分审计：
1. AICD parquet 只有 code 与数字 label。尝试从官方协议、数据集仓库或论文附带代码核验 T2 数字 ID 映射；核验不到时必须保留 numeric_id，不得猜测 family 名称。
2. AICD T1/T2/T3 是不同任务视图，不要把三者行数相加当成互不重复样本。
3. h2_authorbench_dcan：task_id 是 group key；每个任务至少有两个 family。按任务做 train/dev/test，不能按行随机切分。
4. h2_llm_codegen：task_id=scenario:cwe_id；先检查 CSV response 解析、空代码、代码围栏和 model/family 映射。把 simple/secure 当作场景变量，不当作 family。
5. STACAD：以 file_name 作为 group，沿用 folds.npy 或 split_v1；不能把同一个文件的 pair 分到不同折。
6. Droid：有 generator/family，但无公开 task_id，只能作为 generator-held-out 和外部压力测试，不能用作同题语义对齐的核心训练集。
7. CoDET-M4：只做结构化来源指纹控制；target=human/ai 与 model 字段要分开，缺失 model 不得创建新 generator。
8. 输出数据审计结果和切分清单后，再进入模型实验。

第三步先做 P0 强基线，未完成前不要写新方法结论：
1. 在 AICD T2 官方 train/validation/test 上运行字符 n-gram TF-IDF + 线性分类器；先做轻量冒烟子集，再做完整训练。
2. 在 h2_authorbench_dcan 上运行同一套 TF-IDF baseline，并报告 task-held-out family Macro-F1。
3. 在 STACAD 上运行文件级五折 baseline；在 Droid 上运行 generator-held-out 外部测试。
4. 只有已有 encoder 权重且不需要额外占用磁盘时，才运行 CodeBERT/ModernBERT/CodeT5 类 baseline；否则先保存 TF-IDF 和结构化特征结果。
5. 每个结果都记录 Macro-F1、balanced accuracy、per-class recall、混淆矩阵、ECE、每个 generator/family/language 的指标。

第四步实现 DCAN 风格三分支模型，先用小数据验证：
1. semantic branch：输入代码，学习任务/行为表示；在同一个 task_id 内把不同 family 的输出作为语义正对。
2. fingerprint branch：预测 generator/family，并加入 token、格式、AST/控制流统计或已有结构特征；这一分支负责来源指纹。
3. nuisance adversary：从 fingerprint branch 对 task_id/language/length 做对抗去除；从 semantic branch 对 family/generator 做对抗去除。
4. 加入两个表示的正交或交叉协方差惩罚，但先做权重小的消融，不能一次叠加所有损失。
5. 第一轮只比较四个模型：semantic-only、fingerprint-only、late-fusion、full-disentangle。每个模型最多 3 个 seed。
6. 只在有明确 task_id/file_name 的数据上使用语义正对；Droid 不参与这一项 loss。

第五步固定评测协议：
1. closed-set：同 generator 训练和测试。
2. task-held-out：AuthorBench/LLM-CodeGen 按 task_id 或 CWE group 留出。
3. file-held-out：STACAD 按 file_name/fold 留出。
4. unseen-generator：Droid 按 generator 留出；AuthorBench 只把 OpenAI 多 generator 作为辅助折，不能写成多 family H2。
5. AICD 只在官方切分上做 numeric-ID family baseline，标签映射未核验前不能写具体家族名称。
6. 每个主结果同时报告 known-family Macro-F1、balanced accuracy、ECE、unknown-family AUROC 和 generator-level recall。

第六步输出：
1. scripts/ 下保存可重复脚本，所有路径相对项目根目录。
2. d-det/artifacts/acl_dcan_round1/ 下保存 config、metrics.json、predictions 的压缩版本、split manifest、环境版本和 README。
3. d-det/docx/ 下保存一份实验报告，写清数据、切分、损失、seed、GPU、显存峰值和失败实验。
4. 只保留最佳 checkpoint、最终日志和小型预测文件；删除中间 checkpoint、缓存 embedding 和重复转换文件。
5. 最后明确回答：语义/来源分离是否优于同 backbone late-fusion；提升来自哪个数据源；在 task/file/generator held-out 是否仍然成立。
```

## 服务器现有数据的用途

| 数据 | 首要用途 | 禁止的解释 |
|---|---|---|
| `acl_attribution_collection_v1/raw/aicd` | 大规模 AICD Task 2 基线 | 不能在未核验映射时给数字标签命名；不能把三配置相加成独立样本数 |
| `h2_authorbench_dcan` | 同 task 跨 family 语义正对 | C-only，prompt 相同只是语义代理，不是功能等价证明 |
| `h2_llm_codegen` | 同 CWE 跨模型补充对齐 | 安全提示可能产生词法捷径；必须按 task group 切分 |
| `stacad_v2` | 文件级 paired/OOD | `file_name` 是当前可用 group，不自动等于 repository group |
| `h2_droid_full_selected` | generator/family 外部压力测试 | 无 task_id，不能承担核心语义对齐 loss |
| `codet_m4` | 结构化来源指纹控制 | 没有可靠同 task 跨 generator 正对，不作为 DCAN 主训练集 |

## 资源和磁盘规则

服务器上传后只有约 2.7GB 可用空间，实验开始前重新确认：

```bash
cd /root/autodl-tmp/u-det
df -h /root/autodl-tmp
find d-det/data/acl_attribution_collection_v1/raw/aicd -name '*.parquet' | wc -l
```

运行 CPU 预处理时：

```bash
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false
```

不要执行下列操作：

- 不要把 AICD parquet 全部转成 JSONL；
- 不要同时保留 STACAD 压缩包、解压副本和另一份转换副本；
- 不要在本地缓存全量 embedding；
- 不要保存每个 seed 的完整模型，只保存最佳 checkpoint 和 metrics；
- 不要在审计前运行大规模训练；
- 不要删除旧的 `artifacts/`、`runs/`、`checkpoints/`，除非确认是重复缓存并记录删除内容。

## SOTA 判定规则

服务器端 AI 不得因为闭集 Macro-F1 提升就写 SOTA。至少满足以下条件才可以在报告中使用“优于基线”：

1. 同一数据源、同一 backbone、同一 split、同一采样预算下超过强 baseline；
2. task-held-out 或 file-held-out 不同步恶化；
3. unseen-generator 上相对 baseline 有稳定提升；
4. 去除长度、语言、任务文本和来源数据集捷径后，增益仍然存在；
5. 至少 3 个 seed 或 bootstrap 置信区间支持方向一致；
6. 所有 family/generator 映射、许可证和数据来源可追溯。

如果只在 AICD 闭集测试上提升，报告必须写成“闭集改进”，不能写成跨 generator 家族归因 SOTA。

