# d-det AuthorBench 任务条件归因实验 DeepSeek 执行指导

## 0 执行目标

这次实验不再生成数据，也不继续调 SupCon、门控或几何损失。目标是回答两个更基础的问题：

1. 同一任务的内容和实现路径是否主导了 family 归因；
2. 减去任务中心后，family 归因是否更容易跨任务泛化。

本实验是**任务条件归因审计**，不是完整的多 family H2 主实验。公开数据中只有 OpenAI family 同时包含多个 generator，其他 family 各只有一个 generator。因此不能把最终结果写成“多 family 跨 generator H2 已验证”。

## 1 服务器路径和数据

AutoDL 项目目录：

```text
/root/autodl-tmp/u-det/d-det
```

已上传并解压的数据目录：

```text
/root/autodl-tmp/u-det/d-det/data/h2_authorbench
```

文件：

| 文件 | 用途 |
|---|---|
| `core.jsonl` | 1,912 行；239 个 prompt；每个 prompt 有 8 个模型各 1 条代码 |
| `task_index.jsonl` | 239 个任务及其任务级 train/dev/test 划分 |
| `fold_plan.json` | 任务切分和 OpenAI generator 辅助留出方案 |
| `summary.json` | 行数、模型、family、hash 和限制 |
| `README.md` | 数据说明和使用边界 |

数据事实必须先断言：

- 总行数 1,912；任务数 239；每个任务恰好 8 行；
- 模型为 Claude、GPT-4o、GPT-4.1、GPT-4o-mini、Gemini、Qwen、DeepSeek、Llama；
- family 映射为 6 类，其中 OpenAI 包含 3 个 generator；
- 任务切分为 167/36/36 个任务；同一 `task_id` 不得跨 split；
- `replicate_count` 只表示源数据中该 prompt-model 的重复数，不能把重复输出当作独立任务。

不要上传或解压原始 35GB 数据，也不要把完整源压缩包复制到多个目录。

## 2 资源约束

所有命令固定：

```bash
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
```

Python 中还要设置：

```python
torch.set_num_threads(2)
torch.set_num_interop_threads(2)
```

这次不需要重新训练 encoder。特征抽取使用小批量 GPU 推理即可，建议先 `batch_size=8` 冒烟，再视显存改为 16。若 checkpoint 或 tokenizer 不存在，应停止并报告，不要自动联网下载大模型。

## 3 表示提取口径

主实验使用本地未微调 CodeT5-base，避免把 d-det 训练过的表示和归因结论混在一起：

```text
d-det/checkpoints/codet5-base
```

复用 `d-det/encoders/codet5.py` 的加载方式，冻结 encoder，只提取 token hidden state 并做 attention-mask mean pooling，得到 768 维向量 `z`。

CodeT5 的标准上下文长度是 512。默认：

- `max_length=512`；
- 超长代码按头 384 + 尾 128 截断；
- 空代码或有效 token 少于 8 的行跳过并记录；
- bf16/float16 只用于 GPU 推理，缓存保存为 float32 或 float16，并记录 dtype；
- 逐行保存 `task_id`、`model_name`、`family`、`source_sha256`，训练前重新核对顺序。

建议新增脚本：

```text
d-det/scripts/authorbench_taskaware.py
```

缓存和产物放在：

```text
d-det/runs/authorbench_taskaware/
d-det/artifacts/authorbench_taskaware/
```

## 4 必须完成的实验矩阵

### A 数据审计

先不加载 encoder，输出：

```text
artifacts/authorbench_taskaware/audit.json
```

包括任务、模型、family、重复数、代码长度、`nloc`、复杂度、`token_size` 的计数和分布。检查同一 `task_id` 是否完整覆盖 8 个模型，并检查源 hash 是否重复。

### B 四种输入表示

对每个任务的 8 个输出，构造四种表示：

1. `z_raw`：原始 768 维向量；
2. `z_center`：同一任务 8 个向量的无标签均值中心化：

   ```text
   mu_t = mean(z for the eight models of task t)
   z_center = z - mu_t
   ```

3. `z_center_std`：在训练任务上估计每维标准差后，对 `z_center` 标准化；
4. `meta_only`：只使用 `num_lines`、`nloc`、`CC`、`token_size` 等元数据。

`z_center` 是任务条件的转导式视图，因为测试任务的 8 个输出都存在。必须在报告中明确这一点。它不代表单样本部署算法。

### C family 归因

在固定任务级 train/dev/test 划分上，分别训练：

- multinomial Logistic Regression，`C ∈ {0.03, 0.1}`；
- 线性 LDA 或线性 SVM 作为简单交叉检查；
- 不训练深度头，不训练 encoder。

标准化参数、PCA 参数和超参数只能用 train 任务拟合；dev 只用于选择固定的一个配置；test 只做最终评估。

报告：

- family macro-F1；
- balanced accuracy；
- 每个 family recall；
- confusion matrix；
- 任务级 bootstrap 置信区间；
- 与 6 类机会水平约 0.1667 的比较。

必须同时报告 `z_raw`、`z_center`、`z_center_std` 和 `meta_only`，否则无法判断中心化是否真的去掉了任务捷径。

### D model 归因作为捷径诊断

把标签换成 8-way `model_name`，同样比较 `z_raw` 和 `z_center`。这不是论文主任务，而是诊断：

- 如果 model 8-way 很高、family 6-way 也高，可能主要学到了具体模型指纹；
- 如果 `z_center` 后 family 仍保持，而 model 归因下降，说明任务中心化可能去掉了一部分模型或任务捷径；
- 如果二者同时坍塌，只能说明当前表示没有稳定来源信号。

8 类机会水平约为 0.125。不要把 model 归因准确率写成 H2 通过。

### E PCA、聚类和流形诊断

只做描述性诊断，不用它替代最终指标：

- train-fit PCA 的前 2 维散点图：raw 与 center 各一张，颜色分别标 family 和 task split；
- KMeans `k=6` 的 train-only silhouette、ARI 和 family purity；
- LDA 投影的二维图；
- raw 与 center 的类内/类间协方差迹比。

所有图必须标注“描述性，不是 H2 证据”。如果 PCA 图显示团块，不得据此宣称存在 family manifold；如果 KMeans 失败，也不能据此宣称 family 不可判别，判别式和聚类是不同问题。

## 5 OpenAI generator 留出辅助诊断

这个数据只有 OpenAI family 有多个 generator，所以只做一个明确标注为辅助的诊断：

```text
holdout_gpt-4o
holdout_gpt-4.1
holdout_gpt-4o-mini
```

每次把一个 OpenAI model 作为 held-out generator，其他 7 个 model 作为训练侧。测试只统计被留出 generator 的：

- family=OpenAI 的 recall；
- model-name 的 recall；
- raw 与 center 两种表示的差异。

由于测试只覆盖一个 family，这不是整体 `BA_F`，不能和 E35/E36 的 7 family 结果并列。它只能回答：“见过两个 OpenAI generator 后，第三个 OpenAI generator 是否仍落在 OpenAI family 的判别侧”。

同一任务内使用 8 个输出求中心是 transductive 口径；另外再做一个严格任务留出版本，避免把两种口径混在一起：

- 任务条件版本：同一 task 的兄弟输出可用于无标签中心；
- 任务泛化版本：测试 task 完全不参与训练或标准化拟合。

## 6 结果判读规则

本实验不预注册新的“超过多少个百分点就成功”的 H2 门槛，先做机制审计：

1. `z_center` 相比 `z_raw` 在任务留出上提升，且 `meta_only` 很低：支持“任务内容是主要干扰因素，中心化有帮助”；
2. `z_center` 只在同任务转导版本提升，在任务留出版本不提升：说明收益依赖测试时兄弟样本，不能写成单样本归因；
3. model 归因高于 family 归因：说明具体模型捷径强，family 关系未必稳定；
4. OpenAI held-out generator 仍能被归入 OpenAI，且 model 归因下降：只能作为 within-family 的探索性支持；
5. 所有表示都接近机会：当前 CodeT5 表示不能支持任务条件来源归因；
6. 任何单个 family 或单个 prompt 子集变好，只报告异质性，不写成总体结论。

## 7 运行顺序

先运行审计：

```bash
cd /root/autodl-tmp/u-det/d-det
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
/root/miniconda3/envs/udet/bin/python scripts/authorbench_taskaware.py --audit-only
```

再运行小规模特征冒烟：

```bash
/root/miniconda3/envs/udet/bin/python scripts/authorbench_taskaware.py \
  --smoke --batch-size 8 --max-length 512 \
  --out artifacts/authorbench_taskaware_smoke
```

最后运行正式实验：

```bash
/root/miniconda3/envs/udet/bin/python scripts/authorbench_taskaware.py \
  --batch-size 8 --max-length 512 \
  --out artifacts/authorbench_taskaware
```

正式目录至少包含：

```text
audit.json
config.json
manifest.json
metrics.json
predictions.npz
solver.log
report.md
```

脚本默认拒绝覆盖已有正式产物。失败时保留日志并报告原因，不要通过扩大 epoch、改切分、加入测试任务或下载新模型来绕过失败。

## 8 交付要求

DeepSeek 最终只需交付：

1. `scripts/authorbench_taskaware.py`；
2. `artifacts/authorbench_taskaware/`；
3. `report.md` 中的 raw/center/meta-only 对照；
4. PCA/聚类图及其生成命令；
5. 明确是否使用了 GPU、最大显存、CPU 线程数；
6. 明确任务中心化属于 transductive 还是严格任务留出；
7. 明确 OpenAI generator 结果不能代表多 family H2。

不要修改 E35/E36 正式报告，不要把这个 C-only 小数据集的结果外推到 DroidCollection、SemEval 或所有编程语言。
