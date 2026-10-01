# d-det H2 DroidCollection 实验交接指导

## 0. 给执行端的任务边界

本文件是给 DeepSeek 实验编码端的交接说明。目标是在 AutoDL 上完成一次可复现的 H2 generator-held-out 实验，不是继续搜索新的损失或几何结构。

当前研究结论已经固定：E35 在七家族/23 generator 数据上，F1 相对 F0 的两折方向均为正，但 pooled 增量只有约 0.28/0.30 个百分点，低于预先设定的 1 个百分点出口。因此：

- 不修改现有 E35/E35 报告；
- 不继续调 SupCon 温度、损失权重、门控、学习率或 epoch；
- 不添加 DMHM、DCAN、局部密度、协方差、正交或低秩分支；
- 不生成新的合成数据；
- 只完成 DroidCollection 上的数据构型复核和同协议的 H2 候选实验。

如果实现或数据检查发现关键条件不满足，应停止并报告阻塞原因，不要用随机切分、更多调参或替换标签来“修复”结果。

## 1. 研究问题和结论边界

H2 的可检验形式是：同一 `Model_Family` 内、不同 `Generator` 的关系约束，能否在 generator-held-out 条件下保留家族来源信息。

本数据没有公开的 `task_id` 或 `prompt_id`。因此本实验最多支持以下表述：

> DroidCollection 上的跨 generator 家族归因候选证据。

不能写成同任务配对、prompt-invariant 泛化、后训练因果效应或已经学到纯粹的归因空间。

## 2. 数据和服务器路径

AutoDL 项目目录：

```text
/root/autodl-tmp/u-det/d-det
```

数据目录：

```text
/root/autodl-tmp/u-det/d-det/data/h2_droid_subset
```

文件含义：

| 文件 | 用途 |
|---|---|
| `core.jsonl` | 18,000 条主实验样本；14,220 条 `MACHINE_GENERATED`，3,780 条 `HUMAN_GENERATED` 控制样本 |
| `diagnostic_hybrid_adversarial.jsonl` | 2,000 条独立诊断样本；refined 与 adversarial 各 1,000 条，不能混入主实验训练或评分 |
| `summary.json` | 数据源 revision、family/generator 清单、计数和字段 |
| `fold_plan.json` | 两折 generator-held-out 方案和机器可读分配规则 |
| `SHA256SUMS.txt` | 规范 LF 字节校验 |

核心字段：

```text
Code, Generator, Generation_Mode, Source, Language,
Sampling_Params, Rewriting_Params, Label, Model_Family,
split_source, source_row_sha1
```

主数据覆盖 7 个 family、32 个 machine generator、7 种语言。`Source` 可以作为域/来源分层变量，但不能当作 task_id。

处理约束：

- JSONL 必须逐行或分批读取，不要一次性 `json.load` 全文件；
- 不下载 DroidCollection 原始 35GB 数据；
- 特征提取使用小 batch，先以 `batch_size=8` 或 `16` 冒烟；
- 服务器有充足显存，但仍需限制 tokenizer 长度和缓存大小；
- 生成的 feature cache、checkpoint 和 predictions 放在 `runs/` 或 `artifacts/`，不要再次提交大型二进制文件。

## 3. 折叠协议

不要自行重新随机划分。严格读取：

```text
data/h2_droid_subset/fold_plan.json
```

每个 family 的两折中，held-out generator 与 training generator 完全不相交；fold 0 和 fold 1 互补。具体规则：

- 训练机器样本：`Label=MACHINE_GENERATED`、`split_source=train`、`Generator ∈ train_generators`；
- 验证机器样本：`Label=MACHINE_GENERATED`、`split_source=dev`、`Generator ∈ train_generators`；
- 最终测试机器样本：`Label=MACHINE_GENERATED`、`split_source=test`、`Generator ∈ heldout_generators`；
- Human 控制样本按相同的 `split_source` 保留，但不参与 family 标签训练；
- `diagnostic_hybrid_adversarial.jsonl` 完全独立，只做附加鲁棒性分析。

测试 generator 标签没有出现在该折训练侧。因此不能训练一个包含这些 unseen generator 类别的普通 generator 分类器，再把它的结果称为 unseen-generator 识别。主任务是 family attribution；`BA_G` 只表示按 generator 分组的 family recall 平均值，必须同时报告每个 held-out generator 的 recall。

执行前必须打印并断言：

1. 每个 family 的 train/held-out generator 集合均非空；
2. 两侧 generator 交集为 0；
3. 训练、验证、测试的 `source_row_sha1` 无重复；
4. 每个 held-out generator 在最终 test 中有样本；
5. 不读取 `test` 来选择超参数或 early stopping。

## 4. 表示提取

第一版只使用既有 d-det 的冻结 768 维编码流程，不训练 encoder。先阅读并复用：

- `d-det/scripts/flagship_e28_probe.py`
- `d-det/scripts/flagship_e35.py`
- `d-det/configs/ddet_v041.yaml`
- `d-det/encoders/`
- `d-det/models/`

目标是保持与 E28/E35 相同的 tokenizer、截断、pooling、标准化和 768 维输出口径。若已有 checkpoint 或模型资产缺失，必须在报告中明确缺失项；不能静默替换成另一个 encoder 后仍称“复现 E35”。

建议新增独立脚本，例如：

```text
d-det/scripts/h2_droid_e36.py
```

脚本应至少支持：

```text
--data-dir
--out
--batch-size
--max-length
--smoke
--force
```

特征缓存需要保存逐行 `source_row_sha1` 或等价 metadata hash，并在训练前和 JSONL 重新核对，防止特征顺序错位。建议保存：

```text
artifacts/h2_droid_e36/features_*.npz 或 memmap
artifacts/h2_droid_e36/metadata.jsonl
artifacts/h2_droid_e36/manifest.json
```

不要把完整原始代码复制到多个中间文件。

## 5. 唯一主实验：B0、F0、F1

### B0：冻结表示上的加权 family 主轴

对每一折训练机器样本的 768 维特征拟合加权 multinomial Logistic Regression：

```text
C = 0.1
max_iter = 10000
tol = 1e-6
```

权重沿用 E35：

```text
w_i = N / (K * |G_f| * N_g)
```

其中 `K=7`，`G_f` 是该 family 的训练 generator 数，`N_g` 是 generator 样本数。B0 训练完成后冻结，作为所有 F0/F1 的同一主轴。

### F0：来源残差 CE

只训练来源残差，主轴全程冻结：

```text
r = V GELU(U z + b1) + b2
logits = logits_B0(z) + gamma * W_R r
gamma = 0.1 * tanh(eta)
```

阶段 B 修正初始化必须保持：

```text
W_R = 0
eta = 0.5
U Xavier 初始化
V 权重 std = 1e-3
其余 bias = 0
```

训练损失只有 family CE：

```text
F0: L = L_F
```

### F1：只增加跨 generator 正对 SupCon

F1 与 F0 必须共享：

- B0 主轴；
- 残差初始参数；
- 随机种子；
- batch schedule；
- 标准化统计量；
- epoch、optimizer 和 gradient clip。

唯一差异是：

```text
F1: L = L_F + 0.1 * L_cross
tau = 0.1
```

正对严格定义为：

```text
same Model_Family
different Generator
```

不得把同 generator 正对、不同 family 正对或 Human 行混进 `L_cross`。分母使用 batch 内所有非自身样本；没有跨 generator 正对的 anchor 跳过。每折和每个 family 报告有效 anchor 比例。

建议沿用 E35 的 family→generator→sample 均衡 batch。若某一小 family 在 batch 内无法形成正对，只记录并跳过该 anchor，不要改变正对定义来提高覆盖率。

## 6. 指标和预注册判读

主指标只在 held-out test generator 上计算：

```text
BA_F = family recall 的宏平均
BA_G = 各 family 内 held-out generator recall 的宏平均，再对 family 平均
```

同时保存：

- pooled 和逐折 `BA_F`、`BA_G`；
- 每个 family recall；
- 每个 held-out generator recall；
- language、Source、Generation_Mode、代码长度分桶；
- 检测 AUROC 作为次要结果；
- anchor 总数、有效数和比例；
- B0/F0/F1 的初始 logits 一致性与残差漂移量。

预注册出口：

1. 两折中 F1−F0 的 `BA_F` 和 `BA_G` 均为正，且 pooled 两项均至少增加 1 个百分点：H2 获得初步支持，可由指导端决定是否做一次局部几何消融；
2. 方向一致但 pooled 增量小于 1 个百分点：当前 Droid 子集和当前 768 维表示下不支持实用 H2，停止继续堆叠损失；
3. 两折方向不一致或 pooled 增量非正：H2 不支持；
4. anchor 比例过低：结论为“检验不足”，不能写成阴性；
5. 只改善单个 family、role 或 generator：只报告异质性，不能写成总体支持。

随机切分只作迁移损失参照，不作为 H2 通过条件。

## 7. 推荐执行顺序

### 阶段 A：只读审计

先实现或运行轻量审计，输出：

```text
artifacts/h2_droid_e36/audit.json
```

至少包含总行数、split/label/family/generator/language 计数、每折 generator 交集、每折样本数和重复 hash 数。审计不加载 encoder。

### 阶段 B：特征冒烟

使用每个 family 少量样本完成一次 encoder batch，检查：

- 输出维度为 768；
- 无 NaN/Inf；
- metadata 顺序和 hash 对齐；
- GPU 显存和吞吐可接受；
- 同一输入重复编码结果一致。

### 阶段 C：小规模协议冒烟

只跑一个折、每个 family 每个 generator 少量行，验证：

- B0 能收敛；
- F0/F1 初始 logits 与 B0 逐位一致；
- F1 确实产生跨 generator anchor；
- F0/F1 使用同一 batch schedule；
- 输出 metrics/predictions/manifest。

### 阶段 D：完整两折

默认拒绝覆盖已有产物，建议输出：

```text
d-det/artifacts/h2_droid_e36/
├── audit.json
├── config.json
├── manifest.json
├── metrics.json
├── predictions.npz
├── solver.log
└── report.md
```

完整实验结束后先检查 `metrics.json` 和 `manifest.json`，再写报告。不要先根据单个分数修改协议。

## 8. 验收标准

提交给指导端的最小结果必须包括：

1. 新脚本路径和完整运行命令；
2. `audit.json` 的数据计数和 generator 泄漏断言；
3. 特征提取口径、checkpoint、tokenizer、max length、batch size；
4. B0/F0/F1 两折及 pooled 指标；
5. anchor 比例和逐 generator 结果；
6. `manifest.json` 中的 git commit、数据 revision、输入 hash；
7. 明确是否读取过 test、是否使用过 diagnostic 文件；
8. 失败时保留日志和部分产物，不用调参掩盖失败。

最终报告必须把“DroidCollection 上的 generator-held-out 候选结果”和原 E35 结论分开，不能把这次换数据实验写成已经证明 H2。

