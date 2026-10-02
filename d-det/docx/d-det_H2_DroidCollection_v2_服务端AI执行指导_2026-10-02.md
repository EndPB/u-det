# d-det H2 DroidCollection v2：服务端 AI 执行指导

日期：2026-10-02  
适用对象：AutoDL 服务端实验编码 AI  
主任务：在较大 DroidCollection 子集上，完成一次可复现的 generator-held-out 家族归因实验。

## 0. 先读结论

原来的 h2_droid_local_50k 只有约 4.7 万条主数据，只能用于上传、解压和流程冒烟，不能作为 H2 的正式主实验集。

正式主包是：

    data/h2_droid_full_selected/

它包含：

- core.jsonl：146,718 条去重主样本；
- MACHINE_GENERATED：125,718 条；
- HUMAN_GENERATED：21,000 条控制样本；
- 7 个 Model_Family；
- 全部 32 个选定的机器 Generator；
- diagnostic_hybrid_adversarial.jsonl：2,000 条独立诊断样本；
- fold_plan.json：两折 generator-held-out 方案；
- SHA256SUMS.txt：完整性校验。

本轮目标是复现同一个 H2 问题，而不是继续搜索新的损失、调门控或扩大模型结构。

## 1. 服务端文件和完整性检查

假设仓库位于：

    /root/autodl-tmp/u-det

上传后先执行：

    cd /root/autodl-tmp/u-det
    unzip -q d-det/data/h2_droid_full_selected/h2_droid_full_selected_upload.zip \
      -d d-det/data/h2_droid_full_selected_upload
    cd d-det/data/h2_droid_full_selected_upload
    sha256sum -c SHA256SUMS.txt

正式包 ZIP 的 SHA256 应为：

    22074BBE18AF50AE52F5F07A2677750D4F8CB994420EC9DEEAC2F558CFE6B716

如果校验失败，停止实验并报告文件名、实际 hash 和上传日志。不要继续训练。

h2_droid_local_50k 只用于 smoke test，不得替代正式包。不要下载或恢复原始 DroidCollection 的 35GB 全量数据；当前正式包已经是流式筛选、去重后的实验输入。

## 2. 数据语义和论文边界

主任务标签是 Model_Family，不是 Generator。Generator 只用于构造训练/测试隔离。

本数据没有公开的 task_id 或 prompt_id，因此最终结果只能表述为：

> DroidCollection 上的 generator-held-out 家族归因候选证据。

不得表述为同题配对泛化、prompt-invariant 泛化、后训练因果效应或已经学到纯粹的家族归因空间。

core.jsonl 的主要字段：

    Code, Generator, Generation_Mode, Source, Language,
    Sampling_Params, Rewriting_Params, Label, Model_Family,
    split_source, source_row_sha1

处理规则：

- Label=MACHINE_GENERATED 是正式家族归因主样本；
- Label=HUMAN_GENERATED 是人类对照，主要用于检测和域偏移分析，不得把它当成一个模型家族参加 H2 家族分类；
- diagnostic_hybrid_adversarial.jsonl 只做附加诊断，不得混入训练、调参或主测试；
- Source、Language、代码长度和 Generation_Mode 是分析分层变量，不是新的标签；
- 所有 JSONL 必须逐行或分批读取，不得一次性读入内存。

正式主包每个机器生成器的样本量不完全相等，服务端实验必须报告每个 generator 的计数，并使用 generator-aware 的训练权重或采样策略。不得通过复制少数生成器样本来伪造均衡。

## 3. 折叠协议

严格读取：

    data/h2_droid_full_selected/fold_plan.json

不得重新随机划分，也不得用 test 选择超参数。

每一折都必须满足：

1. 每个 family 的训练 generator 和 held-out generator 都非空；
2. 训练侧与 held-out 侧的 generator 交集为 0；
3. 每个 held-out generator 在最终 test 中都有样本；
4. 训练、验证、测试之间没有重复 source_row_sha1；
5. held-out generator 的任何样本都不能出现在该折训练侧。

推荐的主任务取样规则：

- 训练：Label=MACHINE_GENERATED、split_source=train、generator 属于该折训练集合；
- 验证：Label=MACHINE_GENERATED、split_source=dev、generator 属于该折训练集合；
- 测试：Label=MACHINE_GENERATED、split_source=test、generator 属于该折 held-out 集合；
- 人类样本只按预先定义的 split 做检测/域分析，不参与 family CE 或跨 generator SupCon。

实验开始前必须生成：

    artifacts/h2_droid_v2/audit.json

至少记录总行数、split、label、family、generator、language 计数、每折 generator 交集、每折样本数和重复 hash 数。任何断言失败都应停止。

## 4. 表示提取

先复用仓库已有的 E28/E35 冻结编码口径：相同 encoder/checkpoint、tokenizer、max length、截断、pooling、标准化统计，输出维度必须为 768。

不得因为 checkpoint 缺失而静默换用另一个 encoder。若资产缺失，报告缺失路径并暂停正式实验。

先做小批量 smoke test：

- batch_size=8 或 16；
- 检查输出是否为 768 维；
- 检查 NaN/Inf；
- 检查 metadata 顺序和 source_row_sha1；
- 重复编码同一输入，确认结果一致；
- 记录显存峰值。

特征缓存应保留 metadata hash，不要在多个中间文件中复制完整代码。建议目录：

    artifacts/h2_droid_v2/
    ├── audit.json
    ├── features/
    ├── metadata.jsonl
    ├── config.json
    └── manifest.json

## 5. 唯一主实验：B0、F0、F1

### B0：固定家族主轴

在每折训练机器样本的 768 维表示上拟合加权 multinomial Logistic Regression：

    C = 0.1
    max_iter = 10000
    tol = 1e-6

推荐沿用 generator-aware 权重：

    w_i = N / (K * |G_f| * N_g)

其中 K=7，G_f 是该 family 在训练折中的 generator 数，N_g 是该 generator 的训练样本数，N 是训练总数。B0 完成后冻结，F0/F1 必须共享同一 B0 主轴。

### F0：来源残差 CE

固定 B0，仅训练来源残差：

    r = V GELU(U z + b1) + b2
    logits = logits_B0(z) + gamma * W_R r
    gamma = 0.1 * tanh(eta)

初始化沿用阶段 B：

    W_R = 0
    eta = 0.5
    U Xavier 初始化
    V 权重 std = 1e-3
    其余 bias = 0

损失为 F0: L = L_family_CE。

### F1：增加跨 generator 正对约束

F1 与 F0 必须共享 B0 主轴、残差初始参数、随机种子、batch schedule、标准化、optimizer、epoch 和 gradient clip。

唯一差异：

    F1: L = L_family_CE + 0.1 * L_cross
    tau = 0.1

正对严格定义为：

    same Model_Family
    different Generator

不得把同一个 generator、不同 family、HUMAN_GENERATED、diagnostic 文件或 held-out test generator 中的样本作为正对。

分母使用 batch 内所有非自身样本。没有跨 generator 正对的 anchor 必须跳过，并报告有效 anchor 数和比例。不得为了提高比例而改变正对定义。

## 6. 指标和预先固定的判读

主指标只在 held-out test generator 上计算：

    BA_F = 各 family recall 的宏平均

    BA_G = 每个 family 内，各 held-out generator 的 family recall 平均，
           再对 family 做宏平均

必须同时保存 pooled、逐折、逐 family 和逐 held-out generator 结果，并按 Source、Language、Generation_Mode、长度分桶。检测 AUROC 作为次要结果，同时保存 anchor 总数、有效数和比例、F0/F1 初始 logits 是否一致、残差范数和参数漂移。

固定判读：

1. 两折中 F1−F0 的 BA_F 和 BA_G 都为正，且 pooled 两项都至少增加 1 个百分点：H2 获得初步支持，暂停并提交结果，不要自动添加新损失；
2. 方向一致但 pooled 增量小于 1 个百分点：当前 768 维表示和当前数据构型下没有实用 H2 增量；
3. 两折方向不一致或 pooled 增量非正：H2 不支持；
4. anchor 比例过低：结论为“检验不足”，不能直接写成阴性；
5. 只改善一个 family、role 或少数 generator：只能报告异质性，不能写成总体 H2 通过。

随机切分只能作为附加参照，不得作为 H2 通过条件。

## 7. 执行顺序

### 阶段 A：只读审计

先生成 audit.json，确认数据、折叠和 hash 完整性。

### 阶段 B：特征冒烟

只编码少量样本，确认 encoder、显存和 metadata 对齐。

### 阶段 C：小规模协议冒烟

一个 fold、每个 family/generator 少量样本，确认 B0 能运行、F0/F1 初始 logits 与 B0 一致、F1 确实产生跨 generator anchor、F0/F1 使用同一 batch schedule，并且能写出 metrics、predictions、manifest。

### 阶段 D：完整两折

推荐输出：

    artifacts/h2_droid_v2/
    ├── audit.json
    ├── config.json
    ├── manifest.json
    ├── metrics.json
    ├── predictions.npz
    ├── solver.log
    └── report.md

完整实验结束后先提交机器可读的 metrics.json、audit.json、manifest.json 和运行日志，再写自然语言结论。

## 8. 明确禁止事项

本轮禁止：

- 把 47k 小包当正式主实验；
- 下载原始 35GB 全量数据；
- 重新随机切分；
- 读取 test 来调参或 early stop；
- 把 generator 分类准确率当作 unseen-generator 归因结果；
- 把 Human 行或 diagnostic 行混入 H2 主损失；
- 调 SupCon 温度、损失权重、门控、学习率来追求正结果；
- 添加 DMHM、DCAN、局部密度、协方差、正交或低秩分支；
- 静默更换 encoder/checkpoint；
- 在单一 family 或单一 generator 的改善基础上宣布 H2 成功。

## 9. 最终交付

服务端 AI 必须提交：

1. 完整运行命令和新脚本路径；
2. audit.json 与 generator 泄漏断言；
3. encoder/checkpoint/tokenizer/max length/batch size；
4. B0/F0/F1 的两折和 pooled 指标；
5. anchor 比例、逐 family 和逐 generator 结果；
6. manifest.json 中的代码 commit、数据 hash 和配置；
7. 是否读取 test、是否使用 diagnostic 文件的明确说明；
8. 失败时保留日志和部分产物，不得用调参掩盖协议问题。

结果必须把“DroidCollection v2 上的 generator-held-out 候选结果”和既有 E35 结论分开报告。即使 F1 有小幅提升，也不能直接写成已经证明存在稳定、纯粹的家族归因空间。

