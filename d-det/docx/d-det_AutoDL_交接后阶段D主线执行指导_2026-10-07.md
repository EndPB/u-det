# d-det AutoDL 交接后阶段 D 主线执行指导

日期：2026-10-07
服务器基准：commit `83e02c0`，工作树干净
前置状态：交接包、CodeT5-small、Stage A/B/C 校准全部通过

## 0. 这轮校准意味着什么

服务器回传已经确认：

- CodeT5-small 7 个资产齐全，词表 32100，权重和本机产物 4/4 SHA256 一致；
- 交接包 33/33 成员校验通过，overlay 无覆盖损失；
- A：4692 pair records、0 invalid、6/6 admitted folds；
- B1：冻结 CodeT5-small H2 pair raw BA `.6735`，与本机 `.6739` 的差异为 `-.0004`；
- B2：AuthorBench task-heldout BA `.597222`、macro-F1 `.576036`，与本机逐位一致；
- C：P0 只读结果 fusion `.8388`、STACAD `.7330`、Droid `.1645` 均已在位。

这些结果证明代码、数据包、权重、切分和读数可以复现。它们不证明 H1/H2/H3 已经通过，也不构成新的 SOTA 增益。不要因为校准成功而重复同一批 B1/B2/P0 训练。

## 1. 下一阶段唯一目标

阶段 D 的问题是：

> 在已经冻结的 task-aware 数据角色和强基线之上，能否把“任务效应”和“家族来源效应”分开，并在 generator-heldout 关系上得到至少 1 个百分点、跨折同向、检测不下降的增量？

阶段 D 分成两个闸门：

1. **D0 支持矩阵**：只做数据和协议审计，不训练；
2. **D1 H1 证据闭合**：在同一 task-heldout 主赛道上完成控制矩阵，确认是否值得进入唯一 H2 方法实验。

D0 未通过时，不启动神经训练；D1 未通过时，不启动 H2 几何、SupCon、DMHM 或 H3。

## 2. D0：建立支持矩阵（只读）

服务器侧 AI 先在现有工作目录运行：

```bash
cd /root/autodl-tmp/u-det
mkdir -p d-det/artifacts/stage_d_support_2026-10-07/logs
git rev-parse HEAD > d-det/artifacts/stage_d_support_2026-10-07/git_head.txt
git status --short --branch > d-det/artifacts/stage_d_support_2026-10-07/git_status.txt
python scripts/audit_acl_local_readiness.py   --out d-det/artifacts/stage_d_support_2026-10-07
```

此外，必须对 `d-det/data/h2_alignment_v3/fold_plan.json` 和 `pair_index.jsonl` 做一个流式支持审计，输出：

```text
d-det/artifacts/stage_d_support_2026-10-07/alignment_support.json
```

每个 fold 至少记录：

- source、family、heldout_generator、seen_generators；
- train/dev/test 行数、任务数、family 数、generator 数；
- train/dev/test 的 task 交集；
- heldout generator 是否出现在 train/dev；
- train/dev 是否同时有正负 family 对；
- 同 family、不同 generator 的正对数；
- language、base_model、model_role、repository_id 的缺失比例；
- exact code hash 跨 split 数；
- 是否 admitted，以及失败原因。

D0 的硬规则：

- `h2_pair_benchmark_v1` 的 6 个 admitted folds 继续作为正式 pair 诊断；
- `h2_alignment_v3` 中的 Google/Mistral 小折只作 auxiliary diagnostic；不能和 AuthorBench OpenAI、LLM-CodeGen Meta 合并成一个跨 source 平均；
- 没有 train/dev 正对的 fold 不训练；
- 没有 `task_id` 的 Droid 只能做 generator-heldout 压力测试，不能写成同题配对；
- 不把 `model_name` 的字符串事后解释成共同 base 或后训练角色；
- 不把 AICD 数字标签改名为 family。

D0 的交付是 `alignment_support.json`、命令、完整日志和 SHA256；没有这些文件，不进入 D1。

## 3. D1：AuthorBench-DCan task-aware H1 控制矩阵

D1 使用已有 `h2_authorbench_dcan/core.jsonl`，不上传新数据：

- 9498 行；
- 2715 个 task；
- 6 个 family；
- C 语言；
- OpenAI 有 3 个 generator，其余 family 多数只有一个 generator。

因此 D1 是 task-heldout H1-A，不是多 family unseen-generator H2。主 split 必须保持数据包中的 `task_split`，同一个 `task_id` 不得跨 train/dev/test。

### 3.1 只比较预注册视图

按以下顺序完成，同一 split、同一 test-read-once 纪律：

1. metadata-only：family/generator/language/length 等字段只作 shortcut control；
2. word/char TF-IDF：词表只由 train 拟合，C/epoch 只由 dev 选择；
3. frozen CodeT5-small mean pooling：使用包内已校验权重；
4. task-centered 版本：只作为 transductive diagnostic，中心只能由同一 task 的无标签兄弟样本计算；
5. 如已有同切分 P0 fusion、word-TFIDF、style 预测，直接复用并重算指标，不重复训练。

每个视图统一输出：

- family macro-F1、balanced accuracy；
- 每 family recall/precision；
- 500 次 task-cluster bootstrap CI；
- confusion matrix；
- ECE/NLL（有概率输出时）；
- language、length、generator、task-size 分桶；
- 与 P0 fusion `.8388`、word-TFIDF `.8200` 的同切分差值。

### 3.2 D1 通过条件

D1 只有在以下条件同时成立时，才允许进入 H2 方法实验：

1. 至少一个内容表示在严格 task-heldout 上高于机会，并且不是只由 metadata 解释；
2. 相对同切分最强 P0，预注册方法的 macro-F1 或 BA 提升至少约 1 个百分点；
3. 改善在至少两个独立 seed 或两个支持充分的 fold 同向；
4. task-cluster 95% CI 的 paired improvement 不以 0 为主要覆盖区间；
5. task-centered 的增益被明确标成 transductive，不能拿来代表普通单样本部署；
6. detection/control 轴没有被混入 family 主指标。

如果 D1 只复现已有 P0、CodeT5-small 低于 `.8388` 或提升低于 1pt，结论应写成“task-aware 可读性/任务效应审计”，停止新架构探索，转入数据构造。

## 4. 只有 D1 通过后才允许的 H2 实验

H2 使用 `h2_alignment_v3` 的 admitted support 和现有 6 个正式 pair fold；不重新定义正对。

正对严格为：

[
P_{mathrm{cross}}(i)={j:f_j=f_i,;g_j
e g_i}.
]

建议唯一候选是“任务条件化的来源残差读出”，而不是再跑一遍旧 Stage B：

- 冻结已复现的 B0 family head；
- 允许一个小的来源残差或条件化线性读出；
- task/prompt 表示只作为 nuisance/control 输入；
- train/dev/test generator 完全隔离；
- 参数量、优化步数、batch、seed 与 B0 对齐；
- 只设置一个方法臂和一个无方法的 matched control；
- 不同时加入 SupCon、DMHM、低秩、正交、PCGrad、private adapter。

注意：旧 Stage B/E35 已经显示当前固定主轴 + 残差/SupCon 的 pooled 增量只有约 `+.28/.30pt`，未过 1pt 门槛；这条路线不能原样重跑。新实验若没有清晰的数据支持或新的任务条件接口，应直接停止。

H2 主表必须逐 generator 报：

- family BA/F1；
- generator-heldout BA/F1；
- generator 数、test task 数、有效 anchor 数；
- 无正对 anchor 比例；
- 相对 B0 的差值和 paired bootstrap；
- random split 仅作迁移损失参照；
- detection AUROC/F1 和 calibration。

H2 通过条件：至少两个支持充分的 heldout folds、方向同向、总体增量约 1pt 或更高、检测不下降。否则结论是“当前表示/数据协议下未观察到实用 H2 增量”。

## 5. H3 暂停条件

在 D1/H2 通过前，不运行：

- detection-only/family-only/joint 的新联合训练；
- DCAN 完整网络；
- DMHM 原始 768 维密度模型；
- 多个梯度冲突损失；
- 多个 adapter 或低秩组合；
- unknown-family/open-set。

H3 通过后才比较 shared/private adapter、PCGrad 和 conflict-only 梯度约束，并同时报告 family 增益、检测不降、梯度冲突率和梯度范数。单个平方 cosine 不能单独支持 H3。

## 6. 结果目录和回传格式

D0/D1/D2 每个实验使用独立目录，不覆盖校准结果：

```text
d-det/artifacts/stage_d_support_2026-10-07/
d-det/artifacts/stage_d_h1_authorbench_dcan_2026-10-07/
d-det/artifacts/stage_d_h2_relation_2026-10-07/
```

每个实际训练目录必须包含：

```text
config.json
metrics.json
predictions.npz
report.md
SHA256SUMS.txt
logs/
```

服务器侧 AI 的回传必须包括：

1. git HEAD 和工作树；
2. D0 支持矩阵及 admitted/diagnostic-only 判定；
3. 每个视图的完整指标和 CI；
4. 与 P0 的 paired 差值；
5. 是否触发 D1/H2 闸门；
6. 实际读取的 test 文件和时间；
7. 失败项、内存峰值、GPU 显存峰值和下一步唯一建议。

## 7. 明确的停止规则

出现以下任意情况就停止方法扩展：

- task/generator 支持不足；
- 只能通过随机切分获得增益；
- 只在一个 generator 或一个 seed 增益；
- 相对 P0 小于 1pt；
- test 被用于选择 tokenizer、阈值、C、epoch 或方法；
- 检测下降；
- task-centered 增益被误写成单样本能力；
- 需要上传大数据或权重才能继续，但未先证明该资产是当前闸门的必要条件。

当前最优先的工作是 D0 支持矩阵和 D1 同切分 H1 控制矩阵；不是再增加损失、再下载模型或重复已经逐位复现的校准实验。
