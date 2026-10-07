# H2 generator-heldout frozen CodeT5-small pair verification

日期：2026-10-07
状态：完成；本机 RTX 3050 4GB；仅冻结表示 + 线性 pair head；不包含神经网络训练。

## 研究问题

在与词法 TF-IDF 基线相同的六个 admitted generator-heldout 折上，冻结的 CodeT5-small 上下文表示是否能从同任务的代码对中恢复目标 family 的来源关系？这仍是 **pair verification**，不是 K-way family attribution；Google 和 Mistral 两生成器折因留出后 train/dev 没有同族正对，保持 diagnostic-only。

## 固定协议

- 数据索引：`d-det/data/h2_pair_benchmark_v1/{pairs.jsonl,generator_fold_index.jsonl}`。
- admitted folds：AuthorBench/OpenAI 的 `gpt-4.1`、`gpt-4o`、`gpt-4o-mini` 三折；LLM-CodeGen/Meta 的 `codellama`、`llama2`、`llama3` 三折。
- 表示：`Salesforce/codet5-small` encoder，`T5EncoderModel`，冻结全部参数；词表为 32,100 的 RoBERTa BPE。因当前 Transformers 5.4 的 `AddedToken` 配置兼容性问题，脚本显式用 `RobertaTokenizer(vocab=..., merges=...)` 构造 tokenizer，避免旧参数名造成五词表空 tokenizer。
- 输入：不添加 special token；超过 512 token 时取前 384 + 后 128；attention-mask mean pooling，得到 512 维向量。
- pair head：对 `[z_left; z_right]` 训练 `LogisticRegression(class_weight="balanced", max_iter=500)`；不使用 test 标签选模型或阈值。
- 选择与统计：阈值在 dev 以 BA 最大化；test 只评一次；同时保留 task-cluster bootstrap 500 次；总表是六折不加权描述性均值，不是 pooled test estimate。
- 运行命令：`PYTHONPATH=d-det/_vendor HF_HUB_OFFLINE=1 python scripts/h2_generator_heldout_codet5_small.py --batch-size 2`。
- 可复现入口：[`h2_generator_heldout_codet5_small.py`](../../scripts/h2_generator_heldout_codet5_small.py)；原始指标：[`metrics.json`](metrics.json)。

## Aggregate test metrics

| view | BA | F1 | ROC-AUC |
|---|---:|---:|---:|
| raw | **.6739** | .6756 | .7553 |
| ids_only | .6603 | .6646 | .7225 |
| strings_only | .6685 | .6182 | .7442 |
| comments_only | .6517 | .6002 | .7495 |
| all | .6545 | .5767 | .7238 |

逐折 test BA：

| fold | raw | ids | strings | comments | all |
|---|---:|---:|---:|---:|---:|
| OpenAI / gpt-4.1 | .7586 | .7622 | .7540 | .7412 | .7403 |
| OpenAI / gpt-4o | .6821 | .6372 | .6976 | .6819 | .6782 |
| OpenAI / gpt-4o-mini | .7051 | .6648 | .6589 | .6895 | .7186 |
| Meta / codellama | .5932 | .5682 | .5795 | .6045 | .5420 |
| Meta / llama2 | .7045 | .7045 | .6711 | .7057 | .7105 |
| Meta / llama3 | .6000 | .6250 | .6500 | .4875 | .5375 |

raw 视图的 task-cluster bootstrap（每折 500 次，按 test task 重采样）为：OpenAI/gpt-4.1 `.698–.817`、OpenAI/gpt-4o `.590–.756`、OpenAI/gpt-4o-mini `.618–.778`、Meta/codellama `.397–.776`、Meta/llama2 `.585–.824`、Meta/llama3 `.481–.689`。这些区间说明小折尤其是 Meta 折的不确定性很大；JSON 还保存了每个清洗视图的完整 bootstrap 结果。

## 解释边界

1. `raw` 在这组六折中最高，但只比 `strings_only` 高约 0.54 BA 点，且折间方向不一致；因此只能说冻结 CodeT5 表示含有可读来源信息，不能说恢复了 generator-invariant family 因果因素。
2. `all` 没有稳定优于 raw 或清洗视图；这与 TF-IDF 的结果方向一致，说明把所有表面内容混合进表示不自动提高迁移。
3. Meta 三折的 BA 波动（`.5932/.7045/.6000`）和小测试集意味着区间会很宽；不能把它外推成跨 family 结论。
4. 该结果不能直接与 CodeT5-base、已微调 d-det、P0 ensemble 或 K-way 主任务的数字排序；模型规模、输入协议、pair head 和目标任务不同。
5. 所有源数据仍缺乏可直接使用的共同 `base_model/model_role` 控制，所以这是来源可读性/迁移诊断，不是 post-training 的因果估计。

## 运行与资产校验

- `torch 2.11.0+cu128`，Transformers `5.4.0`，CUDA 可用；实际设备 `cuda`，前向 dtype `float16`；运行时间约 340 秒。
- CodeT5-small 权重加载时仅报告 `lm_head.weight` 为 encoder 架构的 unexpected key；该键属于被丢弃的 decoder/language-model head，不影响 encoder-only 表示。
- SHA256：
  - `config.json`: `c067a6377fc235b103a1577fcb8a114e2885b700198bad22e34f84659d0025ca`
  - `vocab.json`: `43bb485f4de0f2fd49b370bef4efab23ea3ab6d0019e72bdd9cec1436a6eaa2b`
  - `merges.txt`: `5d346f84939a98df0cde902df7d60154b461cde1b90dc653ab9b81d74e752a4d`
  - `pytorch_model.bin`: `968fb0f45e1efc8cf3dd50012d1f82ad82098107cbadde2c0fdd8e61bac02908`

## 对 ACL 主线的作用

这轮实验把“强冻结语义表示是否已经有 H1 能力”从猜测变成了可复现实证：在严格 generator-heldout pair 协议中，CodeT5-small 的确高于 TF-IDF，但清洗视图和 all 视图没有稳定增益。下一步应先补足每 family 至少三个 generator 的 task-aware 主数据、输出 task-cluster CI，再考虑 H2 关系损失；不应因这一轮的 BA 直接宣称 generator-invariant attribution 或引入 DCAN/DMHM 堆叠。
