# AuthorBench task-heldout frozen CodeT5-small family baseline

日期：2026-10-07
状态：完成；六类 family 任务的冻结 encoder 对照，不是 generator-heldout H2。

## 问题与协议

在 `h2_authorbench/core.jsonl` 的固定 task-heldout split 上，冻结 CodeT5-small encoder 的上下文表示能否直接支持六类 family attribution？数据为 train/dev/test = 1336/288/288，239 个任务，六类 family；同一任务不会跨 split。该包中大多数 family 只有一个 generator，OpenAI 有三个，因此结果属于 task-heldout H1-A，不能外推为跨 generator 不变性。

- tokenizer：显式 `RobertaTokenizer(vocab=..., merges=...)`，32,100 词表；不加 special token。
- 输入：最大 512 token，超过时取前 384 + 后 128；attention-mask mean pooling，得到 512 维向量。
- 标准化：仅用 train 行拟合 `StandardScaler`。
- 读出：balanced multinomial logistic regression；`C ∈ {0.03, 0.1, 0.3}` 只以 dev macro-F1 选择，test 只评一次。
- 统计：test macro-F1、balanced accuracy 和 500 次按 task 重采样 bootstrap。
- 重跑入口：[`authorbench_codet5_small_family_baseline.py`](../../scripts/authorbench_codet5_small_family_baseline.py)；原始结果：[`metrics.json`](metrics.json)。

## 结果

| split | BA | macro-F1 |
|---|---:|---:|
| dev（选择 C=.03） | .5818 | .5504 |
| test | **.5972** | **.5760** |

test 的逐 family recall 为：claude `.8333`、deepseek `.3611`、gemini `.8889`、llama `.5556`、openai `.5278`、qwen `.4167`。task-cluster bootstrap 的 BA 均值为 `.5974`，95% CI 为 `[.5301, .6698]`。

## 与已有同切分基线的关系

同一 AuthorBench CPU baseline 的 `char_center_std` family macro-F1 为 `.6672`、BA `.6435`；metadata-only macro-F1 为 `.2778`、BA `.3179`，raw char hashing macro-F1 为 `.0909`、BA `.1667`。因此本轮 CodeT5-small `.5760/.5972` 没有超过已经验证的任务中心化字符控制。

这不是 CodeT5-small 无效的普遍结论：模型规模、表示层、输入截断和读出都不同；它证明的是在当前 task-heldout 任务上，冻结上下文 encoder 不能自动替代同题参照/中心化。该负对照支持保留“先分离任务效应，再优化来源流形”的主线。

## 限制

1. 这是 task-heldout K-way family attribution；它不检验每个 family 的 unseen generator。
2. 数据缺少可直接使用的共同 `base_model/model_role` 因果控制。
3. 不能把结果与 Droid generator-heldout BA、H2 pair verification BA 或 CoDET-M4 source-model BA 直接排序。
