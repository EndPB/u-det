# AICD Task 2 numeric-label baseline

日期：2026-10-07
状态：完成；这是可复现的数字标签诊断，标签语义仍未映射。

## 数据边界

本地固定包来自 AICD-Bench 的 T2 配置，revision 为 `b7dc6d82257de6e0146d1e7b0831abd7088633af`。它从公开数据中确定性抽取 train/validation/test，共 48,877 行、12 个整数标签；每个 test 类最多 1,000 行。抽取规则、行数、SHA256 和未解析标签表见同目录 `summary.json`、`audit_index.json`、`label_map_pending.json` 和 `SHA256SUMS.txt`。

官方论文将 AICD-Bench Task 2 定义为模型家族归因，并说明其完整任务包含 12 类（含 human）；公开数据页面当前只暴露 `code` 与整数 `label`。因此本地结果只报告 numeric-ID closed-set 诊断，不能把 0–11 事后命名为模型家族。

- 官方任务说明：[`AICD-Bench 论文`](https://arxiv.org/abs/2602.02079)
- 数据快照：[`AICD-Bench 数据集`](https://huggingface.co/datasets/AICD-bench/AICD-Bench)

## 固定协议

- 文本特征：字符 TF-IDF，3–5 grams，`min_df=2`，最多 80,000 特征，sublinear TF；词表只在 train 拟合。
- 读出：`LogisticRegression(class_weight="balanced", C=1.0, max_iter=1000)`，固定参数。
- 切分：使用本地派生包的 train/validation/test；validation 只作描述，test 只读一次。
- 指标：balanced accuracy、macro-F1 和每个数字标签的 recall/precision；机会 BA 为 `1/12=.0833`。
- 重跑入口：[`aicd_t2_numeric_baseline.py`](../../scripts/aicd_t2_numeric_baseline.py)；原始结果：[`metrics.json`](metrics.json)。

## 结果

| split | n | BA | macro-F1 |
|---|---:|---:|---:|
| train | 33,344 | .6914 | .6780 |
| validation | 3,600 | .4111 | .3948 |
| test | 11,933 | **.2920** | **.2726** |

test 中数字标签 0 和 11 的 recall 分别为 `.7170` 与 `.5970`，标签 7 仅 `.0600`；这说明数字标签之间存在明显难度差异，不能用一个平均值掩盖类别失衡或标签语义未解析的问题。

## 结论边界

1. 字符表面在这个紧凑的 12 类数字空间中有可复现信号，但 test BA `.2920` 远低于训练 BA，提示强泛化鸿沟或切分/标签来源差异。
2. 该结果不能直接与 AuthorBench、Droid 或 CoDET-M4 的 family/source BA 排序，因为标签空间、数据抽取和任务定义不同。
3. 在找到官方 numeric-to-family preprocessing/evaluation mapping 以前，任何“某 family 最难/最好”都不得写入主结论。
4. 这是控制和数据审计结果，不是 AICD 官方完整 Task 2 复现实验；后续若要进 ACL 主表，必须先冻结官方映射、完整 split、重复种子和任务级/来源级统计。\n