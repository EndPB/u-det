# AutoDL H2 pair 首轮实验执行指导（2026-10-04）

## 目标

当前交接包已经完成传输和完整性核验。首轮只回答一个问题：在已有的五个代码视图中，简单的词法相似度是否已经能够区分：

- 正例：同一任务、同一家族、不同 generator；
- 负例：同一任务、不同家族的 hard negative。

这一步是数据和任务 sanity check，不是论文主结果，也不允许据此宣称 SOTA。

## 服务器端位置

外层包：`/root/autodl-tmp/u-det/d-det/data/autodl_h2_bundle_3715711_upload.zip`

已核验 SHA-256：`85e04cb5b9b8c3b438c2258109e6717fbc14374300f90794ca36eb67be71c7f1`

推荐只解压外层包一次，并优先解压 H2 pair 子包：

```bash
cd /root/autodl-tmp/u-det/d-det
mkdir -p data/h2_pair_benchmark_v1
unzip -p data/autodl_h2_bundle_3715711_upload.zip packages/h2_pair_benchmark_v1.zip > /tmp/h2_pair_benchmark_v1.zip
unzip -q /tmp/h2_pair_benchmark_v1.zip -d data/h2_pair_benchmark_v1
python scripts/audit_h2_pair_benchmark_v1.py
```

如果外层包已经解压到 `/root/autodl-tmp/u-det/h2_bundle_3715711/`，则直接使用其中的 `packages/h2_pair_benchmark_v1.zip` 和 `scripts/`，不要再复制一份 23 包集合。

## 环境约束

本轮只使用两个 CPU 线程，避免同时生成多个大矩阵：

```bash
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false
```

不下载 35GB 原始数据，不重新生成代码，不启动旧的 CodeT5 全量训练。首轮 TF-IDF 诊断不需要 GPU；如果后续切换到编码器训练，再显式限制为一张卡并使用小 batch。

## 首轮基线

脚本：`scripts/h2_pair_similarity_baseline.py`

它按 `source` 隔离训练，向量器只在 train split 的左右端点上拟合，计算五个 view 的字符 TF-IDF cosine similarity，并在 dev 上选择阈值。test 只报告一次。test 的 bootstrap 按 `task_id` 聚类，避免把同一任务的多条 pair 当成独立样本。

执行：

```bash
cd /root/autodl-tmp/u-det/d-det
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
python scripts/h2_pair_similarity_baseline.py \
  > artifacts/h2_pair_round1/similarity_baseline.stdout.json
```

输出：`artifacts/h2_pair_round1/similarity_baseline.json`。

必须保存以下字段：每个 source/view 的 dev 阈值、test balanced accuracy、F1、ROC-AUC、average precision、正负平均相似度、按 task 聚类的 bootstrap 95% 区间，以及 `family_pair_balanced_test` 诊断结果。

## 读取结果的规则

1. 先看 `authorbench_dcan` 和 `llm_codegen_v2`，不能合并 family 标签空间。
2. 先看 `raw` 与 `all`，再看 `ids_only`、`strings_only`、`comments_only`。若去掉某个视图后性能大幅下降，只能说明该视图含有可读捷径，不能直接说明存在纯家族空间。
3. 若 test AUC 接近 0.5，先停止模型训练并检查 pair 构造、split、端点哈希和任务聚类。
4. 若 raw 很高而 ids_only/strings_only 也很高，优先报告词法/命名捷径；不要把它写成后训练差异的因果证据。
5. `family_pair_balanced_test` 只是严格平衡诊断子集，LLM-CodeGen v2 测试集只覆盖 9 个 family pair，不能替代全量 test。
6. 任何模型选择只能用 train/dev；test 结果只能在最终冻结协议后读取一次。

## 下一步分支

- 若五视图都接近随机：先审计数据构型，不进入重模型。
- 若 raw/all 明显高、去标识符后下降：先做快捷方式控制，再考虑表示学习。
- 若去快捷方式视图仍稳定高：再在 `triplet_index.jsonl` 和 `task_multi_negative_index.jsonl` 上跑小型对比学习；训练仍按 source 隔离，评估按 task 和 generator fold 分层。
- 若只有一个 source 或一个 split 有增益：只能作为诊断，不能扩展成统一 H2 结论。

## 回传文件

只回传以下小文件，不回传整个 35GB 数据集或中间 checkpoint：

- `artifacts/h2_pair_round1/similarity_baseline.json`
- `artifacts/h2_pair_round1/similarity_baseline.stdout.json`
- `data/h2_pair_benchmark_v1/audit_index.json`
- 运行环境版本和 `git rev-parse HEAD`

