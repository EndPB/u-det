# acl_dcan_round1 —— ACL 数据集合 v1 首轮产物（2026-10-03）

## 布局

- `audit_server_2026-10-03.{json,md}`、`split_manifest_2026-10-03.json` —— 只读审计与切分清单（commit `7cc6262`）。
- `p0/` —— P0 强基线（char n-gram TF-IDF + SGD log_loss；脚本 `scripts/p0_tfidf_baseline.py`）：
  - `p0/aicd_t2/`：AICD T2 官方闭集（**numeric_id，未核验映射不命名家族**）；train 502,149×5ep。
  - `p0/dcan/`：任务留出；`p0/stacad/`：文件级五折（folds.npy）；`p0/droid/`：generator-held-out 两折（外部评测）。
- `dcan_four_models/` —— DCAN 四模型首轮（脚本 `scripts/dcan_four_models.py`；3 seeds + OpenAI 留出辅助）：`config.json`、`metrics.json`、`predictions.npz`。

## 环境与约束

- CPU 预处理 OMP=MKL=2（全流程串行队列）；GPU RTX 3080 Ti（仅 DCAN 编码/训练使用；bf16 推理）。
- 无大缓存：语义特征缓存 29 MB 在 `runs/acl_dcan_round1/emb_dcan_ct5.npz`（不入库）；预测文件为 float16 压缩版。
- Droid 未参与任何损失（仅 P0 外部评测）；语义正对仅同 task 不同 family；AICD 只用数字标签。

## 关键数字（详见 `docx/d-det-acl-dcan-round1-report.md`）

| 项 | 结果 |
|---|---|
| AICD T2 闭集（12 类 numeric_id） | macro-F1 .2444（balAcc .2841，ECE .0634） |
| DCAN 任务留出 TF-IDF | .7639 |
| DCAN late-fusion（3 seeds） | .713 ± .006 |
| DCAN semantic-only / fingerprint-only | .688 ± .013 / .559 ± .002 |
| DCAN full-disentangle（融合 / fp 分支） | .703 ± .005 / .561 ± .003 |
| STACAD 文件级五折 | .4331 ± .0057 |
| Droid generator-held-out（fold_0 / fold_1） | .1604 / .1562 |
| OpenAI 留出 unknown AUROC | .658–.692 |

## 复现（命令）

```bash
cd /root/autodl-tmp/u-det/d-det
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
P=/root/miniconda3/envs/udet/bin/python
$P scripts/p0_tfidf_baseline.py --source dcan            # 79s
$P scripts/p0_tfidf_baseline.py --source aicd_t2 --epochs 5   # 4027s
$P scripts/p0_tfidf_baseline.py --source stacad --epochs 5    # 6750s
$P scripts/p0_tfidf_baseline.py --source droid --epochs 5     # 500s
$P scripts/dcan_four_models.py --seeds 3 --epochs 40     # 250s（特征复用缓存）
```
