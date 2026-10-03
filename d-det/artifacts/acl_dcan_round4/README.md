# acl_dcan_round4 · bdd569 后续执行轮（单变量捷径诊断 + 预算扩展 + 外部审计）

对应报告：`docx/d-det-acl-dcan-round4-report.md`；执行规范：`docx/d-det_AutoDL_bdd569后续执行指导_2026-10-03.md`。
基线协议：bdd569（round3 修正版）；**外部结果独立标注，不与 408-task 主表混合**。

## 内容导航

| 目录 | 说明 |
|---|---|
| `shortcuts_univar/` | **A 单变量捷径诊断**：6 变体（raw/ids/strings/comments/ws/all）×3 seeds TF-IDF（train 拟合、5ep、best-dev+ep5 双口径）；`metrics.json`（dev 曲线/变换示例）、`predictions.npz`（42 键：身份字段+每变体每 seed 概率）、`stats.json`（raw−变体配对 bootstrap B=2000）、config/env/split/hashes、logs |
| `lora_extend/` | **B LoRA 预算扩展**：仅 epochs 12→24、patience 3→5；3 seeds 全 dev 曲线；`metrics.json`、`predictions.npz`、`stats.json`（vs round2 与 vs TF-IDF 配对 CI）、`states/lora_ext_s{0,1,2}.pt`、config/env/split/hashes、logs |
| `external/` | **C 外部泛化审计**（诊断）：`stacad_fold0/`（file-held-out，train 30k 子集）/`droid_fold0/`（generator-held-out）；三族 ×3 seeds + 聚类 bootstrap；各含 config/env/split/hashes/logs |

## 关键数字（速查）

- **A**：raw .7619±.0029；ids −11.8pt\*；comments −6.3pt\*；strings −3.0pt\*；ws ±0.0（char_wb 构造性）；all −31.2pt\*（超可加）。
- **B**：extended .7324/.7462/.7597 → 均值 **.7461**（vs round2 .7286，+1.7pt）；s1/s2 至 24ep 上限仍创新高（欠拟合未完全排除）；s2 与 TF-IDF 统计不可区分（−0.4pt 跨零），均值仍低于 .7639；**不作方法主张**。
- **C**：STACAD fold0：TF-IDF **.3726** > LoRA .3506（+2.2pt\*）> head .2256；Droid fold0：**.1642** / .1322 / .1267（CI 跨零）；无灾难性崩溃、无新方法主张（详见报告 §3）。

## 重跑命令（防覆盖已内置）

```bash
P=/root/miniconda3/envs/udet/bin/python
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 $P scripts/dcan_round4_shortcuts_univar.py && $P scripts/dcan_round4_stats.py
$P scripts/dcan_round4_lora_extend.py --seeds 3 --epochs 24 --patience 5 && $P scripts/dcan_round4_lora_stats.py
$P scripts/dcan_round4_external.py --sources stacad,droid --seeds 3
```

## 边界

test 不参与任何选择；无新下载；OMP=MKL=2；B 为条件执行（A 证据支持可解释捷径假设）；C 结果为诊断性证据；不写 SOTA。
