# d-det ACL SOTA 主线：P0 强基线复现报告（2026-10-04）

> 执行者：服务器 AI；指导文档：`docx/d-det_AutoDL_1f3d547_ACL家族归因SOTA执行指导_2026-10-04.md`
> （已于本轮入库，提交 `0d10823`；锚点 `1f3d547` 是当时 HEAD `ee565d5` 的祖先，差异仅为上一轮收尾记录与指导文档本身，已核验）

## 0. 摘要

本轮按 P0 要求完成了**强基线复现**，分两块：

1. **STACAD 官方公开协议复现**（145k 改写对、7 类、5 折文件级 GroupKFold、纯 CPU）：
   - `classic`：LogReg .4874（官方 .4874，Δ0.0000）/ RF .5240（官方 .5241，Δ−0.0002）；
   - `stack`（meta 层用随包发布的 OOF 输出重跑）：vote .5988 / stack_trees .6004 /
     +TF-IDF .6701 / +CodeBERT .7329 / 完整 canonical .7330——**逐项与官方发布数字一致（|Δ|≤0.0002）**；
   - `learners`：XGBoost **.5988**（官方 .5986）/ LightGBM **.5959**（官方 .5959，逐折 rounds 逐位一致）/
     MLP .5531（官方 .5504）；CatBoost 未安装，引用官方 .5893。
2. **服务器三赛道 P0 基线表 v1**（authorbench 主赛道 + STACAD/Droid 外部赛道）：
   字符/词法 stylometry、CodeT5 监督（复排+统一协议）、结构/stylometry 统计（含 STACAD 官方 105 特征）、
   dev-only 轻量 late fusion，全部带聚类 bootstrap CI 与 ECE。
   - **authorbench 主赛道（task 留出，n=1457）**：新融合基线 **macro-F1 .8388** [.8195, .8620]；
     单模型最强为 word-TFIDF **.8200**；char-TFIDF .7692；LoRA 复排 seed 平均 .7461（逐 seed .7324/.7462/.7597）；
   - **stacad_fold0（file 留出，n=28992）**：**官方 105 特征（变换+stylometry）+ LGBM = .5658**，
     压倒词法（char .3809）与神经（lora .3704 / sem .3033 / head .2290）；融合 **.6089**；
   - **droid_fold0（generator 留出，n=6142）**：最难赛道，全部视图 ≤.165，**融合无增益**（.1540 < 词法 .1645）；
     dev（≈.35）与 test（≈.16）差距巨大 ⇒ 未见生成器归因是当前最大缺口（P2/P3 主攻方向）。

纪律：test 仅评估一次；dev 只用于快照/融合权重；全部产物写入新目录 `artifacts/acl_sota_p0/`，未覆盖任何旧产物。
关键结论：**「成对变换/结构视图」是被验证的强信号（STACAD .5658），「词法」在主赛道仍是最强单视图（.8200），
而两者在「未见 generator」场景都崩溃（droid ≤.17）——P1/P2 的努力方向明确。**

## 1. 锚点、环境与执行记录

- HEAD 校验：`ee565d5`（含 `1f3d547`，追加的收尾提交），本轮新增提交见文末；
- 环境：`/root/miniconda3/envs/udet`（Python 3.12，torch 2.9.1，transformers 5.17.0，sklearn 1.9.1，
  lightgbm 4.7.0，**xgboost-cpu 3.4.1（本轮新装；默认 PyPI xgboost 会拉 nvidia-nccl，改装 CPU 轮子）**）；
- 纯 CPU 阶段将 `OMP_NUM_THREADS=8`（与官方 8 核复现条件一致；神经实验的 OMP=2 规则不适用于此，已记录于 env.json）；
- STACAD 官方代码与结果：`data/stacad_v2/extracted/STACAD-v2/`（含 `scripts/v2_pipeline.py`、
  `cache/results/journal_v2/*.json`、`cache/corpus_v2/features_v2.npz`）；
- 关键对齐校验：features_v2.npz 与 round4-C 行空间**逐行 0/144958 不一致**（key=lang/file/fold/label），
  fold0=28992 与 round4-C test 一致。

## 2. STACAD 官方协议复现（公开基线第一块）

口径（与官方 `v2_pipeline.py` 完全一致）：5 折确定文件级 GroupKFold；SEED=42；早停=训练划分文件级 10%；
指标=逐折 macro-F1 mean±std（+ECE/NLL/Brier）；XGBoost 用官方 tuned protocol.json 参数。

| 阶段 | 模型/候选 | 本轮复现 | 官方数字 | Δ |
|---|---|---|---|---|
| classic | logreg | 0.4874 | 0.4874 | +0.0000 |
| classic | rf | 0.5240 | 0.5241 | −0.0002 |
| stack(meta) | vote_0.50/0.30/0.20 | 0.5988 | 0.5988 | ±0 |
| stack(meta) | vote_equal | 0.5988 | 0.5988 | ±0 |
| stack(meta) | stack_trees | 0.6004 | 0.6004 | ±0 |
| stack(meta) | +tfidf | 0.6701 | 0.6701 | ±0 |
| stack(meta) | +tfidf+lang | 0.6702 | 0.6701 | +0.0001 |
| stack(meta) | +codebert | 0.7048 | 0.7048 | ±0 |
| stack(meta) | +tfidf+codebert | 0.7329 | 0.7328 | +0.0001 |
| stack(meta) | +tfidf+lang+codebert（canonical） | 0.7330 | 0.7330 | ±0 |
| **learners** | **XGBoost（tuned）** | **0.5988** | 0.5986 | +0.0002 |
| **learners** | **LightGBM** | **0.5959** | 0.5959 | ±0 |
| learners | MLP(256,128) | 0.5531 | 0.5504 | +0.0026 |
| learners | CatBoost | （未装机，引用） | 0.5893 | n/a |

XGBoost 逐折 rounds 复现 `[2438, 2360, 2592, 2430, 2250]`（官方 `[2555, 2123, 2578, 2527, 2513]`，xgboost 版本差异）；
**LightGBM 逐折 rounds `[821, 808, 815, 828, 746]` 与官方逐位精确一致**。

self-consistent stack 变体（用**本机重训的 xgb/lgb OOF** 替换官方 OOF、cb 仍用官方）；
与 meta-only 结果的差异全部 ≤0.0005：vote .5983–.5988 / stack_trees .6001 /
+tfidf .6699–.6704 / +codebert .7046 / canonical（+tfidf_codebert）**.7324** /
（+tfidf+lang_codebert）**.7327**——说明本机复现的基学习器与官方产物几乎等价，**端到端自洽**。
全部结果见 `metrics_repro.json` 与 `compare_vs_official.json`（10 项对照 |Δ|≤0.0002，
仅 MLP +0.0026——sklearn 版本差异）。

## 3. 三赛道 P0 基线表

### 3.1 主赛道 authorbench_dcan（6 families，task 留出，n_test=1457，cluster=task_id）

| 模型 | macro-F1 | CI95 | balanced acc | ECE | 说明 |
|---|---|---|---|---|---|
| **fusion_lr（轻量集成）** | **0.8388** | [0.8195, 0.8620] | 0.8319 | 0.0388 | dev 拟合 LR stack（sem_lr, style_lgb, style_lr, tfidf_char, tfidf_word） |
| mean_ensemble | 0.8282 | [0.8046, 0.8483] | 0.8159 | 0.1319 | 等权概率均值 |
| tfidf_word（词法） | 0.8200 | [0.7980, 0.8418] | 0.8125 | 0.0272 | 标识符/词元 1-2gram，SGD 5ep best-dev，3 seeds |
| codet5_lora_ext（复排） | 0.7808 | [0.7590, 0.8033] | 0.7859 | 0.0366 | round4 只重算指标；seed 平均（逐 seed .7324/.7462/.7597，均值 .7461） |
| tfidf_char（字符） | 0.7692 | [0.7434, 0.7908] | 0.7767 | 0.0539 | char_wb(2,4)，同协议 |
| style_lgb（结构统计） | 0.7227 | [0.7007, 0.7468] | 0.7163 | 0.1059 | regex stylometry 97d + LightGBM(800) |
| sem_lr（CodeT5 统一协议） | 0.6528 | [0.6256, 0.6775] | 0.6535 | 0.1792 | 冻结均值池化 768d + 标准化 + LR |
| style_lr | 0.5586 | [0.5343, 0.5858] | 0.5673 | 0.0260 | 同上特征 + LR |
| codet5_head_only（复排） | 0.4216 | [0.3963, 0.4442] | 0.4193 | 0.1105 | round3 audit 只重算指标；seed 平均（逐 seed .4190/.4285/.4014，均值 .4163±.0112） |

要点：
- **词法（word-TFIDF）显著强于字符 TF-IDF**（.8200 vs .7692），且单独已超过 LoRA 微调复排（.7808 seed 平均）；
- 结构统计（97 维 regex）单独 .7227，为「非神经 stylometry」提供了可用的独立视图；
- **轻量 late fusion 把主赛道推到 .8388**（+1.9 分 vs 最强单模型），CI 不重叠于 char/head/lora 行；
  注意该行是 dev 拟合的融合权重（dev=拟合集，未用 test），test 只评估一次。

### 3.2 外部赛道 stacad_fold0（7 classes，file 留出，n_test=28992，cluster=file）

协议 = round4-C（file-level subtrain 30k / dev 3k / test=fold0；官方 105 特征与行空间已逐行对齐）。

| 模型 | macro-F1 | CI95 | balanced acc | ECE | 说明 |
|---|---|---|---|---|---|
| **fusion_lr（轻量集成）** | **0.6089** | [0.6042, 0.6143] | 0.6113 | 0.0259 | dev 拟合 LR stack（feats105_lr, feats105_lgb, sem_lr, tfidf_char, tfidf_word） |
| mean_ensemble | 0.5797 | [0.5740, 0.5857] | 0.5875 | 0.1892 | 等权概率均值 |
| **feats105_lgb（结构+变换统计）** | **0.5658** | [0.5604, 0.5711] | 0.5690 | 0.1252 | **官方 105 特征**（stylometric+transformation）+ LightGBM(800) |
| feats105_lr | 0.4869 | [0.4822, 0.4924] | 0.4926 | 0.0259 | 官方 105 特征 + 标准化 + LR |
| tfidf_char_round4c（复排） | 0.3809 | [0.3757, 0.3859] | 0.3829 | 0.1101 | round4-C 保存预测只重算指标 |
| tfidf_char（本套件重训） | 0.3729 | [0.3676, 0.3775] | 0.3787 | 0.1168 | 与 round4-C .3726 一致（协议复现成功） |
| codet5_lora（复排） | 0.3704 | [0.3657, 0.3765] | 0.3814 | 0.0189 | round4-C 只重算指标 |
| tfidf_word | 0.3657 | [0.3603, 0.3708] | 0.3778 | 0.1324 | 同协议 |
| sem_lr | 0.3033 | [0.2985, 0.3076] | 0.3088 | 0.0619 | 冻结 CodeT5 768d + LR |
| codet5_head_only（复排） | 0.2290 | [0.2249, 0.2329] | 0.2473 | 0.0449 | round4-C 只重算指标 |

要点：**官方 105 特征（成对变换+stylometry）在此赛道压倒性领先**（LGBM .5658 vs 次强词法 .3809）；
融合进一步到 .6089。这与指导的判断一致：**只依赖一个均值池化向量不够，变换/结构视图是关键**。
（注意：官方全量 5 折 OOF 的数更高——canonical_cpu .6701 / 含 CodeBERT .7330——训练规模 116k vs 30k 的差异，两者分表。）

### 3.3 外部赛道 droid_fold0（7 families，generator 留出，n_test=6142，cluster=generator）

| 模型 | macro-F1 | CI95 | balanced acc | ECE | 说明 |
|---|---|---|---|---|---|
| tfidf_char | 0.1645 | [0.1037, 0.1755] | 0.1760 | 0.2756 | 本套件重训（best-dev 逐 seed dev F1 见 metrics.json，≈.35） |
| tfidf_char_round4c（复排） | 0.1634 | [0.1008, 0.1757] | 0.1765 | 0.2801 | round4-C 只重算指标（.1642 复现 ✓） |
| tfidf_word | 0.1628 | [0.1035, 0.1748] | 0.1712 | 0.2833 | 同协议 |
| fusion_lr（轻量集成） | 0.1540 | [0.0867, 0.1749] | 0.1657 | 0.2615 | dev 拟合 LR stack |
| mean_ensemble | 0.1517 | [0.0865, 0.1708] | 0.1616 | 0.1443 | 等权均值 |
| style_lr | 0.1442 | [0.1040, 0.1739] | 0.1541 | 0.0949 | 97d regex + LR |
| sem_lr | 0.1392 | [0.0858, 0.1523] | 0.1453 | 0.2209 | 冻结 CodeT5 768d + LR |
| style_lgb | 0.1379 | [0.0855, 0.1548] | 0.1511 | 0.2772 | 97d regex + LGBM |
| codet5_head_only（复排） | 0.1349 | [0.0871, 0.1729] | 0.1404 | 0.1067 | round4-C 只重算指标（.1322 复现 ✓） |
| codet5_lora（复排） | 0.1277 | [0.0691, 0.1830] | 0.1357 | 0.3414 | round4-C 只重算指标（.1267 复现 ✓） |

要点：这是**最难赛道**——所有视图都压不过 .17，且**融合不带来增益（.1540 低于词法单模型）**。
CI 很宽是因为 cluster=generator（测试生成器数量少，重采样代价高）。
结论：当前特征族不足以支撑「未见 generator 的家族归因」；这正是 P2/P3 要攻的核心缺口，
需要更细的来源指纹 + 任务条件化配对训练，而不是简单融合。

注：dev F1 明显高于 test（词法 dev≈.35 vs test≈.16），说明生成器留出下存在明显的分布偏移，
不可用 dev 数字外推。

## 4. 纪律与偏差说明

- test 不参与任何选择：TF-IDF/SGD 的 epoch 用 dev 选择；融合权重只用 dev 拟合；复排行仅重算指标；
- 未覆盖旧产物：全部新产物在 `artifacts/acl_sota_p0/`；
- 已知偏差与限制：
  1. `OMP_NUM_THREADS=8`（仅 CPU 阶段；与官方复现条件一致；神经阶段规则不变）；
  2. CatBoost 未安装（引用了官方数字）；learners 的 MLP 行为 sklearn 版本差异可能带来小偏差；
  3. xgboost 3.4.1（CPU wheel）与官方 2.x 的树生长实现差异可能造成 rounds/数值小偏移；
  4. STACAD 赛道使用 round4-C 的 subtrain-30k 协议（与官方 5 折全量 OOF 不同，两者分开报告）。

## 5. 产物清单与复现命令

产物根目录：`artifacts/acl_sota_p0/`

| 路径 | 内容 |
|---|---|
| `stacad_official_repro/config.json / env.json` | 复现参数（protocol.json 合并、SEED、SHA-256）与环境 |
| `stacad_official_repro/metrics_repro.json` | 本机复现：classic/learners/stack 全部行（5 折 OOF mean±std、ECE/NLL/Brier） |
| `stacad_official_repro/compare_vs_official.json` | 与随包发布结果的逐项 Δ |
| `stacad_official_repro/logs/{classic_stack,learners}.log` | 运行日志 |
| `server_tracks/authorbench_dcan|stacad_fold0|droid_fold0/{metrics.json,predictions_all.npz,...}` | 三赛道基线表（macro-F1+CI95+balanced acc+ECE+逐类召回） |
| `server_tracks/logs/run.log` | 三赛道运行日志 |
| `summary_p0_v1.{json,md}` | 聚合汇总表 |
| `README.md` | 目录说明与复现脚本索引 |

脚本（本轮新增，均已入库）：`scripts/acl_sota_p0_stacad_repro.py`、
`scripts/acl_sota_p0_baselines.py`、`scripts/acl_sota_p0_summary.py`。

复现命令（服务器，纯 CPU）：

```bash
OMP_NUM_THREADS=8 python scripts/acl_sota_p0_stacad_repro.py --stages classic stack   # 21.6 min
OMP_NUM_THREADS=8 python scripts/acl_sota_p0_stacad_repro.py --stages learners        # 41.2 min（XGB/LGB/MLP 5 折）
OMP_NUM_THREADS=8 python scripts/acl_sota_p0_stacad_repro.py --stages stack           # + self_consist
OMP_NUM_THREADS=8 python scripts/acl_sota_p0_stacad_repro.py --stages merge           # 合并三阶段 JSON
OMP_NUM_THREADS=8 python scripts/acl_sota_p0_baselines.py --tracks authorbench_dcan stacad_fold0 droid_fold0
python scripts/acl_sota_p0_summary.py
```

关键文件 SHA-256 与字节数：见 `artifacts/acl_sota_p0/verify_p0_2026-10-05.json`（17 个文件条目）。

## 6. 下一步（P1 入口）

- 统一多视图特征接口（`scripts/acl_sota_p0_baselines.py` 已提供视图/模型/融合的接口雏形）；
- 融合已显示方向：词法 + 结构统计 + 语义 +（STACAD）变换特征的 late fusion 是 P1 的起点；
- P1 先做**冻结特征 + 线性/树模型的 late fusion**，明确增益来源（family 信息 vs 语言/长度/任务捷径），
  再考虑 P2 任务条件化训练与 P3 三类预注册测试。
