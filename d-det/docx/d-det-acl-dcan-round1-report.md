# d-det · ACL 数据集合 v1 首轮报告：P0 强基线 + DCAN 风格四模型

> 执行规范：`docx/d-det_AutoDL_服务器端AI下一阶段执行指导_2026-10-03.md`（+ `docx/d-det_ACL家族归因数据集合_v1_服务端执行指导_2026-10-03.md`）。
> 脚本：`scripts/p0_tfidf_baseline.py`、`scripts/dcan_four_models.py`；审计脚本 `scripts/audit_server_2026_10_03.py`（commit `7cc6262`）。
> 产物：`artifacts/acl_dcan_round1/`（audit + split_manifest + p0/{aicd_t2,dcan,stacad,droid} + dcan_four_models/ + README.md）。
> 定位：**首轮轻量原型与强基线对照**；不作 SOTA 声明（指导 §SOTA 判定六条未全满足）；与 E35/E36/v2/AuthorBench 报告严格分开。
> 数据前置：审计见 `artifacts/acl_dcan_round1/audit_server_2026-10-03.md`（20 分片、行数=manifest 全对；T2 跨 split 泄露哈希 1,848；STACAD 文件级五折 0 跨折；DCAN 每任务 ≥2 family；LLM-CG 空代码 169/围栏 662；CoDET-M4 缺失 model 13,587）。

## 一句话结论

在 **h2_authorbench_dcan（任务级留出）** 上：P0 字符 n-gram TF-IDF 强基线 **.7639**（macro-F1，5 epoch）；DCAN 风格四模型（冻结 CodeT5-base 均值池化 768d 语义分支 + 31 维结构指纹分支，3 seeds）为 semantic-only **.688±.013**、fingerprint-only **.559±.002**、late-fusion **.713±.006**、full-disentangle **.703±.005**（融合输出；fp 分支 .561±.003 / **full-disentangle 未超普通 late-fusion**（每 seed 混合）。语义与指纹两分支互补（融合 > 两单分支），但 **本轮四组 DCAN 轻量模型均低于同切分 TF-IDF 基线**——只报比较事实，不作 SOTA/总评。其余源：AICD T2 官方闭集（numeric_id）**.2444**；STACAD 文件级五折 **.4331±.0057**；Droid generator-held-out 外部评测 **.1604/.1562**（≈1.1×机会，与 v2 结论一致）。OpenAI 留出 unknown-family AUROC .658–.692。**Droid 未参与任何损失**。

## 1 数据与切分（只用已有数据；不重做审计）

| 来源 | 行/对 | 切分（本轮用法） | 标签口径 |
|---|---|---|---|
| AICD T2 | 1,111,199（train 502,149 / val 101,176 / test 507,874） | 官方 train→val 选择→test 终评 | **numeric_id 0–11（映射 pending：HF 卡空、GitHub 404；不命名家族）** |
| h2_authorbench_dcan | 9,498（train 6,557 / dev 1,484 / test 1,457；1,900/407/408 任务） | **任务级留出**（task_id 不跨 split） | family 6 类；生成器 8 个 |
| STACAD v2 | 144,958 对 / 22,053 文件 | **文件级五折**（folds.npy 与 corpus_v2.py 写入顺序逐 pair 对齐；已核验 0 文件跨折） | 模型 1–7（7 类） |
| Droid selected | 146,718（machine 125,718） | **generator-held-out 两折**（fold_plan；训练=train generator 的 train/dev，测试=held-out generator 的 test） | 7 个机器家族（human 不用） |
| CoDET-M4 / LLM-CG | 500,552 / 1,515 | 本轮未进入实验（资源与优先级） | — |

## 2 P0 强基线（第三步）

统一管线：流式读取 → 训练集抽样拟合 `char_wb(2,4)` TF-IDF（min_df=5，sublinear_tf，≤300k 特征；每篇代码截断 6,000 字符）→ 分块 `transform` + `SGDClassifier(log_loss)` partial_fit（5 epochs，类逆频率样本权重；dev macro-F1 逐 epoch 记录）→ 测试评估（macro-F1 / balanced acc / 逐类召回 / 混淆矩阵 / ECE(top-1,15 bins) / 分组指标）。

| 来源 | 训练量 | macro-F1 | balAcc | ECE | 备注 |
|---|---|---:|---:|---:|---|
| AICD T2（闭集） | 502,149×5ep（4027s） | **.2444** | .2841 | .0634 | dev 轨迹 .183→.228→.238→**.260**→.254；样本极不均衡（类 0=442,096/类 5=1,968）；逐类召回 .12–.77（0=.77、10=.43、11=.42 高，1–4≈.12–.30） |
| h2_authorbench_dcan（任务留出） | 6,557×5ep（79s） | **.7639** | .7755 | .0790 | dev .67→**.797**；逐 generator 召回 .47–.92（gemini .92 / claude .86 / gpt-4.1 .86 高；qwen .51 / deepseek .50 低） |
| STACAD v2（文件级五折） | ~116k×5ep/折（6750s） | **.4331±.0057** | ~.438 | ~.033 | 五折 .4258/.4403/.4358/.4271/.4365；fold0 逐类召回 .26–.66（模型 1=.66 高、模型 6=.26 低）；逐语言 acc .38–.47 |
| Droid（generator-held-out） | 52,251 / 48,348×5ep（500s） | **.1604 / .1562** | .1758 / .1710 | .303 / .312 | 机会 1/7≈.1429；≈1.10–1.12×机会；逐族召回 meta-llama .40/fold0、qwen .43/fold1、microsoft .29–.31，codellama/deepseek ≈.07–.08 |

要点：Droid 外部评测再次显示 **generator-held-out 家族归因贴地**（与 v2/E36 独立方法同向）；AICD 虽是"强基线"，但其 12 类闭集与任务分离仅是数字标签（**不得写家族胜负**，映射未核验）；STACAD 文件级结果说明"文件留出下的模型归因"有明显高于机会但远未饱和的信号（.43 vs 1/7≈.14）。

## 3 DCAN 风格四模型（第四步，h2_authorbench_dcan）

设计（轻量原型，全部遵守资源约束）：
- **语义分支**：冻结 `checkpoints/codet5-base`（未微调）attention-mask 均值池化 → 768d（bf16 推理、float32 缓存 29MB）；可训练 MLP 768→256→128（BN/ReLU/dropout .2）。
- **指纹分支**：31 维结构统计（行数/长度/缩进/空行/注释/花括号/分号/字符分布/关键字频率等；train 集 z-score）。
- **语义正对**：仅同 `task_id` 内不同 family 的样本（batch 按任务聚合 48 任务×≤8 样本；**仅 train 任务**）；SupCon(τ=.1) 权重 0.3。
- 四个模型：M1 semantic-only=CE(family)+0.3·SupCon；M2 fingerprint-only=CE(family)；M3 late-fusion=M1/M2 分支联合训练 + 输出层 LR 融合（train 拟合）；M4 full-disentangle=M2 损失 + 0.3·SupCon(z_s) + 0.3·family-GRL@z_s + 0.2·(长度十分位+提示 KMeans(64) 簇) GRL@z_f + 0.05·交叉协方差；主输出 fingerprint head、次输出 LR([z_s,z_f]) 融合。
- 训练：AdamW 1e-3/wd 1e-4，≤40 epochs（dev macro-F1 选最优，patience 10），seeds 0/1/2；单次运行 250s（含编码缓存复用）。

结果（test 任务；3 seeds mean±std）：

| 模型输出 | macro-F1 | 单 seed |
|---|---|---|
| M1 semantic-only | .688 ± .013 | .675/.684/.706 |
| M2 fingerprint-only | .559 ± .002 | .557/.561/.557 |
| M3 late-fusion（融合） | **.713 ± .006** | .706/.714/.720 |
| M4 full-disentangle（融合） | .703 ± .005 | .709/.697/.703 |
| M4 full-disentangle（fp 分支） | .561 ± .003 | .562/.557/.564 |

- **同 backbone late-fusion 对照（指导第六步问题 1）**：full-disentangle 未优于 late-fusion（.703 vs .713；逐 seed 混合 2/3 反例），对抗/正交项在本设置下**无一致增益**。
- **分支互补（问题 2）**：语义（.688）与指纹（.559）互补，融合提升到 .713；两分支相关性低（见 config 中交叉协方差设计）。
- 校准：fp 分支 ECE 低（.038–.045）但欠准；融合 ECE 高（.219–.230）——融合的 LR 概率未校准。
- **unknown-family AUROC（OpenAI 留出，n=506）**：sem .692 / fp .663 / late-fusion .674 / full-disentangle .658——弱信号，仅作辅助记录；不得写成跨 generator H2。
- 边界：C-only、单一数据集、seeds=3、无双划分复现；DCAN 语义分支持续优于指纹分支的机制（任务监督为主）未做消融细化（下一轮）。

## 4 与指导第六步三个问题的对照

1. **语义/来源分离是否优于同 backbone late-fusion？** —— 否（.703±.005 vs .713±.006，无一致增益）；且两者都低于同切分的 P0 TF-IDF 强基线（.7639）。
2. **提升来自哪个数据源？** —— 本轮主比较只在 `h2_authorbench_dcan`（任务留出）；跨源泛化未评估（Droid 只做外部评测）；AICD/STACAD 只报告各自基线。
3. **在 task/file/generator held-out 是否仍成立？** —— task：融合路径可行但未超强基线；file（STACAD）：基线 .43（明显高于机会）；generator（Droid 外部）：**.16 贴地**——不支持"当前轻量表示在 unseen generator 上有稳定家族归属能力"。

## 5 边界与诚实性声明

- **SOTA 判定六条未全满足**：仅完成同数据/同切分/单 backbone 对照的一部分；无 3 源交叉、无捷径剔除实验、无 CI 跨零检验、无映射核验——一律不写"优于基线/SOTA"，只写"低于/高于本轮强基线"。
- AICD 只用 numeric_id，不命名家族；T1/T2/T3 不合并；Droid 不承担同题语义对齐。
- P0 为固定协议（5 epochs、固定词表抽样、单 SGD 配置），未调参；AICD 全量 single-pass 词表抽样（100k）已有明文记录。
- DCAN 为**首轮原型**：冻结编码器均值池化 + 小型 MLP；对抗/正交权重未做敏感性消融（指导允许权重小、未叠加全部损失；本轮 full-disentangle 的负结果如实报告）。
- 复现：P0 冒烟→全量；DCAN 冒烟→正式；缓存复用；无 test 选择；除开发中的 bug 修复（Droid split 字段、张量 detach）外无事后算法变更。
- 资源：磁盘 2.66GB 可用下完成（无新增大副本；emb 缓存 29MB 在 runs/）；全程 OMP=MKL=2。

## 6 复现命令与交付清单（指导 §6）

```bash
cd /root/autodl-tmp/u-det/d-det
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
P=/root/miniconda3/envs/udet/bin/python
# P0（冒烟版加 --smoke；DCAN 用 --smoke 先验小型路径）
$P scripts/p0_tfidf_baseline.py --source dcan                       # 79s
$P scripts/p0_tfidf_baseline.py --source aicd_t2 --epochs 5         # 4027s
$P scripts/p0_tfidf_baseline.py --source stacad --epochs 5          # 6750s
$P scripts/p0_tfidf_baseline.py --source droid --epochs 5           # 500s
$P scripts/dcan_four_models.py --seeds 3 --epochs 40                # 250s（编码 80s 首跑）
```

交付对照：①脚本 `scripts/p0_tfidf_baseline.py`、`scripts/dcan_four_models.py`（相对项目根路径）；②`artifacts/acl_dcan_round1/{p0,dcan_four_models}/`（config/metrics/predictions 压缩版 + split manifest + README）；③本报告；④保留：正式 log（队列日志要点已入报告；solver 级日志为 stdout 记录）、最佳模型参数为各 seed 的 state（未落盘模型文件——按小数据原型约定，种子+固定协议可复跑，见命令）；⑤GPU=3080 Ti（DCAN 编码/训练；P0 纯 CPU）；⑥中心化/转导口径=本实验无转导表示；⑦OpenAI 仅辅助折；⑧本轮回答见 §4。
