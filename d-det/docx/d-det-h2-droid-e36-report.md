# d-det · H2 报告 E36：DroidCollection 子集 generator-held-out 候选实验

> 执行规范：`docx/d-det_H2_DroidCollection_DeepSeek执行指导_2026-10-01.md`（交接说明）。
> 收尾规范：`docx/d-det_E36后续指导_最终审计与H2收尾_2026-10-01.md`。
> 产物：`artifacts/h2_droid_e36/{audit,config,manifest,metrics}.json + predictions.npz + solver.log + repro_compare.json`；
> **manifest commit=83fad1c（修正后复现基线；与原 2cc9695 产物逐项一致，见 repro_compare.json）**；特征缓存在 `runs/h2_droid_e36/`（不入库）；脚本 `scripts/h2_droid_e36.py`。

## 一句话结论

> 在 DroidCollection 子集上，family attribution 在 generator-held-out 条件下接近机会水平；加入仅跨 generator 正对的残差约束后，两折增量方向不一致（fold_0 +0.14/+0.06pt；fold_1 −0.28/−0.37pt），pooled BA_F/BA_G 分别下降 0.17/0.20 个百分点。因此，在当前 768 维冻结表示和该数据构型下，没有观察到稳定的 H2 增益。**该结果是数据构型和表示条件下的阴性证据，不等价于对所有 task-aware 数据上的 H2 作普遍否定。**
>
> anchor 守卫通过（fold_0 .714 / fold_1 .857）⇒ 正式判读为"不支持"（非"检验不足"）。本数据最多支持"DroidCollection 上的跨 generator 家族归因候选证据"表述；与 E35 严格分开，不外推。

## 摘要表（主指标只在 held-out test generator 上）

| 模型 | fold_0（test 1606） | fold_1（test 1733） | pooled（3339） |
|---|---|---|---|
| B0 冻结加权主轴 | .1608 / .1578 | .1661 / .1722 | .1605 / .1623 |
| F0 残差 CE | .1571 / .1543 | .1697 / .1760 | .1608 / .1625 |
| F1 F0+0.1·L_cross | .1585 / .1549 | .1669 / .1723 | .1591 / .1605 |
| **Δ(F1−F0)** | **+0.14 / +0.06pt** | **−0.28 / −0.37pt** | **−0.17 / −0.20pt** |
| anchor 比例（有效/总） | .714（3960/5544） | .857（5400/6300） | — |
| 检测 AUROC（次要，test） | .9269 | .9230 | — |
| 随机切分参照（探索性诊断，池=原始 train/dev） | .3005 / .3016 | .2966 / .2951 | — |

机会水平 1/7≈.1429；BA_F/BA_G 单位换算为 pt 时乘 100（如 −0.17pt）。

## 1 阶段 A 审计（audit.json；不加载 encoder）

- 数据校验：`SHA256SUMS.txt` 5 文件全部通过（core/diagnostic/fold_plan/summary/README）。
- 行数与标签：18,000 行 = **MACHINE_GENERATED 14,220 + HUMAN_GENERATED 3,780**；split 计数 train 7,020 / dev 6,381 / test 4,599（machine 5,760/5,121/3,339；human 各 1,260）。
- 家族与 generator：7 family、32 个 machine generator（qwen 7 / meta-llama 6 / microsoft 6 / 01-ai 4 / codellama 3 / deepseek-ai 4 / ibm-granite 2）；语言：Python 8,342、Java 3,312、C++ 2,097、JavaScript 1,805、C# 1,437、Go 627、C 380。
- **重复检查**：`source_row_sha1` machine 与 human 在 train/dev/test 间 **跨 split 重复 = 0**；split 内重复 = 0。
- **折断言（全部通过）**：每 family train/held-out generator 集合非空且**交集为 0**；观测 generator ⊆ 计划集合；训练/验证/测试样本非空；**每个 held-out generator 都有 test 样本**（最少 18，最多 180）。fold_0/fold_1 互补（generator 侧互换）。
- 诊断文件 `diagnostic_hybrid_adversarial.jsonl`：2,000 行（MACHINE_REFINED 1,000 + MACHINE_GENERATED_ADVERSARIAL 1,000），**仅计数**，未参与任何训练/评分（见 §7）。

## 2 特征提取口径（与 E28/E35 相同）

- tokenizer：本地 `checkpoints/codet5-base`，`add_special_tokens=False`；`>max_length(1024)` → 头 768 + 尾 256；`<8` 跳过。
- 编码：`runs/v0.4.1_covreg/last.pt`（冻结）+ `runs/flagship_r1/head.pt`（SetPool）+ `m_raw` 均值池化（bf16 autocast）；输出 768 维。
- 实测：18,000 行全部保留（跳过 0）；token 长度 p50/p90/p99/max = 233/599/1024/1024；编码 150.7s（≈120 行/s，bs=16，RTX 3080 Ti）；**同批序重复编码逐位一致**；跨批组成差异 max|Δ|=1.79e-07（bf16 批核效应，记录不阻塞）。
- 缓存：`runs/h2_droid_e36/features.npz`（52MB）+ `metadata.jsonl`，逐行 `source_row_sha1` 并在训练前与 JSONL 逐位复核通过。**复现运行复用同一缓存（hash 不变，见 `repro_compare.json`）。**

## 3 主实验协议（预注册）

- 折：严格读 `fold_plan.json`，不自行划分。训练＝MACHINE+train+train_generators；验证＝MACHINE+dev+train_generators；测试＝MACHINE+test+heldout_generators；Human 按同 split_source 仅用于检测头。
- B0：加权多分类 LR（C=.1、max_iter=10000、tol=1e-6、w=N/(K·|G_f|·N_g)）逐折拟合后**冻结**（两折均收敛）。
- 残差（阶段 B 修正门控）：r=V·GELU(Uz+b1)+b2；ℓ=ℓ0+γ·W_R·r；W_R=0、η0=.5（γ≈.046）、U Xavier、V std=1e-3、b=0。**初始化审计：F0/F1 训练前 logits 与 B0 逐位一致（max|Δ|=0）**。
- F0=L_F；F1=L_F+0.1·L_cross（τ=.1 只除一次；正对=同家族不同 generator；无正对 anchor 跳过；分母含全批；Human/同 generator/异族不入 L_cross）。F0/F1 共享主轴/初始化/种子/批序/标准化/epoch。
- 训练：批 7 族×18=126（族→generator→样本均匀；重抽保护）；22 步/epoch（fold_0，train 2700）/ 25 步/epoch（fold_1，train 3060）×2 epoch；AdamW 1e-3/wd 1e-4/clip 1。
- 训练健康度：L_cross 6.7→4.8（fold_0）、7.3→4.8（fold_1）；γ 稳定 .044–.046；残差参与度 3.2–4.2%；F1 漂移 ‖U‖≈28、‖V‖≈4.6、‖W_R‖≈0.28–0.31（零初始化生长）。

## 4 结果

**4.1 逐族 pooled 召回（B0 / F0 / F1）**：qwen .2225/.2081/.2015；meta-llama .2102/.2022/.2022；microsoft .2558/**.2792**/.2632；01-ai .0879/.0854/.0930；codellama .1155/.1254/**.1287**；deepseek-ai .1116/.1116/.1116；ibm-granite .1198/.1138/.1138。逐族 Δ(F1−F0) 区间 [−1.6pt, +0.8pt]，无 family 超过 ±2pt。

**4.2 逐 held-out generator recall（pooled，B0/F0/F1；规范要求全列）**

| generator | B0 | F0 | F1 || generator | B0 | F0 | F1 |
|---|---|---|---|---|---|---|---|
| Qwen2.5-72B-Instruct | .221 | .198 | .186 | Yi-Coder-9B | .167 | .148 | .185 |
| Qwen2.5-Coder-1.5B | .217 | .174 | .163 | Yi-Coder-1.5B | .083 | .083 | .083 |
| Qwen2.5-Coder-7B-Instruct | .133 | .133 | .128 | Yi-Coder-9B-Chat | .078 | .078 | .078 |
| Qwen2.5-Coder-32B-Instruct | .068 | .055 | .055 | Yi-Coder-1.5B-Chat | .071 | .071 | .079 |
| Qwen2.5-Coder-1.5B-Instruct | .333 | .328 | .311 | CodeLlama-34b-Instruct-hf | .151 | .166 | .158 |
| Qwen2.5-Coder-7B | .160 | .153 | .153 | CodeLlama-7b-hf | .000 | .000 | .000 |
| Qwen2.5-Codder-14B-Instruct | .618 | .579 | .579 | CodeLlama-70b-Instruct-hf | .096 | .103 | .116 |
| Llama-3.3-70B-Instruct | .189 | .161 | .161 | deepseek-coder-6.7b-base | .089 | .089 | .089 |
| Llama-3.1-8B | .412 | .412 | .400 | deepseek-coder-1.3b-base | .115 | .115 | .115 |
| Llama-3.2-1B | .350 | .350 | .350 | deepseek-coder-6.7b-instruct | .122 | .122 | .122 |
| Llama-3.3-70B-Instruct-Turbo | .198 | .177 | .198 | deepseek-coder-1.3b-instruct | .130 | .130 | .130 |
| Llama-3.2-3B | .264 | .292 | .264 | granite-8b-code-base-4k | .133 | .125 | .125 |
| Llama-3.1-8B-Instruct | .111 | .111 | .117 | granite-8b-code-instruct-4k | .077 | .077 | .077 |
| Phi-3-medium-4k-instruct | .203 | .218 | .218 | | | | |
| Phi-3-small-8k-instruct | .318 | .333 | .333 | | | | |
| phi-2 | .056 | .069 | .069 | | | | |
| Phi-3-mini-4k-instruct | .323 | .333 | .312 | | | | |
| Phi-3.5-mini-instruct | .339 | .367 | .344 | | | | |
| phi-4 | .149 | .216 | .149 | | | | |

跨度 0（CodeLlama-7b）到 0.62（Qwen-Codder-14B）；F1−F0 有正有负（如 Yi-Coder-9B +3.7pt、phi-4 −6.7pt、Qwen-72B −1.2pt），**未构成跨 generator 一致的增益**。

**4.3 语言 / Source / Mode / 长度分桶（次要）**：pooled 语言 BA_F（B0/F0/F1）——Python .165/.165/.165（n=1595）、Java .163/.162/.159、C++ .124/.128/.122、JavaScript .171/.167/.162、C# .175/.173/.172、Go .088/.093/.099、C .293/.293/.293；Generation_Mode/Source/长度分桶逐折明细见 `metrics.json`（fold→{B0,F0,F1} 各桶）。

**4.4 anchor（跨 generator 正对覆盖）**：fold_0 整体 .714（逐族：5 个多 generator 族 1.00；codellama/ibm-granite 0.00＝训练侧单 generator）；fold_1 整体 .857（6 族 1.00；ibm-granite 0.00）。**anchor 守卫阈值 .15 通过 → 检验充分**。

**4.5 随机切分参照（探索性诊断，不入出口，不作为 H2 证据）**：按修正口径（池仅含原始 `split_source=train/dev` 的 machine 行，不触及 test）——fold_0 .3005/.3016、fold_1 .2966/.2951 vs gen 主轴 .1608/.1578、.1661/.1722。**该参照仅用于描述 generator migration loss 方向性，不作为支持/反对 H2 的证据**（原口径曾含原始 test 行，已按指导 §一 修正；变化详见 `repro_compare.json` 的 `random_ref` 字段）。

## 5 判读（预注册出口逐条）

1. 出口 1（初步支持）：**不达**——fold_1 两项为负。
2. 出口 2（方向一致但 <1pt）：**不适用**——并不是方向一致。
3. **出口 3（方向不一致或 pooled 非正）：成立 ⇒ H2 在本数据与 768 维冻结表示下不支持。**
4. 出口 4（anchor 过低 ⇒ 检验不足）：不成立（.714/.857）。
5. 出口 5（只改善单 family/role/generator ⇒ 仅报异质性）：记录——F1−F0 逐族/逐 generator 有正有负、无一致结构；不写成任何总体支持。
- 备注（不改变出口）：本数据上 family 主轴（B0）本身在 gen-heldout 下仅 ≈.16（1.12–1.16×机会）；"H2 不支持"与"该表示在当前预算下对该数据来源关系可读性低"两种读法都与数据一致，报告不额外宣称。**检测 AUROC（.9269/.9230）仅作次要报告，不作为 H2 支持证据。**
- 单种子仅作候选证据；本结果只适用于 DroidCollection 子集与当前冻结表示。

## 6 与 E35 的关系（必须分开陈述）

- E35（七家族 SemEval-Task-B 系数据、强主轴承 .5879）：F1−F0 两折 4/4 方向为正但 pooled **+0.28/+0.30pt <1pt** ⇒ 出口 2（不支持）。
- E36（本报告，DroidCollection 子集）：**方向不一致、pooled 非正** ⇒ 出口 3（不支持）。
- 两者是**两个不同数据/协议上的独立证据**，都未达到"H2 初步支持"；**不能合并表述为"H2 已被证明/被证伪"**，也不能把 E36 写成"修复"或"复验"E35。E35 结论与报告不改动。

## 7 验收清单对照（规范 §8）

1. 脚本与命令：`scripts/h2_droid_e36.py`；`OMP_NUM_THREADS=8 python scripts/h2_droid_e36.py [--audit-only|--smoke]`（见 §8）。
2. 审计与断言：`audit.json`（计数、跨 split sha1 重复=0、折 generator 交集=0、held-out 全部有 test）。
3. 特征口径：§2（tokenizer/max_length/batch/encoder asset 全在 `manifest.json → feature_spec`）。
4. 两折及 pooled 指标：§摘要表 + §4（`metrics.json`）。
5. anchor 与逐 generator：§4.4 + §4.2（`metrics.json → folds.*.anchor / per_generator_recall`）。
6. `manifest.json`：git commit、数据 revision、SHA256SUMS 全文件 hash、输入清单 hash、特征缓存 hash。
7. 使用声明：`test_used_for_final_eval_only=true`（仅最终评估，无任何选择）；`random_ref_uses_original_test_rows=false`（随机参照池=原始 train/dev）；`diagnostic_used=false`（仅审计计数）。
8. 失败/中止路径：脚本默认拒绝覆盖产物、断言失败即停止；本次未触发。

### 最终代码审计（指导 §二 十项核对）

| # | 核对项 | 证据 |
|---|---|---|
| 1 | fold_plan train/held-out 交集 0 | `audit.json → folds.*.families.*`（断言通过；复现运行再次断言） |
| 2 | sha1 跨 split 与 split 内重复 0 | `audit.json → dup_sha1`（machine/human 全 0） |
| 3 | test generator 未参与拟合/标准化/早停/调参 | 代码：tr/va 仅含 train_generators；scaler 仅 fit 于训练侧；无 early-stop；超参固定（config.json）；随机参照池已修正为 train/dev |
| 4 | diagnostic 未进入训练/特征拟合/评分 | 代码仅审计计数；`usage.diagnostic_used=false` |
| 5 | F0/F1 共享初始参数/种子/schedule/标准化/optimizer | 代码：deepcopy 初始化、同 RandomState(0) 批序、同 scaler、同 AdamW 配置 |
| 6 | 初始化 F0/F1 logits 与 B0 逐位一致 | `metrics.folds.*.init_maxdiff = 0.0`（两折、复现一致） |
| 7 | 跨 gen 正对=同 family 不同 generator | `cross_pos/supcon_cross_from`（与 E35 同源实现） |
| 8 | BA_G=held-out generator 分组后的 family recall 平均（非 unseen-generator 分类） | `fam_eval` 实现+本报告表述 |
| 9 | 记录 tokenizer/ckpt/max length/batch/特征 hash | `manifest.feature_spec` + `features_cache.sha256` |
| 10 | 只保留一份正式产物 | `artifacts/h2_droid_e36` 为唯一正式目录（smoke/repro 已清理；缓存留在 `runs/` 不入库） |

## 8 复现

```
cd /root/autodl-tmp/u-det/d-det
OMP_NUM_THREADS=8 python scripts/h2_droid_e36.py --audit-only   # 阶段 A（0.4s）
OMP_NUM_THREADS=8 python scripts/h2_droid_e36.py --smoke        # 阶段 B+C（8.1s）
OMP_NUM_THREADS=8 python scripts/h2_droid_e36.py                # 阶段 D 全量（176.0s）
OMP_NUM_THREADS=8 python scripts/h2_droid_e36.py \
  --out artifacts/h2_droid_e36_repro --force                    # 复现（HEAD 83fad1c；缓存复用；14.3s）
```

复现核验：`repro_compare.json` — 主指标（两折+pooled B0/F0/F1、anchor、det、exit）逐项一致；`predictions.npz` 逐位一致；`usage` 新增 2 个声明字段（设计内）；随机参照按指导修正后仅该参照项变化（.2927→.3005、.3090→.2966），不影响 H2 结论口径。

依赖：`data/h2_droid_subset/`（SHA256SUMS 校验通过）、`runs/v0.4.1_covreg/last.pt`、`runs/flagship_r1/head.pt`、`runs/flagship_e27/model_state.pt`、`checkpoints/codet5-base`。**正式产物 8 件**在 `artifacts/h2_droid_e36/`（含 `repro_compare.json` 与 `report.md` 副本）；特征缓存约 57MB 留在 `runs/`（不入库）。
