# d-det · H2 报告 DroidCollection v2：generator-held-out 家族归因（正式主包）

> 执行规范：`docx/d-det_H2_DroidCollection_v2_服务端AI执行指导_2026-10-02.md`。
> 脚本：`scripts/h2_droid_v2.py`（v2 基线提交 `de88cad`，运行 manifest 记录同值；协议与 E36 同构沿用）。
> 数据：`data/h2_droid_full_selected_upload/`（ZIP sha256 `22074bbe…` 与规范一致；SHA256SUMS 5/5 通过——注：`sha256sum -c` 因 CRLF 行尾报错，已用 Python 逐行解析复算，全部 OK；大包不入库，见 `.gitignore`）。
> 产物：`artifacts/h2_droid_v2/{audit,config,manifest,metrics}.json + predictions.npz + solver.log + repro_compare.json`；
> 定位：**DroidCollection v2 上的 generator-held-out 家族归因候选证据**，与 E35 结论严格分开；与 E36 为同一协议家族（同 fold_plan，sha256 `e1b3d131…`），样本量 18k→146.7k。
> 修订（2026-10-02，依据 `docx/d-det_H2_DroidCollection_v2_最终结果审计_2026-10-02.md`）：补充 anchor 覆盖限制——整体 .714/.857 不代表每个 family-fold 单元被直接检验；CodeLlama fold_0 与 IBM Granite 两折无有效跨 generator 正对 ⇒ 标注"未充分检验"。出口 3 判定不变；无重训、无指标/产物变动（`artifacts/h2_droid_v2/report.md` 副本同步）。

## 一句话结论

在 146,718 行正式主包、未微调冻结 768 维表示、严格 generator-held-out 协议下：**两折的 F1−F0 四个差值全部为负（fold_0 −0.73/−0.68pt；fold_1 −0.79/−0.60pt），pooled 下降 0.63/0.60 个百分点 ⇒ 按预注册出口 3：H2 不支持**。整体 anchor 守卫通过（fold_0 .714 / fold_1 .857 ≥ .15）；**但覆盖限制须同时报告**：CodeLlama fold_0 与 IBM Granite 两折无有效跨 generator 正对、未被 L_cross 直接训练 ⇒ 对这些单元只能报告"未充分检验"（见 §5 覆盖限制）。同时注意：本数据上 family 主轴本身贴地（gen-heldout B0 BA_F≈.147–.162，仅 1.03–1.14×机会 1/7≈.1429；随机切分参照 .327–.329，迁移损失 ≈17–18pt）——中心化/残差都建立在一个非常弱的主轴之上。**不写成多 family H2 通过证据；不修改 E35/E36 报告。**

## 摘要表（BA_F / BA_G；机会 1/7≈.1429）

| 模型 | fold_0（test 6,142） | fold_1（test 6,368） | pooled（12,510） |
|---|---|---|---|
| B0 冻结加权主轴 | .1470 / .1488 | .1624 / .1668 | .1543 / .1548 |
| F0 残差 CE | .1493 / .1503 | .1680 / .1697 | .1580 / .1573 |
| F1 F0+0.1·L_cross | .1420 / .1435 | .1601 / .1637 | .1517 / .1513 |
| **Δ(F1−F0)** | **−0.73 / −0.68pt** | **−0.79 / −0.60pt** | **−0.63 / −0.60pt** |
| anchor 有效/总（比例） | 74,700/104,580（.714） | 82,944/96,768（.857） | — |
| 检测 AUROC（次要；test/dev） | .9511 / .9577 | .9553 / .9539 | — |
| 随机切分参照（主轴，池=train/dev；仅参照） | .3287 / .3236 | .3266 / .3198 | — |

## 1 数据与完整性（audit.json）

- 总行数 **146,718** = MACHINE_GENERATED **125,718** + HUMAN_GENERATED **21,000**；7 家族、32 个 machine generator（与 E36 同集合、同 fold_plan）。
- split：train 107,599 / dev 19,609 / test 19,510（machine 100,599 / 12,609 / 12,510；human 7,000×3）。
- `source_row_sha1`：machine/human 全 0 跨 split 重复、0 内重复；折断言（两侧非空、交集 0、held-out 全有 test、观测 generator ⊆ 计划集合）全部通过。
- 计数断言（v2 新增）：146718/125718/21000/107599+19609+19510 逐一断言通过。
- 语言：Python 66,267 / Java 27,785 / C++ 17,835 / JavaScript 14,532 / C# 11,553 / Go 5,383 / C 3,363；Generation_Mode：INSTRUCT 98,697 / COMPLETE 27,021 / Human_Written 21,000。
- 每个 generator 的训练计数在 `audit.json → generator_counts / folds.*.per_train_gen_counts`（训练侧权重按 generator-aware `w=N/(K|G_f|N_g)`，batch 采样为族→generator→样本均匀，未复制任何样本）。

## 2 特征口径（与 E28/E35/E36 相同）

- 冻结 `runs/v0.4.1_covreg/last.pt` + `runs/flagship_r1/head.pt` + SetPool 与 m_raw 均值池化；CodeT5 tokenizer（`add_special_tokens=False`）；max_length=1024（>1024 → 头 768+尾 256）；<8 token 跳过（**0 条**）；输出 768 维；bf16 仅用于 GPU 推理（缓存 float32）。
- 实测：tokenize 17.1s；编码 146,718 行 **1214.6s**（120.8 行/s，bs=16，RTX 3080 Ti）；tok 长度 p50/p90/p99/max = 235/570/1024/1024；**同批序重复编码逐位一致**（跨批组成差异 max|Δ|=1.79e-07，bf16 批核效应，记录在案）。
- 缓存 `runs/h2_droid_v2/features.npz`（sha256 `70ad983f08ccd8a2…`，约 450MB，**不入库**）+ `metadata.jsonl`；逐行 `source_row_sha1` 在训练前与 JSONL 逐位复核通过（两次运行同一缓存，hash 不变）。

## 3 协议（预注册）

- 严格读 `fold_plan.json`（不自行划分）；训练＝machine+train+train_generators、验证＝machine+dev+train_generators、测试＝machine+test+heldout_generators；Human 只用于检测头。
- B0：加权多分类 LR（C=.1、max_iter=10000、tol=1e-6、w=N/(K·|G_f|·N_g)）逐折拟合并**冻结**（两折收敛，n_iter 795/865）。
- F0/F1：阶段 B 修正门控残差（W_R=0、η0=0.5；U Xavier、V std=1e-3）；F1=+0.1·L_cross（τ=.1 只除一次；正对=同家族不同 generator；无正对 anchor 跳过；分母含全批；Human/同 gen/异族不入）。共享主轴/初始化/种子/批序/标准化/优化器/epoch（批 7 族×18；415/384 步每 epoch；2 epoch；AdamW 1e-3/wd 1e-4/clip 1）。**初始化审计：F0/F1 初始 logits 与 B0 逐位一致（两折）**。
- 出口（§6 固定判读）：1) 两折两项为正且 pooled ≥1pt；2) 方向一致但 <1pt；3) 方向不一致或 pooled 非正；4) anchor 过低⇒检验不足；5) 单族/单 generator 改善⇒仅异质性。随机切分仅参照。
- test 只做最终评估（`usage.test_used_for_final_eval_only=true`）；`diagnostic_hybrid_adversarial.jsonl` 仅审计计数、未参与训练/评分。

## 4 结果

**4.1 逐族 pooled 召回（B0 / F0 / F1）**：qwen .2044/.2001/.2115；meta-llama .1953/.2162/.2286；microsoft .2703/.2517/.2426；01-ai .0517/.0747/.0647；codellama .1194/.1212/.1024；deepseek-ai .0885/.1004/.0850；ibm-granite .1504/.1416/.1274。逐族 Δ(F1−F0) 有正有负（meta +1.24pt 最大正、ibm −1.42pt 最大负），**无跨族一致增益**。

**4.2 逐 held-out generator 召回（pooled，B0/F0/F1；规范要求全列）**

| generator | B0 | F0 | F1 || generator | B0 | F0 | F1 |
|---|---|---|---|---|---|---|---|
| Qwen2.5-72B-Instruct | .139 | .149 | .149 | Yi-Coder-9B | .059 | .065 | .036 |
| Qwen2.5-Coder-1.5B | .216 | .236 | .236 | Yi-Coder-1.5B | .050 | .083 | .050 |
| Qwen2.5-Coder-7B-Instruct | .117 | .130 | .135 | Yi-Coder-9B-Chat | .046 | .074 | .069 |
| Qwen2.5-Coder-32B-Instruct | .025 | .028 | .036 | Yi-Coder-1.5B-Chat | .056 | .078 | .071 |
| Qwen2.5-Coder-1.5B-Instruct | .330 | .311 | .336 | CodeLlama-34b-Instruct-hf | .156 | .156 | .144 |
| Qwen2.5-Coder-7B | .171 | .143 | .155 | CodeLlama-7b-hf | .015 | .015 | .015 |
| Qwen2.5-Codder-14B-Instruct | .689 | .640 | .668 | CodeLlama-70b-Instruct-hf | .096 | .100 | .072 |
| Llama-3.3-70B-Instruct | .193 | .210 | .234 | deepseek-6.7b-base | .070 | .086 | .055 |
| Llama-3.1-8B | .440 | .444 | .472 | deepseek-1.3b-base | .087 | .087 | .087 |
| Llama-3.2-1B | .367 | .350 | .350 | deepseek-6.7b-instruct | .106 | .126 | .111 |
| Llama-3.3-70B-Turbo | .115 | .158 | .151 | deepseek-1.3b-instruct | .060 | .036 | .048 |
| Llama-3.2-3B | .298 | .315 | .331 | granite-8b-base-4k | .173 | .154 | .142 |
| Llama-3.1-8B-Instruct | .106 | .133 | .135 | granite-8b-instruct-4k | .085 | .106 | .085 |
| Phi-3-medium-4k-instruct | .214 | .202 | .207 | | | | |
| Phi-3-small-8k-instruct | .321 | .301 | .290 | | | | |
| phi-2 | .043 | .043 | .039 | | | | |
| Phi-3-mini-4k-instruct | .362 | .349 | .343 | | | | |
| Phi-3.5-mini-instruct | .363 | .336 | .315 | | | | |
| phi-4 | .124 | .092 | .087 | | | | |

跨度 .015（CodeLlama-7b-hf）到 .689（Qwen-Codder-14B）；F1−F0 正负混合（如 Qwen2.5-Coder-1.5B-Instruct +2.5pt、Yi-Coder-9B −2.9pt、Llama-3.2-1B ±0pt（F0/F1 完全相同）），未构成一致的跨 generator 增益。

**4.3 anchor 与训练动态**：整体 anchor .714/.857（结构=训练侧多 generator 族的样本占比；fold_0 74,700/104,580、fold_1 82,944/96,768）。**逐 family-fold 覆盖（2026-10-02 审计修正）**：fold_0 的 codellama 训练侧仅 1 个 generator（anchor 0/14,940）、ibm-granite 训练侧仅 1 个（0/14,940）；fold_1 的 codellama 训练侧 2 个（有效 13,824/13,824）、ibm-granite 仍为 1 个（0/13,824）；其余族两折均有效。即 3 个 family-fold 单元（CodeLlama fold_0、IBM Granite 两折）无跨 generator 正对、未被 L_cross 直接训练——整体比例不能写成"所有 family-fold 的 anchor 覆盖充分"。γ 从 .046 单调降到 .019–.023（两臂同降；残差参与度 9.7%–22.7%）——CE 在更大数据上倾向收缩残差；F1 漂移 ‖U‖≈36–38、‖V‖≈12、‖W_R‖≈2.0。首末批 CE：fold0 1.27→1.32、fold1 1.24→1.34（F0/F1 近同步）。

**4.4 语言分桶（pooled BA_F；B0/F0/F1）**：Python .154/.156/.150、Java .162/.168/.159、C++ .141/.138/.138、JavaScript .155/.174/.179、C# .162/.169/.155、Go .098/.095/.083、C .161/.162/.133。Source/Mode/长度分桶见 `metrics.json`。

**4.5 次要读出**：检测 AUROC（train 侧训练、test 评估）.9511/.9553——检测在该数据上很强，但**检出不等于家族归因**（B0 家族读出仍≈机会），不作 H2 证据。

**4.6 与 E36 的关系（同一协议家族）**：fold_plan 完全相同（sha256 一致）；E36（18k 子集）pooled：B0 .1605、F0 .1608、F1 .1591，Δ = −0.17/−0.20pt，出口 3；v2（146.7k 主包）pooled：B0 .1543、F0 .1580、F1 .1517，Δ = −0.63/−0.60pt，出口 3。**两轮独立样本量下均为负向（v2 更负），主轴水平相当且贴地**——这是对同一问题在两个规模上的负向候选证据，彼此独立、不合并宣称。

## 5 判读（预注册出口逐条）

1. 出口 1（初步支持）：**不达**（两折四项全为负）。
2. 出口 2（方向一致但 <1pt）：不适用（增量为负，非"一致为正"）。
3. **出口 3（方向不一致或 pooled 非正）：成立 ⇒ H2 在本数据与 768 维冻结表示下不支持。**
4. 出口 4（anchor 过低）：整体不成立（.714/.857 ≥ .15，`anchor_ok=true`）；但覆盖有限（见下方覆盖限制）——CodeLlama fold_0 与 IBM Granite 两折未被 L_cross 直接训练，对这些单元只能报告"未充分检验"。
5. 出口 5（单族/单 generator 改善⇒异质性）：记录——逐族/逐 generator 增减混合，无一致结构，不写成任何总体支持。

**覆盖限制（2026-10-02 审计修正）**：整体 anchor 比例足以通过 ≥.15 守卫，但该指标不能代表每个 family-fold 都被 H2 损失检验——fold_0 的 CodeLlama 训练侧只有 1 个 generator，fold_0/fold_1 的 IBM Granite 训练侧都只有 1 个 generator，因此这些单元跨 generator anchor=0（fold_1 的 CodeLlama 有 2 个训练 generator，该折有效）。最终表述：
> 在大多数 family-fold 单元有有效跨 generator 正对的正式协议下，F1 未显示增益；CodeLlama fold 0 和 IBM Granite 两折没有被跨 generator 对比损失直接训练，故对这些单元只能报告未充分检验。

此为报告边界修正，不改出口 3 判定、不需要重新训练。

- 边界备注：B0 本身贴地（≈1.03–1.14×机会）且远低于随机参照（迁移损失 ≈17–18pt）；"H2 不支持"与"该表示对跨 generator 家族来源的可读性极低"两种读法都与数据一致，报告不额外宣称。单种子/单划分；1pt 阈值非显著性检验。

## 6 边界与诚实性声明

- 无公开 `task_id/prompt_id` ⇒ 仅支持"DroidCollection 上的 generator-held-out 家族归因候选证据"表述；不写同题配对、prompt-invariant、后处理因果或"纯归因空间"。
- 检测 AUROC 与随机参照均为次要/参照读数，不作 H2 证据；单一 family/generator 的改善只作异质性记录。
- anchor 覆盖边界（2026-10-02 审计修正）：整体 .714/.857 仅表示"大多数" family-fold 单元被跨 generator 正对检验；CodeLlama fold_0 与 IBM Granite 两折无有效正对 ⇒ 只报"未充分检验"，不写"所有 family-fold 的 anchor 覆盖充分"。
- 未读取 test 做任何选择；未使用 diagnostic 文件；未调 SupCon/门控/学习率；未加任何新几何分支；未静默更换 encoder/checkpoint（资产路径见 manifest.feature_spec）。
- 大特征缓存（约 450MB）留在 `runs/h2_droid_v2/`（不入库）；正式产物唯一化（smoke/repro 目录已清理，复现对比见 `repro_compare.json`）。
- 与 E35/E36 结论分开陈述；不修改 E35/E36 级正式报告。

## 7 复现

```
cd /root/autodl-tmp/u-det/d-det
export OMP_NUM_THREADS=8
# 阶段 A（0.4s 级，含 SHA256SUMS 校验与计数断言）
python scripts/h2_droid_v2.py --audit-only
# 阶段 B/C 冒烟（独立目录；每 (split,label,generator) ≤6 行）
python scripts/h2_droid_v2.py --smoke --batch-size 8 \
  --out artifacts/h2_droid_v2_smoke
# 阶段 D 全量（首跑 1577.9s，其中编码 1214.6s；写入正式目录）
python scripts/h2_droid_v2.py
# 复现（缓存复用；manifest commit=de88cad；310.0s）
python scripts/h2_droid_v2.py --out artifacts/h2_droid_v2_repro --force
```
**复现对比**（`repro_compare.json`）：metrics（data/exit/folds/pooled/protocol/usage）逐项一致、`predictions.npz` 全部键逐位一致、特征缓存与数据 hash 一致；manifest commit 由首跑的 `d261950` 修正为含 v2 脚本的 `de88cad`（provenance 修正）。核对后已把 repro 产物替换为唯一正式产物。

## 8 交付清单对照（规范 §9）

① 命令与脚本：上节 + `scripts/h2_droid_v2.py`；② `audit.json` 与 generator 泄漏断言：§1（全过）；③ encoder/checkpoint/tokenizer/max length/batch：§2（另见 `manifest.feature_spec`）；④ B0/F0/F1 两折与 pooled：摘要表；⑤ anchor 比例、逐族、逐 generator：§4.1/4.2/4.3；⑥ manifest：代码 commit `de88cad`、数据文件 sha256、features 缓存 sha、配置（`config.json`）；⑦ 说明：test 仅最终评估、diagnostic 未使用（`metrics.usage`）；⑧ 失败保留：无失败；断言失败即停止的机制在脚本中保留。
