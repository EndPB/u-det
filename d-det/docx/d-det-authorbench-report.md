# d-det · AuthorBench 任务条件归因审计报告

> 执行规范：`docx/d-det_AuthorBench任务条件归因实验_DeepSeek执行指导_2026-10-01.md`（基线提交 `e866c82`）。
> 脚本：`scripts/authorbench_taskaware.py`；产物：`artifacts/authorbench_taskaware/{audit,config,manifest,metrics}.json + predictions.npz + solver.log + 4 张图件`；特征缓存 `runs/authorbench_taskaware/`（不入库）。
> 定位：**任务条件归因审计**，不是多 family H2 实验；C-only 数据；不修改 E35/E36 报告。

## 一句话结论

在未微调 CodeT5-base 的 768 维冻结表示上、按任务级 167/36/36 留出：**family 归因 raw 仅 .3206（macro-F1，约 1.9×机会），任务内无标签中心化（z_center）小幅提升到 .3624，再叠加训练任务标准差的 z_center_std 大幅提升到 .6305（≈3.8×机会；任务级 bootstrap CI [.5711, .6901]）；meta_only .3260 与之相近、并非接近机会**。model 8-way 归因在 raw/center 上高于 family（.3867/.5273 vs .3206/.3624），但在 z_center_std 上 family（.6305）反超 model（.5749）。§5 OpenAI generator 留出（gpt-4o/gpt-4.1/gpt-4o-mini）：family=OpenAI 召回 raw .75–.93、center .86–1.0，**但被留出 generator 在 7-way model 分类器下只有 28–44% 落到已见 OpenAI 模型上**。按指导 §6 机制规则逐条对照（见 §5）：中心化有帮助、任务内容是主要干扰这一说法只得到部分支持；不做 H2 通过/不通过的判定。

## 摘要表（test=36 任务/288 行；机会：family 1/6≈.1667，model 1/8=.125）

| 表示 | family macro-F1 | family balAcc | CI95（任务级） | LDA 对照 | model macro-F1 | model balAcc |
|---|---|---|---|---|---|---|
| z_raw | .3206 | .3457 | [.2722, .3622] | .5955 | .3867 | .4375 |
| z_center（转导中心） | .3624 | .3843 | [.3242, .3997] | .6623 | .5273 | .5625 |
| **z_center_std** | **.6305** | **.6235** | [.5711, .6901] | .6623 | .5749 | .5694 |
| meta_only | .3260 | .3873 | [.2934, .3573] | .3027 | .3049 | .3472 |

## 1 数据审计（audit.json；未加载 encoder）

- 1,912 行 / 239 任务；每任务恰好 8 行 × 8 个不同模型（断言全过）；6 家族（OpenAI 含 3 个 generator，其余各 1 个）。
- 划分：train/dev/test = 167/36/36 任务（1,336/288/288 行）；`task_id` 不跨 split（与 `fold_plan.json` 逐项一致）。
- `source_sha256`：全局重复 0、跨 split 重复 0、任务内重复 0。
- `replicate_count`：源数据同 prompt-model 重复数（本子集每对保留 1 条，重复不视为独立任务）。
- 元数据分布（p50/p90/max）：char_count 2137/4817/20353；num_lines 81/177/667；nloc 57/129/476；CC 3.0/9.5/44.0；token_size 600/1484/5736。
- 语言仅 C；`LLM-AuthorBench` 源压缩包 sha256 `e24399b7…`（summary.json 记录）。

## 2 表示提取口径

- **未微调** CodeT5-base（`checkpoints/codet5-base`，复用 `encoders/codet5.py` 加载方式，冻结）；token hidden state + attention-mask 均值池化 → 768 维。
- `max_length=512`；超长按头 384 + 尾 128 截断（**实测 p50 即为 512，绝大多数样本走截断路径**）；`<8` token 跳过（0 条）。
- bf16 仅用于 GPU 推理，缓存保存 float32；逐行保存 task_id/model_name/family/source_sha256 并在训练前与 JSONL 逐位复核通过。
- 运行资源：OMP=MKL=2、torch 线程 2；GPU（RTX 3080 Ti）batch=8，1,912 行 11.8s（162 行/s），**maxVRAM 914MB**；同批序重复编码逐位一致、跨批组成差异 0.00e+00；**未联网、未调用任何外部 API**。
- 四种表示：z_raw；z_center=z−同任务 8 输出无标签均值（**转导**：测试任务兄弟输出可用；不代表单样本部署算法）；z_center_std=z_center 再按训练任务逐维标准差标准化；meta_only=[char_count, num_lines, nloc, CC, token_size]（原始量纲）。

## 3 协议

- 固定任务级划分；LR 多分类 C∈{0.03,0.1} 以 dev macro-F1 选一（并列取小 C），**全部拟合仅用 train 任务**；dev 仅选择；test 终评；LDA(lsqr, shrinkage=auto) 为交叉检查（无超参）。
- 指标：macro-F1、balanced accuracy、逐类召回、混淆矩阵、任务级 bootstrap（1000×，重采样 36 个 test 任务）CI95。
- 图件为描述性诊断（train-fit PCA / KMeans k=6 / LDA 投影 / 协方差迹比），图内标注“Descriptive only — not H2 evidence”。

## 4 结果

**4.1 family 6-way（逐族召回）**
- z_raw：gemini .778、openai .963，**claude .278 / llama .056 / deepseek 0 / qwen 0** —— 未中心化的读出基本只识别两个家族。
- z_center：claude .389、gemini .889、openai 1.0，deepseek/llama/qwen 仍 ≈0。
- z_center_std：claude .694、deepseek .222、gemini 1.0、llama .611、openai .824、qwen .389 —— 六个家族全部高于各自 raw 值（deepseek/qwen 从 0 提升），混淆矩阵主要残余错误为 deepseek→openai（14/36）与 qwen→deepseek（12/36）。
- meta_only：claude .722、gemini .694、openai .907，deepseek/llama/qwen 0 —— 元数据本身携带“claude/gemini/openai”的结构性捷径。
- LDA 对照与 LR 同向（raw .5955 → center(.std) .6623），且对中心化不敏感（LDA 内置白化）。

**4.2 model 8-way 捷径诊断**：raw .3867 / center .5273 / center_std .5749 / meta .3049（balanced acc 同向）。逐模型：gemini 与 claude 恒高（≥.97/.72）；gpt-4o 在 8-way 中仅 .17（raw）/.31（center）；deepseek .08–.25。**model 化在弱表示上普遍高于 family（捷径 ⇒ 具体模型指纹），但在 z_center_std 上 family（.6305）> model（.5749）**。

**4.3 描述性诊断（均为 train-fit；图件=fig_pca_raw_center_family.png、fig_pca_raw_center_split.png、fig_kmeans.png、fig_lda.png）**
- KMeans k=6：z_raw silhouette .0388 / ARI .0221 / purity .4124；z_center .0208 / .0687 / .4506（聚类弱，不构成 family manifold 证据）。
- 协方差迹比（类内/类间）：z_raw train 33.9 / test 26.9 → z_center 16.9 / 13.7（中心化把类内散度占比减半，但仍远大于 1）。
- PCA/LDA 投影见 4 张图（标注为非 H2 证据）。

**4.4 §5 OpenAI generator 留出（辅助诊断）**

| 留出 generator | 口径 | family=OpenAI 召回（raw→center） | 8-way 指纹召回（raw→center；未做留出，仅对照） | 7-way 中落到已见 OpenAI 模型占比（raw→center） |
|---|---|---|---|---|
| gpt-4.1（n=36 test） | 严格 test | .917→**1.000** | .722→.917 | .444→.417 |
| gpt-4o（n=36 test） | 严格 test | .750→**.944** | .167→.306 | .333→.361 |
| gpt-4o-mini（n=36 test） | 严格 test | .861→.861 | .222→.389 | .278→.417 |
| gpt-4.1（n=72 dev+test） | 任务条件 | .931→.986 | .694→.903 | .361→.389 |
| gpt-4o（n=72 dev+test） | 任务条件 | .833→.958 | .167→.208 | .403→.403 |
| gpt-4o-mini（n=72 dev+test） | 任务条件 | .861→.875 | .194→.375 | .292→.444 |

读法（限定为探索性）：见过两个 OpenAI generator 后，第三个的样本**多数仍落在 OpenAI family 判别侧**（严格口径 raw .75–.92、center .86–1.0，两个口径同向）；但 **model 层面并不收敛到已见的 OpenAI 兄弟**（7-way 只有 28–44% 落到已见 OpenAI 模型上），且该 generator 自身的 8-way 指纹召回差异大（gpt-4.1 高、gpt-4o/mini 低）。这与“family 层面的来源关系弱但非零、generator 层面的 fingerprint 各自独立”一致；**由于只有 OpenAI 一家有多个 generator，不能把它写成多 family H2 结论**。

## 5 判读（指导 §6 六条机制规则逐条对照）

1. **“z_center 相比 z_raw 在任务留出上提升，且 meta_only 很低”**——部分成立：中心化提升（LR .3206→.3624；LDA .5955→.6623；z_center_std 进一步 .6305），但 **meta_only 并不低**（.3260，≈raw；claude/gemini/openai 的元数据捷径显著）。因此“任务内容是主要干扰因素”的强表述不被支持；“中心化有帮助”成立。
2. **“只在同任务转导版本提升”**——不适用：本评估本身就是严格任务留出（test 任务不参与任何拟合），z_center 仍提升；但 **z_center 的构造使用测试任务兄弟输出（转导）**，单样本部署口径下的对应数字应读 z_raw/meta_only。
3. **“model 归因高于 family 归因”**——在 raw/center 上成立（.3867/.5273 vs .3206/.3624），说明弱表示下的具体模型指纹捷径强；在 z_center_std 上 family 反超 model（.6305 vs .5749），该规则不普适。
4. **“OpenAI held-out generator 仍能归入 OpenAI 且 model 归因下降”**——前半成立（§4.4）；model 侧同时观察到 7-way 分配分散、8-way 指纹因 generator 而异——仅作 within-family 探索性记录，不作总体支持。
5. **“所有表示接近机会”**——不成立（z_center_std 显著高于机会）。
6. **异质性**：逐家族提升不均（deepseek .222、qwen .389 仍弱），只报告异质性，不写成总体结论。
- 总口径：本实验为**机制审计**；不判定 H2 通过与否，不与 E35/E36 合并陈述。

## 6 边界与诚实性声明

- **C-only**（239 个 C 任务）；不外推到其他语言、DroidCollection 或 SemEval；不修改 E35/E36 正式报告。
- **OpenAI 是唯一多 generator 家族**；§5 只回答“第三个 OpenAI generator 是否仍落在 OpenAI 判别侧”，不代表多 family 跨 generator H2。
- **z_center 是转导表示**（同任务兄弟输出可用），已在 config/metrics/report 三处标注；严格口径=测试任务不参与训练/标准化/选择，两种口径全程分开。
- 单次划分、单种子；bootstrap 仅针对 test 任务重采样；1pt 类阈值不适用（机制审计）。
- 图件全部为描述性诊断（“Descriptive only — not H2 evidence”）；聚类弱不用于“不可判别”的结论，判别式与聚类是不同问题。
- `meta_only` 未做量纲变换（原始量纲）——其读数偏低可能与尺度有关，报告中不作为强证据方向。

## 7 复现与交付清单（指导 §7/§8）

```
cd /root/autodl-tmp/u-det/d-det
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
/root/miniconda3/envs/udet/bin/python scripts/authorbench_taskaware.py --audit-only
/root/miniconda3/envs/udet/bin/python scripts/authorbench_taskaware.py \
  --smoke --batch-size 8 --max-length 512 --out artifacts/authorbench_taskaware_smoke
/root/miniconda3/envs/udet/bin/python scripts/authorbench_taskaware.py \
  --batch-size 8 --max-length 512 --out artifacts/authorbench_taskaware
```
（正式运行 88.5s；图件 fig_pca_raw_center_family.png / fig_pca_raw_center_split.png / fig_kmeans.png / fig_lda.png 由该命令自动生成。）

对照交付要求：①脚本 `scripts/authorbench_taskaware.py`；②`artifacts/authorbench_taskaware/`（8 件+4 图）；③report 的 raw/center/meta 对照=§摘要表+§4.1；④PCA/聚类图与生成命令=§4.3+上列命令；⑤GPU=是（3080 Ti，batch 8，maxVRAM 914MB），CPU 线程 OMP=MKL=2 + torch 2；⑥中心化口径=转导（任务条件），严格口径已分列；⑦OpenAI 结果不代表多 family H2（§4.4/§6）。
