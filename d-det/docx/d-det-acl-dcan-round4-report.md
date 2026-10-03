# d-det · ACL 数据集合 v1 第四轮报告：单变量捷径诊断 + 预算扩展 + 外部泛化审计（bdd569 修正版协议）

> 执行规范：`docx/d-det_AutoDL_bdd569后续执行指导_2026-10-03.md`。
> 基线协议：bdd569（round3 审计修正版）；切分/身份字段/统计口径与 round3 一致；**不与 408-task 主表混合外部结果**。
> 产物：`artifacts/acl_dcan_round4/{shortcuts_univar, lora_extend, external}/`（各含 config/env/split_manifest/hashes + 日志 + 结果摘要）。
> 定位：诊断轮（不写 SOTA）。

## 0 一句话结论

① 单变量捷径诊断：char TF-IDF 的性能**依赖可解释词法线索**——仅去除标识符 −11.8pt、仅去注释 −6.3pt、仅去字符串/字符常量 −3.0pt（均含任务级配对 CI 且显著），联合清洗 −31.2pt（超可加）；**空白/格式对 char_wb 完全不可见（±0.0pt）**，为分析器构造性结果（不能推广为"格式对一切模型无关"）。② LoRA 预算扩展（12→24ep, patience 5）：均值 **.7461**（vs round2 .7286，+1.7pt；s0/s2 配对 CI 显著），**s1/s2 直到 24ep 上限仍创新高 ⇒ 欠拟合未完全排除**；s2 单 seed 已与 TF-IDF 统计不可区分（−0.4pt，CI 跨零），但均值仍低于 TF-IDF，不作方法主张。③ 外部审计（诊断）：STACAD fold0 TF-IDF .3726 > LoRA .3506（+2.2pt\*）> head .2256；Droid fold0 三族均弱（.164/.132/.127，CI 跨零，与 round1 参照一致）——无灾难性崩溃、无新方法主张。

## 1 A · 单变量捷径诊断（h2_authorbench_dcan，task-held-out；`shortcuts_univar/`）

### 1.1 协议

- C 词法状态机（字符串感知；确定性、无拟合参数）：`ids_only`（非关键字标识符 → `_id`）、`strings_only`（字符串/字符常量 → `""`）、`comments_only`（去 // 与 /* */，字符串内 //、/* 不受影响）、`ws_only`（空白折叠）、`all`（联合）；`raw` 为对照。
- TF-IDF：char_wb(2,4)、min_df=5、sublinear、max_features=300k（**仅 train 拟合**）；SGD log_loss α=2e-6、5 epoch、类逆频率权重；3 seeds；dev 仅记录/选择；test 冻结后评一次。
- 双口径：`best-dev`（统一选择规则）与 `epoch5`（round2 同协议，仅记录不恢复）。
- 配对统计：raw − 变体，任务级 bootstrap B=2000（408 任务）。

### 1.2 结果（test 宏 F1）

| 变体 | best-dev 均值±std | epoch5 均值 | 逐 seed（best） | raw−变体 均值差 |
|---|---|---|---|---|
| raw | **.7619±.0029** | .7672 | .7659 / .7589 / .7609 | — |
| ids_only | **.6436±.0031** | .6523 | .6393 / .6466 / .6448 | **+11.8pt**（CI 全离零） |
| comments_only | **.6989±.0067** | .7041 | .6995 / .7068 / .6905 | **+6.3pt**（CI 全离零） |
| strings_only | **.7323±.0057** | .7323 | .7395 / .7318 / .7256 | **+3.0pt**（CI 全离零） |
| ws_only | .7619±.0029 | .7672 | 同 raw（逐位） | **+0.0pt**（构造性：char_wb 按词切分，空白不可见） |
| all | **.4504±.0056** | .4271 | .4486 / .4580 / .4445 | **+31.2pt**（CI 全离零；超可加：21.1pt 单项之和 → 31.2pt） |

- 逐 seed 配对 CI（raw−变体）：ids [+0.101,+0.153]/[+0.086,+0.136]/[+0.089,+0.143]；comments [+0.049,+0.085]/[+0.030,+0.075]/[+0.048,+0.093]；strings [+0.010,+0.043]/[+0.007,+0.047]/[+0.019,+0.052]；ws 全 [0,0]；all [+0.288,+0.346]/[+0.271,+0.331]/[+0.287,+0.348]。
- **判读**：标识符匿名化是最大单因子；注释为第二；字符串第三，三者皆显著且可分离——**修正 round2/round3 "尚不能区分各清洗项贡献"的限制（对 char_wb TF-IDF 已可区分）**；"联合清洗 → .4572"的旧口径数字因 regex 实现在字符串含 `//`、`/*` 时的误伤与本轮词法实现不同（本轮 epoch5 均值 .4271、best-dev .4504），以本轮词法口径为准。
- 边界：① ws 的零效应是 **char_wb 分析器的构造性质**（按空白分词后仅取词内 n-gram）；对词级/神经模型格式仍可能携带信息，不得外推；② 变换均为"删/替换信号"的操作，联合值超可加说明存在交互（如 `_id` 化后注释删除使残余文本更趋同），不构成机制因果分解的全部；③ task 混淆维度（同任务多 family 的提示词统计）未单独变换，仍待未来。
- 产物：`shortcuts_univar/{metrics.json（含 dev 曲线/变换示例/vocab 大小）, predictions.npz（含 408 任务身份字段与 split_hash）, stats.json, config/env/split_manifest/hashes, logs/run.log}`。

## 2 B · LoRA 预算扩展（条件执行）

### 2.1 执行条件与协议

- **条件依据（指导 §4B）**：A 显示性能主要来自可解释输入捷径（标识符/注释/字符串三类显著、联合 −31.2pt）⇒ 执行 B；否则不训练。
- **唯一变化**：max epoch 12→**24**、patience 3→**5**；其余与 round2 完全一致（同一 encoder/tokenizer/线性头/CE/采样/split/优化器 lr enc 3e-4、head 1e-3、wd 1e-4、bs16×accum2、bf16、截断 384+128、best-dev 选择）。3 seeds；test 仅在 best state 冻结后评一次。

### 2.2 结果（test 宏 F1；`lora_extend/`）

| seed | best dev（epoch） | 运行 epoch | extended test | round2 test | 差值 [CI95] |
|---|---|---|---|---|---|
| 0 | .7521（ep14） | 19（patience 停） | **.7324** | .7088 | **+2.36pt [+0.73,+4.00] *** |
| 1 | .7696（**ep24=上限**） | 24 | **.7462** | .7372 | +0.90pt [−1.04,+2.83] |
| 2 | .7846（**ep24=上限**） | 24 | **.7597** | .7399 | **+1.98pt [+0.01,+4.02] *** |
| 均值 | — | — | **.7461** | .7286 | +1.75pt |

- dev 曲线（`metrics.json`）：s0 在 ep14 达峰后 patience 停止（ep19）；**s1/s2 到 ep24 上限仍逐轮创新高**（s2 尾部：.750→.780→.761→.785）——**12ep 低估了 LoRA 的预算需求，且 24ep 仍未见 s1/s2 平台**。
- 与 TF-IDF（round1 .7639）对照：s0 −3.14pt\*（负显著）、s1 −1.76pt（跨零）、s2 **−0.42pt（跨零）** ⇒ 延长预算后 s2 单 seed 已与 TF-IDF 统计不可区分；**均值 .7461 仍低于 .7639，且无法排除继续加预算再涨** ⇒ 按指导：B 仅用于确认欠拟合，**不作方法主张**。
- 修正记录：round2 曾写"seed0 最优落在 12ep 上限"；扩展后 s0 实际峰值在 ep14（test +2.4pt）⇒ 12ep 上限对 s0 也是截断的。
- 产物：`lora_extend/{metrics.json, predictions.npz, stats.json, states/lora_ext_s{0,1,2}.pt, config/env/split_manifest/hashes.json, logs}`（states 3×1.2MB，哈希记录在 hashes.json）。

## 3 C · 外部泛化审计（诊断性证据）

### 3.1 协议（成本受控；测试集固定）

- **STACAD fold0（file-held-out，一次）**：test=fold0 28,992 行；train=其余折的 **file-level 子集 30,000**（seed 0 抽文件）；dev=3,000（文件级留出，与 train 零共享文件——已断言）。
- **Droid fold0（generator-held-out，一次）**：train 52,251 / dev 6,482 / test 6,142（仅 MACHINE 行；**7 机器家族归因**；test=heldout generators；dev=训练生成器）。
- 三族同 test×3 seeds：TF-IDF（char_wb、5ep、best-dev）/ head-only（冻结 CodeT5 768d 线性头，≤12ep patience3）/ LoRA（round2 同超参，**预算封顶 2ep/patience1**——诊断预算，两个源 best epoch 均落上限）。
- 配对统计：聚类 bootstrap B=2000（STACAD 按**文件**聚类；Droid 按**生成器**聚类）。

### 3.2 结果（test 宏 F1，3 seeds）

| 源 | TF-IDF | head-only | LoRA |
|---|---|---|---|
| STACAD fold0 | **.3726**（.3710/.3706/.3761） | .2256（.2365/.2144/.2259） | .3506（.3436/.3505/.3577） |
| Droid fold0 | **.1642**（.1604/.1702/.1620） | .1322（.1240/.1361/.1366） | .1267（.1321/.1241/.1239） |

- STACAD 配对：TF−LoRA **+2.2pt**（三 seed CI 全显著 ＋）；TF−head +14.7pt\*；LoRA−head +12.5pt\*。
- Droid 配对：TF−head +3.2pt（跨零）；TF−LoRA +3.7pt（跨零）；head−LoRA +0.6pt（跨零）——生成器聚类下 CI 宽，无显著差异。
- 参照：Droid TF-IDF .1642 与 round1 P0（.1604/.1562）一致 ✓；STACAD fold0 TF-IDF .3726 低于 round1 5 折均值 .4331（主因：train 30k 子集 vs round1 ~116k）——预算受控所致，不构成新结论。

### 3.3 判读

- **无灾难性崩溃**：STACAD 上 LoRA 达 TF-IDF 的 **94%**（.351/.373）且显著优于 head-only（+12.5pt）⇒ 适配器在 file-held-out 下仍有效；Droid generator-held-out 上三族均弱（≈1.1–1.4×机会），差异不显著——与 E29/E36/v2 的"跨生成器信号弱"结论一致。
- LoRA 外部预算仅 2ep（两源 best 均在上限）⇒ 神经侧数字为**下界**；即便按此口径，STACAD 也未超 TF-IDF。
- 外部结果**独立于 408-task 主表**（不混合、不宣称 SOTA）；TF-IDF 数字为同预算同切分下重跑（非复用旧值）。

## 4 决策门槛核对（指导 §6）

1. **task-held-out 超 TF-IDF .7639？** 否——B 均值 .7461；s2 单 seed .7597 仍低于 .7639（仅"统计不可区分"）。
2. **外部无崩溃？** **STACAD：是**（LoRA=94% TF-IDF，显著优于 head-only）；**Droid：未达标**（三族均低且不显著优于 TF-IDF，LoRA dev→test（.36→.13）gap 大）。
3. **3 seeds 配对区间支持？** 部分（A/B 多处显著；C 的 Droid 均跨零）。
4. **选择独立于 test？** 是（全部 best-dev；每个 test 只评一次）。
5. **单变量诊断可解释性能变化？** 是（A：标识符/注释/字符串三类显著可分解）。

⇒ **门槛条件 1 未满足**：论文主线保持为"后训练差异驱动的来源可读性分析、闭集归因与跨 generator 迁移审计"，**不声称已得到 SOTA 家族归因器**（按指导 §6 预设出口）。

## 5 资源与边界记录

- 无新下载；OMP=MKL=2；磁盘运行期间保持 ≥2.3GB 可用；round1/round2/round3 产物未覆盖。
- 实验目录均含 `config.json/env.json/split_manifest.json/hashes.json` + 日志（features 缓存不入库，哈希登记于各源 `hashes.json`；LoRA states 3×1.2MB 留服务器，不入库）。
- 教训：外部源类别数 ≠ 6 时 `FTModel` 硬编码头必须重建（本轮已修，冒烟拦截）；budget 封顶的"负结果"必须标注为下界。
