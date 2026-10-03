# d-det · ACL 数据集合 v1 第三轮报告：审计修正轮（评测模式 / 统计口径 / 匹配对照）

> 执行规范：`docx/d-det_AutoDL_701d0df审计修正与下一步指导_2026-10-03.md`。
> 脚本：`scripts/dcan_round3_{mlp,ft,stats,shortcut_fix,bn_diag,head_vs_lora}.py`（另有 round2 LoRA dev 曲线存档）。
> 产物：`artifacts/acl_dcan_round3_audit/`（mode_assertions / config / metrics / dev_curves / 带身份压缩预测 / 数据与代码 hash / env / checkpoint 复算 / 正式日志）。
> 定位：**修正轮，不写 SOTA**。round1/round2 报告中的 MLP 相关数字以本报告为准（两份旧报告已加修正横幅；旧数字本体不改）。

## 一句话结论

修复 MLP 评测模式（表示提取切 `eval()`+`inference_mode`、BN 用 running stats、best state 含 buffers）并全量重跑后：**late-fusion .7381±.0043**（旧 .7134）、**full-disentangle .7356±.0021**（旧 .7030），整体 +2.5~3.3pt；**SupCon 在 late-fusion 上为 +1.1pt 方向性趋势（逐 seed 配对 CI 均跨零，证据不足）**，三种 disentangle 开关**无一致正贡献**（互有胜负，含 1 例 seed 级显著负）；机器级复现成立（no-op 两次训练 dev 曲线逐位一致、概率 maxdiff=0.0）。Q1 结论在 train 拟合长度桶、LR 收敛记录、固定 5ep 协议下**收窄**为"优势依赖被联合清洗改变的信号"。LoRA checkpoint 复算逐位一致（fp16 层）。匹配 head-only（.4163±.0112）vs LoRA（.7286±.0140）：**配对差 −31.2pt，三个 seed CI 全部离零** ⇒ 匹配条件下 adapter 显著改善任务留出（§7）。

## 1 修正记录（透明性）

### 1.1 发现的缺陷与两次自检教训

- **评测模式缺陷（外部审计指出）**：round1 四模型与 round2 消融的 MLP（含 BatchNorm1d+Dropout）在 dev/test 表示提取、融合拟合表示提取时未切 `eval()` → 预测带随机性、依赖评估批组成、BN buffers 被评估数据更新。旧数字不符合独立单样本评测，不能用于严格消融比较。
- **自检教训 ①（round2 期间）**：消融初版误删 `CE(head_s)` 项，导致 head_s 随机——当天发现并修复后重跑（已记录在 round2 日志）。
- **自检教训 ②（本轮）**：round3 套件初版漏 `opt.zero_grad()`（梯度跨 batch 累积→训练发散），数字异常后被**用正确循环独立重写的 bn_diag 对照**揭穿；初版整体归档 `mlp_pre_fix_archive/`（含 `NOTE_INVALID.txt`，全部数字作废），修复后全量重跑（20:41:53 完成，427.4s）。**教训：重实现训练循环必须做单步梯度清零与正确循环对照。**

### 1.2 修正口径（对应指导 §2 的 6 条要求；config.json）

1. 每 epoch 开始所有分支/头/adversary 调 `train()`（循环内显式）；2. dev/test 与融合 LR 的 train 表示提取前调 `eval()`+`torch.inference_mode()`；3. best state 保存完整 `state_dict`（含 BN buffers），恢复后先 `eval()`，评估完继续训练前恢复 `train()`；4. **从头训练**修正版（不用旧 checkpoint 重算）；5. **只修模式：dev 选择规则保持原样**（各臂按自身输出 dev 选 epoch；lf/fd 用融合 dev），融合 LR 只在 train 表示上拟合，未引入新协议；6. 同 seed 修正版 8 臂 ×3 seeds + no-op 检查。`cudnn.deterministic=True, benchmark=False`。

### 1.3 必须检查项的证据（mode_assertions.json 全通过）

| 检查项 | 结果 |
|---|---|
| 同输入两次评估一致 | `double_eval_exact_equal=true`，maxdiff **0.0** |
| eval 前后 BN buffers 不变 | `bn_buffers_unchanged_in_eval=true`（train 对照 `changed=true`） |
| 换评估批组成仅数值容差差异 | maxdiff **2.4e-07** |
| task_id 不跨 split | `task_splits_disjoint=true`（另见 split_manifest.json，复核跨 split=0） |
| test 不参与选择 | `test_excluded_from_selection=true`（epoch/融合器均 train/dev） |
| 损失梯度/正对/覆盖记录 | 探针：所有组件非零梯度、`positive_frac=1.0`；train family 覆盖 {claude 821, deepseek 864, gemini 803, llama 893, openai 2296, qwen 880} |

### 1.4 no-op 复现

同 seed 同配置两次完整训练：dev 曲线**逐位相等**、test 融合概率 **maxdiff=0.0**、test F1 A=B=**0.7384**、best_epoch 16/16 ⇒ 模块初始化、批序、损失、dev 规则、预测全链路确定性成立。

## 2 修正版 MLP 套件（8 臂 × 3 seeds，test 宏 F1；本轮未改 dev 规则）

| 臂 | s0 | s1 | s2 | 均值±std | round2/round1 旧值（标注待修正） |
|---|---|---|---|---|---|
| semantic_only | .6994 | .6844 | .7123 | **.6987±.0114** | .688±.013 |
| fingerprint_only | .5912 | .5771 | .5960 | **.5881±.0080** | .559±.002 |
| late_fusion | .7421 | .7321 | .7402 | **.7381±.0043** | .7134±.006 |
| lf_nosupcon | .7279 | .7275 | .7267 | **.7274±.0005** | .7153±.0069 |
| full_disentangle | .7384 | .7332 | .7353 | **.7356±.0021** | .7030 |
| fd_nosemgrl | .7247 | .7210 | .7384 | **.7280±.0075** | .7073 |
| fd_nofpgrl | .7495 | .7380 | .7286 | **.7387±.0085** | .7074 |
| fd_noorth | .7268 | .7268 | .7556 | **.7364±.0136** | .7071 |

观察：①修正后所有臂提升约 +1.5~3.3pt（旧数字偏差见 §4 归因）；②`fd_nofpgrl` 与 `late_fusion` 并列最高（.7387 / .7381，差在噪声范围）；③fd 相对 lf 差异（fp=74 桶）总体接近；④**目标冲突如实记录**：full_disentangle 同时用 `CE(head_s)` 鼓励 z_s 保留 family 信息、又用 family-GRL 抽取其 family 信息——两项目标天然对抗，相关对比仅按"开关移除"解读，不作因果排除。可开关机制共 **4 项**：SupCon、semantic-GRL、fingerprint-GRL（含长度+prompt 簇两个 nuisance 目标，作一组）、交叉协方差（orth）。

## 3 配对统计（任务级 bootstrap，B=2000，408 test 任务；`stats_paired_bootstrap.json`）

| 对比 | 均值差 | 逐 seed 差值 [CI95] |
|---|---|---|
| late_fusion − lf_nosupcon | **+0.0107** | s0 +0.0142 [−0.0052,+0.0323]；s1 +0.0046 [−0.0166,+0.0263]；s2 +0.0135 [−0.0065,+0.0324] |
| full_disentangle − fd_nosemgrl | +0.0076 | s0 +0.0137 [−0.0025,+0.0308]；s1 +0.0122 [−0.0054,+0.0296]；s2 −0.0031 [−0.0183,+0.0120] |
| full_disentangle − fd_nofpgrl | −0.0031 | s0 −0.0111 [−0.0292,+0.0069]；s1 −0.0047 [−0.0227,+0.0147]；s2 +0.0067 [−0.0101,+0.0231] |
| full_disentangle − fd_noorth | −0.0008 | s0 +0.0116 [−0.0020,+0.0244]；s1 +0.0064 [−0.0127,+0.0245]；**s2 −0.0203 [−0.0361,−0.0051] *** |

**判读（不夸大）**：SupCon 三 seed 同向为正（+1.1pt 均值），但每个 seed 的 CI 均跨零——记为**方向性趋势、证据不足**；sem-GRL 2/3 seed 弱正但同样跨零；fp-GRL 与 orth 方向混合，orth 在 s2 显著为负 ⇒ **四项开关在本设置下均无一致正贡献；按指导口径，此规模不足以做因果排除**（不写"机制无效"）。

## 4 归因：评测模式差异的上界（bn_diag.json，late_fusion s0，正确循环复训）

- A 标准修正口径（running stats / dropout off）**.7246**；D 仅 BN 用批统计 .7379；B 复刻旧缺陷近似（train 模式全开）.7258；C BN 在 train 上重校准后按 A 评估 .7290。
- 解读：BN 口径 A↔D 差 **1.3pt**；dropout D↔B 0.2pt；重校准只能回收部分。**旧结果不是崩盘，但偏差与批组成相关——必须以本报告数字为准**；旧值偏差（如 lf 旧 .7134 vs 修正 .7381）含 BN+dropout+best-epoch 选择偏移。

## 5 Q1 口径修正（shortcuts_fix.json；指导 §4 要求的改写）

- **长度桶**：边界由 **train** 拟合 `[1264, 2033, 3049]` 后应用于 test（修正旧实现用 test 长度算分位的口径不一致）。
- **LR 收敛记录**：metadata_only n_iter 51、codeT5_raw 210、codeT5_center 31、codeT5_center_std 147，全部 `converged=true`；复算宏 F1：.4753 / .4844 / .4522 / .6376（后两者为任务内中心化/标准化，**转导**口径）。
- **filtered TF-IDF**：固定 5 epoch、仅记录 dev、**不恢复 best**（与 round2 一致），并明确声明与 raw/center 的"单次拟合"不属同类协议。
- **报告改写（采用指导给定表述）**："联合匿名化、去注释/字符串和空白归一化后，Macro-F1 从 .7639 降至 .4572；说明当前性能依赖被这些处理改变的信号，**尚不能区分标识符、注释、字面量、格式或任务混淆各自贡献**。"
- 附：分桶观察（探索性）——metadata/filtered 在 q4（最长）acc 明显更高（.6455 / .5979 vs q1–q3 ≈ .39–.55），提示长代码上信号构成不同；**不做因果结论**。单变换诊断（标识符/字符串/注释/空白各一项）**未运行**，列为下一步。

## 6 LoRA checkpoint 复算校验（checkpoint_recompute.json）

- 基座 `checkpoints/codet5-base/pytorch_model.bin` sha256 `053fbafd…e13c`；LoRA r8/α16/dropout .1（q,v，FEATURE_EXTRACTION）；tokenizer=本目录；截断 384+128；family 顺序与 round1 相同。
- 加载 `runs/acl_dcan_round2/best_lora.pt`（seed=0；**ckpt 未保存 best_dev 字段**，round2 原样）复算 test 概率：与 round2 保存的 `lora_s0_probs` **fp16 层逐位一致**（`exact_equal=true`），float32 复算 maxdiff **2.44e-4**（=fp16 量化级）；复算宏 F1 **.7088** = round2 s0 读数 ⇒ **LoRA 三 seed 数字（.7088/.7372/.7399，均值 .7286±.0140）有效**。
- LoRA dev 曲线已存档（`dev_curves_round2_lora.json`）：best dev .7285@12ep / .7283@11ep / .7271@9ep——按指导口径：**只能写"seed0 最优落在 12ep 上限"**，不得写"三 seed 均未平台化"；是否加预算待资源决策。

## 7 匹配条件 head-only vs LoRA（本轮补齐的公平对照）

- 协议（`head_only/{metrics.json,predictions.npz,predictions_meta.json}`）：与 round2 LoRA 完全同 encoder（冻结 CodeT5-base）/ tokenize 512=384+128 / 线性分类头 `Linear(768,6)` / CE / 任务批采样 / head 学习率 1e-3（wd 1e-4）/ bs16×accum2 / ≤12ep、patience3 / seeds [0,1,2]；**唯一差异 = adapter 开关**（LoRA r8/α16 关闭）。
- 结果（test 宏 F1）：**head-only .4190 / .4285 / .4014 → .4163±.0112**（dev best .4080–.4389 @ ep8–12）；LoRA .7088 / .7372 / .7399 → **.7286±.0140**。
- 逐 seed 配对（任务级 bootstrap，B=2000）：s0 **−0.2898 [−0.3227,−0.2562]**；s1 **−0.3087 [−0.3386,−0.2799]**；s2 **−0.3384 [−0.3690,−0.3107]**；**均值 −31.2pt，三个 seed CI 全部离零且为负 ⇒ 匹配条件下 adapter 显著改善任务留出**。
- 口径说明（防误读）：① head-only 线性头收敛快、dev 最优落在 ep8–12，但 round2 Q1 中同特征 sklearn 探针（.4844）作参考上界，继续加预算的预期提升有限；② round2 报告里"head-only .6881"实为 **round1 MLP（2 层+BN+SupCon、40ep、任务批）**，不是本轮匹配线性头——本轮修正后的同特征多层对照为 semantic_only **.6987±.0114**（§2）；③ **差值读数 = "adapter 提供的编码器适配能力"**（LoRA 训练含 enc lr 3e-4，head 双方同为 1e-3）；④ 即便匹配条件下 adapter 把任务留出从 .42 提到 .73，**仍低于同切分 TF-IDF .7639**（−3.5pt）⇒ 不改变"尚无 SOTA 证据"的结论。

## 8 四问答复（对应指导文末）

1. **修复评测后 late-fusion/消融差值是否仍成立？** —— 部分修正：数值普涨（lf .7381 / fd .7356）；**SupCon 由"噪声级"改判为 +1.1pt 方向性趋势（CI 跨零、证据不足）**；sem-GRL / fp-GRL / orth **无一致正贡献**（orth 有 1 例显著负）。不据 3-seed 均值单独宣称"有无贡献"。
2. **匹配条件下 adapter 是否改善任务留出？** —— **是（显著）**：同 encoder/tokenize/线性头/CE/采样/head-lr/预算下，仅关 adapter ⇒ head-only .4163±.0112，LoRA .7286±.0140，配对差 **−31.2pt，三 seed CI 全离零**（§7）；但 .7286 仍 < TF-IDF .7639，且无 file/generator-held-out 同步证据。
3. **联合清洗结论是否收窄？** —— 是（§5）：按指导给定表述改写；未运行单变换诊断，不做机制归因。
4. **是否有证据与资源进入 generator-held-out 方法实验？** —— **否**。本轮未运行任何新的外部评测；维持 round1 压力测试（Droid .1604/.1562）与 STACAD 文件级 .4331 不变；**不能用未运行的外部评测宣称 SOTA**。下一步建议：先做 §5 的单变换诊断与（可选）LoRA 预算决策，再设计 generator-held-out 方法实验。

## 9 边界与产物清单

- 边界：test 不参与任何选择；不新增下载；OMP=MKL=2；磁盘运行期间 ≥2.6GB 可用；`runs/acl_dcan_round2/best_lora.pt` 仍为唯一 checkpoint；`mlp_pre_fix_archive/` 为**无效归档**。
- 产物：`artifacts/acl_dcan_round3_audit/{config.json, env.json, split_manifest.json, mode_assertions.json, mlp/{metrics.json,dev_curves.json,predictions.npz}, mlp_pre_fix_archive/, stats_paired_bootstrap.json, shortcuts_fix.json, bn_diag.json, checkpoint_recompute.json, dev_curves_round2_lora.json, head_only/{metrics.json,predictions.npz,predictions_meta.json}, head_vs_lora.json, hashes.json, README.md, logs/}`。
- 旧产物保留：round1/round2 目录与报告未覆盖（仅加修正横幅）。
