# d-det_AutoDL：`67b4ccd` F2 关系协议再修复与 H3 闸门 执行记录（2026-10-08）

- 服务器：AutoDL（RTX 3080 Ti；`/root/miniconda3/envs/udet/bin/python`；OMP_NUM_THREADS=8）
- 仓库：`/root/autodl-tmp/u-det`；前置提交 `67b4ccd`
- 指导：`d-det/docx/d-det_AutoDL_67b4ccd_F2关系协议再修复与H3闸门指导_2026-10-08.md`
- 本轮为**再修复轮**：不覆盖 `67b4ccd` 旧结果；全部运行 train/dev；未读 test；无生成。
- 新增目录：`f2_member_relation_protocol_fix_2026-10-08/`、`f2_member_relation_permutation_2026-10-08/`、`h3_train_dev_registered_2026-10-08/`（条件执行）

## 0. 合规开关（三目录均记录）

| 开关 | 值 |
|---|---|
| training_allowed | true |
| generation_allowed | false |
| test_read_allowed | false |
| old_test_reuse | false |
| original_bundle_verified | false |

## 1. F2 M-HO 关系协议再修复（核心）

### 1.1 修复内容（对照指导 §3）

| 旧（67b4ccd 版）问题 | 修复 |
|---|---|
| 评测正对含 h、负对不含 h → 标签与 h 身份纠缠 | **P+=(h,m_seen)、P-=(h,m_other)（双方都含同一 h）** |
| partner 未匹配 | partner 按 **size/length/style 贪心匹配**（无放回） |
| pair 非对称 → 位置可泄漏 | 采用 **§3.3 选项 2：双向平均**（训练对增广反向、评测 0.5·(f(i,j)+f(j,i))） |

**过程记录（重要）**：先实现了选项 1（canonical by model_idx），首跑后发现**位置-标签相关**（部分折 h 位于第一位的比例 正=1.000 vs 负=0.000，gap=1.0；如 DS-1.3b 折），违反 §3.3“不能让正负类通过位置得到捷径”→ 弃用并切换选项 2。canonical 版结果保留为 `local/metrics_f2_fix_canonical_interim.json`（interim 证据）。

### 1.2 v2（双向平均）最终结果（11 折）

| 读出 | 均值 | min | max |
|---|---|---|---|
| **q_only** | **0.8139**（CI95 [0.777, 0.842]） | 0.6681 | 0.8693 |
| P0_pair_only（独立对照） | 0.5648 | 0.5165 | 0.6398 |
| q_plus_P0（联合敏感性） | 0.5659 | | |
| cosine | 0.6227 | | |
| **h_only_probe（构造性）** | **0.5000** | 0.4999 | 0.5001 |
| **partner_only_probe（⚠）** | **0.8667** | | |
| task_cross | 0.8242 | | |

- **主比较 Δ(q_only − P0_pair_only) = +0.2491，member-cluster CI95 [0.224, 0.271]**（fold-paired）
- 置换分离：**11/11 折 真值 > null 97.5 分位**（20 seeds 见 §2）
- 逐折 q：0.839 / 0.828 / 0.849 / 0.869 / 0.862 / 0.836 / 0.822 / 0.835 / 0.743 / 0.668 / 0.803

### 1.3 gate（§6）与警告

- 条件 1（正负都含 h）✓；条件 2（顺序不泄漏，双向平均）✓；条件 3（Δ>0）✓；
  条件 4（≥2 系列 ≥2 折同向：11/11）✓；条件 5（20+ 置换分离：11/11）✓；
  条件 6（h identity 构造性排除：h_only=0.5000；task_cross=0.824 与同 h 负类联合看）✓
- **verdict = `member_holdout_relation_candidate`（字面全过）**
- **⚠ partner 身份警告（执行端加测）**：**partner_only_probe 0.8667 > q_only 0.8139，11/11 折 q 低于 partner 探针（平均 −5.3pt）**——仅凭“partner 成员身份”即可解该任务；“同系列关系”与“partner 个体识别”在此协议下**尚未分解**。已写入 gate 字段与报告显著位置；建议下一轮以成员-平衡/关系残差设计分离。

## 2. 置换 null 修复（§5）

目录：`d-det/artifacts/f2_member_relation_permutation_2026-10-08/`

- 每折 **20 个固定 seed**（seed=20261008+100+b）置换训练标签后重训（双向协议）；
- 完整分布报告：null mean/std/2.5/50/97.5 分位 + **真值 percentile** + AUC_norm；
- 结果：**11/11 折 true_percentile = 100**（真值均为 20 个 null 的最大值之上）；null 分布逐折贴近 0.5 带。
- 单次置换口径（0.398）已升级为完整 null 分布口径。S-HO 仍按 67b4ccd 负边界单列（0.467/0.519/0.705），不外推。

## 3. S-resid 补充检查（§7；H3 候选编码前置）

目录：`d-det/artifacts/h3_train_dev_registered_2026-10-08/`（sres 部分）

- 变体：A_only / SA_full（raw S+A）/ S_resid / **S_resid_normmatch**（S⊥ 全局缩放至 mean‖A‖）/ S_resid_shuffle；twin MLP 同配置。
- **task_dev**：A_only 0.9281 / SA_full 0.9309 / **S_resid 0.9687** / **S_resid_normmatch 0.9694** / shuffle 0.9167。
- **model 5 折（model_dev）**：dResid **+3.7~+4.5pt**（5/5 正）；**dNormmatch 同水平全过（+3.5~+4.7pt，部分折甚至略高）**；dSA_full（raw S+A）**±0.7pt ≈ 零增益**；dShuffle −0.2~−1.0pt（5/5 负）。
- **检查结论**：（a）范数匹配保持 → 排除“范数差”解释；（b）**raw S+A 几乎零增益、只有残差化后才有 +4pt —— 增益是“残差化”本身带来的**（行为规则上，“去表面后的来源信号”而非“S 原信息”）；（c）置换回退 → 依赖实例对应。仍不得写后训练因果；S_resid 状态维持 `residualized_representation_candidate`。

## 4. H3 registered（§7；条件=F2 v2 gate 通过，已触发）

目录：`d-det/artifacts/h3_train_dev_registered_2026-10-08/`

- 四配置 c0–c3（L_D / L_F / L_D+L_F / +cos 惩罚 λ=0.1, m=0.3），共享 encoder MLP(1536→256)；M-HO 11 折（训练排除 h）+ S-HO 3 折。

| 配置 | det AUROC（mean） | rel AUROC（mean [min,max]） | grad cos |
|---|---|---|---|
| c0_L_D | 0.9717 | 0.5031 [0.449,0.537] | — |
| c1_L_F | 0.6556 | **0.7068** [0.583,0.750] | — |
| c2_L_D+L_F | **0.9715** | 0.6922 [0.556,0.749] | — |
| c3_L_D+L_F+cos | 0.9715 | 0.6922 [0.556,0.749] | **0.067**（<0.3 惩罚未激活） |

- **Pareto 点**：(det, rel) = c0 (0.972, 0.503) / c1 (0.656, 0.707) / c2 (0.971, 0.692) —— **联合训练几乎“免费”**：相对 c0 检测不掉（−0.02pt）而归因从 0.50 升至 0.69；相对 c1 归因仅 −1.5pt 而检测从 0.66 回至 0.97。
- **S-HO 外推（低功效）**：rel 0.493 / 0.573 / 0.558 / 0.558（c0–c3）——**未见 series 外推仍接近机会（未通过）**，与 F2 S-HO 负边界一致。
- 梯度余弦 0.067（两损失梯度近乎正交→惩罚未触发，接口验证价值）。
- **限制**：H3 rel 水平（0.69~0.71）低于 F2 的 sgd 读数（0.81）——小 MLP 2ep 欠拟合口径；partner 身份警告同样适用；本运行仅 train/dev，非 test 主结果。

## 5. 结果裁定更新（对照指导 §1 与 §9）

| 结果 | 状态 |
|---|---|
| F0-B u=0.7967 | 保留（真正的 unseen-member transfer candidate） |
| F2 M-HO（67b4ccd 版 0.9645） | **冻结**（标签-身份纠缠；已被 v2 取代为历史记录） |
| F2 v2（0.8139） | 新协议 + 双向顺序；gate 通过（字面）+ partner 身份警告 |
| S-HO | 负边界保留（不外推） |
| S-resid | 探索性候选（补充检查通过项：fit-only、标准化、范数匹配、置换） |
| H3 smoke → registered | 接口验证通过后进入 registered train/dev（本目录） |

合法表述（§9）：双侧表示在 task-heldout 与真正 unseen-member 条件下具有来源可读性；残差化表示显示出强探索性增益；**跨 series 迁移与关系归因仍需排除 partner 身份短径后再判断**；后训练因果未建立；H3 未通过（registered 运行结果见 §4）。
