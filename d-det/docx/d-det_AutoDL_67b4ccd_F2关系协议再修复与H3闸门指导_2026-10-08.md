# AutoDL 执行指导：F2 关系协议再修复与 H3 闸门

日期：2026-10-08  
前置提交：`67b4ccd`

本轮 F0-B 修复有效，真正的 unseen-member 结果可以保留。F2 M-HO 仍不能作为关系迁移证据：当前评测正对包含 heldout member (h)，负对完全不包含 (h)，模型可以只识别“是否出现 h”。必须修正关系样本后再决定 H3。

## 1. 本轮结果的裁定

| 结果 | 状态 | 解释 |
|---|---|---|
| F0-B `u=0.7967` | **保留** | 真正的 observed-series unseen-member transfer candidate |
| F2 M-HO `q=0.9645` | **冻结为待复核** | 正负标签与 heldout member 身份纠缠，不能称 relation transfer |
| F2 S-HO | **保留为负边界** | 未见 series 迁移未成立，低功效单列 |
| S-resid | **探索性候选** | 尺度控制后仍强，但尚未是因果或 H3 表示 |
| H3 smoke | **仅接口验证** | 参数和梯度接口通过，不产出主结果 |

## 2. F2 M-HO 的具体协议错误

当前代码构造：

\[
P^+_{eval}=(h,m_{seen}),\qquad
P^-_{eval}=(m_{other},m'_{other}).
\]

因此 (h) 只出现在正类。任何包含成员身份的表示都可以得到很高 AUROC；`mismatched_partner` 高和 `cross_task_swap` 高并不能证明同系列关系或任务不相关，只说明分数可能使用了 heldout member 身份。

## 3. 正确的 M-HO 关系协议

对系列 (s) 和 heldout member (h)：

### 3.1 训练集

\[
P^+_{train}=(m_i,m_j),\quad m_i,m_j\in s\setminus\{h\};
\]

\[
P^-_{train}=(m_i,m_j),\quad m_i\in s' ,m_j\in s'',\quad s'\ne s''.
\]

全部训练对只使用 train tasks。正、负对数量、语言、长度和规模按预注册匹配。

### 3.2 评测集

正类和负类必须都含同一个 (h)：

\[
P^+_{eval}=(h,m_{seen}),\qquad
P^-_{eval}=(h,m_{other}).
\]

其中 (m_{seen}\in s\setminus\{h\})，(m_{other}\notin s)。正负 pair 必须在同一 task、同一 mode，并按 partner 的 size、length、style 分层匹配。这样模型不能用“是否出现 h”解决标签，只能利用 (h) 与 partner 的关系。

### 3.3 pair 顺序和方向

当前 (q=[u_i;u_j;|u_i-u_j|;u_i\odot u_j]) 不是完全对称的。下一版必须二选一并预声明：

1. 将 pair canonicalize，按固定 `unit_id` 排序；或
2. 对每个 pair 同时加入 ((i,j)) 与 ((j,i))，并在评测中平均两个方向。

正负 pair 的顺序规则必须完全一致。不能让正类和负类通过位置得到捷径。

## 4. 必须新增的对照

### 4.1 同 h 的伪关系对照

`h` 在正负两类都出现后，新增：

- `partner_swap`：固定 h，替换 partner 为匹配的其他系列成员；
- `task_same`：同一 task 内评估；
- `task_cross`：只改变 task，但保留 h 和 partner 组合；
- `pair_order_swap`：交换顺序后分数应保持不变或按预声明方式平均。

`task_cross` 高不再被解释为“任务无关正面证据”；它只说明 pair relation 对 task 变换的稳健性，需要和同 h 的负类一起看。

### 4.2 独立 P0 关系基线

分别拟合：

1. `q_only`；
2. `P0_pair_only`；
3. `q_plus_P0`；
4. cosine；
5. 20 次以上 label permutation。

主比较固定为每个 fold 的

\[
\Delta_{q-P0}=AUROC(q\_only)-AUROC(P0\_pair\_only).
\]

`q_plus_P0` 只能是联合敏感性，不得充当 P0-only。

## 5. 置换 null 的修复

单次 permutation 得到 0.398 不能作为充分的“回到机会”证据。每个 fold 使用至少 20 个固定 seed：

\[
\{AUC_b^{perm}\}_{b=1}^{20+}.
\]

同时保存原始方向和方向归一化值：

\[
AUC^{norm}=\max(AUC,1-AUC).
\]

报告 null 的均值、标准差、2.5/50/97.5 分位点，以及真实分数在 null 中的 percentile。不能只报告一个低于 0.5 的数。

## 6. M-HO 通过条件

F2 M-HO 只有在以下条件全部满足时才允许标记 `member_holdout_relation_candidate`：

- 正负评测对都含同一 heldout h；
- pair 顺序不泄漏；
- q-only 超过 P0-only 的 paired delta_mean 为正；
- 至少 2 个系列、至少 2 个 heldout member 折方向一致；
- 20+ permutation null 与真实分数分离；
- partner-swap、task-cross 不再只是 h identity detector；
- S-HO 继续单列，不把 M-HO 外推为 unseen-series。

否则判为 `member_identity_or_relation_unresolved`，不得启动 H3 主结果。

## 7. S-resid 和 H3 的安排

S-resid 在尺度控制后仍有明显增益，保留为探索性表示候选。补充以下检查后，才能作为 H3 的候选编码：

- fit-only residualization；
- fit-only whitening/standardization；
- residual norm 和 A norm 匹配；
- heldout member 上的 A-only、raw S+A、S-resid+A；
- S-resid permutation。

H3 当前只能做 smoke，不做论文主结果。F2 正确协议通过后，再按四配置运行：

\[
\mathcal L_D,
\quad \mathcal L_F,
\quad \mathcal L_D+\alpha\mathcal L_F,
\quad \mathcal L_D+\alpha\mathcal L_F+\lambda\max(0,\cos(g_D,g_F)-m).
\]

报告 detection AUROC、unseen-member attribution、S-HO 外推、梯度余弦和 Pareto 曲线；不能只报告联合 loss。

## 8. 新输出目录

不得覆盖 `67b4ccd`：

- `artifacts/f2_member_relation_protocol_fix_2026-10-08/`
- `artifacts/f2_member_relation_permutation_2026-10-08/`
- `artifacts/h3_train_dev_registered_2026-10-08/`（只有 F2 修复通过后才可执行）

每个目录保存 `hypothesis/config/execution_switches/data_role_matrix/metrics/report/commands/logs/git_head/git_status/SHA256SUMS`。旧 test、生成和权重开关保持关闭。

## 9. 当前合法 ACL 表述

截至本轮，可以写：

> 双侧表示在 task-heldout 和真正 observed-series unseen-member 条件下具有来源可读性；残差化表示显示出强探索性增益，但跨 series 迁移和关系归因仍需排除 heldout-member identity shortcut 后再判断。

暂时不能写：

- “F2 已证明跨 generator 关系迁移”；
- “S-A 机制已经成立”；
- “后训练因果已被识别”；
- “H3 已通过”。
