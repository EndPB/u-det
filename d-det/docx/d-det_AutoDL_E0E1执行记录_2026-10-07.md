# d-det AutoDL 执行记录：E0 证据收尾 + E1 独立数据可行性（2026-10-07）

依据《`d-det/docx/d-det_AutoDL_N2收尾与独立数据可行性指导_2026-10-07.md`》；运行锚点 `9936df1`；
产物目录 `d-det/artifacts/stage_e_evidence_feasibility_2026-10-07/`（e0/ 与 e1/ 各自含日志与 SHA256SUMS）。

## 0 结论速览

- **E0 ✅ 四处差值全部 resolved**：统一原因 = 回传表的 Δ 列是 **paired task-cluster bootstrap 的差值均值**，
  与“两个点估计相减”是不同统计量；两列并列后逐位自洽（未重读 test 文本、未重调模型；仅 argmax 级重算）。
- **E0 P0 定义与预声明时间核对完成**：`mean_ensemble` = 5 成员概率逐元素等权算术平均（非 logits/vote）；
  作为 P0 组件在 N2 读 test 前已声明（`n2_protocol.json`，提交 `7a0d51a`）；“ensemble 优于 fusion”为 post-hoc 观察。
- **E0 闸门重述完成**：H1 可读性未被否定；inductive 增量未通过门槛；转导单独列；跨 generator 不支持 → H2 不启动。
- **E1 = partial**：现成数据只有 openai(AB) 与 meta(LCv2) 两个“≥3 generator”单元；pilot 需要新任务集与新生成
  （未授权）。增量磁盘 ≈15 MB 量级；`training_allowed=false`。

## 1 E0 四处差值对账（原值不覆盖，修正并列）

| issue | 回传 Δ | 点估计差（重算） | bootstrap 差值均值 | 统一口径 | 结论 |
|---|---|---|---|---|---|
| N2 tfidf_word vs fusion | +.0148 | **+.016518** | +.014757 | 同 test（404 行/51 task）、同 3-seed 平均、macro-F1(argmax) | resolved |
| N2 centered vs fusion | +.0278 | **+.027394** | +.027837 | 同上；centered 为确定性 | resolved |
| N2 mean_ensemble vs fusion | +.0663 | **+.067625** | +.066265 | 同上；ensemble 为 5 成员算数平均 | resolved |
| D1 tfidf_word vs fusion | −.0191 | **−.018808** | −.019104 | D1 全量 test（1457 行/408 task） | resolved |

- 原始文件：各 `metrics.json` + `predictions.npz`（sha256 在 `metric_reconciliation.json` 逐行登记）；
  对照口径字段（seed/label set/n/task 数/顺序 example-id 哈希/split 哈希/CI 方法）全部落盘。
- 从 fp16 存档预测重算的 macro-F1 与 `metrics.json` 逐位一致（无 argmax 翻转）。
- **不静默覆盖**：原表（+ .0148 / +.0278 / +.0663 / −.0191）保留；修正列并列展示；都来自**同一统计量家族**：
  点差 = 点估计差；CI 与 frac≤0 来自 500× 配对 task-cluster percentile bootstrap。
- 统计警示：500× 经验重采样频率 ≠ 优于对照的后验概率；N2 有效样本量按 **51 个 test task** 记（不是 404 行）。

## 2 mean_ensemble 定义 / 预声明时间 / 门槛分离

- **定义**（`p0_definition.json`）：$p_{\mathrm{eq}}(f|x)=\frac{1}{M}\sum_{m=1}^{M}p_m(f|x)$，M=5
  （sem_lr, style_lgb, style_lr, tfidf_char, tfidf_word），概率空间逐元素等权平均；类别序 = [claude, deepseek,
  gemini, llama, openai, qwen]；无校准；成员均在 6 类 train 上拟合（无缺类）；N2 上由存档预测重算 max|diff|=3.3e-4（fp16）。
- **fusion_lr**：dev-only LR（log-prob 特征、C=1、max_iter=3000、无样本权重）；无双重 dev 选择；N2 在候选
  dev=348 行上整体复算。
- **预声明时间**：`n2_protocol.json`（提交 `7a0d51a`，早于 N2 test 读取 09:05:32Z）在“P0 复算”清单中明确列出
  “fusion_lr=dev-only LR stack；mean_ensemble=等权均值”——**作为 P0 组件已预声明**；“ensemble 优于 fusion”
  的比较本身为 **post-hoc 观察**，不构成 H2/跨 generator/超越最强 P0 的证据；未来强 P0 候选 = 等权 + fusion 双列，
  在新数据的 train/dev 上冻结。
- **门槛分离**（`gate_reconciliation.json`）：
  - H1 内容可读性：内容视图明显高于 chance(.1667) 与 metadata（N2 .6429 vs .3689；D1 .8200）——**未被否定**；
  - 普通（inductive）增量：点差 +.0165、CI [−.0394,+.0739] 含 0、三 seed 方向不稳 → **未通过进入方法扩展门槛**；
  - 转导诊断：centered +.0274（CI [−.0191,+.0778] 含 0，frac≤0=.13）单独列，不替代 inductive；
  - 跨 generator 迁移：仅 openai 有三 generator → 普遍性不支持，H2 不启动。
  - 原执行器记录（cond1=true / cond2=true 仅转导 / cond3=false）**原样保留**，其读法按上表修正。

## 3 test 暴露账本与数据卫生边界

- 账本（`test_exposure_ledger.json`）：P0（10-05，n=1457/408 task）→ D1（08:35:45Z，n=1457）→
  N2（09:05:32Z，子集 404 行/51 task，**⊂ D1 test**）。同一底层 test；**不得改名/换 split 冒充 fresh test**；
  E0/E1 不新增任何 test 读取。
- exact/ws 重复（跨 split 0 组）与 **norm_lex 骨架碰撞**（25 组/66 行）分开记录：去掉字符串可能抹去常量/IO 规格等语义，
  剔除是**保守协议**，不自动证明原始评测泄漏；剔除实现与原始组在 N1 审计 JSON 中可复核。
- 六族齐全（280 task）为条件选择（改变目标总体）；规则仅用数据组成、在 test 读取前预注册（`7a0d51a`）。

## 4 E1：来源支持 / 同任务覆盖 / 混杂 / 许可 / 新任务可用性（要点）

| dataset | rows | tasks | family/generator | dup audit | test 暴露 | 可独立留出 |
|---|---|---|---|---|---|---|
| h2_authorbench_dcan | 9,498 | 2,715 | 6 vendor 族 / 8 gen（openai=3，其余 1） | exact/ws/lex done | 3 次读取 | no |
| h2_llm_codegen_v2 | 1,512 | 168 | 5 族 / 9 gen（meta=3；google\mistral=2） | not run | partial（pair 22 task） | partial |
| h2_stacad_* | 144,958 | 22,053 | 7 models（非 vendor 分组） | not run | yes | no |
| h2_droid_full_selected | 146,718 | gen-heldout | 32 machine gen | exact 有 | yes | no |
| aicd_t2（numeric） | 1,111,199 | missing | 12 numeric（映射缺失） | exact 有（跨 split 1,848） | yes | no |
| codet_m4/balanced | 500,552 | missing | model 级（含 null 13,587） | not run | yes | no |

- 混杂：AB 与 LCv2 是不同 dataset/任务域；**不得直接拼接**当作“多族支持”的主赛道（可各自作迁移单元）。
- family 语义：gpt-4.x 同 vendor ≠ 已知共享权重谱系；Llama 系列不能凭名字视为唯一同基座对；LCv2 无 base/SFT/DPO 元数据。
- 覆盖细节：AB 每族 generator 数与各 split 行列于 `coverage_matrix.csv`（UTF-8）；10 折 train/dev 正支持与 test task 数
  列于 `dataset_feasibility.json.fold_support`；六族齐全 285→剔除后 280 task。

## 5 pilot 蓝图与出口

- `pilot_design.json`：**status = partial**；需要 ≥3 family × 每族 ≥3 generator 的**同一新任务集**采样：
  现成 triple 仅 **openai（gpt-4.1/4o/4o-mini）** 与 **meta（llama2/llama3/codellama）**；
  google/mistral 各缺 1 个 generator；新任务池缺失。
- 缺口：第 3 个 generator（生成授权）、新任务提示集、闭源 API/预算、license 确认；生成/训练**未请求未执行**。
- 增量磁盘估计：≈300 task × 3 族 × 3 gen ≈ 2,700 输出 ≈ 4 MB 原文 + ~10 MB 索引/缓存（≈15 MB 量级；可复用
  codet5-small / codet5-base / 审计脚本）。
- 功效：`power_unknown`（粗规则：~1,200 task 量级分辨 1pt；需预注册模拟冻结；51 task 的 CI 半宽 ≈4.8pt 实测参照）。
- `training_allowed=false`；`cost_and_generation_authorization=not_requested_here`。

## 6 交付与复现（回传模板 §5）

- commit、目录树、命令、日志、SHA256SUMS：见 `d-det/artifacts/stage_e_evidence_feasibility_2026-10-07/`
  （`commands.txt`、`e0/`、`e1/`、两级 `SHA256SUMS.txt`；提交号见本文件所在提交）。
- 未提交任何权重、代码全文、凭据或冗余缓存；E0/E1 均未新增 test 读取与模型调用。

## 7 唯一下一步

按指导 §4：**E0 完成 + E1 partial → 停止计算**，回传精确缺口（见 §5）与预算；不重复 N2、不加损失、
不启动 H2/H3；新数据生成与外部调用等待明确授权与预注册协议。

---
*生成时间：2026-10-07；记录人：服务器执行 AI。*
