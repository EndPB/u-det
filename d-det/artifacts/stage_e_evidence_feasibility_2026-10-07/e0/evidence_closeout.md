# E0：证据收尾（2026-10-07，post-hoc audit，只用已存产物）

依据《d-det_AutoDL_N2收尾与独立数据可行性指导_2026-10-07.md》§2。**未重载 test 文本、未重调模型、未改超参**；所有数字来自已提交的 `metrics.json` / `predictions.npz` / 协议文件，重算仅做 argmax 级核对。

## 1 差值对账（原值不覆盖，修正并列）

| issue | 回传 Δ | 点估计差 | bootstrap 差值均值 | 结论 |
|---|---|---|---|---|
| N2-word-vs-fusion | +.0148 | +0.0165 | +0.0148 | resolved |
| N2-centered-vs-fusion | +.0278 | +0.0274 | +0.0278 | resolved |
| N2-ensemble-vs-fusion | +.0663 | +0.0676 | +0.0663 | resolved |
| D1-word-vs-fusion | −.0191 | -0.0188 | -0.0191 | resolved |

**统一解释**：回传表的 Δ 列 = paired task-cluster bootstrap 的**差值均值**（delta_mean）；与“两个点估计相减”是两个不同统计量。点差值逐位自洽（见 `metric_reconciliation.json` 逐行字段：同 test、同 seed 组成、同指标定义、同 label set、n/任务数/顺序 id 哈希/split 哈希均已登记）。500× 重采样频率不是后验概率；N2 有效样本量按 51 个 test task 计。

## 2 mean_ensemble 与 P0 定义

- `mean_ensemble`：5 成员（sem_lr, style_lgb, style_lr, tfidf_char, tfidf_word）概率逐元素等权算术平均（概率空间，非 logits/vote）；类别序=['claude', 'deepseek', 'gemini', 'llama', 'openai', 'qwen']；N2 上由存档预测重算的 max|diff|=3.30e-04（fp16 存储口径）。
- `fusion_lr`：dev-only LR（log-prob 特征、C=1、无样本权重），无双重 dev 选择；N2 在候选 dev=348 行上复算。
- **预声明时间**：`n2_protocol.json`（提交 `7a0d51a`，早于 N2 test 读取 09:05:32Z）在 P0 复算清单中明确列出“fusion_lr=dev-only LR stack；mean_ensemble=等权均值”。“ensemble 优于 fusion”的比较本身为 **post-hoc 观察**，不是预注册假设与 H2 证据；未来强 P0 应在独立数据 dev 上冻结（等权与 fusion 双列）。

## 3 闸门重述（不重跑实验）

| 命题 | 本轮表达 |
|---|---|
| H1 内容可读性 | 内容视图（N2 word .6429 / D1 word .8200）均显著高于 chance(.1667) 与 metadata(.3689)；**未被否定** |
| 普通单样本增量 | 未通过进入方法扩展门槛：点差 +0.0165、CI [-0.0394,+0.0739] 含 0、三 seed 方向不稳 |
| 转导诊断 | centered 单独列（+.0274 点差，CI 含 0）；不替代 inductive 条件 |
| 跨 generator 迁移 | 仅 openai 有三 generator；“普遍性”不支持；H2 不启动 |

原执行器记录：cond1=true、cond2=true（**仅转导**）、cond3=false——原样保留；其读法按上表修正。

## 4 数据卫生与集合选择边界

- exact/ws 重复（0 组跨 split）与 **norm_lex 骨架碰撞**（25 组/66 行）分开记录；剔除是**保守协议**，不自动证明原始评测泄漏；去掉字符串可能抹去常量/IO 规格等语义。
- 六族齐全（280 task）为**条件选择**：改变目标总体；规则仅用数据组成、在 test 读取前预注册（`7a0d51a`）。
- 原 test 行剔除后的子集仍屬已暴露集合；E0/E1 不新增任何 test 读取。

## 5 test 暴露账本（要点）

- P0（10-05，n=1457）→ D1（08:35:45Z，n=1457）→ N2（09:05:32Z，子集 404/51 task）——三次读取、同一底层 test。
- N2 子集 ⊂ D1 test tasks：True；不得改名/换 split 冒充 fresh test。

## 6 产出

- `metric_reconciliation.json`、`p0_definition.json`、`gate_reconciliation.json`、`test_exposure_ledger.json`、本文件；命令/日志/SHA256 同目录。
- 运行耗时 0.1s；生成 UTC 2026-10-07T10:37:20.582321+00:00。

