# 主线 C0–C3 一次性批次报告（2026-10-10）

- **判定：`code_conditioned_increment_not_observed`（canonical 复核）** —— C1/C2/C3 全部变体相对冻结 C0 无正增量（−2.70 ～ −4.41pt，≤1/11 折为正）
- 证据等级：**diagnostic**（train/dev only；test 未读；无生成、无权重下载、无样本代码执行）
- 对齐口径：`aligned_under_declared_tolerance`（跨平台微小逐行差异按已声明容差接受，指导 2026-10-10）
- 重启说明：首次尝试 `14:22Z` 被服务器重启打断（日志封存 `logs_pre_restart_2026-10-10T1422Z/`）；重启后 `14:26:34Z` 从头执行，`14:35:38Z` `BATCH_DONE`，全部 job 一次跑完
- 运行 HEAD：`ba2f250`（批次脚本与产物随交付提交；C0 冻结产物运行基准 `a65158f`）

## 1. C0 baseline（冻结 + 批次内重验）

- 来源：`code_conditioned_fresh_c0_canonical_2026-10-10`（成员序 canonical、词法折内拟合、行级映射已审计）
- **批次内 digest 重验：198/198 通过**（dev+inner × 11 折 × 9 数组，float64 sha256 对照 `score_digests.json`）
- dev row **.9442** / task-macro **.9585**；inner .9491 / .9666

## 2. C1 题面条件化（5 视图 × {linear, mlp}；对照 C0；11 折）

| view | head | row | tm | ΔTM vs C0 [CI95] | 正折 |
|---|---|---:|---:|---|---:|
| code_only | linear | .9132 | .9282 | −.0301 [−.0370,−.0229] | 1/11 |
| code_only | **mlp** | **.9212** | **.9315** | **−.0270** [−.0340,−.0207] | 1/11 |
| prompt_only | linear/mlp | .5000 | .5000 | −.4587（构造性无信息） | 0/11 |
| ht_hy | linear | .8759 | .9310 | −.0275 [−.0341,−.0210] | 1/11 |
| ht_hy | mlp | .9181 | .9285 | −.0299 [−.0371,−.0231] | 1/11 |
| full | linear | .8603 | .9142 | −.0441 [−.0526,−.0362] | 0/11 |
| full | mlp | .9138 | .9279 | −.0304 [−.0378,−.0231] | 1/11 |
| psi_only | linear | .5580 | .5993 | −.3596 | 0/11 |
| psi_only | mlp | .5530 | .5823 | −.3761 | 0/11 |

- **full − code_only（mlp，任务内排序）：ΔTM −.0034，CI [−.0078, +.0015]** —— 题面条件化无收益。

## 3. C2 静态代理辅助（full-mlp + λv=0.1；8 代理，parse_ok 零方差排除）

- C2 ：row .9138 / tm **.9290**；**vs C0 −.0295 [−.0369,−.0217]**（1/11 正折）
- **vs C1 full-mlp：ΔTM +.0009 [−.0006,+.0025]** —— 辅助损失空效应
- 长度/风格残差化后 C2−C0：row −.0485 / tm −.0560（去长度/style 后仍负）

## 4. C3 一致性训练（full-mlp + λinv=0.1；comment/ws/rename p=.2/.3/.4，train-only）

- C3：row .9019 / tm **.9253**；**vs C0 row −.0420 [−.0491,−.0343]、tm −.0331 [−.0402,−.0261]**（1/11 正折）
- 变换统计（10,659 行范围内取样）：comment 2,103→**1,677** 接受（426 no_comment）；whitespace 3,161→**1,076**（2,085 no_change）；rename 4,271→**3,666**（334 ast_changed / 121 reflection / 150 no_safe_candidate）；**接受样本 AST 破坏 = 0、接口改变 = 0**
- 变换后归因保持（transformed eval）：comment AUROC .871（sign agree .920）；whitespace .850（pooled .679、sign .763）；rename .895（sign .956）

## 5. 汇总与判定

- C1 全部视图/头、C2、C3 相对冻结 C0 的 task-macro 增量均为负（−2.70 ～ −4.41pt），最多 1/11 折为正；辅助监督（C2）与一致性/不变性训练（C3）均无正增量，且任务条件化（full−code_only）空效应。
- 与历史非 canonical 运行相比（旧 C1 dTM −4.0~−7.2pt、旧 C3 −4.73pt）：方向一致、幅度略缓，差异源于成员序/行序错配修正后的干净输入。
- 结论：**在 canonical 输入、冻结 C0、既定折与 seed 的干净口径下复现 `code_conditioned_increment_not_observed`**。C1–C3 不再重跑；不通过加容量/温度/损失项补救。
- 边界：全部为 train/dev 开发诊断；不读 test、不生成、不下载权重、不执行样本代码；不得写成后训练因果。

## 6. 交付物与后续

- 本目录：`batch_manifest.json`（预注册）/ `hypothesis.md` / `config.json` / `data_role.json` / `metrics.json` / `c0_verification.json` / `report.md` / `commands.txt` / `env.json` / `git_head.txt` / `SHA256SUMS.txt` / `logs/`（含 `logs_pre_restart_…`）/ `c1|c2|c3/` 子目录（各自 metrics/config/hypothesis/report；`local/` 分数缓存按约定不入库）。
- 脚本：`scripts/cc_c1_prompt_conditioned_mainline_2026-10-10.py`、`cc_c2_static_proxy_mainline_2026-10-10.py`、`cc_c3_invariance_mainline_2026-10-10.py`、`run_mainline_batch_2026-10-10.sh`。
- 后续：H3 数据侧（长度平衡、七语言 parser/变体、骨架重复裁定、probes 重跑）由本机并行推进；闸门通过后把 H3 detection/source/invariance 配置追加到下一次完整批次。
