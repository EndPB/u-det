# 主线 C0–C3 一次性批次（2026-10-10）——预注册

- 上游：`AUTODL_AI_HANDOFF_GUIDE_2026-10-10.md` §7（一次批次执行 C0–C3；相对 C0 报告增量）+ ACL v2.0 §5.5/§9（`aligned_under_declared_tolerance`；`c1_c3 = proceed_in_mainline_batch`）
- 输入冻结：canonical 特征 bundle（`d05f8c89…`）+ fold-fit 词法 canonical C0（`code_conditioned_fresh_c0_canonical_2026-10-10`，dev .9442/.9585、inner .9491/.9666）+ `fold_plan.json` 11 个 member-heldout 折 + 固定 seed
- 开关：`test_read=false`、`generation=false`、`weights_downloaded=false`、`code_execution=false`（未读旧 test；未生成文本；未下载权重；未执行样本代码；只训练小型头）

## 问题（预注册）

在 canonical 主线输入下，C1/C2/C3 相对 C0 的任务内排序增量（dev task-macro Δ 与配对 bootstrap CI）是多少？

## 对照与冻结预算（沿用既定实现，不改动）

- **C0 baseline**：复用 canonical C0 运行（确定性；批次内重验 digest 与 score_digests.json 一致）。
- **C1**：5 视图 {code_only, prompt_only, ht_hy, full, psi_only} × 2 头 {linear(LR C=1), small-MLP(d→256→1, 3 seeds×20ep)}；关键对照 full vs code_only。
- **C2**：C1 full-MLP + λv=0.1 静态代理辅助（8 个有方差代理，mask-free MSE）；对照 C1 full-MLP 与 C0。
- **C3**：C1 full 视图 + λinv=0.1 一致性（comment/whitespace/rename，独立 Bernoulli p=.2/.3/.4，固定 seed；接受变体须 AST 等价 mod rename 且接口 hash 不变）；对照 C0。
- 配对显著性：`delta_bootstrap`（task-cluster，seed 20261009，共享重采样索引）。

## 判定规则（预注册）

- 各配置 vs C0 的 dev task-macro Δ：< 1pt、或折间方向混合（正折 < 6/11）→ 记为无稳定增量（与历史结论一致时需要同向复现）。
- 历史参考（旧输入、仅作追溯）：C1 best code_only-MLP TM .9042（ΔTM −4.0pt）、C2 .8998、C3 .8972 —— 本轮为 canonical 输入下的正式复现，不与旧值混写。

## 输出

`d-det/artifacts/mainline_c0c3_batch_2026-10-10/`（§8 格式：batch_manifest.json / hypothesis.md / config.json / data_role.json / metrics.json / report.md / commands.txt / env.json / git_head.txt / SHA256SUMS.txt / logs/ / c1|c2|c3 各任务产物与 score digests）。
