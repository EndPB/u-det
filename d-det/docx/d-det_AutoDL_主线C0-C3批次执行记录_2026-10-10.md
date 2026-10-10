# d-det AutoDL 执行记录：主线 C0–C3 一次性批次（2026-10-10）

## 0. 上游与授权

- 指南（主线恢复裁定）：C0–C3 不再等待微小跨平台差异，按已声明容差 `aligned_under_declared_tolerance`，
  **立即在 AutoDL 一次性完整批次**执行 C0–C3；H3 的 `revise_data` 闸门仅限制 H3 主结果。
- 批次预注册：`d-det/artifacts/mainline_c0c3_batch_2026-10-10/batch_manifest.json`
  （data_sha256、code_commit=`ba2f250`、折/seed/资源/输出目录、`test_read=false`）。
- 开关：`test_read=false`（仅 train/dev）；无生成、无权重下载、无样本代码执行。

## 1. 重启经过（本记录的重点运维项）

- 首次尝试 `14:22Z` 启动，随即被**服务器重启**打断（C1 启动瞬间；当时 CUDA 尚不可用，日志显示
  `device=cpu`；中断日志封存 `logs_pre_restart_2026-10-10T1422Z/`）。
- 重启后盘点：HEAD=`ba2f250`、批次脚本完好、无残留结果；清理后 `14:26:34Z` **从头重跑**
  （CUDA 已可用，`device=cuda`），`14:35:38Z` `BATCH_DONE` —— 全部 job 一次跑完（约 9 分钟）。

## 2. 执行与结果（详见批次 report.md / metrics.json）

- **C0 baseline**：冻结 canonical 运行复用 + **批次内 198/198 digest 重验通过**（dev row .9442 / tm .9585）。
- **C1**（5 视图 × {linear,mlp}）：最好 code_only-mlp .9315，仍 **−2.70pt vs C0**；full-mlp −3.04pt；
  full−code_only −0.34pt（CI 跨 0）→ 题面条件化无收益。
- **C2**（λv=.1 静态代理）：.9290，vs C0 **−2.95pt**；vs C1full **+0.09pt**（空效应）；长度/风格残差化后仍 −5.6pt。
- **C3**（λinv=.1 一致性）：.9253，vs C0 **−3.31pt**（1/11 正折）；接受变体 AST/接口零破坏；
  变换后归因保持 comment .871 / whitespace .850 / rename .895。
- 判定：**`code_conditioned_increment_not_observed`（canonical 复核）**；C1–C3 不再重跑。

## 3. 交付物

- 批次目录 `d-det/artifacts/mainline_c0c3_batch_2026-10-10/`（§8 全件 + c0_verification + c1/c2/c3 子目录；
  `local/` 分数缓存不入库）；
- 脚本 4 件（3 个 mainline + driver）；本记录；随本提交推送。
