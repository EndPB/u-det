# C0 逐点对账：预注册（先于对比运行写定）

日期：2026-10-09。上游指导：`d-det_AutoDL_代码条件后训练_已见家族未见生成器指导_2026-10-09.md` §8；
`d-det_ACL总研究总结与可迭代路线_2026-10-06.md` §50.1。

## 范围

唯一下一步动作：把服务器 C0（`code_conditioned_c0_strong_p0_2026-10-09`，dev row-level .9329 /
task-macro .9442）与本机强 P0 逐点参考（`code_conditioned_p0_reference_2026-10-09`，每折
`p0_scores.npz`）对账。不重拟合、不选新规则、不看 test。

## 已先验确认（写预注册时已核实）

- 参考包 11/11 npz 与 4/4 static_preflight 文件 SHA256 全部匹配清单；
- 参考 `p0 = mean_k z_k(component_k)`（train 折内 z），重构 max|diff| = 0.0（bit 级）；
- 双方 fold 结构一致：CL/Q 折 train 7980 / eval 1368 / pos 171；DS 折 train 7980 / eval 1539 / pos 171；
- 唯一已知差异：train/eval 的 ordered key 哈希不同（双方记录行序不同，非集合不同）。

## 对账步骤（∅ 预注册）

1. 行身份映射：把服务器每折 eval 行按 (member, task) 映射到参考 npz 行（参考行序 =
   成员块（members_order 过滤同系列）× 171 任务字典序）；要求 100% 映射成功且 y 数组逐行相等。
2. 计算逐折：max|Δfused|、mean|Δfused|、Pearson/Spearman(服务器 fused, 参考 p0)；
   逐组件相关（semantic↔linear、char↔char、word↔word、style_meta↔style）与各组件 AUROC 对比。
3. 指标对账：逐折 row-level AUROC、task-macro（两边各自数组）；折间均值；配对 task-cluster
   bootstrap（seed 20261009，B=500，共享重采样）给 (服务器 − 参考) 的 Δ 均值与 95% CI。
4. 静态度量对比：双方 `static_proxies.jsonl` 按 (model_id, task_id) 对齐，逐字段相等性审计。

## 预注册判定（先写死，运行后不得更改）

- `aligned_strict`：全部 11 折 100% 行映射 且 每折 max|Δfused| ≤ 1e-3 且 |ΔrowAUROC| ≤ 1e-3
  且 |ΔtaskMacro| ≤ 1e-3。
- `aligned_at_metric_level`：全部 11 折 100% 行映射 且 每折 |ΔrowAUROC| ≤ 1e-3 且
  |ΔtaskMacro| ≤ 1e-3 且每折 Pearson(fused) ≥ 0.999（逐行分数有残余差异，但排序与指标在
  1e-3 容差内不可区分）。
- 其他情形：`not_aligned`；必须列出逐组件差异来源（semantic 输入块 / lexical 规格 / 其他）。

## 统一 P0 下的 C1–C3 重算（无论判定结果均执行）

参考 `p0_scores.npz` 即“统一强 P0”的权威逐行分数。以参考 p0 为基线，对 C1/C2/C3 的
每折逐行分数重算配对 Δ（row-level 与 task-macro，task-cluster bootstrap 共享索引）：

- 若候选仍低于统一 P0：正式状态 `code_conditioned_increment_not_observed`；
- 若比较方向改变（候选 ≥ 统一 P0 且 CI 支持）：进入复核，不得沿用旧差值；
- 同时重算"候选 − 服务器 C0"旧差值作一致性校验（应与已记录数字一致，验证重算管道正确）。

## 备注

- “重新跑 C1–C3”按比较层面执行（候选分数与统一 P0 的配对重算），不重训候选：候选训练
  不依赖 C0 基线分数，重训只会复现同一批分数；若指导方要求重训，可按同一种子重散一次。
- 全程 test_read=false、generation=false、weights_downloaded=false、code_execution=false。
