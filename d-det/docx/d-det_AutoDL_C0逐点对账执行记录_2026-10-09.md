# d-det AutoDL 执行记录：C0 逐点对账（2026-10-09）

执行方：AutoDL 服务器（/root/autodl-tmp/u-det）。
上游：指导文档 §8「2968ff9 后的唯一补充动作：C0 逐点对账」、ACL §50.1。
开关：train/dev only；test_read=false；generation=false；weights_downloaded=false；
code_execution=false。提交：见文末（已推送）。

---

## 1. 状态

**已完成一次有限对账。判定 `not_aligned`，且根因超出预注册假设（非单纯浮点或规格差异）：
本机 `code_conditioned_p0_reference_2026-10-09` 与本轮 C0 的折设计不同。**

- 逐折 max|Δfused| = 1.05–3.61（预注册容差 1e-3 远未达到）；
- 仅 CL 四折为同设计对比：r(fused)=0.90–0.92，row-level 我方 .9551 vs 本机 .9278
  （服务器 C0 在该子集逐折高 +0.9..+3.9pt）；
- C1–C3 以“本机强 P0（共同行）”重算后仍全部显著为负（dTM −4.7..−7.5pt），
  正式状态 `code_conditioned_increment_not_observed`（相对两种 P0 基线均成立）。

## 2. 结构性发现（v1 对账触发的排查，v2 修正）

本机包 11 个 `p0_scores.npz` = 同一 11 成员的 LOO 折（train=其余 10 成员，7980 行；
eval=heldout(171 正) + 排除集；task/y 数组、行数、哈希均与清单一致），但：

1. **目录名 ↔ 真实 heldout 存在置换**（用 4 模型 char 网格识别每文件 train 缺失成员，
   11/11 精确复现，证据在 `structure_findings.json`）：
   `Q-1.5B→Q-7B`、`Q-14B→DS-1.3b`、`Q-32B→DS-33b`、`Q-7B→DS-6.7b`、
   `DS-1.3b→Q-1.5B`、`DS-33b→Q-14B`、`DS-6.7b→Q-32B`；CL 四个目录恒等。
2. **“同系列排除”划分与本轮及语料 `series` 字段都不同**（两条独立证据：eval 排除集
   + train 侧 linear 块均值符号）：
   - 本机 v2 包：`{CL4} / {Q-1.5B, Q-14B, Q-32B} / {Q-7B, DS-1.3b, DS-33b, DS-6.7b}`
   - 服务器/语料：`{CL4} / {Q4} / {DS3}`
3. 后果：只有 CL 四折两侧同设计；Q/DS 折 eval 集不同（部分成员块仅在一侧）。
   v1 按目录名配对的“整体低相关”（r 0.23–0.61）主要由该错位造成；修正配对后
   r=0.73–0.92，但**仍不达 1e-3**——残余为第二层规格差异（与 §50.1 预判的
   semantic 输入块/lexical 参数一致，无法再反推唯一规格）。

## 3. 输入一致性（已确认的部分）

- 参考包哈希：11/11 npz、4/4 static_preflight 与清单一致；
- 静态代理逐字段对比：10,659/10,659 行 `solution_sha256` 相同、`static_proxy` 深比较
  全等 → 行集与文本一致；
- 融合公式 `p0 = mean_k z_k(train)` 逐位重构（max|diff|=0.0）；
- complete/instruct 模式假设已排除（complete 更差：semantic r .14-.23、AUROC .72-.84）。

## 4. v2 修正对比（共同成员块，1368 行/折；Q-7B 折 855）

| heldout | max\|Δ\| | r | row 我方/本机 | tm 我方/本机 |
|---|---|---|---|---|
| Q-1.5B | 2.005 | .889 | .9709/.9249 | .9825/.9307 |
| Q-14B | 2.086 | .820 | .8453/.9083 | .8663/.9114 |
| Q-32B | 2.049 | .883 | .9380/.9220 | .9657/.9449 |
| Q-7B | 2.583 | .306 | .7531/.9154 | .7120/.9488 |
| CL-13b/34b/70b/7b | 1.05–1.11 | .90–.92 | 见 p0_alignment_v2 | 见 p0_alignment_v2 |
| DS-1.3b/33b/6.7b | 1.96–3.61 | .73–.84 | .9804/.9801 等 | .9841/.9841 等 |

均值 row .9340/.9391；仅 CL 折 .9551/.9278。

## 5. C1–C3 重算（vs 本机强 P0 共同行；paired task-cluster bootstrap，B=500）

| 候选 | dTM vs 本机P0 | dTM vs 服务器C0（同口径） |
|---|---|---|
| c1.code_only_mlp | −4.68pt | −3.66pt |
| c1.full_mlp | −5.09pt | −4.06pt |
| c1.ht_hy_mlp | −4.88pt | −3.85pt |
| c1.code_only_linear | −5.54pt | −4.51pt |
| c1.full_linear | −7.54pt | −6.51pt |
| c2.s_c2 | −5.04pt | −4.02pt |
| c3.s_orig | −5.43pt | −4.41pt |
| c1.psi_only_mlp（噪声对照） | −36.58pt | −35.55pt |

方向、量级与 §50/2968ff9 记录一致；候选训练不依赖基线，比较已在统一 P0 上重算，
无需重训。C4 保持 `not_executed`。

## 6. 建议下一步（二选一，建议 A）

- **A（本机重建参考包，推荐）**：按统一规格重建 11 折 `p0_scores.npz`
  （目录名=heldout；划分 {CL4}/{Q4}/{DS3}；给规格清单：semantic 输入块是否拼接
  style/meta/size、词法 min_df/max_features/solver/seed、融合权重与种子）；服务器
  现有脚本可在 5 分钟内按同规格重跑并复核对账。
- **B（服务器复现本机设计）**：提供 v2 包的 series-map 与组件规格（或原脚本），
  服务器按同设计重跑 C0 后逐点对账。
- 同时请澄清：v2 系列的 `{Q-7B, DS3}` 合并与目录名 7-循环置换是否为有意设计
  （`family_is_confirmed=false` 背景下的另一次系列定义）。

## 7. 产物与哈希

- 对账产物目录：`d-det/artifacts/code_conditioned_c0_alignment_2026-10-09/`
  （README、hypothesis 预注册、report 主报告、structure_findings.json、
  v1/v2 的 alignment/deltas/CSV、static_proxies_compare、scripts/、logs/、SHA256SUMS）。
- 参考包目录（本机提供，未改动）：`code_conditioned_p0_reference_2026-10-09/`、
  `code_conditioned_static_preflight_reference_2026-10-09/`。
- 仓库脚本：`scripts/cc_c0_align.py`、`scripts/cc_c0_align2.py`、`scripts/cc_mode_probe.py`。
- 提交：见仓库 HEAD（本记录与产物一并提交推送）。
