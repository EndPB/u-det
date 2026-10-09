# C0 逐点对账报告（v2：结构性修正后）

日期：2026-10-09。上游：指导文档 §8（唯一补充动作：C0 逐点对账）、ACL §50.1。
预注册：见同目录 `hypothesis.md`（先于对比写定）。本报告在预注册判定基础上，
额外记录 v1（按目录名配对）暴露出的**结构性根因**，以及 v2（按真实 heldout 配对 +
共同成员块）的修正结果。

## 1. 结论摘要

- **判定：`not_aligned`（未通过预注册容差，且更根本地：本机参考包与本轮 C0 的设计不同）。**
  逐折 max|Δfused| = 1.05–3.61（远大于 1e-3）；即使只看设计完全一致的 CL 四折，
  r(fused)=0.90–0.92、Δrow=+0.9..+3.9pt，仍不满足 1e-3。
- **v1（按目录名同折配对）的整体低相关（r 0.23–0.61）主要由“折错位”造成**：
  本机包 11 个 `p0_scores.npz` 是同一 11 成员的 LOO 折，但：
  1. **目录名 ↔ heldout 成员存在置换**（非 CL 的 7 个成员构成一个 7-循环；CL 4 个恒等）；
  2. **“同系列排除”的划分与本轮/语料都不同**：
     - 本机 v2 包：`{CL4} / {Q-1.5B, Q-14B, Q-32B} / {Q-7B, DS-1.3b, DS-33b, DS-6.7b}`
     - 服务器 C0（及语料 `series` 字段）：`{CL4} / {Q-1.5B, Q-7B, Q-14B, Q-32B} / {DS-1.3b, DS-33b, DS-6.7b}`
  3. 因此 **只有 CL 四折两侧同设计**；其余折的 eval 集不同（部分成员块只出现在一侧）。
- **修正后（按真实 heldout 对齐、共同成员块 1368 行/折）：**
  - 融合分逐折 r = 0.73–0.92；CL 四折 0.90–0.92。
  - 指标（共同行）：两侧均值 row-level 我方 .9340 vs 本机 .9391；**CL 折单独看**：
    我方 .9551 vs 本机 .9278（即在本轮唯一可比的子集上，服务器 C0 高于本机旧强 P0）。
  - 残余差异仍有过大（max|Δ|≥1.05）：说明除结构性错位外，**组件规格确有第二层差异**
    （语义输入块、词法参数、融合以外的处理），与 §50.1 的预判一致。
- **C1–C3 在“本机强 P0（共同行）”上重算**：全部显著为负，量级与此前一致或更大：
  | 候选 | vs 本机P0 dTM | vs 服务器C0 dTM（同口径共同行） |
  |---|---|---|
  | c1.code_only_mlp | −4.68pt | −3.66pt |
  | c1.full_mlp | −5.09pt | −4.06pt |
  | c1.ht_hy_mlp | −4.88pt | −3.85pt |
  | c1.code_only_linear | −5.54pt | −4.51pt |
  | c1.full_linear | −7.54pt | −6.51pt |
  | c2.s_c2 | −5.04pt | −4.02pt |
  | c3.s_orig | −5.43pt | −4.41pt |
  | c1.psi_only_mlp（噪声对照） | −36.58pt | −35.55pt |

  → **代码条件增量未观察到**（`code_conditioned_increment_not_observed`）在“相对两种 P0
  基线”下均成立；C4 保持 `not_executed`。

## 2. 对账执行与方法

1. 校验参考包：11/11 `p0_scores.npz` 与 4/4 static_preflight 文件 SHA256 与清单一致。
2. 静态代理逐字段对比：10,659/10,659 行 `solution_sha256` 相同、`static_proxy` 深比较全等
   → **两侧输入行集/文本完全一致**（审计脚本 `cc_c0_align.py` 产物
   `static_proxies_compare.json`）。
3. 结构核验：本机每目录 `y/task/train_task` 布局为“成员块 × 171 个字典序 dev 任务”，
   行数与 fold_plan 计数一致（1368/1539）。
4. v1 逐点对账（按目录名配对）：低相关 + 折级指标大幅错位 → 触发进一步排查。
5. **v2 修正对账（`cc_c0_align2.py`）**：
   - 用 4 个代表折的 char 模型网格（对全 10,659 行打分）识别每文件 train 缺失成员
     = 真实 heldout（11/11 精确复现下表）；
   - 用全部 11 折 eval 块识别每文件“槽位→成员”顺序（每槽 top-1，唯一性断言通过）；
   - 按真实 heldout 配对，在共同成员块（8×171=1368 行；Q-7B 折 5×171=855 行）上比较。
6. 参考 `p0 = mean_k z_k(component_k)`（train 折内 z）在 v1 已逐位重构（max|diff|=0.0）。

## 3. 本机包结构识别（v2 关键证据）

### 3.1 目录名 → 真实 heldout（train 缺失成员识别，11/11）

| 目录名（本机） | 真实 heldout |
|---|---|
| Qwen2.5-Coder-1.5B-Instruct | **Qwen2.5-Coder-7B-Instruct** |
| Qwen2.5-Coder-14B-Instruct | **deepseek-coder-1.3b-instruct** |
| Qwen2.5-Coder-32B-Instruct | **deepseek-coder-33b-instruct** |
| Qwen2.5-Coder-7B-Instruct | **deepseek-coder-6.7b-instruct** |
| CodeLlama-13b/34b/70b/7b-Instruct-hf | 各自（恒等） |
| deepseek-coder-1.3b-instruct | **Qwen2.5-Coder-1.5B-Instruct** |
| deepseek-coder-33b-instruct | **Qwen2.5-Coder-14B-Instruct** |
| deepseek-coder-6.7b-instruct | **Qwen2.5-Coder-32B-Instruct** |

（识别细节与 top-3 证据在 `structure_findings.json`。）

### 3.2 本机 v2 的“同系列排除”划分（两条独立证据）

- **eval 侧**：每折 negatives = “全部成员 − 该 heldout 所属划分”：
  - h=Q-1.5B（文件 DS-1.3b）：negatives = CL4 + DS3 + Q-7B（Q-14B/Q-32B 不出现）；
  - h=DS-1.3b（文件 Q-14B）：negatives = CL4 + Q-1.5B/Q-14B/Q-32B（Q-7B/DS-33b/DS-6.7b 不出现）；
  - h=Q-7B（文件 Q-1.5B）：negatives = CL4 + Q-1.5B/Q-14B/Q-32B（**DS3 整组不出现**）；
  - h=CL-x：negatives = Q4 + DS3（与服务器一致）。
- **train 侧分值符号**：文件 DS-1.3b（h=Q-1.5B）的 `component_train_linear` 块均值：
  Q-14B/Q-32B 两块 = +0.59/+0.58（正类），Q-7B 块 = −0.75（负类）→ 证实其正类 = P2。

结论：本机划分 = `{CL4} / {Q-1.5B, Q-14B, Q-32B} / {Q-7B, DS-1.3b, DS-33b, DS-6.7b}`。
该划分与语料 `series` 字段（= 服务器划分）不同；`family_is_confirmed=false` 的背景下
属于另一次待澄清的系列定义。

## 4. v2 修正对比结果（共同行）

逐折（按真实 heldout 配对；CL 折为完整同设计对比）：

| heldout | 共同行 | max\|Δfused\| | r(fused) | row 我方/本机 | tm 我方/本机 |
|---|---|---|---|---|---|
| Q-1.5B | 1368 | 2.005 | +0.889 | .9709/.9249 | .9825/.9307 |
| Q-14B | 1368 | 2.086 | +0.820 | .8453/.9083 | .8663/.9114 |
| Q-32B | 1368 | 2.049 | +0.883 | .9380/.9220 | .9657/.9449 |
| Q-7B | 855 | 2.583 | +0.306 | .7531/.9154 | .7120/.9488 |
| CL-13b | 1368 | 1.113 | +0.917 | .9557/.9281 | .9607/.9348 |
| CL-34b | 1368 | 1.050 | +0.902 | .9500/.9112 | .9716/.9332 |
| CL-70b | 1368 | 1.081 | +0.913 | .9379/.9038 | .9632/.9273 |
| CL-7b | 1368 | 1.089 | +0.920 | .9767/.9680 | .9774/.9724 |
| DS-1.3b | 1368 | 2.334 | +0.821 | .9804/.9801 | .9841/.9841 |
| DS-33b | 1368 | 1.955 | +0.844 | .9942/.9851 | .9933/.9883 |
| DS-6.7b | 1368 | 3.607 | +0.729 | .9724/.9833 | .9649/.9808 |

均值：row 我方 .9340 / 本机 .9391；**仅 CL 四折：我方 .9551 / 本机 .9278**。

要点：
- v1 的“整体低相关”更正为“折错位 + 规格差异”的叠加；修正后相关升至 0.73–0.92
  但**仍不达 1e-3 逐点容差**（最大差 1.0–3.6），组件级残余差异是第二层原因（规格）。
- CL 四折（设计完全相同）是本轮唯一严格可比子集：**服务器 C0 逐折均不低于本机 P0**
  （+0.9..+3.9pt），且相关 0.90–0.92 —— 说明服务器实现并未劣化，§50.1 担心的
  “服务器低于本机”在共同行上不成立。
- Q/DS 折的差异（如 Q-14B −6.3pt、Q-7B −16.2pt）叠加了设计差异（目标系列不同：
  本机 P2 不含 Q-7B；本机 P3 含 Q-7B）与规格差异，不能单独归因。

## 5. 逐组件相关（v1，按目录名配对；供规格线索）

v1 的组件相关在 CL 折上：semantic↔linear r≈0.46–0.49、char↔char 0.61–0.65、
word 0.58–0.62、style 0.14–0.20。修正配对后 char 类相关普遍上升（见
`p0_alignment_v2.json` 的 `per_member`），语义与 style 仍是最弱环节，与 §50.1 的
“semantic 输入块（embedding/style/meta/size 拼接与否）”和“词法参数（min_df/
max_features/solver）”差异判断相容。**由于结构错位未消，规格不能再从这些数字里
反推到唯一解**；若需精确统一规格，应由本机提供规格清单或按统一规格重建参考包。

## 6. 对 C1–C3 的影响与正式状态

- 无论以“服务器当前 C0”还是“本机强 P0（共同行）”为基线，C1–C3 的 task-macro
  差均为显著负值（0/11 折为正的方向一致）；构造性对照 psi-only ≈ −36pt。
- 预注册路径里的两种情况：
  (a) “候选仍低于统一 P0 → 正式 `code_conditioned_increment_not_observed`” ✓ 成立；
  (b) “基线规格改变了比较 → 重跑 C1–C3 比较” —— 已在**比较层面**完成（候选分数
      与统一 P0 的共同行配对重算；候选训练不依赖基线，重训只会复现同一批分数）。
- C4 未获测试执行授权，保持 `not_executed`。

## 7. 建议的下一步（供本机决策）

1. **澄清系列定义**：`public_series_member_ho_v2` 使用 `{Q-7B, DS3}` 合并、
   拆出 `{Q-1.5B, Q-14B, Q-32B}` 与 `{CL4}` 三个划分——请确认这是有意设计
   （family_is_confirmed=false 下的 v2 系列）还是历史版本残留；同时确认目录名
   7-循环置换的来源（成员列表版本差异）。
2. **若要严格 1e-3 对账**：需在一侧重建参考包：
   - 选项 A（推荐，本机重建）：按统一规格重建 11 折 `p0_scores.npz`（目录名=heldout、
     划分 {CL4}/{Q4}/{DS3}），并给出规格清单（语义输入块、lexical 参数、融合、种子）；
   - 选项 B（服务器复现）：本机提供其确切 series-map 与组件规格（或原脚本），
     服务器按同规格重跑 C0（约 5 分钟）后对账。
3. 在规格统一前，C1–C3 的负结果按“相对两种 P0 基线的开发诊断”引用（本报告 §6）。

## 8. 开关与产物

- 开关：train/dev only；test_read=false；generation=false；weights_downloaded=false；
  code_execution=false。
- 产物：本目录（v1 文件保留作审计轨迹：`p0_alignment.json`、`p0_alignment_per_fold.csv`、
  `unified_p0_deltas.json`、`static_proxies_compare.json`；v2：`p0_alignment_v2.json`、
  `p0_alignment_v2_per_fold.csv`、`unified_p0_deltas_v2.json`、`structure_findings.json`）；
  脚本：`scripts/cc_c0_align.py`、`scripts/cc_c0_align2.py`（仓库内）。
- 哈希：见 `SHA256SUMS.txt`；git 状态见 `git_head.txt` / `git_status.txt`。
