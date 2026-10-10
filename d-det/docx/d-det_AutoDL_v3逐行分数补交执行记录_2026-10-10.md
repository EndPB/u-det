# d-det AutoDL 执行记录：v3 逐行分数数组最小补交（2026-10-10）

## 0. 上游与授权范围

- 指导文档 §14 / ACL §56：本机已用补交的三件 feature 文件覆盖并**在本机重跑 canonical C0**——
  dev row 指标差 `9.22e-6`、task-macro 差 `0`、逐折 dev row 最大差 `5.98e-5`（均 < 1e-3）；
  成员序/行序/词表/行集合 11/11 一致。裁定仍为 `cross_side_score_pending`：**服务器只交付了
  digest、未交付逐行分数数组**，无法计算 `row_score_max_abs`。所需唯一材料 =
  “服务器提供 11 折 train/dev 的压缩逐行分数数组”，不需要 test、权重或原始语料。
- 开关：train/dev only；`test_read=false`；`generation=false`；`weights_downloaded=false`；
  `code_execution=false`（本轮未重跑任何计算，仅复制、验证与登记 v3 canonical 运行既有的分数数组）。

## 1. 本轮动作

1. **补交 22 个逐行分数 npz**（dev×11 + inner×11，共 6.6 MB）：
   目录 `d-det/artifacts/code_conditioned_v3_scorerow_handoff_2026-10-10/`。
   每个文件含 `ev_rows`（10,659 行 records 原序索引，**join key**）、`y`、`taskpos`、
   `fused`、四组件 `s_*`，以及 train 侧 `t_rows`/`t_fused`/`t_s_*`（fit-row 索引序）。
   eval 行序 = taskpos 排序序（块 = ASCII 排序 eval 成员块）——与 canonical C0
   `score_digests.json` 的 digest 约定一致。
2. **digest 级验证（286/286 全过）**：22 文件的全部 `fused`、`s_*`、`t_s_*` 数组
   sha256（float64 C-contiguous 字节）与 `score_digests.json` 逐位一致；
   `fused_stats`（min/max/mean/std）亦一致。
3. **登记材料**：`scorerow_manifest.json`（逐文件 SHA/字节、行序约定、验证计数、
   runtime=Py3.12.14/NumPy2.2.6/sklearn1.9.1/OMP8、gate 说明）、`SHA256SUMS.txt`（25 文件）、
   `compare_template.py`（按 ev_rows join 的便利比对模板，冒烟测试通过：同文件对同为 0.0）、
   `README.md`（用法与闸门）。
4. 未修改任何特征、模型、协议或运行结果；未新跑 C0。

## 2. 下一步（待本机执行）

本机以 `ev_rows` 为 join key 对齐两侧数组，计算：
- 逐行分数最大绝对差（`row_score_max_abs`，fused 为主、四组件供参考）；
- 指标绝对差（`metric_abs`，row-level 与 task-macro）。
两者 **均 ≤ 1e-3** 即关闭闸门（`aligned`），之后才讨论是否在统一规格下重跑 C1–C3；
在此之前 C1–C3 继续停止，开关保持不变。

## 3. 交付物

- `code_conditioned_v3_scorerow_handoff_2026-10-10/`（22 npz + README +
  scorerow_manifest.json + compare_template.py + SHA256SUMS.txt）；
- 本记录；指导 §14 / ACL §56 两份文档随本提交同步。
