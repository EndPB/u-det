# code_conditioned_c0_alignment_2026-10-09

C0 逐点对账（§8 唯一补充动作）产物目录。

- `hypothesis.md`：预注册（先于对比写定；容差/判定先写死）。
- `report.md`：**主报告**（v1→v2 的结构性发现 + 修正对比 + 判定 + 建议）。
- `structure_findings.json`：本机包结构识别证据（目录名→heldout 映射、
  槽位→成员顺序、每槽 top-3 匹配）。
- v1（按目录名配对；保留作审计轨迹）：
  `p0_alignment.json`、`p0_alignment_per_fold.csv`、`unified_p0_deltas.json`、
  `static_proxies_compare.json`（输入行集/文本一致性 10,659/10,659 全等）。
- v2（修正配对：按真实 heldout + 共同成员块）：
  `p0_alignment_v2.json`、`p0_alignment_v2_per_fold.csv`、`unified_p0_deltas_v2.json`。
- `scripts/`：本目录内归档的对账脚本副本（仓库 `scripts/cc_c0_align*.py` 同步）。
- `logs/`：两次对账运行日志。
- 判定：`not_aligned`（结构性错位 + 规格残余差异；详见 report.md §1/§6）。

开关：train/dev only; test_read=false; generation=false; weights_downloaded=false;
code_execution=false。
