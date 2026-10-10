# v3 per-row score handoff（train/dev 逐行分数数组；2026-10-10）

按指导 §14 / ACL §56：本机对账还差的唯一材料 = **11 折 train/dev 逐行分数数组**
（本机明确“只需服务器提供 11 折 train/dev 的压缩逐行分数数组”）。本目录即为此补交。

## 内容

22 个 npz（dev ×11 + inner ×11），每个文件含：

| key | 含义 |
|---|---|
| `ev_rows` | eval 行的全局索引（10,659 行 records 原序）——**对齐用 join key** |
| `y` | eval 行标签（正类 = heldout 成员） |
| `taskpos` | task 在排序后 dev task 列表中的位置（eval 行按此排序） |
| `fused` | 等权 z 融合分（gate 主体） |
| `s_semantic` / `s_char_tfidf` / `s_word_tfidf` / `s_style_meta` | 四组件分数 |
| `t_rows` / `t_fused` / `t_s_*` | train 侧（fit rows）对应数组（fit-row 索引序） |

行序约定：eval 数组 = taskpos 排序序（块 = ASCII 排序的 eval 成员块）；train 数组 =
fit-row 索引序（`t_rows` 给出索引）。与 canonical C0 交付中 `score_digests.json` 的
digest 约定完全一致。

## 验证（已做）

- **286/286 校验通过**：22 个文件内全部 `fused`、`s_*`、`t_s_*` 数组的
  sha256（float64 C-contiguous 字节）与 `score_digests.json` 记录逐位一致；
  `fused_stats`（min/max/mean/std）亦一致（见 `scorerow_manifest.json`）。
- 每个文件的 SHA-256 在 `SHA256SUMS.txt` 与 `scorerow_manifest.json` 中登记。

## 使用方法

1. 用本机 v3 重跑（已覆盖服务器 canonical bundle）的对应折数组与这里逐 fold 对齐；
   **以 `ev_rows` 为 join key** 建立 server↔local 行映射（两侧应产生相同 ev_rows；
   若不相同的行集合，报告交集）。
2. 计算逐行分数最大绝对差（fused；四组件供参考）与指标绝对差
   （row-level、task-macro）。
3. 闸门：`row_score_max_abs ≤ 1e-3` **且** `metric_abs ≤ 1e-3` 才关闭（§13/§14）。
   dev 指标差与逐折最大差已在你侧复算（9.22e-6 / 0 / 5.98e-5），本轮数组提交即用于
   逐行分数闸门的最终判读。

服务器运行环境（解释残余浮点差）：Python 3.12.14 / NumPy 2.2.6 / sklearn 1.9.1 /
OMP_NUM_THREADS=8。附 `compare_template.py`（便利模板，非协议部分）：按 ev_rows
join 后打印逐折 fused/四组件 max|Δ|。

范围：train/dev only；不含 test、权重、生成资产或原始语料。
