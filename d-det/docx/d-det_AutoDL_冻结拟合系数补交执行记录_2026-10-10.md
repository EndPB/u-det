# d-det AutoDL 执行记录：v3 冻结拟合系数补交（2026-10-10）

## 0. 上游与授权范围

- 指导文档 §15 / ACL §57：本机已用补交的 22 个逐行分数文件完成严格对账——行键/标签/
  taskpos/train 行索引 22/22 一致、服务器 286/286 digest 验证通过；**dev fused 逐行
  最大差 `0.007275`、inner `0.016595`（> 1e-3）**，差异集中在 **semantic/LogisticRegression
  数值路径**；裁定 `not_aligned`，C1–C3 继续停止。严格复现的合法路径：
  （a）完全相同的 Linux/Python/BLAS/solver 环境，或（b）**冻结并交换拟合系数**；
  不再补传 test、权重或原始语料。
- 本轮执行**路径（b）**：导出并交付 canonical C0 两个 LR 组件的全部冻结拟合结果 +
  四组件 z 统计。开关：train/dev only；`test_read=false`；`generation=false`；
  `weights_downloaded=false`；`code_execution=false`（未读取 test、未生成、未下载权重、
  未执行被评估代码；仅在本机重拟合以导出参数并做逐位验证）。

## 1. 本轮动作

1. **导出冻结系数**（`scripts/cc_export_frozen_coeffs.py`，OMP=8，10s，22/22 折）：
   - semantic：`StandardScaler(mean_, scale_) + LogisticRegression(coef_, intercept_)`
     （512-d；**coef/intercept 保持 float32 原生精度**）；
   - style_meta：同结构（105-d = style 92 + meta 10 + sizelen[:3]）；
   - z 统计 mu/sd（四组件）：按 canonical run 的**原生 dtype 路径**计算
     （mu=mean(train scores)、sd=max(std,1e-8)；semantic 的 train 分数为 float32，
     这一细节是逐位复现的关键）；
   - 每折导出 `coeffs_{protocol}_fold*.npz`（含 ev_rows/t_rows）。
2. **逐位验证（硬断言，全部通过）**：22/22 折 refit semantic/style_meta 的 eval+train
   分数、以及用交付组件数组 + 冻结 z 统计重构的 fused，与 canonical 交付**逐位一致
   （max|Δ| = 0.0）**；模板独立复跑（OMP=8）22/22 折同样全 0.0。
3. **交付物**：新目录 `d-det/artifacts/code_conditioned_v3_frozen_coeffs_handoff_2026-10-10/`
   （22 系数 npz + `frozen_coeffs_manifest.json`（逐文件 SHA + 每折验证记录 + 约定）+
   `apply_frozen_coeffs_template.py` + `export_frozen_coeffs.py` + `README.md` +
   `SHA256SUMS.txt`；总计 704 KB）。
4. 未新跑 C0；未改动协议、特征或任何既有交付。

## 2. 使用方式（本机）

用模板重建 `StandardScaler`/`LogisticRegression` 对象并**原样赋值**冻结参数（保留
存储 dtype，勿强转 float64），再执行 `transform()/decision_function()` —— 即得与服务器
逐位一致（同线程数下）的 semantic/style_meta 分数；融合使用冻结 mu/sd：
`fused = mean_c((s_c − mu_c)/sd_c)`。建议 `OMP_NUM_THREADS=8`（其它线程数引入
~1e-6 级 float32 rounding 差，仍 ≪1e-3）。

## 3. 预期与下一步

- 替换后本机 fused vs 服务器 fused 残余 = char/word SGD 数值差 + rounding；
  若词法残差仍 > 1e-3，可按需追加导出词法 SGD 系数（float32，约 70 MB/协议）。
- 待本机重算：`row_score_max_abs` 与 `metric_abs` 双 ≤ 1e-3 才关闭闸门；在此之前
  C1–C3 继续停止，开关保持不变。

## 4. 交付物清单

- `code_conditioned_v3_frozen_coeffs_handoff_2026-10-10/`（见 §1.3）；
- `scripts/cc_export_frozen_coeffs.py`（生成脚本）；
- 本记录；指导 §15 / ACL §57 两份文档随本提交同步。
