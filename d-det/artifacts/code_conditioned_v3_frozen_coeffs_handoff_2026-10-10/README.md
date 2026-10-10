# v3 frozen fitted coefficients handoff（冻结拟合系数；2026-10-10）

按指导 §15 / ACL §57 的严格复现路径（b）：**“冻结并交换拟合系数”**。

## 目的

本机与服务器 canonical C0 的残余逐行差（dev fused `7.275e-3`、inner `1.6595e-2`）
集中在 **semantic / LogisticRegression 数值路径**。本包把服务器侧两个 LR 组件的
**全部拟合结果**冻结交付——替换本机重拟合后，semantic、style_meta 与融合的数值路径
由服务器系数决定，残余差只剩 char/word SGD 数值与浮点 rounding。

## 内容

- `coeffs_{protocol}_fold{member}.npz` ×22（dev ×11 + inner ×11）：
  - `sem_mean/sem_scale/sem_coef/sem_intercept`：semantic（512-d；**coef 保持 float32
    原生精度**，与 canonical run 的 dtype 链一致）；
  - `sm_mean/sm_scale/sm_coef/sm_intercept`：style_meta（105-d = style 92 + meta 10 +
    sizelen[:3]）；
  - `mu_*/sd_*`：四组件 z 统计（按 canonical run 的原生 dtype 路径计算，即
    mu=mean(train scores)、sd=max(std,1e-8)）；
  - `ev_rows`/`t_rows`：eval 与 fit 行索引（10,659 行 records 原序）。
- `frozen_coeffs_manifest.json`：逐文件 SHA-256/字节 + **每折验证记录**（refit 分数
  与交付分数逐位一致、fused 重构与交付 fused 逐位一致，全部 `0.0`）+ 约定 + runtime。
- `apply_frozen_coeffs_template.py`：应用模板——重建 `StandardScaler`/
  `LogisticRegression` 对象并赋值冻结参数（**保留存储 dtype，勿强转 float64**）。
- `export_frozen_coeffs.py`：本包生成的 provenance 脚本。
- `SHA256SUMS.txt`。

## 应用配方

```python
# sc = StandardScaler(); sc.mean_ = c["sem_mean"]; sc.scale_ = c["sem_scale"]
# sc.var_ = sc.scale_**2; sc.n_features_in_ = 512
# lr = LogisticRegression(); lr.coef_ = c["sem_coef"].reshape(1,-1)  # float32，勿强转
#   lr.intercept_ = c["sem_intercept"]; lr.classes_ = [0,1]; lr.n_features_in_ = 512
sem = lr.decision_function(sc.transform(hy[ev]))            # 与服务器逐位一致
# 融合（冻结 z 统计）：fused = mean_c((s_c - mu_c)/sd_c)，c 按字母序取四组件
```

建议以 `OMP_NUM_THREADS=8` 应用（与 canonical run 相同）；其它线程数会引入
~1e-6 级 float32 rounding 差（仍 ≪ 1e-3）。

## 验证（已做）

- 导出时（OMP=8）22/22 折断言：refit semantic/style_meta eval+train 分数、fused 重构
  与 canonical 交付**逐位一致（max|Δ|=0.0）**；
- 模板独立复跑（OMP=8）22/22 折：semE=smE=fusedRecon=**0.0**。

## 预期效果与边界

- 替换后本机 fused vs 服务器 fused 的残余 = char/word SGD 数值差 + rounding
  （预计 ≪1e-3 或由词法项主导）；若词法残差仍超阈值，可按需追加导出词法 SGD 系数
  （float32，约 70 MB/协议）。
- 范围：仅 train/dev；无 test、权重、语料或生成资产。
