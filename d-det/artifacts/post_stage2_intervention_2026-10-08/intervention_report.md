# 变体迁移 post-stage2 Phase A 干预矩阵执行报告（2026-10-08）

> 《d-det_AutoDL_后续干预与主干探针指导_2026-10-08》执行记录。
> `exploratory_train_dev_only=true`；`test_read=false`；`generation=false`；`source_status=server_reconstruction_only`；不覆盖 stage1b/stage2 产物。

## 0 摘要

- 变换：A1 格式规范化 / A2 注释屏蔽 / A3 字面量屏蔽（敏感性）；A4=not_executed（指导 §3）。
- 全量变换样本 10659（11 成员 train+dev；test 未读取）；失败样本全部保留原文。
- A0 对照：22 折重放与 stage-1b dev 分数 **max|Δ|=0.000e+00**（pass≤1e-9=True）。
- 读出两组：r1=冻结对象 transform/predict；r2=变换文本重训（原规则）。

## 1 变换审计（规则 / 失败率 / hash / token）

| 变换 | fail | compile 破坏 | token 变化 | 字符变化 | 关键计数 |
|---|---|---|---|---|---|
| A1_format_norm | 0/10659 | 0 | -0.20% | -0.64% | 补尾换行 10659；行尾空白行 16473 |
| A2_comments_masked | 0/10659 | 0 | -12.73% | -17.95% | 行注释 41649；删除字符 1706796 |
| A3_literals_masked | 0/10659 | 0 | -13.34% | -18.42% | 字符串 72140；数字 42820；行数变化的样本 2917 |

hash 联合映射与逐样本映射见 `transforms_audit.json`（joint_hash_map_sha256）与 `local/hash_maps/*.jsonl.gz`（本地）。

## 2 dev 汇总（跨折：系列内折等权 → 系列等权；原始 vs 变换）

| 读出 | 原始 | A1·r1 | A1·r2 | A2·r1 | A2·r2 | A3·r1 | A3·r2 |
|---|---|---|---|---|---|---|---|
| tfidf_char | 0.8834 | 0.8834 | 0.8834 | 0.8519 | 0.8636 | 0.7544 | 0.8263 |
| tfidf_word | 0.9044 | 0.9044 | 0.9044 | 0.8564 | 0.8710 | 0.8043 | 0.8703 |
| sem_base | 0.9240 | 0.8060 | 0.8514 | 0.8371 | 0.9021 | 0.8464 | 0.9038 |
| sem_small | 0.9258 | 0.8037 | 0.8495 | 0.8654 | 0.9074 | 0.8304 | 0.8911 |
| style_lr | 0.9257 | 0.7485 | 0.8001 | 0.8806 | 0.9100 | 0.8551 | 0.8749 |
| style_lgb | 0.9401 | 0.7663 | 0.8412 | 0.6491 | 0.9326 | 0.8637 | 0.8910 |
| metadata_only | 0.8815 | 0.6443 | 0.7043 | 0.8440 | 0.8732 | 0.8236 | 0.8327 |
| size_length_only | 0.6800 | 0.6851 | 0.6864 | 0.6857 | 0.6974 | 0.6096 | 0.6648 |
| P0_fusion | 0.9566 | 0.8774 | 0.8966 | 0.8305 | 0.9460 | 0.8920 | 0.9355 |
| P0_equal | 0.9567 | 0.8983 | 0.9062 | 0.8997 | 0.9453 | 0.8879 | 0.9414 |

（r1=冻结对象仅 transform/predict；r2=变换文本上重训。）

## 3 paired delta（变换 − 原始；同一 picks 序列）

### 3.1 跨折均值（AUROC delta，系列内折等权→系列等权）

| 读出 | A1·r1 | A1·r2 | A2·r1 | A2·r2 | A3·r1 | A3·r2 |
|---|---|---|---|---|---|---|
| tfidf_char | +0.0000 | +0.0000 | -0.0315 | -0.0198 | -0.1290 | -0.0571 |
| tfidf_word | +0.0000 | +0.0000 | -0.0481 | -0.0334 | -0.1001 | -0.0341 |
| sem_base | -0.1180 | -0.0726 | -0.0868 | -0.0218 | -0.0776 | -0.0202 |
| sem_small | -0.1221 | -0.0763 | -0.0604 | -0.0185 | -0.0954 | -0.0347 |
| style_lr | -0.1772 | -0.1256 | -0.0451 | -0.0156 | -0.0706 | -0.0508 |
| style_lgb | -0.1738 | -0.0989 | -0.2911 | -0.0076 | -0.0764 | -0.0491 |
| metadata_only | -0.2372 | -0.1772 | -0.0375 | -0.0084 | -0.0579 | -0.0489 |
| size_length_only | +0.0051 | +0.0064 | +0.0056 | +0.0174 | -0.0704 | -0.0152 |
| P0_fusion | -0.0792 | -0.0600 | -0.1261 | -0.0107 | -0.0646 | -0.0211 |
| P0_equal | -0.0584 | -0.0505 | -0.0570 | -0.0114 | -0.0688 | -0.0152 |

### 3.2 P0_fusion 分负集版本（size_mix / size_matched）

| 变换×读出 | size_mix Δ | size_matched Δ |
|---|---|---|
| A1_format_norm·r1 | -0.0745 | -0.0840 |
| A1_format_norm·r2 | -0.0603 | -0.0597 |
| A2_comments_masked·r1 | -0.1245 | -0.1278 |
| A2_comments_masked·r2 | -0.0125 | -0.0088 |
| A3_literals_masked·r1 | -0.0681 | -0.0611 |
| A3_literals_masked·r2 | -0.0238 | -0.0185 |

（全读出×逐折 CI 见 `paired_delta.json`。）

## 4 r2 C 选择分布（重训在 dev 按原规则）

- A1_format_norm: sem_base=0.03×22, sem_small=0.03×18, metadata_only=1.0×17, size_length_only=1.0×13, style_lr=1.0×8, style_lr=0.3×6
- A2_comments_masked: sem_base=0.03×22, metadata_only=1.0×22, sem_small=0.03×19, size_length_only=1.0×13, style_lr=1.0×13, style_lr=0.3×9
- A3_literals_masked: sem_base=0.03×20, sem_small=0.1×15, metadata_only=1.0×15, size_length_only=1.0×12, style_lr=1.0×10, metadata_only=0.03×7

## 5 判读（对照指导 §5 框架）

- 原对象（r1）与重训（r2）的差值反映**词表/表示不匹配**分量：r2−r1 越大，说明原分类器对表面干预的脆弱性越主要来自表示不匹配而非信号消失。
- r1 与 r2 同时大幅下降且控制读出下降更小 → 支持**表面信号依赖**（H-format 方向）。
- 两者仍保持高分 → 干预后仍有稳定可读成分（继续排除混杂，不得直接称 H-content 成立）。
- 仅 code-layout / size-length 控制高分 → 归为混杂诊断，不推进主干矩阵。
- 所有数字为 train/dev 探索结果；不得写作迁移或 test 证据；不按掉分大小后选主结果。

## 6 Phase B 主干可用性

- encoder-decoder：CodeT5-small（sha `968fb0f45e1efc8c…`）/ CodeT5-base（sha `053fbafd36f4011e…`）可用（现有基线）。
- encoder-only：**not_executed** —— 本地无 encoder-only 代码模型权重（unixcoder/codebert 等不存在）；未获下载授权。
- decoder-only：**not_executed** —— 本地 HF 缓存已清理，无 decoder-only 权重（Qwen/DeepSeek/Yi/SmolLM2 均已删除）；未获下载授权。
- 若获单独授权的最小设计已写入 `backbone_availability.json`（2×3：架构 × 固定读出 × 原始/A2）。

## 7 边界与合规

- `test_read=false`（未读取旧 test、未用 test_scores.npz 做任何选择）；`generation=false`；未下载权重。
- 不覆盖 stage1b/stage2 产物；全部新产物位于本目录。
- `source_status=server_reconstruction_only`、`original_bundle_verified=false`、`claims_of_byte_identity=forbidden` 继续保留。
- A3 仅敏感性分析，不得表述为"去风格后的纯代码"；A2 输出不得解读为因果后训练信号。

