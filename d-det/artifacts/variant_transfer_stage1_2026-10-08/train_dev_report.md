# 变体迁移 stage-1（train/dev）汇报（2026-10-08）

> **source_status = server_reconstruction_only**；original_bundle_verified = false；
> claims_of_byte_identity = forbidden（依据《服务器重构版最小授权与补传豁免》§1）。
> 本报告全部为 **train/dev** 结果：dev 用于预声明选择与诊断，**test 未读取**。

- 运行：22 折（11 heldout × 2 负集版本），runtime 3119s
- 配置 hash：{"execution_config_frozen.json": "6aba9c4dbae37ef30fd2792428400e653356bb660d5ca72a4fe5e50755f76aab", "r1_negative_set_amendment.json": "4d0d1631a6ff06724c10ea2c3b38997e48fec2a1ae14e08fc907ee9eef6cb6ef"}
- 协议：C 网格按 dev AUROC 最大、平局取更小 C；SGD 5ep best-dev×3 seeds 均值；LGBM(800) 固定；AUROC / AP / task-macro AUROC + task-cluster bootstrap 500 (seed 20261008)

## 1. 主要读出（dev AUROC [95% task-cluster bootstrap CI]）

| 折 | 版本 | P0_fusion | P0_equal | best single | best 名称 | metadata_only | size_length_only |
|---|---|---|---|---|---|---|---|
| CodeLlama-Instruct / heldout=7b / size_mix | | 0.955 [0.942, 0.966] | 0.949 [0.935, 0.961] | 0.930 | style_lgb | 0.847 [0.823, 0.868] | 0.848 [0.842, 0.855] |
| CodeLlama-Instruct / heldout=7b / size_matched | | 0.963 [0.953, 0.973] | 0.960 [0.949, 0.970] | 0.940 | tfidf_word | 0.865 [0.842, 0.886] | 0.658 [0.639, 0.676] |
| CodeLlama-Instruct / heldout=13b / size_mix | | 0.961 [0.951, 0.971] | 0.955 [0.942, 0.966] | 0.942 | style_lgb | 0.865 [0.843, 0.884] | 0.806 [0.795, 0.818] |
| CodeLlama-Instruct / heldout=13b / size_matched | | 0.969 [0.960, 0.977] | 0.963 [0.953, 0.973] | 0.947 | style_lgb | 0.884 [0.864, 0.903] | 0.659 [0.637, 0.683] |
| CodeLlama-Instruct / heldout=34b / size_mix | | 0.961 [0.951, 0.972] | 0.958 [0.944, 0.969] | 0.941 | style_lgb | 0.866 [0.842, 0.885] | 0.752 [0.738, 0.766] |
| CodeLlama-Instruct / heldout=34b / size_matched | | 0.970 [0.962, 0.978] | 0.965 [0.953, 0.974] | 0.948 | style_lgb | 0.881 [0.857, 0.903] | 0.665 [0.640, 0.688] |
| CodeLlama-Instruct / heldout=70b / size_mix | | 0.962 [0.950, 0.972] | 0.958 [0.945, 0.969] | 0.941 | style_lgb | 0.862 [0.840, 0.882] | 0.720 [0.703, 0.737] |
| CodeLlama-Instruct / heldout=70b / size_matched | | 0.970 [0.960, 0.979] | 0.966 [0.956, 0.976] | 0.948 | style_lgb | 0.880 [0.857, 0.902] | 0.629 [0.599, 0.657] |
| Qwen2.5-Coder-Instruct / heldout=1.5B / size_mix | | 0.987 [0.981, 0.991] | 0.990 [0.985, 0.994] | 0.978 | style_lgb | 0.958 [0.943, 0.971] | 0.602 [0.578, 0.624] |
| Qwen2.5-Coder-Instruct / heldout=1.5B / size_matched | | 0.990 [0.985, 0.995] | 0.994 [0.989, 0.997] | 0.986 | style_lgb | 0.967 [0.950, 0.980] | 0.622 [0.591, 0.651] |
| Qwen2.5-Coder-Instruct / heldout=7B / size_mix | | 0.971 [0.963, 0.978] | 0.977 [0.970, 0.984] | 0.968 | style_lgb | 0.952 [0.936, 0.964] | 0.621 [0.605, 0.635] |
| Qwen2.5-Coder-Instruct / heldout=7B / size_matched | | 0.980 [0.972, 0.986] | 0.980 [0.972, 0.986] | 0.971 | style_lgb | 0.952 [0.935, 0.967] | 0.576 [0.553, 0.595] |
| Qwen2.5-Coder-Instruct / heldout=14B / size_mix | | 0.978 [0.972, 0.985] | 0.979 [0.972, 0.985] | 0.969 | style_lgb | 0.953 [0.939, 0.966] | 0.655 [0.648, 0.662] |
| Qwen2.5-Coder-Instruct / heldout=14B / size_matched | | 0.977 [0.968, 0.984] | 0.979 [0.971, 0.985] | 0.969 | style_lgb | 0.951 [0.932, 0.965] | 0.576 [0.552, 0.600] |
| Qwen2.5-Coder-Instruct / heldout=32B / size_mix | | 0.976 [0.968, 0.983] | 0.975 [0.968, 0.983] | 0.966 | style_lgb | 0.949 [0.933, 0.963] | 0.723 [0.716, 0.729] |
| Qwen2.5-Coder-Instruct / heldout=32B / size_matched | | 0.978 [0.970, 0.985] | 0.977 [0.969, 0.984] | 0.971 | style_lgb | 0.950 [0.930, 0.966] | 0.588 [0.563, 0.610] |
| DeepSeek-Coder-v1-Instruct / heldout=1.3b / size_mix | | 0.929 [0.913, 0.942] | 0.931 [0.915, 0.944] | 0.907 | style_lgb | 0.821 [0.794, 0.846] | 0.683 [0.657, 0.706] |
| DeepSeek-Coder-v1-Instruct / heldout=1.3b / size_matched | | 0.920 [0.897, 0.939] | 0.924 [0.903, 0.942] | 0.897 | style_lgb | 0.811 [0.781, 0.838] | 0.651 [0.623, 0.683] |
| DeepSeek-Coder-v1-Instruct / heldout=6.7b / size_mix | | 0.928 [0.912, 0.941] | 0.926 [0.910, 0.939] | 0.909 | style_lgb | 0.826 [0.798, 0.849] | 0.707 [0.689, 0.727] |
| DeepSeek-Coder-v1-Instruct / heldout=6.7b / size_matched | | 0.917 [0.899, 0.936] | 0.917 [0.898, 0.935] | 0.890 | style_lgb | 0.811 [0.780, 0.838] | 0.608 [0.579, 0.640] |
| DeepSeek-Coder-v1-Instruct / heldout=33b / size_mix | | 0.927 [0.911, 0.940] | 0.933 [0.918, 0.945] | 0.907 | style_lgb | 0.812 [0.785, 0.835] | 0.885 [0.880, 0.891] |
| DeepSeek-Coder-v1-Instruct / heldout=33b / size_matched | | 0.938 [0.920, 0.954] | 0.946 [0.931, 0.961] | 0.926 | style_lgb | 0.850 [0.822, 0.879] | 0.682 [0.656, 0.710] |

## 2. 负集版本敏感性（P0_fusion dev AUROC）

| 折 | size_mix | size_matched | Δ(matched−mix) |
|---|---|---|---|
| CodeLlama-Instruct / heldout=7b / size_mix | 0.955 | 0.963 | +0.009 |
| CodeLlama-Instruct / heldout=13b / size_mix | 0.961 | 0.969 | +0.007 |
| CodeLlama-Instruct / heldout=34b / size_mix | 0.961 | 0.970 | +0.009 |
| CodeLlama-Instruct / heldout=70b / size_mix | 0.962 | 0.970 | +0.008 |
| Qwen2.5-Coder-Instruct / heldout=1.5B / size_mix | 0.987 | 0.990 | +0.004 |
| Qwen2.5-Coder-Instruct / heldout=7B / size_mix | 0.971 | 0.980 | +0.009 |
| Qwen2.5-Coder-Instruct / heldout=14B / size_mix | 0.978 | 0.977 | -0.002 |
| Qwen2.5-Coder-Instruct / heldout=32B / size_mix | 0.976 | 0.978 | +0.002 |
| DeepSeek-Coder-v1-Instruct / heldout=1.3b / size_mix | 0.929 | 0.920 | -0.009 |
| DeepSeek-Coder-v1-Instruct / heldout=6.7b / size_mix | 0.928 | 0.917 | -0.011 |
| DeepSeek-Coder-v1-Instruct / heldout=33b / size_mix | 0.927 | 0.938 | +0.011 |

## 3. R2 表示迁移（dev AUROC；系列中心距离）

| 折 | base euclid | base cosine | small euclid | small cosine | rank-corr(base euclid, log10 size) |
|---|---|---|---|---|---|
| CodeLlama-Instruct / heldout=7b / size_mix | 0.587 | 0.498 | 0.563 | 0.388 | +0.168 |
| CodeLlama-Instruct / heldout=7b / size_matched | 0.536 | 0.510 | 0.513 | 0.381 | -0.018 |
| CodeLlama-Instruct / heldout=13b / size_mix | 0.602 | 0.507 | 0.584 | 0.547 | +0.168 |
| CodeLlama-Instruct / heldout=13b / size_matched | 0.561 | 0.521 | 0.547 | 0.547 | -0.003 |
| CodeLlama-Instruct / heldout=34b / size_mix | 0.626 | 0.441 | 0.610 | 0.449 | +0.172 |
| CodeLlama-Instruct / heldout=34b / size_matched | 0.591 | 0.446 | 0.576 | 0.447 | -0.036 |
| CodeLlama-Instruct / heldout=70b / size_mix | 0.599 | 0.389 | 0.579 | 0.449 | +0.134 |
| CodeLlama-Instruct / heldout=70b / size_matched | 0.565 | 0.386 | 0.545 | 0.445 | -0.088 |
| Qwen2.5-Coder-Instruct / heldout=1.5B / size_mix | 0.639 | 0.477 | 0.679 | 0.510 | +0.083 |
| Qwen2.5-Coder-Instruct / heldout=1.5B / size_matched | 0.619 | 0.457 | 0.673 | 0.520 | +0.012 |
| Qwen2.5-Coder-Instruct / heldout=7B / size_mix | 0.593 | 0.457 | 0.635 | 0.422 | +0.094 |
| Qwen2.5-Coder-Instruct / heldout=7B / size_matched | 0.597 | 0.473 | 0.634 | 0.394 | +0.130 |
| Qwen2.5-Coder-Instruct / heldout=14B / size_mix | 0.590 | 0.453 | 0.627 | 0.398 | +0.074 |
| Qwen2.5-Coder-Instruct / heldout=14B / size_matched | 0.594 | 0.440 | 0.624 | 0.406 | +0.109 |
| Qwen2.5-Coder-Instruct / heldout=32B / size_mix | 0.580 | 0.603 | 0.622 | 0.552 | +0.074 |
| Qwen2.5-Coder-Instruct / heldout=32B / size_matched | 0.573 | 0.615 | 0.622 | 0.534 | +0.163 |
| DeepSeek-Coder-v1-Instruct / heldout=1.3b / size_mix | 0.475 | 0.417 | 0.511 | 0.543 | +0.029 |
| DeepSeek-Coder-v1-Instruct / heldout=1.3b / size_matched | 0.492 | 0.407 | 0.527 | 0.544 | -0.025 |
| DeepSeek-Coder-v1-Instruct / heldout=6.7b / size_mix | 0.470 | 0.453 | 0.499 | 0.680 | +0.028 |
| DeepSeek-Coder-v1-Instruct / heldout=6.7b / size_matched | 0.510 | 0.473 | 0.522 | 0.613 | -0.016 |
| DeepSeek-Coder-v1-Instruct / heldout=33b / size_mix | 0.449 | 0.531 | 0.491 | 0.530 | +0.041 |
| DeepSeek-Coder-v1-Instruct / heldout=33b / size_matched | 0.476 | 0.522 | 0.515 | 0.554 | -0.044 |

## 4. 汇总

- P0_fusion dev AUROC：min 0.917 / median 0.966 / max 0.990（22 折）
- P0_equal dev AUROC：min 0.917 / median 0.962 / max 0.994
- CodeLlama-Instruct：P0_fusion 折内范围 0.955 – 0.970
- Qwen2.5-Coder-Instruct：P0_fusion 折内范围 0.971 – 0.990
- DeepSeek-Coder-v1-Instruct：P0_fusion 折内范围 0.917 – 0.938

## 5. 解释边界（预声明）

- dev=见尺寸+预声明负集（不含 heldout 变体）；heldout 迁移信号只能由后续 **单独授权的一次性 test read** 评估。
- dev 同时是选择集（C 网格 / SGD best-dev），上述数字为选择内估计，不得当作未见迁移证据。
- 不得表述为：复现本机原件、unseen independent family、后训练因果、无污染确认性 benchmark。
