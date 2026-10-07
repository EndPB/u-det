# 变体迁移 stage-2：test 单次评分汇报（2026-10-08）

> **source_status=server_reconstruction_only**；original_bundle_verified=false；claims_of_byte_identity=forbidden。
> 一次性 test 读取（§4）。此后 test_read_allowed=false；只能对保存分数重算预声明统计。

- 评分行数：1881（11 成员 × 171 test task）；读数 10 + R2 2
- 读数显示名：metadata_only=`code_layout_control`；size_length_only=`oracle_size_length_control`；arms：size_mix（主）/ member-size-matched（length-unweighted）
- 统计：bootstrap 500（seed 20261008，共享抽样序列贯穿所有折/臂）；汇总先系列内折等权、再系列间等权；task-macro 含 multiplicity 修正

## 1. 汇总（heldout_test 迁移读出；mean [95% CI]，跨任务重采样）

| 读出 | heldout AUROC | heldout task-macro | seen AUROC | Δ(heldout−seen) AUROC |
|---|---|---|---|---|
| tfidf_char | 0.854 [0.840,0.869] | 0.897 [0.884,0.911] | 0.879 | -0.025 [-0.028,-0.022] |
| tfidf_word | 0.879 [0.866,0.893] | 0.915 [0.903,0.927] | 0.904 | -0.026 [-0.029,-0.022] |
| sem_base | 0.910 [0.897,0.923] | 0.929 [0.919,0.940] | 0.922 | -0.012 [-0.015,-0.010] |
| sem_small | 0.909 [0.896,0.922] | 0.932 [0.922,0.944] | 0.922 | -0.012 [-0.015,-0.010] |
| style_lr | 0.918 [0.903,0.930] | 0.930 [0.918,0.941] | 0.926 | -0.007 [-0.010,-0.005] |
| style_lgb | 0.933 [0.921,0.943] | 0.935 [0.922,0.946] | 0.943 | -0.009 [-0.012,-0.007] |
| metadata_only | 0.886 [0.870,0.901] | 0.900 [0.884,0.914] | 0.891 | -0.005 [-0.007,-0.004] |
| size_length_only | 0.606 [0.594,0.619] | 0.650 [0.635,0.664] | 0.682 | -0.075 [-0.078,-0.072] |
| P0_fusion | 0.945 [0.936,0.953] | 0.954 [0.946,0.963] | 0.958 | -0.013 [-0.015,-0.010] |
| P0_equal | 0.945 [0.936,0.953] | 0.955 [0.945,0.963] | 0.957 | -0.013 [-0.015,-0.011] |

## 2. R2 radial score（positive-standardized；heldout 域）

| 表示 | AUROC | task-macro |
|---|---|---|
| codet5_base（22 折，等权 mean） | 0.560 | 0.598 |
| codet5_small（22 折，等权 mean） | 0.579 | 0.622 |

## 3. 每折 heldout_test AUROC（P0_fusion / P0_equal / best-member 简表）

| 折 | P0_fusion | P0_equal | size_length(ctrl) | metadata(ctrl) |
|---|---|---|---|---|
| CodeLlama-Instruct / heldout=7b / size_mix | 0.976 [0.964,0.986] | 0.972 | 0.543 | 0.907 |
| CodeLlama-Instruct / heldout=7b / size_matched | 0.983 [0.975,0.990] | 0.981 | 0.220 | 0.930 |
| CodeLlama-Instruct / heldout=13b / size_mix | 0.952 [0.936,0.966] | 0.950 | 0.698 | 0.857 |
| CodeLlama-Instruct / heldout=13b / size_matched | 0.963 [0.950,0.975] | 0.964 | 0.528 | 0.883 |
| CodeLlama-Instruct / heldout=34b / size_mix | 0.937 [0.919,0.954] | 0.926 | 0.844 | 0.841 |
| CodeLlama-Instruct / heldout=34b / size_matched | 0.945 [0.927,0.961] | 0.938 | 0.742 | 0.867 |
| CodeLlama-Instruct / heldout=70b / size_mix | 0.915 [0.893,0.935] | 0.904 | 0.930 | 0.845 |
| CodeLlama-Instruct / heldout=70b / size_matched | 0.928 [0.910,0.946] | 0.919 | 0.667 | 0.871 |
| Qwen2.5-Coder-Instruct / heldout=1.5B / size_mix | 0.920 [0.891,0.946] | 0.905 | 0.252 | 0.923 |
| Qwen2.5-Coder-Instruct / heldout=1.5B / size_matched | 0.921 [0.896,0.944] | 0.929 | 0.659 | 0.920 |
| Qwen2.5-Coder-Instruct / heldout=7B / size_mix | 0.970 [0.956,0.981] | 0.974 | 0.689 | 0.930 |
| Qwen2.5-Coder-Instruct / heldout=7B / size_matched | 0.960 [0.939,0.977] | 0.974 | 0.571 | 0.930 |
| Qwen2.5-Coder-Instruct / heldout=14B / size_mix | 0.983 [0.974,0.990] | 0.987 | 0.559 | 0.949 |
| Qwen2.5-Coder-Instruct / heldout=14B / size_matched | 0.983 [0.974,0.991] | 0.987 | 0.645 | 0.945 |
| Qwen2.5-Coder-Instruct / heldout=32B / size_mix | 0.987 [0.980,0.993] | 0.991 | 0.321 | 0.966 |
| Qwen2.5-Coder-Instruct / heldout=32B / size_matched | 0.988 [0.980,0.994] | 0.991 | 0.802 | 0.964 |
| DeepSeek-Coder-v1-Instruct / heldout=1.3b / size_mix | 0.920 [0.904,0.937] | 0.919 | 0.520 | 0.836 |
| DeepSeek-Coder-v1-Instruct / heldout=1.3b / size_matched | 0.921 [0.903,0.942] | 0.916 | 0.675 | 0.819 |
| DeepSeek-Coder-v1-Instruct / heldout=6.7b / size_mix | 0.923 [0.904,0.940] | 0.921 | 0.758 | 0.829 |
| DeepSeek-Coder-v1-Instruct / heldout=6.7b / size_matched | 0.904 [0.878,0.928] | 0.909 | 0.621 | 0.808 |
| DeepSeek-Coder-v1-Instruct / heldout=33b / size_mix | 0.923 [0.907,0.939] | 0.925 | 0.347 | 0.847 |
| DeepSeek-Coder-v1-Instruct / heldout=33b / size_matched | 0.938 [0.918,0.955] | 0.942 | 0.734 | 0.901 |

## 4. 控制与配对差（P0_fusion 对照；heldout 域；k=22 折分布）

| 对照 | ΔAUROC 均值 [min,max] | Δtask-macro 均值 [min,max] | frac≤0（均值） |
|---|---|---|---|
| fusion−P0_equal | 0.001 [-0.014,0.014] | -0.000 [-0.024,0.015] | 0.500 |
| fusion−size_length_only | 0.341 [-0.015,0.763] | 0.309 [-0.047,0.787] | 0.042 |
| fusion−metadata_only | 0.058 [-0.004,0.104] | 0.053 [0.014,0.116] | 0.051 |

## 5. 分层（heldout 域，P0_fusion，size_mix；仅解释）

| 折 | in_hard n/pos/AUROC | not_hard n/pos/AUROC |
|---|---|---|
| CodeLlama-Instruct / heldout=7b / size_mix | 168/21/0.949 | 1200/150/0.980 |
| CodeLlama-Instruct / heldout=13b / size_mix | 168/21/0.967 | 1200/150/0.949 |
| CodeLlama-Instruct / heldout=34b / size_mix | 168/21/0.910 | 1200/150/0.940 |
| CodeLlama-Instruct / heldout=70b / size_mix | 168/21/0.897 | 1200/150/0.917 |
| Qwen2.5-Coder-Instruct / heldout=1.5B / size_mix | 168/21/0.951 | 1200/150/0.918 |
| Qwen2.5-Coder-Instruct / heldout=7B / size_mix | 168/21/0.932 | 1200/150/0.974 |
| Qwen2.5-Coder-Instruct / heldout=14B / size_mix | 168/21/0.954 | 1200/150/0.985 |
| Qwen2.5-Coder-Instruct / heldout=32B / size_mix | 168/21/0.979 | 1200/150/0.988 |
| DeepSeek-Coder-v1-Instruct / heldout=1.3b / size_mix | 189/21/0.936 | 1350/150/0.918 |
| DeepSeek-Coder-v1-Instruct / heldout=6.7b / size_mix | 189/21/0.935 | 1350/150/0.920 |
| DeepSeek-Coder-v1-Instruct / heldout=33b / size_mix | 189/21/0.940 | 1350/150/0.921 |

## 6. support_gap（匹配负集 vs heldout 尺寸；解释用）

| 折 | heldout size(B) | neg min|Δlog10| |
|---|---|---|
| CodeLlama-Instruct / heldout=7b / size_matched | 7.0 | 0.301 |
| CodeLlama-Instruct / heldout=13b / size_matched | 13.0 | 0.2688 |
| CodeLlama-Instruct / heldout=34b / size_matched | 34.0 | 0.013 |
| CodeLlama-Instruct / heldout=70b / size_matched | 70.0 | 0.3266 |
| Qwen2.5-Coder-Instruct / heldout=1.5B / size_matched | 1.5 | 0.669 |
| Qwen2.5-Coder-Instruct / heldout=7B / size_matched | 7.0 | 0.2688 |
| Qwen2.5-Coder-Instruct / heldout=14B / size_matched | 14.0 | 0.301 |
| Qwen2.5-Coder-Instruct / heldout=32B / size_matched | 32.0 | 0.3912 |
| DeepSeek-Coder-v1-Instruct / heldout=1.3b / size_matched | 1.3 | 0.7312 |
| DeepSeek-Coder-v1-Instruct / heldout=6.7b / size_matched | 6.7 | 0.65 |
| DeepSeek-Coder-v1-Instruct / heldout=33b / size_matched | 33.0 | 0.6734 |

## 7. 边界

- arm=不同负域下的敏感性，不是同样本读出对比；AP 需结合正类率（见 metrics JSON 的 pos_rate）。
- CI 仅反映固定这些模型下的任务变异；bootstrap 频率非后验概率；不挑显著折扩写结论。
- 论文表述上限：固定公开 BigCodeBench 协议下的官方模型系列内尺寸变体迁移；保留模板/清洗/污染未知限制。
- 特征哈希：{'style_sha256': 'e6576a84f98f928019e330d2be7b06b3657d6ee7f6daf889c71cfb0a0e31bec0', 'emb_small_sha256': 'a8c869a6c107e11206c2730dde5b51ec117ecd3c722ecfcc7c1b2c63ee5d039b', 'emb_base_sha256': '2252f5baa885cccd6e0ead6d5d044a4c1a0c3d34a754f20f1ededf081e3be957'}；预测哈希：{'test_scores_npz_sha256': 'a3ae6ee38c35df40a29d1fad29817d87cab1ab9407e5ce0c0e32fc5b505646d2'}
