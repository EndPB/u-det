# P0 汇总（自动生成）

## STACAD 官方协议复现（5 折 OOF macro-F1 mean±std）

| 阶段 | 模型 | repro F1 | official F1 | Δ |
|---|---|---|---|---|
| classic | logreg | 0.4874 | 0.4874 | +0.0000 |
| classic | rf | 0.5240 | 0.5241 | -0.0002 |
| learners | xgb | 0.5988 | 0.5986 | +0.0002 |
| learners | lgb | 0.5959 | 0.5959 | +0.0000 |
| learners | mlp | 0.5531 | 0.5504 | +0.0026 |
| stack | vote_0.50_0.30_0.20 | 0.5988 | 0.5988 | -0.0000 |
| stack | vote_equal | 0.5988 | 0.5988 | -0.0000 |
| stack | stack_trees | 0.6004 | 0.6004 | -0.0000 |
| stack | stack_trees_tfidf | 0.6701 | 0.6701 | +0.0000 |
| stack | stack_trees_tfidf_lang | 0.6701 | 0.6701 | -0.0000 |
| stack | stack_trees_codebert | 0.7047 | 0.7048 | -0.0001 |
| stack | stack_trees_tfidf_codebert | 0.7329 | 0.7328 | +0.0000 |
| stack | stack_trees_tfidf_lang_codebert | 0.7330 | 0.7330 | +0.0000 |

## 赛道：authorbench_dcan（test n=1457，classes=6）

| 模型 | macro-F1 | CI95 | balanced acc | ECE | 来源 |
|---|---|---|---|---|---|
| fusion_lr | 0.8388 | [0.8195, 0.8620] | 0.8319 | 0.0388 | dev-only LR stack（dev=拟合集，dev F1 非 out-of-sample）: sem_lr,style_lgb,style_lr,tfidf_char,tfidf_word |
| mean_ensemble | 0.8282 | [0.8046, 0.8483] | 0.8159 | 0.1319 | 等权概率均值: sem_lr,style_lgb,style_lr,tfidf_char,tfidf_word |
| tfidf_word | 0.8200 | [0.7980, 0.8418] | 0.8125 | 0.0272 | SGD 5ep best-dev，3 seeds（dev F1 ['0.8442', '0.8339', '0.8429']） |
| codet5_lora_ext | 0.7808 | [0.7590, 0.8033] | 0.7859 | 0.0366 | round4 lora_extend（24ep 上限，best-dev）（3 seeds；只重算指标） |
| tfidf_char | 0.7692 | [0.7434, 0.7908] | 0.7767 | 0.0539 | SGD 5ep best-dev，3 seeds（dev F1 ['0.7902', '0.7896', '0.7979']） |
| style_lgb | 0.7227 | [0.7007, 0.7468] | 0.7163 | 0.1059 | regex stylometry 97d + LightGBM(800) |
| sem_lr | 0.6528 | [0.6256, 0.6775] | 0.6535 | 0.1792 | 冻结 codet5_meanpool_768 + StandardScaler+LR(C=1)（统一协议） |
| style_lr | 0.5586 | [0.5343, 0.5858] | 0.5673 | 0.0260 | regex stylometry 97d + 标准化 + LR |
| codet5_head_only | 0.4216 | [0.3963, 0.4442] | 0.4193 | 0.1105 | round3 audit（round3_ft head_only 12ep）（3 seeds；只重算指标） |

## 赛道：stacad_fold0（test n=28992，classes=7）

| 模型 | macro-F1 | CI95 | balanced acc | ECE | 来源 |
|---|---|---|---|---|---|
| fusion_lr | 0.6089 | [0.6042, 0.6143] | 0.6113 | 0.0259 | dev-only LR stack（dev=拟合集，dev F1 非 out-of-sample）: feats105_lgb,feats105_lr,sem_lr,tfidf_char,tfidf_word |
| mean_ensemble | 0.5797 | [0.5740, 0.5857] | 0.5875 | 0.1892 | 等权概率均值: feats105_lgb,feats105_lr,sem_lr,tfidf_char,tfidf_word |
| feats105_lgb | 0.5658 | [0.5604, 0.5711] | 0.5690 | 0.1252 | 官方 105 特征 + LightGBM（800 树；官方协议映射） |
| feats105_lr | 0.4869 | [0.4822, 0.4924] | 0.4926 | 0.0259 | 官方 105 特征 + 标准化 + LR（round4-C 子训练协议内） |
| tfidf_char_round4c | 0.3809 | [0.3757, 0.3859] | 0.3829 | 0.1101 | round4-C TF-IDF（subtrain30k）（3 seeds；只重算指标） |
| tfidf_char | 0.3729 | [0.3676, 0.3775] | 0.3787 | 0.1168 | SGD 5ep best-dev，3 seeds（dev F1 ['0.3659', '0.3710', '0.3658']） |
| codet5_lora | 0.3704 | [0.3657, 0.3765] | 0.3814 | 0.0189 | round4-C LoRA（2ep 上限）（3 seeds；只重算指标） |
| tfidf_word | 0.3657 | [0.3603, 0.3708] | 0.3778 | 0.1324 | SGD 5ep best-dev，3 seeds（dev F1 ['0.3683', '0.3627', '0.3646']） |
| sem_lr | 0.3033 | [0.2985, 0.3076] | 0.3088 | 0.0619 | 冻结 codet5_meanpool_768(fp16x2) + StandardScaler+LR(C=1)（统一协议） |
| codet5_head_only | 0.2290 | [0.2249, 0.2329] | 0.2473 | 0.0449 | round4-C head-only（3 seeds；只重算指标） |

## 赛道：droid_fold0（test n=6142，classes=7）

| 模型 | macro-F1 | CI95 | balanced acc | ECE | 来源 |
|---|---|---|---|---|---|
| tfidf_char | 0.1645 | [0.1037, 0.1755] | 0.1760 | 0.2756 | SGD 5ep best-dev，3 seeds（dev F1 ['0.3533', '0.3512', '0.3575']） |
| tfidf_char_round4c | 0.1634 | [0.1008, 0.1757] | 0.1765 | 0.2801 | round4-C TF-IDF（3 seeds；只重算指标） |
| tfidf_word | 0.1628 | [0.1035, 0.1748] | 0.1712 | 0.2833 | SGD 5ep best-dev，3 seeds（dev F1 ['0.3798', '0.3784', '0.3812']） |
| fusion_lr | 0.1540 | [0.0867, 0.1749] | 0.1657 | 0.2615 | dev-only LR stack（dev=拟合集，dev F1 非 out-of-sample）: sem_lr,style_lgb,style_lr,tfidf_char,tfidf_word |
| mean_ensemble | 0.1517 | [0.0865, 0.1708] | 0.1616 | 0.1443 | 等权概率均值: sem_lr,style_lgb,style_lr,tfidf_char,tfidf_word |
| style_lr | 0.1442 | [0.1040, 0.1739] | 0.1541 | 0.0949 | regex stylometry 97d + 标准化 + LR |
| sem_lr | 0.1392 | [0.0858, 0.1523] | 0.1453 | 0.2209 | 冻结 codet5_meanpool_768(fp16x2) + StandardScaler+LR(C=1)（统一协议） |
| style_lgb | 0.1379 | [0.0855, 0.1548] | 0.1511 | 0.2772 | regex stylometry 97d + LightGBM(800) |
| codet5_head_only | 0.1349 | [0.0871, 0.1729] | 0.1404 | 0.1067 | round4-C head-only（3 seeds；只重算指标） |
| codet5_lora | 0.1277 | [0.0691, 0.1830] | 0.1357 | 0.3414 | round4-C LoRA（2ep 上限）（3 seeds；只重算指标） |
