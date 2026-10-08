# F0：exact-unit 与 observed-series 归因（train/dev）

## F0-A 115-way observed model_id（task-heldout）

| 视图 | top1 | top5 | bal-acc | macro-F1 | task-macro |
|---|---|---|---|---|---|
| h_complete | 0.0645 | 0.2212 | 0.0645 | 0.0401 | 0.0645 |
| h_instruct | 0.0854 | 0.2696 | 0.0854 | 0.0574 | 0.0854 |
| delta | 0.0826 | 0.2440 | 0.0826 | 0.0600 | 0.0826 |
| u | 0.1286 | 0.3536 | 0.1286 | 0.1004 | 0.1286 |
| p0 | 0.0194 | 0.0808 | 0.0194 | 0.0129 | 0.0194 |
（chance top1=0.0087）

## F0-A open-set（model-heldout, max-softmax rejection）
- fold0: AUC=0.5007
- fold3: AUC=0.4697

## F0-B observed_series/member transfer（11 折 heldout member）

| 视图 | 平均 AUROC | min | max |
|---|---|---|---|
| h_instruct | 0.7871 | 0.7569 | 0.8140 |
| delta | 0.6972 | 0.6023 | 0.7391 |
| u | 0.8116 | 0.7705 | 0.8586 |
| p0 | 0.7233 | 0.6212 | 0.8622 |
| h_complete | 0.6707 | 0.6205 | 0.7771 |

> 报告名使用 `observed_series/member transfer`（family_is_confirmed=false）。
