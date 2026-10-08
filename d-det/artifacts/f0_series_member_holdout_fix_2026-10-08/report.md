# F0-B 修复：true member-heldout（corrective rerun，不覆盖旧结果）

- eval positive 与 train model 集合交集=0（断言通过）；折数=11（4+4+3）

## 修复后 aggregate（11 折）

| 视图 | mean AUROC | min | max | 折>0.5 |
|---|---|---|---|---|
| u | 0.7967 | 0.6306 | 0.9168 | 11/11 |
| h_instruct | 0.7775 | 0.6128 | 0.8648 | 11/11 |
| delta | 0.6163 | 0.2731 | 0.8005 | 9/11 |
| p0 | 0.6476 | 0.4263 | 0.9188 | 9/11 |
| h_complete | 0.6570 | 0.5683 | 0.8781 | 11/11 |

## 旧（错误）结果并列（仅作 same-member task-heldout 诊断）

| 视图 | 旧 mean AUROC（错误口径） |
|---|---|
| h_instruct | 0.7871 |
| delta | 0.6972 |
| u | 0.8116 |
| p0 | 0.7233 |
| h_complete | 0.6707 |

## 逐 member（u 视图）

| member | AUROC | task-cluster CI95 | AP |
|---|---|---|---|
| CodeLlama-Instruct::codellama--CodeLlama-7b-Instruct-hf | 0.8772 | [0.847, 0.902] | 0.5913 |
| CodeLlama-Instruct::codellama--CodeLlama-13b-Instruct-hf | 0.7893 | [0.748, 0.825] | 0.4984 |
| CodeLlama-Instruct::codellama--CodeLlama-34b-Instruct-hf | 0.7764 | [0.740, 0.812] | 0.4793 |
| CodeLlama-Instruct::codellama--CodeLlama-70b-Instruct-hf | 0.8126 | [0.779, 0.845] | 0.4877 |
| Qwen2.5-Coder-Instruct::Qwen--Qwen2.5-Coder-1.5B-Instruct | 0.6306 | [0.602, 0.661] | 0.1926 |
| Qwen2.5-Coder-Instruct::Qwen--Qwen2.5-Coder-7B-Instruct | 0.8246 | [0.795, 0.853] | 0.4461 |
| Qwen2.5-Coder-Instruct::Qwen--Qwen2.5-Coder-14B-Instruct | 0.7866 | [0.755, 0.817] | 0.3539 |
| Qwen2.5-Coder-Instruct::Qwen--Qwen2.5-Coder-32B-Instruct | 0.8309 | [0.803, 0.857] | 0.4155 |
| DeepSeek-Coder-v1-Instruct::deepseek-ai--deepseek-coder-1.3b-instruct | 0.7872 | [0.753, 0.819] | 0.4487 |
| DeepSeek-Coder-v1-Instruct::deepseek-ai--deepseek-coder-6.7b-instruct | 0.9168 | [0.894, 0.938] | 0.7266 |
| DeepSeek-Coder-v1-Instruct::deepseek-ai--deepseek-coder-33b-instruct | 0.7312 | [0.705, 0.761] | 0.2795 |

> 报告名：observed series / unseen-member transfer（family_is_confirmed=false）。折支持度不足者照留不删。
