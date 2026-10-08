# F2 置换 null 分布（20 seeds/fold；新证据）

| heldout member | true AUC | null mean | null std | p2.5 | p50 | p97.5 | true pct | 真值>p97.5 |
|---|---|---|---|---|---|---|---|---|
| CodeLlama-Instruct::codellama--CodeLlama-7b-Instruct-hf | 0.8390 | 0.5016 | 0.0404 | 0.4483 | 0.4901 | 0.5877 | 100 | True |
| CodeLlama-Instruct::codellama--CodeLlama-13b-Instruct-hf | 0.8281 | 0.5016 | 0.0390 | 0.4414 | 0.4973 | 0.5747 | 100 | True |
| CodeLlama-Instruct::codellama--CodeLlama-34b-Instruct-hf | 0.8488 | 0.4950 | 0.0344 | 0.4385 | 0.4890 | 0.5445 | 100 | True |
| CodeLlama-Instruct::codellama--CodeLlama-70b-Instruct-hf | 0.8693 | 0.4966 | 0.0347 | 0.4385 | 0.4979 | 0.5575 | 100 | True |
| Qwen2.5-Coder-Instruct::Qwen--Qwen2.5-Coder-1.5B-Instruct | 0.8620 | 0.4914 | 0.0319 | 0.4273 | 0.4961 | 0.5366 | 100 | True |
| Qwen2.5-Coder-Instruct::Qwen--Qwen2.5-Coder-7B-Instruct | 0.8358 | 0.4950 | 0.0309 | 0.4479 | 0.4947 | 0.5597 | 100 | True |
| Qwen2.5-Coder-Instruct::Qwen--Qwen2.5-Coder-14B-Instruct | 0.8220 | 0.4912 | 0.0418 | 0.4319 | 0.4836 | 0.5876 | 100 | True |
| Qwen2.5-Coder-Instruct::Qwen--Qwen2.5-Coder-32B-Instruct | 0.8346 | 0.4845 | 0.0319 | 0.4341 | 0.4840 | 0.5376 | 100 | True |
| DeepSeek-Coder-v1-Instruct::deepseek-ai--deepseek-coder-1.3b-instruct | 0.7426 | 0.5070 | 0.0273 | 0.4613 | 0.4980 | 0.5495 | 100 | True |
| DeepSeek-Coder-v1-Instruct::deepseek-ai--deepseek-coder-6.7b-instruct | 0.6681 | 0.5014 | 0.0222 | 0.4665 | 0.5026 | 0.5346 | 100 | True |
| DeepSeek-Coder-v1-Instruct::deepseek-ai--deepseek-coder-33b-instruct | 0.8026 | 0.4902 | 0.0409 | 0.4224 | 0.5000 | 0.5544 | 100 | True |

- null p50 中位数（跨折）=0.4961；AUC_norm 逐折见 metrics。
