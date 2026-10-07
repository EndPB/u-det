# H2 generator-heldout TF-IDF 基线（本机，2026-10-07）

状态：已完成；CPU 流式/稀疏 TF-IDF，无神经训练。词表只在每折 train endpoint 拟合，阈值只在 dev 选择，test 仅读取一次。

## 结果（6 个具备 train/dev/test 正负支持的折；非加权折均值）

| view | BA | F1 | ROC-AUC |
|---|---:|---:|---:|
| raw | 0.5331 | 0.5982 | 0.5790 |
| ids_only | 0.5551 | 0.6813 | 0.5716 |
| strings_only | 0.5754 | 0.5155 | 0.5761 |
| comments_only | 0.5325 | 0.6682 | 0.5651 |
| all | 0.5153 | 0.6667 | 0.5158 |

## 按 source 的 BA

| source | raw | ids | strings | comments | all |
|---|---:|---:|---:|---:|---:|
| authorbench_dcan | 0.4979 | 0.5120 | 0.5044 | 0.4896 | 0.5276 |
| llm_codegen_v2 | 0.5683 | 0.5982 | 0.6463 | 0.5753 | 0.5030 |

## 折状态

| fold | 状态 | test 行数 |
|---|---|---:|
| authorbench_dcan:openai:holdout=gpt-4.1 | admitted | 229 |
| authorbench_dcan:openai:holdout=gpt-4o | admitted | 223 |
| authorbench_dcan:openai:holdout=gpt-4o-mini | admitted | 218 |
| llm_codegen_v2:google:holdout=Gemni-1.5-pro | diagnostic-only: no train/dev positive support | 48 |
| llm_codegen_v2:google:holdout=codegemma | diagnostic-only: no train/dev positive support | 41 |
| llm_codegen_v2:meta:holdout=codellama | admitted | 51 |
| llm_codegen_v2:meta:holdout=llama2 | admitted | 49 |
| llm_codegen_v2:meta:holdout=llama3 | admitted | 50 |
| llm_codegen_v2:mistral:holdout=codestral | diagnostic-only: no train/dev positive support | 34 |
| llm_codegen_v2:mistral:holdout=mistral | diagnostic-only: no train/dev positive support | 36 |

## 解释边界

- AuthorBench 的三折 OpenAI family-heldout 在所有视图 BA 约 `.49–.53`，当前 lexical pair verification 接近机会。
- LLM-CodeGen v2 的三折 Meta family-heldout 以 `strings_only`（约 `.646`）和 `ids_only`（约 `.598`）最高，但只有三个 generator、每折 test 正例约 38–40，CI 较宽；这是表面来源可读性诊断，不是跨数据集普遍 H2。
- `all` 没有稳定优于清洗视图；这与“直接把所有代码拼起来会更强”的假设不符。
- Google/Mistral 两 generator 的 generator-heldout 折没有 train/dev 正对，已保留为 diagnostic-only，不补造 family 正对。
- 这是同 family pair verification，不是 K-way family attribution，也没有检验后训练 base/instruct 因果。

## 产物与复现

`metrics.json`、`scripts/h2_generator_heldout_tfidf.py`；运行：

```powershell
python scripts/h2_generator_heldout_tfidf.py
```
