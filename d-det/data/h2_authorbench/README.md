# LLM-AuthorBench task-aware subset

这是从公开 LLM-AuthorBench 压缩包中筛出的轻量任务条件数据集，不调用任何模型 API，也不自行生成代码。

## 内容

- core.jsonl：1,912 行，239 个 prompt，每个 prompt 保留 8 个模型各 1 条输出。
- task_index.jsonl：239 个任务及其固定的 train/dev/test 划分。
- fold_plan.json：任务留出主协议，以及只针对 OpenAI 家族的 generator-held-out 辅助诊断。
- summary.json：来源 hash、计数和限制。
- LLM-AuthorBench.json.zip：公开来源压缩包，约 21 MB，用于溯源。

## 适用问题

主协议用于测试：模型是否在没有见过测试任务时，仍能进行 family/model attribution。

辅助协议用于测试：训练见过 GPT-4o、GPT-4.1、GPT-4o-mini 中的两个 generator 后，能否识别留出的第三个 generator 是否仍属于 OpenAI family。

## 不能声称什么

公开数据中只有 OpenAI family 有多个 generator，Claude、Gemini、Qwen、DeepSeek、Llama 各只有一个 generator。因此它不能单独支持“多 family 跨 generator H2 已验证”。它适合做任务捷径审计、中心化前后比较、闭集 family 读出和一个有限的 within-OpenAI 诊断。

## 建议的读取口径

- 训练/验证/测试按 task_split 整组切分，同一个 task_id 不得跨 split。
- 任务条件实验可以在同一任务内比较不同模型，但必须明确这是 task-aware / transductive 口径。
- 任务泛化实验只使用 task_split=test 的完整任务组做最终评估。
- 使用 replicate_count 做敏感性分析，不把重复样本当作独立任务。
- CPU 读取固定为 2 线程；不需要 GPU。后续编码时再使用本地 3050，采用流式批处理即可。

来源仓库：LLMauthorbench/LLMauthorbench。
