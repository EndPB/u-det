import json, zipfile, hashlib, random
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path(r"D:\data\vscode\u-det")
OUT = ROOT / "d-det" / "data" / "h2_authorbench"
ZIP = OUT / "LLM-AuthorBench.json.zip"
OUT.mkdir(parents=True, exist_ok=True)

FAMILY_MAP = {
    "claude-3.5-haiku": "claude",
    "gpt-4o": "openai",
    "gpt-4.1": "openai",
    "gpt-4o-mini": "openai",
    "gemini-2.5-flash-preview-05-20": "gemini",
    "qwen-2.5-72b-instruct": "qwen",
    "deepseek-chat": "deepseek",
    "llama-3.3-70b-instruct": "llama",
}
with zipfile.ZipFile(ZIP) as z:
    raw = json.loads(z.read("LLM-AuthorBench.json"))

by_prompt_model = defaultdict(list)
for idx, row in enumerate(raw):
    by_prompt_model[(row["prompt"], row["model_name"])].append((idx, row))

prompt_models = defaultdict(set)
for prompt, model in by_prompt_model:
    prompt_models[prompt].add(model)
all_models = sorted(FAMILY_MAP)
selected_prompts = sorted(
    prompt for prompt, models in prompt_models.items()
    if set(models) == set(all_models)
)
selected = []
for prompt in selected_prompts:
    task_id = "authorbench_" + hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:12]
    for model in all_models:
        candidates = by_prompt_model[(prompt, model)]
        chosen_idx, chosen = min(candidates, key=lambda x: x[1]["SHA256_checksum"])
        selected.append({
            "task_id": task_id,
            "prompt": prompt,
            "code": chosen["c_code"],
            "model_name": model,
            "family": FAMILY_MAP[model],
            "language": "C",
            "source": "LLM-AuthorBench",
            "source_row_index": chosen_idx,
            "source_sha256": chosen["SHA256_checksum"],
            "replicate_count": len(candidates),
            "char_count": chosen.get("char_count"),
            "num_lines": chosen.get("num_lines"),
            "nloc": chosen.get("nloc"),
            "cyclomatic_complexity": chosen.get("CC"),
            "token_size": chosen.get("token_size"),
        })

rng = random.Random(20261001)
task_ids = sorted({r["task_id"] for r in selected})
rng.shuffle(task_ids)
n = len(task_ids)
n_train = round(n * 0.70)
n_dev = round(n * 0.15)
task_split = {}
for i, task_id in enumerate(task_ids):
    task_split[task_id] = "train" if i < n_train else ("dev" if i < n_train + n_dev else "test")
for r in selected:
    r["task_split"] = task_split[r["task_id"]]

with (OUT / "core.jsonl").open("w", encoding="utf-8", newline="\n") as f:
    for r in selected:
        f.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n")

with (OUT / "task_index.jsonl").open("w", encoding="utf-8", newline="\n") as f:
    seen = set()
    for r in selected:
        if r["task_id"] in seen:
            continue
        seen.add(r["task_id"])
        f.write(json.dumps({
            "task_id": r["task_id"],
            "prompt": r["prompt"],
            "language": "C",
            "task_split": r["task_split"],
            "num_models": len(all_models),
        }, ensure_ascii=False, separators=(",", ":")) + "\n")

openai_models = [m for m in all_models if FAMILY_MAP[m] == "openai"]
generator_folds = []
for heldout in openai_models:
    generator_folds.append({
        "name": "holdout_" + heldout.replace(".", "_"),
        "purpose": "auxiliary_within_openai_family_generator_holdout",
        "train_generators": [m for m in all_models if m != heldout],
        "heldout_generators": [heldout],
        "warning": "Only the OpenAI family has multiple generators in this public dataset; do not report this as a multi-family H2 result."
    })

summary = {
    "dataset": "LLM-AuthorBench task-aware subset",
    "source_archive": "LLM-AuthorBench.json.zip",
    "source_archive_sha256": hashlib.sha256(ZIP.read_bytes()).hexdigest(),
    "selected_rule": "prompts present for all eight model names; one deterministic row per prompt-model pair",
    "rows": len(selected),
    "tasks": len(task_ids),
    "models": all_models,
    "families": sorted(set(FAMILY_MAP.values())),
    "language": ["C"],
    "task_split_counts": dict(Counter(task_split.values())),
    "family_counts": dict(Counter(r["family"] for r in selected)),
    "model_counts": dict(Counter(r["model_name"] for r in selected)),
    "replicate_count_distribution": dict(Counter(r["replicate_count"] for r in selected)),
    "generator_folds": generator_folds,
    "limitations": [
        "The dataset is C-only.",
        "Most families have only one generator; cross-generator H2 is auxiliary and only applies to the OpenAI family.",
        "The primary split holds out tasks, while generator folds are a separate diagnostic protocol.",
        "The source archive contains multiple samples per prompt-model pair; this subset keeps one deterministic sample and records replicate_count."
    ]
}
(OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
(OUT / "fold_plan.json").write_text(json.dumps({
    "seed": 20261001,
    "task_split": task_split,
    "generator_diagnostic_folds": generator_folds,
    "primary_protocol": "task-heldout family attribution",
    "auxiliary_protocol": "within-OpenAI generator-heldout diagnostic"
}, ensure_ascii=False, indent=2), encoding="utf-8")
readme = """# LLM-AuthorBench task-aware subset

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
"""
(OUT / "README.md").write_text(readme, encoding="utf-8")
print(json.dumps({"tasks": len(task_ids), "rows": len(selected), "out": str(OUT), "archive_bytes": ZIP.stat().st_size}, ensure_ascii=False))

