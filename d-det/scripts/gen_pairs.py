#!/usr/bin/env python
"""生成 base / instruct 配对数据（s2 的零标注监督；**有卡时运行**）。

原理（docx/d-det.md §3）：同族 instruct 与 base 的分布比 = e^{r(x)/β} / Z(β)，
即"偏好位移"可从两版输出的差分直接学习 —— **不需要任何人类标注**。

流程：
    1) 读 prompt 集（data/raw/prompts/，由 scripts/prepare.py prompts 下载）；
    2) 各自用**自然用法**生成：base 走 completion（complete_prompt 前缀续写），
       instruct 走 chat 模板（instruct_prompt）——同一道题、各模型最自然的使用方式；
    3) CodeT5 分词器预分词；按 task_id 哈希 8:1:1 切分（同一题不跨 split，防泄漏）；
    4) 写 data/processed/pairs.parquet。

用法（有卡，命令前加 OMP_NUM_THREADS=8）::

    python scripts/gen_pairs.py --limit 100 --out data/processed/pairs_smoke.parquet   # 先冒烟估时
    python scripts/gen_pairs.py --out data/processed/pairs.parquet                     # 全量

首版范围（用户决策，2026-09-23）：**不做测试回测**（"计算语义等价"留到后续严格档），
配对只用"同 prompt"近似；将来可在此脚本加 ``--verify-with-tests``。
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]

# 字段映射：(id 键, base 用键, instruct 用键)；按顺序探测
FIELD_MAP = (
    ("task_id", "complete_prompt", "instruct_prompt"),   # bigcodebench：两种形态都自带
    ("task_id", "prompt", "prompt"),                     # humaneval：两边都用同一题面
)

SCHEMA_FIELDS = ("split", "task_id", "family", "language", "x_plus", "x_minus",
                 "input_ids_plus", "input_ids_minus", "n_tokens_plus", "n_tokens_minus")


def iter_rows(path: Path):
    """探测式读行：parquet / jsonl / jsonl.gz / ndjson / json（坏文件跳过）。"""
    name = path.name.lower()
    try:
        if name.endswith(".parquet"):
            import pyarrow.parquet as pq
            for row in pq.read_table(str(path)).to_pylist():
                yield row
        elif name.endswith((".jsonl", ".ndjson", ".jsonl.gz")):
            opener = gzip.open if name.endswith(".gz") else open
            with opener(str(path), "rt", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if line:
                        yield json.loads(line)
        elif name.endswith(".json"):
            with open(path, encoding="utf-8") as handle:
                blob = json.load(handle)
            if isinstance(blob, list):
                for row in blob:
                    yield row
            elif isinstance(blob, dict):
                yield blob
    except Exception as exc:                        # noqa: BLE001 —— 探测式：坏文件不计入
        print(f"[prompts] 跳过 {path.name}：{exc}")


def load_prompts(prompts_dir: Path, limit: int | None = None) -> list:
    """返回 [{task_id, base_text, instruct_text}]（去重，按 task_id）。"""
    if not prompts_dir.exists():
        raise SystemExit(f"[prompts] 目录不存在：{prompts_dir}"
                         f"（先跑 python scripts/prepare.py prompts）")
    items, seen = [], set()
    files = [p for p in sorted(prompts_dir.rglob("*")) if p.is_file()
             and p.name.lower().endswith((".parquet", ".jsonl", ".jsonl.gz", ".ndjson", ".json"))]
    for path in files:
        for row in iter_rows(path):
            if not isinstance(row, dict):
                continue
            for id_key, base_key, inst_key in FIELD_MAP:
                text_base, text_inst = row.get(base_key), row.get(inst_key)
                if not (isinstance(text_base, str) and isinstance(text_inst, str)
                        and text_base.strip() and text_inst.strip()):
                    continue
                task_id = str(row.get(id_key) or f"{path.stem}-{len(items)}")
                if task_id in seen:
                    break
                seen.add(task_id)
                items.append({"task_id": task_id,
                              "base_text": text_base, "instruct_text": text_inst})
                break
            if limit and len(items) >= limit:
                return items
    return items


def split_of(task_id: str, seed: int) -> str:
    """task 级哈希切分（8:1:1）：同一题的所有样本进同一 split。"""
    digest = hashlib.md5(f"{seed}:{task_id}".encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 10
    return "train" if bucket < 8 else ("val" if bucket == 8 else "test")


def load_model(path: str, device: str):
    """加载因果 LM（bf16）+ tokenizer。"""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not Path(path).exists():
        raise SystemExit(f"[gen] 模型目录不存在：{path}（先跑 python scripts/prepare.py pairmdl）")
    tokenizer = AutoTokenizer.from_pretrained(path)
    if tokenizer.pad_token is None:                      # Qwen 系常见：无 pad token，用 eos 兼任
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16)
    model = model.to(device).eval()
    return tokenizer, model


@torch.no_grad()
def gen_batch(model, tokenizer, texts: list, args, chat: bool, seed: int) -> list:
    """批处理生成（左 padding）。``chat=True`` 时套 instruct 侧模板（base 侧走纯 completion）。

    批内共享同一个随机种子（``seed``）——逐条精确控制 RNG 在批处理下不可行，
    可复现性以"同一批同种子"为准（实验对比用同一数据即可）。
    """
    torch.manual_seed(seed)
    if chat and getattr(tokenizer, "chat_template", None):
        texts = [tokenizer.apply_chat_template(
            [{"role": "user", "content": t}], tokenize=False, add_generation_prompt=True)
            for t in texts]
    tokenizer.padding_side = "left"                     # decoder-only 生成的标准做法
    enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True,
                    max_length=args.max_prompt_tokens).to(args.device)
    out = model.generate(
        **enc, do_sample=True, temperature=args.temperature, top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        pad_token_id=(tokenizer.pad_token_id if tokenizer.pad_token_id is not None
                      else tokenizer.eos_token_id),
    )
    new_tokens = out[:, enc["input_ids"].shape[1]:]
    return [tokenizer.decode(row, skip_special_tokens=True) for row in new_tokens]


def main() -> int:
    parser = argparse.ArgumentParser(description="生成 base/instruct 配对（s2 的零标注监督）")
    parser.add_argument("--pair-name", default="qwen2.5-coder-0.5b")
    parser.add_argument("--base-model", default=None,
                        help="默认 checkpoints/<pair-name>-base")
    parser.add_argument("--instruct-model", default=None,
                        help="默认 checkpoints/<pair-name>-instruct")
    parser.add_argument("--codet5", default=str(ROOT / "checkpoints" / "codet5-base"),
                        help="预分词用的 CodeT5 分词器路径（与训练口径一致）")
    parser.add_argument("--prompts-dir", default=str(ROOT / "data" / "raw" / "prompts"))
    parser.add_argument("--out", default=str(ROOT / "data" / "processed" / "pairs.parquet"))
    parser.add_argument("--limit", type=int, default=None, help="只用前 N 个 prompt（冒烟）")
    parser.add_argument("--per-prompt", type=int, default=1, help="每个 prompt 生成几对")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-prompt-tokens", type=int, default=1024)
    parser.add_argument("--min-chars", type=int, default=32, help="过短的生成直接跳过")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8, help="生成批大小（越大越快；8~16 推荐）")
    parser.add_argument("--batch-write", type=int, default=256, help="每攒多少条落一次盘")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("[gen] 需要 GPU（这是生成任务）；无卡模式请勿运行")

    from transformers import AutoTokenizer
    import pyarrow as pa
    import pyarrow.parquet as pq

    base_path = args.base_model or str(ROOT / "checkpoints" / f"{args.pair_name}-base")
    inst_path = args.instruct_model or str(ROOT / "checkpoints" / f"{args.pair_name}-instruct")

    prompts = load_prompts(Path(args.prompts_dir), args.limit)
    if not prompts:
        raise SystemExit("[prompts] 没找到可用 prompt（检查 data/raw/prompts/ 的内容与字段）")
    print(f"[gen] prompts: {len(prompts)}（来自 {args.prompts_dir}）")

    corpus_tok = AutoTokenizer.from_pretrained(args.codet5)
    base_tok, base_model = load_model(base_path, args.device)
    inst_tok, inst_model = load_model(inst_path, args.device)
    print(f"[gen] base={base_path}  instruct={inst_path}  device={args.device}")

    schema = pa.schema([
        ("split", pa.string()), ("task_id", pa.string()), ("family", pa.string()),
        ("language", pa.string()),
        ("x_plus", pa.string()), ("x_minus", pa.string()),
        ("input_ids_plus", pa.list_(pa.int32())), ("input_ids_minus", pa.list_(pa.int32())),
        ("n_tokens_plus", pa.int32()), ("n_tokens_minus", pa.int32()),
    ])
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written, skipped, counts = 0, 0, {"train": 0, "val": 0, "test": 0}

    def flush(writer, rows):
        table = pa.table({
            "split": pa.array([r["split"] for r in rows], pa.string()),
            "task_id": pa.array([r["task_id"] for r in rows], pa.string()),
            "family": pa.array([r["family"] for r in rows], pa.string()),
            "language": pa.array([r["language"] for r in rows], pa.string()),
            "x_plus": pa.array([r["x_plus"] for r in rows], pa.string()),
            "x_minus": pa.array([r["x_minus"] for r in rows], pa.string()),
            "input_ids_plus": pa.array([r["input_ids_plus"] for r in rows],
                                       pa.list_(pa.int32())),
            "input_ids_minus": pa.array([r["input_ids_minus"] for r in rows],
                                        pa.list_(pa.int32())),
            "n_tokens_plus": pa.array([r["n_tokens_plus"] for r in rows], pa.int32()),
            "n_tokens_minus": pa.array([r["n_tokens_minus"] for r in rows], pa.int32()),
        }, schema=schema)
        writer.write_table(table)

    tasks = []
    for index, item in enumerate(prompts):
        for rep in range(args.per_prompt):
            tasks.append({"task_id": item["task_id"], "rep": rep,
                          "base_text": item["base_text"],
                          "instruct_text": item["instruct_text"],
                          "seed": args.seed * 1000003 + index * 101 + rep})
    print(f"[gen] 任务数 {len(tasks)}（batch_size={args.batch_size}）")

    rows = []
    with pq.ParquetWriter(out_path, schema) as writer:
        bar = tqdm(total=len(tasks), desc="gen pairs")
        for start in range(0, len(tasks), args.batch_size):
            chunk = tasks[start:start + args.batch_size]
            x_minus_list = gen_batch(base_model, base_tok,
                                     [t["base_text"] for t in chunk],
                                     args, chat=False, seed=chunk[0]["seed"])
            x_plus_list = gen_batch(inst_model, inst_tok,
                                    [t["instruct_text"] for t in chunk],
                                    args, chat=True, seed=chunk[0]["seed"] + 7)
            for task, x_minus, x_plus in zip(chunk, x_minus_list, x_plus_list):
                if len(x_minus.strip()) < args.min_chars or len(x_plus.strip()) < args.min_chars:
                    skipped += 1
                    continue
                split = split_of(task["task_id"] + f"#{task['rep']}", args.seed)
                counts[split] += 1
                ids_plus = corpus_tok(x_plus)["input_ids"]
                ids_minus = corpus_tok(x_minus)["input_ids"]
                rows.append({
                    "split": split,
                    "task_id": task["task_id"],
                    "family": args.pair_name,
                    "language": "python",
                    "x_plus": x_plus,
                    "x_minus": x_minus,
                    "input_ids_plus": ids_plus,
                    "input_ids_minus": ids_minus,
                    "n_tokens_plus": len(ids_plus),
                    "n_tokens_minus": len(ids_minus),
                })
            bar.update(len(chunk))
            if len(rows) >= args.batch_write:
                flush(writer, rows)
                written += len(rows)
                rows = []
        if rows:
            flush(writer, rows)
            written += len(rows)
        bar.close()

    size = out_path.stat().st_size / 1048576 if out_path.exists() else 0.0
    print(f"[gen] 完成：{out_path}（{written} 对，跳过 {skipped} 条过短；"
          f"train/val/test = {counts['train']}/{counts['val']}/{counts['test']}；"
          f"{size:.1f} MB）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
