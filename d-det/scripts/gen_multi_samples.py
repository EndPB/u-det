#!/usr/bin/env python
"""E22：同题多次采样生成（坍缩度读数前置；**有卡时运行**）。

协议（预注册，2026-09-27 定稿；生成前不得改动）：
  - 任务集：6 族配对共有的 329 题（与全部既有读数同题可比）；
  - 主实验：每题每族 N=8 次采样，temperature=0.7 / top_p=0.95（instruct 侧 chat 模板）；
  - 温度消融：每族排序后前 50 题，另跑 0.2 / 1.0 两档（N=8）；
  - 种子：逐题独立（seed_base + 1000×题序；批内 8 条共享批种子——既有管线口径）。

用法：
  python scripts/gen_multi_samples.py --family qwen05 \
      --model checkpoints/qwen2.5-coder-0.5b-instruct \
      --out data/processed/multisample_qwen05_t0.7.parquet

输出 schema：task_id, family, sample_idx, temperature, seed, x, n_chars, n_tokens_c5
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gen_pairs import load_prompts, load_model, gen_batch  # noqa: E402

PARQ = {
    "qwen05": "data/processed/pairs.parquet",
    "qwen15": "data/processed/pairs_qwen15.parquet",
    "ds13": "data/processed/pairs_ds13.parquet",
    "yi15": "data/processed/pairs_yi15.parquet",
    "granite2b": "data/processed/pairs_granite2b.parquet",
    "smollm2": "data/processed/pairs_smollm2.parquet",
}


def common_tasks() -> list:
    sets = []
    for rel in PARQ.values():
        t = pq.read_table(ROOT / rel, columns=["task_id"])
        sets.append(set(map(str, t.column("task_id").to_pylist())))
    return sorted(set.intersection(*sets))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True)
    ap.add_argument("--model", required=True, help="instruct 模型目录")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--max-prompt-tokens", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=None, help="只用排序后前 N 题（温度消融=50）")
    ap.add_argument("--seed-base", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--prompts-dir", default=str(ROOT / "data" / "raw" / "prompts"))
    ap.add_argument("--codet5", default=str(ROOT / "checkpoints" / "codet5-base"))
    ap.add_argument("--flush-every", type=int, default=16, help="每 N 题落一次盘")
    args = ap.parse_args()

    common = common_tasks()
    print(f"[e22gen] 6 族共同题 = {len(common)}", flush=True)

    prompts = load_prompts(Path(args.prompts_dir))
    by_id = {p["task_id"]: p for p in prompts}
    tasks = [t for t in common if t in by_id]
    missing = set(common) - set(tasks)
    if missing:
        print(f"[e22gen] 警告：{len(missing)} 题在 prompts 中缺失（跳过）", flush=True)
    if args.limit:
        tasks = tasks[:args.limit]
    print(f"[e22gen] 本次生成 {len(tasks)} 题 × {args.n} 采样 = {len(tasks)*args.n} 条"
          f"（family={args.family}, T={args.temperature}）", flush=True)

    from transformers import AutoTokenizer
    corpus_tok = AutoTokenizer.from_pretrained(args.codet5)
    inst_tok, inst_model = load_model(args.model, args.device)
    ns = SimpleNamespace(device=args.device,
                         max_prompt_tokens=args.max_prompt_tokens,
                         temperature=args.temperature, top_p=args.top_p,
                         max_new_tokens=args.max_new_tokens)

    schema = pa.schema([
        ("task_id", pa.string()), ("family", pa.string()), ("sample_idx", pa.int32()),
        ("temperature", pa.float32()), ("seed", pa.int64()),
        ("x", pa.string()), ("n_chars", pa.int32()), ("n_tokens_c5", pa.int32()),
    ])
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    t0 = time.time()
    with pq.ParquetWriter(out_path, schema) as writer:
        bar = tqdm(total=len(tasks), desc=f"e22gen {args.family} T={args.temperature}")
        for i, tid in enumerate(tasks):
            seed = args.seed_base + 1000 * i
            text = by_id[tid]["instruct_text"]
            outs = gen_batch(inst_model, inst_tok, [text] * args.n, ns, chat=True, seed=seed)
            for j, x in enumerate(outs):
                rows.append({
                    "task_id": tid, "family": args.family, "sample_idx": j,
                    "temperature": args.temperature, "seed": seed,
                    "x": x, "n_chars": len(x),
                    "n_tokens_c5": len(corpus_tok(x)["input_ids"]),
                })
    def table_of(rs):
        return pa.table({
            "task_id": pa.array([r["task_id"] for r in rs], pa.string()),
            "family": pa.array([r["family"] for r in rs], pa.string()),
            "sample_idx": pa.array([r["sample_idx"] for r in rs], pa.int32()),
            "temperature": pa.array([r["temperature"] for r in rs], pa.float32()),
            "seed": pa.array([r["seed"] for r in rs], pa.int64()),
            "x": pa.array([r["x"] for r in rs], pa.string()),
            "n_chars": pa.array([r["n_chars"] for r in rs], pa.int32()),
            "n_tokens_c5": pa.array([r["n_tokens_c5"] for r in rs], pa.int32()),
        }, schema=schema)

    rows = []
    t0 = time.time()
    with pq.ParquetWriter(out_path, schema) as writer:
        bar = tqdm(total=len(tasks), desc=f"e22gen {args.family} T={args.temperature}")
        for i, tid in enumerate(tasks):
            seed = args.seed_base + 1000 * i
            text = by_id[tid]["instruct_text"]
            outs = gen_batch(inst_model, inst_tok, [text] * args.n, ns, chat=True, seed=seed)
            for j, x in enumerate(outs):
                rows.append({
                    "task_id": tid, "family": args.family, "sample_idx": j,
                    "temperature": args.temperature, "seed": seed,
                    "x": x, "n_chars": len(x),
                    "n_tokens_c5": len(corpus_tok(x)["input_ids"]),
                })
            bar.update(1)
            if (i + 1) % args.flush_every == 0:
                writer.write_table(table_of(rows))
                rows = []
        if rows:
            writer.write_table(table_of(rows))
            rows = []

    dt = (time.time() - t0) / 60
    print(f"[e22gen] {args.family} T={args.temperature} 完成：{len(tasks)} 题 × {args.n} "
          f"= {len(tasks)*args.n} 条，用时 {dt:.1f} min（{dt*60/max(1,len(tasks)):.1f}s/题）"
          f" -> {out_path.name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
