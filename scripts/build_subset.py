#!/usr/bin/env python
"""均衡子集构建（单文件脚本）：CoDET-M4 均衡采样 + HybridCodeAuthorship 全量，预分词后落盘。

流程：
    m4      ：流式扫全量 parquet（50 万行）-> 按 (target × 语言 × 字符长度桶) 蓄水池采样候选
              -> 预分词得到精确 token 长度 -> 按 (类别 × token 长度桶 × 语言) 二次均衡
              -> 按 code 哈希切 train/val/test（8:1:1）-> data/processed/m4.parquet
    hybrid  ：全量读取 -> 行级 Attribution 映射为 token 级标签（按行内字符位置对齐）
              -> 过滤过短样本 -> 按 RecordId 哈希切分 -> data/processed/hybrid.parquet

用法::

    python scripts/build_subset.py                # 两者都建（m4 每类 1w）
    python scripts/build_subset.py m4 --n 10000
    python scripts/build_subset.py hybrid
"""

from __future__ import annotations

import argparse
import hashlib
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataio.hybrid import read_hybrid, tokenize_with_line_labels  # noqa: E402

# token 长度分桶边界（用于"各种长度均衡"）；字符桶按约 3.5 字符/token 对齐
TOKEN_EDGES = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
CHAR_EDGES = [100, 200, 400, 900, 1800, 3600, 7200, 14000, 28000]
SPLITS = (("train", 8), ("val", 1), ("test", 1))   # 按哈希取模 10 分组


def bucket_of(n: int, edges: list[int]) -> int:
    """返回 n 落在第几个桶（edges 为桶下界，依次递增）。"""
    for i in range(len(edges) - 1, -1, -1):
        if n >= edges[i]:
            return i
    return -1


def split_of(key: str) -> str:
    """按内容哈希分组切分：同一哈希不会跨集。"""
    digest = int.from_bytes(hashlib.blake2b(key.encode("utf-8", "ignore"), digest_size=8).digest(), "little")
    r = digest % 10
    acc = 0
    for name, weight in SPLITS:
        acc += weight
        if r < acc:
            return name
    return "test"


def waterfill(supply: dict, quota: int) -> dict:
    """把 quota 尽量均匀地分配到 supply 各单元格（受供给上限约束）。"""
    take = {k: 0 for k in supply}
    remaining = min(quota, sum(supply.values()))
    while remaining > 0:
        active = [k for k in supply if take[k] < supply[k]]
        if not active:
            break
        share = max(1, remaining // len(active))
        moved = 0
        for k in active:
            add = min(share, supply[k] - take[k], remaining)
            take[k] += add
            remaining -= add
            moved += add
        if moved == 0:
            break
    return take


def balanced_pick(items: list[dict], quota: int, axes: list[str], seed: int) -> list[dict]:
    """按 axes（如 ['target', 'bucket', 'lang']）分层，尽量均匀地选出 quota 条。"""
    cells = defaultdict(list)
    for it in items:
        cells[tuple(it[a] for a in axes)].append(it)
    supply = {k: len(v) for k, v in cells.items()}
    take = waterfill(supply, quota)
    rng = random.Random(seed)
    picked = []
    for key, n in take.items():
        if n <= 0:
            continue
        pool = cells[key]
        rng.shuffle(pool)
        picked.extend(pool[:n])
    return picked


def table(rows: list[tuple], row_name: str) -> str:
    """把 (行键, 列键) 计数渲染成文本表。"""
    counts = Counter(rows)
    row_keys = sorted({r for r, _ in rows}, key=str)
    col_keys = sorted({c for _, c in rows}, key=str)
    lines = ["  " + f"{row_name:<12}" + "".join(f"{str(c):>10}" for c in col_keys) + f"{'合计':>10}"]
    for r in row_keys:
        cells = [counts.get((r, c), 0) for c in col_keys]
        lines.append("  " + f"{str(r):<12}" + "".join(f"{v:>10}" for v in cells) + f"{sum(cells):>10}")
    totals = [sum(counts.get((r, c), 0) for r in row_keys) for c in col_keys]
    lines.append("  " + f"{'合计':<12}" + "".join(f"{v:>10}" for v in totals) + f"{sum(totals):>10}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# m4
# --------------------------------------------------------------------------- #
def build_m4(args, tokenizer) -> Path:
    from dataio.m4 import iter_m4

    seed = args.seed
    rng = random.Random(seed)
    seen: set[int] = set()
    reservoirs: dict[tuple, list[dict]] = defaultdict(list)
    counts: Counter = Counter()
    lang_counter, split_counter = Counter(), Counter()
    row = 0
    for batch in iter_m4(columns=["cleaned_code", "target", "language", "model", "split"]):
        for code, target, lang, model, raw_split in zip(
            batch["cleaned_code"], batch["target"], batch["language"], batch["model"], batch["split"]
        ):
            row += 1
            if row % 100000 == 0:
                print(f"  [m4] 已扫描 {row} 行…")
            if not code or target not in ("human", "ai") or not lang:
                continue
            code = code.strip()
            if not (CHAR_EDGES[0] <= len(code) <= CHAR_EDGES[-1]):
                continue
            digest = hashlib.blake2b(code.encode("utf-8", "ignore"), digest_size=8).digest()
            key = int.from_bytes(digest, "little")
            if key in seen:                      # 跨类别同文去重（同一段代码只留一条）
                continue
            seen.add(key)
            lang_counter[lang] += 1
            split_counter[raw_split] += 1
            cell = (target, lang, bucket_of(len(code), CHAR_EDGES))
            item = {"code": code, "target": target, "lang": lang, "model": model, "hash": key}
            reservoir = reservoirs[cell]
            counts[cell] += 1
            if len(reservoir) < args.cand_per_cell:            # 蓄水池采样（算法 R，数量无偏）
                reservoir.append(item)
            else:
                j = rng.randrange(counts[cell])
                if j < args.cand_per_cell:
                    reservoir[j] = item
    print(f"  [m4] 扫描完成：{row} 行；语言分布 {dict(lang_counter.most_common(8))}")
    print(f"  [m4] 原始 split 列分布：{dict(split_counter)}")

    # 语言只保留供给最多的前 K 个，避免长尾噪声
    top_langs = {lang for lang, _ in lang_counter.most_common(args.langs)}
    candidates = [it for group in reservoirs.values() for it in group if it["lang"] in top_langs]
    print(f"  [m4] 候选池 {len(candidates)} 条（{len(top_langs)} 种语言）-> 预分词…")

    # 预分词：得到精确 token 长度，同时直接存下 input_ids
    final: list[dict] = []
    for start in range(0, len(candidates), 512):
        chunk = candidates[start:start + 512]
        enc = tokenizer([c["code"] for c in chunk], add_special_tokens=False,
                        truncation=True, max_length=args.max_tokens)
        for item, ids in zip(chunk, enc["input_ids"]):
            n = len(ids)
            if n < args.min_tokens:
                continue
            item["ids"] = ids
            item["n_tokens"] = n
            item["bucket"] = bucket_of(n, TOKEN_EDGES)
            final.append(item)

    # 二次均衡：类别 × token 长度桶 × 语言
    per_class = args.n
    picked: list[dict] = []
    for target in ("human", "ai"):
        pool = [it for it in final if it["target"] == target]
        if not pool:
            continue
        use = balanced_pick(pool, per_class, ["bucket", "lang"], seed)
        for it in use:
            it["split"] = split_of(str(it["hash"]))
            it["label"] = 0 if target == "human" else 1
        picked.extend(use)
        print(f"  [m4] {target}: 候选 {len(pool)} -> 选中 {len(use)}")
        print(table([(it["bucket"], it["lang"]) for it in use], "桶\\语言"))

    out = Path(args.processed_dir) / "m4.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = {
        "split": pa.array([it["split"] for it in picked], pa.string()),
        "target": pa.array([it["target"] for it in picked], pa.string()),
        "language": pa.array([it["lang"] for it in picked], pa.string()),
        "model": pa.array([it["model"] for it in picked], pa.string()),
        "code": pa.array([it["code"] for it in picked], pa.string()),
        "input_ids": pa.array([it["ids"] for it in picked], pa.list_(pa.int32())),
        "n_tokens": pa.array([it["n_tokens"] for it in picked], pa.int32()),
    }
    pq.write_table(pa.table(fields), out)
    print(f"  [m4] 写出 {out}（{len(picked)} 条，{out.stat().st_size / 1048576:.1f} MB）")
    print(table([(it["bucket"], it["split"]) for it in picked], "桶\\split"))
    return out


# --------------------------------------------------------------------------- #
# hybrid
# --------------------------------------------------------------------------- #
def build_hybrid(args, tokenizer) -> Path:
    rows = read_hybrid(args.hybrid_raw)
    print(f"  [hybrid] 原始 {len(rows)} 条 -> 行级分词与标签对齐…")
    picked, skipped = [], Counter()
    for i, row in enumerate(rows):
        if i % 2000 == 0 and i:
            print(f"  [hybrid] 已处理 {i}/{len(rows)}…")
        code = row["AICode"]
        attribution = row["Attribution"]
        if not code or not attribution:
            skipped["空"] += 1
            continue
        if sum(1 for a in attribution if a == "AI") == 0:
            skipped["无 AI 行"] += 1
            continue
        tokens = tokenize_with_line_labels(tokenizer, code, attribution)
        n = len(tokens["input_ids"])
        if n < args.min_tokens:
            skipped["过短"] += 1
            continue
        if n > args.hybrid_max_tokens:
            skipped["过长"] += 1
            continue
        tokens.update({
            "RecordId": row["RecordId"], "ModelId": row["ModelId"], "Language": row["Language"],
            "code": code, "label": 1, "n_tokens": n, "bucket": bucket_of(n, TOKEN_EDGES),
            "split": split_of(str(row["RecordId"])),
        })
        picked.append(tokens)

    print(f"  [hybrid] 保留 {len(picked)} 条；跳过 {dict(skipped)}")
    out = Path(args.processed_dir) / "hybrid.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = {
        "split": pa.array([p["split"] for p in picked], pa.string()),
        "RecordId": pa.array([p["RecordId"] for p in picked], pa.string()),
        "ModelId": pa.array([p["ModelId"] for p in picked], pa.string()),
        "Language": pa.array([p["Language"] for p in picked], pa.string()),
        "code": pa.array([p["code"] for p in picked], pa.string()),
        "input_ids": pa.array([p["input_ids"] for p in picked], pa.list_(pa.int32())),
        "tok_labels": pa.array([p["tok_labels"] for p in picked], pa.list_(pa.int32())),
        "line_of_token": pa.array([p["line_of_token"] for p in picked], pa.list_(pa.int32())),
        "line_label": pa.array([p["line_label"] for p in picked], pa.list_(pa.int32())),
        "label": pa.array([p["label"] for p in picked], pa.int32()),
        "n_tokens": pa.array([p["n_tokens"] for p in picked], pa.int32()),
    }
    pq.write_table(pa.table(fields), out)
    print(f"  [hybrid] 写出 {out}（{len(picked)} 条，{out.stat().st_size / 1048576:.1f} MB）")
    print(table([(p["bucket"], p["split"]) for p in picked], "桶\\split"))
    print("  [hybrid] 生成模型分布：" + str(dict(Counter(p["ModelId"] for p in picked))))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="构建均衡子集（m4 / hybrid）")
    parser.add_argument("target", nargs="?", default="all", choices=["m4", "hybrid", "all"])
    parser.add_argument("--n", type=int, default=10000, help="m4 每类目标数量（默认 1w）")
    parser.add_argument("--langs", type=int, default=6, help="m4 保留的语言数（按供给排名）")
    parser.add_argument("--min-tokens", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=4096, help="m4 单条 token 上限（超出截断）")
    parser.add_argument("--hybrid-max-tokens", type=int, default=16384, help="hybrid 单条 token 上限（超长丢弃）")
    parser.add_argument("--cand-per-cell", type=int, default=1200, help="m4 每个 (类别×语言×字符桶) 蓄水池大小")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--processed-dir", default=str(ROOT / "data" / "processed"))
    parser.add_argument("--m4-raw", default=str(ROOT / "data" / "raw" / "CoDET-M4" / "dataset_without_comments.parquet"))
    parser.add_argument("--hybrid-raw", default=str(ROOT / "data" / "raw" / "HybridCodeAuthorship" / "hybrid.parquet"))
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(ROOT / "checkpoints" / "codet5-base"))
    print(f"[build_subset] 分词器：{type(tokenizer).__name__}（vocab {tokenizer.vocab_size}）")

    if args.target in ("m4", "all"):
        print("[build_subset] === CoDET-M4 均衡子集 ===")
        build_m4(args, tokenizer)
    if args.target in ("hybrid", "all"):
        print("[build_subset] === HybridCodeAuthorship ===")
        build_hybrid(args, tokenizer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
