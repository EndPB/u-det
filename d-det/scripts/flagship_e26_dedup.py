#!/usr/bin/env python
"""E26 前置诊断 E：ids→code 恢复 + 跨 split 近重复审计（不训练、不生成）。

路径（经 semeval_prepare_big.py / semeval_prepare.py 核实）：
  b_train ← task_b_training_set.parquet（抽样后重排）
  b_val   ← task_b_validation_set.parquet（抽样）
  b_test  ← task_b_test_set_sample.parquet（全量 1K）
  分词规则：CodeT5 `add_special_tokens=False`；n>1024 → ids[:768]+ids[-256:]；n<8 跳过。
恢复：对三个源文件全量（再）分词 → ids 哈希匹配 → 找回采样样本的 code。
审计：精确（原文 / 去空白归一）；近重复（标识符 5-gram MinHash，16 带 LSH，验证 Jaccard）。

输出：runs/flagship_e26/e26_dedup.json
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from argparse import Namespace
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from flagship_e25_stage0 import load_corpus_tagged  # noqa: E402

OUT = ROOT / "runs/flagship_e26"
RAW = {
    "train": ROOT / "data/raw/SemEval-2026-Task13/task_b/task_b_training_set.parquet",
    "val": ROOT / "data/raw/SemEval-2026-Task13/task_b/task_b_validation_set.parquet",
    "test": ROOT / "data/raw/SemEval-2026-Task13/task_b/task_b_test_set_sample.parquet",
}
ARGS = Namespace(smoke=False, max_len=2048, train_human=8000, train_fam=1200,
                 val_human=1500, val_fam=500, test_human=500, test_fam=200,
                 unseen_cap=600, enc_bs=32)
TOK_RE = re.compile(r"[A-Za-z_]\w*|\d+|[^\sA-Za-z0-9_]")


def ids_md5(ids) -> str:
    return hashlib.md5(np.ascontiguousarray(ids, dtype=np.int64).tobytes()).hexdigest()


def shingles(code: str, k: int = 5):
    toks = TOK_RE.findall(code)
    if len(toks) < k:
        return np.empty(0, dtype=np.int64)
    hs = {hash(tuple(toks[i:i + k])) & 0xFFFFFFFF for i in range(len(toks) - k + 1)}
    return np.fromiter(hs, dtype=np.int64)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    corpus = load_corpus_tagged(ARGS)

    # ---- 查询哈希表 ----
    queries = defaultdict(list)                       # hash -> [(split, idx)]
    for sp, docs in corpus.items():
        for i, d in enumerate(docs):
            queries[ids_md5(d.ids)].append((sp, i))
    print(f"[dedup] 采样样本 {sum(len(v) for v in corpus.values())}，"
          f"唯一哈希 {len(queries)}", flush=True)

    tok = AutoTokenizer.from_pretrained(str(ROOT / "checkpoints/codet5-base"))
    recovered = {}                                    # (sp, idx) -> {"code", "gen", "lang"}
    coverage_src = {}
    for src_sp, path in RAW.items():
        pf = pq.ParquetFile(str(path))
        n_rows = pf.metadata.num_rows
        t1 = time.time(); seen = 0; hit = 0
        for batch in pf.iter_batches(batch_size=2048,
                                     columns=["code", "generator", "language"]):
            codes = batch.column("code").to_pylist()
            gens = batch.column("generator").to_pylist()
            langs = batch.column("language").to_pylist()
            enc = tok(codes, add_special_tokens=False)
            for j, ids in enumerate(enc["input_ids"]):
                seen += 1
                cands = [ids]
                if len(ids) > 1024:
                    cands.append(ids[:768] + ids[-256:])
                for cd in cands:
                    h = ids_md5(np.asarray(cd))
                    if h in queries:
                        for (sp, i) in queries[h]:
                            if corpus[sp][i].split == src_sp and (sp, i) not in recovered:
                                recovered[(sp, i)] = {"code": codes[j],
                                                      "gen": str(gens[j]),
                                                      "lang": str(langs[j])}
                                hit += 1
        coverage_src[src_sp] = {"rows": n_rows, "seen": seen, "matched": hit,
                                "minutes": round((time.time() - t1) / 60, 1)}
        print(f"[dedup] 源 {src_sp}: {hit} 命中 / {seen} 行 / {n_rows} 总 "
              f"（{coverage_src[src_sp]['minutes']} min）", flush=True)

    coverage = {sp: {"n": len(docs), "recovered": sum(
        1 for i in range(len(docs)) if (sp, i) in recovered)}
        for sp, docs in corpus.items()}
    print(f"[dedup] 覆盖率 {coverage}", flush=True)

    # ---- 近重复审计（仅恢复样本） ----
    keys = sorted(recovered.keys(), key=lambda k: (k[0], k[1]))
    codes = [recovered[k]["code"] for k in keys]
    norm = ["".join(c.split()) for c in codes]
    norm_md5 = [hashlib.md5(c.encode("utf-8", "ignore")).hexdigest() for c in norm]
    exact_groups = defaultdict(list)
    for idx, h in enumerate(norm_md5):
        exact_groups[h].append(idx)
    exact_dups = {h: g for h, g in exact_groups.items() if len(g) > 1}

    t2 = time.time()
    sh = [shingles(c) for c in codes]                  # 每个样本的去重 shingle 数组
    rng = np.random.RandomState(0)
    A = rng.randint(1, 1 << 31, size=64); B = rng.randint(0, 1 << 31, size=64)
    P = (1 << 61) - 1
    sigs = []
    for s in sh:
        if len(s) == 0:
            sigs.append(np.zeros(64, dtype=np.int64)); continue
        X = s[:, None]
        sig = ((X * A[None, :] + B[None, :]) % P).min(0)
        sigs.append(sig)
    sigs = np.stack(sigs)                              # (N, 64)
    print(f"[dedup] shingle+签名 {len(sh)} 条 / {(time.time()-t2)/60:.1f} min；"
          f"平均 shingles {np.mean([len(s) for s in sh]):.0f}", flush=True)

    pairs = set()
    for b in range(16):
        buckets = defaultdict(list)
        for i in range(len(keys)):
            buckets[tuple(sigs[i, b * 4:(b + 1) * 4].tolist())].append(i)
        for group in buckets.values():
            if len(group) < 2:
                continue
            for a in range(len(group)):
                for c in range(a + 1, len(group)):
                    pairs.add((group[a], group[c]))
    print(f"[dedup] LSH 候选对 {len(pairs)}", flush=True)

    def jac(i, j):
        si, sj = sh[i], sh[j]
        if len(si) == 0 or len(sj) == 0:
            return 0.0
        inter = np.intersect1d(si, sj, assume_unique=True).size
        return float(inter) / (len(si) + len(sj) - inter)

    audit_summary_note = "exact=去空白归一精确重复；near=标识符 5-gram Jaccard≥0.7"
    ex_summary = {}
    near = defaultdict(list)
    for (i, j) in pairs:
        J = jac(i, j)
        if J >= 0.7:
            ki, kj = keys[i], keys[j]
            cat = f"{ki[0]}↔{kj[0]}"
            near[cat].append({"J": round(J, 3),
                              "a": {"split": ki[0], "gen": recovered[ki]["gen"]},
                              "b": {"split": kj[0], "gen": recovered[kj]["gen"]}})
    ex_summary = {}
    for h, g in exact_dups.items():
        cats = Counter()
        for idx in g:
            cats[keys[idx][0]] += 1
        cat = "↔".join(sorted(cats))
        ex_summary.setdefault(cat, []).append(
            {"n": len(g), "splits": dict(cats),
             "gen": [recovered[keys[i]]["gen"] for i in g[:3]]})
    report = {
        "coverage_source": coverage_src,
        "coverage_sampled": coverage,
        "exact_norm_dup_groups": {k: v[:3] for k, v in ex_summary.items()},
        "exact_norm_dup_group_count": {k: len(v) for k, v in ex_summary.items()},
        "near_dup_pairs": {k: {"n": len(v), "examples": v[:3]}
                           for k, v in sorted(near.items())},
        "near_dup_pair_counts": {k: len(v) for k, v in sorted(near.items())},
        "method": {"shingle": "identifier 5-gram", "minhash": 64, "lsH": "16x4",
                   "threshold": 0.7},
        "timing_min": round((time.time() - t0) / 60, 1),
    }
    (OUT / "e26_dedup.json").write_text(json.dumps(report, ensure_ascii=False,
                                                   indent=2, default=str))
    print(f"[dedup] 完成 {report['timing_min']} min → 近重复对 "
          f"{report['near_dup_pair_counts']}；精确组 {report['exact_norm_dup_group_count']}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
