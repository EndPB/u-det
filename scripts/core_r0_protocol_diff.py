"""R0：complete/instruct 协议差异读出（《核心假设后训练差异与归因多实验指导》§7）。

阶段：
  --stage feats : 加载 115 dual × (train+dev 969 tasks) × 2 modes 文本（test 不装载）→
                  CodeT5-small/base 冻结嵌入 + style/meta/size 特征（本地缓存，gitignore）。
  --stage eval  : 按切分核对每个读出进行 fit/评测（task/model/size/seen），输出结果 JSON。
  --stage report: 汇总 + 报告。

边界：training_allowed=true / generation_allowed=false / test_read_allowed=false；
exploratory 阶段产物 = train/dev only；R0 标签=protocol_conditioned_difference。
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import variant_transfer_stage1_execute as s1  # noqa: E402

OUT = ROOT / "d-det/artifacts/r0_protocol_diff_2026-10-08"
LOCAL = OUT / "local"
RECORDS = ROOT / "d-det/data/public_same_task_full_2026-10-07/records.jsonl"
SPLIT_CSV = ROOT / "d-det/artifacts/public_full_receive_2026-10-08/prereg/split_index.csv"
LOG: list[str] = []


def log(m: str):
    print(m, flush=True)
    LOG.append(m)


def load_split():
    split_of = {}
    with SPLIT_CSV.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            split_of[r["task_id"]] = r["split"]
    return split_of


def load_dual_models():
    model_modes = defaultdict(set)
    for line in open(RECORDS, encoding="utf-8"):
        d = json.loads(line)
        if d.get("subset") == "full":
            model_modes[d["model_id"]].add(d["generation_mode"])
    dual = sorted(m for m, ms in model_modes.items() if ms == {"complete", "instruct"})
    return dual


def build_index():
    split_of = load_split()
    dual = load_dual_models()
    tasks = sorted([t for t, sp in split_of.items() if sp in ("train", "dev")])
    rows = []  # (model, task, mode, split)
    for m in dual:
        for t in tasks:
            for mode in ("complete", "instruct"):
                rows.append((m, t, mode, split_of[t]))
    return dual, tasks, rows


def stage_feats(args):
    t0 = time.time()
    LOCAL.mkdir(parents=True, exist_ok=True)
    (OUT / "logs").mkdir(exist_ok=True)
    dual, tasks, rows = build_index()
    n = len(rows)
    log(f"[feats] units: {len(dual)} models x {len(tasks)} tasks x 2 modes = {n}")
    assert n == len(dual) * len(tasks) * 2

    # ---- 加载文本（test tasks 从不装载；多行组取 code_sha256 序第一行）----
    log("[feats] load texts (train/dev only; multi-row groups keep first by sha order)")
    want = {f"{r[0]}|{r[1]}|{r[2]}": i for i, r in enumerate(rows)}
    picked = {}
    dup_groups = 0
    with RECORDS.open(encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d.get("subset") != "full":
                continue
            k = f"{d['model_id']}|{d['task_id']}|{d['generation_mode']}"
            if k not in want:
                continue
            cs = d.get("solution_sha256") or d.get("code_sha256") or ""
            text = d.get("solution") or d.get("code") or ""
            cur = picked.get(k)
            if cur is None:
                picked[k] = (cs, text)
            else:
                dup_groups += 1
                if cs < cur[0]:
                    picked[k] = (cs, text)
    log(f"[feats] texts picked={len(picked)} dup_extra_rows={dup_groups}")
    texts = []
    valid = np.ones(n, dtype=bool)
    shas = []
    for i, (m, t, mode, sp) in enumerate(rows):
        k = f"{m}|{t}|{mode}"
        cs, tx = picked.get(k, ("", ""))
        if not tx:
            valid[i] = False
        texts.append(tx if tx else " ")
        shas.append(cs)

    # ---- style/meta/sizelen ----
    log("[feats] style/meta/size (CPU)")
    t = time.time()
    style_mat = np.array([s1.style_features(tx) for tx in texts], dtype=np.float64)
    meta_mat = style_mat[:, s1.META_IDX].copy()
    log(f"  style {style_mat.shape} in {time.time()-t:.1f}s")
    # size (params) per model; parsed from name
    import re
    def parse_params(mid):
        mm = re.search(r"(\d+(?:\.\d+)?)\s*[bB](?![a-zA-Z])", mid)
        return float(mm.group(1)) if mm else np.nan
    sizeB = np.array([parse_params(m) for (m, _, _, _) in rows], dtype=np.float64)
    szfeat = np.array([[np.log10(sizeB[i]) if not np.isnan(sizeB[i]) else np.nan,
                        style_mat[i, 0], style_mat[i, 1]] for i in range(n)], dtype=np.float64)
    np.savez_compressed(LOCAL / "style_meta.npz",
                        style=style_mat.astype(np.float32), meta=meta_mat.astype(np.float32),
                        sizelen=szfeat.astype(np.float32), valid=valid, sizeB=sizeB)
    log(f"  saved style_meta.npz")
    # 文本缓存（本地；供 P0/TF-IDF 读出使用；不进 git）
    with gzip.open(LOCAL / "texts.jsonl.gz", "wt", encoding="utf-8") as f:
        for i, (m, t, mode, sp) in enumerate(rows):
            f.write(json.dumps({"i": i, "text": texts[i]}, ensure_ascii=False) + "\n")
    log(f"  saved texts.jsonl.gz")

    # ---- CodeT5 embeddings ----
    log("[feats] CodeT5-small embedding (GPU)")
    t = time.time()
    emb_small = s1.encode_codet5(s1.MODEL_SMALL, texts, device_batch=16)
    log(f"  small {emb_small.shape} in {time.time()-t:.1f}s")
    np.savez_compressed(LOCAL / "emb_small.npz", emb=emb_small.astype(np.float32))
    log("[feats] CodeT5-base embedding (GPU)")
    t = time.time()
    emb_base = s1.encode_codet5(s1.MODEL_BASE, texts, device_batch=8)
    log(f"  base {emb_base.shape} in {time.time()-t:.1f}s")
    np.savez_compressed(LOCAL / "emb_base.npz", emb=emb_base.astype(np.float32))

    # ---- row index ----
    idx = {"models": dual, "tasks": tasks, "n": n,
           "order": "model-major: for model: for task: for mode[complete,instruct]",
           "row_keys": [f"{m}|{t}|{mode}" for (m, t, mode, sp) in rows],
           "splits": [sp for (_, _, _, sp) in rows],
           "shas": shas, "valid": valid.tolist(),
           "env": {"python": sys.version.split()[0], "numpy": np.__version__,
                   "sklearn": __import__("sklearn").__version__}}
    (LOCAL / "row_index.json").write_text(json.dumps(idx, ensure_ascii=False), encoding="utf-8")
    man = {"schema": "r0_feature_manifest_v1", "generated_utc": datetime.now(timezone.utc).isoformat(),
           "n_rows": n, "n_models": len(dual), "n_tasks": len(tasks),
           "invalid_text_rows": int((~valid).sum()),
           "files": {}}
    for fn in ("style_meta.npz", "emb_small.npz", "emb_base.npz"):
        p = LOCAL / fn
        man["files"][fn] = {"bytes": p.stat().st_size, "sha256": sha256_file(p)}
    (LOCAL / "feature_manifest.json").write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "logs" / "r0_feats.log").write_text("\n".join(LOG) + "\n", encoding="utf-8")
    log(f"[feats] done in {time.time()-t0:.1f}s")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["feats", "eval", "report"])
    ap.add_argument("--split", default="")
    ap.add_argument("--fold", type=int, default=-1)
    ap.add_argument("--rep", default="base", choices=["base", "small"])
    ap.add_argument("--light", action="store_true", help="跳过 TF-IDF/LGBM 读出")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "logs").mkdir(exist_ok=True)
    if args.stage == "feats":
        stage_feats(args)
    elif args.stage == "eval":
        import core_r0_eval as ev
        ev.run_split(args.split, args.fold, heavy=not args.light, rep=args.rep)
    elif args.stage == "report":
        import core_r0_eval as ev
        ev.make_report()


if __name__ == "__main__":
    main()
