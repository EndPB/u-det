#!/usr/bin/env python
"""临时探针：旗舰轮实验（E24）前的编码吞吐与长度分布测量。

输出：docs/s、tokens/s、n_tokens 分位数（b_train / b_val / b_test）。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from encoders import build_encoder  # noqa: E402
from models import build_model  # noqa: E402


def main() -> int:
    for f in ("b_train", "b_val", "b_test"):
        t = pq.read_table(ROOT / f"data/processed/semeval_big/{f}.parquet",
                          columns=["n_tokens"])
        n = np.asarray(t.column("n_tokens").to_pylist())
        q = np.percentile(n, [50, 90, 95, 99, 100]).astype(int)
        print(f"[probe] {f}: n={len(n)} n_tokens p50/p90/p95/p99/max = {list(q)}", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    with open(ROOT / "configs/ddet_v041.yaml", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    enc_cfg = dict(cfg["encoder"])
    enc_cfg["path"] = str(ROOT / enc_cfg["path"])
    enc = build_encoder(**enc_cfg)
    dual = build_model("dual", encoder=enc, dim=enc.hidden_size,
                       pool=cfg["model"].get("pool", "mean"),
                       s2_rank=cfg["model"].get("s2_rank", 1))
    ck = torch.load(str(ROOT / "runs/v0.4.1_covreg/last.pt"), map_location="cpu",
                    weights_only=False)
    dual.load_state_dict(ck["state"], strict=False)
    enc = dual.encoder.to(device).eval()

    t = pq.read_table(ROOT / "data/processed/semeval_big/b_train.parquet",
                      columns=["ids"])
    ids_all = t.column("ids").to_pylist()
    rng = np.random.RandomState(0)
    pick = rng.choice(len(ids_all), 256, replace=False)
    docs = [np.asarray(ids_all[i][:2048], dtype="int64") for i in pick]
    docs.sort(key=len)
    print(f"[probe] 样本 256 条，平均长度 {np.mean([len(d) for d in docs]):.0f} tokens", flush=True)

    def batch_of(chunk):
        maxL = max(len(d) for d in chunk)
        ids = torch.zeros(len(chunk), maxL, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for j, d in enumerate(chunk):
            ids[j, :len(d)] = torch.from_numpy(d)
            mask[j, :len(d)] = 1
        return ids.to(device), mask.to(device)

    bs = 16
    t0 = time.time()
    ntok = 0
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16,
                                         enabled=device == "cuda"):
        for i in range(0, len(docs), bs):
            chunk = docs[i:i + bs]
            ids, mask = batch_of(chunk)
            out = enc(ids, mask)
            ntok += int(mask.sum())
    dt = time.time() - t0
    print(f"[probe] 编码 {len(docs)} 条 / {ntok} tokens：{dt:.1f}s → "
          f"{len(docs)/dt:.1f} docs/s，{ntok/dt:.0f} tokens/s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
