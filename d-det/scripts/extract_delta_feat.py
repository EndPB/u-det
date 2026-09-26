#!/usr/bin/env python
"""通用：pairs parquet → Δ 特征 npz（CodeT5 v1.0 池化 h，Δ = h(x+) − h(x−)）。

用法：python scripts/extract_delta_feat.py --pairs data/processed/pairs_X.parquet \
        --out runs/v0.5_disc/d_X.npz
与 kernel_e11 特征口径一致（encode_codes：[1024] 截断 + v0.4.1_covreg 编码器）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import yaml
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from encoders import build_encoder  # noqa: E402
from models import build_model  # noqa: E402
from kernel_e2_loss import encode_codes  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    t = pq.read_table(ROOT / args.pairs)
    xp = t.column("x_plus").to_pylist()
    xm = t.column("x_minus").to_pylist()
    tasks = [str(x) for x in t.column("task_id").to_pylist()] if "task_id" in t.schema.names \
        else [str(i) for i in range(len(xp))]
    print(f"[delta] {args.pairs}: {len(xp)} 对", flush=True)

    with open(ROOT / "configs/ddet_v041.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    enc_cfg = dict(cfg["encoder"])
    enc_cfg["path"] = str(ROOT / enc_cfg["path"])
    enc = build_encoder(**enc_cfg)
    dual = build_model("dual", encoder=enc, dim=enc.hidden_size,
                       pool=cfg["model"].get("pool", "mean"),
                       s2_rank=cfg["model"].get("s2_rank", 1))
    ck = torch.load(str(ROOT / "runs/v0.4.1_covreg/last.pt"), map_location="cpu",
                    weights_only=False)
    dual.load_state_dict(ck["state"], strict=False)
    dual = dual.to(device).eval()
    tok = AutoTokenizer.from_pretrained(enc_cfg["path"])

    hp = encode_codes(dual, xp, tok, device).numpy()
    hm = encode_codes(dual, xm, tok, device).numpy()
    d = (hp - hm).astype("float32")
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, d=d, tasks=np.array(tasks, dtype=object))
    print(f"[delta] 写出 {out}（d={d.shape}）", flush=True)
    print("[delta] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
