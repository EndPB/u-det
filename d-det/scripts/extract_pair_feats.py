#!/usr/bin/env python
"""K2-full 前置：为 6 族配对文本缓存双侧池化特征 h(x+) / h(x−)（v1.0 冻结编码器）。

与 extract_delta_feat.py 的区别：不存差值 Δ，而是分别存两侧池化向量（768 维），
供 DCAN 式"训练版残差分离"（scripts/kernel_e16_dcan.py）使用：
  · 单侧任务（文本级族归因）：样本 = h
  · 配对任务（配对级族归因）：输入 = Δ 或 [h+; h−]
附带任务号（题级切分与 GroupKFold 用）。

运行：OMP_NUM_THREADS=8 /root/miniconda3/envs/udet/bin/python scripts/extract_pair_feats.py
输出：runs/kernel_e16/h_{family}.npz（h_plus, h_minus, tasks）
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

OUT = ROOT / "runs/kernel_e16"
PARQ = {
    "qwen05": "data/processed/pairs.parquet",
    "qwen15": "data/processed/pairs_qwen15.parquet",
    "ds13": "data/processed/pairs_ds13.parquet",
    "yi15": "data/processed/pairs_yi15.parquet",
    "granite2b": "data/processed/pairs_granite2b.parquet",
    "smollm2": "data/processed/pairs_smollm2.parquet",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="只处理单族（冒烟用）")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[feat] device={device}", flush=True)

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

    for name, rel in PARQ.items():
        if args.only and name != args.only:
            continue
        out = OUT / f"h_{name}.npz"
        if out.exists():
            print(f"[feat] 跳过 {name}（已存在）", flush=True)
            continue
        t = pq.read_table(ROOT / rel)
        xp = t.column("x_plus").to_pylist()
        xm = t.column("x_minus").to_pylist()
        tasks = [str(x) for x in t.column("task_id").to_pylist()]
        hp = encode_codes(dual, xp, tok, device).numpy()
        hm = encode_codes(dual, xm, tok, device).numpy()
        np.savez_compressed(out, h_plus=hp.astype("float32"),
                            h_minus=hm.astype("float32"),
                            tasks=np.array(tasks, dtype=object))
        print(f"[feat] {name}: {len(tasks)} 对 -> {out.name}", flush=True)
    print("[feat] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
