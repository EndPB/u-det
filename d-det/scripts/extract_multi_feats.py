#!/usr/bin/env python
"""E22 特征：为多采样文本缓存 v1.0 冻结编码器的池化特征（768 维；**有卡时运行**）。

输入：data/processed/multisample_<family>_t{0.7,0.2,1.0}.parquet
输出：runs/kernel_e22/multi_<family>_t{...}.npz（emb, tasks, sample_idx, temperature, n_chars）

运行：OMP_NUM_THREADS=8 /root/miniconda3/envs/udet/bin/python scripts/extract_multi_feats.py
"""
from __future__ import annotations

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

OUT = ROOT / "runs/kernel_e22"
FAMILIES = ["qwen05", "qwen15", "ds13", "yi15", "granite2b", "smollm2"]
TEMPS = ["t0.7", "t0.2", "t1.0"]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[e22feat] device={device}", flush=True)

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

    done = 0
    for name in FAMILIES:
        for tt in TEMPS:
            src = ROOT / f"data/processed/multisample_{name}_{tt}.parquet"
            dst = OUT / f"multi_{name}_{tt}.npz"
            if not src.exists():
                continue
            if dst.exists():
                print(f"[e22feat] 跳过 {name} {tt}（已存在）", flush=True)
                continue
            t = pq.read_table(src)
            xs = t.column("x").to_pylist()
            tasks = [str(v) for v in t.column("task_id").to_pylist()]
            sidx = np.asarray(t.column("sample_idx").to_pylist(), dtype="int64")
            n_chars = np.asarray(t.column("n_chars").to_pylist(), dtype="int64")
            emb = encode_codes(dual, xs, tok, device).numpy()
            np.savez_compressed(dst, emb=emb.astype("float32"),
                                tasks=np.array(tasks, dtype=object),
                                sample_idx=sidx, n_chars=n_chars,
                                temperature=np.float64(float(tt[1:])))
            done += 1
            print(f"[e22feat] {name} {tt}: {len(xs)} 条 -> {dst.name}", flush=True)
    print(f"[e22feat] done（{done} 个新文件）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
