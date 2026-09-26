#!/usr/bin/env python
"""通用单 run probs 导出（run 目录内 + runs/semeval_r 别名），支持任意编码器配置。

用法：python scripts/semeval_dump_run.py --run runs/semeval_qwen_b --task b \
        --config configs/ddet_qwen.yaml --init none [--data-dir data/processed/semeval_qwen]
init: 与训练一致（仅占位；probs 由 best.pt 状态决定）；--init none 时跳过底座权重加载。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from encoders import build_encoder  # noqa: E402
from models import build_model  # noqa: E402
from semeval_finetune import SevDataset, TaskModel, pad_collate  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--task", default="b")
    ap.add_argument("--config", default="configs/ddet_v041.yaml")
    ap.add_argument("--data-dir", default="data/processed/semeval")
    ap.add_argument("--init", default="runs/v0.4.1_covreg/last.pt")
    ap.add_argument("--readout-rank", type=int, default=0)
    ap.add_argument("--tag", default=None, help="runs/semeval_r 别名（默认 run 目录名）")
    ap.add_argument("--clean", action="store_true", help="导出后删除 run 内 best.pt（省盘）")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    with open(ROOT / args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    enc_cfg = dict(cfg["encoder"])
    enc_cfg["path"] = str(ROOT / enc_cfg["path"])
    enc = build_encoder(**enc_cfg)
    enc.requires_grad_(False)
    dual = build_model("dual", encoder=enc, dim=enc.hidden_size,
                       pool=cfg["model"].get("pool", "mean"),
                       s2_rank=cfg["model"].get("s2_rank", 1))
    if str(args.init).lower() not in ("none", "no", ""):
        ck0 = ROOT / args.init
        if ck0.exists():
            ck = torch.load(str(ck0), map_location="cpu", weights_only=False)
            dual.load_state_dict(ck.get("state", {}), strict=False)
            print(f"[dump_run] 底座初始权重 ← {args.init}")
    ncls = {"a": 2, "b": 11, "c": 4}[args.task]
    model = TaskModel(dual, args.task, ncls, s2_rank=args.readout_rank).to(device)
    run = ROOT / args.run
    st = torch.load(str(run / "best.pt"), map_location="cpu", weights_only=False)
    miss, unexp = model.load_state_dict(st["state"], strict=False)
    print(f"[dump_run] {args.run} missing={len(miss)} unexpected={len(unexp)}")
    model.eval()
    tag = args.tag or run.name
    (ROOT / "runs/semeval_r").mkdir(parents=True, exist_ok=True)
    for split in ("val", "test"):
        ds = SevDataset(str(ROOT / args.data_dir / f"{args.task}_{split}.parquet"))
        dl = DataLoader(ds, batch_size=8, collate_fn=pad_collate, num_workers=0)
        probs, Y = [], []
        with torch.no_grad():
            for b in dl:
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                    lg = model.logits(b["input_ids"].to(device),
                                      b["attention_mask"].to(device))
                probs.append(torch.softmax(lg.float(), -1).cpu())
                Y.extend(b["labels"].tolist())
        P = torch.cat(probs).numpy()
        np.savez_compressed(run / f"probs_{split}.npz", probs=P, y=np.array(Y))
        np.savez_compressed(ROOT / "runs/semeval_r" / f"probs_{tag}_{split}.npz",
                            probs=P, y=np.array(Y))
        print(f"[dump_run] {tag}/{split} OK（n={len(Y)}）", flush=True)
    if args.clean:
        for f in ("last.pt", "best.pt"):
            p = run / f
            if p.exists():
                p.unlink()
                print(f"[dump_run] 已删除 {p}")
    print("[dump_run] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
