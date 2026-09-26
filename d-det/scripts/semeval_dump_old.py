#!/usr/bin/env python
"""既有 B 臂 probs 导出（供集成），支持低秩读出结构；完成后调用方清理 .pt 省盘。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from encoders import build_encoder  # noqa: E402
from models import build_model  # noqa: E402
from semeval_finetune import SevDataset, TaskModel, pad_collate  # noqa: E402

OUT = ROOT / "runs/semeval_r"
TARGETS = [("b", "runs/semeval_b", 0), ("b_r4", "runs/semeval_b_r4", 4),
           ("b_r8", "runs/semeval_b_r8", 8), ("b_r11", "runs/semeval_b_r11", 11),
           ("b_r8init", "runs/semeval_b_r8init", 8)]


def loader(split: str) -> DataLoader:
    ds = SevDataset(str(ROOT / f"data/processed/semeval/b_{split}.parquet"))
    return DataLoader(ds, batch_size=8, collate_fn=pad_collate, num_workers=0)


def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with open(ROOT / "configs/ddet_v041.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    enc_cfg = dict(cfg["encoder"])
    enc_cfg["path"] = str(ROOT / enc_cfg["path"])
    enc = build_encoder(**enc_cfg)
    dual = build_model("dual", encoder=enc, dim=enc.hidden_size,
                       pool=cfg["model"].get("pool", "mean"),
                       s2_rank=cfg["model"].get("s2_rank", 1))
    tok = AutoTokenizer.from_pretrained(enc_cfg["path"])  # noqa: F841
    OUT.mkdir(parents=True, exist_ok=True)
    for tag, run_dir, rk in TARGETS:
        ck = ROOT / run_dir / "best.pt"
        if not ck.exists():
            print(f"[dump] {tag}: 缺 {ck}，跳过", flush=True)
            continue
        model = TaskModel(dual, "b", 11, s2_rank=rk).to(device)
        st = torch.load(str(ck), map_location="cpu", weights_only=False)
        model.load_state_dict(st["state"], strict=False)
        model.eval()
        for split in ("val", "test"):
            probs, Y = [], []
            with torch.no_grad():
                for b in tqdm(loader(split), desc=f"{tag}/{split}", unit="it", leave=False):
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                        lg = model.logits(b["input_ids"].to(device),
                                          b["attention_mask"].to(device))
                    probs.append(torch.softmax(lg.float(), -1).cpu())
                    Y.extend(b["labels"].tolist())
            np.savez_compressed(OUT / f"probs_{tag}_{split}.npz",
                                probs=torch.cat(probs).numpy(), y=np.array(Y))
            print(f"[dump] {tag}/{split} OK（n={len(Y)}）", flush=True)
    print("[dump_old] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
