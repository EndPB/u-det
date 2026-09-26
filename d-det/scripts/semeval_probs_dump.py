#!/usr/bin/env python
"""B/C 直连头 probs（val+test 导出，供堆叠）；补全冻结特征缓存（b_val 完整版、c_val）。

输出：runs/semeval_r/probs_{b,c}_{val,test}.npz（probs/logits/y/lang）
      runs/semeval_zeroshot/feat/{b,c}_val.npz（完整 h/s1/s2/y/lang）
"""

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

FEAT = ROOT / "runs/semeval_zeroshot/feat"
OUT = ROOT / "runs/semeval_r"


def loader(task: str, split: str) -> DataLoader:
    ds = SevDataset(str(ROOT / f"data/processed/semeval/{task}_{split}.parquet"))
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
    ck = torch.load(str(ROOT / "runs/v0.4.1_covreg/last.pt"), map_location="cpu", weights_only=False)
    dual.load_state_dict(ck["state"], strict=False)
    dual = dual.to(device).eval()

    # ---------- 1) 冻结特征补全（b_val 覆盖、c_val 新增） ----------
    for task, split in (("b", "val"), ("c", "val")):
        H, S1, S2, Y = [], [], [], []
        with torch.no_grad():
            for b in tqdm(loader(task, split), desc=f"feat {task}_{split}", unit="it"):
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                    s1, s2, h = dual.forward_feat(b["input_ids"].to(device),
                                                  b["attention_mask"].to(device))
                H.append(h.float().cpu())
                S1.append(s1.float().cpu().reshape(-1))
                S2.append(s2.float().cpu().reshape(len(b["labels"]), -1))
                Y.extend(b["labels"].tolist())
        np.savez_compressed(FEAT / f"{task}_{split}.npz",
                            h=torch.cat(H).numpy().astype("float32"),
                            s1=torch.cat(S1).numpy(), s2=torch.cat(S2).numpy(),
                            y=np.array(Y), lang=np.array(SevDataset(
                                str(ROOT / f"data/processed/semeval/{task}_{split}.parquet")).langs,
                                dtype=object))
        print(f"[feat] {task}_{split} 完整缓存 OK（n={len(Y)}）", flush=True)

    # ---------- 2) B/C 直连头 probs ----------
    OUT.mkdir(parents=True, exist_ok=True)
    for task, ckpt, ncls in (("b", "runs/semeval_b/last.pt", 11),
                             ("c", "runs/semeval_c/last.pt", 4)):
        model = TaskModel(dual, task, ncls).to(device)
        st = torch.load(str(ROOT / ckpt), map_location="cpu", weights_only=False)
        miss, unexp = model.load_state_dict(st["state"], strict=False)
        print(f"[ckpt] {ckpt} missing={len(miss)} unexpected={len(unexp)}", flush=True)
        model.eval()
        for split in ("val", "test"):
            probs, logits, Y = [], [], []
            with torch.no_grad():
                for b in tqdm(loader(task, split), desc=f"probs {task}_{split}", unit="it"):
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                        lg = model.logits(b["input_ids"].to(device),
                                          b["attention_mask"].to(device))
                    logits.append(lg.float().cpu())
                    probs.append(torch.softmax(lg.float(), -1).cpu())
                    Y.extend(b["labels"].tolist())
            np.savez_compressed(OUT / f"probs_{task}_{split}.npz",
                                probs=torch.cat(probs).numpy(),
                                logits=torch.cat(logits).numpy(),
                                y=np.array(Y))
            print(f"[probs] {task}_{split} OK（n={len(Y)}）", flush=True)
    print("[probs_dump] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
