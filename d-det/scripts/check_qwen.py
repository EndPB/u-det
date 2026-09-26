#!/usr/bin/env python
"""Qwen 编码器接入冒烟（项目规约：新骨干/新读出上线前必须过"首步梯度非零 + 显存包络"）。"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from encoders import build_encoder  # noqa: E402


def main() -> int:
    assert torch.cuda.is_available(), "需要 GPU"
    dev = "cuda"
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "configs/ddet_qwen.yaml"
    with open(ROOT / cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    enc_cfg = dict(cfg["encoder"])
    enc_cfg["path"] = str(ROOT / enc_cfg["path"])
    enc = build_encoder(**enc_cfg).to(dev)
    D = enc.hidden_size
    print(f"[check] hidden_size={D}，num_layers={enc.num_layers}", flush=True)

    torch.manual_seed(0)
    # ---- 1) 前向形状 ----
    ids = torch.randint(0, 1000, (2, 256), device=dev)
    mask = torch.ones_like(ids)
    enc.train()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        feats = enc(ids, mask)
    print("[check] forward shape:", tuple(feats.shape), feats.dtype, flush=True)
    assert feats.shape == (2, 256, D), feats.shape

    # ---- 2) 首步梯度非零（死锁家族规约）----
    head = nn.Linear(D, 11).to(dev)
    loss = F.cross_entropy(head(feats.mean(1).float()), torch.randint(0, 11, (2,), device=dev))
    loss.backward()
    train_p = [(n, p) for n, p in enc.named_parameters() if p.requires_grad]
    nz = sum(1 for _, p in train_p if p.grad is not None and p.grad.abs().sum() > 0)
    frac = nz / max(len(train_p), 1)
    print(f"[check] 可训练参数 {len(train_p)} 个 / {sum(p.numel() for _, p in train_p)/1e6:.2f}M，"
          f"首步非零梯度 {nz} 个（{frac:.1%}）", flush=True)
    assert loss.item() == loss.item() and frac >= 0.5, "首步梯度冒烟未通过"
    enc.zero_grad(set_to_none=True)

    # ---- 3) 显存包络（大模型用 batch=2；默认 batch=4 × seq=1024 训练态，含反传）----
    bs = 2 if D > 2000 else 4
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    ids = torch.randint(0, 1000, (bs, 1024), device=dev)
    mask = torch.ones_like(ids)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        feats = enc(ids, mask)
    loss = F.cross_entropy(head(feats.mean(1).float()), torch.randint(0, 11, (bs,), device=dev))
    loss.backward()
    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"[check] batch{bs}×1024 峰值显存 {peak:.2f} GB", flush=True)
    assert peak < 11.0, f"显存超包络：{peak:.2f} GB"

    print("QWEN_SMOKE_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
