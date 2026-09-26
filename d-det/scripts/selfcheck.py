#!/usr/bin/env python
"""接线自检：跑任何正式训练之前先过这一关（零训练成本）。

检查项：
    S1 起点等价：未训练的模型对任意输入 s1 ≡ s2 ≡ 0（零初始化读出的直接结论；
       若失败说明有旁路把非零值泄漏进了读出通路）
    S2 首步梯度：m4 的 BCE 让 w1 拿到非零梯度、且不给 w2 梯度；
       pair 的 margin 让 w2 拿到非零梯度、且不给 w1 梯度
       —— 防御 "α=0 与 w=0 同时成立" 那类乘法死锁（models/scores.py 注记 1）
    S3 参数清单：可训练 / 冻结参数与初始 cos(w1,w2)

用法::

    # 有卡（按 configs/ddet_base.yaml 的编码器）
    OMP_NUM_THREADS=8 python scripts/selfcheck.py --config configs/ddet_base.yaml

    # 无卡 / CPU 冒烟（换轻量编码器，秒级）
    OMP_NUM_THREADS=8 python scripts/selfcheck.py --config configs/ddet_base.yaml \\
        --cpu --encoder codet5tok --set encoder.freeze=true

说明：自检会强制关掉 encoder.compile / static（不是检查目标，且避免编译环境差异）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from encoders import build_encoder                     # noqa: E402
from models import build_model                         # noqa: E402
from train import apply_overrides, load_config, readout_cos   # noqa: E402

CODE_A = '''def add(a, b):\n    """Add two numbers."""\n    return a + b\n'''
CODE_B = '''def add(a,b):\n  return a + b\n'''


def main() -> int:
    parser = argparse.ArgumentParser(description="d-det 接线自检")
    parser.add_argument("--config", default="configs/ddet_base.yaml")
    parser.add_argument("--encoder", default=None)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--set", action="append", default=None, metavar="段.键=值")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args)
    cfg["encoder"]["compile"] = False           # 自检与编译无关，关掉以规避环境差异
    cfg["encoder"]["static"] = False

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print(f"[selfcheck] device={device} encoder={cfg['encoder']['name']} "
          f"model={cfg['model']['name']}")

    enc_cfg = dict(cfg["encoder"])
    if not Path(enc_cfg["path"]).is_absolute():
        enc_cfg["path"] = str(ROOT / enc_cfg["path"])        # 允许从任意 cwd 运行
    encoder = build_encoder(**enc_cfg)
    mcfg = dict(cfg["model"])
    model = build_model(mcfg.pop("name", "dual"), encoder=encoder,
                        dim=encoder.hidden_size, **mcfg)
    model.to(device)
    model.train()

    tokenizer = AutoTokenizer.from_pretrained(enc_cfg["path"])

    def encode(text: str):
        ids = tokenizer(text, return_tensors="pt")["input_ids"].to(device)
        return ids, torch.ones_like(ids)

    failures = []

    # ---- S1 起点等价 ----
    ids_a, mask_a = encode(CODE_A)
    ids_b, mask_b = encode(CODE_B)
    with torch.no_grad():
        s1a, s2a = model(ids_a, mask_a)
        s1b, s2b = model(ids_b, mask_b)
    dev_s1 = max(float(s1a.abs().max()), float(s1b.abs().max()))
    dev_s2 = max(float(s2a.abs().max()), float(s2b.abs().max()))
    ok = dev_s1 < 1e-6 and dev_s2 < 1e-6
    print(f"[S1] 起点等价：max|s1|={dev_s1:.3e}  max|s2|={dev_s2:.3e}"
          f"  ⇒ {'PASS' if ok else 'FAIL'}")
    if not ok:
        failures.append("S1 起点等价（零初始化读出未生效或有旁路）")

    # ---- S2a 首步梯度：m4 的 BCE → w1 ----
    model.zero_grad(set_to_none=True)
    s1, _ = model(ids_a, mask_a)
    l1 = F.binary_cross_entropy_with_logits(s1.float(), torch.tensor([1.0], device=device))
    l1.backward()
    g1 = float(model.w1.weight.grad.norm()) if model.w1.weight.grad is not None else 0.0
    g1_w2 = float(model.w2.weight.grad.norm()) if model.w2.weight.grad is not None else 0.0
    ok = g1 > 0.0 and g1_w2 == 0.0
    print(f"[S2a] m4→w1：|∂L1/∂w1|={g1:.3e}（期望 >0）  |∂L1/∂w2|={g1_w2:.3e}（期望 0）"
          f"  ⇒ {'PASS' if ok else 'FAIL'}")
    if not ok:
        failures.append("S2a（BCE 对 w1 的梯度异常 —— 查零初始化/损失接线）")

    # ---- S2b 首步梯度：pair 的 margin → w2 ----
    model.zero_grad(set_to_none=True)
    _, s2_plus = model(ids_a, mask_a)
    _, s2_minus = model(ids_b, mask_b)
    margin = float(cfg.get("loss", {}).get("margin", 1.0))
    if int(model.s2_rank) == 1:
        gap = (s2_plus - s2_minus).squeeze(-1)
    else:
        gap = (s2_plus - s2_minus).norm(dim=-1)
    l2 = F.relu(margin - gap).mean()
    l2.backward()
    g2 = float(model.w2.weight.grad.norm()) if model.w2.weight.grad is not None else 0.0
    g2_w1 = float(model.w1.weight.grad.norm()) if model.w1.weight.grad is not None else 0.0
    ok = g2 > 0.0 and g2_w1 == 0.0
    print(f"[S2b] pair→w2：|∂L2/∂w2|={g2:.3e}（期望 >0）  |∂L2/∂w1|={g2_w1:.3e}（期望 0）"
          f"  ⇒ {'PASS' if ok else 'FAIL'}")
    if not ok:
        failures.append("S2b（margin 对 w2 的梯度异常 —— 查 hinge 方向与零初始化）")

    # ---- S3 参数清单 ----
    n_all = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_read = model.w1.weight.numel() + model.w2.weight.numel()
    print(f"[S3] 参数：总计 {n_all / 1e6:.2f}M；可训练 {n_train / 1e6:.3f}M"
          f"（读出 w1/w2 = {n_read}，pool={model.pool}，s2_rank={model.s2_rank}）")
    print(f"[S3] 初始 cos(w1,w2) = {readout_cos(model):+.4f}"
          f"（零向量按 0 处理；训练中应保持在低位）")

    if failures:
        print(f"[selfcheck] FAIL：{failures}")
        return 1
    print("[selfcheck] 全部通过 ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
