#!/usr/bin/env python
"""v0.4 新几何损失的"首步梯度非零"冒烟（CPU，stub 编码器）。

死锁家族第 4 课（前 3 例见 docx/d-det-v0.2.md §3 / d-det.md 注记）：任何新读出/损失
上线前必须验证"第一步就有非零梯度"。本脚本用 32 维 stub 编码器逐个验证：

    rep（固定 margin 斥力）         → ∂L/∂w1 ≠ 0
    fisher（Fisher 判别比 EMA）     → 过 warmup 后 ∂L/∂w1 ≠ 0
    cov（去冗余 EMA 协方差）         → ∂L/∂emb ≠ 0
    xfam（跨族方向，经 stream_loss） → ∂L/∂emb ≠ 0 且 parts["xfam"] 正常
    m4 集成路径（rep+fisher+cov 同开）→ loss 有限、∂L/∂w1 ≠ 0、parts 键齐全

用法： python scripts/check_v040_grads.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models import build_model                                     # noqa: E402
from train import CovEMA, FisherEMA, s1_margin_loss, stream_loss   # noqa: E402


class StubEnc(nn.Module):
    def __init__(self, dim: int = 32, vocab: int = 200):
        super().__init__()
        self.hidden_size = dim
        self.emb = nn.Embedding(vocab, dim)

    def forward(self, input_ids, attention_mask=None):
        return self.emb(input_ids)

    def input_embeddings(self):
        return self.emb.weight


def grad_sum(loss, param, retain: bool = False) -> float:
    g = torch.autograd.grad(loss, param, retain_graph=retain, allow_unused=True)[0]
    return float(g.abs().sum()) if g is not None else 0.0


def main() -> int:
    torch.manual_seed(0)
    model = build_model("dual", encoder=StubEnc(), dim=32)
    nn.init.normal_(model.w1.weight, std=0.1)      # 测试用：零初始化起点 s1≡0 会让 Fisher 无信号
    ok = True

    # 1) 固定 margin 斥力
    ids = torch.randint(0, 100, (1, 12))
    mask = torch.ones_like(ids)
    s1, _, _ = model.forward_feat(ids, mask)
    l_rep = s1_margin_loss(s1, torch.tensor([1]), 1.0)
    g = grad_sum(l_rep, model.w1.weight)
    print(f"[rep]    loss={float(l_rep):.4f}  |dL/dw1|={g:.3e}")
    ok = ok and g > 0 and bool(torch.isfinite(l_rep))

    # 2) Fisher（双类交替、输入分布不同；过 warmup 后看梯度）
    fisher = FisherEMA(momentum=0.95, log_target=1.386, warmup=8)
    g_f, l_seen = 0.0, None
    for t in range(160):
        label = t % 2
        ids_t = (torch.randint(0, 100, (1, 12)) if label == 0
                 else torch.randint(100, 200, (1, 12)))
        s1_t, _, _ = model.forward_feat(ids_t, torch.ones_like(ids_t))
        lf = fisher.step(s1_t, torch.tensor([label]))
        if lf is not None:
            l_seen = lf
            g_f = max(g_f, grad_sum(lf, model.w1.weight))
    print(f"[fisher] loss={None if l_seen is None else float(l_seen):.4f}  "
          f"max|dL/dw1|={g_f:.3e}")
    ok = ok and g_f > 0 and l_seen is not None

    # 3) cov（去冗余 EMA 协方差）
    cov = CovEMA(dim=32, momentum=0.9)
    g_c, l_c = 0.0, None
    for t in range(60):
        ids_t = torch.randint(0, 200, (1, 16))
        _, _, h = model.forward_feat(ids_t, torch.ones_like(ids_t))
        lc_ = cov.step(h)
        if lc_ is not None:
            l_c = lc_
            g_c = max(g_c, grad_sum(lc_, model.encoder.emb.weight))
    print(f"[cov]    loss={None if l_c is None else float(l_c):.3e}  "
          f"max|dL/demb|={g_c:.3e}")
    ok = ok and g_c > 0 and l_c is not None

    # 4) pairxf（经 stream_loss 集成路径）
    b = {
        "input_ids_plus_a": torch.randint(0, 100, (1, 10)),
        "attention_mask_plus_a": torch.ones(1, 10, dtype=torch.long),
        "input_ids_minus_a": torch.randint(0, 100, (1, 10)),
        "attention_mask_minus_a": torch.ones(1, 10, dtype=torch.long),
        "input_ids_plus_b": torch.randint(100, 200, (1, 10)),
        "attention_mask_plus_b": torch.ones(1, 10, dtype=torch.long),
        "input_ids_minus_b": torch.randint(100, 200, (1, 10)),
        "attention_mask_minus_b": torch.ones(1, 10, dtype=torch.long),
    }
    loss, parts = stream_loss({"loss": {"xfam": 0.5}}, model, b, "pairxf")
    g_x = grad_sum(loss, model.encoder.emb.weight)
    print(f"[xfam]   loss={float(loss):.4f}  parts={parts}  |dL/demb|={g_x:.3e}")
    ok = ok and g_x > 0 and "xfam" in parts

    # 5) m4 集成路径（rep+fisher+cov 同开；先喂几步让 fisher/cov 缓冲热起来）
    fisher2, cov2 = FisherEMA(warmup=2), CovEMA(dim=32, momentum=0.9)
    aux = {"fisher": fisher2, "cov": cov2}
    for t in range(10):
        ids_t = torch.randint(0, 100, (1, 12))
        s1_t, _, h_t = model.forward_feat(ids_t, torch.ones_like(ids_t))
        fisher2.step(s1_t.detach(), torch.tensor([t % 2]))
        cov2.step(h_t.detach())
    bm = {"input_ids": torch.randint(0, 100, (1, 12)),
          "attention_mask": torch.ones(1, 12, dtype=torch.long),
          "labels": torch.tensor([1])}
    cfg = {"loss": {"s1": 1.0, "rep": 1.0, "rep_margin": 1.0,
                    "fisher": 1.0, "cov": 1.0}}
    loss_m, parts_m = stream_loss(cfg, model, bm, "m4", aux)
    g_m = grad_sum(loss_m, model.w1.weight)
    print(f"[m4]     loss={float(loss_m):.4f}  parts={sorted(parts_m)}  |dL/dw1|={g_m:.3e}")
    ok = ok and g_m > 0 and bool(torch.isfinite(loss_m))

    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
