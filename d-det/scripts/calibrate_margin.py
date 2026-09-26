#!/usr/bin/env python
"""从 v0.3.0 的 m4 分数 dump 标定 loss.rep_margin（固定 margin 斥力）。

约定：signed = (2y−1)·s1（正类 = AI 样本 y=1）。建议 margin 取 signed 的 P5 分位
（≈5% 样本落在 margin 带内、获得斥力；其余梯度≈0），下限 0.5 防过小。

用法：
    python scripts/calibrate_margin.py [--scores runs/v0.3.0_fullft/scores_m4_test.pt]
输出末行：MARGIN=<值>（供队列 grep）
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="runs/v0.3.0_fullft/scores_m4_test.pt")
    ap.add_argument("--quantile", type=float, default=0.05)
    ap.add_argument("--floor", type=float, default=0.5)
    args = ap.parse_args()
    path = Path(args.scores)
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[1] / path
    if not path.exists():
        print(f"[calib] 找不到 {path}（先对基座跑评测 --dump-scores）")
        return 1
    d = torch.load(path, map_location="cpu")
    s1 = d["s1"].float()
    y = d["label"].float()
    signed = (2 * y - 1) * s1
    qs = {q: float(torch.quantile(signed, q)) for q in (0.01, 0.05, 0.10, 0.25)}
    m = round(max(float(args.floor), qs[args.quantile]), 3)
    frac = float((signed < m).float().mean())
    ai, hu = signed[y == 1], signed[y == 0]
    print(f"[calib] n={len(s1)} AI={int((y == 1).sum())} Human={int((y == 0).sum())}")
    print(f"[calib] signed 分位：P1={qs[0.01]:.3f} P5={qs[0.05]:.3f} "
          f"P10={qs[0.10]:.3f} P25={qs[0.25]:.3f}")
    print(f"[calib] AI: mean={float(ai.mean()):.3f} std={float(ai.std()):.3f} | "
          f"Human: mean={float(hu.mean()):.3f} std={float(hu.std()):.3f}")
    print(f"[calib] margin={m}（signed < margin 的样本占 {frac:.1%}）")
    print(f"MARGIN={m}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
