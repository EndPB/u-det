#!/usr/bin/env python
"""E17e：配对视图的对称-反对称分解（位置 vs 位移）——理论改写（§3.12）的判据实验。

对配对视图做分解：
  对称分量 S = (h⁺ + h⁻)/2      （双侧共享的"位置签名"）
  反对称分量 A = (h⁺ − h⁻)/2     （= Δ/2，纯"位移"）
在 全量（未中心化）与 对齐（逐题中心化）两种口径下，比较：
  视图：S 单独 / A 单独 / [S;A] / [h⁺;h⁻]（参照）
判读（预注册）：若 S 单独 ≈ [h⁺;h⁻] 且 ≫ A 单独 → "位置主导、位移冗余"成立。
输出：runs/kernel_e17/symsplit.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from kernel_e17_framework import cv_disc, load_full, summarize, NAMES  # noqa: E402
from kernel_e17b_views import load_h_aligned  # noqa: E402

RES = ROOT / "runs/kernel_e17"


def run_views(Xp, Xm, y, g, tag, report):
    S = 0.5 * (Xp + Xm)
    A = 0.5 * (Xp - Xm)
    views = {"S_alone": S, "A_alone": A, "S_A": np.hstack([S, A]),
             "pair_ref": np.hstack([Xp, Xm])}
    for name, Xv in views.items():
        m = summarize(cv_disc(Xv, y, g), y)
        report[f"{tag}_{name}"] = {"acc": m["acc"], "balanced_acc": m["balanced_acc"]}
        print(f"[e17e] {tag} {name:<9}: acc {m['acc']:.4f} / bal {m['balanced_acc']:.4f}",
              flush=True)


def main() -> int:
    report = {}
    # 全量（未中心化）
    Xd, Xhp, Xhm, yf, gf = load_full()
    run_views(Xhp, Xhm, yf, gf, "full", report)
    # 对齐（逐题中心化）
    HP, HM, common = load_h_aligned()
    HPc = HP - HP.mean(axis=0, keepdims=True)
    HMc = HM - HM.mean(axis=0, keepdims=True)
    Xp = np.vstack(list(HPc))
    Xm = np.vstack(list(HMc))
    y = np.concatenate([[n] * len(common) for n in NAMES])
    g = np.concatenate([np.array(common) for _ in NAMES])
    run_views(Xp, Xm, y, g, "aligned", report)
    out = RES / "symsplit.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n[e17e] 写出 {out}\n[e17e] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
