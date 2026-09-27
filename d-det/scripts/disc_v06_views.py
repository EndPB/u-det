#!/usr/bin/env python
"""v0.6 规范评测：视图升级（[h⁺;h⁻] / 逐题中心化）的正式口径（不新增数据；纯 CPU）。

依据 E17 系列（内核报告 §3.11）：
  · 非转导（单对即可部署）：[h⁺;h⁻] DiscHead —— 全量 5703 按题 GroupKFold；
  · 转导（有同题兄弟样本）：[h̃⁺;h̃⁻]（族组内中心化）—— 对齐 329 题；
  · 参照：Δ 视图旧口径（0.7626 / 0.7740）。
产出：runs/v0.6_views/{eval_v06.json, head_hp_hm.pkl}
      （head_hp_hm.pkl = 全量数据拟合的部署头，供对新配对直接推理）
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from models.disc import DiscHead  # noqa: E402
from kernel_e17_framework import load_full, cv_disc, summarize, NAMES  # noqa: E402
from kernel_e17b_views import load_h_aligned  # noqa: E402

RES = ROOT / "runs/v0.6_views"


def main() -> int:
    RES.mkdir(parents=True, exist_ok=True)
    rep = {}

    # ---------- 非转导：[h+;h-]（主口径，单对即可） ----------
    Xd, Xhp, Xhm, yf, gf = load_full()
    Xpair = np.hstack([Xhp, Xhm])
    rep["nontransductive_hp_hm_full"] = summarize(cv_disc(Xpair, yf, gf), yf)
    print(f"[v06] 非转导 [h+;h-] 全量: acc {rep['nontransductive_hp_hm_full']['acc']} / "
          f"bal {rep['nontransductive_hp_hm_full']['balanced_acc']}", flush=True)

    Xtriple = np.hstack([Xd, Xhp, Xhm])
    m = summarize(cv_disc(Xtriple, yf, gf), yf)
    rep["nontransductive_d_hp_hm_full"] = {"acc": m["acc"], "balanced_acc": m["balanced_acc"]}
    print(f"[v06] 非转导 [d;h+;h-] 全量: acc {m['acc']} / bal {m['balanced_acc']}", flush=True)

    m = summarize(cv_disc(Xd, yf, gf), yf)
    rep["ref_delta_full"] = {"acc": m["acc"], "balanced_acc": m["balanced_acc"]}
    print(f"[v06] 参照 Δ 全量: acc {m['acc']} / bal {m['balanced_acc']}", flush=True)

    # 部署头：全量拟合
    head = DiscHead().fit(Xpair, yf)
    head.save(RES / "head_hp_hm.pkl")
    print(f"[v06] 部署头已存 {RES / 'head_hp_hm.pkl'}", flush=True)

    # ---------- 转导：中心化 [h~+;h~-]（对齐 329 题） ----------
    HP, HM, common = load_h_aligned()
    HPc = HP - HP.mean(axis=0, keepdims=True)
    HMc = HM - HM.mean(axis=0, keepdims=True)
    Xc = np.hstack([np.vstack(list(HPc)), np.vstack(list(HMc))])
    y = np.concatenate([[n] * len(common) for n in NAMES])
    g = np.concatenate([np.array(common) for _ in NAMES])
    rep["transductive_centered_hp_hm_aligned"] = summarize(cv_disc(Xc, y, g), y)
    print(f"[v06] 转导 中心化[h~+;h~-] 对齐: acc {rep['transductive_centered_hp_hm_aligned']['acc']} / "
          f"bal {rep['transductive_centered_hp_hm_aligned']['balanced_acc']}", flush=True)

    rep["meta"] = {
        "protocol": "按题 GroupKFold=5（同题同折）；中心化=族组内逐题去均值（不用标签）",
        "views": {"pair": "[h+;h-]", "triple": "[d;h+;h-]", "centered": "[h~+;h~-]"},
        "source": "E17 系列；docx/d-det-kernel.md §3.11",
    }
    out = RES / "eval_v06.json"
    out.write_text(json.dumps(rep, ensure_ascii=False, indent=2))
    print(f"\n[v06] 写出 {out}\n[v06] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
