#!/usr/bin/env python
"""E19 判读：跨编码器稳健性——原始 CodeT5 特征 vs v1.0（自训）特征。

背景（问题清单 B4）：v0.7 全部结论建立在自训编码器上（且其训练时见过 qwen05 家族数据），
需用"中立编码器"（原始未微调 CodeT5-base，extract_pair_feats_raw.py 提取）复跑核心判读。

对比项（v1 参考值写在代码注释里）：
  ① 六族全量：Δ → [h⁺;h⁻]（v1: 0.7626 → 0.8271）
  ② 对齐 329 题：pair → center（v1: 0.7827 → 0.8571）
  ③ 关键二分：granite2b/smollm2（v1: .8502/.9057/.9646）、
              qwen05/qwen15（v1: .7335/.7906/.9036）、qwen15/ds13（v1: .8845/.9404/.9772）

预注册判读：若"pair > Δ"且"center > pair"的增益方向在全部比较中保持，
且对齐中心化 ≥ 0.80 → **结论稳健**（增益不是编码器自举的伪信号）；
若方向丢失/反转 → 主结论需携带风险标注（如实上报）。
输出：runs/kernel_e19/robust.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from kernel_e17_framework import cv_disc, summarize  # noqa: E402

F19 = ROOT / "runs/kernel_e19"
RES = F19
NAMES = ["qwen05", "qwen15", "ds13", "yi15", "granite2b", "smollm2"]
T1_PAIRS = [("granite2b", "smollm2"), ("qwen05", "qwen15"), ("qwen15", "ds13")]


def load_fam(name):
    z = np.load(F19 / f"h_raw_{name}.npz", allow_pickle=True)
    return (np.asarray(z["h_plus"], dtype="float64"),
            np.asarray(z["h_minus"], dtype="float64"),
            [str(v) for v in z["tasks"]])


def run_task(names, tag, report):
    fams = [load_fam(n) for n in names]
    common = None
    for _, _, t in fams:
        s = set(t)
        common = s if common is None else (common & s)
    common = sorted(common)
    HP, HM = [], []
    for hp, hm, t in fams:
        idx = {tt: i for i, tt in enumerate(t)}
        sel = [idx[tt] for tt in common]
        HP.append(hp[sel]); HM.append(hm[sel])
    HP = np.stack(HP); HM = np.stack(HM)
    y = np.concatenate([[n] * len(common) for n in names])
    g = np.concatenate([np.array(common) for _ in names])
    HPc = HP - HP.mean(axis=0, keepdims=True)
    HMc = HM - HM.mean(axis=0, keepdims=True)
    views = {
        "delta": np.vstack(list(HP - HM)),
        "pair": np.hstack([np.vstack(list(HP)), np.vstack(list(HM))]),
        "pair_center": np.hstack([np.vstack(list(HPc)), np.vstack(list(HMc))]),
    }
    entry = {"n_tasks": len(common), "n_samples": int(len(y))}
    for v, X in views.items():
        m = summarize(cv_disc(X, y, g), y)
        entry[v] = {"acc": m["acc"], "balanced_acc": m["balanced_acc"]}
    report[tag] = entry
    print(f"[e19] {tag:<24}: Δ {entry['delta']['acc']:.4f} → pair {entry['pair']['acc']:.4f} "
          f"→ center {entry['pair_center']['acc']:.4f}", flush=True)


def run_attribution(report):
    HP_l, HM_l, y_l, g_l = [], [], [], []
    for n in NAMES:
        hp, hm, t = load_fam(n)
        HP_l.append(hp); HM_l.append(hm)
        y_l += [n] * len(t); g_l += t
    HPf = np.vstack(HP_l); HMf = np.vstack(HM_l)
    yf = np.array(y_l); gf = np.array(g_l)
    entry = {"n_samples": int(len(yf))}
    for v, X in (("delta", HPf - HMf), ("pair", np.hstack([HPf, HMf]))):
        m = summarize(cv_disc(X, yf, gf), yf)
        entry[v] = {"acc": m["acc"], "balanced_acc": m["balanced_acc"]}
    report["attribution_full"] = entry
    print(f"[e19] 归因·全量: Δ {entry['delta']['acc']:.4f} → pair {entry['pair']['acc']:.4f}",
          flush=True)

    # 对齐 329 中心化
    fams = [load_fam(n) for n in NAMES]
    common = None
    for _, _, t in fams:
        s = set(t)
        common = s if common is None else (common & s)
    common = sorted(common)
    HP, HM = [], []
    for hp, hm, t in fams:
        idx = {tt: i for i, tt in enumerate(t)}
        sel = [idx[tt] for tt in common]
        HP.append(hp[sel]); HM.append(hm[sel])
    HP = np.stack(HP); HM = np.stack(HM)
    HPc = HP - HP.mean(axis=0, keepdims=True)
    HMc = HM - HM.mean(axis=0, keepdims=True)
    y = np.concatenate([[n] * len(common) for n in NAMES])
    g = np.concatenate([np.array(common) for _ in NAMES])
    entry2 = {"n_tasks": len(common), "n_samples": int(len(y))}
    for v, X in (("pair", np.hstack([np.vstack(list(HP)), np.vstack(list(HM))])),
                 ("pair_center", np.hstack([np.vstack(list(HPc)), np.vstack(list(HMc))]))):
        m = summarize(cv_disc(X, y, g), y)
        entry2[v] = {"acc": m["acc"], "balanced_acc": m["balanced_acc"]}
    report["attribution_aligned"] = entry2
    print(f"[e19] 归因·对齐: pair {entry2['pair']['acc']:.4f} → center "
          f"{entry2['pair_center']['acc']:.4f}", flush=True)


def main() -> int:
    missing = [n for n in NAMES if not (F19 / f"h_raw_{n}.npz").exists()]
    if missing:
        raise SystemExit(f"缺特征：{missing}（先跑 extract_pair_feats_raw.py）")
    report = {}
    for a, b in T1_PAIRS:
        run_task([a, b], f"binary_{a}_vs_{b}", report)
    run_attribution(report)

    # 预注册判读
    ok_dirs = all(report[k]["pair"]["acc"] > report[k]["delta"]["acc"]
                  for k in report if k.startswith("binary_") or k == "attribution_full")
    ok_center = (report["attribution_aligned"]["pair_center"]["acc"]
                 > report["attribution_aligned"]["pair"]["acc"]
                 and report["attribution_aligned"]["pair_center"]["acc"] >= 0.80)
    verdict = ("稳健：增益方向全部保持且对齐中心化 ≥0.80 → 非编码器自举产物"
               if ok_dirs and ok_center else
               "未达稳健线——主结论需携带风险标注（见方向/数值异常项）")
    report["verdict"] = verdict
    print(f"\n[e19] 判读: {verdict}")
    out = RES / "robust.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"[e19] 写出 {out}\n[e19] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
