#!/usr/bin/env python
"""v0.5 判别头 6 族训练/验证（各族全量 + 394 同题对齐版）。

数据：runs/v0.5_disc/d_{qwen05,qwen15,ds13,yi15,granite2b,qwen3b}.npz（存在即用）
- 全量版：各族全部 Δ；按题 GroupKFold（同题 q/d 同折）
- 对齐版：6 族共同题子集（严格同题公平）
输出：runs/v0.5_disc/{head6.pkl, eval6.json}
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

OUT = ROOT / "runs/v0.5_disc"
NAMES = ["qwen05", "qwen15", "ds13", "yi15", "granite2b", "smollm2", "qwen3b"]


def main() -> int:
    data = {}
    for n in NAMES:
        p = OUT / f"d_{n}.npz"
        if p.exists():
            d = np.load(p, allow_pickle=True)
            data[n] = (np.asarray(d["d"], dtype="float32"),
                       np.array([str(t) for t in d["tasks"]]))
            print(f"[t6] {n}: {data[n][0].shape[0]} 对", flush=True)
    if len(data) < 3:
        raise SystemExit(f"可用族不足（{list(data)}）")

    res = {"families": list(data), "sizes": {k: int(v[0].shape[0]) for k, v in data.items()}}

    # ---- 全量版 ----
    X = np.vstack([v[0] for v in data.values()])
    y = np.concatenate([np.array([k] * len(v[0])) for k, v in data.items()])
    g = np.concatenate([v[1] for v in data.values()])
    head = DiscHead().fit(X, y)
    head.save(OUT / "head6.pkl")
    res["full"] = {k: round(v, 4) for k, v in DiscHead().cross_val(X, y, g).items()
                   if k in ("acc", "balanced_acc", "n", "n_classes")}
    print(f"[t6] 全量（{X.shape[0]} 样本 × {len(data)} 族）: {res['full']}", flush=True)

    # ---- 394 同题对齐版（共同题交集）----
    common = None
    for v in data.values():
        common = set(v[1].tolist()) if common is None else (common & set(v[1].tolist()))
    common = sorted(common or [])
    if len(common) >= 50:
        idx = {k: {t: i for i, t in enumerate(v[1].tolist())} for k, v in data.items()}
        X2 = np.vstack([v[0][[idx[k][t] for t in common]] for k, v in data.items()])
        y2 = np.concatenate([np.array([k] * len(common)) for k in data])
        g2 = np.concatenate([np.array(common) for _ in data])
        res["aligned_common"] = {"n_common_tasks": len(common)}
        res["aligned_common"].update({k: round(v, 4) for k, v in
                                      DiscHead().cross_val(X2, y2, g2).items()
                                      if k in ("acc", "balanced_acc")})
        print(f"[t6] 对齐版（{len(common)} 共同题）: {res['aligned_common']}", flush=True)

    (OUT / "eval6.json").write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print(json.dumps(res, ensure_ascii=False, indent=2))
    print("[t6] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
