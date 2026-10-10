#!/usr/bin/env python3
"""Convenience template: join server vs local v3 C0 per-fold score arrays on ev_rows.

NOT part of the protocol — the local side may use its own comparison code.
Usage:
  python compare_template.py --server <dir_with_server_npz> [--local <dir>]
                             [--protocol dev|inner|both]

Server side = this handoff dir (default if --server omitted: script directory).
For each fold file it loads the server npz, finds the same-name file in the
local dir (or accepts a user-supplied mapping), joins rows by `ev_rows`,
and prints max|delta| for fused + the four components (server arrays in the
npz are already in taskpos-sorted order; join makes order robust).

Read-only: nothing is written.
"""
import argparse
import json
from pathlib import Path

import numpy as np

COMPS = ["semantic", "char_tfidf", "word_tfidf", "style_meta"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument("--local", type=Path, required=True)
    ap.add_argument("--protocol", choices=["dev", "inner", "both"], default="both")
    a = ap.parse_args()

    protos = ["dev", "inner"] if a.protocol == "both" else [a.protocol]
    worst = {}
    for prot in protos:
        for sv in sorted(a.server.glob(f"scores_{prot}_fold*.npz")):
            lo = a.local / sv.name
            if not lo.exists():
                print(f"[skip] local file missing: {sv.name}")
                continue
            zs, zl = np.load(sv), np.load(lo)
            # join on ev_rows
            s_idx = {int(r): i for i, r in enumerate(zs["ev_rows"])}
            l_idx = {int(r): i for i, r in enumerate(zl["ev_rows"])}
            common = sorted(set(s_idx) & set(l_idx))
            si = np.array([s_idx[r] for r in common])
            li = np.array([l_idx[r] for r in common])
            y_eq = bool(np.array_equal(np.asarray(zs["y"])[si], np.asarray(zl["y"])[li]))
            row = {"n_common": len(common), "y_equal": y_eq,
                   "fused": float(np.abs(zs["fused"][si] - zl["fused"][li]).max()),
                   **{f"s_{c}": float(np.abs(zs[f"s_{c}"][si] - zl[f"s_{c}"][li]).max())
                      for c in COMPS}}
            print(prot, sv.name, json.dumps({k: (round(v, 6) if isinstance(v, float) else v)
                                             for k, v in row.items()}))
            for k, v in row.items():
                if isinstance(v, float):
                    worst[prot] = max(worst.get(prot, 0.0), v)
    print("worst max|delta| per protocol:", {k: round(v, 6) for k, v in worst.items()})
    print("gate reminder: row_score_max_abs <= 1e-3 AND metric_abs <= 1e-3 (guidance §13/§14)")


if __name__ == "__main__":
    main()
