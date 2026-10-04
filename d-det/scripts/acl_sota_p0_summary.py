#!/usr/bin/env python
"""ACL SOTA P0：汇总表（读取 STACAD 官方复现 + 三赛道基线套件产物，生成 summary）。

输出：artifacts/acl_sota_p0/summary_p0_v1.json / summary_p0_v1.md
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
P0 = ROOT / "artifacts" / "acl_sota_p0"


def fmt(x, p=4):
    return "n/a" if x is None else f"{x:.{p}f}"


def main():
    out = {"official_stacad": {}, "tracks": {}}

    rp = P0 / "stacad_official_repro" / "metrics_repro.json"
    if rp.exists():
        out["official_stacad"]["metrics"] = json.load(open(rp))
    cp = P0 / "stacad_official_repro" / "compare_vs_official.json"
    if cp.exists():
        out["official_stacad"]["compare"] = json.load(open(cp))

    for track in ("authorbench_dcan", "stacad_fold0", "droid_fold0"):
        mp = P0 / "server_tracks" / track / "metrics.json"
        if mp.exists():
            out["tracks"][track] = json.load(open(mp))

    json.dump(out, open(P0 / "summary_p0_v1.json", "w"), indent=2, ensure_ascii=False)

    L = ["# P0 汇总（自动生成）", ""]
    if rp.exists():
        L.append("## STACAD 官方协议复现（5 折 OOF macro-F1 mean±std）\n")
        L.append("| 阶段 | 模型 | repro F1 | official F1 | Δ |")
        L.append("|---|---|---|---|---|")
        cmp = out["official_stacad"].get("compare", {})
        for stage, rows in cmp.items():
            for k, r in rows.items():
                d = r["delta"]
                dstr = "" if d is None else f"{d:+.4f}"
                L.append(f"| {stage} | {k} | {fmt(r['repro_f1'])} | {fmt(r['official_f1'])} | {dstr} |")
        L.append("")
    for track, pack in out["tracks"].items():
        L.append(f"## 赛道：{track}（test n={pack['n_test']}，classes={len(pack['classes'])}）\n")
        L.append("| 模型 | macro-F1 | CI95 | balanced acc | ECE | 来源 |")
        L.append("|---|---|---|---|---|---|")
        rows = pack["rows"]
        order = sorted(rows, key=lambda k: -(rows[k]["macro_f1"] or 0))
        for k in order:
            r = rows[k]
            ci = f"[{fmt(r.get('ci95_low'))}, {fmt(r.get('ci95_high'))}]" if "ci95_low" in r else ""
            L.append(f"| {k} | {fmt(r['macro_f1'])} | {ci} | {fmt(r['balanced_acc'])} | "
                     f"{fmt(r['ece'])} | {r.get('provenance', '')} |")
        L.append("")
    (P0 / "summary_p0_v1.md").write_text("\n".join(L), encoding="utf-8")
    print("wrote", P0 / "summary_p0_v1.json", "and summary_p0_v1.md")


if __name__ == "__main__":
    main()
