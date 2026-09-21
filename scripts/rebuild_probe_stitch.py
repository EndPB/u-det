"""从两次 cut-and-stitch 日志里重建合并后的 ``probe_stitch.json``。

背景：`--stage stitch` 原本是**覆盖**写 `runs/<tag>/probe_stitch.json`，
于是"补跑 k=2"把第一次的 k=0/3/4 覆盖掉了（B2 类陷阱：后续动作覆盖正式产物）。
脚本已改为**合并写入**；本文件负责把已被覆盖的那部分从日志里恢复。

用法：``OMP_NUM_THREADS=1 python scripts/rebuild_probe_stitch.py``
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/v0.4.5/probe_stitch.json"
LOGS = ["/tmp/probe_stitch_v045.log", "/tmp/probe_stitch_k2.log"]

PAT = re.compile(
    r"\[P7/(raw|bridge)\] 去掉尾 (\d+) 层（截断到第 (\d+) 层）："
    r"m4=([\d.]+) line=([\d.]+) chunk=([\d.]+) token=([\d.]+)"
)


def main() -> int:
    rows: dict[str, dict] = {}
    for log in LOGS:
        p = Path(log)
        if not p.exists():
            print(f"[warn] 缺日志 {log}")
            continue
        txt = p.read_text(encoding="utf-8", errors="ignore")
        for m in PAT.finditer(txt):
            variant, k, layer, m4, line, chunk, token = m.groups()
            rows.setdefault(k, {"layer": int(layer)})[variant] = {
                "m4": {"sample_f1": float(m4)},
                "hybrid": {"line_f1": float(line), "chunk_f1": float(chunk),
                           "token_f1": float(token)},
                "source": log,
            }
    if not rows:
        print("没有从日志里解析到任何行，先确认 --stage stitch 跑过")
        return 1
    blob = {
        "ckpt": "runs/v0.4.5/best.pt",
        "ks": sorted(int(k) for k in rows),
        "bridge": True,
        "rows": rows,
        "note": ("本文件由两次运行合并而成（原文件被后续 k=2 运行覆盖）："
                 "k=0/3/4 来自 /tmp/probe_stitch_v045.log，k=2 来自 /tmp/probe_stitch_k2.log。"
                 "每行的 metrics 只保留了主指标（m4 sample_f1 / hybrid line-chunk-token f1），"
                 "完整 evaluate 输出见各自日志。"),
    }
    OUT.write_text(json.dumps(blob, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[ok] 已写 {OUT.relative_to(ROOT)}：ks={blob['ks']}")
    for k in blob["ks"]:
        r = rows[str(k)]
        for v in ("raw", "bridge"):
            if v in r:
                h = r[v]["hybrid"]
                print(f"   k={k} {v:>6}  m4={r[v]['m4']['sample_f1']:.4f} "
                      f"line={h['line_f1']:.4f} chunk={h['chunk_f1']:.4f} token={h['token_f1']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
