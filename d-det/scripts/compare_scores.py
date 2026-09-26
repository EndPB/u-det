"""对比两个 run 的同流分数结构（消融分析用）。

用法：
    python scripts/compare_scores.py --a runs/v0.1.0 --b runs/v0.1.0_ab_nopair
                                     [--stream m4] [--split test]

输出：s1 相关性（Pearson/Spearman）、两版各自错误集与重叠、
按标签/长度/生成模型分组的差异。
（只读 scores_*.pt，不占 GPU。）
"""
import argparse
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def load_scores(run_dir: Path, stream: str, split: str):
    path = run_dir / f"scores_{stream}_{split}.pt"
    if not path.exists():
        raise SystemExit(f"找不到 {path}")
    blob = torch.load(path, map_location="cpu", weights_only=False)
    return blob


def main() -> int:
    ap = argparse.ArgumentParser(description="对比两个 run 的同流分数结构")
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--stream", default="m4")
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    ra = Path(args.a) if Path(args.a).is_absolute() else ROOT / args.a
    rb = Path(args.b) if Path(args.b).is_absolute() else ROOT / args.b
    A = load_scores(ra, args.stream, args.split)
    B = load_scores(rb, args.stream, args.split)

    sa = A["s1"].numpy().astype(np.float64)
    sb = B["s1"].numpy().astype(np.float64)
    y = A["label"].numpy().astype(int)
    assert (y == B["label"].numpy().astype(int)).all(), "两版样本顺序不一致！"
    la = A["length"].numpy().astype(int)
    pa = 1.0 / (1.0 + np.exp(-sa))
    pb = 1.0 / (1.0 + np.exp(-sb))
    da = (pa >= 0.5).astype(int)
    db = (pb >= 0.5).astype(int)

    tag_a = ra.name
    tag_b = rb.name
    print(f"stream={args.stream} split={args.split} n={len(y)} AI={y.mean():.3f}")
    print(f"[s1] pearson={np.corrcoef(sa, sb)[0, 1]:.4f} "
          f"spearman={np.corrcoef(sa.argsort().argsort(), sb.argsort().argsort())[0, 1]:.4f}")
    print(f"[mean s1] {tag_a} {sa.mean():+.3f} | {tag_b} {sb.mean():+.3f}")
    ea, eb = da != y, db != y
    print(f"[acc] {tag_a} {(da == y).mean():.4f} | {tag_b} {(db == y).mean():.4f}")
    print(f"[错误] {tag_a}={ea.sum()} {tag_b}={eb.sum()} 重叠={(ea & eb).sum()} "
          f"{tag_a}独有={(ea & ~eb).sum()} {tag_b}独有={(~ea & eb).sum()}")
    for name, m in [("human", y == 0), ("ai", y == 1)]:
        if m.sum() == 0:
            continue
        print(f"  {name}: 错{tag_a}={((ea & m).sum())} / 错{tag_b}={((eb & m).sum())} / "
              f"重叠={(ea & eb & m).sum()}")
    for lo, hi in [(0, 512), (512, 2048), (2048, 8192), (8192, 10**9)]:
        m = (la >= lo) & (la < hi)
        if m.sum() == 0:
            continue
        print(f"  len[{lo},{hi}): n={m.sum()} mean|dp|={np.abs(pa - pb)[m].mean():.4f} "
              f"错{tag_a}={(ea & m).sum()}/错{tag_b}={(eb & m).sum()}")

    meta = A.get("meta")
    if meta:
        models = np.array([(m or {}).get("model") or "human?" if isinstance(m, dict) else "?"
                           for m in meta], dtype=object)
        for mm in sorted(set(map(str, models))):
            m = np.array([str(x) == mm for x in models])
            if m.sum() == 0:
                continue
            print(f"  model={mm:<10} n={m.sum():<5} AI占比={y[m].mean():.2f} "
                  f"acc{tag_a}={(da[m] == y[m]).mean():.3f} acc{tag_b}={(db[m] == y[m]).mean():.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
