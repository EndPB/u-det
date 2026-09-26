"""配对泛化探针：用指定 ckpt 的 s2 对「任意 pair parquet」报方向正确率。

用途：跨模型族 / 跨规模泛化（零训练成本）——例如用 v0.1.0（Qwen2.5-0.5B 对训练）
的 s2 去判别 Qwen2.5-1.5B 的 base/instruct 配对，验证「零标注自适应」叙事。

要点：
- 直接读 parquet 自带的 ``input_ids_plus/minus``（与生成时口径逐字节一致，
  避免任何分词差异）；
- 判定口径与 ``train.py --eval`` 相同：rank=1 用带符号差值 (s2₊−s2₋)，
  rank>1 用 L2 范数；dir_acc = 差值为正的比例；
- 只读不写 run 目录（--out 可由调用方指定路径）。
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import pyarrow.parquet as pq
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from encoders import build_encoder                    # noqa: E402
from models import build_model                        # noqa: E402
from train import apply_overrides, load_config        # noqa: E402


@torch.no_grad()
def score_s2(model, ids, device):
    t = torch.tensor([ids], device=device)
    _, s2 = model(t, torch.ones_like(t))
    return s2.reshape(-1).float().cpu()


def main() -> int:
    ap = argparse.ArgumentParser(description="配对泛化探针（s2 方向正确率）")
    ap.add_argument("--config", default="configs/ddet_base.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--pair-file", required=True)
    ap.add_argument("--splits", default="val,test")
    ap.add_argument("--out", default=None, help="结果 json 路径（可选）")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--set", action="append", default=None, metavar="段.键=值")
    args = ap.parse_args()

    cfg = apply_overrides(load_config(args.config), args)
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"

    enc_cfg = dict(cfg["encoder"])
    if not Path(enc_cfg["path"]).is_absolute():
        enc_cfg["path"] = str(ROOT / enc_cfg["path"])
    encoder = build_encoder(**enc_cfg)
    mcfg = dict(cfg["model"])
    model = build_model(mcfg.pop("name", "dual"), encoder=encoder,
                        dim=encoder.hidden_size, **mcfg)
    ckpt_path = Path(args.ckpt) if Path(args.ckpt).is_absolute() else ROOT / args.ckpt
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(ckpt["state"], strict=False)
    if missing or unexpected:
        print("[pairgen] 注意：未加载 %d 项、多余 %d 项"
              % (len(missing), len(unexpected)))
    model.to(device).eval()
    print("[pairgen] ckpt=%s（s2_rank=%d）pair_file=%s"
          % (args.ckpt, model.s2_rank, args.pair_file))

    pf = Path(args.pair_file)
    if not pf.is_absolute():
        pf = ROOT / pf
    table = pq.read_table(pf)

    summary = {}
    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        idx = [i for i, s in enumerate(table.column("split").to_pylist()) if s == split]
        if not idx:
            continue
        gaps = []
        for i in idx:
            s2p = score_s2(model, table.column("input_ids_plus")[i].as_py(), device)
            s2m = score_s2(model, table.column("input_ids_minus")[i].as_py(), device)
            if model.s2_rank == 1:
                gaps.append(float(s2p[0] - s2m[0]))
            else:
                gaps.append(float((s2p - s2m).norm()))
        n = len(gaps)
        pos = sum(1 for g in gaps if g > 0)
        summary[split] = {
            "n": n,
            "dir_acc": pos / n,
            "delta_mean": statistics.fmean(gaps),
            "delta_median": statistics.median(gaps),
        }
        print("[pairgen] %-5s n=%-4d dir_acc=%.4f  Δmean=%+.3f  Δmedian=%+.3f"
              % (split, n, summary[split]["dir_acc"],
                 summary[split]["delta_mean"], summary[split]["delta_median"]))

    if args.out:
        out = Path(args.out)
        if not out.is_absolute():
            out = ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump({"ckpt": args.ckpt, "pair_file": str(pf), "summary": summary},
                      f, ensure_ascii=False, indent=2)
        print("[pairgen] 结果已写入 %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
