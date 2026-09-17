"""诊断：m4 样本级判别随序列长度、特征尺度的表现。

用法：
    python scripts/diag_m4.py --ckpt runs/v02/best.pt --split val --limit 2000
    python scripts/diag_m4.py --split train --limit 500      # 看训练集是否可拟合

输出：按长度分桶的 acc/f1 + 预测概率分布 + 各尺度特征单独池化后的线性探针准确率。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from report import build_report  # noqa: E402
from train import build_model, make_dataset, setup_cuda  # noqa: E402

BUCKETS = [(0, 256), (256, 512), (512, 1024), (1024, 2048), (2048, 4096), (4096, 1 << 30)]


def f1(probs: torch.Tensor, labels: torch.Tensor) -> tuple[float, float, float]:
    pred = (probs > 0.5).float()
    tp = float(((pred == 1) & (labels == 1)).sum())
    fp = float(((pred == 1) & (labels == 0)).sum())
    fn = float(((pred == 0) & (labels == 1)).sum())
    p = tp / max(tp + fp, 1e-9)
    r = tp / max(tp + fn, 1e-9)
    return 2 * p * r / max(p + r, 1e-9), p, r


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/udet_base.yaml")
    parser.add_argument("--ckpt", default="runs/v02/best.pt")
    parser.add_argument("--split", default="val")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--probe", action="store_true", help="额外做各尺度线性探针")
    args = parser.parse_args()

    setup_cuda()
    cfg = yaml.safe_load(open(args.config))
    tokenizer = AutoTokenizer.from_pretrained(str(cfg["encoder"]["path"]))
    report = build_report(cfg["report"]["name"], tokenizer=tokenizer,
                          max_tokens=cfg["report"].get("max_tokens", 64))
    model, _ = build_model(cfg, tokenizer)
    device = "cuda"
    if args.ckpt and Path(args.ckpt).exists():
        ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["state"], strict=False)
        print(f"[diag] 载入 {args.ckpt}（epoch {ckpt.get('epoch')}）")
    model.to(device).eval()

    ds = make_dataset(cfg, "m4", args.split, report, False, args.limit or None)
    print(f"[diag] m4/{args.split}: {len(ds)} 条")

    recs = []           # (length, prob, label)
    feats = {}          # 尺度 -> [池化特征]
    labels_all = []
    with torch.no_grad():
        for i in range(len(ds)):
            item = ds[i]
            ids = torch.as_tensor(item["input_ids"], dtype=torch.long, device=device)
            if ids.dim() == 1:
                ids = ids.unsqueeze(0)
            if args.probe:
                feats_in = model.encoder(ids)
                levels, _ = model.backbone(feats_in, return_features=True)
                for k, lv in enumerate(levels):
                    feats.setdefault(k, []).append(lv.mean(dim=1).squeeze(0).float().cpu())
            sample_logits, _ = model(ids)
            prob = float(torch.softmax(sample_logits.float(), dim=-1)[0, 1])
            label = int(torch.as_tensor(item["label"]).reshape(-1)[0])
            recs.append((ids.shape[1], prob, label))
            labels_all.append(label)
            if (i + 1) % 500 == 0:
                print(f"  {i + 1}/{len(ds)}")

    labels = torch.tensor([r[2] for r in recs], dtype=torch.float32)
    probs = torch.tensor([r[1] for r in recs])
    print(f"\n整体: n={len(recs)} 正类占比={labels.mean():.3f} "
          f"平均概率={probs.mean():.3f} 概率std={probs.std():.3f}")
    print(f"      acc={float(((probs > 0.5) == labels.bool()).float().mean()):.4f}")
    for lo, hi in BUCKETS:
        idx = [j for j, r in enumerate(recs) if lo <= r[0] < hi]
        if len(idx) < 20:
            continue
        p, l = probs[idx], labels[idx]
        f, pr, rc = f1(p, l)
        name = f"{lo}~{hi if hi < 1 << 30 else 'inf'}"
        print(f"  len {name:>12}: n={len(idx):5d} acc={float(((p > 0.5) == l.bool()).float().mean()):.4f} "
              f"f1={f:.4f} p={pr:.4f} r={rc:.4f} 平均概率={float(p.mean()):.3f}")

    if args.probe:
        print("\n线性探针（各尺度平均池化特征 -> label，5 折内逻辑回归近似：闭式最小二乘）")
        for k in sorted(feats):
            X = torch.stack(feats[k]).double()
            X = torch.cat([X, torch.ones(len(X), 1, dtype=torch.float64)], dim=1)
            y = labels.double().unsqueeze(1)
            w = torch.linalg.lstsq(X, y).solution
            pred = (X @ w).squeeze(1)
            acc = float(((pred > 0.5) == labels.bool()).float().mean())
            print(f"  尺度 {k}: 特征维度 {X.shape[1] - 1:4d}  训练集线性可分性 acc={acc:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
