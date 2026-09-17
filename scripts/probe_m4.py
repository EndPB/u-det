"""线性探针：m4 样本级信号在各尺度特征上的可解码性（train 拟合 -> val 评测）。

回答的问题：v0.2 样本级偏低，是"瓶颈特征里没有信息"还是"样本头没学好"。

用法：
    python scripts/probe_m4.py --ckpt runs/v03/best.pt --train-limit 6000
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


@torch.no_grad()
def collect(model, ds, device, levels_out):
    feats = {}
    labels = []
    for i in range(len(ds)):
        item = ds[i]
        ids = torch.as_tensor(item["input_ids"], dtype=torch.long, device=device).unsqueeze(0)
        hidden = model.encoder(ids)
        levels, _ = model.backbone(hidden, return_features=True)
        for k in range(levels_out or len(levels)):
            feats.setdefault(k, []).append(levels[k].mean(dim=1).squeeze(0).float().cpu())
        labels.append(int(torch.as_tensor(item["label"]).reshape(-1)[0]))
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{len(ds)}")
    return feats, torch.tensor(labels, dtype=torch.float32)


def fit_eval(Xtr, ytr, Xva, yva):
    def design(X):
        return torch.cat([X, torch.ones(len(X), 1, dtype=X.dtype)], dim=1)

    A = design(Xtr.double()).T @ design(Xtr.double()) + 1e-2 * torch.eye(Xtr.shape[1] + 1, dtype=torch.float64)
    w = torch.linalg.solve(A, design(Xtr.double()).T @ ytr.double().unsqueeze(1))
    logit = design(Xva.double()) @ w
    pred = (logit.squeeze(1) > 0.5).float()
    acc = float((pred == yva).float().mean())
    tp = float(((pred == 1) & (yva == 1)).sum())
    fp = float(((pred == 1) & (yva == 0)).sum())
    fn = float(((pred == 0) & (yva == 1)).sum())
    p = tp / max(tp + fp, 1e-9)
    r = tp / max(tp + fn, 1e-9)
    return acc, 2 * p * r / max(p + r, 1e-9)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/udet_base.yaml")
    ap.add_argument("--ckpt", default="runs/v03/best.pt")
    ap.add_argument("--train-limit", type=int, default=6000)
    ap.add_argument("--val-limit", type=int, default=0)
    args = ap.parse_args()

    setup_cuda()
    cfg = yaml.safe_load(open(args.config))
    tokenizer = AutoTokenizer.from_pretrained(str(cfg["encoder"]["path"]))
    report = build_report(cfg["report"]["name"], tokenizer=tokenizer,
                          max_tokens=cfg["report"].get("max_tokens", 64))
    model, _ = build_model(cfg, tokenizer)
    device = "cuda"
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state"], strict=False)
    model.to(device).eval()
    print(f"[probe] 载入 {args.ckpt}（epoch {ckpt.get('epoch')}）")

    tr = make_dataset(cfg, "m4", "train", report, False, args.train_limit or None)
    va = make_dataset(cfg, "m4", "val", report, False, args.val_limit or None)
    print(f"[probe] train={len(tr)} val={len(va)}")
    ftr, ytr = collect(model, tr, device, 0)
    fva, yva = collect(model, va, device, 0)

    print("\n尺度  特征dim   val acc   val f1   （线性探针：train 拟合）")
    for k in sorted(ftr):
        Xtr, Xva = torch.stack(ftr[k]), torch.stack(fva[k])
        acc, f1 = fit_eval(Xtr, ytr, Xva, yva)
        name = "瓶颈L/16" if k == 0 else f"L/{2 ** (len(ftr) - 1 - k)}"
        print(f"  {k}  {Xtr.shape[1]:5d}   {acc:.4f}  {f1:.4f}   ({name})")

    idx = torch.randint(0, len(va), (len(va),))
    print(f"\n参照：朴素规则 全部预测正类 val acc={float(yva.mean()):.4f}")
    lo = torch.tensor([va[i]["input_ids"].__len__() < 1024 for i in range(len(va))], dtype=torch.float32)
    print(f"参照：纯长度规则（<1024 猜正）val acc={float((lo == yva).float().mean()):.4f}")
    del idx
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
