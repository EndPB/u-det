"""零训练诊断：哪一层 CodeT5 给出最好的"无上下文 token 特征"。

U-Det 的编码器是逐 token、无上下文的（seq_len=1）。取**最后一层**隐状态是习惯，
但 CodeT5 的最后一层是为**上下文**服务的——喂长度=1 是它预训练时几乎不会遇到的形态。
词嵌入层（layer=0）才是"无上下文特征"的经典来源。

这里对若干层各构一张 (vocab, D) 查表，用**同一个上下文无关读出**
（token 平均池化 → 线性探针，train 拟合 → val 评测）比较其 m4 判别力 —— 不训练任何模型。

用法：
    python scripts/probe_lut_layers.py --layers -1 0 3 6 9 --train-limit 8000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from encoders.codet5 import CodeT5TokenEncoder  # noqa: E402
from report import build_report  # noqa: E402
from train import make_dataset, setup_cuda  # noqa: E402


def collect_ids(cfg, split, limit, report):
    ds = make_dataset(cfg, "m4", split, report, False, limit or None)
    ids = [torch.as_tensor(ds[i]["input_ids"], dtype=torch.long) for i in range(len(ds))]
    labels = torch.tensor([int(torch.as_tensor(ds[i]["label"]).reshape(-1)[0]) for i in range(len(ds))],
                          dtype=torch.float32)
    return ids, labels


def pooled(lut: torch.Tensor, ids: list[torch.Tensor]) -> torch.Tensor:
    """token 平均池化（上下文无关读出）。"""
    return torch.stack([lut[i.to(lut.device)].float().mean(0).cpu() for i in ids])


def fit_eval(Xtr, ytr, Xva, yva):
    def design(X):
        return torch.cat([X.double(), torch.ones(len(X), 1, dtype=torch.float64)], dim=1)

    A = design(Xtr).T @ design(Xtr) + 1e-2 * torch.eye(Xtr.shape[1] + 1, dtype=torch.float64)
    w = torch.linalg.solve(A, design(Xtr).T @ ytr.double().unsqueeze(1))
    pred = (design(Xva) @ w).squeeze(1) > 0.5
    acc = float((pred.float() == yva).float().mean())
    tp = float(((pred == 1) & (yva == 1)).sum())
    fp = float(((pred == 1) & (yva == 0)).sum())
    fn = float(((pred == 0) & (yva == 1)).sum())
    p = tp / max(tp + fp, 1e-9)
    r = tp / max(tp + fn, 1e-9)
    return acc, 2 * p * r / max(p + r, 1e-9)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/udet_v03.yaml")
    ap.add_argument("--layers", type=int, nargs="+", default=[-1, 0, 3, 6, 9])
    ap.add_argument("--train-limit", type=int, default=8000)
    ap.add_argument("--val-limit", type=int, default=0)
    args = ap.parse_args()

    setup_cuda()
    cfg = yaml.safe_load(open(args.config))
    tokenizer = AutoTokenizer.from_pretrained(str(cfg["encoder"]["path"]))
    report = build_report(cfg["report"]["name"], tokenizer=tokenizer,
                          max_tokens=cfg["report"].get("max_tokens", 64))

    print("[1/2] 收集 m4 的 token 序列（train / val）")
    ids_tr, y_tr = collect_ids(cfg, "train", args.train_limit, report)
    ids_va, y_va = collect_ids(cfg, "val", args.val_limit, report)
    print(f"      train={len(ids_tr)} val={len(ids_va)}")

    print("\n[2/2] 逐层构表 + 上下文无关线性探针（不训练任何模型）")
    print("  layer   特征dim    val acc   val f1")
    for layer in args.layers:
        enc = CodeT5TokenEncoder(path=cfg["encoder"]["path"], lut=f"/tmp/lut_layer{layer}.pt",
                                 layer=layer, freeze=True)
        lut = enc.lut.float().cuda()
        acc, f1 = fit_eval(pooled(lut, ids_tr), y_tr, pooled(lut, ids_va), y_va)
        tag = "最后一层" if layer == -1 else ("词嵌入层" if layer == 0 else f"第 {layer} 层")
        print(f"  {layer:>5}    768      {acc:.4f}  {f1:.4f}   ({tag})")
        del lut, enc
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
