#!/usr/bin/env python
"""SemEval 适配微调：冻结骨干 v1.0（编码器 + 原 s1/s2/tok 头）+ 任务头，2ep 预算。

参考 docx/semeval.md：
- 类别不平衡 → 子集已平衡/封顶（prepare）+ sqrt/inv 类权重（本脚本）± focal；
- 泛化 → 头尾截断（prepare）+ 空白扰动增强（本脚本，训练时）+ 分语言/分域评测；
- 口径 → macro-F1（官方 scorer）；A 额外做阈值校准（val 上网格搜）。

任务头：A 复用原 s1 头（+可学 bias）；B/C 新建 Linear(D, K)。

用法（冒烟 / 正式）：
    python scripts/semeval_finetune.py --task a --limit 300 --epochs 1
    python scripts/semeval_finetune.py --task c --weights sqrt
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from encoders import build_encoder  # noqa: E402
from models import build_model  # noqa: E402


# --------------------------------------------------------------------------- #
def pad_collate(batch: list) -> dict:
    length = max(len(b["input_ids"]) for b in batch)
    ids = torch.zeros(len(batch), length, dtype=torch.long)
    mask = torch.zeros(len(batch), length, dtype=torch.long)
    for r, b in enumerate(batch):
        n = len(b["input_ids"])
        ids[r, :n] = torch.tensor(b["input_ids"], dtype=torch.long)
        mask[r, :n] = 1
    return {
        "input_ids": ids,
        "attention_mask": mask,
        "labels": torch.tensor([b["label"] for b in batch], dtype=torch.long),
        "language": [b["language"] for b in batch],
    }


class SevDataset(Dataset):
    def __init__(self, path: str, augment: bool = False, aug_p: float = 0.3,
                 nl_ids: list | None = None, seed: int = 0):
        t = pq.read_table(path)
        self.ids = t.column("ids").to_pylist()
        self.labels = t.column("label").to_pylist()
        self.langs = t.column("language").to_pylist()
        self.augment, self.aug_p = augment, aug_p
        self.nl_ids = nl_ids or []
        self.rng = random.Random(seed)          # 数据增强专用随机源（train 时逐条调用）

    def __len__(self):
        return len(self.ids)

    def _aug_ws(self, ids: list) -> list:
        """空白扰动：删一个换行 token 或插一组换行（对应 R_void 工件；低概率触发）。"""
        if self.rng.random() >= self.aug_p or not self.nl_ids:
            return ids
        ids = list(ids)
        nl_set = set(self.nl_ids)
        pos = [i for i, t in enumerate(ids) if t in nl_set]
        if pos and self.rng.random() < 0.5:
            del ids[self.rng.choice(pos)]
        else:
            at = self.rng.randint(0, len(ids))
            ids[at:at] = self.nl_ids
        return ids

    def __getitem__(self, i):
        ids = self.ids[i]
        if self.augment:
            ids = self._aug_ws(ids)
        return {"input_ids": ids, "label": int(self.labels[i]), "language": self.langs[i]}


class TaskModel(nn.Module):
    """冻结骨干（dual）之上加任务头；A 复用 s1 头 + bias。

    s2_rank（=--readout-rank）>0 时（B/C）：读出经 s2 低秩子空间 z = w2_r(h)
    （row0 由 v1.0 w2 初始化，其余 N(0,1e-3) 防死锁）→ 任务分类器 head2。
    """

    def __init__(self, dual, task: str, num_classes: int, s2_rank: int = 0,
                 readout_init: float = 1e-3):
        super().__init__()
        self.dual = dual
        self.task = task
        self.num_classes = num_classes
        self.s2_rank = int(s2_rank)
        self.use_sub = task != "a" and self.s2_rank > 0
        if task == "a":
            self.head = None
            self.bias = nn.Parameter(torch.zeros(1))
        elif self.use_sub:
            self.w2_new = nn.Linear(dual.dim, self.s2_rank, bias=False)
            with torch.no_grad():
                self.w2_new.weight.zero_()
                self.w2_new.weight[0].copy_(dual.w2.weight.detach().reshape(-1))
                if self.s2_rank > 1:
                    self.w2_new.weight[1:].normal_(0, readout_init)
            self.head2 = nn.Linear(self.s2_rank, num_classes)
            nn.init.normal_(self.head2.weight, std=0.02)
            nn.init.zeros_(self.head2.bias)
        else:
            self.head = nn.Linear(dual.dim, num_classes)
            nn.init.normal_(self.head.weight, std=0.02)
            nn.init.zeros_(self.head.bias)

    def logits(self, input_ids, attention_mask):
        h = self.dual.features(input_ids, attention_mask)
        if self.task == "a":
            return (self.dual.w1(h).squeeze(-1) + self.bias)
        if self.use_sub:
            return self.head2(self.w2_new(h))
        return self.head(h)


def make_class_weight(counts: dict, mode: str, n_cls: int, device) -> torch.Tensor | None:
    if mode == "none":
        return None
    w = torch.ones(n_cls)
    for c in range(n_cls):
        n = max(int(counts.get(c, 0)), 1)
        w[c] = 1.0 / n if mode == "inv" else 1.0 / (n ** 0.5)
    w = w / w.mean()
    return w.to(device)


def focal_ce(logits, labels, weight, gamma: float):
    logp = F.log_softmax(logits.float(), dim=-1)
    pt = logp.gather(1, labels[:, None]).squeeze(1).exp()
    ce = F.nll_loss(logp, labels, weight=weight, reduction="none")
    return (((1.0 - pt) ** gamma) * ce).mean()


@torch.no_grad()
def collect_preds(model, loader, device, amp, amp_dtype):
    model.eval()
    probs, labels, langs = [], [], []
    for batch in tqdm(loader, desc="eval", leave=False):
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        with torch.autocast(amp_dtype, dtype=torch.bfloat16, enabled=amp):
            logits = model.logits(ids, mask)
        if model.task == "a":
            p = torch.sigmoid(logits.float()).cpu().numpy()
        else:
            p = torch.softmax(logits.float(), -1).cpu().numpy()
        probs.append(p)
        labels.extend(batch["labels"].tolist())
        langs.extend(batch["language"])
    probs = np.concatenate(probs, 0)
    return probs, np.array(labels), langs


def metrics_report(task, probs, labels, thr: float | None = None) -> dict:
    if task == "a":
        p = probs
        t = 0.5 if thr is None else float(thr)
        pred = (p >= t).astype(int)
    else:
        p = probs
        pred = probs.argmax(1)
        t = None
    rep = {"macro_f1": float(f1_score(labels, pred, average="macro")),
           "acc": float((pred == labels).mean()), "n": int(len(labels))}
    if t is not None:
        rep["threshold"] = t
    per_class = {}
    for c in sorted(set(labels.tolist()) | set(pred.tolist())):
        tp = int(((pred == c) & (labels == c)).sum())
        fp = int(((pred == c) & (labels != c)).sum())
        fn = int(((pred != c) & (labels == c)).sum())
        prec = tp / max(tp + fp, 1e-9)
        rec = tp / max(tp + fn, 1e-9)
        per_class[str(c)] = {"p": round(prec, 4), "r": round(rec, 4),
                             "f1": round(2 * prec * rec / max(prec + rec, 1e-9), 4),
                             "n": int((labels == c).sum())}
    rep["per_class"] = per_class
    return rep


def tune_threshold(probs, labels) -> float:
    best_t, best_f1 = 0.5, -1.0
    for t in np.arange(0.05, 0.96, 0.01):
        f1 = f1_score(labels, (probs >= t).astype(int), average="macro")
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t


def by_language(task, probs, labels, langs, thr) -> dict:
    out = {}
    for lg in sorted(set(langs)):
        idx = np.array([l == lg for l in langs])
        if idx.sum() == 0:
            continue
        rep = metrics_report(task, probs[idx], labels[idx], thr)
        out[lg] = {"n": rep["n"], "macro_f1": round(rep["macro_f1"], 4),
                   "acc": round(rep["acc"], 4)}
    return out


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["a", "b", "c"], required=True)
    ap.add_argument("--data-dir", default="data/processed/semeval")
    ap.add_argument("--init", default="runs/v0.4.1_covreg/last.pt")
    ap.add_argument("--config", default="configs/ddet_v041.yaml")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr-encoder", type=float, default=2e-5)
    ap.add_argument("--batch", type=int, default=4, help="每批样本数（eq. 批 = batch × grad-accum）")
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--weights", choices=["none", "sqrt", "inv"], default="sqrt")
    ap.add_argument("--loss", choices=["ce", "focal"], default="ce")
    ap.add_argument("--gamma", type=float, default=2.0)
    ap.add_argument("--aug-ws", type=float, default=0.3)
    ap.add_argument("--freeze-encoder", action="store_true")
    ap.add_argument("--readout-rank", type=int, default=0,
                    help="B/C：s2 低秩子空间读出秩（0=直连任务头；r>0 → z=w2_r(h)→head2）")
    ap.add_argument("--readout-init", type=float, default=1e-3,
                    help="低秩读出的非首行初始化标准差（防死锁用；默认 1e-3）")
    ap.add_argument("--limit", type=int, default=None, help="冒烟：每集合只用前 N 条")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-last", action="store_true",
                    help="只存 best.pt，不存 last.pt（大模型省盘）")
    args = ap.parse_args()

    import yaml
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = device == "cuda"
    amp_dtype = "cuda" if amp else "cpu"

    with open(ROOT / args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    enc_cfg = dict(cfg["encoder"])
    enc_cfg["path"] = str(ROOT / enc_cfg["path"])
    enc = build_encoder(**enc_cfg)
    enc.requires_grad_(not args.freeze_encoder)
    dual = build_model("dual", encoder=enc, dim=enc.hidden_size,
                       pool=cfg["model"].get("pool", "mean"),
                       s2_rank=cfg["model"].get("s2_rank", 1))
    if str(args.init).lower() in ("none", "no", ""):
        print("[init] 跳过权重加载（--init none；如 Qwen 骨干：读出全新初始化）") 
    else:
        ck = torch.load(str(ROOT / args.init), map_location="cpu", weights_only=False)
        missing, unexpected = dual.load_state_dict(ck.get("state", {}), strict=False)
        print(f"[init] {args.init}（epoch {ck.get('epoch')}）missing={len(missing)} unexpected={len(unexpected)}")

    num_classes = {"a": 2, "b": 11, "c": 4}[args.task]
    model = TaskModel(dual, args.task, num_classes, s2_rank=args.readout_rank,
                      readout_init=args.readout_init).to(device)

    tok = AutoTokenizer.from_pretrained(enc_cfg["path"])
    nl_ids = tok.encode("\n", add_special_tokens=False)
    print(f"[aug] 空白扰动 p={args.aug_ws} nl_ids={nl_ids}")

    data_dir = ROOT / args.data_dir
    tr_path, va_path, te_path = (data_dir / f"{args.task}_{s}.parquet"
                                 for s in ("train", "val", "test"))
    ds_tr = SevDataset(str(tr_path), augment=args.aug_ws > 0, aug_p=args.aug_ws,
                       nl_ids=nl_ids, seed=args.seed)
    ds_va = SevDataset(str(va_path))
    ds_te = SevDataset(str(te_path))
    if args.limit:
        for ds in (ds_tr, ds_va, ds_te):
            ds.ids = ds.ids[:args.limit]
            ds.labels = ds.labels[:args.limit]
            ds.langs = ds.langs[:args.limit]
    counts = Counter(ds_tr.labels)
    print(f"[data] train n={len(ds_tr)} counts={dict(sorted(counts.items()))} | "
          f"val n={len(ds_va)} | test n={len(ds_te)}")

    loader_tr = DataLoader(ds_tr, batch_size=args.batch, shuffle=True, collate_fn=pad_collate,
                           num_workers=0)
    loader_va = DataLoader(ds_va, batch_size=args.batch, collate_fn=pad_collate, num_workers=0)
    loader_te = DataLoader(ds_te, batch_size=args.batch, collate_fn=pad_collate, num_workers=0)

    enc_params = [p for p in dual.encoder.parameters() if p.requires_grad]
    other = [p for n, p in model.named_parameters()
             if p.requires_grad and not n.startswith("dual.encoder.")]
    groups = [{"params": other, "lr": args.lr}]
    if enc_params:
        groups.append({"params": enc_params, "lr": args.lr_encoder})
    opt = torch.optim.AdamW(groups, lr=args.lr, weight_decay=0.01)
    accum = max(1, args.grad_accum)
    updates = max(1, len(loader_tr) // accum) * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g["lr"] for g in groups],
                                                total_steps=updates, pct_start=0.05)
    loss_weight = make_class_weight(counts, args.weights, num_classes, device)
    if loss_weight is not None:
        print(f"[loss] {args.loss} weights={args.weights} -> {loss_weight.tolist()}")

    out_dir = ROOT / (args.out or f"runs/semeval_{args.task}")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "args.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    best_f1 = -1.0
    for epoch in range(args.epochs):
        model.train()
        if args.freeze_encoder:
            model.dual.encoder.eval()
        bar = tqdm(loader_tr, desc=f"epoch {epoch}", unit="it")
        running, seen = 0.0, 0
        opt.zero_grad(set_to_none=True)
        for step, batch in enumerate(bar):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            with torch.autocast(amp_dtype, dtype=torch.bfloat16, enabled=amp):
                logits = model.logits(ids, mask)
                if args.task == "a":
                    loss = F.binary_cross_entropy_with_logits(logits.float(),
                                                              labels.float())
                elif args.loss == "focal":
                    loss = focal_ce(logits, labels, loss_weight, args.gamma)
                else:
                    loss = F.cross_entropy(logits.float(), labels, weight=loss_weight)
            (loss / accum).backward()
            running += float(loss.detach())
            seen += 1
            if (step + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters()
                                                if p.requires_grad], 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
            if (step + 1) % 200 == 0:
                bar.set_postfix(loss=f"{running / max(seen, 1):.4f}")
        bar.close()

        probs_va, labels_va, langs_va = collect_preds(model, loader_va, device, amp, amp_dtype)
        thr = tune_threshold(probs_va, labels_va) if args.task == "a" else None
        rep_va = metrics_report(args.task, probs_va, labels_va, thr)
        print(f"[val {epoch}] macro_f1={rep_va['macro_f1']:.4f} acc={rep_va['acc']:.4f} "
              f"thr={rep_va.get('threshold')}")
        with open(out_dir / "metrics.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"epoch": epoch, "train_loss": running / max(seen, 1),
                                "val": rep_va}) + "\n")
        state = {k: v for k, v in model.state_dict().items()}
        if not args.no_last:
            torch.save({"state": state, "epoch": epoch, "task": args.task,
                        "args": vars(args)}, out_dir / "last.pt")
        if rep_va["macro_f1"] > best_f1:
            best_f1 = rep_va["macro_f1"]
            torch.save({"state": state, "epoch": epoch, "task": args.task,
                        "args": vars(args)}, out_dir / "best.pt")
            print(f"[ckpt] best.pt (val macro_f1={best_f1:.4f})")

    # 最终评测：val（含阈值）/ test（用 val 调出的阈值）
    probs_va, labels_va, langs_va = collect_preds(model, loader_va, device, amp, amp_dtype)
    thr = tune_threshold(probs_va, labels_va) if args.task == "a" else None
    va = metrics_report(args.task, probs_va, labels_va, thr)
    va["by_language"] = by_language(args.task, probs_va, labels_va, langs_va, thr)
    probs_te, labels_te, langs_te = collect_preds(model, loader_te, device, amp, amp_dtype)
    te = metrics_report(args.task, probs_te, labels_te, thr)
    te["by_language"] = by_language(args.task, probs_te, labels_te, langs_te, thr)
    if args.task == "a":
        te["macro_f1@0.5"] = round(metrics_report("a", probs_te, labels_te, 0.5)["macro_f1"], 4)
    result = {"task": args.task, "init": args.init, "weights": args.weights,
              "loss": args.loss, "epochs": args.epochs, "n_train": len(ds_tr),
              "val": va, "test": te}
    np.savez_compressed(out_dir / "probs_val.npz", probs=np.asarray(probs_va),
                        y=np.asarray(labels_va))
    np.savez_compressed(out_dir / "probs_test.npz", probs=np.asarray(probs_te),
                        y=np.asarray(labels_te))
    with open(out_dir / "eval.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[final] val macro_f1={va['macro_f1']:.4f} | test macro_f1={te['macro_f1']:.4f}"
          f"{' @thr=' + str(round(thr, 2)) if thr else ''}")
    print(f"[out] {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
