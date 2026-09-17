#!/usr/bin/env python
"""U-Det 训练 / 评测主干（根目录单文件）。

模型：逐 token 冻结编码器（查表）-> 1D U-Net（瓶颈可插 Transformer）-> 样本级头 + 多尺度 token 级头。
数据：双流并行（同一 step 同时参与，共享一个 optimizer step）——
      m4（样本级：human 类 + AI 类）与 hybrid（token 级：行级标注引导上采样）。
损失：样本级 CE + 各上采样尺度 token 级 BCE（标签按尺度池化，权重由粗到细）。# 双流：每个 step 从 m4 与 hybrid 各取一个 batch，两个 forward 后累加梯度，只做一次 opt.step()
# （两个任务同时端到端训练；两流步数不一致时用 cycle() 轮转较短的那条，epoch 长度取两者最大值）。
用法::

    python train.py --config configs/udet_base.yaml                  # 训练
    python train.py --config configs/udet_base.yaml --eval --ckpt runs/xxx/best.pt   # 评测
    python train.py --limit 200 --epochs 1 --max-length 512 --tag smoke             # 冒烟
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from dataio import build_dataset, collate
from encoders import build_encoder
from models import PooledClassifier, SampleHead, TokenHeads, build_mid, build_unet, downsample_mask
from report import build_report

ROOT = Path(__file__).resolve().parent
IGNORE = -100


# --------------------------------------------------------------------------- #
# 基础工具
# --------------------------------------------------------------------------- #
def resolve(path: str) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def load_config(path: str) -> dict:
    with open(resolve(path), encoding="utf-8") as f:
        return yaml.safe_load(f)


def apply_overrides(cfg: dict, args) -> dict:
    """命令行覆盖配置（冒烟用）。"""
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.batch_size is not None:
        for name in cfg["train"]["batch_size"]:
            cfg["train"]["batch_size"][name] = args.batch_size
    if args.lr is not None:
        cfg["train"]["lr"] = args.lr
    if args.max_length is not None:
        for name in cfg["train"]["max_length"]:
            cfg["train"]["max_length"][name] = args.max_length
        cfg["train"]["eval_max_length"] = args.max_length
    if args.no_m4:
        cfg["train"]["streams"] = [s for s in cfg["train"]["streams"] if s != "m4"]
    if args.no_hybrid:
        cfg["train"]["streams"] = [s for s in cfg["train"]["streams"] if s != "hybrid"]
    if args.report is not None:
        cfg["report"]["name"] = args.report
    if args.mid is not None:
        cfg["model"]["mid"]["name"] = args.mid
    if args.model is not None:
        cfg["model"]["name"] = args.model
    if args.freeze_encoder:
        cfg["encoder"]["freeze"] = True
    if args.unfreeze_encoder:
        cfg["encoder"]["freeze"] = False
    return cfg


def sliding_windows(total: int, max_length: int, stride: int | None = None) -> list[tuple[int, int]]:
    """把长度 total 切成 [start, end) 窗口（评测时滑窗合并用）。"""
    if total <= max_length:
        return [(0, total)]
    stride = stride or max(1, max_length // 2)
    out = []
    start = 0
    while True:
        end = min(start + max_length, total)
        out.append((start, end))
        if end >= total:
            return out
        start += stride


def cycle(loader):
    """无限循环取数（双流长度不一致时循环较短的流）。"""
    while True:
        yield from loader


def truncate(dataset, limit: int | None):
    """冒烟用：等距抽取 limit 条（跨类别/语言均匀，不是只取头部）。"""
    if not limit or len(dataset) <= limit:
        return dataset
    total = len(dataset)
    index = [i * total // limit for i in range(limit)]
    for attr in ("ids", "codes", "labels", "meta", "tok", "line_of_token", "line_label"):
        seq = getattr(dataset, attr, None)
        if isinstance(seq, list) and len(seq) == total:
            setattr(dataset, attr, [seq[i] for i in index])
    return dataset


def split_counts(dataset) -> dict:
    from collections import Counter
    keys = [k for k in ("language", "model_id") if any(k in m for m in dataset.meta[:50])]
    info = {k: dict(Counter(str(m.get(k)) for m in dataset.meta)) for k in keys}
    return {"n": len(dataset), "labels": dict(Counter(dataset.labels)), **info}


# --------------------------------------------------------------------------- #
# 模型组装
# --------------------------------------------------------------------------- #
class UDet(nn.Module):
    """编码器（逐 token 查表）-> U-Net（含瓶颈模块）-> 样本级头 + 多尺度 token 级头。"""

    def __init__(self, encoder: nn.Module, unet: nn.Module, sample_head: nn.Module, token_heads: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.unet = unet
        self.sample_head = sample_head
        self.token_heads = token_heads
        self.align_multiple = 2 ** unet.depth          # 多尺度池化要求长度对齐到 2^depth

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        feats = self.encoder(input_ids, attention_mask)                       # (B, L, D)
        _, scales, bottleneck = self.unet(
            feats.transpose(1, 2), mask=attention_mask, return_features=True  # (B, D, L) -> ...
        )
        down_mask = downsample_mask(attention_mask, 1, self.unet.depth)
        sample_logits = self.sample_head(bottleneck, down_mask)               # (B, 2)
        token_logits = self.token_heads([bottleneck] + list(scales))          # 由粗到细，每层 (B, 1, L_k)
        return sample_logits, token_logits


# 直接二分类基线的名称（编码器 + 池化 + 线性头，无 U-Net / 无 token 头）
BASELINE_MODELS = ("codet5cls", "pooled")


def build_model(cfg: dict, tokenizer) -> tuple[nn.Module, nn.Module]:
    """按 cfg 组装模型：默认 U-Det；model.name 为基线名时构建直接二分类基线。

    编码器是否可训练由 ``encoder.freeze`` 控制（逐 token 查表编码器无参数，恒为“冻结”）。
    """
    encoder = build_encoder(**cfg["encoder"])
    freeze = bool(cfg["encoder"].get("freeze", True))
    if freeze:
        encoder.eval()
        encoder.requires_grad_(False)

    mcfg = dict(cfg["model"])
    name = mcfg.pop("name", "unet1d")
    if name in BASELINE_MODELS:                                   # 基线：编码器 -> 池化 -> 线性分类
        bcfg = dict(mcfg.get("baseline", {}))
        return PooledClassifier(encoder, dim=encoder.hidden_size, **bcfg), encoder

    mid = build_mid(**mcfg.pop("mid", {"name": "none"}))
    mcfg.pop("in_channels", None)
    mcfg.pop("baseline", None)
    unet = build_unet(name, in_channels=encoder.hidden_size,
                      out_channels=None, mid=mid, **mcfg)      # out_channels=None：只当特征提取器

    hcfg = dict(cfg.get("heads", {}))
    sample_head = SampleHead(dim=unet.features[-1], hidden=hcfg.pop("sample_hidden", None),
                             dropout=hcfg.pop("sample_dropout", 0.0))
    token_heads = TokenHeads(list(reversed(unet.features)), out=1)
    return UDet(encoder, unet, sample_head, token_heads), encoder


# --------------------------------------------------------------------------- #
# 损失
# --------------------------------------------------------------------------- #
def token_loss(logit_scales: list[torch.Tensor], tok_labels: torch.Tensor,
               weights: list[float], ignore: int = IGNORE) -> torch.Tensor | None:
    """各尺度 token 级 BCE：标签按尺度平均池化成软标签（对应不同文本长度精度）。"""
    valid = (tok_labels != ignore)
    if not valid.any():
        return None
    target = tok_labels.clamp(min=0).float()
    length = tok_labels.shape[-1]
    total, total_w = 0.0, 0.0
    for logits, weight in zip(logit_scales, weights):
        k = max(1, math.ceil(length / logits.shape[-1]))
        num = F.avg_pool1d((target * valid).unsqueeze(1), k, k, ceil_mode=True)
        den = F.avg_pool1d(valid.float().unsqueeze(1), k, k, ceil_mode=True)
        keep = (den > 0).float()
        if keep.sum() == 0:
            continue
        soft = num / den.clamp(min=1e-6)
        loss = F.binary_cross_entropy_with_logits(logits, soft, weight=keep, reduction="sum") / keep.sum()
        total = total + weight * loss
        total_w += weight
    return None if total_w == 0 else total / total_w


def batch_losses(cfg: dict, sample_logits: torch.Tensor, token_logits: list[torch.Tensor] | None,
                 batch: dict, device: str) -> tuple[torch.Tensor, dict]:
    """一个 batch 的总损失与分项（样本级 + token 级；基线模型无 token 头时只算样本级）。"""
    loss_cfg = cfg.get("loss", {})
    labels = batch["labels"].to(device)

    class_weights = loss_cfg.get("class_weights")
    weight = None if not class_weights else torch.tensor(class_weights, device=device, dtype=torch.float32)
    l_sample = F.cross_entropy(sample_logits.float(), labels, weight=weight)

    l_token = None
    if token_logits is not None and loss_cfg.get("token", 1.0) > 0:
        tok_labels = batch["tok_labels"].to(device)
        l_token = token_loss(token_logits, tok_labels,
                             loss_cfg.get("scale_weights", [1.0] * len(token_logits)))
    parts = {"sample": float(l_sample.detach()), "token": 0.0}
    total = loss_cfg.get("sample", 1.0) * l_sample
    if l_token is not None:
        total = total + loss_cfg.get("token", 1.0) * l_token
        parts["token"] = float(l_token.detach())
    return total, parts


# --------------------------------------------------------------------------- #
# 评测
# --------------------------------------------------------------------------- #
def _chunk_runs(flags: list[int]) -> list[tuple[int, int]]:
    """把 0/1 序列里连续的 1 归并成 [start, end) 段。"""
    runs, start = [], None
    for i, f in enumerate(flags):
        if f and start is None:
            start = i
        elif not f and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(flags)))
    return runs


def chunk_match(pred: list[int], gold: list[int], iou_threshold: float = 0.5) -> tuple[int, int, int]:
    """片段级（连续 AI 行段）匹配：按 IoU >= 阈值贪心匹配，返回 (命中, 预测段数, 标注段数)。"""
    pred_runs, gold_runs = _chunk_runs(pred), _chunk_runs(gold)
    used, tp = set(), 0
    for p in pred_runs:
        best, best_iou = -1, 0.0
        for j, g in enumerate(gold_runs):
            if j in used:
                continue
            inter = max(0, min(p[1], g[1]) - max(p[0], g[0]))
            union = max(p[1], g[1]) - min(p[0], g[0])
            iou = inter / union if union else 0.0
            if iou > best_iou:
                best, best_iou = j, iou
        if best >= 0 and best_iou >= iou_threshold:
            used.add(best)
            tp += 1
    return tp, len(pred_runs), len(gold_runs)


def prf1(pred: list[int], gold: list[int]) -> dict:
    tp = sum(1 for p, g in zip(pred, gold) if p == 1 and g == 1)
    fp = sum(1 for p, g in zip(pred, gold) if p == 1 and g == 0)
    fn = sum(1 for p, g in zip(pred, gold) if p == 0 and g == 1)
    precision = tp / max(tp + fp, 1e-9)
    recall = tp / max(tp + fn, 1e-9)
    return {"p": precision, "r": recall, "f1": 2 * precision * recall / max(precision + recall, 1e-9)}


@torch.no_grad()
def evaluate(model: UDet, dataset, name: str, cfg: dict, device: str, report=None) -> dict:
    """滑窗评测（超长样本按窗口合并）：m4 出样本级 ACC/F1；hybrid 出行级 / token 级 / 片段级指标。"""
    model.eval()
    max_length = cfg["train"]["eval_max_length"]
    batch_size = cfg["train"].get("eval_batch_size", 16)
    is_hybrid = name == "hybrid"

    jobs = []                                                  # (样本下标, start, end)
    for i in range(len(dataset)):
        for start, end in sliding_windows(len(dataset.ids[i]), max_length):
            jobs.append((i, start, end))

    sample_logits = torch.zeros(len(dataset), 2)
    token_hits = [[0, 0, 0, 0] for _ in range(len(dataset))]    # tp/fp/fn/tn（token 级）
    line_sum = [None] * len(dataset)                            # 行级概率累加（hybrid）
    line_cnt = [None] * len(dataset)
    report_cache: dict[int, list[int]] = {}

    for chunk in itertools.batched(jobs, batch_size):
        ids = []
        for i, start, end in chunk:
            piece = list(dataset.ids[i][start:end])
            if start == 0 and report is not None:               # 报告前缀只加在该样本第一个窗口
                if i not in report_cache:
                    report_cache[i] = report.ids(dataset.codes[i])
                piece = list(report_cache[i]) + piece
            ids.append(piece)
        multiple = int(getattr(model, "align_multiple", 8))
        length = math.ceil(max(len(x) for x in ids) / multiple) * multiple
        input_ids = torch.zeros((len(ids), length), dtype=torch.long)
        attention_mask = torch.zeros((len(ids), length), dtype=torch.long)
        for row, x in enumerate(ids):
            input_ids[row, :len(x)] = torch.tensor(x, dtype=torch.long)
            attention_mask[row, :len(x)] = 1
        input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)

        amp = cfg["train"].get("amp", True) and device == "cuda"
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            logits_sample, token_logits = model(input_ids, attention_mask)
        has_token = token_logits is not None                       # 基线模型没有 token 头
        probs = torch.sigmoid(token_logits[-1].float())[:, 0, :] if has_token else None   # 最细尺度 (B, L)

        for row, (i, start, end) in enumerate(chunk):
            sample_logits[i] += logits_sample[row].float().cpu()
            if not has_token or not is_hybrid:
                continue
            n = end - start
            offset = probs.shape[1] - n                                    # 报告前缀占位
            p = probs[row, offset:].tolist()
            tok = dataset.tok[i][start:end]
            lines = dataset.line_of_token[i][start:end]
            if line_sum[i] is None:
                line_sum[i], line_cnt[i] = {}, {}
            for prob, tok_label, line in zip(p, tok, lines):
                if tok_label == IGNORE:
                    continue
                line_sum[i][line] = line_sum[i].get(line, 0.0) + prob
                line_cnt[i][line] = line_cnt[i].get(line, 0) + 1
                hit = 0 if prob > 0.5 else 3                               # 0=tp 3=tn，需要再看标签
                if prob > 0.5 and tok_label == 0:
                    hit = 1
                elif prob <= 0.5 and tok_label == 1:
                    hit = 2
                token_hits[i][hit] += 1

    metrics = {"n": len(dataset)}
    gold = [dataset.labels[i] for i in range(len(dataset))]
    pred = sample_logits.argmax(-1).tolist()
    metrics["sample_acc"] = sum(1 for p, g in zip(pred, gold) if p == g) / max(len(gold), 1)
    metrics["sample_f1"] = prf1(pred, gold)["f1"]

    if is_hybrid and has_token:
        pred_lines, gold_lines = [], []
        chunk_tp = chunk_pred = chunk_gold = 0
        for i in range(len(dataset)):
            line_label = dataset.line_label[i]
            picked = [int(line_sum[i].get(k, 0.0) / max(line_cnt[i].get(k, 0), 1) > 0.5)
                      for k in range(len(line_label))]
            pred_lines.extend(picked)
            gold_lines.extend(line_label)
            tp_i, pred_i, gold_i = chunk_match(picked, line_label)      # 逐文件统计片段级
            chunk_tp += tp_i
            chunk_pred += pred_i
            chunk_gold += gold_i
        metrics.update({f"line_{k}": v for k, v in prf1(pred_lines, gold_lines).items()})
        precision = chunk_tp / max(chunk_pred, 1e-9)
        recall = chunk_tp / max(chunk_gold, 1e-9)
        metrics["chunk_p"] = precision
        metrics["chunk_r"] = recall
        metrics["chunk_f1"] = 2 * precision * recall / max(precision + recall, 1e-9)
        tp, fp, fn = (sum(h[j] for h in token_hits) for j in range(3))
        precision = tp / max(tp + fp, 1e-9)
        recall = tp / max(tp + fn, 1e-9)
        metrics["token_f1"] = 2 * precision * recall / max(precision + recall, 1e-9)
    elif is_hybrid:
        print("[eval] 提示：当前模型无 token 头，hybrid 只能出样本级指标")
    return metrics


# --------------------------------------------------------------------------- #
# 训练
# --------------------------------------------------------------------------- #
def make_dataset(cfg: dict, name: str, split: str, tokenizer, report, train: bool, limit: int | None):
    data_cfg = cfg["data"]
    return truncate(
        build_dataset(
            name,
            file=str(resolve(data_cfg["processed_dir"]) / data_cfg[f"{name}_file"]),
            split=split,
            report=report,
            max_length=cfg["train"]["max_length"][name],
            train=train,
            seed=cfg["train"].get("seed", 0),
        ),
        limit,
    )


def train(cfg: dict, args) -> None:
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    torch.manual_seed(cfg["train"].get("seed", 0))
    tokenizer = AutoTokenizer.from_pretrained(str(resolve(cfg["encoder"]["path"])))
    report = build_report(cfg["report"]["name"], tokenizer=tokenizer,
                          max_tokens=cfg["report"].get("max_tokens", 64))
    model, encoder = build_model(cfg, tokenizer)
    model.to(device)

    streams = cfg["train"]["streams"]
    loaders = {}
    for name in streams:
        dataset = make_dataset(cfg, name, "train", tokenizer, report, True, args.limit)
        print(f"[data] {name} train: {split_counts(dataset)}")
        loaders[name] = DataLoader(
            dataset,
            batch_size=cfg["train"]["batch_size"][name],
            shuffle=True, drop_last=True, collate_fn=collate,
            num_workers=cfg["train"].get("num_workers", 0),
        )

    params = [p for p in model.parameters() if p.requires_grad]
    n_param = sum(p.numel() for p in params)
    print(f"[model] 可训练参数 {n_param / 1e6:.2f}M（编码器 {encoder.__class__.__name__}"
          f"{'冻结' if not any(p.requires_grad for p in encoder.parameters()) else '微调'}"
          f"，含报告：{report.name}）")

    # 参数分组：编码器（若可训练）单独给学习率 train.lr_encoder（默认同全局）
    lr = cfg["train"]["lr"]
    lr_encoder = cfg["train"].get("lr_encoder", lr)
    enc_params = [p for p in encoder.parameters() if p.requires_grad]
    other_params = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("encoder.")]
    groups = [{"params": other_params, "lr": lr}]
    if enc_params:
        groups.append({"params": enc_params, "lr": lr_encoder})
    opt = torch.optim.AdamW(groups, lr=lr, weight_decay=cfg["train"].get("weight_decay", 0.01))
    if any(len(loader) == 0 for loader in loaders.values()):
        raise SystemExit("某个数据流不足一个 batch（减小 train.batch_size 或增大子集）")
    steps_per_epoch = max(len(loader) for loader in loaders.values())
    total_steps = steps_per_epoch * cfg["train"]["epochs"]
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[g["lr"] for g in groups], total_steps=max(total_steps, 1),
        pct_start=cfg["train"].get("warmup", 0.05),
    )

    run_dir = resolve(cfg["train"].get("out_dir", "runs")) / args.tag
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    log_path = run_dir / "metrics.csv"
    log_file = open(log_path, "w", newline="", encoding="utf-8")
    logger = csv.writer(log_file)
    logger.writerow(["epoch", "step", "loss", "loss_sample", "loss_token", "lr", "sec"])

    amp = cfg["train"].get("amp", True) and device == "cuda"
    monitor = cfg["train"].get("monitor", "mean")
    best = -1.0
    print(f"[train] streams={streams} steps/epoch={steps_per_epoch} epochs={cfg['train']['epochs']} amp={amp}")

    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        if not any(p.requires_grad for p in model.encoder.parameters()):
            model.encoder.eval()                              # 冻结编码器保持 eval（关掉其 dropout）
        for name, dataset in zip(streams, [loaders[s].dataset for s in streams]):
            if hasattr(dataset, "epoch"):
                dataset.epoch = epoch                 # 窗口裁剪在该 epoch 内可复现
        iters = {name: cycle(loader) for name, loader in loaders.items()}
        running = {"loss": 0.0, "sample": 0.0, "token": 0.0}
        start_time = time.time()
        for step in range(steps_per_epoch):
            opt.zero_grad(set_to_none=True)
            step_parts = {"sample": 0.0, "token": 0.0}
            step_loss = 0.0
            for name in streams:                                  # 每个 step：两流各一个 batch，同时参与
                batch = next(iters[name])
                batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                    sample_logits, token_logits = model(batch["input_ids"], batch["attention_mask"])
                    loss, parts = batch_losses(cfg, sample_logits, token_logits, batch, device)
                loss.backward()
                step_loss += float(loss.detach())
                for key in step_parts:
                    step_parts[key] += parts[key]
            torch.nn.utils.clip_grad_norm_(params, cfg["train"].get("grad_clip", 1.0))
            opt.step()
            scheduler.step()

            for key, value in step_parts.items():
                running[key] += value
            running["loss"] += step_loss
            if (step + 1) % cfg["train"].get("log_every", 20) == 0 or step == steps_per_epoch - 1:
                elapsed = time.time() - start_time
                n = step + 1
                lr = scheduler.get_last_lr()[0]
                print(f"[epoch {epoch} {n}/{steps_per_epoch}] loss={running['loss'] / n:.4f} "
                      f"sample={running['sample'] / n:.4f} token={running['token'] / n:.4f} "
                      f"lr={lr:.2e} {elapsed / n:.2f}s/it")
                logger.writerow([epoch, n, f"{running['loss'] / n:.6f}", f"{running['sample'] / n:.6f}",
                                 f"{running['token'] / n:.6f}", f"{lr:.3e}", f"{elapsed:.1f}"])
                log_file.flush()
        log_file.flush()

        # ---- 验证（各流 val 集） ----
        results = {}
        for name in streams:
            dataset = make_dataset(cfg, name, "val", tokenizer, report, False, args.limit)
            if len(dataset) == 0:
                continue
            metrics = evaluate(model, dataset, name, cfg, device, report)
            results[name] = metrics
            print(f"[val {epoch}] {name}: " + json.dumps({k: round(v, 4) for k, v in metrics.items()}))
        score = _monitor_score(results, monitor)
        train_avg = {k: v / max(steps_per_epoch, 1) for k, v in running.items()}
        with open(run_dir / "metrics.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"epoch": epoch, "train": train_avg, "val": results}) + "\n")
        if score > best:
            best = score
            # 除冻结编码器外的完整 state_dict（含 BatchNorm running 统计量）；微调编码器时会一并保存
            keep_encoder = any(p.requires_grad for p in model.encoder.parameters())
            state = {k: v for k, v in model.state_dict().items()
                     if keep_encoder or not k.startswith("encoder.")}
            torch.save({"cfg": cfg, "epoch": epoch, "score": score, "metrics": results, "state": state},
                       run_dir / "best.pt")
            print(f"[ckpt] 保存 best.pt（{monitor}={score:.4f}）")

    log_file.close()
    print(f"[train] 完成，产物目录：{run_dir}")


def _monitor_score(results: dict, monitor: str) -> float:
    if monitor == "m4_f1":
        return results.get("m4", {}).get("sample_f1", 0.0)
    if monitor == "line_f1":
        return results.get("hybrid", {}).get("line_f1", 0.0)
    values = [v.get("sample_f1", 0.0) for v in results.values()]
    values += [v.get("line_f1", 0.0) for v in results.values() if "line_f1" in v]
    return sum(values) / max(len(values), 1)


# --------------------------------------------------------------------------- #
# 评测入口
# --------------------------------------------------------------------------- #
def run_eval(cfg: dict, args) -> None:
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(str(resolve(cfg["encoder"]["path"])))
    report = build_report(cfg["report"]["name"], tokenizer=tokenizer,
                          max_tokens=cfg["report"].get("max_tokens", 64))
    model, _ = build_model(cfg, tokenizer)
    ckpt = torch.load(resolve(args.ckpt), map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(ckpt["state"], strict=False)
    if missing or unexpected:
        print(f"[eval] 注意：未加载 {len(missing)} 项（如 {missing[:3]}）、多余 {len(unexpected)} 项")
    model.to(device)
    print(f"[eval] 载入 {args.ckpt}（epoch {ckpt.get('epoch')}），配置来自当前 yaml/命令行")
    result = {}
    for name in cfg["train"]["streams"]:
        for split in ("val", "test"):
            dataset = make_dataset(cfg, name, split, tokenizer, report, False, args.limit)
            if len(dataset) == 0:
                continue
            metrics = evaluate(model, dataset, name, cfg, device, report)
            result[f"{name}/{split}"] = metrics
            print(f"[eval] {name}/{split}: " + json.dumps({k: round(v, 4) for k, v in metrics.items()}))
    out = Path(resolve(args.ckpt)).parent / "eval.json"       # 落盘，便于 scripts/compare.py 汇总
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"ckpt": str(args.ckpt), "epoch": ckpt.get("epoch"), "metrics": result},
                  f, ensure_ascii=False, indent=2)
    print(f"[eval] 结果已写入 {out}")


def main() -> int:
    parser = argparse.ArgumentParser(description="U-Det 训练/评测")
    parser.add_argument("--config", default="configs/udet_base.yaml")
    parser.add_argument("--tag", default="udet-base")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None, help="每个数据集只用前 N 条（冒烟）")
    parser.add_argument("--no-m4", action="store_true")
    parser.add_argument("--no-hybrid", action="store_true")
    parser.add_argument("--report", default=None, help="覆盖 report.name（none/handcrafted）")
    parser.add_argument("--mid", default=None, help="覆盖 model.mid.name（none/transformer）")
    parser.add_argument("--model", default=None, help="覆盖 model.name（unet1d / codet5cls 基线）")
    parser.add_argument("--freeze-encoder", action="store_true", help="冻结编码器（基线默认冻结）")
    parser.add_argument("--unfreeze-encoder", action="store_true", help="解冻编码器（微调基线）")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    cfg = apply_overrides(load_config(args.config), args)
    if args.eval:
        if not args.ckpt:
            raise SystemExit("--eval 需要 --ckpt 指定权重")
        run_eval(cfg, args)
    else:
        train(cfg, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
