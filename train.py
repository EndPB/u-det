#!/usr/bin/env python
"""U-Det 训练 / 评测主干（根目录单文件）。

模型（v0.2）：逐 token 编码器（查表，可冻结/可训练）
            -> 频域 U-Net（FFT 上下采样 + 8 层瓶颈）
            -> 样本级头（瓶颈池化）+ 多尺度 token 级头（5 个尺度：L/16 … L）。
数据：双流并行（m4 样本级 / hybrid token 级），**不截断代码**；训练与评测一律 batch=1，
      用梯度累积凑有效批大小（FFT 是全局算子，批内 padding 会污染频谱语义）。
损失：样本级 CE + 各尺度 token 级 BCE（标签按尺度池化，权重由粗到细）。

用法::

    python train.py --config configs/udet_base.yaml --tag v02
    python train.py --config configs/udet_base.yaml --eval --ckpt runs/v02/best.pt
    python train.py --limit 100 --epochs 1 --tag smoke
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from dataio import build_dataset, collate
from encoders import build_encoder
from models import PooledClassifier, SampleHead, TokenHeads, build_hier
from report import build_report

ROOT = Path(__file__).resolve().parent
IGNORE = -100


# --------------------------------------------------------------------------- #
# 基础工具
# --------------------------------------------------------------------------- #
def resolve(path: str) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def setup_cuda() -> None:
    """cuFFT plan 缓存上限（频域核内部已按 2 的幂对齐，plan 种类很少，256 足够）。"""
    if torch.cuda.is_available():
        torch.backends.cuda.cufft_plan_cache.max_size = 256


def load_config(path: str) -> dict:
    with open(resolve(path), encoding="utf-8") as f:
        return yaml.safe_load(f)


def apply_overrides(cfg: dict, args) -> dict:
    """命令行覆盖配置（冒烟 / 消融用）。"""
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.lr is not None:
        cfg["train"]["lr"] = args.lr
    if args.grad_accum is not None:
        cfg["train"]["grad_accum"] = args.grad_accum
    if args.no_m4:
        cfg["train"]["streams"] = [s for s in cfg["train"]["streams"] if s != "m4"]
    if args.no_hybrid:
        cfg["train"]["streams"] = [s for s in cfg["train"]["streams"] if s != "hybrid"]
    if args.report is not None:
        cfg["report"]["name"] = args.report
    if args.model is not None:
        cfg["model"]["name"] = args.model
    if args.freeze_encoder:
        cfg["encoder"]["freeze"] = True
    if args.unfreeze_encoder:
        cfg["encoder"]["freeze"] = False
    return cfg


def limit_dataset(dataset, limit: int | None):
    """冒烟用：等距抽取 limit 条（跨类别/语言均匀）。"""
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
    lengths = sorted(len(x) for x in dataset.ids)
    info["tokens"] = {"中位": lengths[len(lengths) // 2], "max": lengths[-1]} if lengths else {}
    return {"n": len(dataset), "labels": dict(Counter(dataset.labels)), **info}


def cycle(loader):
    """无限循环取数（两流长度不一致时循环较短的流）。"""
    while True:
        yield from loader


# --------------------------------------------------------------------------- #
# 模型组装
# --------------------------------------------------------------------------- #
BASELINE_MODELS = ("codet5cls", "pooled")


class UDet(nn.Module):
    """逐 token 编码器 -> 频域 U-Net -> 样本级头 + 多尺度 token 级头（整段序列，batch=1）。"""

    def __init__(self, encoder: nn.Module, backbone: nn.Module, sample_head: nn.Module,
                 token_heads: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.backbone = backbone
        self.sample_head = sample_head
        self.token_heads = token_heads

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None):
        feats = self.encoder(input_ids)                          # (B, L, D)，整段不截断
        levels, _ = self.backbone(feats, return_features=True)   # 由粗到细（L/16 … L）
        sample_logits = self.sample_head(levels[0])              # 瓶颈池化 -> (B, 2)
        token_logits = self.token_heads(levels)                  # 每个尺度 (B, 1, L_k)
        return sample_logits, token_logits


def build_model(cfg: dict, tokenizer) -> tuple[nn.Module, nn.Module]:
    """按 cfg 组装模型：默认频域 U-Net；model.name 为基线名时构建直接二分类基线。"""
    encoder = build_encoder(**cfg["encoder"])
    freeze = bool(cfg["encoder"].get("freeze", True))
    if freeze:
        encoder.eval()
        encoder.requires_grad_(False)

    mcfg = dict(cfg["model"])
    name = mcfg.pop("name", "hier")
    if name in BASELINE_MODELS:                                  # 基线：编码器 -> 池化 -> 线性分类
        bcfg = dict(mcfg.get("baseline", {}))
        return PooledClassifier(encoder, dim=encoder.hidden_size, **bcfg), encoder

    mcfg.pop("baseline", None)
    backbone = build_hier(name, **mcfg)
    heads = dict(cfg.get("heads", {}))
    sample_head = SampleHead(backbone.dim, hidden=heads.get("sample_hidden"),
                             dropout=heads.get("sample_dropout", 0.0))
    token_heads = TokenHeads([backbone.dim] * (backbone.depth + 1), out=1)
    return UDet(encoder, backbone, sample_head, token_heads), encoder


# --------------------------------------------------------------------------- #
# 损失
# --------------------------------------------------------------------------- #
def token_loss(logit_scales: list[torch.Tensor], tok_labels: torch.Tensor,
               weights: list[float], ignore: int = IGNORE) -> torch.Tensor | None:
    """各尺度 token 级 BCE：标签按尺度平均池化成软标签（不同文本长度精度）。"""
    valid = (tok_labels != ignore)
    if not valid.any():
        return None
    target = tok_labels.clamp(min=0).float()
    length = tok_labels.shape[-1]
    total, total_w = 0.0, 0.0
    for logits, weight in zip(logit_scales, weights):
        want = logits.shape[-1]                                  # 该尺度真实长度（可能 ceil 过）
        k = max(1, math.ceil(length / want))
        pad = max(0, want * k - length)                          # 补齐到整数倍，池化后恰好 want 个 bin
        t = F.pad(target, (0, pad)) if pad else target
        v = F.pad(valid.float(), (0, pad)) if pad else valid.float()
        num = F.avg_pool1d((t * v).unsqueeze(1), k, k)
        den = F.avg_pool1d(v.unsqueeze(1), k, k)
        keep = (den > 0).float()
        if keep.sum() == 0:
            continue
        soft = num / den.clamp(min=1e-6)
        loss = F.binary_cross_entropy_with_logits(logits, soft, weight=keep, reduction="sum") / keep.sum()
        total = total + weight * loss
        total_w += weight
    return None if total_w == 0 else total / total_w


def batch_losses(cfg: dict, sample_logits: torch.Tensor, token_logits: list[torch.Tensor] | None,
                 batch: dict, device: str, use_sample: bool = True) -> tuple[torch.Tensor, dict]:
    """一个样本的总损失与分项（样本级 + token 级；基线模型无 token 头时只算样本级）。

    Args:
        use_sample: 是否计入样本级 CE。hybrid 流的样本标签恒为 1（只保留含 AI 行的文件），
            若计入会把样本头拉向“永远输出正类”的退化解，故只对 m4（human/AI 均衡）生效。
    """
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
    total = loss_cfg.get("sample", 1.0) * l_sample if use_sample else torch.zeros((), device=device)
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
def evaluate(model: nn.Module, dataset, name: str, cfg: dict, device: str) -> dict:
    """整段序列逐样本评测（不截断、不滑窗）：m4 出样本级指标；hybrid 出行级/token/片段级指标。"""
    model.eval()
    amp = cfg["train"].get("amp", True) and device == "cuda"
    is_hybrid = name == "hybrid"

    sample_logits = torch.zeros(len(dataset), 2)
    token_hits = [[0, 0, 0, 0] for _ in range(len(dataset))]       # tp/fp/fn/tn
    line_sum = [None] * len(dataset)
    line_cnt = [None] * len(dataset)
    has_token = True

    for i in tqdm(range(len(dataset)), desc=f"eval {name}", leave=False):
        item = dataset[i]                                          # 已含报告前缀 / token 标签
        ids = torch.tensor([item["input_ids"]], dtype=torch.long, device=device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            logits_sample, token_logits = model(ids, torch.ones_like(ids))
        has_token = token_logits is not None
        sample_logits[i] += logits_sample[0].float().cpu()
        if not has_token or not is_hybrid:
            continue
        probs = torch.sigmoid(token_logits[-1].float())[0, 0].cpu().tolist()   # 最细尺度 = L
        if line_sum[i] is None:
            line_sum[i], line_cnt[i] = {}, {}
        for prob, tok_label, line in zip(probs, item["tok_labels"], item["line_of_token"]):
            if tok_label == IGNORE:
                continue
            line_sum[i][line] = line_sum[i].get(line, 0.0) + prob
            line_cnt[i][line] = line_cnt[i].get(line, 0) + 1
            hit = 0 if prob > 0.5 else 3
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
            tp_i, pred_i, gold_i = chunk_match(picked, line_label)
            chunk_tp += tp_i
            chunk_pred += pred_i
            chunk_gold += gold_i
        metrics.update({f"line_{k}": v for k, v in prf1(pred_lines, gold_lines).items()})
        precision = chunk_tp / max(chunk_pred, 1e-9)
        recall = chunk_tp / max(chunk_gold, 1e-9)
        metrics["chunk_p"], metrics["chunk_r"] = precision, recall
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
def make_dataset(cfg: dict, name: str, split: str, report, train: bool, limit: int | None):
    data_cfg = cfg["data"]
    return limit_dataset(
        build_dataset(
            name,
            file=str(resolve(data_cfg["processed_dir"]) / data_cfg[f"{name}_file"]),
            split=split,
            report=report,
            train=train,
        ),
        limit,
    )


def train(cfg: dict, args) -> None:
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    setup_cuda()
    torch.manual_seed(cfg["train"].get("seed", 0))
    tokenizer = AutoTokenizer.from_pretrained(str(resolve(cfg["encoder"]["path"])))
    report = build_report(cfg["report"]["name"], tokenizer=tokenizer,
                          max_tokens=cfg["report"].get("max_tokens", 64))
    model, encoder = build_model(cfg, tokenizer)
    model.to(device)

    streams = cfg["train"]["streams"]
    loaders = {}
    for name in streams:                                   # 默认 batch=1：整段序列，不做 padding
        dataset = make_dataset(cfg, name, "train", report, True, args.limit)
        bs = int(cfg["train"].get("batch_size", {}).get(name, 1)) if isinstance(cfg["train"].get("batch_size"), dict) else 1
        print(f"[data] {name} train: {split_counts(dataset)}  batch={bs}")
        loaders[name] = DataLoader(
            dataset, batch_size=bs, shuffle=True, collate_fn=collate,
            num_workers=cfg["train"].get("num_workers", 0),
        )

    params = [p for p in model.parameters() if p.requires_grad]
    n_param = sum(p.numel() for p in params)
    frozen = not any(p.requires_grad for p in encoder.parameters())
    print(f"[model] 可训练参数 {n_param / 1e6:.2f}M（编码器 {encoder.__class__.__name__}"
          f"{'冻结' if frozen else '微调'}，含报告：{report.name}）")

    lr = cfg["train"]["lr"]
    lr_encoder = cfg["train"].get("lr_encoder", lr)
    enc_params = [p for p in encoder.parameters() if p.requires_grad]
    other_params = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("encoder.")]
    groups = [{"params": other_params, "lr": lr}]
    if enc_params:
        groups.append({"params": enc_params, "lr": lr_encoder})
    opt = torch.optim.AdamW(groups, lr=lr, weight_decay=cfg["train"].get("weight_decay", 0.01))

    if any(len(loader) == 0 for loader in loaders.values()):
        raise SystemExit("某个数据流为空（检查 --limit / 子集）")
    steps_per_epoch = max(len(loader) for loader in loaders.values())
    accum = max(1, int(cfg["train"].get("grad_accum", 1)))
    sample_streams = set(cfg["train"].get("sample_streams", streams))     # 参与样本级损失的流
    print(f"[loss] 样本级损失仅统计：{sorted(sample_streams)}（其余流的样本标签退化）")
    updates_per_epoch = max(1, steps_per_epoch // accum)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[g["lr"] for g in groups], total_steps=updates_per_epoch * cfg["train"]["epochs"],
        pct_start=cfg["train"].get("warmup", 0.05),
    )

    run_dir = resolve(cfg["train"].get("out_dir", "runs")) / args.tag
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    log_file = open(run_dir / "metrics.csv", "w", newline="", encoding="utf-8")
    logger = csv.writer(log_file)
    logger.writerow(["epoch", "step", "loss", "loss_sample", "loss_token", "lr", "sec"])

    amp = cfg["train"].get("amp", True) and device == "cuda"
    monitor = cfg["train"].get("monitor", "mean")
    tokens_seen = 0
    best = -1.0
    print(f"[train] streams={streams} steps/epoch={steps_per_epoch} accum={accum} "
          f"updates/epoch={updates_per_epoch} epochs={cfg['train']['epochs']} amp={amp}")

    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        if frozen:
            model.encoder.eval()                           # 冻结编码器保持 eval（关掉其 dropout）
        iters = {name: cycle(loader) for name, loader in loaders.items()}
        running = {"loss": 0.0, "sample": 0.0, "token": 0.0}
        start_time = time.time()
        opt.zero_grad(set_to_none=True)
        bar = tqdm(range(steps_per_epoch), desc=f"epoch {epoch}", unit="it")
        for step in bar:
            step_parts = {"sample": 0.0, "token": 0.0}
            for name in streams:                           # 两流各取一个样本，同时参与
                batch = next(iters[name])
                batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                    sample_logits, token_logits = model(batch["input_ids"], batch.get("attention_mask"))
                    loss, parts = batch_losses(cfg, sample_logits, token_logits, batch, device,
                                               use_sample=name in sample_streams)
                (loss / (accum * len(streams))).backward()          # 归一到全部流的样本均值
                step_parts["token"] += parts["token"]
                if name in sample_streams:                         # 只记录真正回传的样本级损失
                    step_parts["sample"] += parts["sample"]
                tokens_seen += int(batch["input_ids"].numel())

            if (step + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(params, cfg["train"].get("grad_clip", 1.0))
                opt.step()
                scheduler.step()
                opt.zero_grad(set_to_none=True)

            for key, value in step_parts.items():
                running[key] += value
            running["loss"] += step_parts["sample"] + step_parts["token"]
            if (step + 1) % cfg["train"].get("log_every", 20) == 0 or step == steps_per_epoch - 1:
                elapsed = time.time() - start_time
                done = step + 1
                bar.set_postfix(loss=f"{running['loss'] / done:.3f}",
                                sample=f"{running['sample'] / done:.3f}",
                                token=f"{running['token'] / done:.3f}",
                                lr=f"{scheduler.get_last_lr()[0]:.2e}",
                                tok_s=f"{tokens_seen / max(elapsed, 1e-6):.0f}")
                logger.writerow([epoch, done, f"{running['loss'] / done:.6f}",
                                 f"{running['sample'] / done:.6f}",
                                 f"{running['token'] / done:.6f}",
                                 f"{scheduler.get_last_lr()[0]:.3e}", f"{elapsed:.1f}"])
                log_file.flush()
        bar.close()

        results = {}
        for name in streams:
            dataset = make_dataset(cfg, name, "val", report, False, args.limit)
            if len(dataset) == 0:
                continue
            metrics = evaluate(model, dataset, name, cfg, device)
            results[name] = metrics
            print(f"[val {epoch}] {name}: " + json.dumps({k: round(v, 4) for k, v in metrics.items()}))
        score = _monitor_score(results, monitor)
        train_avg = {k: v / max(steps_per_epoch, 1) for k, v in running.items()}
        with open(run_dir / "metrics.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"epoch": epoch, "train": train_avg, "val": results}) + "\n")
        if score > best:
            best = score
            # 只存可训练的编码器参数（冻结底座/LUT buffer 不入库，checkpoint 保持小体积）
            trainable = {name for name, p in model.named_parameters() if p.requires_grad}
            state = {k: v for k, v in model.state_dict().items()
                     if not k.startswith("encoder.") or k in trainable}
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
    setup_cuda()
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
            dataset = make_dataset(cfg, name, split, report, False, args.limit)
            if len(dataset) == 0:
                continue
            metrics = evaluate(model, dataset, name, cfg, device)
            result[f"{name}/{split}"] = metrics
            print(f"[eval] {name}/{split}: " + json.dumps({k: round(v, 4) for k, v in metrics.items()}))
    out = Path(resolve(args.ckpt)).parent / "eval.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"ckpt": str(args.ckpt), "epoch": ckpt.get("epoch"), "metrics": result},
                  f, ensure_ascii=False, indent=2)
    print(f"[eval] 结果已写入 {out}")


def main() -> int:
    parser = argparse.ArgumentParser(description="U-Det 训练/评测（v0.2 频域 U-Net）")
    parser.add_argument("--config", default="configs/udet_base.yaml")
    parser.add_argument("--tag", default="udet-base")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None, help="每个数据集只用前 N 条（冒烟）")
    parser.add_argument("--no-m4", action="store_true")
    parser.add_argument("--no-hybrid", action="store_true")
    parser.add_argument("--report", default=None, help="覆盖 report.name（none/handcrafted）")
    parser.add_argument("--model", default=None, help="覆盖 model.name（hier / codet5cls 基线）")
    parser.add_argument("--freeze-encoder", action="store_true", help="冻结编码器")
    parser.add_argument("--unfreeze-encoder", action="store_true", help="解冻编码器（微调）")
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
