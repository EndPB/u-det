#!/usr/bin/env python
"""d-det 训练 / 评测主入口（根目录单文件）。

模型：编码器（codet5blk：分块 + LoRA）→ 池化 → 双分数读出（s1 / s2，零初始化扩张，
      见 models/scores.py 的三条"工程落点"注记）。
数据：双流并行，恒 batch=1（编码器分块处理，长度无上限）：

    m4    人类 / AI 样本级标签        → s1 的 BCE（人机分离，λa）
    pair  同 prompt 的 base/instruct  → s2 的 hinge margin（零标注偏好位移，λb）

总损失（docx/d-det.md §5）：

    L = λa · BCE(σ(s1), y)  +  λb · max(0, m − s2(x₊) + s2(x₋))  +  λc · cos²(w1, w2)

    （λd"旧维语义保持"在从零训练的 v0.1 **不启用**：没有旧网络可锚定；
      config 键 loss.anchor 保留，配成非 0 会直接报错而非静默忽略。）

用法::

    python train.py --config configs/ddet_base.yaml --tag v0.1.0
    python train.py --config configs/ddet_base.yaml --eval --ckpt runs/v0.1.0/best.pt --dump-scores
    python train.py --config configs/ddet_base.yaml --tag smoke --limit 50 --epochs 1

工程铁律（继承自 u-det，见 ../docx/lessons.md）：
    * checkpoint 存 ``model.state_dict()`` 的可训练子集（不是 named_parameters）；
    * ``--eval`` 比对训练产物目录里的 config.yaml（数学相关段），不一致**直接中止**
      （历史上用错 config 评测曾静默产出垃圾结果）；
    * 带 ``--limit`` 的冒烟评测写 ``eval_limit<N>.json``，**绝不覆盖**正式 eval.json；
    * 架构 / 数据对比必须对齐 epoch 预算（2 轮的数字只能用来排优先级）。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataio import COLLATES, build_dataset
from encoders import build_encoder
from models import build_model

ROOT = Path(__file__).resolve().parent


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
    """命令行覆盖配置（冒烟 / 消融用）。

    一律用 getattr 读取（而不是直接属性访问）：这样只定义了部分参数的调用方
    （如 scripts/selfcheck.py / probe_sensitivity.py 的 argparse）也能安全复用本函数。
    """
    if getattr(args, "epochs", None) is not None:
        cfg["train"]["epochs"] = args.epochs
    if getattr(args, "lr", None) is not None:
        cfg["train"]["lr"] = args.lr
    if getattr(args, "grad_accum", None) is not None:
        cfg["train"]["grad_accum"] = args.grad_accum
    if getattr(args, "margin", None) is not None:
        cfg.setdefault("loss", {})["margin"] = args.margin
    if getattr(args, "encoder", None) is not None:
        cfg["encoder"]["name"] = args.encoder
    if getattr(args, "model", None) is not None:
        cfg["model"]["name"] = args.model
    if getattr(args, "freeze_encoder", False):
        cfg["encoder"]["freeze"] = True
    if getattr(args, "unfreeze_encoder", False):
        cfg["encoder"]["freeze"] = False
    if getattr(args, "no_m4", False):
        cfg["train"]["streams"] = [s for s in cfg["train"]["streams"] if s != "m4"]
    if getattr(args, "no_pair", False):
        cfg["train"]["streams"] = [s for s in cfg["train"]["streams"] if s != "pair"]
    if not cfg["train"]["streams"]:
        raise SystemExit("[cfg] 所有数据流都被去掉了（--no-m4 / --no-pair）")
    if getattr(args, "set", None):                     # 通用覆盖（单变量实验用）
        for item in args.set:
            key, _, raw = str(item).partition("=")
            if not _ or not key:
                raise SystemExit(f"--set 的格式应为 段.键=值，收到 {item!r}")
            node = cfg
            for part in key.split(".")[:-1]:
                node = node.setdefault(part, {})
            node[key.split(".")[-1]] = yaml.safe_load(raw)   # YAML 解析 ⇒ 布尔/数字/列表都能写
            print(f"[cfg] --set {key} = {node[key.split('.')[-1]]!r}")
    return cfg


def check_anchor(cfg: dict) -> None:
    """loss.anchor（λd 旧维语义保持）在 v0.1 未实现 —— 宁可报错，不可静默忽略。"""
    anchor = float(cfg.get("loss", {}).get("anchor", 0.0) or 0.0)
    if anchor != 0.0:
        raise SystemExit(
            "[loss] loss.anchor != 0，但该项（旧维语义保持）在从零训练的 v0.1 未实现"
            "（没有旧网络可锚定）。请保持 loss.anchor=0；若需要请先实现再启用。")


def limit_dataset(dataset, limit: int | None):
    """冒烟用：等距抽取 limit 条（跨类别 / 语言均匀）。"""
    if not limit or len(dataset) <= limit:
        return dataset
    total = len(dataset)
    index = [i * total // limit for i in range(limit)]
    for attr in ("ids", "codes", "labels", "meta",
                 "ids_plus", "ids_minus", "codes_plus", "codes_minus",
                 "ids_plus_a", "ids_minus_a", "ids_plus_b", "ids_minus_b",
                 "codes_plus_a", "codes_minus_a", "codes_plus_b", "codes_minus_b",
                 "tok", "line_of_token", "line_label"):
        seq = getattr(dataset, attr, None)
        if isinstance(seq, list) and len(seq) == total:
            setattr(dataset, attr, [seq[i] for i in index])
    return dataset


def split_counts(dataset) -> dict:
    """数据集概览（打印用）：样本数 / 类别 / 语言 / 长度。"""
    from collections import Counter
    info = {"n": len(dataset)}
    keys = [k for k in ("language", "model", "family")
            if dataset.meta and any(k in m for m in dataset.meta[:50])]
    for key in keys:
        info[key] = dict(Counter(str(m.get(key)) for m in dataset.meta))
    ids = getattr(dataset, "ids", None)
    if ids:
        lengths = sorted(len(x) for x in ids)
        info["tokens"] = {"中位": lengths[len(lengths) // 2], "max": lengths[-1]}
    return info


def cycle(loader):
    """无限循环取数（两流长度不一致时循环较短的流）。"""
    while True:
        yield from loader


def readout_cos(model) -> float:
    """w1 与 w2（各列）方向余弦绝对值的均值：1=退化到同一方向，0=正交。"""
    with torch.no_grad():
        w1, w2 = model.readout_vectors()
        w1 = w1.double()
        w2 = w2.double()
        num = (w2.T @ w1).abs()                       # (r,)
        den = w2.norm(dim=0) * w1.norm() + 1e-12      # (r,)
        return float((num / den).mean())


def _diag_batches(cfg: dict, args, device: str, n: int = 16) -> dict:
    """固定小子集（m4 / pair 各 n 条）——梯度快照诊断用（不参与训练更新）。"""
    out = {}
    for name in ("m4", "pair"):
        dataset = make_dataset(cfg, name, "train", None)
        k = min(n, len(dataset))
        index = list(range(0, len(dataset), max(1, len(dataset) // k)))[:k]
        if name == "m4":
            out["m4"] = (
                [torch.tensor(dataset[i]["input_ids"], dtype=torch.long, device=device)
                 for i in index],
                [int(dataset.labels[i]) for i in index],
            )
        else:
            out["pair"] = (
                [torch.tensor(dataset[i]["input_ids_plus"], dtype=torch.long, device=device)
                 for i in index],
                [torch.tensor(dataset[i]["input_ids_minus"], dtype=torch.long, device=device)
                 for i in index],
            )
    return out


def _grad_conflict(model, batches: dict, enc_params, device: str, amp: bool,
                   amp_dtype: str, lc: dict) -> dict:
    """固定 batch 上分别求 L1 / L2 的编码器梯度（逐条累积；峰值 = 单样本图）。

    逐条 `autograd.grad`：不触碰 `.grad`（训练中途安全），长样本也不会 OOM
    （容器 cgroup 限额 30GB；整批 pad 一次回传在长样本上实测被 kill）。
    """
    was_training = model.training
    model.eval()

    def _acc(loss, acc):
        grads = torch.autograd.grad(loss, enc_params, allow_unused=True)
        grads = [torch.zeros_like(p) if g is None else g for g, p in zip(grads, enc_params)]
        return grads if acc is None else [a + b for a, b in zip(acc, grads)]

    ids_list, labels = batches["m4"]
    g1, l1_sum = None, 0.0
    for ids, y in zip(ids_list, labels):
        with torch.autocast(amp_dtype, dtype=torch.bfloat16, enabled=amp):
            l1i = F.binary_cross_entropy_with_logits(
                model(ids, torch.ones_like(ids))[0].float(),
                torch.tensor([y], dtype=torch.float32, device=device))
        g1 = _acc(l1i, g1)
        l1_sum += float(l1i.detach())
    l1 = l1_sum / max(len(ids_list), 1) * float(lc.get("s1", 1.0))

    plus_list, minus_list = batches["pair"]
    margin = float(lc.get("margin", 1.0))
    g2, l2_sum = None, 0.0
    for ip, im in zip(plus_list, minus_list):
        with torch.autocast(amp_dtype, dtype=torch.bfloat16, enabled=amp):
            s2p = model(ip, torch.ones_like(ip))[1]
            s2m = model(im, torch.ones_like(im))[1]
            d = (s2p - s2m).reshape(-1)
            gap = d.norm() if d.numel() > 1 else d[0]
            l2i = F.relu(margin - gap)
        g2 = _acc(l2i, g2)
        l2_sum += float(l2i.detach())
    l2 = l2_sum / max(len(plus_list), 1) * float(lc.get("s2", 1.0))

    v1 = torch.cat([g.reshape(-1).float() for g in g1])
    v2 = torch.cat([g.reshape(-1).float() for g in g2])
    n1, n2 = float(v1.norm()), float(v2.norm())
    cos = float((v1 @ v2) / (v1.norm() * v2.norm() + 1e-12)) if min(n1, n2) > 0 else float("nan")
    if was_training:
        model.train()
    return {"cos_enc": cos, "g1_norm": n1, "g2_norm": n2,
            "ratio_g2_g1": (n2 / n1 if n1 > 0 else float("nan")),
            "l1": l1, "l2": l2}


def mlm_aux_loss(model, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                 mask_token_id, prob: float = 0.15):
    """掩码复原辅助：随机 15% 位置替换为 <extra_id_0>，用绑定词嵌入在原位重建。

    与 s1 共用同一条样本、额外一次编码器前向；无掩码位置时返回 None（跳过）。
    """
    if mask_token_id is None:
        return None
    keep = attention_mask.to(torch.bool)
    m = (torch.rand(input_ids.shape, device=input_ids.device) < float(prob)) & keep
    if int(m.sum()) == 0:
        return None
    masked = input_ids.clone()
    masked[m] = int(mask_token_id)
    h = model.encoder(masked, attention_mask)                 # (B, L, D) token 级特征
    emb_w = model.encoder.input_embeddings()                  # (V, D) 绑定词嵌入
    return F.cross_entropy(F.linear(h[m].float(), emb_w.float()), input_ids[m])


def s1_margin_loss(s1: torch.Tensor, labels: torch.Tensor, margin: float):
    """固定 margin 的饱和斥力（v0.4）：softplus(m − (2y−1)·s1)。

    把每个样本"正确侧"的分数推过 margin：越接近 0（越含糊）力越大；
    越过 margin 后梯度指数衰减→0（不再为已分对的样本持续施力）。
    """
    signed = (2.0 * labels.float() - 1.0) * s1.float()
    return F.softplus(float(margin) - signed).mean()


class FisherEMA:
    """s1 轴的 Fisher 判别比（类别 EMA 统计的在线近似，batch=1 友好）。

        R = (μA − μH)² / (σA² + σH²)        损失 = softplus(log_target − log R)

    含义：把同类聚拢（σ² 小）且异类拉开（Δμ 大）到目标比值后不再施力。
    当前样本以 (1−momentum) 权重进入统计（梯度只经这一项；历史统计为 detached 缓冲）。
    """

    def __init__(self, momentum: float = 0.95, log_target: float = 1.386,
                 warmup: int = 20, var_floor: float = 1e-3, min_gap: float = 0.05):
        self.m = float(momentum)
        self.log_target = float(log_target)
        self.warmup = int(warmup)
        self.var_floor = float(var_floor)
        self.min_gap = float(min_gap)
        self.mu: dict = {0: None, 1: None}
        self.var: dict = {0: None, 1: None}
        self.count: dict = {0: 0, 1: 0}

    def step(self, s1: torch.Tensor, labels: torch.Tensor):
        """更新缓冲并返回损失（未过 warmup / 类内差距过小返回 None）。"""
        s1 = s1.reshape(-1).float()
        labels = labels.reshape(-1)
        mu_new, var_new = dict(self.mu), dict(self.var)
        for cls in (0, 1):
            mask = labels == cls
            if not bool(mask.any()):
                continue
            sb = s1[mask]
            if self.mu[cls] is None:                       # 冷启动：直接取当前批
                mu_new[cls] = sb.mean()
                var_new[cls] = ((sb - sb.mean().detach()) ** 2).mean()
            else:                                          # EMA 一步（梯度只经当前批）
                old_mu = self.mu[cls]
                mu_new[cls] = (1 - self.m) * sb.mean() + self.m * old_mu
                var_new[cls] = (self.m * self.var[cls]
                                + (1 - self.m) * ((sb - old_mu.detach()) ** 2).mean())
        loss = None
        if (mu_new[0] is not None and mu_new[1] is not None
                and self.count[0] >= self.warmup and self.count[1] >= self.warmup):
            gap = mu_new[1] - mu_new[0]                      # μAI − μHuman
            if abs(float(gap.detach())) >= self.min_gap:     # 冷启动防 1/gap 尖峰
                var = (var_new[0] + var_new[1]).clamp_min(self.var_floor)
                log_r = 2.0 * torch.log(gap.abs() + 1e-8) - torch.log(var)
                loss = F.softplus(self.log_target - log_r)
        for cls in (0, 1):                                   # 缓冲 detached 更新
            if mu_new[cls] is None:
                continue
            self.mu[cls] = mu_new[cls].detach()
            self.var[cls] = var_new[cls].detach()
            self.count[cls] += int((labels == cls).sum())
        return loss


class CovEMA:
    """池化特征的 EMA 协方差 → 相关矩阵非对角平方惩罚（Barlow/VICReg-lite，batch=1）。

    去冗余对照臂用（v0.4.1）：压低不同维度的相关性（减少冗余），不做"信息压制"。
    """

    def __init__(self, dim: int, momentum: float = 0.99, eps: float = 1e-6):
        self.dim = int(dim)
        self.m = float(momentum)
        self.eps = float(eps)
        self.mean = None
        self.m2 = None

    def step(self, h: torch.Tensor):
        x = h.float().reshape(-1, self.dim).mean(dim=0)      # batch=1 → (D,)
        outer = torch.outer(x, x)
        if self.mean is None:                                # 冷启动只填缓冲
            self.mean = x.detach().clone()
            self.m2 = outer.detach().clone()
            return None
        mean_new = self.m * self.mean + (1 - self.m) * x
        m2_new = self.m * self.m2 + (1 - self.m) * outer
        cov = m2_new - torch.outer(mean_new, mean_new)
        std = cov.diagonal().clamp_min(self.eps).sqrt()
        corr = cov / torch.outer(std, std)
        off = corr - torch.diag(corr.diagonal())
        loss = (off ** 2).sum() / self.dim ** 2
        self.mean = mean_new.detach()
        self.m2 = m2_new.detach()
        return loss


# --------------------------------------------------------------------------- #
# 损失
# --------------------------------------------------------------------------- #
def stream_loss(cfg: dict, model, batch: dict, name: str, aux: dict | None = None):
    """按数据流计算损失。返回 (loss, parts)（parts 只用于日志；aux = 几何项状态）。"""
    lc = cfg.get("loss", {})
    if name == "m4":
        lam_rep = float(lc.get("rep", 0.0) or 0.0)
        lam_fisher = float(lc.get("fisher", 0.0) or 0.0)
        lam_cov = float(lc.get("cov", 0.0) or 0.0)
        need_h = (lam_rep > 0) or (lam_fisher > 0) or (lam_cov > 0)
        if need_h:                                   # 几何项复用同一次前向拿 h
            s1, _, h = model.forward_feat(batch["input_ids"], batch["attention_mask"])
        else:
            s1, _ = model(batch["input_ids"], batch["attention_mask"])
            h = None
        l1 = F.binary_cross_entropy_with_logits(s1.float(), batch["labels"].float())
        total = lc.get("s1", 1.0) * l1
        parts = {"s1": float(l1.detach())}
        if lam_rep > 0:                              # 固定 margin 饱和斥力
            l_rep = s1_margin_loss(s1, batch["labels"],
                                   float(lc.get("rep_margin", 1.0)))
            total = total + lam_rep * l_rep
            parts["rep"] = float(l_rep.detach())
        if lam_fisher > 0 and aux and aux.get("fisher") is not None:
            l_fisher = aux["fisher"].step(s1, batch["labels"])
            if l_fisher is not None:
                total = total + lam_fisher * l_fisher
                parts["fisher"] = float(l_fisher.detach())
        if lam_cov > 0 and aux and aux.get("cov") is not None and h is not None:
            l_cov = aux["cov"].step(h)
            if l_cov is not None:
                total = total + lam_cov * l_cov
                parts["cov"] = float(l_cov.detach())
        w_mlm = float(lc.get("mlm", 0.0) or 0.0)
        if w_mlm > 0:
            l_mlm = mlm_aux_loss(model, batch["input_ids"], batch["attention_mask"],
                                 lc.get("mlm_token_id"), float(lc.get("mlm_prob", 0.15)))
            if l_mlm is not None:
                total = total + w_mlm * l_mlm
                parts["mlm"] = float(l_mlm.detach())
        return total, parts
    if name == "pair":
        _, s2_plus = model(batch["input_ids_plus"], batch["attention_mask_plus"])
        _, s2_minus = model(batch["input_ids_minus"], batch["attention_mask_minus"])
        margin = float(lc.get("margin", 1.0))
        if int(model.s2_rank) == 1:                   # 与原文公式逐字对应
            gap = (s2_plus - s2_minus).squeeze(-1)    # 期望 ≥ margin
        else:                                         # 低秩版：到子空间的距离（注记 1）
            gap = (s2_plus - s2_minus).norm(dim=-1)
        l2 = F.relu(margin - gap).mean()
        return lc.get("s2", 1.0) * l2, {"s2": float(l2.detach())}
    if name == "pairxf":
        # 跨族 Δ 方向引力：cos(Δh_A, Δh_B)（同 task_id、不同族家族）
        _, _, h_pa = model.forward_feat(batch["input_ids_plus_a"],
                                        batch["attention_mask_plus_a"])
        _, _, h_ma = model.forward_feat(batch["input_ids_minus_a"],
                                        batch["attention_mask_minus_a"])
        _, _, h_pb = model.forward_feat(batch["input_ids_plus_b"],
                                        batch["attention_mask_plus_b"])
        _, _, h_mb = model.forward_feat(batch["input_ids_minus_b"],
                                        batch["attention_mask_minus_b"])
        da = (h_pa - h_ma).reshape(1, -1)
        db = (h_pb - h_mb).reshape(1, -1)
        cos = F.cosine_similarity(da, db, dim=-1, eps=1e-8).mean()
        l_xf = 1.0 - cos
        return lc.get("xfam", 1.0) * l_xf, {"xfam": float(l_xf.detach())}
    if name == "hybrid":
        tok_logits = model.token_logits(batch["input_ids"], batch["attention_mask"])
        tok = batch["tok_labels"].float()
        valid = tok != -100
        if not bool(valid.any()):
            return torch.zeros((), device=tok_logits.device), {"token": 0.0}
        l_tok = F.binary_cross_entropy_with_logits(tok_logits[valid].float(), tok[valid])
        return lc.get("token", 1.0) * l_tok, {"token": float(l_tok.detach())}
    raise KeyError(f"未知数据流 {name!r}（可选：m4 / pair / hybrid / pairxf）")


def orth_reg(model, eps: float = 1e-8) -> torch.Tensor:
    """cos²(w1, w2)（跨秩取各列均值）。

    零向量处：分子 = 0、分母 clamp 到 eps ⇒ **值与梯度都是 0**（不会 NaN）。
    """
    w1 = model.w1.weight.reshape(-1).float()
    w2 = model.w2.weight.float().reshape(model.s2_rank, model.dim)
    num = (w2 @ w1).pow(2)                                   # (r,)
    den = (w1.dot(w1) * (w2 * w2).sum(-1)).clamp_min(eps)    # (r,)
    return (num / den).mean()


# --------------------------------------------------------------------------- #
# 评测
# --------------------------------------------------------------------------- #
def _chunk_runs(flags: list) -> list:
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


def chunk_match(pred: list, gold: list, iou_threshold: float = 0.5):
    """片段级匹配（连续 AI 行段，IoU≥阈值贪心）→ (命中, 预测段数, 标注段数)。u-det 原样。"""
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


def prf1(pred: list, gold: list) -> dict:
    """0/1 序列的 p/r/f1（u-det 原样）。"""
    tp = sum(1 for p, g in zip(pred, gold) if p == 1 and g == 1)
    fp = sum(1 for p, g in zip(pred, gold) if p == 1 and g == 0)
    fn = sum(1 for p, g in zip(pred, gold) if p == 0 and g == 1)
    precision = tp / max(tp + fp, 1e-9)
    recall = tp / max(tp + fn, 1e-9)
    return {"p": precision, "r": recall,
            "f1": 2 * precision * recall / max(precision + recall, 1e-9)}


@torch.no_grad()
def evaluate(model, dataset, name: str, cfg: dict, device: str,
             dump_path: str | None = None) -> dict:
    """逐样本整段评测（不截断）：

        m4   → s1 的 acc / f1 / AUC（人机分离）
        pair → s2 的方向正确率 dir_acc 与 Δs2 统计（偏好位移）

    ``dump_path`` 非空时把**逐样本原始分数**存成 ``.pt``（含 s1 / s2 / 标签 / 长度），
    供 scripts/analyze_scores.py 离线做阈值扫描、长度分桶、2D GMM 与象限分析。
    """
    model.eval()
    amp = cfg["train"].get("amp", True) and device == "cuda"
    amp_dtype = "cuda" if device == "cuda" else "cpu"
    n = len(dataset)
    rank = int(model.s2_rank)

    if name == "m4":
        s1_all = torch.zeros(n)
        s2_all = torch.zeros(n, rank)
        lengths = [0] * n
        for i in tqdm(range(n), desc="eval m4", leave=False):
            item = dataset[i]
            ids = torch.tensor([item["input_ids"]], dtype=torch.long, device=device)
            with torch.autocast(amp_dtype, dtype=torch.bfloat16, enabled=amp):
                s1, s2 = model(ids, torch.ones_like(ids))
            s1_all[i] = s1[0].float().cpu()
            s2_all[i] = s2[0].float().cpu()
            lengths[i] = int(ids.shape[1])
        labels = [int(dataset.labels[i]) for i in range(n)]
        prob = torch.sigmoid(s1_all).tolist()
        pred = [int(p > 0.5) for p in prob]
        tp = sum(1 for p, g in zip(pred, labels) if p == 1 and g == 1)
        fp = sum(1 for p, g in zip(pred, labels) if p == 1 and g == 0)
        fn = sum(1 for p, g in zip(pred, labels) if p == 0 and g == 1)
        precision = tp / max(tp + fp, 1e-9)
        recall = tp / max(tp + fn, 1e-9)
        try:
            auc = float(roc_auc_score(labels, prob))
        except ValueError:                       # 冒烟时可能只有单类
            auc = float("nan")
        metrics = {
            "n": n,
            "acc": sum(1 for p, g in zip(pred, labels) if p == g) / max(n, 1),
            "f1": 2 * precision * recall / max(precision + recall, 1e-9),
            "auc": auc,
        }
        if dump_path:
            torch.save({
                "stream": "m4", "n": n,
                "s1": s1_all, "s2": s2_all,
                "label": torch.tensor(labels, dtype=torch.int8),
                "length": torch.tensor(lengths, dtype=torch.int32),
                "meta": [dict(m) for m in dataset.meta],
            }, dump_path)
            print(f"[eval] 逐样本分数已存：{dump_path}（n={n}）")
        return metrics

    if name == "pair":
        s1p_all = torch.zeros(n)
        s2p_all = torch.zeros(n, rank)
        s1m_all = torch.zeros(n)
        s2m_all = torch.zeros(n, rank)
        for i in tqdm(range(n), desc="eval pair", leave=False):
            item = dataset[i]
            ip = torch.tensor([item["input_ids_plus"]], dtype=torch.long, device=device)
            im = torch.tensor([item["input_ids_minus"]], dtype=torch.long, device=device)
            with torch.autocast(amp_dtype, dtype=torch.bfloat16, enabled=amp):
                s1p, s2p = model(ip, torch.ones_like(ip))
                s1m, s2m = model(im, torch.ones_like(im))
            s1p_all[i] = s1p[0].float().cpu()
            s2p_all[i] = s2p[0].float().cpu()
            s1m_all[i] = s1m[0].float().cpu()
            s2m_all[i] = s2m[0].float().cpu()
        gap = s2p_all - s2m_all
        gap = gap.squeeze(-1) if rank == 1 else gap.norm(dim=-1)     # (n,)
        metrics = {
            "n": n,
            "dir_acc": float((gap > 0).float().mean()),              # s2(x₊) > s2(x₋) 的比例
            "delta_mean": float(gap.mean()),
            "delta_median": float(gap.median()),
        }
        if dump_path:
            torch.save({
                "stream": "pair", "n": n,
                "s1_plus": s1p_all, "s2_plus": s2p_all,
                "s1_minus": s1m_all, "s2_minus": s2m_all,
                "meta": [dict(m) for m in dataset.meta],
            }, dump_path)
            print(f"[eval] 逐样本分数已存：{dump_path}（n={n}）")
        return metrics

    if name == "hybrid":
        line_sum = [dict() for _ in range(n)]
        line_cnt = [dict() for _ in range(n)]
        hits = [0, 0, 0, 0]                       # tp / fp / fn / tn（token 级，阈值 0.5）
        for i in tqdm(range(n), desc="eval hybrid", leave=False):
            item = dataset[i]
            ids = torch.tensor([item["input_ids"]], dtype=torch.long, device=device)
            with torch.autocast(amp_dtype, dtype=torch.bfloat16, enabled=amp):
                tok_logits = model.token_logits(ids, torch.ones_like(ids))
            probs = torch.sigmoid(tok_logits[0].float()).cpu().tolist()
            for prob, lab, line in zip(probs, item["tok_labels"], item["line_of_token"]):
                if lab == -100:
                    continue
                line_sum[i][line] = line_sum[i].get(line, 0.0) + prob
                line_cnt[i][line] = line_cnt[i].get(line, 0) + 1
                hit = 0 if prob > 0.5 else 3
                if prob > 0.5 and lab == 0:
                    hit = 1
                elif prob <= 0.5 and lab == 1:
                    hit = 2
                hits[hit] += 1
        pred_lines, gold_lines = [], []
        chunk_tp = chunk_pred = chunk_gold = 0
        for i in range(n):
            gold = dataset.line_label[i]
            picked = [int(line_sum[i].get(k, 0.0) / max(line_cnt[i].get(k, 0), 1) > 0.5)
                      for k in range(len(gold))]
            pred_lines.extend(picked)
            gold_lines.extend(gold)
            tp_i, p_i, g_i = chunk_match(picked, gold)
            chunk_tp += tp_i
            chunk_pred += p_i
            chunk_gold += g_i
        metrics = {"n": n}
        metrics.update({f"line_{k}": v for k, v in prf1(pred_lines, gold_lines).items()})
        precision = chunk_tp / max(chunk_pred, 1e-9)
        recall = chunk_tp / max(chunk_gold, 1e-9)
        metrics["chunk_p"], metrics["chunk_r"] = precision, recall
        metrics["chunk_f1"] = 2 * precision * recall / max(precision + recall, 1e-9)
        precision = hits[0] / max(hits[0] + hits[1], 1e-9)
        recall = hits[0] / max(hits[0] + hits[2], 1e-9)
        metrics["token_f1"] = 2 * precision * recall / max(precision + recall, 1e-9)
        return metrics

    if name == "pairxf":
        # 跨族方向一致性诊断：cos(Δh_A, Δh_B) 与两侧方向正确率
        cos_list = []
        ok_a = ok_b = 0
        for i in tqdm(range(n), desc="eval pairxf", leave=False):
            item = dataset[i]
            outs = {}
            for side in ("plus_a", "minus_a", "plus_b", "minus_b"):
                ids = torch.tensor([item[f"input_ids_{side}"]], dtype=torch.long,
                                   device=device)
                with torch.autocast(amp_dtype, dtype=torch.bfloat16, enabled=amp):
                    _, s2, h = model.forward_feat(ids, torch.ones_like(ids))
                outs[side] = (s2[0].float().cpu(), h[0].float().cpu())
            da = outs["plus_a"][1] - outs["minus_a"][1]
            db = outs["plus_b"][1] - outs["minus_b"][1]
            cos = float(F.cosine_similarity(da.reshape(1, -1), db.reshape(1, -1),
                                            dim=-1, eps=1e-8))
            cos_list.append(cos)
            ga = float((outs["plus_a"][0] - outs["minus_a"][0]).squeeze())
            gb = float((outs["plus_b"][0] - outs["minus_b"][0]).squeeze())
            ok_a += int(ga > 0)
            ok_b += int(gb > 0)
        return {"n": n,
                "dir_cos_mean": sum(cos_list) / max(n, 1),
                "dir_cos_pos": sum(1 for c in cos_list if c > 0) / max(n, 1),
                "a_dir_acc": ok_a / max(n, 1),
                "b_dir_acc": ok_b / max(n, 1)}

    raise KeyError(f"未知数据流 {name!r}（可选：m4 / pair / hybrid / pairxf）")


def monitor_score(results: dict, monitor: str) -> float:
    """选 best.pt 用的综合分（mean = **实际评测过的流**的可用指标等权平均）。

    注意：只统计 results 里存在的流 —— 单流消融（--no-pair / --no-m4）时，
    缺失流不能被当成 0 计入，否则会把分数系统性地压低。
    """
    if monitor == "m4_auc":
        return float(results.get("m4", {}).get("auc", 0.0) or 0.0)
    if monitor == "pair_acc":
        return float(results.get("pair", {}).get("dir_acc", 0.0) or 0.0)
    values = []
    for stream, key in (("m4", "auc"), ("pair", "dir_acc"), ("hybrid", "line_f1")):
        if stream not in results:
            continue
        value = results[stream].get(key, float("nan"))
        values.append(0.0 if (value is None or math.isnan(value)) else value)
    return sum(values) / max(len(values), 1)


# --------------------------------------------------------------------------- #
# 数据入口
# --------------------------------------------------------------------------- #
def make_dataset(cfg: dict, name: str, split: str, limit: int | None):
    data_cfg = cfg["data"]
    if name == "pairxf":                     # 跨族配对：两个文件按 task_id 对齐
        files = [resolve(data_cfg["processed_dir"]) / f for f in data_cfg["pairxf_files"]]
        for f in files:
            if not f.exists():
                raise SystemExit(f"[data] 找不到 {f}（pairxf 需要两份配对文件）")
        return limit_dataset(
            build_dataset("pairxf", files=[str(f) for f in files], split=split), limit)
    path = resolve(data_cfg["processed_dir"]) / data_cfg[f"{name}_file"]
    if name == "pair" and not path.exists():
        raise SystemExit(
            f"[data] 找不到 {path} —— 需要先跑 scripts/gen_pairs.py 生成配对数据"
            f"（见 README 的『有卡阶段』一节）；或先只用 m4 单流训练（--no-pair）。")
    return limit_dataset(
        build_dataset(name, file=str(path), split=split),
        limit,
    )


# --------------------------------------------------------------------------- #
# 训练
# --------------------------------------------------------------------------- #
def train(cfg: dict, args) -> None:
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    torch.manual_seed(cfg["train"].get("seed", 0))

    enc_cfg = dict(cfg["encoder"])
    enc_cfg["path"] = str(resolve(enc_cfg["path"]))          # 允许从任意 cwd 运行
    encoder = build_encoder(**enc_cfg)
    frozen = bool(cfg["encoder"].get("freeze", True))
    if frozen:
        encoder.eval()
        encoder.requires_grad_(False)
    mcfg = dict(cfg["model"])
    model_name = mcfg.pop("name", "dual")
    model = build_model(model_name, encoder=encoder,
                        dim=encoder.hidden_size, **mcfg)

    # 续跑（补 epoch 用）：只载模型权重，优化器 / LR 计划重新起（与 u-det 同语义）
    resume_epoch, resume_score = -1, -1.0
    if args.resume:
        ckpt = torch.load(resolve(args.resume), map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(ckpt.get("state", {}), strict=False)
        if missing or unexpected:
            print(f"[resume] 注意：未匹配 {len(missing)} 项（如 {missing[:3]}）、"
                  f"多余 {len(unexpected)} 项")
        resume_epoch = int(ckpt.get("epoch", -1))
        resume_score = float(ckpt.get("score", -1.0))
        print(f"[resume] 载入 {args.resume}（已训到 epoch {resume_epoch}，"
              f"monitor={resume_score:.4f}），本次追加 {cfg['train']['epochs']} 个 epoch")
    model.to(device)

    streams = list(cfg["train"]["streams"])
    loaders = {}
    for name in streams:                                     # 恒 batch=1：整段序列，不截断
        dataset = make_dataset(cfg, name, "train", args.limit)
        print(f"[data] {name} train: {split_counts(dataset)}")
        workers = cfg["train"].get("num_workers", 0)
        if name == "pairxf":                 # 两个 parquet 的 join 结果较小，单 worker 即可
            workers = cfg["train"].get("num_workers_pairxf", workers)
        loaders[name] = DataLoader(
            dataset, batch_size=1, shuffle=True, collate_fn=COLLATES[name],
            num_workers=workers,
        )

    params = [p for p in model.parameters() if p.requires_grad]
    n_param = sum(p.numel() for p in params)
    n_readout = model.w1.weight.numel() + model.w2.weight.numel()
    print(f"[model] {model_name}：可训练 {n_param / 1e6:.3f}M"
          f"（其中读出 w1/w2 = {n_readout}）"
          f"编码器 {encoder.__class__.__name__}"
          f"{'冻结' if frozen else ('全参微调' if getattr(encoder, 'full_ft', False) else '微调(LoRA)')}"
          f"，pool={model.pool} s2_rank={model.s2_rank}")

    # 掩码复原辅助：解析一次掩码 token id（T5 哨兵 <extra_id_0>）
    lam_mlm = float(cfg.get("loss", {}).get("mlm", 0.0) or 0.0)
    if lam_mlm > 0:
        from transformers import AutoTokenizer
        _tok = AutoTokenizer.from_pretrained(enc_cfg["path"])
        cfg["loss"]["mlm_token_id"] = int(_tok.convert_tokens_to_ids("<extra_id_0>"))
        print(f"[loss] 掩码复原开启：λmlm={lam_mlm} mask_id={cfg['loss']['mlm_token_id']} "
              f"prob={cfg['loss'].get('mlm_prob', 0.15)}")

    # 几何辅助项（v0.4）：状态对象挂在这里，经 aux 传入 stream_loss
    lam_rep = float(cfg.get("loss", {}).get("rep", 0.0) or 0.0)
    lam_fisher = float(cfg.get("loss", {}).get("fisher", 0.0) or 0.0)
    lam_xfam = float(cfg.get("loss", {}).get("xfam", 0.0) or 0.0)
    lam_cov = float(cfg.get("loss", {}).get("cov", 0.0) or 0.0)
    aux = {}
    if lam_fisher > 0:
        aux["fisher"] = FisherEMA(momentum=float(cfg["loss"].get("fisher_momentum", 0.95)),
                                  log_target=float(cfg["loss"].get("fisher_target", 1.386)),
                                  warmup=int(cfg["loss"].get("fisher_warmup", 20)),
                                  min_gap=float(cfg["loss"].get("fisher_min_gap", 0.05)))
    if lam_cov > 0:
        aux["cov"] = CovEMA(dim=int(encoder.hidden_size),)
    if lam_rep or lam_fisher or lam_xfam or lam_cov:
        print(f"[geom] rep={lam_rep}(margin={cfg.get('loss', {}).get('rep_margin', 1.0)}) "
              f"fisher={lam_fisher} xfam={lam_xfam} cov={lam_cov}")

    lr = cfg["train"]["lr"]
    lr_encoder = cfg["train"].get("lr_encoder", lr)
    enc_params = [p for p in encoder.parameters() if p.requires_grad]
    other_params = [p for n, p in model.named_parameters()
                    if p.requires_grad and not n.startswith("encoder.")]
    groups = [{"params": other_params, "lr": lr}]
    if enc_params:
        groups.append({"params": enc_params, "lr": lr_encoder})
    opt = torch.optim.AdamW(groups, lr=lr, weight_decay=cfg["train"].get("weight_decay", 0.01))

    # 归因诊断：固定小 batch 上的双流梯度快照（--diag-gradconf K；0=关）
    diag_every = int(getattr(args, "diag_gradconf", 0) or 0)
    diag_state = None
    if diag_every > 0:
        if not enc_params:
            print("[gradconf] 编码器已冻结，跳过梯度快照")
        else:
            diag_state = _diag_batches(cfg, args, device)
            print(f"[gradconf] 开启：每 {diag_every} step 记录 ∇L1/∇L2 冲突"
                  f"（16 m4 + 16 pair 固定子集）")

    if any(len(loader) == 0 for loader in loaders.values()):
        raise SystemExit("某个数据流为空（检查 --limit / 子集）")
    steps_per_epoch = max(len(loader) for loader in loaders.values())
    accum = max(1, int(cfg["train"].get("grad_accum", 1)))
    updates_per_epoch = max(1, steps_per_epoch // accum)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[g["lr"] for g in groups],
        total_steps=updates_per_epoch * cfg["train"]["epochs"],
        pct_start=cfg["train"].get("warmup", 0.05),
    )

    run_dir = resolve(cfg["train"].get("out_dir", "runs")) / args.tag
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    append_log = bool(args.resume) and (run_dir / "metrics.csv").exists()   # 续跑：原 csv 追加
    log_file = open(run_dir / "metrics.csv", "a" if append_log else "w",
                    newline="", encoding="utf-8")
    logger = csv.writer(log_file)
    if not append_log:
        logger.writerow(["epoch", "step", "loss", "loss_s1", "loss_s2", "loss_orth",
                         "loss_mlm", "loss_token", "loss_rep", "loss_fisher",
                         "loss_xfam", "loss_cov", "lr", "sec"])

    amp = cfg["train"].get("amp", True) and device == "cuda"
    amp_dtype = "cuda" if device == "cuda" else "cpu"
    lam_orth = float(cfg.get("loss", {}).get("orth", 0.0) or 0.0)
    best = resume_score
    epochs = int(cfg["train"]["epochs"])
    start_epoch = resume_epoch + 1
    print(f"[train] streams={streams} steps/epoch={steps_per_epoch} accum={accum} "
          f"updates/epoch={updates_per_epoch} epochs={epochs}"
          f"（epoch {start_epoch}…{start_epoch + epochs - 1}）"
          f" amp={amp} λa={cfg['loss'].get('s1', 1.0)} λb={cfg['loss'].get('s2', 1.0)} "
          f"λc={lam_orth} margin={cfg['loss'].get('margin', 1.0)}")

    for offset in range(epochs):
        epoch = start_epoch + offset
        model.train()
        if frozen:
            model.encoder.eval()                 # 冻结编码器保持 eval（关掉其 dropout）
        iters = {name: cycle(loader) for name, loader in loaders.items()}
        running = {"loss": 0.0, "s1": 0.0, "s2": 0.0, "orth": 0.0, "mlm": 0.0,
                   "token": 0.0, "rep": 0.0, "fisher": 0.0, "xfam": 0.0, "cov": 0.0}
        start_time = time.time()
        opt.zero_grad(set_to_none=True)
        bar = tqdm(range(steps_per_epoch), desc=f"epoch {epoch}", unit="it")
        for step in bar:
            step_parts = {"s1": 0.0, "s2": 0.0, "orth": 0.0, "mlm": 0.0,
                          "token": 0.0, "rep": 0.0, "fisher": 0.0, "xfam": 0.0, "cov": 0.0}
            for name in streams:                 # 各流各取一个样本，同时参与
                batch = next(iters[name])
                batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                         for k, v in batch.items()}
                with torch.autocast(amp_dtype, dtype=torch.bfloat16, enabled=amp):
                    loss, parts = stream_loss(cfg, model, batch, name, aux)
                (loss / (accum * len(streams))).backward()
                for key, value in parts.items():
                    step_parts[key] += value
            if lam_orth > 0:                     # 正交保险：不依赖数据流，每步回传
                l_orth = orth_reg(model)
                (lam_orth * l_orth / (accum * len(streams))).backward()
                step_parts["orth"] = float(l_orth.detach())

            if (step + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(params, cfg["train"].get("grad_clip", 1.0))
                opt.step()
                scheduler.step()
                opt.zero_grad(set_to_none=True)

            if diag_state is not None and (step + 1) % diag_every == 0:
                try:
                    rec = _grad_conflict(model, diag_state, enc_params, device, amp,
                                         amp_dtype, cfg.get("loss", {}))
                    with open(run_dir / "gradconf.jsonl", "a", encoding="utf-8") as f:
                        f.write(json.dumps({"epoch": epoch, "step": step + 1, **rec}) + "\n")
                except Exception as exc:             # 诊断绝不打断训练
                    print(f"[gradconf] 跳过（{type(exc).__name__}: {exc}）")

            for key, value in step_parts.items():
                running[key] += value
            running["loss"] += sum(step_parts.values())
            if (step + 1) % cfg["train"].get("log_every", 200) == 0 or step == steps_per_epoch - 1:
                elapsed = time.time() - start_time
                done = step + 1
                bar.set_postfix(loss=f"{running['loss'] / done:.3f}",
                                s1=f"{running['s1'] / done:.3f}",
                                s2=f"{running['s2'] / done:.3f}",
                                lr=f"{scheduler.get_last_lr()[0]:.2e}")
                logger.writerow([epoch, done, f"{running['loss'] / done:.6f}",
                                 f"{running['s1'] / done:.6f}",
                                 f"{running['s2'] / done:.6f}",
                                 f"{running['orth'] / done:.6f}",
                                 f"{running['mlm'] / done:.6f}",
                                 f"{running['token'] / done:.6f}",
                                 f"{running['rep'] / done:.6f}",
                                 f"{running['fisher'] / done:.6f}",
                                 f"{running['xfam'] / done:.6f}",
                                 f"{running['cov'] / done:.6f}",
                                 f"{scheduler.get_last_lr()[0]:.3e}", f"{elapsed:.1f}"])
                log_file.flush()
        bar.close()

        results = {}
        for name in streams:
            dataset = make_dataset(cfg, name, "val", args.limit)
            if len(dataset) == 0:
                continue
            metrics = evaluate(model, dataset, name, cfg, device)
            results[name] = metrics
            print(f"[val {epoch}] {name}: "
                  + json.dumps({k: round(v, 4) for k, v in metrics.items()}))
        score = monitor_score(results, cfg["train"].get("monitor", "mean"))
        cos = readout_cos(model)
        print(f"[diag {epoch}] cos(w1,w2) = {cos:+.4f}（0=正交；越大越退化为同一方向）")
        train_avg = {k: v / max(steps_per_epoch, 1) for k, v in running.items()}
        with open(run_dir / "metrics.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"epoch": epoch, "train": train_avg, "val": results,
                                "cos_w1w2": cos}) + "\n")

        # 只存可训练子集的 state（冻结底座/LUT buffer 不入库，checkpoint 保持小体积）
        trainable = {n for n, p in model.named_parameters() if p.requires_grad}
        state = {k: v for k, v in model.state_dict().items()
                 if not k.startswith("encoder.") or k in trainable}
        torch.save({"cfg": cfg, "epoch": epoch, "score": score,
                    "metrics": results, "state": state}, run_dir / "last.pt")
        if score > best:
            best = score
            shutil.copyfile(run_dir / "last.pt", run_dir / "best.pt")
            print(f"[ckpt] 保存 best.pt（{cfg['train'].get('monitor', 'mean')}={score:.4f}）")
        elif not (run_dir / "best.pt").exists():
            # 续跑且未超过 resume 分数：保底写一份 best.pt（=last），避免下游脚本 FileNotFound
            # （2026-09-24 实测踩坑：v0.4.1 monitor 0.9232 ≤ resume 0.9233 → best.pt 缺失 → 队列评测全崩）
            shutil.copyfile(run_dir / "last.pt", run_dir / "best.pt")
            print(f"[ckpt] 本次未超过 resume 分数（{score:.4f} ≤ {best:.4f}），"
                  f"best.pt 回退为 last.pt（保底）")

    log_file.close()
    print(f"[train] 完成，产物目录：{run_dir}")


# --------------------------------------------------------------------------- #
# 评测入口
# --------------------------------------------------------------------------- #
def run_eval(cfg: dict, args) -> None:
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    enc_cfg = dict(cfg["encoder"])
    enc_cfg["path"] = str(resolve(enc_cfg["path"]))          # 允许从任意 cwd 运行
    encoder = build_encoder(**enc_cfg)
    frozen = bool(cfg["encoder"].get("freeze", True))
    if frozen:
        encoder.eval()
        encoder.requires_grad_(False)
    mcfg = dict(cfg["model"])
    model = build_model(mcfg.pop("name", "dual"), encoder=encoder,
                        dim=encoder.hidden_size, **mcfg)
    ckpt = torch.load(resolve(args.ckpt), map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(ckpt["state"], strict=False)
    if missing or unexpected:
        print(f"[eval] 注意：未加载 {len(missing)} 项（如 {missing[:3]}）、"
              f"多余 {len(unexpected)} 项")
    model.to(device)
    print(f"[eval] 载入 {args.ckpt}（epoch {ckpt.get('epoch')}），配置来自当前 yaml/命令行")

    # ★ 防呆：评测配置必须与**训练时**的一致，否则会**静默**产出垃圾结果（u-det 实测踩过）。
    #   只比对决定"前向语义"的段（encoder / model）；分块大小 / static / compile / ckpt 等
    #   数学无关键允许不同。
    trained_cfg_path = Path(resolve(args.ckpt)).parent / "config.yaml"
    if trained_cfg_path.exists():
        with open(trained_cfg_path, encoding="utf-8") as f:
            trained_cfg = yaml.safe_load(f) or {}
        neutral = {"block_batch", "static", "compile", "ckpt", "chunk", "num_workers"}
        diffs = []
        for section in ("encoder", "model"):
            old, new = (trained_cfg.get(section) or {}), (cfg.get(section) or {})
            for key in sorted(set(old) | set(new)):
                if key in neutral:
                    continue
                if old.get(key) != new.get(key):
                    diffs.append(f"{section}.{key}: 训练={old.get(key)!r}  评测={new.get(key)!r}")
        if diffs:
            print("[eval] ⚠⚠ 评测配置与训练配置不一致 —— 这会静默产出错误结果，已中止：")
            for line in diffs[:15]:
                print(f"          {line}")
            if len(diffs) > 15:
                print(f"          ...（共 {len(diffs)} 处）")
            print(f"[eval]  训练配置：{trained_cfg_path}")
            raise SystemExit(2)
        print(f"[eval] 配置一致性检查通过（比对自 {trained_cfg_path}）")

    result = {}
    for name in cfg["train"]["streams"]:
        for split in ("val", "test"):
            dataset = make_dataset(cfg, name, split, args.limit)
            if len(dataset) == 0:
                continue
            dump = None
            if args.dump_scores and name != "pairxf":
                dump = str(Path(resolve(args.ckpt)).parent / f"scores_{name}_{split}.pt")
            metrics = evaluate(model, dataset, name, cfg, device, dump_path=dump)
            result[f"{name}/{split}"] = metrics
            print(f"[eval] {name}/{split}: "
                  + json.dumps({k: round(v, 4) for k, v in metrics.items()}))

    # ★ 防呆：--limit 的冒烟评测绝不能覆盖正式 eval.json（u-det 实测踩过）
    suffix = f"_limit{args.limit}" if args.limit else ""
    out = Path(resolve(args.ckpt)).parent / f"eval{suffix}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"ckpt": str(args.ckpt), "epoch": ckpt.get("epoch"), "metrics": result},
                  f, ensure_ascii=False, indent=2)
    if args.limit:
        print(f"[eval] ⚠ 检测到 --limit {args.limit}（冒烟评测）⇒ 结果写入 {out.name}，"
              f"**不会覆盖**正式的 eval.json")
    print(f"[eval] 结果已写入 {out}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(description="d-det 训练/评测（s1 众数坍缩 + s2 偏好位移）")
    parser.add_argument("--config", default="configs/ddet_base.yaml")
    parser.add_argument("--tag", default="ddet-base")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None, help="每个数据集只用前 N 条（冒烟）")
    parser.add_argument("--margin", type=float, default=None, help="覆盖 loss.margin")
    parser.add_argument("--no-m4", action="store_true", help="去掉 m4 流（只训 s2）")
    parser.add_argument("--no-pair", action="store_true", help="去掉 pair 流（只训 s1）")
    parser.add_argument("--model", default=None, help="覆盖 model.name（dual）")
    parser.add_argument("--encoder", default=None,
                        help="覆盖 encoder.name（codet5blk / codet5lora / codet5tok / codet5）")
    parser.add_argument("--freeze-encoder", action="store_true", help="冻结编码器")
    parser.add_argument("--unfreeze-encoder", action="store_true", help="解冻编码器（LoRA 微调）")
    parser.add_argument("--set", action="append", default=None, metavar="段.键=值",
                        help="通用配置覆盖，可重复；例：--set loss.orth=0.0")
    parser.add_argument("--resume", default=None,
                        help="从该 checkpoint 续跑（只载模型权重；--epochs 变成"
                             "\"本次追加几个 epoch\"）")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--dump-scores", action="store_true",
                        help="评测时把逐样本 s1/s2 分数存到 runs/<tag>/scores_<流>_<split>.pt，"
                             "供 scripts/analyze_scores.py 离线分析")
    parser.add_argument("--diag-gradconf", type=int, default=0, metavar="K",
                        help="每 K step 在固定小 batch 上记录 ∇L1/∇L2 梯度冲突（归因用；0=关）")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    cfg = apply_overrides(load_config(args.config), args)
    check_anchor(cfg)
    if args.eval:
        if not args.ckpt:
            raise SystemExit("--eval 需要 --ckpt 指定权重")
        run_eval(cfg, args)
    else:
        train(cfg, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
