#!/usr/bin/env python
"""U-Det 训练 / 评测主干（根目录单文件）。

模型（v0.2）：逐 token 编码器（查表，可冻结/可训练）
            -> 频域 U-Net（FFT 上下采样 + 8 层瓶颈）
            -> 样本级头（瓶颈池化）+ 多尺度 token 级头（5 个尺度：L/16 … L）。
数据：双流并行（m4 样本级 / hybrid token 级），**不截断代码**；训练与评测一律 batch=1，
      用梯度累积凑有效批大小（FFT 是全局算子，批内 padding 会污染频谱语义）。
损失：样本级 CE + 各尺度 token 级 BCE（标签按尺度池化，权重由粗到细）。
v0.4 另加两项可关的辅助损失（见 `loss` 段）：

  * ``cons_pool`` A 层级一致性：粗尺度 ← 细化预测的池化（自蒸馏 / 池化可交换）；
  * ``cons_pos``  B2 跳跃引导下采样：下采样路径的窗口内**逐 token 位置探针**。

另外 ``token_target`` 可把粗尺度的聚合目标从 mean（比例）换成 max / lse（是否有 AI）。

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
import shutil
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
from models import (PooledClassifier, PositionProbes, SampleHead, TokenHeads,
                    WindowedContextClassifier, build_hier, position_targets)
from report import build_report

ROOT = Path(__file__).resolve().parent
IGNORE = -100
#: batch_losses 返回的"非样本级"损失项名（用于日志/轮均）
LOSS_KEYS = ("token", "cons_pool", "cons_pos")


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
    if args.encoder is not None:
        cfg["encoder"]["name"] = args.encoder
    if args.freeze_encoder:
        cfg["encoder"]["freeze"] = True
    if args.unfreeze_encoder:
        cfg["encoder"]["freeze"] = False
    if args.layer is not None:
        cfg["encoder"]["layer"] = args.layer
    if args.block_batch is not None:                   # 分块编码器：降低显存的自保守旋钮
        cfg["encoder"]["block_batch"] = args.block_batch
    if args.no_cons:                                   # v0.4 消融：关掉 A（池化自蒸馏）+ B2（位置探针）
        cfg.setdefault("loss", {})
        cfg["loss"]["cons_pool"] = 0.0
        cfg["loss"]["cons_pos"] = 0.0
    if args.token_target is not None:                  # v0.4 消融：统一覆盖各尺度聚合目标
        cfg.setdefault("loss", {})
        cfg["loss"]["token_target"] = [args.token_target] * 5
    if args.no_gate:                                   # v0.4 消融：退回 v0.3 的纯跨注意力跳连
        cfg.setdefault("model", {})["gate"] = False
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
BASELINE_MODELS = ("codet5cls", "pooled", "codet5win", "window")
#: 用滑动窗口全长覆盖（而非 max_length 截断）的基线名
WINDOW_MODELS = ("codet5win", "window")


class UDet(nn.Module):
    """逐 token 编码器 -> 窗口注意力 U-Net -> 样本级头 + 多尺度 token 级头（整段序列，batch=1）。

    v0.4 额外可挂一组 **下采样位置探针**（``pos_probes``）：从下采样路径的每一级解出
    "窗口内逐 token"的 logits，逼下采样保留窗口内的位置结构（loss.cons_pos > 0 时启用）。
    """

    accepts_aux = True

    def __init__(self, encoder: nn.Module, backbone: nn.Module, sample_head: nn.Module,
                 token_heads: nn.Module, pos_probes: nn.Module | None = None):
        super().__init__()
        self.encoder = encoder
        self.backbone = backbone
        self.sample_head = sample_head
        self.token_heads = token_heads
        self.pos_probes = pos_probes

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None,
                return_aux: bool = False, report: torch.Tensor | None = None):
        """返回 ``(sample_logits, token_logits)``；``return_aux=True`` 时多返一个 aux dict。

        ``report``：v0.4.6 的文档级统计向量，**只喂样本头** ——
        按要求不接瓶颈输入，因为那会改动整条上采样通路、连带改变 token 级。
        """
        feats = self.encoder(input_ids)                          # (B, L, D)，整段不截断
        if return_aux and self.pos_probes is not None:
            levels, _, downs = self.backbone(feats, return_features=True, return_down=True)
        else:
            levels, _ = self.backbone(feats, return_features=True)
            downs = None
        sample_logits = self.sample_head(levels, report=report)   # 多尺度池化 (+报告) -> (B, 2)
        token_logits = None if self.token_heads is None else self.token_heads(levels, feats)
        if not return_aux:
            return sample_logits, token_logits
        aux = {"downs": downs, "pos_logits": None, "spans": None}
        if downs is not None:
            aux["pos_logits"] = self.pos_probes(downs)
            aux["spans"] = self.pos_probes.spans
        return sample_logits, token_logits, aux


def build_model(cfg: dict, tokenizer) -> tuple[nn.Module, nn.Module]:
    """按 cfg 组装模型：默认频域 U-Net；model.name 为基线名时构建直接二分类基线。"""
    encoder = build_encoder(**cfg["encoder"])
    freeze = bool(cfg["encoder"].get("freeze", True))
    if freeze:
        encoder.eval()
        encoder.requires_grad_(False)

    mcfg = dict(cfg["model"])
    name = mcfg.pop("name", "hier")
    if name in BASELINE_MODELS:                                  # 基线：编码器 -> 池化(可滑窗) -> 分类
        bcfg = dict(mcfg.get("baseline", {}))
        if name in WINDOW_MODELS:
            strides = bcfg.pop("stride_by_stream", None) or None           # 只给 train/evaluate 读
            print(f"[model] 基线 WindowedContextClassifier（滑动窗口全长覆盖）"
                  f"window={bcfg.get('window')} stride={bcfg.get('stride') or 'win/2'} "
                  f"按流覆盖={strides} window_batch={bcfg.get('window_batch', 32)} "
                  f"token_head={bool(bcfg.get('token_head', True))}")
            return WindowedContextClassifier(encoder, dim=encoder.hidden_size, **bcfg), encoder
        return PooledClassifier(encoder, dim=encoder.hidden_size, **bcfg), encoder

    mcfg.pop("baseline", None)
    backbone = build_hier(name, **mcfg)
    heads = dict(cfg.get("heads", {}))
    # v0.4.4：样本头可以读多个尺度（默认 1 = 只读瓶颈，与历史行为一致）。
    # 主干只有瓶颈 1 个尺度时（sample_only）自动夹紧，避免 LayerNorm 宽度不匹配。
    n_avail = 1 if getattr(backbone, "sample_only", False) else backbone.depth + 1
    n_req = int(heads.get("sample_levels", 1))
    n_levels = max(1, min(n_req, n_avail))
    if n_levels != n_req:
        print(f"[model] heads.sample_levels 请求 {n_req}，但主干只有 {n_avail} 个尺度 ⇒ 夹到 {n_levels}")
    # v0.4.6：报告改为文档级向量时，从 heads 段读它的维度（0 = 不接，与历史逐位一致）
    rep_dim = int(heads.get("report_dim", 0) or 0)
    rep_proj = int(heads.get("report_proj", 32))
    sample_head = SampleHead(backbone.dim, hidden=heads.get("sample_hidden"),
                             dropout=heads.get("sample_dropout", 0.0), n_levels=n_levels,
                             report_dim=rep_dim, report_proj=rep_proj)
    if rep_dim > 0:
        print(f"[model] 样本头接入**文档级报告向量**（{rep_dim} -> {rep_proj} 维，"
              f"只喂样本头、不进序列、不进瓶颈）")
    if n_levels > 1:
        print(f"[model] 样本头读 {n_levels} 个尺度（由粗到细，跨度 64→1）：各自池化后拼接，"
              f"宽度 {backbone.dim * n_levels}" + (f" + {rep_proj}（报告）" if rep_dim > 0 else "")
              + f"（vs 只读瓶颈的 {backbone.dim}）")
    if getattr(backbone, "sample_only", False):
        token_heads = None
        print("[model] 单任务模式（主干 sample_only=True）：不建 token 头、不建上采样/跳连路径，"
              "只做样本级分类（token 级损失会被 batch_losses 自动跳过）")
    else:
        token_heads = TokenHeads(
            [backbone.dim] * (backbone.depth + 1), out=1,
            bypass_dim=backbone.dim if heads.get("token_bypass") else None)
        if heads.get("token_bypass"):
            print(f"[model] token 头旁路（v0.4.5）：最细尺度额外拼接**编码器全长特征**"
                  f"（{backbone.dim} -> {2 * backbone.dim}），因为主干各尺度都在瓶颈之后")
    loss_cfg = cfg.get("loss", {})
    pos_probes = None
    if float(loss_cfg.get("cons_pos", 0.0) or 0.0) > 0:            # 只在启用位置探针时才建参数
        spans = getattr(backbone, "down_strides", None)
        if not spans:
            raise SystemExit(f"model.name={name!r} 不支持 loss.cons_pos（无 down_strides）")
        pos_probes = PositionProbes(backbone.dim, spans=spans,
                                    hidden=int(loss_cfg.get("cons_pos_hidden", 256)),
                                    dropout=float(heads.get("sample_dropout", 0.0)))
        print(f"[model] 位置探针 spans={pos_probes.spans}（下采样路径逐 token 位置约束）")
    return UDet(encoder, backbone, sample_head, token_heads, pos_probes), encoder


# --------------------------------------------------------------------------- #
# 损失
# --------------------------------------------------------------------------- #
def _aggregate_target(t: torch.Tensor, v: torch.Tensor, k: int, mode: str, beta: float):
    """把 (B, L) 的硬标签按窗口 k 聚合成 (B, 1, L_k) 目标。

    ``mean``：窗口内 AI **比例**（v0.3 原行为）；``max``：窗口内**是否有** AI；
    ``lse``：软 max（β 越大越接近 max，β→0 退化为 mean）。
    无效位由 ``v`` 掩掉，分母只数有效 token。
    """
    if mode == "mean" or k == 1:
        num = F.avg_pool1d((t * v).unsqueeze(1), k, k)
        den = F.avg_pool1d(v.unsqueeze(1), k, k)
        return num / den.clamp(min=1e-6)
    if mode == "max":
        masked = (t + (1.0 - v) * -1e4).unsqueeze(1)
        return F.max_pool1d(masked, k, k).clamp(0.0, 1.0)
    if mode == "lse":
        num = F.avg_pool1d((torch.exp(beta * t) * v).unsqueeze(1), k, k) * k      # Σ exp(β·y)
        den = F.avg_pool1d(v.unsqueeze(1), k, k) * k                              # |valid|
        return torch.log(num.clamp(min=1e-6) / den.clamp(min=1.0)) / beta
    raise ValueError(f"未知 token_target 模式 {mode!r}，可选：mean / max / lse")


def scale_weights_length(logit_scales: list[torch.Tensor], nmin: float = 64.0,
                         tau: float = 0.5) -> list[float]:
    """v0.4.6：长度感知的逐尺度权重  w_i = σ((log2 n_i − log2 N_min) / τ)。

    ``n_i`` 取**该尺度的真实位置数** ``logits.shape[-1]``（而不是 L/s_i）——
    它就是 `_aggregate_target` 用的同一个量，两者必须一致。
    短文档的粗尺度位置数少 ⇒ w 被压低 ⇒ 权重自动向**高分辨率尺度**集中，
    与「短文本更该参考浅层/高分辨率」的直觉一致。

    ★ 已知局限（探针实测确认，见 docx/u-det-v0.4.md §8.12.5-④）：
      本函数只有两个退化 regime ——
        · n_i ≫ N_min  ⇒ 5 个权重全趋 1（**均匀**，实测是最差的非退化解）；
        · n_i ≪ N_min  ⇒ 权重 ∝ (1/s_i)^(1/τ)（**与 L 无关**，实测 L=293 与 L=8192 几乎逐位相同）。
      所以它**无法表达“短文档温和、长文档激进”**；长度依赖只存在于过渡带 L ~ N_min·s_i。
      当前取值 N_min=64 / τ=0.5 是探针在冻结 v0.4.5 上扫出来的加权 BCE：
      静态 0.3994 / N_min=8 0.4199（**更差**）/ N_min=64 0.3921（更优）。
    """
    lo = math.log2(max(float(nmin), 1e-9))
    return [1.0 / (1.0 + math.exp(-((math.log2(max(float(t.shape[-1]), 1.0)) - lo) / tau)))
            for t in logit_scales]


def token_loss(logit_scales: list[torch.Tensor], tok_labels: torch.Tensor,
               weights: list[float], ignore: int = IGNORE,
               modes: list[str] | None = None, beta: float = 4.0,
               weight_mode: str = "static", nmin: float = 64.0,
               tau: float = 0.5) -> torch.Tensor | None:
    """各尺度 token 级 BCE：标签按尺度池化成目标（默认平均池化软标签）。

    Args:
        modes: 逐尺度（由粗到细）的聚合方式，见 `_aggregate_target`；None = 全部 mean。
        beta: ``lse`` 模式的锐化系数。
        weight_mode: ``static``（默认，直接用 ``weights``，与历史**逐位一致**）
            或 ``length``（v0.4.6，改用 `scale_weights_length`）。
    """
    valid = (tok_labels != ignore)
    if not valid.any():
        return None
    if weight_mode == "length":
        weights = scale_weights_length(logit_scales, nmin, tau)
    elif weight_mode != "static":
        raise ValueError(f"未知 weight_mode {weight_mode!r}，可选：static / length")
    target = tok_labels.clamp(min=0).float()
    length = tok_labels.shape[-1]
    modes = list(modes or ["mean"] * len(logit_scales))
    if len(modes) != len(logit_scales):
        raise ValueError(f"token_target 长度必须等于尺度数 {len(logit_scales)}（收到 {len(modes)}）")
    total, total_w = 0.0, 0.0
    for logits, weight, mode in zip(logit_scales, weights, modes):
        want = logits.shape[-1]                                  # 该尺度真实长度（可能 ceil 过）
        k = max(1, math.ceil(length / want))
        pad = max(0, want * k - length)                          # 补齐到整数倍，池化后恰好 want 个 bin
        t = F.pad(target, (0, pad)) if pad else target
        v = F.pad(valid.float(), (0, pad)) if pad else valid.float()
        den = F.avg_pool1d(v.unsqueeze(1), k, k)
        keep = (den > 0).float()
        if keep.sum() == 0:
            continue
        soft = _aggregate_target(t, v, k, mode, beta)
        loss = F.binary_cross_entropy_with_logits(logits, soft, weight=keep, reduction="sum") / keep.sum()
        total = total + weight * loss
        total_w += weight
    return None if total_w == 0 else total / total_w


def _pool_ratio(length: int, length_target: int) -> tuple[int, int]:
    """从长度 ``length`` 平均池化到 ``length_target``：返回 (池化核 k, 末尾补齐量)。"""
    k = max(1, math.ceil(length / length_target))
    return k, max(0, length_target * k - length)


def _mask_at(tok_labels: torch.Tensor, length_target: int, ignore: int = IGNORE) -> torch.Tensor:
    """把"是否有效 token"掩码池化到 ``length_target``，返回 (B, 1, L_t) 的有效比例。"""
    valid = (tok_labels != ignore).float().unsqueeze(1)
    length = valid.shape[-1]
    k, pad = _pool_ratio(length, length_target)
    if pad:
        valid = F.pad(valid, (0, pad))
    return F.avg_pool1d(valid, k, k)[:, :, :length_target]


def cons_pool_loss(logit_scales: list[torch.Tensor], tok_labels: torch.Tensor,
                   weights: list[float] | None = None, detach: bool = True,
                   ignore: int = IGNORE) -> torch.Tensor | None:
    """v0.4 · A：层级一致性（池化可交换 / 自蒸馏）。

    要求**更粗一级**的预测等于**更细一级**预测的窗口池化值：

        BCE(logits_k,  pool(σ(logits_{k+1})).detach())

    与现有 token 级损失的区别在**目标**：现在粗尺度的目标是把标签池化成比例（一个 64 长的窗口
    只有 1 个 AI token 时目标 0.016，糊且带标注噪声），这里的目标是**模型自己更细一级的预测**池化而来
    ——更细一级能看到全分辨率特征（下采样前），它的池化是一个更锐、方差更小的估计。
    信号从"细/skip 侧"流向"粗/下采样侧"，这正是"跳跃连接引导下采样"。

    Args:
        weights: 相邻尺度对的权重（长度 = 尺度数 - 1）。
        detach: True = 自蒸馏（目标不回传，防止两侧一起塌到常数）。
    """
    if len(logit_scales) < 2:
        return None
    total, total_w = 0.0, 0.0
    for k in range(len(logit_scales) - 1):
        coarse, fine = logit_scales[k], logit_scales[k + 1]
        length_c, length_f = coarse.shape[-1], fine.shape[-1]
        mask = _mask_at(tok_labels, length_f, ignore)                     # (B, 1, L_f)
        keep_f = (mask > 0).float()
        p = torch.sigmoid(fine.detach() if detach else fine)
        p = (p * keep_f)
        r, pad = _pool_ratio(length_f, length_c)                          # L_f -> L_c
        if pad:
            p, keep_f = F.pad(p, (0, pad)), F.pad(keep_f, (0, pad))
        num = F.avg_pool1d(p, r, r)
        den = F.avg_pool1d(keep_f, r, r)
        keep = (den > 0).float()
        if keep.sum() == 0:
            continue
        pooled = (num / den.clamp(min=1e-6))[:, :, :length_c]
        keep = keep[:, :, :length_c]
        if pooled.shape[-1] < length_c:                                   # r 取整造成的尾部长度差
            pad_c = length_c - pooled.shape[-1]
            pooled, keep = F.pad(pooled, (0, pad_c)), F.pad(keep, (0, pad_c))
        loss = F.binary_cross_entropy_with_logits(coarse, pooled, weight=keep,
                                                  reduction="sum") / keep.sum().clamp(min=1)
        w = 1.0 if not weights else float(weights[k])
        total = total + w * loss
        total_w += w
    return None if total_w == 0 else total / total_w


def position_loss(probe_logits: list[torch.Tensor], spans, tok_labels: torch.Tensor,
                  weights: list[float] | None = None, ignore: int = IGNORE) -> torch.Tensor | None:
    """v0.4 · B2：跳跃连接引导下采样（下采样路径的窗口内逐 token 位置探针）。

    ``probe_logits[i]`` 形状 ``(B, L_k, span)``，第 j 个位置的第 t 个偏移对应原始 token
    ``j * span + t``；直接对该 token 的硬标签算 BCE。于是下采样**自己**必须保住窗口内的位置/边界，
    而不是只交出"窗口内 AI 比例"这一个数。

    注意这条约束挂在**下采样路径**上（不是上采样/skip 那一侧），skip 无法替它兜底。
    """
    if not probe_logits:
        return None
    total, total_w = 0.0, 0.0
    for i, (logits, span) in enumerate(zip(probe_logits, spans)):
        target, keep = position_targets(tok_labels, int(span), ignore)     # (B, L_k, span)
        length_k = min(logits.shape[1], target.shape[1])
        logits, target, keep = logits[:, :length_k], target[:, :length_k], keep[:, :length_k]
        if keep.sum() == 0:
            continue
        loss = F.binary_cross_entropy_with_logits(logits.float(), target, weight=keep,
                                                  reduction="sum") / keep.sum()
        w = 1.0 if not weights else float(weights[i])
        total = total + w * loss
        total_w += w
    return None if total_w == 0 else total / total_w


def batch_losses(cfg: dict, sample_logits: torch.Tensor, token_logits: list[torch.Tensor] | None,
                 batch: dict, device: str, use_sample: bool = True,
                 aux: dict | None = None) -> tuple[torch.Tensor, dict]:
    """一个样本的总损失与分项（样本级 + token 级 + v0.4 辅助项；基线模型无 token 头时只算样本级）。

    Args:
        use_sample: 是否计入样本级 CE。hybrid 流的样本标签恒为 1（只保留含 AI 行的文件），
            若计入会把样本头拉向“永远输出正类”的退化解，故只对 m4（human/AI 均衡）生效。
        aux: ``UDet.forward(..., return_aux=True)`` 的第三返回值（下采样位置 probe 的 logits）。

    损失组成（v0.4）::

        L = sample·CE + token·BCE(多尺度) + cons_pool·A + cons_pos·B2
    """
    loss_cfg = cfg.get("loss", {})
    labels = batch["labels"].to(device)
    class_weights = loss_cfg.get("class_weights")
    weight = None if not class_weights else torch.tensor(class_weights, device=device, dtype=torch.float32)
    l_sample = F.cross_entropy(sample_logits.float(), labels, weight=weight)

    tok_labels = None
    l_token = None
    if token_logits is not None and loss_cfg.get("token", 1.0) > 0:
        tok_labels = batch["tok_labels"].to(device)
        l_token = token_loss(token_logits, tok_labels,
                             loss_cfg.get("scale_weights", [1.0] * len(token_logits)),
                             modes=loss_cfg.get("token_target"),
                             beta=float(loss_cfg.get("token_target_beta", 4.0)),
                             weight_mode=str(loss_cfg.get("token_weight_mode", "static")),
                             nmin=float(loss_cfg.get("token_weight_nmin", 64.0)),
                             tau=float(loss_cfg.get("token_weight_tau", 0.5)))
    parts = {"sample": float(l_sample.detach()), "token": 0.0, "cons_pool": 0.0, "cons_pos": 0.0}
    total = loss_cfg.get("sample", 1.0) * l_sample if use_sample else torch.zeros((), device=device)
    if l_token is not None:
        total = total + loss_cfg.get("token", 1.0) * l_token
        parts["token"] = float(l_token.detach())

    # ---------------- v0.4 辅助损失 ---------------- #
    if tok_labels is not None and token_logits is not None and loss_cfg.get("cons_pool", 0.0) > 0:
        l_cons = cons_pool_loss(token_logits, tok_labels,
                                weights=loss_cfg.get("cons_pool_weights"),
                                detach=bool(loss_cfg.get("cons_pool_detach", True)))
        if l_cons is not None:
            total = total + float(loss_cfg["cons_pool"]) * l_cons
            parts["cons_pool"] = float(l_cons.detach())
    if (loss_cfg.get("cons_pos", 0.0) > 0 and tok_labels is not None
            and aux and aux.get("pos_logits")):
        l_pos = position_loss(aux["pos_logits"], aux["spans"], tok_labels,
                              weights=loss_cfg.get("cons_pos_weights"))
        if l_pos is not None:
            total = total + float(loss_cfg["cons_pos"]) * l_pos
            parts["cons_pos"] = float(l_pos.detach())
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


def stream_strides(cfg: dict, model: nn.Module) -> dict:
    """按流覆盖窗口步长（只有 `WindowedContextClassifier` 支持）。

    样本级流（m4）不需要重叠 ⇒ 用 stride=window 可把窗口数减半；
    token 级流（hybrid）需要接缝平滑 ⇒ 用 window//2。
    模型不支持时返回空 dict，调用方据此决定要不要传 ``stride=``。
    """
    if not getattr(model, "accepts_stride", False):
        return {}
    bcfg = cfg.get("model", {}).get("baseline") or {}
    return dict(bcfg.get("stride_by_stream") or {})


@torch.no_grad()
def evaluate(model: nn.Module, dataset, name: str, cfg: dict, device: str,
             dump_path: str | None = None) -> dict:
    """整段序列逐样本评测（不截断、不滑窗）：m4 出样本级指标；hybrid 出行级/token/片段级指标。

    ``dump_path`` 非空时，额外把**逐样本的原始输出**存成 ``.pt``（见函数末尾）：
    长度 / 标签 / 样本概率 / 逐行概率 / 逐 token 概率。这样后续的**阈值扫描、长度分桶、
    错误分析**都能离线做，不必再占 GPU 重跑一遍评测。
    """
    model.eval()
    amp = cfg["train"].get("amp", True) and device == "cuda"
    is_hybrid = name == "hybrid"
    strides = stream_strides(cfg, model)                          # 按流覆盖窗口步长（空=不传）

    sample_logits = torch.zeros(len(dataset), 2)
    token_hits = [[0, 0, 0, 0] for _ in range(len(dataset))]       # tp/fp/fn/tn
    line_sum = [None] * len(dataset)
    line_cnt = [None] * len(dataset)
    lengths = [0] * len(dataset)                                  # 原始 token 数（长度分桶用）
    line_prob = [None] * len(dataset)
    token_probs = [None] * len(dataset)
    token_labels = [None] * len(dataset)
    has_token = True

    for i in tqdm(range(len(dataset)), desc=f"eval {name}", leave=False):
        item = dataset[i]                                          # 已含报告前缀 / token 标签
        ids = torch.tensor([item["input_ids"]], dtype=torch.long, device=device)
        extra = {"stride": strides.get(name)} if strides else {}
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            out = model(ids, torch.ones_like(ids), **extra, report=report_tensor(item, device))
        logits_sample, token_logits = out[0], out[1]
        has_token = token_logits is not None
        sample_logits[i] += logits_sample[0].float().cpu()
        lengths[i] = int(ids.shape[1])
        if not has_token or not is_hybrid:
            continue
        probs = torch.sigmoid(token_logits[-1].float())[0, 0].cpu().tolist()   # 最细尺度 = L
        if dump_path:                                             # 原始逐 token 概率与标签
            token_probs[i] = torch.tensor(probs, dtype=torch.float16)
            token_labels[i] = torch.tensor(item["tok_labels"], dtype=torch.int16)
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
            if dump_path:
                line_prob[i] = torch.tensor(
                    [line_sum[i].get(k, 0.0) / max(line_cnt[i].get(k, 0), 1)
                     for k in range(len(line_label))], dtype=torch.float16)
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

    if dump_path:                                     # ★ 原始逐样本数据落盘（离线分析用）
        blob = {
            "name": name,
            "n": len(dataset),
            "length": torch.tensor(lengths, dtype=torch.int32),
            "label": torch.tensor([dataset.labels[i] for i in range(len(dataset))],
                                  dtype=torch.int8),
            "sample_prob": torch.softmax(sample_logits, dim=-1)[:, 1],   # 类别 1 的概率
        }
        if is_hybrid and has_token:
            blob["line_label"] = [torch.tensor(dataset.line_label[i], dtype=torch.int8)
                                  for i in range(len(dataset))]
            blob["line_prob"] = line_prob
            blob["token_prob"] = token_probs          # 最细尺度（= 原始 token 数）的逐 token 概率
            blob["token_label"] = token_labels
        out_path = Path(dump_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(blob, out_path)
        print(f"[eval] 逐样本原始数据已存：{out_path}（n={blob['n']}）")
    return metrics


# --------------------------------------------------------------------------- #
# 训练
# --------------------------------------------------------------------------- #
def make_report(cfg: dict, tokenizer):
    """按 ``cfg["report"]`` 构建报告对象（v0.4.6：额外支持 mode / stats_mean / stats_std）。

    ★ **唯一入口** —— train 与各诊断脚本都必须走这里。
      否则 `report.mode=vector` 时标准化常数会对不上：模型会拿到**未标准化**的向量，
      而**不会报任何错**（这正是本项目的 B 类陷阱）。
    """
    rcfg = dict(cfg.get("report", {}))
    name = rcfg.pop("name", "handcrafted")
    rcfg.pop("mode", None)                    # mode 由 dataset 读，不传给报告类
    kwargs = {"max_tokens": int(rcfg.pop("max_tokens", 64))}
    for k in ("precision", "stats_mean", "stats_std"):
        if rcfg.get(k) is not None:
            kwargs[k] = rcfg[k]
    return build_report(name, tokenizer=tokenizer, **kwargs)


def report_tensor(item: dict, device: str):
    """从 dataset item 取报告向量并转成 (1, d) 张量；没有（prefix/none 模式）则返回 None。"""
    r = item.get("report")
    return None if r is None else torch.tensor([r], dtype=torch.float32, device=device)


def make_dataset(cfg: dict, name: str, split: str, report, train: bool, limit: int | None):
    data_cfg = cfg["data"]
    return limit_dataset(
        build_dataset(
            name,
            file=str(resolve(data_cfg["processed_dir"]) / data_cfg[f"{name}_file"]),
            split=split,
            report=report,
            train=train,
            report_mode=cfg.get("report", {}).get("mode", "prefix"),
        ),
        limit,
    )


def train(cfg: dict, args) -> None:
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    setup_cuda()
    torch.manual_seed(cfg["train"].get("seed", 0))
    tokenizer = AutoTokenizer.from_pretrained(str(resolve(cfg["encoder"]["path"])))
    report = make_report(cfg, tokenizer)
    model, encoder = build_model(cfg, tokenizer)

    # ---- 续跑（补 epoch 用）：只载模型权重，优化器 / LR 计划重新起 ----
    resume_epoch, resume_score = -1, -1.0
    if args.resume:
        ckpt = torch.load(resolve(args.resume), map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(ckpt.get("state", {}), strict=False)
        if missing or unexpected:
            print(f"[resume] 注意：未匹配 {len(missing)} 项（如 {missing[:3]}）、多余 {len(unexpected)} 项")
        resume_epoch = int(ckpt.get("epoch", -1))
        resume_score = float(ckpt.get("score", -1.0))
        print(f"[resume] 载入 {args.resume}（已训到 epoch {resume_epoch}，monitor={resume_score:.4f}）"
              f"，本次追加 {cfg['train']['epochs']} 个 epoch")
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
    append_log = bool(args.resume) and (run_dir / "metrics.csv").exists()   # 续跑：保留原 csv 追加
    log_file = open(run_dir / "metrics.csv", "a" if append_log else "w", newline="", encoding="utf-8")
    logger = csv.writer(log_file)
    if not append_log:
        logger.writerow(["epoch", "step", "loss", "loss_sample", "loss_token",
                         "loss_cons_pool", "loss_cons_pos", "lr", "sec"])

    amp = cfg["train"].get("amp", True) and device == "cuda"
    monitor = cfg["train"].get("monitor", "mean")
    # 只有启用位置探针（B2）时才额外走 return_down 分支，其余情况与 v0.3 完全相同
    want_aux = float(cfg.get("loss", {}).get("cons_pos", 0.0) or 0.0) > 0 and getattr(model, "accepts_aux", False)
    strides = stream_strides(cfg, model)                          # 按流覆盖窗口步长（空=不传）
    if strides:
        print(f"[model] 按流覆盖窗口步长：{strides}")
    if want_aux:
        print("[loss] 启用下采样位置探针（B2）；以及："
              f"cons_pool={cfg['loss'].get('cons_pool', 0.0)} cons_pos={cfg['loss'].get('cons_pos', 0.0)} "
              f"token_target={cfg['loss'].get('token_target')}")
    tokens_seen = 0
    best = resume_score                                       # 续跑时不把已存的好权重覆盖掉
    epochs = int(cfg["train"]["epochs"])
    start_epoch = resume_epoch + 1
    print(f"[train] streams={streams} steps/epoch={steps_per_epoch} accum={accum} "
          f"updates/epoch={updates_per_epoch} epochs={epochs}"
          f"（epoch {start_epoch}…{start_epoch + epochs - 1}）amp={amp}")

    for offset in range(epochs):
        epoch = start_epoch + offset
        model.train()
        if frozen:
            model.encoder.eval()                           # 冻结编码器保持 eval（关掉其 dropout）
        iters = {name: cycle(loader) for name, loader in loaders.items()}
        running = {"loss": 0.0, "sample": 0.0, **{k: 0.0 for k in LOSS_KEYS}}
        warned_nograd: set[str] = set()
        start_time = time.time()
        opt.zero_grad(set_to_none=True)
        bar = tqdm(range(steps_per_epoch), desc=f"epoch {epoch}", unit="it")
        for step in bar:
            step_parts = {"sample": 0.0, **{k: 0.0 for k in LOSS_KEYS}}
            for name in streams:                           # 两流各取一个样本，同时参与
                batch = next(iters[name])
                batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
                extra = {"return_aux": True} if want_aux else {}
                if strides:
                    extra["stride"] = strides.get(name)           # 样本级流不需要重叠
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                    out = model(batch["input_ids"], batch.get("attention_mask"), **extra,
                                report=batch.get("report"))
                    loss, parts = batch_losses(cfg, out[0], out[1], batch, device,
                                               use_sample=name in sample_streams,
                                               aux=out[2] if len(out) > 2 else None)
                if loss.requires_grad:
                    (loss / (accum * len(streams))).backward()          # 归一到全部流的样本均值
                else:
                    # v0.4.3 单任务消融会走到这里：该流既不在 sample_streams、又没有 token 头
                    # ⇒ batch_losses 返回常量 zeros，无 grad_fn。
                    # ★ 注意保留除法的 len(streams) 因子：不 backwar d 与"除以 2"是两件事，
                    #   前者只是跳过零梯度，后者才是梯度尺度 ⇒ 这样另一流的尺度与 v0.4.2 完全一致。
                    if name not in warned_nograd:
                        warned_nograd.add(name)
                        print(f"[loss] 流 {name} 本步无可回传损失（无 token 头且样本级不计该流）"
                              f"⇒ 跳过 backward；len(streams) 因子保留以维持另一个流的梯度尺度")
                for key in LOSS_KEYS:
                    step_parts[key] += parts[key]
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
            running["loss"] += step_parts["sample"] + sum(step_parts[k] for k in LOSS_KEYS)
            if (step + 1) % cfg["train"].get("log_every", 20) == 0 or step == steps_per_epoch - 1:
                elapsed = time.time() - start_time
                done = step + 1
                bar.set_postfix(loss=f"{running['loss'] / done:.3f}",
                                sample=f"{running['sample'] / done:.3f}",
                                token=f"{running['token'] / done:.3f}",
                                cons=f"{running['cons_pool'] / done:.3f}/{running['cons_pos'] / done:.3f}",
                                lr=f"{scheduler.get_last_lr()[0]:.2e}",
                                tok_s=f"{tokens_seen / max(elapsed, 1e-6):.0f}")
                logger.writerow([epoch, done, f"{running['loss'] / done:.6f}",
                                 f"{running['sample'] / done:.6f}",
                                 f"{running['token'] / done:.6f}",
                                 f"{running['cons_pool'] / done:.6f}",
                                 f"{running['cons_pos'] / done:.6f}",
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
        score = _monitor_score(results, monitor, sample_streams)
        train_avg = {k: v / max(steps_per_epoch, 1) for k, v in running.items()}
        with open(run_dir / "metrics.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"epoch": epoch, "train": train_avg, "val": results}) + "\n")
        # 只存可训练的编码器参数（冻结底座/LUT buffer 不入库，checkpoint 保持小体积）
        trainable = {name for name, p in model.named_parameters() if p.requires_grad}
        state = {k: v for k, v in model.state_dict().items()
                 if not k.startswith("encoder.") or k in trainable}
        # 每轮都存 last.pt（便于反复续跑）；best.pt 直接复制，不再重复序列化 475MB
        torch.save({"cfg": cfg, "epoch": epoch, "score": score, "metrics": results, "state": state},
                   run_dir / "last.pt")
        if score > best:
            best = score
            shutil.copyfile(run_dir / "last.pt", run_dir / "best.pt")
            print(f"[ckpt] 保存 best.pt（{monitor}={score:.4f}）")

    log_file.close()
    print(f"[train] 完成，产物目录：{run_dir}")


def _monitor_score(results: dict, monitor: str, sample_streams=None) -> float:
    """选 best.pt 用的综合分。

    ``mean`` 只统计**有意义的**指标：样本级 F1 只取 `sample_streams` 里的流
   （hybrid 样本标签恒为 1，它的 sample_f1 随模型变好反而下降，会把好 epoch 压下去），
    再加上各流的 line_f1。
    """
    if monitor == "m4_f1":
        return results.get("m4", {}).get("sample_f1", 0.0)
    if monitor == "line_f1":
        return results.get("hybrid", {}).get("line_f1", 0.0)
    keep = set(sample_streams) if sample_streams else set(results)
    values = [v.get("sample_f1", 0.0) for k, v in results.items() if k in keep]
    values += [v.get("line_f1", 0.0) for v in results.values() if "line_f1" in v]
    return sum(values) / max(len(values), 1)


# --------------------------------------------------------------------------- #
# 评测入口
# --------------------------------------------------------------------------- #
def run_eval(cfg: dict, args) -> None:
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    setup_cuda()
    tokenizer = AutoTokenizer.from_pretrained(str(resolve(cfg["encoder"]["path"])))
    report = make_report(cfg, tokenizer)
    model, _ = build_model(cfg, tokenizer)
    ckpt = torch.load(resolve(args.ckpt), map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(ckpt["state"], strict=False)
    if missing or unexpected:
        print(f"[eval] 注意：未加载 {len(missing)} 项（如 {missing[:3]}）、多余 {len(unexpected)} 项")
    model.to(device)
    print(f"[eval] 载入 {args.ckpt}（epoch {ckpt.get('epoch')}），配置来自当前 yaml/命令行")
    # ★ 防呆：评测配置必须与**训练时**的配置一致，否则会**静默**产出垃圾结果。
    #   实测踩过（2026-09-20）：用 v0.4.1 的 config（编码器 codet5lora，逐 token）
    #   去评测 v0.4.2 的权重（编码器 codet5blk，分块），编码器语义完全不同，
    #   m4 从 0.97 掉到 0.67、line 从 0.79 掉到 0.01 —— 而 state_dict 的键基本对齐，
    #   所以**不报任何错**。这里比对训练产物目录里的 config.yaml，不一致直接中止。
    #   只比对决定「前向语义」的段；loss/train 段不影响评测，不查。
    #   数学无关的键（分块大小/是否静态/compile/ckpt/分块前向的 chunk）允许不同。
    trained_cfg_path = Path(resolve(args.ckpt)).parent / "config.yaml"
    if trained_cfg_path.exists():
        with open(trained_cfg_path, encoding="utf-8") as f:
            trained_cfg = yaml.safe_load(f) or {}
        neutral = {"block_batch", "static", "compile", "ckpt", "chunk"}
        diffs = []
        for sec in ("encoder", "model", "heads", "report"):
            old, new = (trained_cfg.get(sec) or {}), (cfg.get(sec) or {})
            for key in sorted(set(old) | set(new)):
                if key in neutral:
                    continue
                if old.get(key) != new.get(key):
                    diffs.append(f"{sec}.{key}: 训练={old.get(key)!r}  评测={new.get(key)!r}")
        if diffs:
            print("[eval] ⚠⚠ 评测配置与训练配置不一致 —— 这会静默产出错误结果，已中止：")
            for d in diffs[:15]:
                print(f"          {d}")
            if len(diffs) > 15:
                print(f"          ...（共 {len(diffs)} 处）")
            print(f"[eval]  训练配置：{trained_cfg_path}")
            raise SystemExit(2)
        print(f"[eval] 配置一致性检查通过（比对自 {trained_cfg_path}）")
    result = {}
    for name in cfg["train"]["streams"]:
        for split in ("val", "test"):
            dataset = make_dataset(cfg, name, split, report, False, args.limit)
            if len(dataset) == 0:
                continue
            dump = None
            if getattr(args, "dump_raw", False):
                dump = str(Path(resolve(args.ckpt)).parent / f"raw_{name}_{split}.pt")
            metrics = evaluate(model, dataset, name, cfg, device, dump_path=dump)
            result[f"{name}/{split}"] = metrics
            print(f"[eval] {name}/{split}: " + json.dumps({k: round(v, 4) for k, v in metrics.items()}))
    # ★ 防呆：带 --limit 的是**冒烟评测**，绝不能覆盖正式的 eval.json。
    #   实测踩过（2026-09-21）：为验证配置一致性防呆而跑的一条 `--eval --limit 3`，
    #   把 v0.4.4 正式的 eval.json 与 raw_*.pt 全部覆盖成了 3 样本版本（m4 显示 1.0000），
    #   而它**不报任何错**。现在冒烟结果写到 eval_limit<N>.json。
    suffix = f"_limit{args.limit}" if args.limit else ""
    out = Path(resolve(args.ckpt)).parent / f"eval{suffix}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"ckpt": str(args.ckpt), "epoch": ckpt.get("epoch"), "metrics": result},
                  f, ensure_ascii=False, indent=2)
    if args.limit:
        print(f"[eval] ⚠ 检测到 --limit {args.limit}（冒烟评测）⇒ 结果写入 {out.name}，"
              f"**不会覆盖**正式的 eval.json")
    print(f"[eval] 结果已写入 {out}")


def main() -> int:
    parser = argparse.ArgumentParser(description="U-Det 训练/评测（窗口注意力 U-Net + 双任务）")
    parser.add_argument("--config", default="configs/udet_base.yaml")
    parser.add_argument("--tag", default="udet-base")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None, help="每个数据集只用前 N 条（冒烟）")
    parser.add_argument("--no-m4", action="store_true")
    parser.add_argument("--no-hybrid", action="store_true")
    parser.add_argument("--report", default=None, help="覆盖 report.name（none/handcrafted）")
    parser.add_argument("--model", default=None, help="覆盖 model.name（codec / hier / codet5cls）")
    parser.add_argument("--encoder", default=None, help="覆盖 encoder.name（codet5blk / codet5lora / codet5tok / codet5）")
    parser.add_argument("--layer", type=int, default=None, help="覆盖 encoder.layer（-1 最后一层 / 0 词嵌入层）")
    parser.add_argument("--block-batch", type=int, default=None,
                        help="覆盖 encoder.block_batch（分块编码器每次前向并几块，越小越省显存）")
    parser.add_argument("--freeze-encoder", action="store_true", help="冻结编码器")
    parser.add_argument("--unfreeze-encoder", action="store_true", help="解冻编码器（微调）")
    parser.add_argument("--no-cons", action="store_true", help="v0.4：关掉 cons_pool + cons_pos")
    parser.add_argument("--token-target", default=None,
                        help="v0.4：覆盖 loss.token_target（mean / max / lse，全尺度统一）")
    parser.add_argument("--no-gate", action="store_true", help="v0.4：关掉跳跃融合门控")
    parser.add_argument("--resume", default=None,
                        help="从该 checkpoint 续跑（只载模型权重；--epochs 变成“本次追加几个 epoch”）")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--dump-raw", action="store_true",
                        help="评测时把逐样本原始输出（长度/标签/样本概率/逐行概率/逐 token 概率）"
                             "存到 runs/<tag>/raw_<流>_<split>.pt，供离线做阈值扫描与分桶分析")
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
