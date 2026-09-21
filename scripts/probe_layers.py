"""零训练探针 P5~P7：**逐层编码器信息量** —— 决定「编码器尾部删几层搬到瓶颈」。

问：把 CodeT5 编码器**尾部 k 层删掉**、把那 k 层搬到主干瓶颈，代价是多少？k 取 2 / 3 / 4？

为什么不能直接试：删+搬是一条**双变量**改动（编码器少 k 层、瓶颈多 k 层），
按 lessons A7 结果不可归因；而且「小步快跑 2 epoch」的经验在本项目**七次全错**
（lessons A1）—— 架构对比必须对齐 epoch 预算，一轮就是 2 小时。
所以先用**零训练探针**把「删」的代价量出来，再决定要不要花那 2 小时。

三段，全部零训练：

  P5  逐 **token** 线性探针：用 hybrid 的 token 标签，在**每一层**的逐 token 特征上
      拟合一个线性分类器（train 拟合 → test 评测）。
      ⇒ 「这一层还剩多少**可线性解码**的 token 级信息」。
      ⇒ **截断到第 k 层的代价 = P5(12) − P5(k)**，这是「删」的代价的上界。
  P6  逐 **文档** 线性探针：m4 上按样本池化后拟合线性分类器。做**两种分辨率**：
      (a) 全分辨率平均池化（= 编码器输出的自然读数，主干能拿到的信息总量）；
      (b) **瓶颈分辨率**（先降到 ceil(L/64) 再平均，与主干最粗一级的带宽一致）。
      ⇒ (b) 回答「搬到瓶颈有没有用」：若在 L/64 分辨率下深层**早已不再涨**，
        说明瓶颈是**带宽**受限而不是**层数**不足 ⇒ 往瓶颈堆层无益。
  P7  cut-and-stitch 实测（``--stage stitch``）：把**训练好的整条主干与头**直接接在
      第 k 层特征上，报真实的 test 指标。
      ⇒ 给出「只删、不搬、且**不重训**」的**下界**（悲观值）。
      ⇒ k=0（= 层 12，原样）必须与官方 ``eval.json`` 一致 —— 这是内建自检。

读法（P5/P6 的边际列才是决策依据）：
    某一层上的探针值几乎不涨 ⇒ 那一层对下游**没贡献可线性解码的信息** ⇒ 删掉便宜。
    若 8→9→10→11→12 五层平坦，则删 4 层（截断到层 8）代价很小，实验值得跑；
    若从某一层开始明显上升，则**最多删到那一层之前**。

已知局限（必须写进结论里）：
  * 线性探针测的是**信息是否还在**，不是「训练好的主干能不能用上」。真实主干是
    非线性的 4 层瓶颈 + 上采样，所以探针给出的 Δ 是**代价的上界**（可能高估损失）。
  * 反之 P7 的 stitch 是**下界**（主干没见过第 8 层的特征分布，属 OOD 输入）。
    两个方向夹住真值，正是本探针的设计意图。
  * 删层会连带删掉这些层的 LoRA 适配器；而「搬」到瓶颈可以把 LoRA 一起搬过去
    （维度完全相同 768/12 头/ratio 4）。这一点探针量不了，只能在文档里写明。

用法::

    # ① 探针（认信息量）：一次前向 × 2 split，约 10~20 分钟
    OMP_NUM_THREADS=8 python scripts/probe_layers.py --stage probe \
        --ckpt runs/v0.4.5/best.pt --config configs/udet_v045.yaml

    # ② cut-and-stitch 实测：每个 k 一遍 test，约 3~4 分钟/k
    OMP_NUM_THREADS=8 python scripts/probe_layers.py --stage stitch \
        --ckpt runs/v0.4.5/best.pt --config configs/udet_v045.yaml --ks 0 2 3 4

结果落盘：``runs/<tag>/probe_layers.json``（P5/P6）与 ``probe_stitch.json``（P7）。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train import (IGNORE, build_model, chunk_match, evaluate, make_dataset,  # noqa: E402
                   make_report, prf1, resolve, setup_cuda)

BUCKETS = [(0, 512), (512, 2048), (2048, 8192), (8192, 1 << 30)]


# --------------------------------------------------------------------------- #
# 0. 工具：AUC / F1 / 分桶 / 流式岭回归
# --------------------------------------------------------------------------- #
def auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """精确 AUC（Mann-Whitney），并列取平均秩。``labels`` 为 0/1。"""
    scores = scores.double()
    labels = labels.bool()
    npos = int(labels.sum())
    nneg = int(labels.numel() - npos)
    if npos == 0 or nneg == 0:
        return float("nan")
    uniq, inv, cnt = torch.unique(scores, return_inverse=True, return_counts=True)
    cum = cnt.cumsum(0).double()
    rank_mean = (cum - cnt.double() + cum + 1.0) / 2.0        # 并列值的平均秩
    ranks = rank_mean[inv]
    return float((ranks[labels].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def prf(scores: torch.Tensor, labels: torch.Tensor, thr: float) -> dict:
    pred = scores > thr
    gold = labels.bool()
    tp = float((pred & gold).sum())
    fp = float((pred & ~gold).sum())
    fn = float((~pred & gold).sum())
    p = tp / max(tp + fp, 1e-9)
    r = tp / max(tp + fn, 1e-9)
    return {"p": p, "r": r, "f1": 2 * p * r / max(p + r, 1e-9)}


def prior_threshold(scores: torch.Tensor, p_pos: float) -> float:
    """**不偷看标签**的阈值：让判正比例等于训练集正类比例（分位数校准）。

    线性探针输出的是「±1 回归值」，0 并不是类平衡点；本项目的 ``prob > 0.5``
    约定对应的是「半数」，在类别极不平衡的 token 级上完全不可用。
    这里取 test 分数的 (1−p_pos) 分位，只用测试集**无标签**的分数分布 ⇒ 无泄漏。
    """
    if not 0.0 < p_pos < 1.0:
        return 0.0
    return float(torch.quantile(scores.double().flatten().cpu().float(), 1.0 - p_pos))


def fmt(x: float, width: int = 16, digits: int = 4) -> str:
    """nan（单类别 ⇒ AUC 无定义）显示成 --，避免拿 nan 下结论。"""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return f"{'--':>{width}}"
    return f"{x:>{width}.{digits}f}"


def bucket_of(length: int) -> int:
    for b, (lo, hi) in enumerate(BUCKETS):
        if lo <= length < hi:
            return b
    return -1


class RidgeStream:
    """流式岭回归（闭式解）：只累积 ``Σx / Σxxᵀ / Σx·y / n``，**永不存特征**。

    为什么能只靠累积量算标准化：``z = (x−μ)/σ`` 是 x 的**仿射**变换，
    而 μ 取训练集精确均值 ⇒ ``Σz ≡ 0`` ⇒ 偏置项与 z 正交、截距就是 ȳ。于是

        Σzzᵀ = A M A,   M = Σxxᵀ − (Σx)μᵀ − μ(Σx)ᵀ + nμμᵀ,   A = diag(1/σ)
        Σz·y = A (Σx·y − μ Σy)

    这是一次前向就能跑完整个 train 的关键（7554 万 token × 768 维存不下）。
    """

    def __init__(self, dim: int):
        self.dim = dim
        self.n = 0
        self.npos = 0
        self.nneg = 0
        self.sy = 0.0
        self.G = None
        self.sx = None
        self.sxy = None

    @torch.no_grad()
    def add(self, x: torch.Tensor, y: torch.Tensor) -> None:
        """x (n, dim)；y (n,) 取值 ±1。矩阵乘在 fp32（快），累加在 fp64（稳）。"""
        xf = x.float()
        yf = y.float()
        g = (xf.t() @ xf).double()
        s = xf.sum(0).double()
        if self.G is None:
            self.G = torch.zeros_like(g)
            self.sx = torch.zeros_like(s)
            self.sxy = torch.zeros_like(s)
        self.G += g
        self.sx += s
        self.sxy += (xf.t() @ yf).double()
        self.sy += float(yf.sum())
        self.n += int(xf.shape[0])
        self.npos += int((yf > 0).sum())
        self.nneg += int((yf < 0).sum())

    @torch.no_grad()
    def fit(self, lam: float = 1e-2) -> dict:
        G = self.G.cpu().double() if self.G is not None else torch.zeros(self.dim, self.dim,
                                                                        dtype=torch.float64)
        sx = self.sx.cpu().double() if self.sx is not None else torch.zeros(self.dim,
                                                                           dtype=torch.float64)
        sxy = self.sxy.cpu().double() if self.sxy is not None else torch.zeros(self.dim,
                                                                              dtype=torch.float64)
        n = max(self.n, 1)
        mu = sx / n
        var = (G.diagonal() / n - mu * mu).clamp(min=1e-12)
        sd = var.sqrt()
        M = G - torch.outer(sx, mu) - torch.outer(mu, sx) + n * torch.outer(mu, mu)
        ZtZ = M / sd[:, None] / sd[None, :]
        q = (sxy - mu * self.sy) / sd
        eye = torch.eye(self.dim, dtype=torch.float64)
        w = torch.linalg.solve(ZtZ + lam * eye, q)
        return {
            "mu": mu, "sd": sd, "w": w, "b": self.sy / n,
            "n": self.n, "p_pos": self.npos / max(self.npos + self.nneg, 1),
            "mean_sq": float(G.diagonal().sum() / n),      # E‖x‖²：层间尺度诊断
        }

    @staticmethod
    def to_device(packed: dict, device) -> dict:
        """把权重搬到特征所在设备（fp32 足够打分；fp64 只在拟合时需要）。"""
        out = dict(packed)
        for k in ("mu", "sd", "w"):
            out[k] = packed[k].float().to(device)
        return out

    @torch.no_grad()
    def score(self, x: torch.Tensor, packed: dict) -> torch.Tensor:
        z = (x.float() - packed["mu"]) / packed["sd"]
        return z @ packed["w"] + packed["b"]


# --------------------------------------------------------------------------- #
# 1. 逐层隐状态：复刻分块编码器的分块/补齐/还原，但返回**全部** 13 层
# --------------------------------------------------------------------------- #
def _all_states(enc, ids: torch.Tensor, mask: torch.Tensor):
    """(nb, K) -> hidden_states 元组（0 = 词嵌入，i = 第 i 层；T5 无 final LN，末层即输出）。"""
    out = enc.model(input_ids=ids, attention_mask=mask,
                    output_hidden_states=True, return_dict=True)
    return out.hidden_states


@torch.no_grad()
def encoder_states(enc, input_ids: torch.Tensor, attention_mask: torch.Tensor):
    """复刻 ``CodeT5BlockEncoder.forward`` 的分块逻辑，返回 ``list[(B, L, D)]``（共 num_layers+1）。

    ★ 与 ``enc(input_ids, attention_mask)``（``layer=-1``）必须**逐位一致**，
      由 ``--selfcheck`` 验证；不一致说明这里的分块/还原逻辑与编码器漂了。
    """
    batch, length = input_ids.shape
    keep = attention_mask
    k, bb = enc.block, enc.block_batch
    nblk = (length + k - 1) // k
    if enc.static:
        nblk = -(-nblk // bb) * bb
    pad_len = nblk * k - length
    ids, mask = input_ids, attention_mask
    if pad_len:
        ids = F.pad(ids, (0, pad_len), value=enc.pad_id)
        mask = F.pad(mask, (0, pad_len), value=0)
    ids = ids.reshape(batch * nblk, k)
    mask = mask.reshape(batch * nblk, k)
    dead = mask.sum(-1) == 0
    if bool(dead.any()):
        mask = mask.clone()
        mask[dead, 0] = 1
    pieces = None
    for start in range(0, ids.shape[0], bb):
        states = _all_states(enc, ids[start:start + bb], mask[start:start + bb])
        if pieces is None:
            pieces = [[] for _ in states]
        for li, h in enumerate(states):
            pieces[li].append(h)
    outs = []
    for chunks in pieces:
        h = chunks[0] if len(chunks) == 1 else torch.cat(chunks, 0)
        h = h.reshape(batch, nblk * k, -1)[:, :length]
        outs.append(h * keep.unsqueeze(-1).to(h.dtype))
    return outs


def build(cfg: dict, ckpt: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(str(resolve(cfg["encoder"]["path"])))
    model, enc = build_model(cfg, tokenizer)
    blob = torch.load(resolve(ckpt), map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(blob["state"], strict=False)
    model.to(device).eval()
    print(f"[probe] 载入 {ckpt}（epoch {blob.get('epoch')}，缺失 {len(missing)} 项 / "
          f"多余 {len(unexpected)} 项 —— 冻结底座按路径确定性加载，属正常）")
    return model, enc, tokenizer


def selfcheck(enc, device: str) -> None:
    """分块复刻的逐位自检：states[-1] 必须与编码器自己的输出完全相等。"""
    torch.manual_seed(0)
    for length in (1, 127, 128, 129, 300):
        ids = torch.randint(5, 3000, (1, length), device=device)
        mask = torch.ones_like(ids)
        states = encoder_states(enc, ids, mask)
        ref = enc(ids, mask)
        d = float((states[-1].float() - ref.float()).abs().max())
        flag = "OK " if d < 5e-5 else "!! "
        print(f"[selfcheck] L={length:>4} 层数={len(states)} max|states[-1] − enc(x)| = {d:.3e} {flag}")
        if d >= 5e-5:
            raise SystemExit("分块复刻与编码器不一致，先修 encoder_states 再跑探针")


# --------------------------------------------------------------------------- #
# 2. P5：逐 token 线性探针
# --------------------------------------------------------------------------- #
@torch.no_grad()
def probe_token(model, enc, cfg, report, device, layers, limit_train, limit_test,
                lam: float) -> dict:
    """hybrid train 拟合 → test 评测；逐层 token F1 / AUC / 行级 F1 / 片段级 F1。"""
    ds_tr = make_dataset(cfg, "hybrid", "train", report, True, limit_train or None)
    ds_te = make_dataset(cfg, "hybrid", "test", report, False, limit_test or None)
    n_layer = enc.num_layers + 1
    print(f"\n[P5] hybrid train={len(ds_tr)} / test={len(ds_te)}，层 0..{n_layer - 1}（0=词嵌入）")

    acc = [RidgeStream(enc.hidden_size) for _ in range(n_layer)]
    t0 = time.time()
    for i in range(len(ds_tr)):
        item = ds_tr[i]
        ids = torch.tensor([item["input_ids"]], dtype=torch.long, device=device)
        mask = torch.ones_like(ids)
        states = encoder_states(enc, ids, mask)
        tok = torch.tensor([item["tok_labels"]], dtype=torch.long, device=device)
        valid = (tok != IGNORE).reshape(-1)
        y = torch.where(tok.reshape(-1)[valid] == 1, 1.0, -1.0)
        for li in layers:
            acc[li].add(states[li][0][valid], y)
        if (i + 1) % 200 == 0:
            print(f"   train [{i + 1}/{len(ds_tr)}] {time.time() - t0:.0f}s", flush=True)
    print(f"   train 完成 {time.time() - t0:.0f}s")

    packed = {li: RidgeStream.to_device(acc[li].fit(lam), device) for li in layers}

    # test：逐层算分，并顺带做行级 / 片段级聚合
    scores = {li: [] for li in layers}          # 每条样本的 (n,) 分数
    line_of = []                                # 每条样本的 line_of_token（用于行级）
    line_label = []
    lengths = []
    t0 = time.time()
    for i in range(len(ds_te)):
        item = ds_te[i]
        ids = torch.tensor([item["input_ids"]], dtype=torch.long, device=device)
        mask = torch.ones_like(ids)
        states = encoder_states(enc, ids, mask)
        tok = torch.tensor([item["tok_labels"]], dtype=torch.long, device=device)
        valid = (tok != IGNORE).reshape(-1).cpu()
        for li in layers:
            scores[li].append(acc[li].score(states[li][0][valid.to(device)], packed[li]).cpu())
        line_of.append(torch.tensor(item["line_of_token"], dtype=torch.long)[valid])
        line_label.append(list(item["line_label"]))
        lengths.append(int(valid.sum()))
        if (i + 1) % 200 == 0:
            print(f"   test  [{i + 1}/{len(ds_te)}] {time.time() - t0:.0f}s", flush=True)

    gold_tok_all = {li: [] for li in layers}
    for i in range(len(ds_te)):
        item = ds_te[i]
        tok = torch.tensor([item["tok_labels"]], dtype=torch.long).reshape(-1)
        v = tok != IGNORE
        gold = torch.where(tok[v] == 1, 1, 0)
        for li in layers:
            gold_tok_all[li].append(gold)

    out = {}
    for li in layers:
        sc = torch.cat(scores[li])
        y = torch.cat(gold_tok_all[li])
        p = packed[li]
        thr = prior_threshold(sc, p["p_pos"])
        row = {
            "n_token": int(sc.numel()),
            "auc": auc(sc, y),
            "thr": thr,
            "f1": prf(sc, y, thr)["f1"],
            "f1_thr0": prf(sc, y, 0.0)["f1"],
            "mean_sq": p["mean_sq"],
        }
        # 行级 / 片段级：把逐 token 分数按 line_of_token 平均
        pred_lines, gold_lines = [], []
        chunk = [0, 0, 0]
        for i in range(len(ds_te)):
            lo = line_of[i]
            sc_i = scores[li][i]
            n_line = len(line_label[i])
            ssum = torch.zeros(n_line, dtype=torch.float64)
            scnt = torch.zeros(n_line, dtype=torch.float64)
            ssum.scatter_add_(0, lo, sc_i.double())
            scnt.scatter_add_(0, lo, torch.ones_like(sc_i).double())
            picked = [int(ssum[k] / max(scnt[k].item(), 1) > thr) for k in range(n_line)]
            pred_lines.extend(picked)
            gold_lines.extend(line_label[i])
            tp, npred, ngold = chunk_match(picked, line_label[i])
            chunk[0] += tp
            chunk[1] += npred
            chunk[2] += ngold
        row["line_f1"] = prf1(pred_lines, gold_lines)["f1"]
        cp = chunk[0] / max(chunk[1], 1e-9)
        cr = chunk[0] / max(chunk[2], 1e-9)
        row["chunk_f1"] = 2 * cp * cr / max(cp + cr, 1e-9)
        out[li] = row
        print(f"   layer {li:>2}  auc={row['auc']:.4f}  f1={row['f1']:.4f} "
              f"（thr0 时 {row['f1_thr0']:.4f}）  line={row['line_f1']:.4f} "
              f"chunk={row['chunk_f1']:.4f}  E‖x‖²={row['mean_sq']:.1f}")
    return {"layers": layers, "rows": out, "n_train": len(ds_tr), "n_test": len(ds_te)}


# --------------------------------------------------------------------------- #
# 3. P6：逐文档线性探针（全分辨率 + 瓶颈分辨率）
# --------------------------------------------------------------------------- #
def pool_mean(h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """(1, L, D) -> (1, D)：掩码平均池化。"""
    m = mask.unsqueeze(-1).to(h.dtype)
    return (h * m).sum(1) / m.sum(1).clamp(min=1.0)


def pool_stride(h: torch.Tensor, mask: torch.Tensor, stride: int = 64) -> torch.Tensor:
    """(1, L, D) -> (1, D)：先降到 ceil(L/stride) 个位置（不重叠窗口均值），再对位置取均值。

    这模拟的是**主干最粗一级的带宽**（``levels[0]`` 跨度 64）：瓶颈只能看到约 L/64 个位置。
    """
    length = h.shape[1]
    n = max(1, math.ceil(length / stride))
    pad = n * stride - length
    hm = h * mask.unsqueeze(-1).to(h.dtype)
    mk = mask
    if pad:
        hm = F.pad(hm, (0, 0, 0, pad))
        mk = F.pad(mk, (0, pad))
    hm = hm.reshape(1, n, stride, -1).sum(2)
    mk = mk.reshape(1, n, stride).sum(2)
    return (hm / mk.clamp(min=1.0).unsqueeze(-1)).mean(1)


@torch.no_grad()
def probe_doc(model, enc, cfg, report, device, layers, limit_train, limit_test, lam: float) -> dict:
    """m4 train 拟合 → test 评测；逐层 doc AUC / acc / F1，两种池化分辨率，按长度分桶。"""
    ds_tr = make_dataset(cfg, "m4", "train", report, True, limit_train or None)
    ds_te = make_dataset(cfg, "m4", "test", report, False, limit_test or None)
    n_layer = enc.num_layers + 1
    print(f"\n[P6] m4 train={len(ds_tr)} / test={len(ds_te)}，层 0..{n_layer - 1}")

    acc = {(li, p): RidgeStream(enc.hidden_size) for li in layers for p in ("full", "s64")}
    t0 = time.time()
    for i in range(len(ds_tr)):
        item = ds_tr[i]
        ids = torch.tensor([item["input_ids"]], dtype=torch.long, device=device)
        mask = torch.ones_like(ids)
        states = encoder_states(enc, ids, mask)
        y = torch.tensor([1.0 if item["label"] == 1 else -1.0], device=device)
        for li in layers:
            h = states[li]
            acc[(li, "full")].add(pool_mean(h, mask), y)
            acc[(li, "s64")].add(pool_stride(h, mask), y)
        if (i + 1) % 1000 == 0:
            print(f"   train [{i + 1}/{len(ds_tr)}] {time.time() - t0:.0f}s", flush=True)
    print(f"   train 完成 {time.time() - t0:.0f}s")
    packed = {k: RidgeStream.to_device(v.fit(lam), device) for k, v in acc.items()}

    sc = {k: [] for k in packed}
    labels, lengths = [], []
    for i in range(len(ds_te)):
        item = ds_te[i]
        ids = torch.tensor([item["input_ids"]], dtype=torch.long, device=device)
        mask = torch.ones_like(ids)
        states = encoder_states(enc, ids, mask)
        labels.append(item["label"])
        lengths.append(int(ids.shape[1]))
        for li in layers:
            for p in ("full", "s64"):
                v = pool_mean(states[li], mask)[0] if p == "full" else pool_stride(states[li], mask)[0]
                sc[(li, p)].append(float(acc[(li, p)].score(v.unsqueeze(0), packed[(li, p)])[0]))
        if (i + 1) % 500 == 0:
            print(f"   test  [{i + 1}/{len(ds_te)}] ", flush=True)

    y = torch.tensor(labels)
    L = torch.tensor(lengths)
    bidx = torch.tensor([bucket_of(int(v)) for v in lengths])
    out = {}
    for li in layers:
        row = {}
        for p in ("full", "s64"):
            s = torch.tensor(sc[(li, p)])
            thr = prior_threshold(s, packed[(li, p)]["p_pos"])
            pr = prf(s, y, thr)
            acc_ = float(((s > thr).long() == y).float().mean())
            bucket = {}
            for b, (lo, hi) in enumerate(BUCKETS):
                m = bidx == b
                if int(m.sum()) == 0:
                    continue
                s_b, y_b = s[m], y[m]
                thr_b = prior_threshold(s_b, float(y_b.float().mean()))
                bucket[f"[{lo},{'inf' if hi > (1 << 29) else hi})"] = {
                    "n": int(m.sum()),
                    "acc": float(((s_b > thr_b).long() == y_b).float().mean()),
                    "auc": auc(s_b, y_b),
                }
            row[p] = {"auc": auc(s, y), "acc": acc_, "f1": pr["f1"], "thr": thr,
                      "mean_sq": packed[(li, p)]["mean_sq"], "bucket": bucket}
        out[li] = row
        print(f"   layer {li:>2}  full: auc={row['full']['auc']:.4f} acc={row['full']['acc']:.4f}"
              f" f1={row['full']['f1']:.4f}   s64: auc={row['s64']['auc']:.4f}"
              f" acc={row['s64']['acc']:.4f}")
    return {"layers": layers, "rows": out, "n_train": len(ds_tr), "n_test": len(ds_te)}


# --------------------------------------------------------------------------- #
# 4. P7：cut-and-stitch 实测（复用 evaluate，保证指标算法与官方一致）
# --------------------------------------------------------------------------- #
class StitchedEncoder(nn.Module):
    """把真实编码器**第 layer 层**的输出当作编码器输出（删掉它之后的所有层）。

    用它替换 ``model.encoder`` 后，``train.evaluate`` 可以原样复用 ——
    指标算法、报告拼接、autocast 全都与官方评测一致。

    ``bridge`` 非空时，先把该层特征**线性桥接**回第 12 层（P7b）：
    ``h → (h − μ)/σ @ W + b``。它的作用是**消掉分布漂移**这个混淆：
    直接替换（raw）时主干没见过截断层特征的分布，度量的是「删层 + 分布漂移」；
    桥接后主干收到的向量在**均值/协方差意义上**与它训练时见过的接近，
    度量的更接近「删层本身」。因此 **raw 是悲观下界、bridge 是乐观上界**。
    """

    def __init__(self, real: nn.Module, layer: int, bridge: dict | None = None):
        super().__init__()
        self.real = real
        self.layer = int(layer)
        self.hidden_size = real.hidden_size
        self.bridge = None
        if bridge is not None:
            self.bridge = {k: bridge[k] for k in ("mu", "sd", "W", "b")}

    @property
    def num_layers(self) -> int:
        return self.layer

    def forward(self, input_ids, attention_mask=None, **kwargs):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        h = encoder_states(self.real, input_ids, attention_mask)[self.layer]
        if self.bridge is None:
            return h
        dtype = h.dtype
        z = (h.float() - self.bridge["mu"].to(h.device)) / self.bridge["sd"].to(h.device)
        out = z @ self.bridge["W"].to(h.device) + self.bridge["b"].to(h.device)
        return out.to(dtype)


class BridgeFit:
    """流式累积「第 src 层 → 第 tgt 层」的线性桥接统计量（P7b 用）。

    只用 ``Σ x_s x_sᵀ / Σ x_s x_tᵀ / Σ x_s / Σ x_t / n`` 就能闭式解出：
        ZᵀZ = A M A        （与 RidgeStream 同一套仿射技巧）
        ZᵀX_t = A (Σ x_s x_tᵀ − μ_s Σ x_tᵀ)
        W = (ZᵀZ + λI)^{-1} ZᵀX_t,   b = mean(x_t)  （因为 Σz ≡ 0）
    """

    def __init__(self, dim: int):
        self.dim = dim
        self.n = 0
        self.Gs = self.C = self.ss = self.st = None

    @torch.no_grad()
    def add(self, xs: torch.Tensor, xt: torch.Tensor) -> None:
        xs, xt = xs.float(), xt.float()
        gs = (xs.t() @ xs).double()
        c = (xs.t() @ xt).double()
        ss = xs.sum(0).double()
        st = xt.sum(0).double()
        if self.Gs is None:
            self.Gs, self.C = torch.zeros_like(gs), torch.zeros_like(c)
            self.ss, self.st = torch.zeros_like(ss), torch.zeros_like(st)
        self.Gs += gs
        self.C += c
        self.ss += ss
        self.st += st
        self.n += int(xs.shape[0])

    @torch.no_grad()
    def fit(self, lam: float = 1e-2) -> dict:
        n = max(self.n, 1)
        Gs, C = self.Gs.cpu().double(), self.C.cpu().double()
        ss, st = self.ss.cpu().double(), self.st.cpu().double()
        mu = ss / n
        sd = (Gs.diagonal() / n - mu * mu).clamp(min=1e-12).sqrt()
        M = Gs - torch.outer(ss, mu) - torch.outer(mu, ss) + n * torch.outer(mu, mu)
        ZtZ = M / sd[:, None] / sd[None, :]
        ZtXt = (C - torch.outer(mu, st)) / sd[:, None]
        eye = torch.eye(self.dim, dtype=torch.float64)
        W = torch.linalg.solve(ZtZ + lam * eye, ZtXt)
        return {"mu": mu, "sd": sd, "W": W, "b": st / n, "n": self.n}


@torch.no_grad()
def fit_bridges(enc, cfg, report, device, layers, n_layer, n_hy: int, n_m4: int,
                lam: float) -> dict:
    """在 train 集上拟合「第 ℓ 层 → 第 12 层」的线性桥接（P7b 的准备工作）。"""
    srcs = [li for li in layers if li != n_layer - 1]
    if not srcs:
        return {}
    acc = {li: BridgeFit(enc.hidden_size) for li in srcs}
    for name, limit in (("hybrid", n_hy), ("m4", n_m4)):
        if not limit:
            continue
        ds = make_dataset(cfg, name, "train", report, True, limit)
        t0 = time.time()
        for i in range(len(ds)):
            item = ds[i]
            ids = torch.tensor([item["input_ids"]], dtype=torch.long, device=device)
            states = encoder_states(enc, ids, torch.ones_like(ids))
            xt = states[n_layer - 1][0]
            for li in srcs:
                acc[li].add(states[li][0], xt)
            if (i + 1) % 400 == 0:
                print(f"   bridge-fit {name} [{i + 1}/{len(ds)}] {time.time() - t0:.0f}s", flush=True)
        print(f"   bridge-fit {name} 完成（{len(ds)} 条，{time.time() - t0:.0f}s）")
    return {li: acc[li].fit(lam) for li in srcs}


def run_stitch(model, enc, cfg, report, device, ks, limit_test,
               bridges: dict | None = None) -> dict:
    """对每个 k 报两组读数：``raw``（直接替换）与 ``bridge``（线性桥接后替换）。"""
    ds_m4 = make_dataset(cfg, "m4", "test", report, False, limit_test or None)
    ds_hy = make_dataset(cfg, "hybrid", "test", report, False, limit_test or None)
    n_layer = enc.num_layers + 1
    variants = ["raw"] + (["bridge"] if bridges else [])
    out = {}
    for k in ks:
        layer = n_layer - 1 - k                      # 删掉尾部 k 层 ⇒ 截断在第 (12−k) 层
        out[k] = {"layer": layer}
        for v in variants:
            if v == "bridge" and layer not in (bridges or {}):
                continue
            model.encoder = StitchedEncoder(enc, layer, bridges.get(layer) if v == "bridge" else None)
            t0 = time.time()
            m4 = evaluate(model, ds_m4, "m4", cfg, device)
            hy = evaluate(model, ds_hy, "hybrid", cfg, device)
            out[k][v] = {"m4": m4, "hybrid": hy}
            print(f"[P7/{v}] 去掉尾 {k} 层（截断到第 {layer} 层）："
                  f"m4={m4['sample_f1']:.4f} line={hy['line_f1']:.4f} "
                  f"chunk={hy['chunk_f1']:.4f} token={hy['token_f1']:.4f}"
                  f"  [{time.time() - t0:.0f}s]", flush=True)
    model.encoder = enc
    return out


# --------------------------------------------------------------------------- #
# 5. 报表
# --------------------------------------------------------------------------- #
def report_probe(p5: dict, p6: dict, ks=(0, 1, 2, 3, 4, 6)) -> None:
    layers = p5["layers"]
    R = p5["rows"]
    print("\n" + "=" * 96)
    print("P5｜逐层 token 级线性探针（hybrid train→test；冻结编码器 + 单层线性分类器）")
    print("=" * 96)
    print(f"{'层':>4}{'token-AUC':>12}{'ΔAUC':>10}{'token-F1':>11}{'ΔF1':>10}"
          f"{'行级F1':>10}{'片段F1':>10}{'E‖x‖²':>12}")
    prev = None
    for li in layers:
        r = R[li]
        d_a = "" if prev is None else f"{r['auc'] - prev['auc']:+.4f}"
        d_f = "" if prev is None else f"{r['f1'] - prev['f1']:+.4f}"
        print(f"{li:>4}{r['auc']:>12.4f}{d_a:>10}{r['f1']:>11.4f}{d_f:>10}"
              f"{r['line_f1']:>10.4f}{r['chunk_f1']:>10.4f}{r['mean_sq']:>12.1f}")
        prev = r
    print("（Δ 列 = 相对**上一层**的边际：正得越多说明这一层越不可删；0 附近 = 删了不心疼）")

    base = R[layers[-1]]
    print("\n" + "=" * 96)
    print("P5b｜★ 截断到第 k 层 ⇒ 「删掉尾部 k 层」的 token 级代价（AUC 无阈值，最稳）")
    print("=" * 96)
    print(f"{'删掉层数':>10}{'截断层':>8}{'token-AUC':>12}{'ΔAUC vs 全 12 层':>20}"
          f"{'token-F1':>11}{'ΔF1':>10}{'行级F1':>10}{'Δ':>10}")
    for k in ks:
        li = layers[-1] - k
        if li not in R:
            continue
        r = R[li]
        print(f"{k:>10}{li:>8}{r['auc']:>12.4f}{r['auc'] - base['auc']:>+20.4f}"
              f"{r['f1']:>11.4f}{r['f1'] - base['f1']:>+10.4f}"
              f"{r['line_f1']:>10.4f}{r['line_f1'] - base['line_f1']:>+10.4f}")

    if not p6:
        return
    print("\n" + "=" * 96)
    print("P6｜逐层文档级线性探针（m4 train→test）")
    print("      full = 全分辨率平均池化（编码器输出的信息总量）")
    print("      s64  = **瓶颈分辨率**（先降到 ceil(L/64) 再平均，与 levels[0] 带宽一致）")
    print("=" * 96)
    print(f"{'层':>4}{'full-AUC':>11}{'Δ':>9}{'full-acc':>11}{'s64-AUC':>11}{'Δ':>9}{'s64-acc':>11}")
    prev = None
    for li in p6["layers"]:
        r = p6["rows"][li]
        d1 = "" if prev is None else f"{r['full']['auc'] - prev['full']['auc']:+.4f}"
        d2 = "" if prev is None else f"{r['s64']['auc'] - prev['s64']['auc']:+.4f}"
        print(f"{li:>4}{r['full']['auc']:>11.4f}{d1:>9}{r['full']['acc']:>11.4f}"
              f"{r['s64']['auc']:>11.4f}{d2:>9}{r['s64']['acc']:>11.4f}")
        prev = r
    print("\n★ s64 的 Δ 列若在深层已归零 ⇒ 瓶颈是**带宽**受限，往瓶颈堆层不会有用。")

    print("\n" + "=" * 96)
    print("P6b｜文档级按长度分桶（s64 分辨率，full 见 json）")
    print("=" * 96)
    head = f"{'层':>4}" + "".join(f"{('[' + str(lo) + ',' + ('inf' if hi > (1 << 29) else str(hi)) + ')'):>16}"
                                 for lo, hi in BUCKETS)
    print(head + f"{'总体':>12}")
    for li in p6["layers"]:
        r = p6["rows"][li]
        line = f"{li:>4}"
        for lo, hi in BUCKETS:
            key = f"[{lo},{'inf' if hi > (1 << 29) else hi})"
            b = r["s64"]["bucket"].get(key)
            line += fmt(b["auc"], 16) if b else f"{'--':>16}"
        print(line + f"{r['s64']['auc']:>12.4f}")


def report_stitch(st: dict, official: dict | None, ks) -> None:
    variants = [v for v in ("raw", "bridge") if any(v in st.get(k, {}) for k in ks)]
    base = {v: st.get(0, {}).get(v) for v in variants}
    print("\n" + "=" * 96)
    print("P7｜cut-and-stitch 实测：把**训练好的主干与头**接在第 (12−k) 层特征上（不重训）")
    print("    raw    = 直接接上去（主干没见过该层特征的分布 ⇒ 悲观下界）")
    print("    bridge = 先把该层特征**线性映射回第 12 层**再接（消掉分布漂移 ⇒ 乐观上界）")
    print("=" * 96)
    print(f"{'删掉层数':>8}{'截断层':>7}{'变体':>8}{'m4-F1':>10}{'Δ':>9}"
          f"{'行级F1':>10}{'Δ':>9}{'片段F1':>10}{'Δ':>9}{'tokenF1':>10}{'Δ':>9}")
    for k in ks:
        if k not in st:
            continue
        r = st[k]
        for v in variants:
            if v not in r:
                continue
            m4, hy = r[v]["m4"], r[v]["hybrid"]
            b = base[v]
            d = [""] * 4 if b is None else [
                f"{m4['sample_f1'] - b['m4']['sample_f1']:+.4f}",
                f"{hy['line_f1'] - b['hybrid']['line_f1']:+.4f}",
                f"{hy['chunk_f1'] - b['hybrid']['chunk_f1']:+.4f}",
                f"{hy['token_f1'] - b['hybrid']['token_f1']:+.4f}",
            ]
            print(f"{k:>8}{r['layer']:>7}{v:>8}{m4['sample_f1']:>10.4f}{d[0]:>9}"
                  f"{hy['line_f1']:>10.4f}{d[1]:>9}{hy['chunk_f1']:>10.4f}{d[2]:>9}"
                  f"{hy['token_f1']:>10.4f}{d[3]:>9}")
    print("\n★ k=0（截断层 12，原样）是内建自检，``raw`` 必须与官方 eval.json 一致：")
    if 0 in st and "raw" in st[0]:
        a = st[0]["raw"]
        if official:
            for key, path in (("m4", ("m4", "sample_f1")), ("line", ("hybrid", "line_f1")),
                              ("token", ("hybrid", "token_f1"))):
                got = a[path[0]][path[1]]
                want = official[f"{path[0]}/test"][path[1]]
                print(f"   {key:<6}{got:.6f} vs 官方 {want:.6f}   Δ={got - want:+.2e}")
        else:
            print("   （没找到 eval.json，跳过）")
    print("★ raw 与 bridge 之间的**宽度**就是“分布漂移”占了多少；"
          "两者都差 ⇒ 删层真的丢信息；只有 raw 差 ⇒ 主要是主干没适配。")


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description="逐层编码器信息量探针（P5/P6/P7）")
    ap.add_argument("--ckpt", default="runs/v0.4.5/best.pt")
    ap.add_argument("--config", default="configs/udet_v045.yaml")
    ap.add_argument("--stage", default="probe", choices=["probe", "stitch", "all"])
    ap.add_argument("--layers", type=int, nargs="+", default=None, help="要探的层（默认全部）")
    ap.add_argument("--ks", type=int, nargs="+", default=[0, 2, 3, 4],
                    help="stitch 的「删掉层数」")
    ap.add_argument("--bridge", action="store_true",
                    help="P7b：先拟合「第 ℓ 层 → 第 12 层」的线性桥接，再去掉分布漂移重新度量")
    ap.add_argument("--bridge-hybrid", type=int, default=1200, help="拟合桥接用的 hybrid train 条数")
    ap.add_argument("--bridge-m4", type=int, default=3000, help="拟合桥接用的 m4 train 条数")
    ap.add_argument("--limit-hybrid-train", type=int, default=2000)
    ap.add_argument("--limit-m4-train", type=int, default=8000)
    ap.add_argument("--limit-test", type=int, default=0)
    ap.add_argument("--lam", type=float, default=1e-2, help="岭回归正则（特征已标准化）")
    args = ap.parse_args()

    setup_cuda()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = yaml.safe_load(open(str(resolve(args.config)), encoding="utf-8"))
    model, enc, tokenizer = build(cfg, args.ckpt, device)
    report = make_report(cfg, tokenizer)
    selfcheck(enc, device)

    run_dir = Path(resolve(args.ckpt)).parent
    n_layer = enc.num_layers + 1
    layers = args.layers if args.layers else list(range(n_layer))
    layers = sorted({li for li in layers if 0 <= li < n_layer})

    if args.stage in ("probe", "all"):
        p5 = probe_token(model, enc, cfg, report, device, layers,
                         args.limit_hybrid_train, args.limit_test, args.lam)
        p6 = probe_doc(model, enc, cfg, report, device, layers,
                       args.limit_m4_train, args.limit_test, args.lam)
        blob = {"ckpt": args.ckpt, "config": args.config, "lam": args.lam,
                "p5_token": p5, "p6_doc": p6}
        (run_dir / "probe_layers.json").write_text(
            json.dumps(blob, ensure_ascii=False, indent=1), encoding="utf-8")
        report_probe(p5, p6, tuple(args.ks))
        print(f"\n[probe] 已落盘 {run_dir / 'probe_layers.json'}")

    if args.stage in ("stitch", "all"):
        official = None
        ev = run_dir / "eval.json"
        if ev.exists():
            official = json.loads(ev.read_text(encoding="utf-8"))["metrics"]
        bridges = None
        if args.bridge:
            print(f"\n[P7b] 拟合线性桥接（hybrid {args.bridge_hybrid} + m4 {args.bridge_m4} 条 train）…")
            bridges = fit_bridges(enc, cfg, report, device, layers, n_layer,
                                  args.bridge_hybrid, args.bridge_m4, args.lam)
        st = run_stitch(model, enc, cfg, report, device, args.ks, args.limit_test, bridges)
        # ★ 合并而不是覆盖：同一 ckpt 可能分几次跑不同的 k（先跑 0/3/4、再补 2），
        #   直接覆盖会把上一次的结果丢掉（B2 类陷阱）。
        out_path = run_dir / "probe_stitch.json"
        merged = {"ckpt": args.ckpt, "ks": [], "bridge": bool(args.bridge), "rows": {}}
        if out_path.exists():
            old = json.loads(out_path.read_text(encoding="utf-8"))
            merged["ks"] = [int(k) for k in old.get("ks", [])]
            merged["rows"] = {str(k): v for k, v in old.get("rows", {}).items()}
            merged["bridge"] = bool(old.get("bridge")) or bool(args.bridge)
        for k, v in st.items():
            merged["rows"][str(k)] = v
        merged["ks"] = sorted({int(k) for k in merged["rows"]})
        out_path.write_text(json.dumps(merged, ensure_ascii=False, indent=1, default=str),
                            encoding="utf-8")
        report_stitch({int(k): v for k, v in merged["rows"].items()}, official,
                      tuple(merged["ks"]))
        print(f"\n[stitch] 已落盘 {out_path}（本次 {list(st)}，文件内累计 {merged['ks']}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
