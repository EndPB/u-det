"""零训练探针 P1 / P2：**逐尺度 token 表现** + **尺度权重离线定标**。

不训练，只做一次前向，回答两个决定 v0.4.6 是否值得改的问题：

    P1  短文档里 token 级 F1 是否随"尺度变粗"而下降？
        （= 长度感知尺度权重 w_i 的**前提**；不成立就别改）
    P2  长度权重 w_i = σ((log2 n_i − log2 N_min)/τ) 与静态权重 [0.2..1.0]，
        在各长度桶上的加权 BCE 谁更优？τ 取多少？

为什么用 BCE 而不是 F1 做 P2 的判据：F1 不可逐样本相加，
而 `token_loss` 实际优化的就是"逐尺度 BCE 的加权平均"，所以 BCE 才是与改动一致的度量。
F1 仍然照算，用来给 P1 一个**可解释**的读数。

用法::

    # 先落盘（约 3~6 分钟，一次前向）
    OMP_NUM_THREADS=8 python scripts/probe_scales.py --ckpt runs/v0.4.5/best.pt --split test

    # 再反复离线分析（不占 GPU、可随便扫 τ）
    python scripts/probe_scales.py --load --taus 0.25 0.5 1 2 4 --nmin 8

落盘位置：``runs/<tag>/probe_scales_<split>.pt``（与 ckpt 同目录，tag 从路径推断）。
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from report import build_report  # noqa: E402
from train import _aggregate_target, build_model, make_dataset, setup_cuda  # noqa: E402
from train import IGNORE  # noqa: E402

BUCKETS = [(0, 512), (512, 2048), (2048, 8192), (8192, 1 << 30)]


# --------------------------------------------------------------------------- #
# 权重函数（与 Step 1 将要实现的完全同一套定义）
# --------------------------------------------------------------------------- #
def length_weights(ns: list[int], nmin: float = 8.0, tau: float = 1.0) -> list[float]:
    """w_i = σ((log2 n_i − log2 N_min) / τ)；n_i = 该尺度的真实位置数。"""
    lo = math.log2(max(float(nmin), 1e-9))
    return [1.0 / (1.0 + math.exp(-((math.log2(max(float(n), 1.0)) - lo) / tau))) for n in ns]


def normalized(ws: list[float]) -> list[float]:
    s = sum(ws) or 1.0
    return [w / s for w in ws]


# --------------------------------------------------------------------------- #
# 前向采样（P1 的数据来源）
# --------------------------------------------------------------------------- #
@torch.no_grad()
def collect(model, ds, cfg: dict, device: str) -> dict:
    """逐样本前向，记录长度 / 逐尺度位置数 / 逐尺度 BCE / 逐尺度 token F1。"""
    loss_cfg = cfg.get("loss", {})
    modes = loss_cfg.get("token_target")
    beta = float(loss_cfg.get("token_target_beta", 4.0))
    rec = {"len": [], "n": [], "bce": [], "f1": [], "p": [], "r": [], "n_rep": []}

    for i in range(len(ds)):
        item = ds[i]
        ids = torch.as_tensor(item["input_ids"], dtype=torch.long, device=device).unsqueeze(0)
        # ★ 必须保持 (1, L)：`token_loss` 里的 tok_labels 是二维的，
        #   若这里用 1-D，`v.unsqueeze(1)` 会把 batch 维当成序列长度，avg_pool1d 直接报错。
        tok = torch.as_tensor(item["tok_labels"], dtype=torch.long, device=device).unsqueeze(0)
        _, token_logits = model(ids, torch.ones_like(ids))
        if token_logits is None:
            raise SystemExit("该 ckpt 没有 token 头，无法做本探针")

        valid = (tok != IGNORE)                       # (1, L)
        length = tok.shape[-1]
        target = tok.clamp(min=0).float()             # (1, L)
        bces, f1s, ps, rs, ns = [], [], [], [], []

        for si, lg in enumerate(token_logits):
            lg = lg.reshape(-1).float()
            want = lg.shape[0]
            ns.append(want)
            # ---- BCE：与 token_loss 逐行同构（同样的池化目标、同样的 keep 掩码）----
            k = max(1, math.ceil(length / want))
            pad = max(0, want * k - length)
            t = F.pad(target, (0, pad)) if pad else target
            v = F.pad(valid.float(), (0, pad)) if pad else valid.float()
            den = F.avg_pool1d(v.unsqueeze(1), k, k)          # (1, 1, want)
            keep = (den > 0).float()
            mode = (modes[si] if modes and si < len(modes) else "mean")
            soft = _aggregate_target(t, v, k, mode, beta)      # (1, 1, want)
            lg3 = lg.reshape(1, 1, want)
            if float(keep.sum()) > 0:
                b = F.binary_cross_entropy_with_logits(lg3, soft, weight=keep, reduction="sum")
                bces.append(float(b / keep.sum()))
            else:
                bces.append(float("nan"))
            # ---- token F1：把该尺度上采样回原长度再与真实 token 标签比（可解释）----
            up = lg3.repeat_interleave(k, dim=-1)[:, :, :length].reshape(-1)
            pred = (up > 0) & valid.reshape(-1)
            gt = (tok.reshape(-1) == 1) & valid.reshape(-1)
            tp = float((pred & gt).sum())
            fp = float((pred & ~gt).sum())
            fn = float((~pred & gt).sum())
            p = tp / max(tp + fp, 1e-9)
            r = tp / max(tp + fn, 1e-9)
            f1s.append(2 * p * r / max(p + r, 1e-9))
            ps.append(p)
            rs.append(r)

        rec["len"].append(int(valid.sum()))          # ★ 真实代码长度（剔除报告前缀）
        rec["n_rep"].append(int((tok == IGNORE).sum()))
        rec["n"].append(ns)
        rec["bce"].append(bces)
        rec["f1"].append(f1s)
        rec["p"].append(ps)
        rec["r"].append(rs)
        if (i + 1) % 200 == 0:
            print(f"  [{i + 1}/{len(ds)}] …", flush=True)

    for k in ("len", "n", "n_rep"):
        rec[k] = torch.tensor(rec[k])
    rec["bce"] = torch.tensor(rec["bce"])
    rec["f1"] = torch.tensor(rec["f1"])
    rec["p"] = torch.tensor(rec["p"])
    rec["r"] = torch.tensor(rec["r"])
    return rec


# --------------------------------------------------------------------------- #
# 分析
# --------------------------------------------------------------------------- #
def _bucket(lengths: torch.Tensor) -> list[int]:
    idx = torch.full_like(lengths, -1)
    for b, (lo, hi) in enumerate(BUCKETS):
        idx[(lengths >= lo) & (lengths < hi)] = b
    return idx


def report_p1(rec: dict) -> None:
    L, f1, bce, n = rec["len"], rec["f1"], rec["bce"], rec["n"]
    bidx = _bucket(L)
    ns = n.shape[1]
    print("\n" + "=" * 78)
    print("P1｜逐尺度 token F1（把该尺度上采样回原长度后算，阈值 0.5）")
    print("=" * 78)
    head = f"{'长度区间':<14}{'n':>6}  " + "".join(f"{'s=' + str(int(n[0, i])):>9}" for i in range(ns))
    print(head)
    for b, (lo, hi) in enumerate(BUCKETS):
        m = bidx == b
        if int(m.sum()) == 0:
            continue
        hi_s = "∞" if hi > (1 << 29) else str(hi)
        row = f"[{lo},{hi_s})".ljust(14) + f"{int(m.sum()):>6}  "
        row += "".join(f"{float(f1[m][:, i].mean()):>9.4f}" for i in range(ns))
        print(row)
    print("-" * 78)
    row = "ALL".ljust(14) + f"{len(L):>6}  "
    row += "".join(f"{float(f1[:, i].mean()):>9.4f}" for i in range(ns))
    print(row)
    print("（尺度由粗到细，步长 64/32/16/4/1；读数应**随尺度变细而升高**——这是全局趋势）\n")

    print("=" * 78)
    print("P1b｜★ 关键判据：**短桶里从最细 → 最粗的 F1 落差**")
    print("=" * 78)
    print(f"{'长度区间':<14}{'n':>6}{'最细(s=1)':>12}{'最粗(s=64)':>12}{'落差':>10}")
    for b, (lo, hi) in enumerate(BUCKETS):
        m = bidx == b
        if int(m.sum()) == 0:
            continue
        fine = float(f1[m][:, -1].mean())
        coarse = float(f1[m][:, 0].mean())
        hi_s = "∞" if hi > (1 << 29) else str(hi)
        print(f"[{lo},{hi_s})".ljust(14) + f"{int(m.sum()):>6}{fine:>12.4f}{coarse:>12.4f}{coarse - fine:>10.4f}")
    print("\n若短桶落差明显大于长桶 ⇒ w_i 的「短文档更信细尺度」前提成立。\n")

    print("=" * 78)
    print("P2｜加权 BCE：静态权重 vs 长度权重（同一批逐尺度 BCE，只换权重）")
    print("=" * 78)
    print("（BCE 越小越好；这里报的是「按样本平均的加权 BCE」）")
    return


def report_p2(rec: dict, static_w: list[float], taus: list[float], nmin: float) -> None:
    L, bce, n = rec["len"], rec["bce"], rec["n"]
    bidx = _bucket(L)
    print(f"\n{'权重方案':<22}" + "".join(
        f"{('[' + str(lo) + ',' + ('∞' if hi > (1 << 29) else str(hi)) + ')'):>14}" for lo, hi in BUCKETS) + f"{'总计':>12}")
    print("-" * 115)

    def wbce(ws_fn) -> list[float]:
        out = []
        for i in range(len(L)):
            w = ws_fn(n[i].tolist())
            b = bce[i]
            ok = torch.isfinite(b)
            wv = torch.tensor(w, dtype=torch.float32)[ok]
            bv = b[ok]
            out.append(float((wv * bv).sum() / wv.sum().clamp(min=1e-9)))
        return out

    def row(name: str, vals: list[float]) -> None:
        v = torch.tensor(vals)
        line = f"{name:<22}"
        for b, (lo, hi) in enumerate(BUCKETS):
            m = bidx == b
            line += f"{float(v[m].mean()):>14.4f}" if int(m.sum()) else f"{'--':>14}"
        print(line + f"{float(v.mean()):>12.4f}")

    print(f"归一化后的静态权重 = {[round(x, 4) for x in normalized(static_w)]}")
    row("静态 [0.2..1.0]", wbce(lambda _: static_w))
    base = None
    for tau in taus:
        vals = wbce(lambda ns, t=tau: length_weights(ns, nmin, t))
        if base is None:
            base = vals
        row(f"长度 τ={tau:g} (N_min={nmin:g})", vals)
    print("\n（每一列都可横向比较：数字越小越好）\n")


def report_shape(static_w: list[float], taus: list[float], nmin: float) -> None:
    print("=" * 78)
    print("权重形状速查（归一化，由粗到细）")
    print("=" * 78)
    print(f"{'L':>7}{'n_i':>28}   {'长度权重':<40}{'静态':<40}")
    for L in (64, 293, 1024, 4096, 8192, 32768):
        ns = [max(1, math.ceil(L / s)) for s in (64, 32, 16, 4, 1)]
        for tau in (taus[0],):
            print(f"{L:>7}{str(ns):>28}   "
                  f"{str([round(x, 3) for x in normalized(length_weights(ns, nmin, tau))]):<40}"
                  f"{str([round(x, 3) for x in normalized(static_w)]):<40}")
    print()


def main() -> int:
    ap = argparse.ArgumentParser(description="P1/P2 逐尺度 token 探针")
    ap.add_argument("--ckpt", default="runs/v0.4.5/best.pt")
    ap.add_argument("--config", default="configs/udet_v045.yaml")
    ap.add_argument("--split", default="test", choices=["test", "val"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--load", action="store_true", help="跳过前向，直接读已落盘的探测结果")
    ap.add_argument("--taus", type=float, nargs="+", default=[0.25, 0.5, 1.0, 2.0, 4.0])
    ap.add_argument("--nmin", type=float, default=8.0)
    args = ap.parse_args()

    run_dir = Path(args.ckpt).parent
    cache = run_dir / f"probe_scales_{args.split}.pt"

    if args.load:
        if not cache.exists():
            raise SystemExit(f"没有 {cache}，先跑一次不带 --load")
        rec = torch.load(cache, map_location="cpu", weights_only=False)
        static_w = rec["static_w"]
        print(f"[probe] 读入 {cache}")
    else:
        setup_cuda()
        cfg = yaml.safe_load(open(args.config))
        tokenizer = AutoTokenizer.from_pretrained(str(cfg["encoder"]["path"]))
        report = build_report(cfg["report"]["name"], tokenizer=tokenizer,
                              max_tokens=cfg["report"].get("max_tokens", 64))
        model, _ = build_model(cfg, tokenizer)
        ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["state"], strict=False)
        model.to("cuda").eval()
        print(f"[probe] 载入 {args.ckpt}（epoch {ckpt.get('epoch')}）")
        ds = make_dataset(cfg, "hybrid", args.split, report, False, args.limit or None)
        print(f"[probe] hybrid/{args.split}: {len(ds)} 条，开始一次前向 …")
        static_w = list(cfg.get("loss", {}).get("scale_weights", [1.0, 1.0, 1.0, 1.0, 1.0]))
        rec = collect(model, ds, cfg, "cuda")
        rec["static_w"] = static_w
        rec["split"] = args.split
        rec["ckpt"] = args.ckpt
        torch.save(rec, cache)
        print(f"[probe] 已落盘 {cache}")

    if torch.is_tensor(rec["static_w"]) or isinstance(rec["static_w"], list):
        static_w = [float(x) for x in rec["static_w"]]
    report_shape(static_w, args.taus, args.nmin)
    report_p1(rec)
    report_p2(rec, static_w, args.taus, args.nmin)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
