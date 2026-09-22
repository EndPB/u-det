"""零训练探针 P11：**瓶颈的「带宽」到底是不是限制** —— 池化步长的对照扫描。

动机来自 P10 的意外结果：把主干各尺度拿去跟「文档级 / token 级标签」做线性探针，
发现 **v0.4.5 的主干反而比 v0.4.4 更好**（最细尺度 token F1 0.8271 vs 0.8174），
瓶颈的文档级信息也**没有下降**（AUC 0.9889 vs 0.9877）。
⇒ §8.11.6 的「旁路削弱了信息保值」这条机制**没有拿到表征层面的证据**。

于是问题回到 §8.8.3 / §8.16.6 那条老诊断：**瓶颈是带宽受限**（一篇 1024 token 的文档
在 L/64 上只剩 16 个位置，而滑窗基线在同长度上仍有逐 token 的 512 窗口）。

本探针把「带宽」直接做成一个可扫的变量：**同一层的特征，按不同的池化步长
s ∈ {64, 32, 16, 8, 4, 1} 降到 ceil(L/s) 个位置，再各自拟一个文档级线性探针。**

判据：
    * 若 s 从 64 降到 32/16 时 AUC 明显上升 ⇒ **带宽确实是限制**
      ⇒ 「让 `divs` 对中短文档自适应（少下一次采样）」值得作为那**一次**训练；
    * 若各 s 基本持平 ⇒ 带宽不是限制，那条路也不该走。

用法::

    OMP_NUM_THREADS=8 python scripts/probe_bandwidth.py --ckpt runs/v0.4.5/best.pt \
        --config configs/udet_v045.yaml --limit-train 3000
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from probe_layers import RidgeStream, encoder_states, auc, prior_threshold, prf  # noqa: E402
from train import build_model, make_dataset, make_report, resolve, setup_cuda     # noqa: E402
from transformers import AutoTokenizer                                           # noqa: E402

STRIDES = (64, 32, 16, 8, 4, 1)
BUCKETS = [(0, 512), (512, 1024), (1024, 2048), (2048, 8192), (8192, 1 << 30)]


def pooled(x: torch.Tensor, stride: int) -> torch.Tensor:
    """(1, L, D) -> (D,)：先降到 ceil(L/stride) 个位置（不重叠窗口均值），再对位置取均值。"""
    if stride <= 1:
        return x[0].mean(0)
    length = x.shape[1]
    n = max(1, math.ceil(length / stride))
    pad = n * stride - length
    if pad:
        x = torch.nn.functional.pad(x, (0, 0, 0, pad))
    return x.reshape(1, n, stride, -1).mean(2).mean(1)[0]


def main() -> int:
    ap = argparse.ArgumentParser(description="P11：瓶颈带宽的池化步长对照")
    ap.add_argument("--ckpt", default="runs/v0.4.5/best.pt")
    ap.add_argument("--config", default="configs/udet_v045.yaml")
    ap.add_argument("--limit-train", type=int, default=3000)
    ap.add_argument("--split", default="test", choices=["test", "val"])
    ap.add_argument("--level", type=int, default=0, help="用第几个尺度（0 = 瓶颈）")
    args = ap.parse_args()

    setup_cuda()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = yaml.safe_load(open(str(resolve(args.config)), encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(str(resolve(cfg["encoder"]["path"])))
    report = make_report(cfg, tokenizer)
    model, enc = build_model(cfg, tokenizer)
    blob = torch.load(resolve(args.ckpt), map_location="cpu", weights_only=False)
    model.load_state_dict(blob["state"], strict=False)
    model.to(device).eval()
    print(f"[P11] 载入 {args.ckpt}（epoch {blob.get('epoch')}），探第 {args.level} 个尺度")

    n_lv = enc.num_layers + 1
    acc = {(s, lv): RidgeStream(enc.hidden_size) for s in STRIDES for lv in [args.level]}
    ds_tr = make_dataset(cfg, "m4", "train", report, True, args.limit_train or None)
    for i in range(len(ds_tr)):
        it = ds_tr[i]
        ids = torch.tensor([it["input_ids"]], dtype=torch.long, device=device)
        states = encoder_states(enc, ids, torch.ones_like(ids))
        y = torch.tensor([1.0 if it["label"] == 1 else -1.0], device=device)
        for s in STRIDES:
            acc[(s, args.level)].add(pooled(states[args.level], s).unsqueeze(0), y)
        if (i + 1) % 1000 == 0:
            print(f"   fit [{i + 1}/{len(ds_tr)}]", flush=True)
    packed = {k: RidgeStream.to_device(v.fit(1e-2), device) for k, v in acc.items()}
    print("   fit 完成")

    ds_te = make_dataset(cfg, "m4", args.split, report, False, None)
    sc = {s: [] for s in STRIDES}
    labels, lengths = [], []
    for i in range(len(ds_te)):
        it = ds_te[i]
        ids = torch.tensor([it["input_ids"]], dtype=torch.long, device=device)
        states = encoder_states(enc, ids, torch.ones_like(ids))
        labels.append(int(it["label"]))
        lengths.append(int(ids.shape[1]))
        for s in STRIDES:
            v = pooled(states[args.level], s).unsqueeze(0)
            sc[s].append(float(acc[(s, args.level)].score(v, packed[(s, args.level)])[0]))
        if (i + 1) % 500 == 0:
            print(f"   score [{i + 1}/{len(ds_te)}]", flush=True)

    y = torch.tensor(labels)
    L = torch.tensor(lengths)
    print(f"\n{'=' * 88}")
    print(f"P11｜第 {args.level} 个尺度（{'瓶颈' if args.level == 0 else 'levels[%d]' % args.level}）"
          f"按不同池化步长降采样后的**文档级线性探针**（m4 {args.split}，n={len(labels)}）")
    print(f"     步长 s ⇒ 位置数 ⌈L/s⌉；s=1 就是全分辨率（无降采样）")
    print(f"{'=' * 88}")
    print(f"{'步长 s':>8}{'AUC':>10}{'acc':>10}{'F1':>10}{'位置数中位':>14}")
    for s in STRIDES:
        v = torch.tensor(sc[s])
        thr = prior_threshold(v, packed[(s, args.level)]["p_pos"])
        npos = torch.tensor([math.ceil(int(x) / s) for x in lengths])
        print(f"{s:>8}{auc(v, y):>10.4f}{float(((v > thr).long() == y).float().mean()):>10.4f}"
              f"{prf(v, y, thr)['f1']:>10.4f}{int(npos.median()):>14d}")
    print("\n判据：s 从 64 降到 32/16 时 AUC 明显上升 ⇒ **带宽确实是限制**，"
          "「divs 对短文档自适应」值得作为那一次训练；")
    print("     各 s 基本持平 ⇒ 带宽不是限制，那条路也不该走。")

    print(f"\n{'─' * 88}\n按长度分桶（AUC）：看带宽问题在哪个长度区间最严重")
    head = f"{'区间':>16}{'n':>6}" + "".join(f"{'s=' + str(s):>9}" for s in STRIDES)
    print(head)
    for lo, hi in BUCKETS:
        sel = (L >= lo) & (L < hi)
        if int(sel.sum()) == 0:
            continue
        hi_s = "inf" if hi >= 1 << 30 else str(hi)
        row = f"{'[' + str(lo) + ',' + hi_s + ')':>16}{int(sel.sum()):>6}"
        for s in STRIDES:
            v = torch.tensor(sc[s])[sel]
            yy = y[sel]
            a = auc(v, yy)
            row += f"{'--':>9}" if a != a else f"{a:>9.4f}"
        print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
