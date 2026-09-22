"""零训练探针 P9：**预测「额外一路头」（v0.4.9）在 token 侧能拿到什么**。

背景：v0.4.9 = v0.4.5 + 把旁路从「拼进最细头」改成「额外一路头」。
它改变了**两个**读出器的分工：

    主干最细头（768→1）  ← 回到 v0.4.4 的形状，**不再**看到编码器 feats
    额外一路头（768→1）  ← 只吃编码器 feats
    报告/评测           = 两者 logits 的**均值**

m4（文档级）那一边没法零训练预测（它取决于"主干是否回到 v0.4.4 的受力状态"，
是机制假设）。但 **token 那一侧可以预测**，因为我们可以用**现有的两个 run 的产物**
拼出两个读出器的代理：

    v0.4.4 的最细头      = 一个**没有旁路**、在主干上训出来的读出器
                           ⇒ 正是 v0.4.9 主干那一级所处的受力状态
    hybrid train 上拟合的线性探针（脊回归，吃编码器 layer-12 的 feats）
                           = v0.4.9 额外一路头的代理（同样的输入、同样的目标）

于是把两者的 logits 平均，就得到 **v0.4.9 在 token 侧的预测值**，
再与 v0.4.4（0.8187）和 v0.4.5（0.8281）的实测对比 ——
**一次 GPU 前向（约 3 分钟）+ 纯 CPU 离线**，不用训两小时。

用法::

    OMP_NUM_THREADS=8 python scripts/probe_ensemble.py \
        --config configs/udet_v045.yaml --ckpt runs/v0.4.5/best.pt \
        --surrogate runs/v0.4.4 --concat runs/v0.4.5 --split test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_raw import hybrid_metrics_from_tokens                        # noqa: E402
from probe_layers import RidgeStream, encoder_states                      # noqa: E402
from train import (IGNORE, build_model, make_dataset, make_report,        # noqa: E402
                   resolve, setup_cuda)
from transformers import AutoTokenizer                                    # noqa: E402

BUCKETS = [(0, 512), (512, 2048), (2048, 8192), (8192, 1 << 30)]


def logit(p: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    p = p.float().clamp(eps, 1 - eps)
    return torch.log(p / (1 - p))


# --------------------------------------------------------------------------- #
@torch.no_grad()
def feats_token_logits(cfg, ckpt: str, device: str, limit_train: int,
                       split: str) -> tuple[list[torch.Tensor], dict]:
    """在 hybrid train 上拟合「编码器 feats → token 标签」的线性读出，再对 test 打分。"""
    tokenizer = AutoTokenizer.from_pretrained(str(resolve(cfg["encoder"]["path"])))
    model, enc = build_model(cfg, tokenizer)
    blob = torch.load(resolve(ckpt), map_location="cpu", weights_only=False)
    model.load_state_dict(blob["state"], strict=False)
    model.to(device).eval()
    print(f"[P9] 载入 {ckpt}（epoch {blob.get('epoch')}），用它取编码器 layer-12 的 feats")
    report = make_report(cfg, tokenizer)
    n_layer = enc.num_layers + 1

    ds_tr = make_dataset(cfg, "hybrid", "train", report, True, limit_train or None)
    acc = RidgeStream(enc.hidden_size)
    for i in range(len(ds_tr)):
        it = ds_tr[i]
        ids = torch.tensor([it["input_ids"]], dtype=torch.long, device=device)
        tok = torch.tensor([it["tok_labels"]], dtype=torch.long, device=device).reshape(-1)
        valid = tok != IGNORE
        feats = encoder_states(enc, ids, torch.ones_like(ids))[n_layer - 1][0][valid]
        acc.add(feats, torch.where(tok[valid] == 1, 1.0, -1.0))
        if (i + 1) % 500 == 0:
            print(f"   fit [{i + 1}/{len(ds_tr)}]")
    packed = RidgeStream.to_device(acc.fit(1e-2), device)
    print(f"   fit 完成（{len(ds_tr)} 条，{acc.n} 个 token，正类比例 {packed['p_pos']:.3f}）")

    ds_te = make_dataset(cfg, "hybrid", split, report, False, None)
    out, meta = [], {"line_of_token": [], "line_label": [], "token_label": [], "length": []}
    for i in range(len(ds_te)):
        it = ds_te[i]
        ids = torch.tensor([it["input_ids"]], dtype=torch.long, device=device)
        tok = torch.tensor([it["tok_labels"]], dtype=torch.long, device=device).reshape(-1)
        valid = (tok != IGNORE)
        feats = encoder_states(enc, ids, torch.ones_like(ids))[n_layer - 1][0][valid]
        sc = acc.score(feats, packed).cpu()
        full = torch.zeros(tok.numel())
        full[valid.cpu()] = sc
        out.append(full)
        meta["line_of_token"].append(torch.tensor(it["line_of_token"], dtype=torch.long))
        meta["line_label"].append(torch.tensor(it["line_label"], dtype=torch.long))
        meta["token_label"].append(tok.cpu())
        meta["length"].append(int(valid.sum()))
        if (i + 1) % 300 == 0:
            print(f"   score [{i + 1}/{len(ds_te)}]")
    meta["n"] = len(ds_te)
    return out, meta


# --------------------------------------------------------------------------- #
def variant(blob, meta, thr: float, probs_fn) -> dict:
    return hybrid_metrics_from_tokens(blob, thr, probs_fn)


def main() -> int:
    ap = argparse.ArgumentParser(description="P9：预测「额外一路头」在 token 侧能拿到什么")
    ap.add_argument("--config", default="configs/udet_v045.yaml")
    ap.add_argument("--ckpt", default="runs/v0.4.5/best.pt")
    ap.add_argument("--surrogate", default="runs/v0.4.4",
                    help="「主干最细头」的代理（该 run 没有旁路）")
    ap.add_argument("--concat", default="runs/v0.4.5",
                    help="当前基线（旁路拼进原头）")
    ap.add_argument("--split", default="test", choices=["test", "val"])
    ap.add_argument("--limit-hybrid-train", type=int, default=2000)
    ap.add_argument("--cache", default="runs/v0.4.5/probe_feats_token.pt")
    ap.add_argument("--reuse", action="store_true", help="复用已缓存的 feats 读出分数")
    args = ap.parse_args()

    setup_cuda()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cache = Path(resolve(args.cache))
    if args.reuse and cache.exists():
        print(f"[P9] 复用 {cache}")
        blob_c = torch.load(cache, weights_only=False)
        feats_logits, meta = blob_c["feats_logits"], blob_c["meta"]
    else:
        cfg = yaml.safe_load(open(str(resolve(args.config)), encoding="utf-8"))
        feats_logits, meta = feats_token_logits(cfg, args.ckpt, device,
                                                args.limit_hybrid_train, args.split)
        cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"feats_logits": feats_logits, "meta": meta, "split": args.split,
                    "ckpt": args.ckpt, "config": args.config}, cache)
        print(f"[P9] 已落盘 {cache}")

    # ---- 读两个已有 run 的 dump，做对齐自检 ----
    ref = torch.load(resolve(args.surrogate) / f"raw_hybrid_{args.split}.pt",
                     map_location="cpu", weights_only=False)
    cur = torch.load(resolve(args.concat) / f"raw_hybrid_{args.split}.pt",
                     map_location="cpu", weights_only=False)
    n = min(len(feats_logits), ref["n"], cur["n"])
    same = all(torch.equal(ref["token_label"][i].long(), meta["token_label"][i].long())
               for i in range(n))
    print(f"[P9] 自检：{args.surrogate} 的 token_label 与本次数据集逐位一致 → {same}")
    if not same:
        raise SystemExit("token_label 对不上，先查 report 前缀/切分是否一致")

    blob = {"n": n, "token_label": ref["token_label"][:n],
            "line_of_token": meta["line_of_token"][:n],
            "line_label": meta["line_label"][:n]}

    cand = {
        f"A {args.surrogate}（无旁路的最细头 = 主干代理）": lambda i: ref["token_prob"][i].float(),
        f"B {args.concat}（旁路拼进原头 = 现基线）": lambda i: cur["token_prob"][i].float(),
        "C 额外头的代理（feats 线性读出，单独）": lambda i: torch.sigmoid(feats_logits[i]),
        "D ★ 预测 v0.4.9 = mean(A 主干代理, C 额外代理)":
            lambda i: torch.sigmoid(0.5 * (logit(ref["token_prob"][i].float())
                                           + feats_logits[i])),
    }

    print(f"\n{'=' * 92}\nP9｜用现有产物预测 v0.4.9 的 token 侧（split={args.split}，n={n}）\n{'=' * 92}")
    print(f"{'方案':<46}{'thr=0.5 行级':>13}{'片段':>9}{'token':>9}"
          f"{'最优thr':>9}{'行级':>9}{'片段':>9}{'token':>9}")
    grid = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    for name, fn in cand.items():
        m5 = variant(blob, meta, 0.50, fn)
        best = max(((t, variant(blob, meta, t, fn)) for t in grid),
                   key=lambda r: r[1]["token_f1"])
        line = (f"{name:<46}{m5['line_f1']:>13.4f}{m5['chunk_f1']:>9.4f}{m5['token_f1']:>9.4f}"
                f"{best[0]:>9.2f}{best[1]['line_f1']:>9.4f}{best[1]['chunk_f1']:>9.4f}"
                f"{best[1]['token_f1']:>9.4f}")
        print(line)
    print("（后三列是**在每个方案自己的最优阈值**下 —— 这是诊断用的口径，选择用了测试集，有泄漏，"
          "不能当正式成绩）")

    # ---- 混合权重扫描 ----
    print(f"\n{'─' * 92}\n混合权重扫描：logit 空间 w·主干代理 + (1−w)·额外代理（w=0.5 即 v0.4.9 的官方口径）")
    print(f"{'w':>6}{'thr=0.5 行级':>14}{'片段':>10}{'token':>10}{'最优thr':>10}{'token(最优)':>13}")
    for w in (0.0, 0.25, 0.5, 0.75, 1.0):
        fn = (lambda i, w=w: torch.sigmoid(w * logit(ref["token_prob"][i].float())
                                          + (1 - w) * feats_logits[i]))
        m5 = variant(blob, meta, 0.50, fn)
        bt, bm = max(((t, variant(blob, meta, t, fn)) for t in grid),
                     key=lambda r: r[1]["token_f1"])
        print(f"{w:>6.2f}{m5['line_f1']:>14.4f}{m5['chunk_f1']:>10.4f}{m5['token_f1']:>10.4f}"
              f"{bt:>10.2f}{bm['token_f1']:>13.4f}")
    print("  w=1.0 就是 A（主干代理单独）；w=0.0 是 C（额外代理单独）。"
          "\n  若 w 的最优点在 0.5 附近或 0.5 明显优于 1.0 ⇒ 额外头提供**互补**信息；"
          "\n  若 0.5 明显差于 1.0 ⇒ 额外头在**拖后腿**（那 v0.4.9 的设计需要改，比如降权）")

    # ---- 长度分桶（只看关键两行）----
    length = torch.tensor([len(t) for t in meta["token_label"][:n]])
    real_len = torch.tensor(meta["length"][:n])
    print(f"\n{'─' * 92}\n按长度分桶（thr=0.5，line F1）：看额外头对**短文档**（当前最大弱项）有没有用")
    print(f"{'区间':>16}{'n':>6}{'A 主干代理':>13}{'B 现基线':>11}{'D 预测 v0.4.9':>16}")
    for lo, hi in BUCKETS:
        sel = (real_len >= lo) & (real_len < hi)
        if int(sel.sum()) == 0:
            continue
        hi_s = "inf" if hi >= 1 << 30 else str(hi)
        idx = torch.nonzero(sel).reshape(-1).tolist()
        sub = {"n": len(idx), "token_label": [blob["token_label"][i] for i in idx],
               "line_of_token": [blob["line_of_token"][i] for i in idx],
               "line_label": [blob["line_label"][i] for i in idx]}
        vals = []
        for fn in (cand[f"A {args.surrogate}（无旁路的最细头 = 主干代理）"],
                   cand[f"B {args.concat}（旁路拼进原头 = 现基线）"],
                   cand["D ★ 预测 v0.4.9 = mean(A 主干代理, C 额外代理)"]):
            # ★ 必须把子集的**局部下标**映射回全集的全局下标，否则 probs_fn 会取错样本
            vals.append(hybrid_metrics_from_tokens(
                sub, 0.50, lambda j, fn=fn, idx=idx: fn(idx[j]))["line_f1"])
        print(f"{'[' + str(lo) + ',' + hi_s + ')':>16}{len(idx):>6}"
              f"{vals[0]:>13.4f}{vals[1]:>11.4f}{vals[2]:>16.4f}")
    print("\n★ 局限：m4（文档级）这一边**预测不了** —— 它取决于「主干是否回到 v0.4.4 的受力状态」，"
          "那是机制假设，只能靠那次训练验。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
