"""零训练探针 P10：**直接检验 §8.11.6 的机制假设**（「去掉捷径能不能救回文档级」）。

机制假设（§8.11.6）：
    旁路让最细尺度的 token 头**直接拿到编码器 feats** ⇒ 主干那一级不必再保留
    细粒度信息 ⇒ 削弱了「token 级任务替主干做**信息保值**」的正则化 ⇒ 文档级掉分。

这个假设有一个**零训练就能查的直接推论**：
    ⇒ v0.4.5 的**主干**（up 路径最细那一级 `levels[4]`）里，"可线性解码"的
      token 级信息应当**少于** v0.4.4（v0.4.4 没有旁路，主干一直在承受 token 级压力）。

而 `levels[4]` → token 头正是一个**线性读出**（`Linear(768→1)`），
所以"线性探针"测的就是那个头**能做到的上限** —— 这是最贴题的度量。

同理，文档级那一侧：v0.4.4 的**瓶颈**（`levels[0]`）里可线性解码的文档级信息
应当**不少于** v0.4.5（因为 v0.4.4 的主干一直在为样本级保值）。

三种可能的结局（都能被本探针区分）：
    (i)  v0.4.5 的 `levels[4]` token 探针明显更低 ⇒ **机制成立**，
        那么「额外一路头」（v0.4.9）值得跑：它把压力还回主干；
    (ii) 两者差不多 ⇒ 机制**不成立**，文档级的损失另有原因
        （例如"多出来的那 769 个参数 / 头的干扰"），v0.4.9 就不该指望 m4 回来；
    (iii) v0.4.5 反而更高 ⇒ 旁路让主干**更好**了，机制方向就是错的。

用法::

    OMP_NUM_THREADS=8 python scripts/probe_mechanism.py \
        --ckpts runs/v0.4.4/best.pt runs/v0.4.5/best.pt
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_raw import hybrid_metrics_from_tokens                      # noqa: E402
from probe_layers import RidgeStream, encoder_states                    # noqa: E402
from train import (IGNORE, build_model, make_dataset, make_report,      # noqa: E402
                   resolve, setup_cuda)
from transformers import AutoTokenizer                                  # noqa: E402


def auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    scores, labels = scores.double(), labels.bool()
    npos, nneg = int(labels.sum()), int((~labels).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    uniq, inv, cnt = torch.unique(scores, return_inverse=True, return_counts=True)
    cum = cnt.cumsum(0).double()
    ranks = ((cum - cnt.double() + cum + 1.0) / 2.0)[inv]
    return float((ranks[labels].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def prf(scores: torch.Tensor, labels: torch.Tensor, thr: float) -> float:
    pred, gold = scores > thr, labels.bool()
    tp = float((pred & gold).sum())
    fp = float((pred & ~gold).sum())
    fn = float((~pred & gold).sum())
    p, r = tp / max(tp + fp, 1e-9), tp / max(tp + fn, 1e-9)
    return 2 * p * r / max(p + r, 1e-9)


@torch.no_grad()
def _levels(model, enc, ids):
    """一次前向，返回 ``(levels, feats)``；levels 由粗到细。"""
    feats = enc(ids)
    levels, _ = model.backbone(feats, return_features=True)
    return levels, feats


def _level_targets(tok: torch.Tensor, length_k: int):
    """把 ``(L,)`` 的逐 token 标签/有效性按窗口聚合成 ``(L_k,)`` 的**硬标签 + keep 掩码**。

    ★ 必须逐尺度聚合：主干只有**最细**那一级是 stride 1（长度 = L），
      粗尺度（跨度 64/32/16/4）的长度是 ``ceil(L/s)``。
      直接用 token 级的掩码去索引粗尺度会立刻报
      ``IndexError: The shape of the mask [2299] ... does not match ... [36, 768]``
      （本脚本第一版就是这么炸的）。
    聚合方式：窗口内**过半为 AI** 记 +1（与 `train._aggregate_target` 的取向一致）。
    """
    L = tok.numel()
    k = max(1, math.ceil(L / length_k))
    pad = length_k * k - L
    v = (tok != IGNORE).float()
    t = tok.clamp(min=0).float()
    if pad:
        v, t = F.pad(v, (0, pad)), F.pad(t, (0, pad))
    den = F.avg_pool1d(v.view(1, 1, -1), k, k).view(-1)
    pos = F.avg_pool1d(t.view(1, 1, -1), k, k).view(-1)
    keep = den > 0
    frac = pos / den.clamp(min=1e-9)
    return torch.where(frac > 0.5, 1.0, -1.0), keep


@torch.no_grad()
def probe_hybrid_token(model, enc, cfg, report, device, n_hy, split):
    """逐尺度线性探针（**流式**）：第 k 个尺度的特征 → 该尺度上的 token 标签。

    返回 ``(per_level_auc, finest_metrics)``：

    * ``per_level_auc``：在**该尺度自己的分辨率**上算 AUC（跨尺度 / 跨 ckpt 可比）；
    * ``finest_metrics``：最细尺度（stride 1，长度 = L）上采样回原长度后的
      行级 / 片段级 / token 级 F1 —— 这是 token 头**真正**的输入，直接对标它的上限。

    ★ 必须流式：最细尺度的特征是 (L, 768)，L 可到 1 万 ⇒ 存 1500 条会 OOM。
    """
    n_lv = model.backbone.depth + 1
    ds_tr = make_dataset(cfg, "hybrid", "train", report, True, n_hy or None)
    acc = [RidgeStream(enc.hidden_size) for _ in range(n_lv)]
    for i in range(len(ds_tr)):
        it = ds_tr[i]
        ids = torch.tensor([it["input_ids"]], dtype=torch.long, device=device)
        tok = torch.tensor(it["tok_labels"], dtype=torch.long).reshape(-1)
        levels, _ = _levels(model, enc, ids)
        for k in range(n_lv):
            y, keep = _level_targets(tok, levels[k].shape[1])
            keep = keep.to(device)
            if bool(keep.any()):
                acc[k].add(levels[k][0][keep], y.to(device)[keep])
        if (i + 1) % 400 == 0:
            print(f"   fit [{i + 1}/{len(ds_tr)}]", flush=True)
    packed = [RidgeStream.to_device(a.fit(1e-2), device) for a in acc]

    ds_te = make_dataset(cfg, "hybrid", split, report, False, None)
    blob = {"n": len(ds_te), "token_label": [], "line_of_token": [], "line_label": []}
    sc_lv = [[] for _ in range(n_lv)]
    y_lv = [[] for _ in range(n_lv)]
    finest = []
    for i in range(len(ds_te)):
        it = ds_te[i]
        ids = torch.tensor([it["input_ids"]], dtype=torch.long, device=device)
        tok = torch.tensor(it["tok_labels"], dtype=torch.long).reshape(-1)
        levels, _ = _levels(model, enc, ids)
        for k in range(n_lv):
            y, keep = _level_targets(tok, levels[k].shape[1])
            keep = keep.to(device)
            s = acc[k].score(levels[k][0][keep], packed[k]).cpu()
            sc_lv[k].append(s)
            y_lv[k].append(y[keep.cpu()])
        # 最细尺度：铺回原长度，供行级/片段级用
        valid = (tok != IGNORE)
        full = torch.zeros(tok.numel())
        full[valid.cpu()] = sc_lv[-1][-1]
        finest.append(full)
        blob["token_label"].append(tok.cpu())
        blob["line_of_token"].append(torch.tensor(it["line_of_token"], dtype=torch.long))
        blob["line_label"].append(torch.tensor(it["line_label"], dtype=torch.long))
        if (i + 1) % 400 == 0:
            print(f"   score [{i + 1}/{len(ds_te)}]", flush=True)

    per_level_auc = {k: auc(torch.cat(sc_lv[k]), (torch.cat(y_lv[k]) > 0).long())
                     for k in range(n_lv)}
    finest_metrics = hybrid_metrics_from_tokens(
        blob, 0.50, lambda i: torch.sigmoid(finest[i]))
    return per_level_auc, finest_metrics


def probe_m4_doc(model, enc, cfg, report, device, n_m4):
    """瓶颈（levels[0]）→ 文档级标签的线性探针（均值池化后），返回 AUC / F1。"""
    def collect(split, limit):
        ds = make_dataset(cfg, "m4", split, report, False, limit or None)
        pooled, labels = [], []
        for i in range(len(ds)):
            it = ds[i]
            ids = torch.tensor([it["input_ids"]], dtype=torch.long, device=device)
            levels, _ = _levels(model, enc, ids)
            pooled.append(levels[0][0].mean(0).detach())
            labels.append(int(it["label"]))
            if (i + 1) % 1000 == 0:
                print(f"   m4/{split} [{i + 1}/{len(ds)}]", flush=True)
        return pooled, labels

    tr_x, tr_y = collect("train", n_m4)
    acc = RidgeStream(enc.hidden_size)
    for x, y in zip(tr_x, tr_y):
        acc.add(x.unsqueeze(0), torch.tensor([1.0 if y == 1 else -1.0], device=device))
    packed = RidgeStream.to_device(acc.fit(1e-2), device)
    te_x, te_y = collect("test", 0)
    sc = torch.cat([acc.score(x.unsqueeze(0), packed).cpu() for x in te_x])
    y = torch.tensor(te_y)
    thr = float(torch.quantile(sc.float(), 1 - packed["p_pos"]))
    return {"auc": auc(sc, y), "f1": prf(sc, y, thr), "n": len(te_y)}


def main() -> int:
    ap = argparse.ArgumentParser(description="P10：检验「旁路削弱信息保值」的机制假设")
    ap.add_argument("--ckpts", nargs="+", default=["runs/v0.4.4/best.pt", "runs/v0.4.5/best.pt"])
    ap.add_argument("--config", default=None, help="默认取每个 ckpt 同目录的 config.yaml")
    ap.add_argument("--limit-hybrid-train", type=int, default=1200)
    ap.add_argument("--limit-m4-train", type=int, default=2000)
    ap.add_argument("--split", default="test", choices=["test", "val"])
    ap.add_argument("--skip-m4", action="store_true")
    args = ap.parse_args()

    setup_cuda()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = {}
    for ck in args.ckpts:
        cfg_path = args.config or str(Path(resolve(ck)).parent / "config.yaml")
        cfg = yaml.safe_load(open(str(resolve(cfg_path)), encoding="utf-8"))
        cfg["encoder"]["compile"] = False                     # 探针不需要，省掉编译时间
        tokenizer = AutoTokenizer.from_pretrained(str(resolve(cfg["encoder"]["path"])))
        report = make_report(cfg, tokenizer)
        model, enc = build_model(cfg, tokenizer)
        blob = torch.load(resolve(ck), map_location="cpu", weights_only=False)
        model.load_state_dict(blob["state"], strict=False)
        model.to(device).eval()
        print(f"\n[P10] ==== {ck}（epoch {blob.get('epoch')}，config={cfg_path}）====")
        hy_auc, hy_fin = probe_hybrid_token(model, enc, cfg, report, device,
                                            args.limit_hybrid_train, args.split)
        row = {"auc": hy_auc, "finest": hy_fin}
        if not args.skip_m4:
            row["m4"] = probe_m4_doc(model, enc, cfg, report, device, args.limit_m4_train)
        results[ck] = row
        del model
        torch.cuda.empty_cache()

    # ---------------- 报表 ---------------- #
    print("\n" + "=" * 92)
    print("P10｜主干各尺度的「→ token 标签」线性探针（hybrid test）")
    print("      levels 由粗到细：跨度 64 / 32 / 16 / 4 / 1")
    print("=" * 92)
    heads = list(results)
    n_lv = len(results[heads[0]]["auc"])
    stride = [64, 32, 16, 4, 1][:n_lv]
    print(f"{'尺度(跨度)':>12}" + "".join(f"{h:>22}" for h in heads))
    print(f"{'':>12}" + "".join(f"{'AUC（本尺度分辨率）':>22}" for _ in heads))
    for k in range(n_lv):
        line = f"{'L/' + str(stride[k]):>12}"
        for h in heads:
            line += f"{results[h]['auc'][k]:>22.4f}"
        print(line)
    print("\n" + "=" * 92)
    print("P10b｜★ 最细尺度（跨度 1 = token 头真正的输入）上采样回原长度后的 F1")
    print("=" * 92)
    print(f"{'ckpt':>28}{'行级 F1':>12}{'片段 F1':>12}{'token F1':>12}")
    for h in heads:
        m = results[h]["finest"]
        print(f"{h:>28}{m['line_f1']:>12.4f}{m['chunk_f1']:>12.4f}{m['token_f1']:>12.4f}")
    print("★ 机制假设（§8.11.6）预测：**有旁路**的那一版在最细尺度上**更低**"
          "（主干被免除了保值压力）。")

    if not args.skip_m4:
        print("\n" + "=" * 92)
        print("P10c｜瓶颈（levels[0]，均值池化）→ 文档级标签 线性探针（m4 test）")
        print("=" * 92)
        print(f"{'ckpt':>28}{'AUC':>10}{'F1':>10}{'n':>8}")
        for h in heads:
            m = results[h]["m4"]
            print(f"{h:>28}{m['auc']:>10.4f}{m['f1']:>10.4f}{m['n']:>8}")
        print("★ 机制假设预测 —— **没有旁路**的那一版在瓶颈上**更强**（它一直在为样本级保值）。")
    print("\n⚠️ 这是**冻结模型上的线性读出**，测的是「信息还在不在」，不是「训练后谁更好」。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
