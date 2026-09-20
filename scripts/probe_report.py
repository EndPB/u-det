"""零训练探针 P3 / P4：**报告统计量的信息量** + **报告污染的量化**。

P3  只用那 7 维手工统计量做**闭式线性探针**，预测文档级标签（train 拟合 → val/test 评）。
    问：改动 2（把报告喂给文档级头）有没有指望？
    ★ 必须**按长度分桶**看，否则会被 C2「长度—标签混淆」骗到（纯长度规则 val acc 就有 0.659）。
    另外看**互补性**：在 v0.4.5 已经分错的样本上，探针是否更准。

P4  报告前缀在瓶颈里占多少带宽：ceil(n_rep/64) / ceil(L/64)，按长度分桶。
    这是纯算术，不需要模型。

全程 CPU，不占 GPU。

用法::

    OMP_NUM_THREADS=8 python scripts/probe_report.py --config configs/udet_v045.yaml
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from report import build_report                       # noqa: E402
from report.handcrafted import stats as hstats        # noqa: E402
from train import make_dataset                        # noqa: E402

KEYS = ["void", "indent", "tail", "charh", "bigh", "namev", "linev"]
BUCKETS = [(0, 256), (256, 512), (512, 1024), (1024, 2048), (2048, 1 << 30)]


def collect(cfg: dict, split: str, tokenizer=None):
    """返回 (X 7 维统计量, y 标签, L 代码长度, n_rep 报告 token 数)。

    ``tokenizer=None`` 时 n_rep 一律为 0（P4 会拿真 tokenizer 另算）。
    """
    rep = build_report(cfg["report"]["name"], tokenizer=tokenizer)
    ds = make_dataset(cfg, "m4", split, None, False, None)      # report=None ⇒ ids 就是纯代码
    X, y, L, nr = [], [], [], []
    for i in range(len(ds)):
        it = ds[i]
        s = hstats(it["code"])
        X.append([float(s[k]) for k in KEYS])
        y.append(int(it["label"]))
        L.append(len(it["input_ids"]))
        nr.append(len(rep.ids(it["code"])) if tokenizer is not None else 0)
    return (torch.tensor(X, dtype=torch.float64), torch.tensor(y),
            torch.tensor(L), torch.tensor(nr))


def ridge_fit(X: torch.Tensor, y: torch.Tensor, lam: float = 1e-2):
    """闭式最小二乘（与项目里「闭式线性底线」同一套做法）：标准特征 + 偏置，目标 ±1。

    返回 (w, mu, sd)；预测时必须复用同一套标准化参数。
    """
    mu, sd = X.mean(0), X.std(0).clamp(min=1e-9)
    Z = torch.cat([(X - mu) / sd, torch.ones(X.shape[0], 1, dtype=X.dtype)], 1)
    t = (y.to(X.dtype) * 2 - 1)
    A = Z.T @ Z + lam * torch.eye(Z.shape[1], dtype=X.dtype)
    w = torch.linalg.solve(A, Z.T @ t)
    return w, mu, sd


def ridge_pred(X: torch.Tensor, pack) -> torch.Tensor:
    w, mu, sd = pack
    Z = torch.cat([(X - mu) / sd, torch.ones(X.shape[0], 1, dtype=X.dtype)], 1)
    return (Z @ w) > 0


def acc(pred: torch.Tensor, y: torch.Tensor) -> float:
    return float((pred == y.bool()).float().mean())


def main() -> int:
    ap = argparse.ArgumentParser(description="P3/P4 报告探针")
    ap.add_argument("--config", default="configs/udet_v045.yaml")
    ap.add_argument("--run", default="runs/v0.4.5")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(cfg["encoder"]["path"]))

    data = {}
    for split in ("train", "val", "test"):
        data[split] = collect(cfg, split, tokenizer)
        print(f"[probe] m4/{split}: {len(data[split][1])} 条")

    Xtr, ytr, Ltr, _ = data["train"]
    pack = ridge_fit(Xtr, ytr)

    print("\n" + "=" * 76)
    print("P3｜只用 7 维手工统计量的闭式线性探针（文档级）")
    print("=" * 76)
    print(f"{'split':<8}{'n':>6}{'acc':>10}   | 按长度分桶 acc")
    for split in ("val", "test"):
        X, y, L, _ = data[split]
        p = ridge_pred(X, pack)
        print(f"{split:<8}{len(y):>6}{acc(p, y):>10.4f}")
    Xte, yte, Lte, _ = data["test"]
    pte = ridge_pred(Xte, pack)
    print(f"\n{'长度区间':<16}{'n':>6}{'探针 acc':>11}{'原版规则(长度)acc':>18}")
    for lo, hi in BUCKETS:
        m = (Lte >= lo) & (Lte < hi)
        if int(m.sum()) == 0:
            continue
        hi_s = "∞" if hi > (1 << 29) else str(hi)
        len_rule = (Lte[m] < 1024)                     # 粗糙的"纯长度"对照：短=ai
        print(f"[{lo},{hi_s})".ljust(16) + f"{int(m.sum()):>6}{acc(pte[m], yte[m]):>11.4f}"
              f"{acc(len_rule, yte[m]):>18.4f}")
    print("（若探针 acc 与「纯长度规则」相当 ⇒ 它学到的只是长度，不是风格）\n")

    # ---- 互补性：在 v0.4.5 已经分错的样本上，探针是否更准 ----
    raw = Path(args.run) / "raw_m4_test.pt"
    if raw.exists():
        blob = torch.load(raw, map_location="cpu", weights_only=False)
        prob = blob["sample_prob"].float()
        if prob.dim() > 1:
            prob = prob[:, 1]
        lab = blob["label"].long()
        wrong = (prob > 0.5).long() != lab
        print("=" * 76)
        print("P3b｜互补性：只在 v0.4.5 **分错**的样本上看")
        print("=" * 76)
        print(f"v0.4.5 test acc = {acc((prob > 0.5), lab):.4f}，错 {int(wrong.sum())}/{len(lab)} 条")
        if int(wrong.sum()) > 0:
            print(f"这些错样本里探针 acc = {acc(pte[wrong], yte[wrong]):.4f}"
                  f"（随机=0.5；≥0.65 才算有互补信息）")
        print()
    else:
        print(f"[probe] 没有 {raw}，跳过互补性分析\n")

    # ---------------- P4 ----------------
    print("=" * 76)
    print("P4｜报告前缀在**瓶颈**里占的带宽比  ceil(n_rep/64) / ceil(L/64)")
    print("=" * 76)
    rep = build_report(cfg["report"]["name"], tokenizer=tokenizer)
    print("（当前 report.mode 是前缀 token；这里用真 tokenizer 实算 n_rep）")
    print(f"{'长度区间':<16}{'n':>6}{'中位 L':>9}{'中位 n_rep':>11}{'瓶颈占比中位':>14}{'≥50% 的比例':>13}")
    for lo, hi in BUCKETS:
        m = (Lte >= lo) & (Lte < hi)
        if int(m.sum()) == 0:
            continue
        hi_s = "∞" if hi > (1 << 29) else str(hi)
        nrep = data["test"][3][m]
        lv = Lte[m]
        if int(nrep.max()) == 0:
            print(f"[{lo},{hi_s})".ljust(16) + f"{int(m.sum()):>6}{'--':>9}{'--':>11}"
                  f"{'(需 tokenizer，见下)':>14}")
            continue
        frac = torch.ceil(nrep.float() / 64) / torch.ceil(lv.float() / 64).clamp(min=1)
        print(f"[{lo},{hi_s})".ljust(16) + f"{int(m.sum()):>6}{int(lv.median()):>9}"
              f"{int(nrep.median()):>11}{float(frac.median()):>14.3f}"
              f"{float((frac >= 0.5).float().mean()):>13.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
