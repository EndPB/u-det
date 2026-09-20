"""v0.4.6 自检：长度感知的 token 尺度权重（改动 1）+ 报告改文档级向量（改动 2）。

重点不是"能跑"，而是**向后兼容**：
    1. `weight_mode="static"`（默认）必须与历史**逐位一致** —— 否则 v0.4.5 的 ckpt 不可复现；
    2. `SampleHead(report_dim=0)` 必须与历史**逐位一致** —— 否则老 ckpt 加载行为会变；
    3. `report.mode="vector"` 必须**不改** input_ids / tok_labels —— 这是本次改动的核心承诺。

用法::

    OMP_NUM_THREADS=8 python scripts/check_v046.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.heads import SampleHead                    # noqa: E402
from report import build_report                        # noqa: E402
from report.handcrafted import VECTOR_KEYS             # noqa: E402
from train import scale_weights_length, token_loss     # noqa: E402

OK, FAIL = "\033[32mOK\033[0m", "\033[31mFAIL\033[0m"
bad = 0


def check(name: str, cond: bool, extra: str = "") -> None:
    global bad
    if not cond:
        bad += 1
    print(f"  [{OK if cond else FAIL}] {name}" + (f"   {extra}" if extra else ""))


def norm(ws):
    s = sum(ws) or 1.0
    return [w / s for w in ws]


# --------------------------------------------------------------------------- #
def t1_weight_function() -> None:
    print("\n=== 1. 权重函数 w_i = σ((log2 n_i − log2 N_min)/τ) ===")
    ns = [torch.zeros(1, 1, k) for k in (8, 16, 32, 128, 512)]
    w = scale_weights_length(ns, nmin=64.0, tau=0.5)
    # 与探针脚本里的同名公式必须一致（两处实现、一个定义）
    lo = math.log2(64.0)
    ref = [1 / (1 + math.exp(-((math.log2(max(k, 1)) - lo) / 0.5))) for k in (8, 16, 32, 128, 512)]
    check("与探针脚本的公式逐位一致", all(abs(a - b) < 1e-15 for a, b in zip(w, ref)),
          f"w={[round(x, 4) for x in w]}")
    check("N_min 处权重为 0.5", abs(1 / (1 + math.exp(-((math.log2(64) - lo) / 0.5))) - 0.5) < 1e-12)

    # 形状：短样本 n_i 小 ⇒ 粗尺度被压低；归一化后向细尺度集中
    # ★ 期望值来自实际采用的 N_min=64 / τ=0.5（不是 N_min=8 那张表）
    for L, want_fine in ((64, 0.965), (293, 0.608), (1024, 0.382)):
        ns2 = [torch.zeros(1, 1, max(1, math.ceil(L / s))) for s in (64, 32, 16, 4, 1)]
        nw = norm(scale_weights_length(ns2, 64.0, 0.5))
        check(f"L={L}: 归一化权重形状", abs(nw[-1] - want_fine) < 0.03,
              f"由粗到细 {[round(x, 3) for x in nw]}")
        check(f"L={L}: 权重随尺度变细单调不减",
              all(nw[i] <= nw[i + 1] + 1e-9 for i in range(len(nw) - 1)))


def t2_token_loss_static_identical() -> None:
    print("\n=== 2. token_loss：static 模式与历史逐位一致 ===")
    torch.manual_seed(0)
    L = 512
    tok = torch.randint(0, 2, (1, L))
    tok[0, : 20] = -100                                     # 模拟报告前缀被忽略
    scales = [torch.randn(1, 1, max(1, math.ceil(L / s))) for s in (64, 32, 16, 4, 1)]
    w = [0.2, 0.4, 0.6, 0.8, 1.0]
    a = token_loss(scales, tok, w, modes=["lse", "lse", "lse", "mean", "mean"], beta=4.0)
    b = token_loss(scales, tok, w, modes=["lse", "lse", "lse", "mean", "mean"], beta=4.0,
                   weight_mode="static")
    check("省略 weight_mode == 'static'（逐位）", float((a - b).abs()) == 0.0,
          f"static={float(a):.10f}")
    c = token_loss(scales, tok, w, modes=["lse", "lse", "lse", "mean", "mean"], beta=4.0,
                   weight_mode="length", nmin=64.0, tau=0.5)
    check("length 模式改变了损失（说明真的生效）", abs(float(a - c)) > 1e-9,
          f"length={float(c):.10f}  差={float(c - a):+.6f}")
    # length 模式应等价于「显式传入同一套权重」
    wl = scale_weights_length(scales, 64.0, 0.5)
    d = token_loss(scales, tok, wl, modes=["lse", "lse", "lse", "mean", "mean"], beta=4.0)
    check("length 模式 == 显式传入 scale_weights_length 的结果（逐位）",
          float((c - d).abs()) == 0.0)
    try:
        token_loss(scales, tok, w, weight_mode="nope")
        check("非法 weight_mode 应当报错", False)
    except ValueError:
        check("非法 weight_mode 报错", True)


def t3_sample_head_backcompat() -> None:
    print("\n=== 3. SampleHead：report_dim=0 与历史逐位一致 ===")
    torch.manual_seed(0)
    h = SampleHead(768, hidden=256, n_levels=5)
    check("report_dim 默认 0", h.report_dim == 0)
    check("宽度 = dim*n_levels（未被加宽）", h.width == 768 * 5)
    xs = [torch.randn(1, k, 768) for k in (8, 16, 32, 128, 512)]
    o1 = h(xs)
    o2 = h(xs, report=None)
    check("不传 report 与显式 None 逐位一致", float((o1 - o2).abs().max()) == 0.0)
    try:
        h(xs, report=torch.zeros(1, 7))
        check("report_dim=0 但传了 report ⇒ 应报错（防静默忽略）", False)
    except ValueError:
        check("report_dim=0 时传 report 报错（防静默忽略）", True)

    h2 = SampleHead(768, hidden=256, n_levels=5, report_dim=7, report_proj=32)
    check("report_dim=7 时宽度 +32", h2.width == 768 * 5 + 32)
    n_add = sum(p.numel() for p in h2.parameters()) - sum(p.numel() for p in h.parameters())
    check("参数增量 < 10K", n_add < 10_000, f"实际 +{n_add} 参数")
    try:
        h2(xs)
        check("report_dim>0 时不传 report ⇒ 应报错", False)
    except ValueError:
        check("report_dim>0 时缺 report 报错", True)
    try:
        h2(xs, report=torch.zeros(1, 5))
        check("report 维度不符 ⇒ 应报错", False)
    except ValueError:
        check("report 维度不符报错", True)
    o3 = h2(xs, report=torch.randn(1, 7))
    check("report_dim>0 时前向正常", o3.shape == (1, 2), f"out={tuple(o3.shape)}")


def t4_report_vector_mode() -> None:
    print("\n=== 4. report.mode='vector'：报告不进序列 ===")
    cfg = yaml.safe_load(open("configs/udet_v046.yaml"))
    rcfg = cfg["report"]
    rep = build_report(rcfg["name"], max_tokens=rcfg.get("max_tokens", 64),
                       stats_mean=rcfg["stats_mean"], stats_std=rcfg["stats_std"])
    code = "def foo(a, b):\n    if a:\n        return  b\n\n\n"
    v = rep.vector(code)
    check("vector 长度 == 7", len(v) == 7, f"{[round(x, 3) for x in v]}")
    check("全部截断在 ±5 内", all(-5.0 <= x <= 5.0 for x in v))
    check("VECTOR_KEYS 顺序固定", VECTOR_KEYS == ("void", "indent", "tail", "charh",
                                                  "bigh", "namev", "linev"))

    from dataio import build_dataset
    from transformers import AutoTokenizer
    from train import resolve
    tok = AutoTokenizer.from_pretrained(str(resolve(cfg["encoder"]["path"])))
    rep_tok = build_report(rcfg["name"], tokenizer=tok, max_tokens=rcfg.get("max_tokens", 64),
                           stats_mean=rcfg["stats_mean"], stats_std=rcfg["stats_std"])
    kw = dict(file=str(resolve(cfg["data"]["processed_dir"]) / cfg["data"]["m4_file"]),
              split="val", train=False)
    ds_pre = build_dataset("m4", report=rep_tok, report_mode="prefix", **kw)
    ds_vec = build_dataset("m4", report=rep, report_mode="vector", **kw)
    ds_non = build_dataset("m4", report=None, report_mode="vector", **kw)
    ip, iv, ino = ds_pre[0], ds_vec[0], ds_non[0]
    check("vector 模式的 ids == 不带报告时的 ids（报告没进序列）",
          iv["input_ids"] == ino["input_ids"])
    check("prefix 模式的 ids 更长（确实加了前缀）",
          len(ip["input_ids"]) > len(iv["input_ids"]),
          f"prefix={len(ip['input_ids'])} vector={len(iv['input_ids'])}")
    check("vector 模式给出 report 向量", iv["report"] is not None and len(iv["report"]) == 7)
    check("prefix 模式不给 report 向量", ip["report"] is None)
    check("vector 模式的 tok_labels 与不带报告时一致", iv["tok_labels"] == ino["tok_labels"])
    # tok_labels 的「前缀屏蔽」要在**有 token 标签的数据集**上验（m4 的 tok_labels 恒为 None）
    kwh = dict(file=str(resolve(cfg["data"]["processed_dir"]) / cfg["data"]["hybrid_file"]),
               split="val", train=False)
    hp = build_dataset("hybrid", report=rep_tok, report_mode="prefix", **kwh)[0]
    hv = build_dataset("hybrid", report=rep_tok, report_mode="vector", **kwh)[0]
    hn = build_dataset("hybrid", report=None, report_mode="vector", **kwh)[0]
    n_pre = len(hp["input_ids"]) - len(hv["input_ids"])
    check("hybrid prefix 模式：前缀 tok_labels 全为 -100",
          n_pre > 0 and hp["tok_labels"][:n_pre] == [-100] * n_pre, f"前缀 {n_pre} 个 token")
    check("hybrid vector 模式：tok_labels 与「不带报告」完全一致",
          hv["tok_labels"] == hn["tok_labels"])
    check("hybrid vector 模式：ids 与「不带报告」完全一致",
          hv["input_ids"] == hn["input_ids"])

    from dataio import collate
    b = collate([iv])
    check("collate 产出 (1,7) 的 report 张量",
          b["report"] is not None and tuple(b["report"].shape) == (1, 7))
    bp = collate([ip])
    check("prefix 模式下 collate 的 report 为 None", bp["report"] is None)


def main() -> int:
    t1_weight_function()
    t2_token_loss_static_identical()
    t3_sample_head_backcompat()
    t4_report_vector_mode()
    print(f"\n{'=' * 60}\n{'全部通过' if bad == 0 else f'{bad} 项失败'}\n{'=' * 60}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
