"""特征侧归因诊断：加 s2 流为什么会拖低 s1？（零训练，纯用已有 checkpoint）

候选机制（对应 v0.2 文档待写的「归因」一节）：
  H1 表征冲突：pair 铰链经共享 LoRA 改编码器几何 → s1 可用信息受损；
  H2 读出冲突：cos² 正交正则 + w2 生长 → w1 被挤离判别方向；
  H3 尺度失配：margin=1.0 相对 ‖Δh‖ 自然尺度过大 → 表示被过度形变。

测量（每个 ckpt 独立，全部零训练）：
  1) oracle 读出：在冻结 h 上训 LR 探针解 m4 —— full 与 head/mid/tail 片段分别评估
     → 「特征里还剩多少可线性解码的人机信息」；片段上的损失 = 表征损伤直接证据；
  2) 方向几何：cos(w1, m4-probe 方向)、cos(w1, pair-probe 方向)、
     cos(m4-probe, pair-probe)（H2：正交约束把 w1 挤开多远）；
  3) Δh 尺度：‖h₊−h₋‖ 分布 vs margin=1.0 与 ‖w2‖（H3）；
  4) pair 探针 AUC / dir_acc（在特征上直接拟合 base/instruct 方向的可行性 = "s2 探针化"前置检查）；
  5) 梯度冲突快照：固定 batch 上 cos(∇L1, ∇L2)_enc、‖∇L1‖、‖∇L2‖（λb / PCGrad 方向）。

用法：
  OMP_NUM_THREADS=32 python scripts/diag_features.py --cpu \
      --runs pretrained runs/v0.1.0_ab_nopair runs/v0.1.0 runs/v0.2.2_ab_abmil
输出：每个 run 目录写 diag_features.json；stdout 汇总表。

口径：m4 = 预分词 ids（add_special_tokens=False）；pair = 训练同款 input_ids_*；
     片段 = tokenizer(add_special_tokens=False)（与 probe_fragments 修正后一致）。
"""
from __future__ import annotations

import argparse
import gc
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from dataio import build_dataset              # noqa: E402
from encoders import build_encoder            # noqa: E402
from models import build_model                # noqa: E402
from probe_fragments import make_fragments    # noqa: E402


def load_run(tag: str, device: str):
    cfg_path = (ROOT / "configs/ddet_base.yaml") if tag == "pretrained" \
        else (ROOT / "runs" / tag / "config.yaml")
    cfg = yaml.safe_load(open(cfg_path, encoding="utf-8"))
    enc_cfg = dict(cfg["encoder"])
    enc_cfg["path"] = str((ROOT / enc_cfg["path"]).resolve())
    if device == "cpu":
        enc_cfg["compile"] = False               # CPU 上 torch.compile 无益
    enc = build_encoder(**enc_cfg)
    mcfg = dict(cfg["model"])
    model = build_model(mcfg.pop("name", "dual"), encoder=enc, dim=enc.hidden_size, **mcfg)
    if tag != "pretrained":
        ck = torch.load(ROOT / "runs" / tag / "best.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(ck["state"], strict=False)
    model.to(device).eval()
    tok = AutoTokenizer.from_pretrained(enc_cfg["path"])
    print(f"    [{tag}] 模型载入完成（{device}）", flush=True)
    return model, tok, cfg


@torch.no_grad()
def feat_of(model, tok, code_or_ids, device):
    if isinstance(code_or_ids, str):
        ids = tok(code_or_ids, return_tensors="pt",
                  add_special_tokens=False)["input_ids"].to(device)
    else:
        ids = torch.tensor([code_or_ids], dtype=torch.long, device=device)
    return model.features(ids, torch.ones_like(ids))[0].float().cpu().numpy()


def pick_strat(labels, n, rng):
    idx0 = [i for i, y in enumerate(labels) if y == 0]
    idx1 = [i for i, y in enumerate(labels) if y == 1]
    rng.shuffle(idx0)
    rng.shuffle(idx1)
    k0 = min(len(idx0), n // 2)
    k1 = min(len(idx1), n - k0)
    return idx0[:k0] + idx1[:k1]


def raw_coef(clf):
    """把标准化空间里的 LR 系数换回原始特征空间方向。"""
    lr = clf.named_steps["logisticregression"]
    sc = clf.named_steps["standardscaler"]
    return (lr.coef_.reshape(-1) / np.maximum(sc.scale_, 1e-12)).astype(np.float64)


def cos(u, v):
    u = np.asarray(u, dtype=np.float64).ravel()
    v = np.asarray(v, dtype=np.float64).ravel()
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu < 1e-12 or nv < 1e-12:
        return float("nan")
    return float(u @ v / (nu * nv))


def _pad(seqs, pad_id, device):
    m = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), m), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros(len(seqs), m, dtype=torch.long, device=device)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = torch.tensor(s, dtype=torch.long, device=device)
        mask[i, :len(s)] = 1
    return ids, mask


def grad_snapshot(model, cfg, tok, device, ds_m4, ds_pr, n=16):
    """固定 mini-batch 上分别求 L1 / L2 的编码器梯度（逐条累积；峰值 = 单样本图）。

    逐条 `autograd.grad`：不触碰 `.grad`（训练中安全），长样本也不会撑爆内存
    （容器 cgroup 限额 30GB；整批 pad 一次回传在长样本上实测被 OOM kill）。
    m4 / pair 训练集由调用方传入（避免重复整体载入）。
    """
    enc_params = [p for nm, p in model.named_parameters()
                  if nm.startswith("encoder.") and p.requires_grad]
    if not enc_params:
        return None
    im = list(range(0, len(ds_m4), max(1, len(ds_m4) // n)))[:n]
    ip = list(range(0, len(ds_pr), max(1, len(ds_pr) // n)))[:n]

    def _one(seq):
        return torch.tensor([seq], dtype=torch.long, device=device)

    def _acc(loss, acc):
        grads = torch.autograd.grad(loss, enc_params, allow_unused=True)
        grads = [torch.zeros_like(p) if g is None else g for g, p in zip(grads, enc_params)]
        return grads if acc is None else [a + b for a, b in zip(acc, grads)]

    was = model.training
    model.eval()
    margin = float(cfg.get("loss", {}).get("margin", 1.0))
    g1, l1_sum = None, 0.0
    for i in im:
        ids = _one(ds_m4[i]["input_ids"])
        y = torch.tensor([float(ds_m4.labels[i])], dtype=torch.float32, device=device)
        l1i = F.binary_cross_entropy_with_logits(
            model(ids, torch.ones_like(ids))[0].float(), y)
        g1 = _acc(l1i, g1)
        l1_sum += float(l1i.detach())
    g2, l2_sum = None, 0.0
    for i in ip:
        ipd, imd = _one(ds_pr[i]["input_ids_plus"]), _one(ds_pr[i]["input_ids_minus"])
        sp = model(ipd, torch.ones_like(ipd))[1]
        sm = model(imd, torch.ones_like(imd))[1]
        d = (sp - sm).reshape(-1)
        gap = d.norm() if d.numel() > 1 else d[0]
        l2i = F.relu(margin - gap)
        g2 = _acc(l2i, g2)
        l2_sum += float(l2i.detach())
    if was:
        model.train()

    v1 = torch.cat([g.reshape(-1).float() for g in g1])
    v2 = torch.cat([g.reshape(-1).float() for g in g2])
    n1, n2 = float(v1.norm()), float(v2.norm())
    return {"cos_enc": (float((v1 @ v2) / (v1.norm() * v2.norm() + 1e-12))
                        if min(n1, n2) > 1e-12 else float("nan")),
            "g1_norm": n1, "g2_norm": n2,
            "ratio_g2_g1": (n2 / n1 if n1 > 1e-12 else float("nan")),
            "l1": l1_sum / max(len(im), 1), "l2": l2_sum / max(len(ip), 1)}


def run_one(tag: str, args, device: str):
    rng = random.Random(0)
    model, tok, cfg = load_run(tag, device)
    data = cfg["data"]
    out = {"run": tag, "pool": cfg["model"].get("pool"),
           "s2_rank": int(cfg["model"].get("s2_rank", 1))}

    # ---- m4 线性探针（oracle 读出） ----
    ds_tr = build_dataset("m4", file=str(ROOT / data["processed_dir"] / data["m4_file"]),
                          split="train")
    ds_te = build_dataset("m4", file=str(ROOT / data["processed_dir"] / data["m4_file"]),
                          split="test")
    print(f"    [{tag}] m4 数据集载入完成（tr={len(ds_tr)} te={len(ds_te)}）", flush=True)
    itr = pick_strat(ds_tr.labels, args.m4_n, rng)
    H_tr = np.stack([feat_of(model, tok, ds_tr[i]["input_ids"], device) for i in itr])
    y_tr = np.array([int(ds_tr.labels[i]) for i in itr])
    ite = pick_strat(ds_te.labels, args.m4_n, rng)
    H_te = np.stack([feat_of(model, tok, ds_te[i]["input_ids"], device) for i in ite])
    y_te = np.array([int(ds_te.labels[i]) for i in ite])
    clf_m4 = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
    clf_m4.fit(H_tr, y_tr)
    out["m4_probe"] = {
        "acc_te": float((clf_m4.predict(H_te) == y_te).mean()),
        "auc_te": float(roc_auc_score(y_te, clf_m4.decision_function(H_te))),
        "n_tr": len(itr), "n_te": len(ite)}
    d_m4 = raw_coef(clf_m4)

    # ---- 片段（同一 m4 探针直接打分；独立分层抽样） ----
    sel = pick_strat(ds_te.labels, args.frag_n, rng)
    groups = {"full": [], "head": [], "mid": [], "tail": []}
    for i in sel:
        frags = make_fragments(ds_te.codes[i], 8, 0.4)
        for name in groups:
            text = frags.get(name)
            if text is None or not text.strip():
                continue
            h = feat_of(model, tok, text, device)
            groups[name].append(int(clf_m4.predict(h[None])[0]) == int(ds_te.labels[i]))
    out["frag_probe"] = {k: {"acc": float(np.mean(v)) if v else None, "n": len(v)}
                         for k, v in groups.items()}

    # ---- pair 探针 + Δh 尺度 ----
    ds_pt = build_dataset("pair", file=str(ROOT / data["processed_dir"] / data["pair_file"]),
                          split="train")
    ds_pv = build_dataset("pair", file=str(ROOT / data["processed_dir"] / data["pair_file"]),
                          split="val")
    ds_px = build_dataset("pair", file=str(ROOT / data["processed_dir"] / data["pair_file"]),
                          split="test")
    print(f"    [{tag}] pair 数据集载入完成（tr={len(ds_pt)} va={len(ds_pv)} te={len(ds_px)}）",
          flush=True)
    ptr = list(range(0, len(ds_pt), max(1, len(ds_pt) // args.pair_n)))[:args.pair_n]
    Hp_tr = np.stack([feat_of(model, tok, ds_pt[i]["input_ids_plus"], device) for i in ptr])
    Hm_tr = np.stack([feat_of(model, tok, ds_pt[i]["input_ids_minus"], device) for i in ptr])
    nv = min(len(ds_pv), args.pair_eval_cap) if args.pair_eval_cap else len(ds_pv)
    nx = min(len(ds_px), args.pair_eval_cap) if args.pair_eval_cap else len(ds_px)
    Hp_list, Hm_list = [], []
    for i in range(nv):
        Hp_list.append(feat_of(model, tok, ds_pv[i]["input_ids_plus"], device))
        Hm_list.append(feat_of(model, tok, ds_pv[i]["input_ids_minus"], device))
    for i in range(nx):
        Hp_list.append(feat_of(model, tok, ds_px[i]["input_ids_plus"], device))
        Hm_list.append(feat_of(model, tok, ds_px[i]["input_ids_minus"], device))
    Hp_ev = np.stack(Hp_list)
    Hm_ev = np.stack(Hm_list)
    n_ev = nv + nx

    clf_pair = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
    clf_pair.fit(np.r_[Hp_tr, Hm_tr], np.r_[np.ones(len(ptr)), np.zeros(len(ptr))])
    d_pair = raw_coef(clf_pair)
    dec_p = clf_pair.decision_function(Hp_ev)
    dec_m = clf_pair.decision_function(Hm_ev)
    out["pair_probe"] = {
        "auc_ev": float(roc_auc_score(np.r_[np.ones(n_ev), np.zeros(n_ev)],
                                      np.r_[dec_p, dec_m])),
        "dir_acc_ev": float((dec_p > dec_m).mean()), "n_ev": n_ev}

    dlt = Hp_ev - Hm_ev
    norms = np.linalg.norm(dlt, axis=1)
    unit = dlt / np.maximum(norms[:, None], 1e-12)
    out["delta"] = {
        "norm_median": float(np.median(norms)), "norm_p90": float(np.percentile(norms, 90)),
        "consistency": float(np.linalg.norm(unit.mean(axis=0))), "n": n_ev}
    w1 = model.w1.weight.detach().cpu().numpy().reshape(-1).astype(np.float64)
    w2 = model.w2.weight.detach().cpu().numpy().reshape(model.s2_rank, -1).astype(np.float64)
    w2v = w2.reshape(-1)
    margin = float(cfg.get("loss", {}).get("margin", 1.0))
    gap2 = dlt @ w2v if model.s2_rank == 1 else None
    out["readout"] = {
        "w1_norm": float(np.linalg.norm(w1)), "w2_norm": float(np.linalg.norm(w2v)),
        "cos_w1_m4probe": cos(w1, d_m4), "cos_w1_pairprobe": cos(w1, d_pair),
        "cos_w2_pairprobe": cos(w2v, d_pair), "cos_m4probe_pairprobe": cos(d_m4, d_pair),
        "s2gap_median": (float(np.median(gap2)) if gap2 is not None else None),
        "frac_gap_ge_margin": (float((gap2 >= margin).mean()) if gap2 is not None else None)}

    # ---- 梯度冲突快照 ----
    try:
        out["gradconf"] = grad_snapshot(model, cfg, tok, device, ds_tr, ds_pt)
    except Exception as exc:                       # 诊断绝不打断
        out["gradconf"] = {"error": f"{type(exc).__name__}: {exc}"}

    # ---- 保存 + 打印 ----
    if tag != "pretrained":
        with open(ROOT / "runs" / tag / "diag_features.json", "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
    fp = out["frag_probe"]
    gc = out.get("gradconf") or {}

    def _fmt(c):
        return "None" if c is None else format(c, ".4f")

    print(f"[{tag}] pool={out['pool']} m4probe acc={out['m4_probe']['acc_te']:.4f} "
          f"auc={out['m4_probe']['auc_te']:.4f}")
    print(f"    frag acc: " + " ".join(
        f"{k}={_fmt(fp[k]['acc'])}(n={fp[k]['n']})" for k in ("full", "head", "mid", "tail")))
    print(f"    pairprobe auc={out['pair_probe']['auc_ev']:.4f} "
          f"dir={out['pair_probe']['dir_acc_ev']:.4f} | "
          f"cos w1·m4={out['readout']['cos_w1_m4probe']:+.3f} "
          f"w1·pair={out['readout']['cos_w1_pairprobe']:+.3f} "
          f"w2·pair={out['readout']['cos_w2_pairprobe']:+.3f} "
          f"m4·pair={out['readout']['cos_m4probe_pairprobe']:+.3f}")
    print(f"    Δh med={out['delta']['norm_median']:.3f} p90={out['delta']['norm_p90']:.3f} "
          f"cons={out['delta']['consistency']:.3f} | "
          f"s2gap med={out['readout']['s2gap_median']} "
          f"frac≥m={out['readout']['frac_gap_ge_margin']} | "
          f"gradcos={gc.get('cos_enc')} |g1|={gc.get('g1_norm')} |g2|={gc.get('g2_norm')} "
          f"l1={gc.get('l1')} l2={gc.get('l2')}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="特征侧归因诊断（零训练）")
    ap.add_argument("--runs", nargs="+",
                    default=["pretrained", "runs/v0.1.0_ab_nopair", "runs/v0.1.0",
                             "runs/v0.2.2_ab_abmil"])
    ap.add_argument("--m4-n", type=int, default=300)
    ap.add_argument("--pair-n", type=int, default=200)
    ap.add_argument("--frag-n", type=int, default=150)
    ap.add_argument("--pair-eval-cap", type=int, default=None,
                    help="限制 pair 评测条数（冒烟用；默认全量）")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    device = "cpu" if (args.cpu or not torch.cuda.is_available()) else "cuda"
    tags = [r.split("/")[-1].rstrip("/") for r in args.runs]
    print(f"[diag] device={device} runs={tags} m4_n={args.m4_n} pair_n={args.pair_n} "
          f"frag_n={args.frag_n}")
    summary = {}
    for tag in tags:
        try:
            summary[tag] = run_one(tag, args, device)
        except Exception as exc:
            print(f"[{tag}] 失败：{type(exc).__name__}: {exc}")
        finally:
            gc.collect()
    print("\n# 汇总（用于 v0.2 文档归因节的表）")
    for tag, rec in summary.items():
        print(f"  {tag}: m4probe={rec['m4_probe']['acc_te']:.4f} "
              f"mid={rec['frag_probe']['mid']['acc']} "
              f"pairprobe={rec['pair_probe']['auc_ev']:.4f} "
              f"cos_w1_pair={rec['readout']['cos_w1_pairprobe']:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
