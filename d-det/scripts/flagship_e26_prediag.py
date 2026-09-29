#!/usr/bin/env python
"""E26 前置诊断 A/B/C/D（不训练新模型、不生成样本）。

依据 `docx/d-det_E25复核与E26前置诊断_2026-09-29.md`：
  A. E24 s_top 零梯度验证：wq 参数范数；train/val s_top 分布（features.npz 末维）；
     固定训练 batch 上一次 E24 全损失 forward/backward → wq.weight / wq.bias / W1 / W2 / bD
     梯度范数；各损失分量有限性。
  B. μ vs μ+log v 家族探针·训练内正则网格（C ∈ 预注册集合），同一 val 比较，保存全部设置；
     test_seen 不参与（不读取）。
  C. unseen 三来源 split 的准确样本数（meta.json 口径）。
  D. 旗舰/强模型子集：纳入标准（写死）+ 逐 generator × split 名单与样本数。

输出：runs/flagship_e26/e26_diagnostics.json
"""
from __future__ import annotations

import json
import re
import sys
import time
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from encoders import build_encoder  # noqa: E402
from models import build_model  # noqa: E402
from flagship_round1 import (SetPool, _batches, collate, SEEN_LABS, EPS,  # noqa: E402
                             LAM)
from flagship_e25_stage0 import load_corpus_tagged  # noqa: E402

OUT = ROOT / "runs/flagship_e26"
R25 = ROOT / "runs/flagship_e25"
ARGS = Namespace(smoke=False, max_len=2048, train_human=8000, train_fam=1200,
                 val_human=1500, val_fam=500, test_human=500, test_fam=200,
                 unseen_cap=600, enc_bs=32)
C_GRID = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]          # 预注册网格（训练内）
# 旗舰子集纳入标准（写死；主观判定声明）：
#   闭源前沿 API：GPT-4o / GPT-4o-mini；
#   ≥7B 开源 instruct 旗舰：Llama-3.1-405B / Llama-3.3-70B / Llama-3.2-90B-Vision /
#   Llama-4-Scout、Qwen2.5-Coder-32B/7B(-Instruct)、gemma-3-27b/12b-it、phi-4、
#   Yi-Coder-9B-Chat、DeepSeek-V3、Devstral、Mistral-7B-Instruct。
#   排除：≤2B 小型模型、base/非 instruct 变体（如 deepseek-coder-6.7b-base、Qwen-7B base）。
FLAGSHIP_PATTERNS = [
    r"^GPT-4o", r"Llama-3\.1-405B", r"Llama-3\.3-70B", r"Llama-3\.2-90B-Vision",
    r"Llama-4-Scout", r"Qwen2\.5-Coder-32B", r"Qwen2\.5-Coder-7B-Instruct",
    r"gemma-3-27b", r"gemma-3-12b", r"phi-4", r"Yi-Coder-9B", r"DeepSeek-V3",
    r"Devstral", r"Mistral-7B-Instruct",
]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    report = {}

    # ================= A. s_top 零梯度验证 =================
    hp = torch.load(ROOT / "runs/flagship_r1/head.pt", map_location="cpu",
                    weights_only=False)
    head = SetPool(768, 512)
    head.load_state_dict(hp["head"])
    wq_norm = float(head.wq.weight.norm())
    bq = float(head.wq.bias)
    report["A_head_params"] = {"wq_weight_norm": wq_norm, "wq_bias": bq}

    feats = dict(np.load(R25 / "features.npz", allow_pickle=True))
    st_top_tr = feats["z_train"][:, 1536]
    st_top_v = feats["z_val"][:, 1536]
    report["A_stop_distribution"] = {
        "train": {"min": float(st_top_tr.min()), "max": float(st_top_tr.max()),
                  "std": float(st_top_tr.std())},
        "val": {"min": float(st_top_v.min()), "max": float(st_top_v.max()),
                "std": float(st_top_v.std())}}
    print(f"[e26] A: wq norm {wq_norm:.3e} bias {bq:.3e}；s_top std "
          f"train {st_top_tr.std():.3e}", flush=True)

    # 一次 forward/backward（E24 全损失；固定 batch）
    with open(ROOT / "configs/ddet_v041.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    enc_cfg = dict(cfg["encoder"])
    enc_cfg["path"] = str(ROOT / enc_cfg["path"])
    enc0 = build_encoder(**enc_cfg)
    dual = build_model("dual", encoder=enc0, dim=enc0.hidden_size,
                       pool=cfg["model"].get("pool", "mean"),
                       s2_rank=cfg["model"].get("s2_rank", 1))
    ck = torch.load(str(ROOT / "runs/v0.4.1_covreg/last.pt"), map_location="cpu",
                    weights_only=False)
    dual.load_state_dict(ck["state"], strict=False)
    enc = dual.encoder.to(device).eval()
    for p in enc.parameters():
        p.requires_grad = False
    head = head.to(device)
    st = dict(np.load(ROOT / "runs/flagship_r1/stats.npz", allow_pickle=True))
    bD = torch.nn.Parameter(torch.tensor([float(hp["bD"])], device=device))

    corpus = load_corpus_tagged(ARGS)
    batch = next(iter(_batches(corpus["train"], 64, True, seed=0)))
    ids, mask = collate(batch, device)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16,
                                         enabled=device == "cuda"):
        h = enc(ids, mask)
    z = head(h, mask)
    T = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32, device=device)
    mu_c, v_c = T(st["mu"]), T(st["v"])
    pi = T(st["pi"])
    r = z[:, None, :] - mu_c[None]
    G = 0.5 * ((r * r) / v_c[None] + torch.log(v_c)[None]).sum(-1) / z.shape[1]
    g_ai = -torch.logsumexp(torch.log(pi[1:]) - G[:, 1:], dim=1)
    sD = (G[:, 0] - g_ai + torch.log(pi[1:].sum() / pi[0]) + bD)
    y = torch.as_tensor([d.y_ai for d in batch], dtype=torch.float32, device=device)
    L_D = F.binary_cross_entropy_with_logits(sD, y)
    delta, muH, vH = T(st["delta"]), T(st["mu_H"]), T(st["vH"])
    alpha = ((z - muH) / vH) @ delta / ((delta ** 2 / vH).sum() + EPS)
    rF = z - muH - alpha[:, None] * delta
    log_pf = -0.5 * (((rF[:, None, :] - T(st["fam_res_mu"])[None]) ** 2
                      / T(st["fam_res_v"])[None]
                      + torch.log(T(st["fam_res_v"]))[None]).sum(-1)) / z.shape[1] \
        + torch.log(pi[1:])[None]
    log_pf = log_pf - torch.logsumexp(log_pf, dim=1, keepdim=True)
    ai_idx = [i for i, d in enumerate(batch) if d.y_ai == 1]
    fam_t = torch.as_tensor([SEEN_LABS.index(batch[i].fam) for i in ai_idx],
                            device=device)
    L_F = F.nll_loss(log_pf[ai_idx], fam_t)

    def supcon(emb, labels, langs, tau=0.1):
        e = emb / emb.norm(dim=1, keepdim=True).clamp(min=1e-9)
        sim = e @ e.t() / tau
        n = len(labels)
        same = torch.eq(labels[:, None], labels[None, :]) & ~torch.eye(
            n, dtype=torch.bool, device=emb.device)
        diff = same & torch.ne(langs[:, None], langs[None, :])
        use = torch.where(diff.sum(1, keepdim=True) > 0, diff, same)
        cntp = use.sum(1)
        valid = cntp > 0
        sim = sim - sim.max(1, keepdim=True).values.detach()
        logz_ = torch.logsumexp(sim.masked_fill(
            torch.eye(n, dtype=torch.bool, device=emb.device), -1e9), dim=1)
        lp = ((sim - logz_[:, None]) * use).sum(1) / cntp.clamp(min=1)
        return -lp[valid].mean()
    y_lab = torch.as_tensor([d.y_ai for d in batch], device=device)
    lang_l = torch.as_tensor([hash(d.lang) % 64 for d in batch], device=device)
    L_CD = supcon(z, y_lab, lang_l)
    fam_l = torch.as_tensor([SEEN_LABS.index(batch[i].fam) for i in ai_idx],
                            device=device)
    lang_ai = torch.as_tensor([hash(batch[i].lang) % 64 for i in ai_idx],
                              device=device)
    L_CF = supcon(rF[ai_idx], fam_l, lang_ai)
    loss = LAM["D"] * L_D + LAM["F"] * L_F + LAM["CD"] * L_CD + LAM["CF"] * L_CF
    head.zero_grad(); bD.grad = None
    loss.backward()
    grads = {"wq.weight": float(head.wq.weight.grad.norm()) if head.wq.weight.grad is not None else None,
             "wq.bias": float(head.wq.bias.grad.abs().max()) if head.wq.bias.grad is not None else None,
             "W1.weight": float(head.W1.weight.grad.norm()) if head.W1.weight.grad is not None else None,
             "W2.weight": float(head.W2.weight.grad.norm()) if head.W2.weight.grad is not None else None,
             "bD": float(bD.grad.norm()) if bD.grad is not None else None}
    finite = bool(np.isfinite(float(loss)))
    report["A_backward"] = {"loss_total": float(loss), "finite": finite,
                            "components": {"L_D": float(L_D), "L_F": float(L_F),
                                           "L_CD": float(L_CD), "L_CF": float(L_CF)},
                            "grad_norms": grads}
    print(f"[e26] A: loss {float(loss):.4f} finite={finite} | grads {grads}",
          flush=True)

    # ================= B. 正则网格（μ vs μ+log v） =================
    y_tr = np.array([d.y_ai for d in corpus["train"]])
    ai_tr = np.array([d.y_ai == 1 for d in corpus["train"]])
    fam_tr = np.array([SEEN_LABS.index(d.fam) for d in corpus["train"] if d.y_ai])
    ai_val = np.array([d.y_ai == 1 and d.fam in SEEN_LABS for d in corpus["val"]])
    fam_val = np.array([SEEN_LABS.index(d.fam) for d, s in zip(corpus["val"], ai_val)
                        if s])
    z_tr, z_v = feats["z_train"], feats["z_val"]
    sets = {"raw_mean": (feats["rm_train"], feats["rm_val"]),
            "mu": (z_tr[:, :768], z_v[:, :768]),
            "mu+logv": (z_tr[:, :1536], z_v[:, :1536])}
    # 注：features.npz 中 z 的末维（s_top）恒为 0，z ≡ μ+log v，不再重复列入网格。
    grid = {}
    for name, (Xtr, Xv) in sets.items():
        sc = StandardScaler().fit(Xtr[ai_tr])
        Xa = sc.transform(Xtr[ai_tr]); Xb = sc.transform(Xv[ai_val])
        rows = []
        for C in C_GRID:
            lr = LogisticRegression(max_iter=2000, C=C).fit(Xa, fam_tr)
            p = lr.predict(Xb)
            rows.append({"C": C,
                         "val_acc": round(float((p == fam_val).mean()), 4),
                         "val_bal": round(float(balanced_accuracy_score(fam_val, p)), 4)})
        grid[name] = rows
        best = max(rows, key=lambda r: r["val_bal"])
        print(f"[e26] B·{name:<9} best bal {best['val_bal']} @C={best['C']} "
              f"（全网格见 JSON）", flush=True)
    report["B_reg_grid"] = {"grid": C_GRID, "note": "训练折拟合、val 选择；test_seen 未读取",
                            "results": grid}

    # ================= C. unseen 计数（meta.json 口径） =================
    meta = json.loads((R25 / "meta.json").read_text())
    from collections import Counter
    cnt = Counter(meta["unseen"]["split"])
    gens = Counter(meta["unseen"]["generator"])
    report["C_unseen_counts"] = {"by_split": dict(cnt), "total": sum(cnt.values()),
                                 "by_generator": dict(gens)}
    print(f"[e26] C: unseen counts {dict(cnt)}", flush=True)

    # ================= D. 旗舰子集（标准写死） =================
    fam_table = {}
    for split in ("train", "val", "test_seen", "unseen"):
        for g in meta[split]["generator"]:
            fam_table.setdefault(g, {"train": 0, "val": 0, "test_seen": 0, "unseen": 0})
            fam_table[g][split] += 1
    flagged = {g: c for g, c in fam_table.items()
               if any(re.search(p, g) for p in FLAGSHIP_PATTERNS)}
    tot = {s: sum(c[s] for c in flagged.values())
           for s in ("train", "val", "test_seen", "unseen")}
    report["D_flagship"] = {"criteria_patterns": FLAGSHIP_PATTERNS,
                            "per_generator": flagged, "totals": tot,
                            "all_generators": fam_table}
    print(f"[e26] D: 旗舰子集 totals {tot}（{len(flagged)} generators）", flush=True)

    report["timing_min"] = round((time.time() - t0) / 60, 1)
    (OUT / "e26_diagnostics.json").write_text(json.dumps(report, ensure_ascii=False,
                                                         indent=2, default=str))
    print(f"[e26] 完成 {report['timing_min']} min → {OUT/'e26_diagnostics.json'}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
