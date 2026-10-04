#!/usr/bin/env python
"""H2 pair round2（步骤 2）：轻量距离约束臂（学习式投影 + 余弦 BCB）。

- 复用 round2 特征缓存（冻结 CodeT5 均值池化，features/*.npz），不重编码；
- φ(z)=normalize(Wz+b)（768→256，两侧共享）；相似度 s=α·cos(φL,φR)+β（α,β 可学习）；
- loss=BCE(σ(γ·s), y)（γ=4 固定）；train 拟合、dev 选 epoch+阈值、test 一次、3 seeds；
- 目的：判断"轻量对比/距离约束"是否比线性头更接近/超越 lexical shortcut。
输出：artifacts/h2_pair_round2/metrics_contrastive.json（不覆盖 metrics.json）
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
DATA = ROOT / "data" / "h2_pair_benchmark_v1"
OUT = ROOT / "artifacts" / "h2_pair_round2"
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from h2_pair_similarity_baseline import choose_threshold, cluster_bootstrap, metrics  # noqa: E402

VIEWS = ("raw", "ids_only", "strings_only", "comments_only", "all")


def read_jsonl(p: Path):
    with p.open(encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def load_feats(source: str, view: str):
    z = np.load(OUT / "features" / f"{source}_{view}.npz", allow_pickle=True)
    return dict(zip(z["shas"].tolist(), z["feats"]))


def run_view(srows, view, table, seeds=(0, 1, 2), device="cuda", epochs=60, patience=10):
    zl = np.stack([table[r["left_view_code_sha256"][view]] for r in srows]).astype(np.float32)
    zr = np.stack([table[r["right_view_code_sha256"][view]] for r in srows]).astype(np.float32)
    y = np.array([r["pair_label"] for r in srows], dtype=int)
    splits = np.array([r["task_split"] for r in srows])
    tr, dv, te = splits == "train", splits == "dev", splits == "test"
    per_seed = []
    for seed in seeds:
        torch.manual_seed(seed); np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        proj = torch.nn.Linear(768, 256).to(device)
        alpha = torch.nn.Parameter(torch.tensor(8.0, device=device))
        beta = torch.nn.Parameter(torch.tensor(0.0, device=device))
        opt = torch.optim.AdamW([{"params": proj.parameters(), "lr": 1e-3},
                                 {"params": [alpha, beta], "lr": 1e-2}], weight_decay=1e-4)
        L = torch.tensor(zl[tr], device=device); R = torch.tensor(zr[tr], device=device)
        Ld = torch.tensor(zl[dv], device=device); Rd = torch.tensor(zr[dv], device=device)
        Lt = torch.tensor(zl[te], device=device); Rt = torch.tensor(zr[te], device=device)
        ytr = torch.tensor(y[tr], device=device)
        best = {"ba": -1, "state": None, "epoch": 0, "threshold": 0.5}
        curve = []
        n = L.shape[0]
        for ep in range(1, epochs + 1):
            proj.train()
            rng = np.random.RandomState(seed * 100 + ep)
            order = torch.tensor(rng.permutation(n), device=device)
            for i in range(0, n, 256):
                idx = order[i:i + 256]
                opt.zero_grad()
                l = F.normalize(proj(L[idx]), dim=1); r = F.normalize(proj(R[idx]), dim=1)
                s = alpha * (l * r).sum(1) + beta
                loss = F.binary_cross_entropy_with_logits(s, ytr[idx].float())
                loss.backward(); opt.step()
            proj.eval()
            with torch.inference_mode():
                l = F.normalize(proj(Ld), dim=1); r = F.normalize(proj(Rd), dim=1)
                sc = (alpha * (l * r).sum(1) + beta).cpu().numpy()
            th = choose_threshold(sc, y[dv])
            ba = float((np.mean(sc[y[dv] == 1] >= th) + np.mean(sc[y[dv] == 0] < th)) / 2)
            curve.append({"epoch": ep, "dev_ba": ba, "threshold": float(th)})
            if ba > best["ba"]:
                best = {"ba": ba, "epoch": ep, "threshold": float(th),
                        "state": {k: v.detach().clone() for k, v in proj.state_dict().items()},
                        "alpha": float(alpha.detach()), "beta": float(beta.detach())}
            if ep - best["epoch"] >= patience:
                break
        proj.load_state_dict(best["state"]); proj.eval()
        with torch.inference_mode():
            l = F.normalize(proj(Lt), dim=1); r = F.normalize(proj(Rt), dim=1)
            st = (best["alpha"] * (l * r).sum(1) + best["beta"]).cpu().numpy()
        per_seed.append({"seed": seed, "best_epoch": best["epoch"], "best_dev_ba": best["ba"],
                         "alpha": best["alpha"], "beta": best["beta"], "dev_curve": curve,
                         "test": metrics(st, y[te], best["threshold"]),
                         "test_task_cluster_bootstrap": cluster_bootstrap(
                             [r for r in srows if r["task_split"] == "test"], st, y[te], best["threshold"])})
        del proj; torch.cuda.empty_cache()
    return per_seed


def main():
    t0 = time.time()
    rows = read_jsonl(DATA / "pairs.jsonl")
    out = {"config": {"script": "scripts/h2_pair_contrastive_probe.py",
                      "features": "round2 features 缓存（冻结 CodeT5 mean-pool 768d）",
                      "model": "phi=normalize(Wz+b)(768→256)；s=alpha*cos(phiL,phiR)+beta；BCE；alpha/beta 可学习",
                      "protocol": "train 拟合、dev 选 epoch+阈值、test 一次、3 seeds、task 聚类 bootstrap"},
           "sources": {}}
    for source in ("authorbench_dcan", "llm_codegen_v2"):
        srows = [r for r in rows if r["source"] == source]
        out["sources"][source] = {"n": len(srows), "views": {}}
        for view in VIEWS:
            table = load_feats(source, view)
            per_seed = run_view(srows, view, table)
            mean_ba = float(np.mean([p["test"]["balanced_accuracy"] for p in per_seed]))
            mean_auc = float(np.mean([p["test"]["roc_auc"] for p in per_seed if p["test"]["roc_auc"] is not None]))
            out["sources"][source]["views"][view] = {
                "per_seed": per_seed, "mean_test_ba": mean_ba, "mean_test_auc": mean_auc}
            print(f"[r2con] {source}/{view}: mean testBA {mean_ba:.4f} (AUC {mean_auc:.4f})", flush=True)
    (OUT / "metrics_contrastive.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"[r2con] done {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
