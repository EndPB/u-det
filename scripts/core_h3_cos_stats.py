"""H3 c3 梯度余弦激活统计（§1.1-3）：逐批记录 cos、罚项、梯度范数。

对 11 M-HO + 3 S-HO 折重跑 c3 单配置，逐 batch 记录：
  cos(∇L_D, ∇L_F)（encoder 参数）、罚项值 relu(cos−0.3)、梯度范数 ‖g_D‖/‖g_F‖。
汇总：激活率（cos>m 的比例）、罚项均值、负余弦率、梯度范数分布。
说明 c2/c3 输出相同不足以证明梯度分离有效（本统计给出实际激活证据）。

输出：artifacts/h3_definition_and_human_support_2026-10-08/metrics_h3_cos_stats.json
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import core_r0_eval as ev  # noqa: E402
import variant_transfer_stage1_execute as s1  # noqa: E402
import core_h3_registered as H3R  # noqa: E402

OUT = ROOT / "d-det/artifacts/h3_definition_and_human_support_2026-10-08"
SEED = 20261008
M_COS = 0.3
LOG: list[str] = []


def log(m):
    print(m, flush=True)
    LOG.append(m)


def main():
    import torch
    import torch.nn as nn
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "logs").mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    A = ev.load_assets()
    splits = {t: A["splits"][A["TASK_I"][t] * 2] for t in A["tasks"]}
    tr_tasks = [t for t in A["tasks"] if splits[t] == "train"]
    dv_tasks = [t for t in A["tasks"] if splits[t] == "dev"]
    U = H3R.build_u(A)
    log(f"[c3] tasks train={len(tr_tasks)} dev={len(dv_tasks)}")

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = nn.Sequential(nn.Linear(1536, 256), nn.ReLU())
            self.head_d = nn.Linear(256, 1)
            self.head_f = nn.Sequential(nn.Linear(256 * 4, 64), nn.ReLU(), nn.Linear(64, 1))
        def z(self, x):
            return self.enc(x)
        def d(self, z):
            return self.head_d(z).squeeze(-1)
        def f(self, za, zb):
            return self.head_f(torch.cat([za, zb, (za - zb).abs(), za * zb], -1)).squeeze(-1)

    def unit_mat(models, tasks):
        return np.stack([U[(m, t)] for m in models for t in tasks]).astype(np.float32)

    def run_c3(data, epochs=2, lr=2e-3, bs=2048, seed=SEED):
        torch.manual_seed(seed)
        net = Net().to(device)
        opt = torch.optim.Adam(net.parameters(), lr=lr)
        rng = np.random.default_rng(seed)
        Xdet = torch.tensor(unit_mat(data["det_models"], tr_tasks), device=device)
        n_cells = Xdet.shape[0]
        pairs_tr = data["fit_pos"] + data["fit_neg"]
        y_tr = np.array([1] * len(data["fit_pos"]) + [0] * len(data["fit_neg"]), dtype=np.float32)
        UA_t = torch.tensor(np.stack([U[(a, t)] for a, b, t in pairs_tr]).astype(np.float32), device=device)
        UB_t = torch.tensor(np.stack([U[(b, t)] for a, b, t in pairs_tr]).astype(np.float32), device=device)
        y_tt = torch.tensor(y_tr, device=device)
        rec = []
        for ep in range(epochs):
            perm = rng.permutation(n_cells)
            for i in range(0, n_cells, bs):
                sel = torch.tensor(perm[i:i + bs], device=device)
                Xb = Xdet[sel]
                Xp = torch.cat([Xb[:, :768], -Xb[:, 768:]], dim=1)
                loss_d = torch.nn.functional.softplus(-(net.d(net.z(Xb)) - net.d(net.z(Xp)))).mean()
                idx2 = torch.tensor(rng.integers(0, len(pairs_tr), size=min(bs, len(pairs_tr))), device=device)
                za = UA_t[idx2]; zb = UB_t[idx2]; yb = y_tt[idx2]
                loss_f = 0.5 * (torch.nn.functional.binary_cross_entropy_with_logits(net.f(net.z(za), net.z(zb)), yb)
                                + torch.nn.functional.binary_cross_entropy_with_logits(net.f(net.z(zb), net.z(za)), yb))
                gd = torch.autograd.grad(loss_d, net.enc.parameters(), retain_graph=True, allow_unused=True, create_graph=True)
                gf = torch.autograd.grad(loss_f, net.enc.parameters(), retain_graph=True, allow_unused=True, create_graph=True)
                gdv = torch.cat([g.ravel() for g in gd if g is not None])
                gfv = torch.cat([g.ravel() for g in gf if g is not None])
                cos_t = torch.nn.functional.cosine_similarity(gdv, gfv, dim=0)
                pen = float(torch.relu(cos_t - M_COS).detach())
                rec.append({"cos": float(cos_t.detach()), "penalty": pen,
                            "gd_norm": float(gdv.norm()), "gf_norm": float(gfv.norm())})
                loss = loss_d + loss_f + 0.1 * torch.relu(cos_t - M_COS)
                opt.zero_grad(); loss.backward(); opt.step()
        return rec

    records = {}
    for series in s1.SERIES:
        for h in s1.SERIES[series]:
            data = H3R.fold_data_mho(A, U, series, h, tr_tasks, dv_tasks)
            rec = run_c3(data)
            records[f"MHO::{series}::{h}"] = rec
            cosv = np.array([r["cos"] for r in rec])
            log(f"[c3] MHO {series}::{h}: n_batch={len(rec)} activation_rate={(cosv > M_COS).mean():.3f} "
                f"cos_mean={cosv.mean():.3f} neg_rate={(cosv < 0).mean():.3f}")
    for hold in s1.SERIES:
        data = H3R.fold_data_sho(A, U, hold, tr_tasks, dv_tasks)
        rec = run_c3(data)
        records[f"SHO::{hold}"] = rec
        cosv = np.array([r["cos"] for r in rec])
        log(f"[c3] SHO {hold}: n_batch={len(rec)} activation_rate={(cosv > M_COS).mean():.3f} cos_mean={cosv.mean():.3f}")

    all_cos = np.concatenate([np.array([r["cos"] for r in v]) for v in records.values()])
    all_pen = np.concatenate([np.array([r["penalty"] for r in v]) for v in records.values()])
    all_gd = np.concatenate([np.array([r["gd_norm"] for r in v]) for v in records.values()])
    all_gf = np.concatenate([np.array([r["gf_norm"] for r in v]) for v in records.values()])
    summary = {"schema": "h3_cos_stats_v1", "generated_utc": datetime.now(timezone.utc).isoformat(),
               "m_cos_threshold": M_COS, "n_folds": len(records), "n_batches_total": int(len(all_cos)),
               "activation_rate": float((all_cos > M_COS).mean()),
               "penalty_mean": float(all_pen.mean()),
               "negative_cos_rate": float((all_cos < 0).mean()),
               "cos": {"mean": float(all_cos.mean()), "std": float(all_cos.std()),
                       "p05": float(np.percentile(all_cos, 5)), "p50": float(np.percentile(all_cos, 50)),
                       "p95": float(np.percentile(all_cos, 95))},
               "grad_norms": {"gd_mean": float(all_gd.mean()), "gf_mean": float(all_gf.mean()),
                              "gd_median": float(np.median(all_gd)), "gf_median": float(np.median(all_gf))},
               "note": ("c2/c3 输出相同不能证明梯度分离有效；本统计给出实际激活证据（cos>m 比例、"
                        "罚项均值、梯度范数）。正对齐惩罚的目的与效果需在正式协议前重新论证。"),
               "records": {k: {"n": len(v), "activation_rate": float((np.array([r['cos'] for r in v]) > M_COS).mean())}
                           for k, v in records.items()}}
    (OUT / "metrics_h3_cos_stats.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "logs" / "c3_stats.log").write_text("\n".join(LOG) + "\n", encoding="utf-8")
    log(f"[c3] done in {time.time()-t0:.1f}s; activation_rate={summary['activation_rate']:.4f}")


if __name__ == "__main__":
    main()
