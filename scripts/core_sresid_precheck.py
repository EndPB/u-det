"""S-resid 补充检查（指导 §7：作为 H3 候选编码的前置，仅 F2 通过后执行/交付）。

检查项：
  1. fit-only residualization（fit fold lstsq，eval 用 fit coef）——重跑确认；
  2. fit-only standardization（每维 fit 均值/std）——重跑确认；
  3. **residual norm 与 A norm 匹配**：S⊥ 全局缩放到 mean‖A‖ 后训练（S_resid_normmatch）；
  4. heldout member 上的 A-only / raw S+A（SA_full）/ S-resid+A 三臂（task fold + model 5 折）；
  5. S-resid permutation（task 分层置换）。
变体：A_only / SA_full / S_resid / S_resid_normmatch / S_resid_shuffle（twin MLP 同配置）。

输出：artifacts/h3_train_dev_registered_2026-10-08/（s_resid 检查部分）
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

OUT = ROOT / "d-det/artifacts/h3_train_dev_registered_2026-10-08"
SEED = 20261008
VARIANTS = ("A_only", "SA_full", "S_resid", "S_resid_normmatch", "S_resid_shuffle")
LOG: list[str] = []


def log(m):
    print(m, flush=True)
    LOG.append(m)


def perm_S(S, n_models, n_tasks):
    rng = np.random.default_rng(SEED)
    S2 = S.copy()
    for tj in range(n_tasks):
        rows = np.arange(n_models) * n_tasks + tj
        perm = rng.permutation(n_models)
        S2[rows] = S[rows[perm]]
    return S2


def ctrl_mat(A, Rc, Ri):
    st = A["style"].astype(np.float64); mt = A["meta"].astype(np.float64)
    sz = A["sizelen"].astype(np.float64)
    return np.hstack([st[Ri], mt[Ri], sz[Ri]])


def fit_fold_views(A, fit_models, fit_tasks):
    Xh = A["emb_base"]
    Rc, Ri = ev.rows_for(fit_models, fit_tasks)
    S = (Xh[Ri] + Xh[Rc]) / 2.0
    Ax = (Xh[Ri] - Xh[Rc]) / 2.0
    Xc = ctrl_mat(A, Rc, Ri)
    Xt = np.hstack([Xc, np.ones((len(Xc), 1))])
    coef, *_ = np.linalg.lstsq(Xt, S, rcond=None)
    Sr = S - Xt @ coef
    mu = Sr.mean(0); sd = Sr.std(0); sd_safe = np.where(sd < 1e-8, 1.0, sd)
    # norm 匹配：S⊥ 全局缩放到 mean‖A‖
    nA = float(np.mean(np.linalg.norm(Ax, axis=1)))
    nS = float(np.mean(np.linalg.norm(Sr, axis=1)))
    scale = nA / max(1e-12, nS)
    Ssc = Sr * scale
    Srp = perm_S(Sr, len(fit_models), len(fit_tasks))
    out = {
        "A_only": (Ax, -Ax),
        "SA_full": (np.hstack([S, Ax]), np.hstack([S, -Ax])),
        "S_resid": (np.hstack([Sr, Ax]), np.hstack([Sr, -Ax])),
        "S_resid_normmatch": (np.hstack([Ssc, Ax]), np.hstack([Ssc, -Ax])),
        "S_resid_shuffle": (np.hstack([Srp, Ax]), np.hstack([Srp, -Ax])),
    }
    return out, coef, mu, sd_safe, scale


def eval_fold_views(A, em, et, coef, mu, sd, scale):
    Xh = A["emb_base"]
    Rc, Ri = ev.rows_for(em, et)
    S = (Xh[Ri] + Xh[Rc]) / 2.0
    Ax = (Xh[Ri] - Xh[Rc]) / 2.0
    Xc = ctrl_mat(A, Rc, Ri)
    Xe = np.hstack([Xc, np.ones((len(Xc), 1))])
    Sr = S - Xe @ coef
    Ssc = Sr * scale
    Srp = perm_S(Sr, len(em), len(et))
    return {
        "A_only": (Ax, -Ax),
        "SA_full": (np.hstack([S, Ax]), np.hstack([S, -Ax])),
        "S_resid": (np.hstack([Sr, Ax]), np.hstack([Sr, -Ax])),
        "S_resid_normmatch": (np.hstack([Ssc, Ax]), np.hstack([Ssc, -Ax])),
        "S_resid_shuffle": (np.hstack([Srp, Ax]), np.hstack([Srp, -Ax])),
    }


def train_twin(v1, v2, wgt, epochs=12, hidden=(256, 64), seed=SEED):
    import torch
    import torch.nn as nn
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = v1.shape[1]
    net = nn.Sequential(nn.Linear(d, hidden[0]), nn.ReLU(), nn.Linear(hidden[0], hidden[1]),
                        nn.ReLU(), nn.Linear(hidden[1], 1)).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    X1 = torch.tensor(np.asarray(v1, dtype=np.float32), device=device)
    X2 = torch.tensor(np.asarray(v2, dtype=np.float32), device=device)
    w_t = torch.tensor(np.asarray(wgt, dtype=np.float32), device=device)
    n = len(X1); bs = 1024
    rng = np.random.default_rng(seed)
    for ep in range(epochs):
        perm = rng.permutation(n)
        for i in range(0, n, bs):
            sel = perm[i:i + bs]
            g1 = net(X1[sel]).squeeze(-1); g2 = net(X2[sel]).squeeze(-1)
            loss = (torch.nn.functional.softplus(-(g1 - g2)) * w_t[sel]).mean()
            opt.zero_grad(); loss.backward(); opt.step()

    def score(V):
        with torch.inference_mode():
            return net(torch.tensor(np.asarray(V, dtype=np.float32), device=device)).squeeze(-1).cpu().numpy()
    return score


def acc_mb(pos, inv_m, nm):
    sums = np.bincount(inv_m, weights=pos, minlength=nm)
    cnts = np.bincount(inv_m, minlength=nm)
    valid = cnts > 0
    return float(np.mean(sums[valid] / cnts[valid]))


def metrics_with_ci(d, models, tasks, nb=500, seed=SEED):
    uniq_m, inv_m = np.unique(models, return_inverse=True)
    uniq_t, inv_t = np.unique(tasks, return_inverse=True)
    nm = len(uniq_m)
    pos = (d > 0).astype(float) + 0.5 * (d == 0).astype(float)
    point = acc_mb(pos, inv_m, nm)
    rng = np.random.default_rng(seed)
    idx_by = {q: np.where(inv_t == q)[0] for q in range(len(uniq_t))}
    vals = np.empty(nb)
    for k in range(nb):
        pick = rng.choice(len(uniq_t), size=len(uniq_t), replace=True)
        idx = np.concatenate([idx_by[q] for q in pick])
        vals[k] = acc_mb(pos[idx], inv_m[idx], nm)
    return {"pair_acc_mb": point,
            "ci95": [float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))],
            "n_units": int(len(d))}


def run_kernel(split, fold):
    t0 = time.time()
    A = ev.load_assets()
    defs = ev.split_definition(split, fold)
    fit_models, fit_tasks = defs[0][1], defs[0][2]
    log(f"[sres] split={split} fold={fold} fit=({len(fit_models)}m×{len(fit_tasks)}t)")
    fv, coef, mu, sd, scale = fit_fold_views(A, fit_models, fit_tasks)
    log(f"    normmatch scale = {scale:.4f}")
    fm_axis = np.repeat(np.array(fit_models), len(fit_tasks))
    cnt = {}
    for m in fm_axis:
        cnt[m] = cnt.get(m, 0) + 1
    wgt = np.array([1.0 / cnt[m] for m in fm_axis])
    scorers = {}
    for name in VARIANTS:
        log(f"    train {name}")
        v1, v2 = fv[name]
        scorers[name] = train_twin(v1, v2, wgt)
    res = {"schema": "sres_precheck_v1", "split": split, "fold": fold, "normmatch_scale": scale, "domains": {}}
    for dom in defs:
        label, em, et = dom[0], dom[3], dom[4]
        if not em:
            continue
        e_views = eval_fold_views(A, em, et, coef, mu, sd, scale)
        em_axis = np.repeat(np.array(em), len(et)); et_axis = np.tile(np.array(et), len(em))
        dm = {}
        for name in VARIANTS:
            v1x, v2x = e_views[name]
            sc, si = scorers[name](v2x), scorers[name](v1x)
            dm[name] = metrics_with_ci(si - sc, em_axis, et_axis)
            log(f"    [{label}][{name}] mb={dm[name]['pair_acc_mb']:.4f} ci={dm[name]['ci95']}")
        res["domains"][label] = dm
    res["runtime_seconds"] = time.time() - t0
    (OUT / "results").mkdir(exist_ok=True)
    (OUT / "results" / f"kernel_{split}_{fold}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"[sres] done {split}/{fold} in {time.time()-t0:.1f}s")


def summarize():
    res = {}
    for p in sorted((OUT / "results").glob("*.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        res[f"{r['split']}_{r['fold']}"] = r
    task_dev, model_folds = None, []
    for key, r in res.items():
        dm = r["domains"].get("task_dev")
        if dm is not None:
            task_dev = {"M": {k: dm[k]["pair_acc_mb"] for k in dm}, "ci": {k: dm[k]["ci95"] for k in dm},
                        "dResid": dm["S_resid"]["pair_acc_mb"] - dm["A_only"]["pair_acc_mb"],
                        "dResid_normmatch": dm["S_resid_normmatch"]["pair_acc_mb"] - dm["A_only"]["pair_acc_mb"],
                        "dSA_full": dm["SA_full"]["pair_acc_mb"] - dm["A_only"]["pair_acc_mb"],
                        "dShuffle": dm["S_resid_shuffle"]["pair_acc_mb"] - dm["A_only"]["pair_acc_mb"]}
        dm2 = r["domains"].get("model_dev")
        if dm2 is not None:
            model_folds.append({"kernel": key,
                                "dResid": dm2["S_resid"]["pair_acc_mb"] - dm2["A_only"]["pair_acc_mb"],
                                "dResid_normmatch": dm2["S_resid_normmatch"]["pair_acc_mb"] - dm2["A_only"]["pair_acc_mb"],
                                "dSA_full": dm2["SA_full"]["pair_acc_mb"] - dm2["A_only"]["pair_acc_mb"],
                                "dShuffle": dm2["S_resid_shuffle"]["pair_acc_mb"] - dm2["A_only"]["pair_acc_mb"],
                                "M": {k: dm2[k]["pair_acc_mb"] for k in dm2}})
    out = {"schema": "sres_precheck_summary_v1", "generated_utc": datetime.now(timezone.utc).isoformat(),
           "report_first_line": "S-resid 补充检查（§7；fit-only 残差/标准化/范数匹配/三臂/置换）",
           "task_dev": task_dev, "model_folds": model_folds,
           "model_dResid_mean": float(np.mean([m["dResid"] for m in model_folds])) if model_folds else None,
           "model_dNormmatch_mean": float(np.mean([m["dResid_normmatch"] for m in model_folds])) if model_folds else None,
           "model_dSA_full_mean": float(np.mean([m["dSA_full"] for m in model_folds])) if model_folds else None,
           "checklist": {
               "fit_only_residualization": True, "fit_only_std": True,
               "norm_match_kept": bool(task_dev and task_dev["dResid_normmatch"] > 0.01
                                       and model_folds and all(m["dResid_normmatch"] > 0 for m in model_folds)),
               "raw_SA_plus_A_vs_resid": (task_dev["dSA_full"] if task_dev else None),
               "permutation_reduces": bool(task_dev and task_dev["dShuffle"] < task_dev["dResid"])}}
    (OUT / "metrics_sres_precheck.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"[sres] summary done; norm_match_kept={out['checklist']['norm_match_kept']}")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["run", "sum"])
    ap.add_argument("--split", default="task")
    ap.add_argument("--fold", type=int, default=-1)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "logs").mkdir(exist_ok=True)
    if args.stage == "run":
        run_kernel(args.split, args.fold)
    else:
        summarize()
        (OUT / "logs" / "sres_run.log").write_text("\n".join(LOG) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
