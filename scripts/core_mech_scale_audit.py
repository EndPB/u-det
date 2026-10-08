"""机制判词统一 + S-resid 尺度控制审计（指导 §4）。

A) canonical mechanism_verdict.json：raw_SA_mechanism=unresolved（依 §4.1）；
   所有后续报告只引用该文件（旧目录 report.md/mechanism_claim.json 不改，仅记录差异）。
B) S-resid 尺度控制（每 fit fold）：
   S⊥ = S − Ŝ(C)，C=[style,meta,size,length]（fit fold lstsq）。
   变体（twin MLP 完全同配置 hidden=(256,64)/12ep/seed 20261008 = matched-capacity）：
     A_only / S_resid（原尺度）/ S_resid_std（fit 每维标准化）/ S_resid_shuffle（task 分层置换）
   审计量：每维均值(std)、协方差 top 特征值/能量占比、‖S‖/‖S⊥‖/‖A‖ 范数分布。
   判词：task 与 model-heldout 均强 + std 变体稳健 → residualized_representation_candidate；
         否则 exploratory_weak / scale_suspect。不得写后训练因果。

输出：artifacts/r0_mechanism_scale_audit_2026-10-08/
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

OUT = ROOT / "d-det/artifacts/r0_mechanism_scale_audit_2026-10-08"
OLD_MECH = ROOT / "d-det/artifacts/r0_mechanism_controls_2026-10-08"
SEED = 20261008
VARIANTS = ("A_only", "S_resid", "S_resid_std", "S_resid_shuffle")
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


def fit_transform_stats(Xc, S):
    Xt = np.hstack([Xc, np.ones((len(Xc), 1))])
    coef, *_ = np.linalg.lstsq(Xt, S, rcond=None)
    S_res = S - Xt @ coef
    mu = S_res.mean(0)
    sd = S_res.std(0)
    sd_safe = np.where(sd < 1e-8, 1.0, sd)
    return coef, mu, sd_safe


def fit_views(A, fit_models, fit_tasks):
    Xh = A["emb_base"]
    Rc, Ri = ev.rows_for(fit_models, fit_tasks)
    S = (Xh[Ri] + Xh[Rc]) / 2.0
    Ax = (Xh[Ri] - Xh[Rc]) / 2.0
    Xc = ctrl_mat(A, Rc, Ri)
    coef, mu, sd = fit_transform_stats(Xc, S)
    Xt = np.hstack([Xc, np.ones((len(Xc), 1))])
    Sr = S - Xt @ coef
    Srs = (Sr - mu) / sd
    Srp = perm_S(Sr, len(fit_models), len(fit_tasks))
    out = {
        "A_only": (Ax, -Ax),
        "S_resid": (np.hstack([Sr, Ax]), np.hstack([Sr, -Ax])),
        "S_resid_std": (np.hstack([Srs, Ax]), np.hstack([Srs, -Ax])),
        "S_resid_shuffle": (np.hstack([Srp, Ax]), np.hstack([Srp, -Ax])),
    }
    audit = {
        "S_resid_per_dim": {"mean_abs_max": float(np.abs(mu - Sr.mean(0)).max()),
                            "std_min": float(sd.min()), "std_median": float(np.median(sd)),
                            "std_max": float(sd.max())},
        "norms": {k: {"mean": float(np.mean(np.linalg.norm(v, axis=1))),
                      "std": float(np.std(np.linalg.norm(v, axis=1))),
                      "p05": float(np.percentile(np.linalg.norm(v, axis=1), 5)),
                      "p95": float(np.percentile(np.linalg.norm(v, axis=1), 95))}
                  for k, v in (("S", S), ("S_resid", Sr), ("A", Ax))},
    }
    # 协方差 top 特征值（能量占比）
    for tag, M in (("S", S), ("S_resid", Sr)):
        C = (M - M.mean(0)).T @ (M - M.mean(0)) / len(M)
        evals = np.linalg.eigvalsh(C)[::-1]
        top10 = evals[:10]
        audit[f"cov_{tag}"] = {"top10_eigvals": [float(x) for x in top10],
                               "top10_energy_ratio": float(top10.sum() / max(1e-12, evals.sum())),
                               "eigval_median": float(np.median(evals))}
    return out, coef, mu, sd, audit


def eval_views(A, em, et, coef, mu, sd):
    Xh = A["emb_base"]
    Rc, Ri = ev.rows_for(em, et)
    S = (Xh[Ri] + Xh[Rc]) / 2.0
    Ax = (Xh[Ri] - Xh[Rc]) / 2.0
    Xc = ctrl_mat(A, Rc, Ri)
    Xe = np.hstack([Xc, np.ones((len(Xc), 1))])
    Sr = S - Xe @ coef
    Srs = (Sr - mu) / sd
    Srp = perm_S(Sr, len(em), len(et))
    return {
        "A_only": (Ax, -Ax),
        "S_resid": (np.hstack([Sr, Ax]), np.hstack([Sr, -Ax])),
        "S_resid_std": (np.hstack([Srs, Ax]), np.hstack([Srs, -Ax])),
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
            "n_units": int(len(d)), "n_models": int(nm), "n_tasks": int(len(uniq_t))}


def run_kernel(split, fold):
    t0 = time.time()
    A = ev.load_assets()
    defs = ev.split_definition(split, fold)
    fit_models, fit_tasks = defs[0][1], defs[0][2]
    log(f"[audit] split={split} fold={fold} fit=({len(fit_models)}m×{len(fit_tasks)}t)")
    fv, coef, mu, sd, audit = fit_views(A, fit_models, fit_tasks)
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
    res = {"schema": "mech_scale_kernel_v1", "split": split, "fold": fold, "audit": audit, "domains": {}}
    for dom in defs:
        label, em, et = dom[0], dom[3], dom[4]
        if not em:
            continue
        ev_v = eval_views(A, em, et, coef, mu, sd)
        em_axis = np.repeat(np.array(em), len(et)); et_axis = np.tile(np.array(et), len(em))
        dm = {}
        for name in VARIANTS:
            v1x, v2x = ev_v[name]
            sc, si = scorers[name](v2x), scorers[name](v1x)
            d = si - sc
            dm[name] = metrics_with_ci(d, em_axis, et_axis)
            log(f"    [{label}][{name}] mb={dm[name]['pair_acc_mb']:.4f} ci={dm[name]['ci95']}")
        res["domains"][label] = dm
    res["runtime_seconds"] = time.time() - t0
    (OUT / "results").mkdir(exist_ok=True)
    (OUT / "results" / f"{split}_{fold}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"[audit] done {split}/{fold} in {time.time() - t0:.1f}s")


def summarize():
    """读全部结果 → metrics_scale_audit.json + canonical mechanism_verdict.json。"""
    res = {}
    for p in sorted((OUT / "results").glob("*.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        res[f"{r['split']}_{r['fold']}"] = r
    # 旧 mech deltas（raw SA 依据）
    raw_basis = {}
    oldc = OLD_MECH / "mechanism_claim.json"
    if oldc.exists():
        oc = json.loads(oldc.read_text(encoding="utf-8"))
        raw_basis = {"old_claim_file": "r0_mechanism_controls_2026-10-08/mechanism_claim.json",
                     "old_verdict_string": oc.get("verdict"),
                     "old_deltas": {k: {kk: (float(np.mean(vv)) if isinstance(vv, list) else float(vv))
                                        for kk, vv in v.items() if kk.startswith("d")}
                                    for k, v in oc.get("deltas", {}).items()}}
    # 汇总新结果
    task_dev = None; model_folds = []
    for key, r in res.items():
        dm = r["domains"].get("task_dev")
        if dm is not None:
            task_dev = {"kernel": key, "dResid": dm["S_resid"]["pair_acc_mb"] - dm["A_only"]["pair_acc_mb"],
                        "dResidStd": dm["S_resid_std"]["pair_acc_mb"] - dm["A_only"]["pair_acc_mb"],
                        "dShuffle": dm["S_resid_shuffle"]["pair_acc_mb"] - dm["A_only"]["pair_acc_mb"],
                        "M": {k: dm[k]["pair_acc_mb"] for k in dm},
                        "ci": {k: dm[k]["ci95"] for k in dm}}
        dm2 = r["domains"].get("model_dev")
        if dm2 is not None:
            model_folds.append({"kernel": key,
                                "dResid": dm2["S_resid"]["pair_acc_mb"] - dm2["A_only"]["pair_acc_mb"],
                                "dResidStd": dm2["S_resid_std"]["pair_acc_mb"] - dm2["A_only"]["pair_acc_mb"],
                                "dShuffle": dm2["S_resid_shuffle"]["pair_acc_mb"] - dm2["A_only"]["pair_acc_mb"],
                                "M": {k: dm2[k]["pair_acc_mb"] for k in dm2}})
    d_res = [m["dResid"] for m in model_folds]
    d_std = [m["dResidStd"] for m in model_folds]
    metrics = {"schema": "mech_scale_audit_v1", "generated_utc": datetime.now(timezone.utc).isoformat(),
               "task_dev": task_dev, "model_folds": model_folds,
               "model_dResid_mean": float(np.mean(d_res)) if d_res else None,
               "model_dResidStd_mean": float(np.mean(d_std)) if d_std else None,
               "model_dResid_all_positive": bool(all(x > 0 for x in d_res)) if d_res else None,
               "audits": {k: r["audit"] for k, r in res.items()}}
    (OUT / "metrics_scale_audit.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=1), encoding="utf-8")

    # canonical verdict
    strong_task = task_dev is not None and task_dev["dResid"] > 0.01
    strong_model = bool(d_res) and all(x > 0 for x in d_res) and float(np.mean(d_res)) > 0.01
    std_ok = task_dev is not None and task_dev["dResidStd"] > 0.01 and float(np.mean(d_std)) > 0.0
    shuffle_down = task_dev is not None and task_dev["dShuffle"] < task_dev["dResid"]
    if strong_task and strong_model and std_ok:
        s_status = "residualized_representation_candidate"
    elif strong_task and strong_model and not std_ok:
        s_status = "exploratory_weak_scale_suspect"
    else:
        s_status = "exploratory_weak"
    verdict = {
        "schema": "mechanism_verdict_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "canonical": True,
        "note": "本文件为本轮机制判词唯一 canonical 来源；旧目录的报告仅保留历史记录。",
        "raw_SA_mechanism": "unresolved",
        "raw_SA_basis": raw_basis,
        "raw_SA_summary": ("task-dev dSA≈+0.47pt、model 折 +0.16~0.27pt 且 CI 重叠；task-dev shuffle 未形成强分离 —— "
                           "按指导 §4.1 判定 raw_SA_mechanism=unresolved；S_shuffle 仅辅助诊断。"),
        "s_resid_status": s_status,
        "s_resid_evidence": {"task_dev": task_dev, "model_folds_mean": {"dResid": float(np.mean(d_res)) if d_res else None,
                                                                       "dResidStd": float(np.mean(d_std)) if d_std else None},
                             "strong_task_gt_1pt": bool(strong_task), "strong_model_all_positive": bool(strong_model),
                             "std_variant_robust": bool(std_ok), "shuffle_reduces_vs_resid": bool(shuffle_down)},
        "caveats": ["S⊥ 原尺度在 MLP 上可被放大；std 变体是尺度控制的直接对照；",
                    "残差化表示的机制未解释，不得写后训练因果；",
                    "旧 r0_mechanism_controls_2026-10-08 的 claim 字符串 SA_interaction_conditional_evidence 不采用。"],
    }
    (OUT / "mechanism_verdict.json").write_text(json.dumps(verdict, ensure_ascii=False, indent=1), encoding="utf-8")

    L = ["# R0 机制裁决（canonical）与 S-resid 尺度控制", "",
         f"- **raw_SA_mechanism = unresolved**（指导 §4.1；旧 claim 字符串不采用）",
         f"- **s_resid_status = {s_status}**", "",
         "## 尺度控制（task_dev）", ""]
    if task_dev:
        L += ["| 变体 | M_b | CI95 |", "|---|---|---|"]
        for k, v in task_dev["M"].items():
            L.append(f"| {k} | {v:.4f} | [{task_dev['ci'][k][0]:.3f}, {task_dev['ci'][k][1]:.3f}] |")
        L += ["", f"- dResid={task_dev['dResid']:+.4f} / dResidStd={task_dev['dResidStd']:+.4f} / dShuffle={task_dev['dShuffle']:+.4f}"]
    L += ["", "## model 5 折（model_dev 域）", "",
          "| fold | dResid | dResidStd | dShuffle |", "|---|---|---|---|"]
    for m in model_folds:
        L.append(f"| {m['kernel']} | {m['dResid']:+.4f} | {m['dResidStd']:+.4f} | {m['dShuffle']:+.4f} |")
    L += ["", f"- model dResid 均值={metrics['model_dResid_mean']:+.4f}；全折为正={metrics['model_dResid_all_positive']}"]
    L += ["", "## 审计量（fit fold 摘要）"]
    if res:
        first = next(iter(res.values()))["audit"]
        L.append(f"- S_resid 每维 std: min={first['S_resid_per_dim']['std_min']:.3g} median={first['S_resid_per_dim']['std_median']:.3g} max={first['S_resid_per_dim']['std_max']:.3g}")
        L.append(f"- 范数: ‖S⊥‖ mean={first['norms']['S_resid']['mean']:.2f} vs ‖S‖ mean={first['norms']['S']['mean']:.2f} vs ‖A‖ mean={first['norms']['A']['mean']:.2f}")
        L.append(f"- cov top10 能量占比: S={first['cov_S']['top10_energy_ratio']:.3f} / S⊥={first['cov_S_resid']['top10_energy_ratio']:.3f}")
    L += ["", "> 结论引用方式：一切报告统一引用 `mechanism_verdict.json`（raw_SA=unresolved；S⊥ 状态见上）。"]
    (OUT / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    log(f"[audit] verdict: raw_SA=unresolved; s_resid={s_status}")


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
        (OUT / "logs" / "audit_run.log").write_text("\n".join(LOG) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
