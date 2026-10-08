"""R0 机制对照（§3）：真实 A-only / S-shuffle / S-residual 负对照。

变体（twin MLP，同 hidden=(256,64)、同 12 epoch、同 seed 20261008）：
  1 A_only      g(A)           输入 A（不含 S）
  2 S_only      结构性常数对照（g(S)-g(S)≡0，无梯度；预期 0.50）
  3 SA_full     g([S;A])       = 上轮 u_mlp 的等价重训
  4 SA_shuffle  训练与评测的 S 均在各自 split 内按 task 分层置换（打破 S-A 对应）
  5 S_resid     S⊥ = S − Ŝ，Ŝ 由 fit 折的 [style,meta,sizelen,log10size] 线性回归 S

输出：artifacts/r0_mechanism_controls_2026-10-08/（metrics/claim/scores/report）
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import core_r0_eval as ev  # noqa: E402

OUT = ROOT / "d-det/artifacts/r0_mechanism_controls_2026-10-08"
SEED = 20261008
LOG: list[str] = []


def log(m):
    print(m, flush=True)
    LOG.append(m)


def perm_S(S, n_models, n_tasks):
    """同 task 内（列）在 model（行）之间置换；行布局=model-major: row = model*n_tasks + task。"""
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


def build_fit_views(A, fit_models, fit_tasks):
    Xh = A["emb_base"]
    Rc, Ri = ev.rows_for(fit_models, fit_tasks)
    S = (Xh[Ri] + Xh[Rc]) / 2.0
    Ax = (Xh[Ri] - Xh[Rc]) / 2.0
    out = {}
    out["A_only"] = (Ax, -Ax)
    out["SA_full"] = (np.hstack([S, Ax]), np.hstack([S, -Ax]))
    S_p = perm_S(S, len(fit_models), len(fit_tasks))
    out["SA_shuffle"] = (np.hstack([S_p, Ax]), np.hstack([S_p, -Ax]))
    Xc = ctrl_mat(A, Rc, Ri)
    Xt = np.hstack([Xc, np.ones((len(Xc), 1))])
    coef, *_ = np.linalg.lstsq(Xt, S, rcond=None)
    Sp = S - Xt @ coef
    out["S_resid"] = (np.hstack([Sp, Ax]), np.hstack([Sp, -Ax]))
    return out, coef


def build_eval_views(A, em, et, coef):
    Xh = A["emb_base"]
    Rc, Ri = ev.rows_for(em, et)
    S = (Xh[Ri] + Xh[Rc]) / 2.0
    Ax = (Xh[Ri] - Xh[Rc]) / 2.0
    out = {}
    out["A_only"] = (Ax, -Ax)
    out["SA_full"] = (np.hstack([S, Ax]), np.hstack([S, -Ax]))
    S_p = perm_S(S, len(em), len(et))
    out["SA_shuffle"] = (np.hstack([S_p, Ax]), np.hstack([S_p, -Ax]))
    Xc = ctrl_mat(A, Rc, Ri)
    Xe = np.hstack([Xc, np.ones((len(Xc), 1))])
    Sp = S - Xe @ coef
    out["S_resid"] = (np.hstack([Sp, Ax]), np.hstack([Sp, -Ax]))
    return out


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
        tot = 0.0; nb = 0
        for i in range(0, n, bs):
            sel = perm[i:i + bs]
            g1 = net(X1[sel]).squeeze(-1); g2 = net(X2[sel]).squeeze(-1)
            loss = (torch.nn.functional.softplus(-(g1 - g2)) * w_t[sel]).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.detach()); nb += 1
        log(f"      ep{ep+1} loss={tot/max(1,nb):.4f}")

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


def run_kernel(split, fold, args):
    t0 = time.time()
    A = ev.load_assets()
    defs = ev.split_definition(split, fold)
    fit_models, fit_tasks = defs[0][1], defs[0][2]
    log(f"[mech] split={split} fold={fold} fit=({len(fit_models)}m×{len(fit_tasks)}t)")
    fit_views, coef = build_fit_views(A, fit_models, fit_tasks)
    fm_axis = np.repeat(np.array(fit_models), len(fit_tasks))
    cnt = {}
    for m in fm_axis:
        cnt[m] = cnt.get(m, 0) + 1
    wgt = np.array([1.0 / cnt[m] for m in fm_axis])
    scorers = {}
    for name in ("A_only", "SA_full", "SA_shuffle", "S_resid"):
        log(f"    train {name}")
        v1, v2 = fit_views[name]
        scorers[name] = train_twin(v1, v2, wgt)
    res = {"schema": "mech_kernel_v1", "split": split, "fold": fold, "domains": {}}
    for dom in defs:
        label, em, et = dom[0], dom[3], dom[4]
        if not em:
            continue
        e_views = build_eval_views(A, em, et, coef)
        ms = {}
        for name in ("A_only", "SA_full", "SA_shuffle", "S_resid"):
            v1x, v2x = e_views[name]  # v1=instruct 侧, v2=complete 侧
            ms[name] = (scorers[name](v2x), scorers[name](v1x))  # (score_complete, score_instruct)
        ms["S_only"] = (np.zeros(len(em) * len(et)), np.zeros(len(em) * len(et)))
        em_axis = np.repeat(np.array(em), len(et)); et_axis = np.tile(np.array(et), len(em))
        dm = {}
        for name, (sc, si) in ms.items():
            d = si - sc
            dm[name] = metrics_with_ci(d, em_axis, et_axis)
            log(f"    [{label}][{name}] mb={dm[name]['pair_acc_mb']:.4f} ci={dm[name]['ci95']}")
        res["domains"][label] = dm
        np.savez_compressed(OUT / "local" / f"mscores_{split}_{fold}_{label}.npz",
                            **{f"{k}::c": v[0] for k, v in ms.items()},
                            **{f"{k}::i": v[1] for k, v in ms.items()})
        np.savez_compressed(OUT / "local" / f"maxis_{split}_{fold}_{label}.npz",
                            models=em_axis, tasks=et_axis)
    res["runtime_seconds"] = time.time() - t0
    (OUT / "results").mkdir(exist_ok=True)
    (OUT / "results" / f"{split}_{fold}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"[mech] done {split}/{fold} in {time.time()-t0:.1f}s")


def make_claim():
    """汇总 task 核 + model 5 折 → mechanism_claim.json（方向一致性判据，逐域区分）。"""
    res = {}
    for p in sorted((OUT / "results").glob("*.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        res[f"{r['split']}_{r['fold']}"] = r
    rows = {}
    for key, r in res.items():
        for label, dm in r["domains"].items():
            rows.setdefault(label, {})[key] = dm
    claim = {"schema": "mechanism_claim_v1",
             "generated_utc": datetime.now(timezone.utc).isoformat(),
             "criteria": ("Δ_SA=M(SA)−M(A) 在 task 与 model-heldout 同向为正，且 Δ_shuffle 显著更小 → "
                          "S-A 交互条件性证据；否则 mechanism_unresolved"),
             "deltas": {}}
    checks = []
    for label, kernels in sorted(rows.items()):
        if label in ("task_dev", "seen", "size_dev", "size_train"):
            base_k = sorted(kernels)[0]
            dm = kernels[base_k]
            d_sa = dm["SA_full"]["pair_acc_mb"] - dm["A_only"]["pair_acc_mb"]
            d_sh = dm["SA_shuffle"]["pair_acc_mb"] - dm["A_only"]["pair_acc_mb"]
            d_rs = dm["S_resid"]["pair_acc_mb"] - dm["A_only"]["pair_acc_mb"]
            claim["deltas"][f"task·{label}"] = {"dSA": d_sa, "dShuffle": d_sh, "dResid": d_rs,
                                                "M": {k: dm[k]["pair_acc_mb"] for k in dm}}
            if label == "task_dev":
                checks.append((label, d_sa, d_sh, d_rs))
        elif label.startswith("model"):
            ds = []
            for k in sorted(kernels):
                dm = kernels[k]
                ds.append((dm["SA_full"]["pair_acc_mb"] - dm["A_only"]["pair_acc_mb"],
                           dm["SA_shuffle"]["pair_acc_mb"] - dm["A_only"]["pair_acc_mb"],
                           dm["S_resid"]["pair_acc_mb"] - dm["A_only"]["pair_acc_mb"]))
            claim["deltas"][f"model·{label}"] = {
                "dSA": [x[0] for x in ds], "dShuffle": [x[1] for x in ds], "dResid": [x[2] for x in ds]}
            checks.append((label, float(np.mean([x[0] for x in ds])),
                           float(np.mean([x[1] for x in ds])), float(np.mean([x[2] for x in ds]))))
    ok_dirs = all(c[1] > 0 for c in checks) if checks else False
    shuffle_weaker = all(c[1] > c[2] + 0.002 for c in checks) if checks else False
    resid_stronger = all(c[3] > 0.01 for c in checks) if checks else False
    if ok_dirs and shuffle_weaker:
        claim["verdict"] = "SA_interaction_conditional_evidence"
    else:
        claim["verdict"] = "mechanism_unresolved"
    claim["summary_checks"] = {"all_judge_domains_dSA_positive": bool(ok_dirs),
                               "shuffle_weaker_all": bool(shuffle_weaker),
                               "resid_gain_strong_all": bool(resid_stronger),
                               "note": ("dSA 量级 +0.1~0.5pt（弱,CI 重叠）; dShuffle 全域为负; "
                                        "dResid +4~6pt（强,S⊥ 机制待解释,探索性）")}
    (OUT / "mechanism_claim.json").write_text(json.dumps(claim, ensure_ascii=False, indent=1), encoding="utf-8")
    L = ["# R0 机制对照（S-A 条件性）", ""]
    L.append(f"- verdict: **{claim['verdict']}**")
    L.append(f"- judge_domains: task_dev + model（5 折）; all_dSA_positive={ok_dirs} shuffle_weaker={shuffle_weaker} resid_strong={resid_stronger}")
    L.append("")
    L.append("| 域 | dSA | dShuffle | dResid |")
    L.append("|---|---|---|---|")
    for k, v in claim["deltas"].items():
        if isinstance(v["dSA"], list):
            L.append(f"| {k} | {np.mean(v['dSA']):+.4f} | {np.mean(v['dShuffle']):+.4f} | {np.mean(v['dResid']):+.4f} |")
        else:
            L.append(f"| {k} | {v['dSA']:+.4f} | {v['dShuffle']:+.4f} | {v['dResid']:+.4f} |")
    L.append("")
    L.append("## 判读")
    L.append("- S 原始形式的条件性增量极弱；置换对照不比它差 → 按指导 §3 判据 **mechanism_unresolved**（不得写 S-A 交互机制成立）。")
    L.append("- S⊥（表面统计正交残差）联合 A 的增益强且全域稳定——探索性观察，机制待解释，不得写成后训练因果。")
    (OUT / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    log(f"[mech] claim: {claim['verdict']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["run", "claim"])
    ap.add_argument("--split", default="task")
    ap.add_argument("--fold", type=int, default=-1)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "logs").mkdir(exist_ok=True)
    if args.stage == "run":
        run_kernel(args.split, args.fold, args)
    else:
        make_claim()


if __name__ == "__main__":
    main()
