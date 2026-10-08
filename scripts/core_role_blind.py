"""实验 B：角色未知的 S-resid（role-blind residual；train/dev）。

接口（指导 §3）：
  C_sym = [(C1+C2)/2; |C1-C2|; C1⊙C2]（交换不变）；S~C_sym 回归（fit fold，ridge + 内层 5-fold
  cross-fit 产生训练残差；eval 用全 fit 系数）。R=S-Ŝ(C_sym) 交换不变；A=(e2-e1)/2 变号。
断言组：随机 50% 翻转一致性 / 交换不变性（R 同、A 变号、C_sym 同）/ 角色名无依赖（代码层）。
三臂 + 参照：A_only / [0;A]（匹配容量）/ SA_full / **role-blind R+A** / oracle（C_instruct 残差 + A，
上界）/ R_shuffle。3 seeds；task_dev + model 5 折；fit-only 标准化（真正应用）。
禁止读取 heldout h 的 train 行（本实验不涉及 h 留出探针；数据访问声明已修正）。

输出：artifacts/role_blind_residual_2026-10-08/
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

OUT = ROOT / "d-det/artifacts/role_blind_residual_2026-10-08"
SEED = 20261008
SEEDS = (0, 1, 2)
LOG: list[str] = []


def log(m):
    print(m, flush=True)
    LOG.append(m)


def ctrl_rows(A, R):
    st = A["style"][R].astype(np.float64); mt = A["meta"][R].astype(np.float64)
    sz = A["sizelen"][R].astype(np.float64)
    return np.hstack([st, mt, sz])


def build_views(A, models, tasks):
    Xh = A["emb_base"]
    R0, R1 = ev.rows_for(models, tasks)  # (complete, instruct)
    C1 = ctrl_rows(A, R0); C2 = ctrl_rows(A, R1)
    C_sym = np.hstack([(C1 + C2) / 2.0, np.abs(C1 - C2), C1 * C2])
    S = (Xh[R1] + Xh[R0]) / 2.0
    Ax = (Xh[R1] - Xh[R0]) / 2.0
    return C_sym, C1, C2, S, Ax


def perm_rows(X, n_models, n_tasks):
    rng = np.random.default_rng(SEED)
    X2 = X.copy()
    for tj in range(n_tasks):
        rows = np.arange(n_models) * n_tasks + tj
        X2[rows] = X[rows[rng.permutation(n_models)]]
    return X2


def fit_fold(A, fit_models, fit_tasks):
    """fit fold 的视图与统计量。返回 dict。"""
    C_sym, C1, C2, S, Ax = build_views(A, fit_models, fit_tasks)
    from sklearn.linear_model import Ridge
    n = len(S)
    oof = np.zeros_like(S)
    fold_of = np.arange(n) % 5
    for k in range(5):
        tr = fold_of != k; dv = fold_of == k
        rg = Ridge(alpha=1.0).fit(C_sym[tr], S[tr])
        oof[dv] = rg.predict(C_sym[dv])
    ridge = Ridge(alpha=1.0).fit(C_sym, S)
    R = S - oof
    Xt = np.hstack([C2, np.ones((len(C2), 1))])
    coef_o, *_ = np.linalg.lstsq(Xt, S, rcond=None)
    R_or = S - Xt @ coef_o
    R_sh = perm_rows(R, len(fit_models), len(fit_tasks))
    return {"S": S, "A": Ax, "R": R, "R_or": R_or, "R_sh": R_sh, "ridge": ridge,
            "coef_o": coef_o, "C_sym": C_sym}


def make_views(stats, D, fit=False):
    """由单元级 D(dict: S,A,R,R_or,R_sh) 构造变体 (v1, v2)；fit=True 时记录标准化统计。"""
    out = {}
    def std_pair(name, W):
        if fit:
            mu = W.mean(0); sd = W.std(0); sd[sd < 1e-8] = 1.0
            stats[name] = (mu, sd)
        else:
            mu, sd = stats[name]
        W1 = (W - mu) / sd
        W2 = W1.copy()
        W2[:, -D["A"].shape[1]:] *= -1.0  # A 块取负（反对称侧）
        return W1.astype(np.float32), W2.astype(np.float32)
    zero = np.zeros_like(D["S"])
    out["A_only"] = std_pair("A_only", D["A"])
    out["zero_A"] = std_pair("zero_A", np.hstack([zero, D["A"]]))
    out["SA_full"] = std_pair("SA_full", np.hstack([D["S"], D["A"]]))
    out["roleblind"] = std_pair("roleblind", np.hstack([D["R"], D["A"]]))
    out["oracle"] = std_pair("oracle", np.hstack([D["R_or"], D["A"]]))
    out["rb_shuffle"] = std_pair("rb_shuffle", np.hstack([D["R_sh"], D["A"]]))
    return out


def eval_fold(A, em, et, ff, stats):
    C_sym, C1, C2, S, Ax = build_views(A, em, et)
    R = S - ff["ridge"].predict(C_sym)
    Xt = np.hstack([C2, np.ones((len(C2), 1))])
    R_or = S - Xt @ ff["coef_o"]
    R_sh = perm_rows(R, len(em), len(et))
    D = {"S": S, "A": Ax, "R": R, "R_or": R_or, "R_sh": R_sh}
    return make_views(stats, D, fit=False)


def train_twin(v1, v2, wgt, epochs=10, hidden=(256, 64), seed=0):
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


def metrics_ci(d, models, tasks, nb=500, seed=SEED):
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
    return point, [float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))]


def interface_audit(A, fit_models, fit_tasks):
    """断言组：交换不变性 / 50% 翻转一致性 / 角色名无依赖（结构声明）。"""
    C_sym, C1, C2, S, Ax = build_views(A, fit_models, fit_tasks)
    n = len(S)
    rng = np.random.default_rng(SEED)
    sel = rng.choice(n, size=min(300, n), replace=False)
    # 交换行序重建（行对互换）→ C_sym 应相同、A 应变号
    Xh = A["emb_base"]
    R0, R1 = ev.rows_for(fit_models, fit_tasks)
    C1s = ctrl_rows(A, R1); C2s = ctrl_rows(A, R0)
    C_sym_sw = np.hstack([(C1s + C2s) / 2.0, np.abs(C1s - C2s), C1s * C2s])
    S_sw = (Xh[R0] + Xh[R1]) / 2.0
    Ax_sw = (Xh[R0] - Xh[R1]) / 2.0
    audit = {
        "swap_Csym_identical": bool(np.allclose(C_sym[sel], C_sym_sw[sel], atol=1e-9)),
        "swap_S_identical": bool(np.allclose(S[sel], S_sw[sel], atol=1e-9)),
        "swap_A_negated": bool(np.allclose(Ax[sel], -Ax_sw[sel], atol=1e-12)),
        "n_checked": int(len(sel)),
        "role_name_dependence": "none (C_sym uses only the multiset of per-side features; no mode labels)",
        "flip50_note": "50% 随机翻转与交换等价（row-order swap），由 swap_* 断言覆盖；训练/评测不使用 mode 先验构造控制残差",
    }
    # 残差交换不变（需要 fit coef）：用全 fit ridge
    from sklearn.linear_model import Ridge
    rg = Ridge(alpha=1.0).fit(C_sym, S)
    R = S - rg.predict(C_sym)
    R_sw = S_sw - rg.predict(C_sym_sw)
    audit["swap_R_identical"] = bool(np.allclose(R[sel], R_sw[sel], atol=1e-8))
    return audit


def run_kernel(split, fold):
    t0 = time.time()
    A = ev.load_assets()
    defs = ev.split_definition(split, fold)
    fit_models, fit_tasks = defs[0][1], defs[0][2]
    log(f"[B] split={split} fold={fold} fit=({len(fit_models)}m×{len(fit_tasks)}t)")
    ff = fit_fold(A, fit_models, fit_tasks)
    if split == "task" and fold == -1:
        audit = interface_audit(A, fit_models, fit_tasks)
        (OUT / "interface_audit.json").write_text(json.dumps(
            {"schema": "role_blind_interface_audit_v1", "generated_utc": datetime.now(timezone.utc).isoformat(),
             **audit}, ensure_ascii=False, indent=1), encoding="utf-8")
        log(f"[B] interface audit: {audit['swap_Csym_identical']}/{audit['swap_A_negated']}/"
            f"{audit.get('swap_R_identical')}")
        assert audit["swap_Csym_identical"] and audit["swap_A_negated"] and audit["swap_R_identical"]
    stats = {}
    Dfit = {"S": ff["S"], "A": ff["A"], "R": ff["R"], "R_or": ff["R_or"], "R_sh": ff["R_sh"]}
    views_fit = make_views(stats, Dfit, fit=True)
    fm_axis = np.repeat(np.array(fit_models), len(fit_tasks))
    cnt = {}
    for m in fm_axis:
        cnt[m] = cnt.get(m, 0) + 1
    wgt = np.array([1.0 / cnt[m] for m in fm_axis])
    scorers = {}
    for name, (v1, v2) in views_fit.items():
        for s in SEEDS:
            log(f"    train {name} seed{s}")
            scorers.setdefault(name, []).append(train_twin(v1, v2, wgt, seed=s))
    res = {"schema": "role_blind_kernel_v1", "split": split, "fold": fold, "domains": {}}
    for dom in defs:
        label, em, et = dom[0], dom[3], dom[4]
        if not em:
            continue
        evs = eval_fold(A, em, et, ff, stats)
        em_axis = np.repeat(np.array(em), len(et)); et_axis = np.tile(np.array(et), len(em))
        dm = {}
        for name, (v1x, v2x) in evs.items():
            scs = np.mean([sc(v2x) for sc in scorers[name]], axis=0)
            sis = np.mean([sc(v1x) for sc in scorers[name]], axis=0)
            d = sis - scs
            pt, ci = metrics_ci(d, em_axis, et_axis)
            dm[name] = {"pair_acc_mb": pt, "ci95": ci}
            log(f"    [{label}][{name}] mb={pt:.4f} ci={ci}")
        res["domains"][label] = dm
    res["runtime_seconds"] = time.time() - t0
    (OUT / "results").mkdir(exist_ok=True)
    (OUT / "results" / f"{split}_{fold}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"[B] done {split}/{fold} in {time.time()-t0:.1f}s")


def summarize():
    import glob
    res = {}
    for p in sorted((OUT / "results").glob("*.json")):
        r = json.loads(Path(p).read_text(encoding="utf-8"))
        res[f"{r['split']}_{r['fold']}"] = r
    task_dev = res.get("task_-1", {}).get("domains", {}).get("task_dev")
    model_folds = []
    for k, r in res.items():
        if k.startswith("model_") and "model_dev" in r["domains"]:
            model_folds.append(r["domains"]["model_dev"])
    variants = list(task_dev.keys()) if task_dev else []
    out = {"schema": "role_blind_summary_v1", "generated_utc": datetime.now(timezone.utc).isoformat(),
           "task_dev": task_dev, "model_folds": {f"fold{i}": dm for i, dm in enumerate(model_folds)},
           "deltas_task": {}, "deltas_model_mean": {}}
    if task_dev:
        for name in variants:
            if name == "roleblind":
                continue
            out["deltas_task"][f"roleblind_minus_{name}"] = task_dev["roleblind"]["pair_acc_mb"] - task_dev[name]["pair_acc_mb"]
        for name in variants:
            out["deltas_model_mean"][f"roleblind_minus_{name}"] = (
                float(np.mean([dm["roleblind"]["pair_acc_mb"] - dm[name]["pair_acc_mb"] for dm in model_folds]))
                if model_folds else None)
    (OUT / "metrics_role_blind.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    L = ["# 实验 B：角色未知残差（role-blind）", ""]
    if task_dev:
        L += ["## task_dev（M_b）", "", "| 变体 | M_b | CI95 |", "|---|---|---|"]
        for name in variants:
            L.append(f"| {name} | {task_dev[name]['pair_acc_mb']:.4f} | {task_dev[name]['ci95']} |")
        L += ["", "## Δ（roleblind − 基）"]
        for k, v in out["deltas_task"].items():
            L.append(f"- {k}: {v:+.4f}")
    if model_folds:
        L += ["", "## model 5 折均值 Δ"]
        for k, v in out["deltas_model_mean"].items():
            L.append(f"- {k}: {v:+.4f}" if v is not None else f"- {k}: —")
    (OUT / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    sign = None
    if out["deltas_model_mean"].get("roleblind_minus_oracle") is not None:
        sign = out["deltas_model_mean"]["roleblind_minus_oracle"]
    log(f"[B] summary: model Δ(rb−oracle)={sign}")


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
        (OUT / "logs" / "rb_run.log").write_text("\n".join(LOG) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
