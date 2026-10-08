"""role_blind corrective rerun（7ee36fb 修复裁定 §2）。

修复点：
  1. 标准化双视图从**原始构造**分别计算：[main, A] 与 [main, -A] 各自标准化；
     A 块采用指导首选方案：均值严格=0、尺度=sqrt(mean(A^2))（RMS）→ 交换后 A 块严格反号；
     主块（S/R/zero）用 fit 均值/标准差。**禁止对标准化 A 块直接取负**。
  2. 补标准化交换断言（max|V1_A+V2_A|=0、主块相同、flip 一致）+ 记录 mu_A/sigma_A。
  3. torch.manual_seed(seed)/numpy/python/cuda seed；保存每 seed 初始化哈希、final loss、分数哈希。
  4. 内层 cross-fit 按 **task 分组**（同 task 所有 model 同折）。
  5. 指标统一 model-balanced pair-direction accuracy；paired CI 用**同一 task 抽样序列**。
  6. oracle 视图=仅原始 instruct 侧计算并复用于两方向（受辅助信息上界，非机制证据）。

smoke：先跑冻结 task fold 的完整变换断言（--stage smoke）；
全量：3 seeds × task/model folds（--stage run --split task|model --fold）。

输出：artifacts/role_blind_residual_corrective_2026-10-08/
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import core_r0_eval as ev  # noqa: E402

OUT = ROOT / "d-det/artifacts/role_blind_residual_corrective_2026-10-08"
SEED = 20261008
SEEDS = (0, 1, 2)
LOG: list[str] = []
A_BLOCK = 768


def log(m):
    print(m, flush=True)
    LOG.append(m)


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ctrl_rows(A, R):
    st = A["style"][R].astype(np.float64); mt = A["meta"][R].astype(np.float64)
    sz = A["sizelen"][R].astype(np.float64)
    return np.hstack([st, mt, sz])


def build_raw(A, models, tasks):
    Xh = A["emb_base"]
    R0, R1 = ev.rows_for(models, tasks)
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
    C_sym, C1, C2, S, Ax = build_raw(A, fit_models, fit_tasks)
    from sklearn.linear_model import Ridge
    n = len(S)
    nt = len(fit_tasks)
    task_of = np.tile(np.arange(nt), len(fit_models))
    fold_of = task_of % 5  # 内层 cross-fit 按 task 分组（同 task 所有 model 同折）
    oof = np.zeros_like(S)
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
    return {"S": S, "A": Ax, "R": R, "R_or": R_or, "R_sh": R_sh, "ridge": ridge, "coef_o": coef_o}


def make_std_params(fit_main, fit_A):
    """主块 (mu, sd)；A 块 (0, rms_A)（指导首选方案，冻结）。"""
    mu = fit_main.mean(0); sd = fit_main.std(0); sd[sd < 1e-8] = 1.0
    rms = float(np.sqrt(np.mean(fit_A ** 2)))
    return {"main_mu": mu, "main_sd": sd, "a_rms": rms,
            "a_mu_raw": float(fit_A.mean()), "a_rms_value": rms}


def std_view(params, X_raw):
    """X_raw=[main, A]（原始，A 可能已取负）；主块用 fit mu/sd、A 块用 RMS（mu=0）。"""
    d_main = X_raw.shape[1] - A_BLOCK
    Y = np.empty_like(X_raw, dtype=np.float64)
    Y[:, :d_main] = (X_raw[:, :d_main] - params["main_mu"]) / params["main_sd"]
    Y[:, d_main:] = X_raw[:, d_main:] / params["a_rms"]
    return Y


VIEW_SPECS = {
    "A_only": lambda D: (D["A"], -D["A"]),
    "zero_A": lambda D: (np.hstack([np.zeros_like(D["S"]), D["A"]]),
                         np.hstack([np.zeros_like(D["S"]), -D["A"]])),
    "SA_full": lambda D: (np.hstack([D["S"], D["A"]]), np.hstack([D["S"], -D["A"]])),
    "roleblind": lambda D: (np.hstack([D["R"], D["A"]]), np.hstack([D["R"], -D["A"]])),
    "oracle": lambda D: (np.hstack([D["R_or"], D["A"]]), np.hstack([D["R_or"], -D["A"]])),
    "rb_shuffle": lambda D: (np.hstack([D["R_sh"], D["A"]]), np.hstack([D["R_sh"], -D["A"]])),
}


def build_std_views(fit_data, fit=True, params_store=None):
    """返回 {view: (v1_std, v2_std)}；fit=True 时从原始统计量构造 params_store。"""
    out = {}
    for name, fn in VIEW_SPECS.items():
        v1_raw, v2_raw = fn(fit_data)
        if fit:
            # 主块统计用该视图的主块原始数据
            d_main = v1_raw.shape[1] - A_BLOCK
            p = make_std_params(v1_raw[:, :d_main], fit_data["A"])
            params_store[name] = p
        else:
            p = params_store[name]
        v1 = std_view(p, v1_raw); v2 = std_view(p, v2_raw)
        out[name] = (v1.astype(np.float32), v2.astype(np.float32))
    return out


def train_twin(v1, v2, wgt, epochs=10, hidden=(256, 64), seed=0):
    import torch
    import torch.nn as nn
    set_seeds(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = v1.shape[1]
    net = nn.Sequential(nn.Linear(d, hidden[0]), nn.ReLU(), nn.Linear(hidden[0], hidden[1]),
                        nn.ReLU(), nn.Linear(hidden[1], 1)).to(device)
    init_hash = hashlib.sha1(repr([p.detach().cpu().numpy().tobytes() for p in net.parameters()]).encode()).hexdigest()[:12]
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    X1 = torch.tensor(np.asarray(v1, dtype=np.float32), device=device)
    X2 = torch.tensor(np.asarray(v2, dtype=np.float32), device=device)
    w_t = torch.tensor(np.asarray(wgt, dtype=np.float32), device=device)
    n = len(X1); bs = 1024
    rng = np.random.default_rng(seed)
    last = None
    for ep in range(epochs):
        perm = rng.permutation(n)
        for i in range(0, n, bs):
            sel = perm[i:i + bs]
            g1 = net(X1[sel]).squeeze(-1); g2 = net(X2[sel]).squeeze(-1)
            loss = (torch.nn.functional.softplus(-(g1 - g2)) * w_t[sel]).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            last = float(loss.detach())

    def score(V):
        with torch.inference_mode():
            return net(torch.tensor(np.asarray(V, dtype=np.float32), device=device)).squeeze(-1).cpu().numpy()
    del X1, X2  # 释放训练副本（score 不依赖；降常驻内存）
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return score, {"init_hash": init_hash, "final_loss": last}


def acc_mb(pos, inv_m, nm):
    sums = np.bincount(inv_m, weights=pos, minlength=nm)
    cnts = np.bincount(inv_m, minlength=nm)
    valid = cnts > 0
    return float(np.mean(sums[valid] / cnts[valid]))


def paired_ci(scores_by_arm, y_units, models, tasks, pairs, nb=500, seed=SEED):
    """同一 task 抽样序列的配对差值 CI（model-balanced pair-direction accuracy）。"""
    uniq_m, inv_m = np.unique(models, return_inverse=True)
    uniq_t, inv_t = np.unique(tasks, return_inverse=True)
    nm = len(uniq_m)
    pos = {k: (v > 0).astype(float) + 0.5 * (v == 0).astype(float) for k, v in scores_by_arm.items()}
    rng = np.random.default_rng(seed)
    idx_by = {q: np.where(inv_t == q)[0] for q in range(len(uniq_t))}
    deltas = {f"{a}-{b}": [] for a, b in pairs}
    for _ in range(nb):
        pick = rng.choice(len(uniq_t), size=len(uniq_t), replace=True)
        idx = np.concatenate([idx_by[q] for q in pick])
        accs = {k: acc_mb(pos[k][idx], inv_m[idx], nm) for k in pos}
        for a, b in pairs:
            deltas[f"{a}-{b}"].append(accs[a] - accs[b])
    return {k: {"mean": float(np.mean(v)), "ci95": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]}
            for k, v in deltas.items()}


def run_kernel(split, fold, smoke=False, skip_seen=False):
    t0 = time.time()
    A = ev.load_assets()
    defs = ev.split_definition(split, fold)
    fit_models, fit_tasks = defs[0][1], defs[0][2]
    log(f"[rb] split={split} fold={fold} fit=({len(fit_models)}m×{len(fit_tasks)}t) smoke={smoke}")
    ff = fit_fold(A, fit_models, fit_tasks)
    Dfit = {k: ff[k] for k in ("S", "A", "R", "R_or", "R_sh")}
    params = {}
    views_fit = build_std_views(Dfit, fit=True, params_store=params)
    # ---------- 完整变换断言 ----------
    audit = {}
    p_sa = params["SA_full"]
    v1_raw = np.hstack([Dfit["S"], Dfit["A"]]); v2_raw = np.hstack([Dfit["S"], -Dfit["A"]])
    v1 = std_view(p_sa, v1_raw); v2 = std_view(p_sa, v2_raw)
    audit["std_swap_A_strict_negate"] = bool(np.abs(v1[:, -A_BLOCK:] + v2[:, -A_BLOCK:]).max() == 0.0)
    audit["std_swap_main_identical"] = bool(np.abs(v1[:, :-A_BLOCK] - v2[:, :-A_BLOCK]).max() == 0.0)
    audit["flip_of_std_equals_std_of_flip_Amax"] = float(np.abs(np.hstack([v1[:, :-A_BLOCK], -v1[:, -A_BLOCK:]]) - v2).max())
    audit["mu_A_raw"] = p_sa["a_mu_raw"]
    audit["sigma_A_used_rms"] = p_sa["a_rms_value"]
    # 原始层交换断言
    C_sym, C1, C2, S, Ax = build_raw(A, fit_models, fit_tasks)
    Xh = A["emb_base"]
    R0, R1 = ev.rows_for(fit_models, fit_tasks)
    C1s = ctrl_rows(A, R1); C2s = ctrl_rows(A, R0)
    C_sym_sw = np.hstack([(C1s + C2s) / 2.0, np.abs(C1s - C2s), C1s * C2s])
    sel = np.random.default_rng(SEED).choice(len(S), size=min(300, len(S)), replace=False)
    audit["raw_swap_Csym_identical"] = bool(np.allclose(C_sym[sel], C_sym_sw[sel], atol=1e-9))
    audit["raw_swap_A_negated"] = bool(np.allclose(Ax[sel], -((Xh[R0] - Xh[R1]) / 2.0)[sel], atol=1e-12))
    log(f"[rb] audit: {json.dumps({k: v for k, v in audit.items() if k.startswith(('std','raw','flip'))}, ensure_ascii=False)}")
    if not (audit["std_swap_A_strict_negate"] and audit["std_swap_main_identical"]
            and audit["flip_of_std_equals_std_of_flip_Amax"] == 0.0):
        raise RuntimeError("standardization swap assertions failed — stop branch")

    if smoke:
        (OUT / "interface_audit.json").write_text(json.dumps(
            {"schema": "rb_corrective_interface_audit_v1", "generated_utc": datetime.now(timezone.utc).isoformat(),
             "stage": "smoke", "split": split, "fold": fold, **audit}, ensure_ascii=False, indent=1), encoding="utf-8")

    fm_axis = np.repeat(np.array(fit_models), len(fit_tasks))
    cnt = {}
    for m in fm_axis:
        cnt[m] = cnt.get(m, 0) + 1
    wgt = np.array([1.0 / cnt[m] for m in fm_axis])
    tag = "smoke" if smoke else f"{split}_{fold}"
    scorers = {}
    seed_meta = {}
    for name, (v1s, v2s) in views_fit.items():
        for s in (SEEDS if not smoke else SEEDS[:1]):
            sc, meta = train_twin(v1s, v2s, wgt, seed=s)
            scorers.setdefault(name, {})[s] = sc
            seed_meta[f"{name}::s{s}"] = meta
    import gc
    del views_fit, Dfit  # 释放 fit 视图大数组（ff 保留：域评估需 ff["ridge"]/coef_o）
    gc.collect()
    res = {"schema": "rb_corrective_kernel_v1", "split": split, "fold": fold,
           "audit": audit, "seed_meta": seed_meta, "domains": {}}
    for dom in defs:
        if skip_seen and dom[0] in ("seen",):
            continue
        label, em, et = dom[0], dom[3], dom[4]
        if not em:
            continue
        ff_eval = {
            "S": None, "A": None, "R": None, "R_or": None, "R_sh": None}
        C_sym_e, C1_e, C2_e, S_e, Ax_e = build_raw(A, em, et)
        R_e = S_e - ff["ridge"].predict(C_sym_e)
        Xt_e = np.hstack([C2_e, np.ones((len(C2_e), 1))])
        R_or_e = S_e - Xt_e @ ff["coef_o"]
        R_sh_e = perm_rows(R_e, len(em), len(et))
        De = {"S": S_e, "A": Ax_e, "R": R_e, "R_or": R_or_e, "R_sh": R_sh_e}
        ev_views = build_std_views(De, fit=False, params_store=params)
        em_axis = np.repeat(np.array(em), len(et)); et_axis = np.tile(np.array(et), len(em))
        arm_scores = {}
        per_seed_scores = {}
        for name, (v1x, v2x) in ev_views.items():
            d_seeds = []
            for s in scorers[name]:
                sc = scorers[name][s]
                d = sc(v1x) - sc(v2x)
                d_seeds.append(d)
                per_seed_scores[f"{name}::s{s}"] = d.astype(np.float32)
            arm_scores[name] = np.mean(d_seeds, axis=0)
        np.savez_compressed(OUT / "local" / f"scores_{tag}_{label}.npz", **per_seed_scores,
                            models=em_axis, tasks=et_axis)
        # acc + paired CI（同一 task 抽样序列）
        dm = {}
        for name, d in arm_scores.items():
            pos = (d > 0).astype(float) + 0.5 * (d == 0).astype(float)
            uniq_m, inv_m = np.unique(em_axis, return_inverse=True)
            dm[name] = {"pair_dir_acc_mb": acc_mb(pos, inv_m, len(uniq_m))}
        pairs = [("roleblind", "zero_A"), ("roleblind", "A_only"), ("roleblind", "SA_full"),
                 ("roleblind", "oracle"), ("roleblind", "rb_shuffle")]
        dm["paired_ci"] = paired_ci(arm_scores, None, em_axis, et_axis, pairs)
        # order-swap identity audit（d 恒等反号）
        rb_v1, rb_v2 = ev_views["roleblind"]
        d_orig = arm_scores["roleblind"]
        d_swap = np.mean([scorers["roleblind"][s](rb_v2) - scorers["roleblind"][s](rb_v1) for s in scorers["roleblind"]], axis=0)
        dm["order_swap_identity_max_abs"] = float(np.abs(d_orig + d_swap).max())
        # 分数哈希
        dm["scores_sha1"] = hashlib.sha1(np.asarray(arm_scores["roleblind"]).tobytes()).hexdigest()[:12]
        res["domains"][label] = dm
        log(f"    [{label}] roleblind={dm['roleblind']['pair_dir_acc_mb']:.4f} zero_A={dm['zero_A']['pair_dir_acc_mb']:.4f} "
            f"oracle={dm['oracle']['pair_dir_acc_mb']:.4f} SA={dm['SA_full']['pair_dir_acc_mb']:.4f} "
            f"Δ(rb-zeroA)={dm['paired_ci']['roleblind-zero_A']['mean']:+.4f}")
    res["runtime_seconds"] = time.time() - t0
    (OUT / "results").mkdir(exist_ok=True)
    (OUT / "results" / f"{tag}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"[rb] done {tag} in {time.time()-t0:.1f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["smoke", "run", "sum"])
    ap.add_argument("--split", default="task")
    ap.add_argument("--fold", type=int, default=-1)
    ap.add_argument("--skip-seen", action="store_true",
                    help="跳过 seen 域（cgroup 32GB 内存限制下 fit 视图+评分并存风险）")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "logs").mkdir(exist_ok=True)
    if args.stage == "smoke":
        run_kernel("task", -1, smoke=True)
    elif args.stage == "run":
        run_kernel(args.split, args.fold, smoke=False, skip_seen=args.skip_seen)
    else:
        summarize()
    (OUT / "logs" / "rb_run.log").write_text("\n".join(LOG) + "\n", encoding="utf-8")


def summarize():
    res = {}
    for p in sorted((OUT / "results").glob("*.json")):
        if p.stem == "smoke":
            continue
        r = json.loads(p.read_text(encoding="utf-8"))
        res[f"{r['split']}_{r['fold']}"] = r
    task = res.get("task_-1", {})
    model = [r for k, r in res.items() if k.startswith("model_")]
    out = {"schema": "rb_corrective_summary_v1", "generated_utc": datetime.now(timezone.utc).isoformat(),
           "task_dev": task.get("domains", {}).get("task_dev"),
           "model_folds": {k: res[k]["domains"].get("model_dev") for k in res if k.startswith("model_")}}
    # model 折均值 Δ
    if out["model_folds"]:
        keys = [k for k in out["model_folds"] if out["model_folds"][k]]
        deltas = {}
        for pair in ("roleblind-zero_A", "roleblind-A_only", "roleblind-SA_full", "roleblind-oracle", "roleblind-rb_shuffle"):
            vals = [out["model_folds"][k]["paired_ci"][pair]["mean"] for k in keys]
            deltas[pair] = {"mean": float(np.mean(vals)), "min": float(np.min(vals)), "max": float(np.max(vals))}
        out["model_deltas"] = deltas
    (OUT / "metrics_role_blind_corrective.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
