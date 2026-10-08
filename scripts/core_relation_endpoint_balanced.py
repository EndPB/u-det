"""实验 A：端点平衡的关系增量（48 折；train/dev）。

标签=same_observed_series（任意同系列对为正）；48 折=每系列各留出 1 成员（4x4x3）。
训练权重按 plan（每类总权 1；端点两侧相等）；评测权重（每 anchor 两侧 1/3；partner 平衡）。
读出：endpoint-only sanity（trivial 分数，加权应≈0.5）/ independent-source composition /
symmetric q（[|diff|;prod]，加权 LR，伪特征 scaler fit-fold）/ strong P0（partial：style+meta 组合 +
P0 对称对 LR）/ relation residual（冻结 compose + γ r(q_sym)，γ0=0）。
共享 task-cluster bootstrap（500，所有折同一 task 序列）；置换 20 次（折0）。
主比较：residual − composition、residual − P0（paired delta CI）。

输出：artifacts/relation_endpoint_balanced_2026-10-08/
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

PLAN = ROOT / "d-det/artifacts/endpoint_balanced_relation_plan_2026-10-08"
OUT = ROOT / "d-det/artifacts/relation_endpoint_balanced_2026-10-08"
SEED = 20261008
SEEDS = (0, 1, 2)
LOG: list[str] = []


def log(m):
    print(m, flush=True)
    LOG.append(m)


def weighted_auc(y, s, w):
    idx = np.argsort(-s, kind="mergesort")
    ys, ss, ws = y[idx], s[idx], w[idx]
    Wp = ws[ys == 1].sum(); Wn = ws[ys == 0].sum()
    total = 0.0; i = 0; cum_p = 0.0; n = len(s)
    while i < n:
        j = i
        while j < n and ss[j] == ss[i]:
            j += 1
        sp = ws[i:j][ys[i:j] == 1].sum(); sn = ws[i:j][ys[i:j] == 0].sum()
        total += sn * (cum_p + 0.5 * sp)
        cum_p += sp
        i = j
    return total / (Wp * Wn + 1e-12)


def weighted_ap(y, s, w):
    idx = np.argsort(-s, kind="mergesort")
    ys, ss, ws = y[idx], s[idx], w[idx]
    Wp = ws[ys == 1].sum()
    cum_tp = 0.0; cum_fp = 0.0; ap = 0.0; i = 0; n = len(s); prev_rec = 0.0
    while i < n:
        j = i
        while j < n and ss[j] == ss[i]:
            j += 1
        tp = ws[i:j][ys[i:j] == 1].sum(); fp = ws[i:j][ys[i:j] == 0].sum()
        cum_tp += tp; cum_fp += fp
        rec = cum_tp / Wp
        prec = cum_tp / (cum_tp + cum_fp + 1e-12)
        ap += (rec - prev_rec) * prec
        prev_rec = rec
        i = j
    return ap


def unweighted_auc(y, s):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s))


def sgd_weighted(Xtr, ytr, wtr, Xdv, seeds=SEEDS, epochs=5, bs=4096, alpha=1e-6):
    from sklearn.linear_model import SGDClassifier
    Xtr = np.asarray(Xtr, dtype=np.float32); Xdv = np.asarray(Xdv, dtype=np.float32)
    classes = np.array([0, 1])
    scores = np.zeros(len(Xdv))
    for seed in seeds:
        clf = SGDClassifier(loss="log_loss", alpha=alpha, random_state=seed)
        rng = np.random.default_rng(seed)
        for ep in range(epochs):
            perm = rng.permutation(len(Xtr))
            for i in range(0, len(perm), bs):
                sel = perm[i:i + bs]
                clf.partial_fit(Xtr[sel], ytr[sel], classes=classes, sample_weight=wtr[sel])
        scores += clf.decision_function(Xdv)
    return scores / len(seeds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="all", help="'all' or comma list e.g. 0 or 0,1,2")
    ap.add_argument("--perm-folds", default="0", help="folds for 20x permutation (default fold 0)")
    ap.add_argument("--skip-perm", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "logs").mkdir(exist_ok=True)
    (OUT / "local").mkdir(exist_ok=True)
    plan = json.loads((PLAN / "endpoint_balanced_plan.json").read_text(encoding="utf-8"))
    folds = plan["folds"]
    A = ev.load_assets()
    splits = {t: A["splits"][A["TASK_I"][t] * 2] for t in A["tasks"]}
    tr_tasks = [t for t in A["tasks"] if splits[t] == "train"]
    dv_tasks = [t for t in A["tasks"] if splits[t] == "dev"]
    nt, nv = len(tr_tasks), len(dv_tasks)
    all_members = [m for s in plan["series"] for m in plan["series_members"][s]]
    rowidx = {m: i for i, m in enumerate(all_members)}
    tidx_tr = {t: i for i, t in enumerate(tr_tasks)}
    tidx_dv = {t: i for i, t in enumerate(dv_tasks)}
    log(f"[A] tasks train={nt} dev={nv}; members={len(all_members)}; folds={len(folds)}")

    H = A["emb_base"]
    Um = np.zeros((len(all_members), nt, 1536), dtype=np.float32)
    for m in all_members:
        mi = A["MODEL_I"][m]
        for k, t in enumerate(tr_tasks):
            r = mi * A["nT"] * 2 + A["TASK_I"][t] * 2
            hc, hi = H[r], H[r + 1]
            Um[rowidx[m], k] = np.concatenate([(hc + hi) / 2.0, (hi - hc) / 2.0])
    Udv = np.zeros((len(all_members), nv, 1536), dtype=np.float32)
    for m in all_members:
        mi = A["MODEL_I"][m]
        for k, t in enumerate(dv_tasks):
            r = mi * A["nT"] * 2 + A["TASK_I"][t] * 2
            hc, hi = H[r], H[r + 1]
            Udv[rowidx[m], k] = np.concatenate([(hc + hi) / 2.0, (hi - hc) / 2.0])

    def surf_mat(members, tasks, mode=1):
        """[style, meta, sizelen] on instruct rows for members × tasks: (nM, nT, F)"""
        Fs = []
        for m in members:
            mi = A["MODEL_I"][m]
            rows = np.array([mi * A["nT"] * 2 + A["TASK_I"][t] * 2 + mode for t in tasks])
            Fs.append(np.hstack([A["style"][rows], A["meta"][rows], A["sizelen"][rows]]))
        return np.stack(Fs).astype(np.float64)  # (nM, nT, F)

    def pair_feats(Umat, pair_midx, pair_tidx, feat_fn):
        """向量化 pair 特征：pair_midx=(n_pair,2) 成员行号；pair_tidx=task 列号。"""
        i, j = pair_midx[:, 0], pair_midx[:, 1]
        ui = Umat[i, pair_tidx]; uj = Umat[j, pair_tidx]
        return feat_fn(ui, uj)

    def qsym(ui, uj):
        return np.hstack([np.abs(ui - uj), ui * uj]).astype(np.float32)

    metrics_all = {}
    target_folds = list(range(len(folds))) if args.folds == "all" else [int(x) for x in args.folds.split(",")]
    for fid in target_folds:
        fold = folds[fid]
        tf0 = time.time()
        M = fold["train_members"]
        # 训练对（含 task 展开）：pos
        tr_pos = []  # (a, b, t)
        tr_pos_w = []
        for p in fold["train_pos"]:
            w0 = p["weight"] / nt
            for t in tr_tasks:
                tr_pos.append((p["a"], p["b"], t)); tr_pos_w.append(w0)
        tr_neg = []
        tr_neg_w = []
        for p in fold["train_neg"]:
            w0 = p["weight"] / nt
            for t in tr_tasks:
                tr_neg.append((p["a"], p["b"], t)); tr_neg_w.append(w0)
        wp = np.array(tr_pos_w, dtype=np.float64); wn = np.array(tr_neg_w, dtype=np.float64)
        # 统一缩放至均值 1（整体）
        wtr = np.concatenate([wp, wn])
        wtr = wtr / wtr.mean()
        ytr = np.concatenate([np.ones(len(tr_pos)), np.zeros(len(tr_neg))]).astype(np.float32)
        pairs_tr = tr_pos + tr_neg

        # ---------- 评测对展开 ----------
        ev_pos, ev_neg = [], []
        ev_pw, ev_nw = [], []
        for p in fold["eval_pos"]:
            w0 = p["weight"] / nv
            for t in dv_tasks:
                ev_pos.append((p["anchor"], p["partner"], t)); ev_pw.append(w0)
        for p in fold["eval_neg"]:
            w0 = p["weight"] / nv
            for t in dv_tasks:
                ev_neg.append((p["anchor"], p["partner"], t)); ev_nw.append(w0)
        ev_pairs = ev_pos + ev_neg
        yev = np.concatenate([np.ones(len(ev_pos)), np.zeros(len(ev_neg))]).astype(np.int64)
        wev = np.concatenate([ev_pw, ev_nw]).astype(np.float64)
        ev_taskcol = np.array([tidx_dv[t] for a, b, t in ev_pairs])

        # ---------- 训练特征 ----------
        # 索引：pt_idx (n_pair, 2) 成员行号 + task 列号（tr）
        pt_midx = np.array([[rowidx[a], rowidx[b]] for a, b, t in pairs_tr])
        pt_tidx = np.array([tidx_tr[t] for a, b, t in pairs_tr])
        Xq_tr = pair_feats(Um, pt_midx, pt_tidx, qsym)
        # 评测特征
        pe_midx = np.array([[rowidx[a], rowidx[b]] for a, b, t in ev_pairs])
        pe_tidx = np.array([tidx_dv[t] for a, b, t in ev_pairs])
        Xq_ev = pair_feats(Udv, pe_midx, pe_tidx, qsym)
        # fit-only scaler
        mu = Xq_tr.mean(0); sd = Xq_tr.std(0); sd[sd < 1e-8] = 1.0
        Xq_tr_s = ((Xq_tr - mu) / sd).astype(np.float32)
        Xq_ev_s = ((Xq_ev - mu) / sd).astype(np.float32)

        # ---------- 读出 1：endpoint-only sanity（trivial 分数） ----------
        # 两种与标签无关的端点分数：sizeB 与 ‖u‖ 差（不含成员身份文本）
        def unorm_feats(Umat, midx, tidx):
            return np.linalg.norm(Umat[midx[:, 0], tidx], axis=1) - np.linalg.norm(Umat[midx[:, 1], tidx], axis=1)
        sizeB = {m: float(A["sizeB"][A["MODEL_I"][m] * A["nT"] * 2] or 0) for m in all_members}
        s_anchor_size = np.array([sizeB[a] for a, b, t in ev_pairs])
        s_partner_size = np.array([sizeB[b] for a, b, t in ev_pairs])
        s_unorm_diff = unorm_feats(Udv, pe_midx, pe_tidx)

        # ---------- 读出 2：independent-source composition ----------
        from sklearn.linear_model import LogisticRegression
        unit_members_tr = [m for s in plan["series"] for m in M[s]]
        unit_series_tr = np.array([plan["series"].index(s) for s in plan["series"] for m in M[s] for _ in tr_tasks])
        Xunit_tr = np.stack([Um[rowidx[m], k] for m in unit_members_tr for k in range(nt)]).astype(np.float64)
        clf_ser = LogisticRegression(max_iter=2000, C=1.0).fit(Xunit_tr, unit_series_tr)
        Xunit_ev = np.stack([Udv[rowidx[m], k] for m in all_members for k in range(nv)])
        P_ev = clf_ser.predict_proba(Xunit_ev)  # (11*nv, 3)
        p_by = {(m, k): P_ev[rowidx[m] * nv + k] for m in all_members for k in range(nv)}
        s_compose_ev = np.array([float(p_by[(a, k)] @ p_by[(b, k)]) for a, b, t in ev_pairs
                                 for k in [tidx_dv[t]]])
        # 训练 compose 分数（供 residual 冻结 baseline 用）
        P_tr = clf_ser.predict_proba(Xunit_tr)
        p_bytr = {}
        for mi_, m in enumerate(unit_members_tr):
            for k in range(nt):
                p_bytr[(m, k)] = P_tr[mi_ * nt + k]
        s_compose_tr = np.array([float(p_bytr[(a, k)] @ p_bytr[(b, k)]) for a, b, t in pairs_tr
                                 for k in [tidx_tr[t]]])

        # ---------- 读出 3：symmetric q 加权 LR ----------
        s_q = sgd_weighted(Xq_tr_s, ytr, wtr, Xq_ev_s)

        # ---------- 读出 4：strong P0（partial：style+meta+sizelen） ----------
        # (a) 系列分类器 on surface（unit 级）
        Sf_all_tr = surf_mat(unit_members_tr, tr_tasks)  # (nM, nt, F)
        Sf_tr = Sf_all_tr.reshape(-1, Sf_all_tr.shape[-1]).astype(np.float64)
        clf_sur = LogisticRegression(max_iter=2000, C=1.0).fit(Sf_tr, unit_series_tr)
        Sf_all_ev = surf_mat(all_members, dv_tasks)
        Sf_ev = Sf_all_ev.reshape(-1, Sf_all_ev.shape[-1]).astype(np.float64)
        Ps_ev = clf_sur.predict_proba(Sf_ev)
        ps_by = {(m, k): Ps_ev[rowidx[m] * nv + k] for m in all_members for k in range(nv)}
        s_p0_compose = np.array([float(ps_by[(a, k)] @ ps_by[(b, k)]) for a, b, t in ev_pairs for k in [tidx_dv[t]]])
        # (b) P0 对称对特征 LR
        P0tr = np.stack([surf_mat([m], tr_tasks)[0] for m in all_members])  # (11, nt, F)
        P0dv = np.stack([surf_mat([m], dv_tasks)[0] for m in all_members])
        def p0pair(P0m, midx, tidx):
            fi = P0m[midx[:, 0], tidx]; fj = P0m[midx[:, 1], tidx]
            return np.hstack([np.abs(fi - fj), fi + fj]).astype(np.float32)
        Xp0_tr = p0pair(P0tr, pt_midx, pt_tidx); Xp0_ev = p0pair(P0dv, pe_midx, pe_tidx)
        mu0 = Xp0_tr.mean(0); sd0 = Xp0_tr.std(0); sd0[sd0 < 1e-8] = 1.0
        s_p0_pair = sgd_weighted((Xp0_tr - mu0) / sd0, ytr, wtr, (Xp0_ev - mu0) / sd0)

        # ---------- 读出 5：relation residual（冻结 compose + γ r(q_sym)） ----------
        import torch
        import torch.nn as nn
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        eps = 1e-6

        def run_residual(seed, labels):
            torch.manual_seed(seed)
            base_tr = np.log(np.clip(s_compose_tr, eps, 1 - eps) / (1 - np.clip(s_compose_tr, eps, 1 - eps)))
            base_ev = np.log(np.clip(s_compose_ev, eps, 1 - eps) / (1 - np.clip(s_compose_ev, eps, 1 - eps)))
            r_net = nn.Sequential(nn.Linear(3072, 64), nn.ReLU(), nn.Linear(64, 1)).to(device)
            # 输出零初始化（E33 教训）：r≡0 初始 ⇒ 残差贡献恒零（=compose 基线）且梯度全活；
            # 等价实现指导的“γ0=0”意图（字面 γ=0 且无输出零初始化会构成双死门）。
            nn.init.zeros_(r_net[-1].weight); nn.init.zeros_(r_net[-1].bias)
            gamma = torch.ones(1, requires_grad=True, device=device)
            opt = torch.optim.Adam(list(r_net.parameters()) + [gamma], lr=1e-3)
            Xt = torch.tensor(Xq_tr_s, device=device)
            bt = torch.tensor(base_tr, dtype=torch.float32, device=device)
            yt = torch.tensor(labels, device=device)
            wt = torch.tensor(wtr, dtype=torch.float32, device=device)
            rng = np.random.default_rng(seed)
            for ep in range(8):
                perm = rng.permutation(len(Xt))
                for i in range(0, len(perm), 4096):
                    sel = torch.tensor(perm[i:i + 4096], device=device)
                    r_out = torch.tanh(r_net(Xt[sel]).squeeze(-1))
                    logits = bt[sel] + gamma * r_out
                    loss = (torch.nn.functional.binary_cross_entropy_with_logits(logits, yt[sel], reduction="none") * wt[sel]).mean()
                    opt.zero_grad(); loss.backward(); opt.step()
            with torch.inference_mode():
                Xe = torch.tensor(Xq_ev_s, device=device)
                r_ev = torch.tanh(r_net(Xe).squeeze(-1)).cpu().numpy()
            return base_ev + float(gamma.detach().cpu()) * r_ev, float(gamma.detach().cpu())

        resid_runs = [run_residual(s, ytr) for s in SEEDS]
        s_resid = np.mean([r[0] for r in resid_runs], axis=0)
        gammas = [r[1] for r in resid_runs]

        # ---------- 指标 ----------
        reads = {"endpoint_anchor_size": s_anchor_size, "endpoint_partner_size": s_partner_size,
                 "endpoint_unorm_diff": s_unorm_diff, "compose": s_compose_ev,
                 "q_sym": s_q, "P0_compose": s_p0_compose, "P0_pair": s_p0_pair, "residual": s_resid}
        m = {}
        for k, s in reads.items():
            m[k] = {"weighted_auc": float(weighted_auc(yev, s, wev)),
                    "weighted_ap": float(weighted_ap(yev, s, wev)),
                    "unweighted_auc": unweighted_auc(yev, s)}
        np.savez_compressed(OUT / "local" / f"eval_scores_fold{fid}.npz",
                            y=yev, w=wev, taskcol=ev_taskcol,
                            **{f"s_{k}": np.asarray(v, dtype=np.float64) for k, v in reads.items()})
        metrics_all[fid] = m
        m["residual_gammas"] = gammas
        log(f"[A] fold{fid} ({fold['heldout']['CodeLlama-Instruct'][:20]}...) "
            f"endpoint(anchor/partner)={m['endpoint_anchor_size']['weighted_auc']:.3f}/{m['endpoint_partner_size']['weighted_auc']:.3f} "
            f"compose={m['compose']['weighted_auc']:.4f} q={m['q_sym']['weighted_auc']:.4f} "
            f"P0c={m['P0_compose']['weighted_auc']:.4f} P0p={m['P0_pair']['weighted_auc']:.4f} "
            f"resid={m['residual']['weighted_auc']:.4f} gamma={gammas} ({time.time()-tf0:.0f}s)")
        # ---------- 置换（代表折，20 seeds；pair 级置换：同 unordered pair 的所有 task 行共标签） ----------
        if not args.skip_perm and str(fid) == args.perm_folds.split(",")[0]:
            perm_records = []
            n_perm = 20
            keys = [(a, b) if a <= b else (b, a) for a, b, t in pairs_tr]
            uniq = sorted(set(keys))
            t_obs_q = weighted_auc(yev, s_q, wev)
            t_obs_r = weighted_auc(yev, s_resid, wev)
            for bperm in range(n_perm):
                rngp = np.random.default_rng(SEED + 200 + bperm)
                bits = {k: int(rngp.integers(0, 2)) for k in uniq}
                y_perm = np.array([bits[k] for k in keys], dtype=np.float32)
                s_qp = sgd_weighted(Xq_tr_s, y_perm, wtr, Xq_ev_s)
                pr = run_residual(SEEDS[0], y_perm)
                s_rp = pr[0]
                perm_records.append({"seed": SEED + 200 + bperm,
                                     "q_sym_auc": float(weighted_auc(yev, s_qp, wev)),
                                     "residual_auc": float(weighted_auc(yev, s_rp, wev))})
            tq = np.array([r["q_sym_auc"] for r in perm_records])
            trr = np.array([r["residual_auc"] for r in perm_records])
            perm_out = {"fold": fid, "n_perm": n_perm,
                        "T_obs_q_sym": t_obs_q, "T_obs_residual": t_obs_r,
                        "q_sym": {"values": [float(x) for x in tq], "mean": float(tq.mean()),
                                  "p_value": float((1 + int((tq >= t_obs_q).sum())) / (n_perm + 1))},
                        "residual": {"values": [float(x) for x in trr], "mean": float(trr.mean()),
                                     "p_value": float((1 + int((trr >= t_obs_r).sum())) / (n_perm + 1))}}
            (OUT / "metrics_permutation.json").write_text(json.dumps(perm_out, ensure_ascii=False, indent=1), encoding="utf-8")
            log(f"[A] permutation fold{fid}: q p={perm_out['q_sym']['p_value']:.3f} "
                f"resid p={perm_out['residual']['p_value']:.3f}")
        (OUT / "metrics_partial.json").write_text(json.dumps(
            {"schema": "f2_balanced_partial_v1", "folds_done": sorted(metrics_all.keys()), "metrics": metrics_all},
            ensure_ascii=False, indent=1), encoding="utf-8")

    (OUT / "metrics_partial.json").write_text(json.dumps(
        {"schema": "f2_balanced_partial_v1", "folds_done": sorted(metrics_all.keys()), "metrics": metrics_all},
        ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "logs" / "run_a.log").write_text("\n".join(LOG) + "\n", encoding="utf-8")
    log(f"[A] folds done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
