"""H3 registered（train/dev；F2 v2 协议通过后执行；不读 test、不产出 test 主结果）。

四配置（同架构、同数据、参数量匹配）：
  c0: L_D                                    （协议检测：单元 [S;D] vs [S;-D] 翻转对）
  c1: L_F                                    （关系归因：v2 同 h 协议对，双向平均）
  c2: L_D + L_F
  c3: L_D + L_F + lam*relu(cos(grad_D, grad_F) - m)   （lambda=0.1, m=0.3）
M-HO：11 折（逐 member 留出）；S-HO：3 折（逐 series 留出，低功效）。
评测：det AUROC（held-out member 的 dev 单元翻转对）；rel AUROC（v2 协议 pos/neg，双向平均；
      P0 参照引用 f2 v2 值，不重训）；梯度余弦记录；Pareto=四配置 (det, rel) 点。

输出：artifacts/h3_train_dev_registered_2026-10-08/（H3 部分）
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import core_r0_eval as ev  # noqa: E402
import variant_transfer_stage1_execute as s1  # noqa: E402

OUT = ROOT / "d-det/artifacts/h3_train_dev_registered_2026-10-08"
SEED = 20261008
LOG: list[str] = []
CONFIGS = [("c0_L_D", 1.0, 0.0, 0.0), ("c1_L_F", 0.0, 1.0, 0.0),
           ("c2_L_D+L_F", 1.0, 1.0, 0.0), ("c3_L_D+L_F+cos", 1.0, 1.0, 0.1)]


def log(m):
    print(m, flush=True)
    LOG.append(m)


def build_u(A):
    H = A["emb_base"]
    U = {}
    for ss in s1.SERIES:
        for m in s1.SERIES[ss]:
            mi = A["MODEL_I"][m]
            for t in A["tasks"]:
                ti = A["TASK_I"][t]
                r = mi * A["nT"] * 2 + ti * 2
                hc, hi = H[r], H[r + 1]
                U[(m, t)] = np.concatenate([(hc + hi) / 2.0, (hi - hc) / 2.0])
    return U


def partner_key(A, m, t):
    r = A["MODEL_I"][m] * A["nT"] * 2 + A["TASK_I"][t] * 2 + 1
    size = A["sizeB"][A["MODEL_I"][m] * A["nT"] * 2]
    st = A["style"][r].astype(np.float64)
    ls = float(np.log10(size)) if np.isfinite(size) and size > 0 else 0.0
    return np.array([ls, st[0] / 500.0, float(np.mean(st)) / 50.0, float(A["sizelen"][r, 0]) / 5.0])


def match_partners(A, h, seen, others, t):
    P = {m: partner_key(A, m, t) for m in seen + others}
    dists = sorted(((float(np.linalg.norm(P[a] - P[b])), a, b) for a in seen for b in others), key=lambda x: x[0])
    ua, ub, pairs = set(), set(), []
    for d, a, b in dists:
        if a in ua or b in ub:
            continue
        pairs.append((a, b)); ua.add(a); ub.add(b)
    return pairs


def keyf2(A, p):
    pi = A["sizeB"][A["MODEL_I"][p[0]] * A["nT"] * 2]
    pj = A["sizeB"][A["MODEL_I"][p[1]] * A["nT"] * 2]
    size_d = abs(np.log10(pi) - np.log10(pj)) if (pi and pj and np.isfinite(pi) and np.isfinite(pj)) else 99
    ri = A["MODEL_I"][p[0]] * A["nT"] * 2 + A["TASK_I"][p[2]] * 2 + 1
    rj = A["MODEL_I"][p[1]] * A["nT"] * 2 + A["TASK_I"][p[2]] * 2 + 1
    len_d = abs(float(A["style"][ri, 0]) - float(A["style"][rj, 0])) / 500.0
    st_d = float(np.mean(np.abs(A["style"][ri].astype(np.float64) - A["style"][rj].astype(np.float64)))) / 50.0
    return size_d * 3.0 + len_d + st_d * 0.1


def match_negs(A, pos, cand):
    pb, cb = defaultdict(list), defaultdict(list)
    for a, b, t in pos:
        pb[t].append((a, b, t))
    for a, b, t in cand:
        cb[t].append((a, b, t))
    neg = []
    for t, plist in pb.items():
        neg.extend(sorted(cb[t], key=lambda p: keyf2(A, p))[:len(plist)])
    return neg


def fold_data_mho(A, U, series, h, tr_tasks, dv_tasks):
    members = s1.SERIES[series]
    seen = [m for m in members if m != h]
    others = [m for ss in s1.SERIES if ss != series for m in s1.SERIES[ss]]
    fit_pos = [(seen[i], seen[j], t) for i in range(len(seen)) for j in range(i + 1, len(seen)) for t in tr_tasks]
    cand = [(others[i], others[j], t) for i in range(len(others)) for j in range(i + 1, len(others)) for t in tr_tasks]
    fit_neg = match_negs(A, fit_pos, cand)
    ev_pos, ev_neg = [], []
    for t in dv_tasks:
        for m_seen, m_other in match_partners(A, h, seen, others, t):
            ev_pos.append((h, m_seen, t)); ev_neg.append((h, m_other, t))
    return {"fit_pos": fit_pos, "fit_neg": fit_neg, "ev_pos": ev_pos, "ev_neg": ev_neg,
            "det_models": seen, "det_eval_model": h}


def fold_data_sho(A, U, hold, tr_tasks, dv_tasks):
    rest = [ss for ss in s1.SERIES if ss != hold]
    rest_members = [m for ss in rest for m in s1.SERIES[ss]]
    hold_members = s1.SERIES[hold]
    fit_pos = []
    for ss in rest:
        ms = s1.SERIES[ss]
        fit_pos += [(ms[i], ms[j], t) for i in range(len(ms)) for j in range(i + 1, len(ms)) for t in tr_tasks]
    cand = [(rest_members[i], rest_members[j], t) for i in range(len(rest_members)) for j in range(i + 1, len(rest_members))
            if s1.SERIES_OF[rest_members[i]] != s1.SERIES_OF[rest_members[j]] for t in tr_tasks]
    fit_neg = match_negs(A, fit_pos, cand)
    ev_pos = [(hold_members[i], hold_members[j], t) for i in range(len(hold_members)) for j in range(i + 1, len(hold_members))
              for t in dv_tasks]
    cand_ev = [(hm, m, t) for hm in hold_members for m in rest_members for t in dv_tasks]
    ev_neg = match_negs(A, ev_pos, cand_ev)
    return {"fit_pos": fit_pos, "fit_neg": fit_neg, "ev_pos": ev_pos, "ev_neg": ev_neg,
            "det_models": rest_members, "det_eval_model": None}


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
    U = build_u(A)
    log(f"[h3r] tasks train={len(tr_tasks)} dev={len(dv_tasks)}; u cells={len(U)}")

    from sklearn.metrics import roc_auc_score

    def unit_mat(models, tasks):
        X = np.stack([U[(m, t)] for m in models for t in tasks]).astype(np.float32)
        return X

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

    def train_cfg(data, wd, wf, lam, epochs=2, lr=2e-3, bs=2048, seed=SEED):
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
        cos_hist = []
        for ep in range(epochs):
            perm = rng.permutation(n_cells)
            for i in range(0, n_cells, bs):
                sel = torch.tensor(perm[i:i + bs], device=device)
                loss = None; loss_d = None; loss_f = None
                if wd > 0:
                    Xb = Xdet[sel]
                    Xp = torch.cat([Xb[:, :768], -Xb[:, 768:]], dim=1)
                    loss_d = torch.nn.functional.softplus(-(net.d(net.z(Xb)) - net.d(net.z(Xp)))).mean()
                    loss = wd * loss_d
                if wf > 0:
                    idx2 = torch.tensor(rng.integers(0, len(pairs_tr), size=min(bs, len(pairs_tr))), device=device)
                    za = UA_t[idx2]; zb = UB_t[idx2]; yb = y_tt[idx2]
                    loss_f = 0.5 * (torch.nn.functional.binary_cross_entropy_with_logits(net.f(net.z(za), net.z(zb)), yb)
                                    + torch.nn.functional.binary_cross_entropy_with_logits(net.f(net.z(zb), net.z(za)), yb))
                    loss = (loss + wf * loss_f) if loss is not None else (wf * loss_f)
                if lam > 0 and loss_d is not None and loss_f is not None:
                    try:
                        gd = torch.autograd.grad(wd * loss_d, net.enc.parameters(), retain_graph=True,
                                                 allow_unused=True, create_graph=True)
                        gf = torch.autograd.grad(wf * loss_f, net.enc.parameters(), retain_graph=True,
                                                 allow_unused=True, create_graph=True)
                        gdv = torch.cat([g.ravel() for g in gd if g is not None])
                        gfv = torch.cat([g.ravel() for g in gf if g is not None])
                        cos_t = torch.nn.functional.cosine_similarity(gdv, gfv, dim=0)
                        cos_hist.append(float(cos_t.detach()))
                        loss = loss + lam * torch.relu(cos_t - 0.3)
                    except RuntimeError:
                        pass
                opt.zero_grad(); loss.backward(); opt.step()
        return net, (float(np.mean(cos_hist)) if cos_hist else None)

    def eval_cfg(net, data, dv_tasks):
        net.eval()
        with torch.inference_mode():
            det_auc = None
            if data["det_eval_model"] is not None:
                Xh = torch.tensor(unit_mat([data["det_eval_model"]], dv_tasks), device=device)
                Xp = torch.cat([Xh[:, :768], -Xh[:, 768:]], dim=1)
                g = net.d(net.z(Xh)).cpu().numpy(); gp = net.d(net.z(Xp)).cpu().numpy()
                det_auc = float(roc_auc_score(np.r_[np.ones(len(g)), np.zeros(len(gp))], np.r_[g, gp]))

            def score_batch(pairs):
                ZA = torch.tensor(np.stack([U[(a, t)] for a, b, t in pairs]).astype(np.float32), device=device)
                ZB = torch.tensor(np.stack([U[(b, t)] for a, b, t in pairs]).astype(np.float32), device=device)
                s = 0.5 * (net.f(net.z(ZA), net.z(ZB)) + net.f(net.z(ZB), net.z(ZA)))
                return s.cpu().numpy()

            sp = score_batch(data["ev_pos"]); sn = score_batch(data["ev_neg"])
            rel_auc = float(roc_auc_score(np.r_[np.ones(len(sp)), np.zeros(len(sn))], np.r_[sp, sn]))
        net.train()
        return det_auc, rel_auc

    results = {"schema": "h3_registered_v1", "generated_utc": datetime.now(timezone.utc).isoformat(),
               "report_first_line": "registered train/dev（F2 v2 通过后执行；不读 test）",
               "m_ho": {}, "s_ho": {}, "configs": [c[0] for c in CONFIGS]}
    for series in s1.SERIES:
        for h in s1.SERIES[series]:
            data = fold_data_mho(A, U, series, h, tr_tasks, dv_tasks)
            fold_res = {}
            for name, wd, wf, lam in CONFIGS:
                net, cosm = train_cfg(data, wd, wf, lam)
                det, rel = eval_cfg(net, data, dv_tasks)
                fold_res[name] = {"det_auroc": det, "rel_auroc": rel, "grad_cos_mean": cosm}
                log(f"[h3r] MHO {series}::{h} {name}: det={det} rel={rel:.4f} cos={cosm}")
            results["m_ho"][f"{series}::{h}"] = fold_res
    for hold in s1.SERIES:
        data = fold_data_sho(A, U, hold, tr_tasks, dv_tasks)
        fold_res = {}
        for name, wd, wf, lam in CONFIGS:
            net, cosm = train_cfg(data, wd, wf, lam)
            det, rel = eval_cfg(net, data, dv_tasks)
            fold_res[name] = {"det_auroc": det, "rel_auroc": rel, "grad_cos_mean": cosm}
            log(f"[h3r] SHO {hold} {name}: rel={rel:.4f} cos={cosm}")
        results["s_ho"][hold] = fold_res

    # 汇总
    agg = {}
    for name in [c[0] for c in CONFIGS]:
        ds = [v[name]["det_auroc"] for v in results["m_ho"].values() if v[name]["det_auroc"] is not None]
        rs = [v[name]["rel_auroc"] for v in results["m_ho"].values()]
        cs = [v[name]["grad_cos_mean"] for v in results["m_ho"].values() if v[name]["grad_cos_mean"] is not None]
        agg[name] = {"det_auroc_mean": float(np.mean(ds)) if ds else None,
                     "rel_auroc_mean": float(np.mean(rs)),
                     "rel_auroc_min": float(np.min(rs)), "rel_auroc_max": float(np.max(rs)),
                     "grad_cos_mean": float(np.mean(cs)) if cs else None}
    agg_sho = {name: {"rel_auroc_mean": float(np.mean([v[name]["rel_auroc"] for v in results["s_ho"].values()]))}
               for name in agg}
    results["aggregate"] = agg
    results["aggregate_s_ho"] = agg_sho
    results["pareto_points"] = {name: {"det": agg[name]["det_auroc_mean"], "rel": agg[name]["rel_auroc_mean"]} for name in agg}
    (OUT / "metrics_h3_registered.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")

    L = ["# H3 registered（train/dev；不读 test）", "",
         "| 配置 | det AUROC（mean） | rel AUROC（mean [min,max]） | grad cos |", "|---|---|---|---|"]
    for name in agg:
        a = agg[name]
        det_s = "—" if a["det_auroc_mean"] is None else f"{a['det_auroc_mean']:.4f}"
        cos_s = "—" if a["grad_cos_mean"] is None else f"{a['grad_cos_mean']:.3f}"
        L.append(f"| {name} | {det_s} | {a['rel_auroc_mean']:.4f} [{a['rel_auroc_min']:.4f},{a['rel_auroc_max']:.4f}] | {cos_s} |")
    L += ["", "## S-HO 外推（3 折，低功效）", ""]
    for name in agg_sho:
        L.append(f"- {name}: rel AUROC mean={agg_sho[name]['rel_auroc_mean']:.4f}")
    L += ["", "> registered train/dev；det=held-out member 的协议检测；rel=v2 同 h 协议（partner 身份警告见 f2 fix 报告）；",
          "> Pareto 点见 metrics；P0 参照引用 f2 v2（不重训）。"]
    (OUT / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (OUT / "logs" / "h3r_run.log").write_text("\n".join(LOG) + "\n", encoding="utf-8")
    log(f"[h3r] done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
