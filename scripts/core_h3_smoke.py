"""H3 代码 smoke（仅工程验证；不产出论文主结果）。

四配置（同一模型与数据，只改损失组合；参数量天然匹配）：
  c0: L_D                                  （协议检测：instruct vs complete 单元方向）
  c1: L_F                                  （关系归因：同 observed series 对 vs 跨系列对）
  c2: L_D + L_F
  c3: L_D + L_F + lam * max(0, cos(gD_grad, gF_grad) - m)
记录：loss 轨迹、检测 AUROC（dev tasks）、held-out member 关系 AUROC/BA（1 折）、
      共享编码器上 ∇L_D 与 ∇L_F 的梯度余弦、运行时间。

⚠ smoke_only=true；数据子集=300 train tasks × 11 members；2 epoch；
  一切数字仅供管线验证，严禁作为论文主结果引用。不读 test。

输出：artifacts/h3_smoke_only_2026-10-08/
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

OUT = ROOT / "d-det/artifacts/h3_smoke_only_2026-10-08"
SEED = 20261008
N_TASKS = 300
HOLDOUT_MEMBER = "deepseek-ai--deepseek-coder-33b-instruct"
LOG: list[str] = []


def log(m):
    print(m, flush=True)
    LOG.append(m)


def unit_u(A, members, tasks):
    """member-major 单元 u 矩阵 + 行索引（[S;D] 1536d）。"""
    H = A["emb_base"]
    rows = []
    for m in members:
        mi = A["MODEL_I"][m]
        ti = np.array([A["TASK_I"][t] for t in tasks])
        rows.append(mi * A["nT"] * 2 + ti * 2)
    Rc = np.concatenate(rows)
    Ri = Rc + 1
    hc, hi = H[Rc], H[Ri]
    S = (hc + hi) / 2.0; D = (hi - hc) / 2.0
    return np.hstack([S, D])


def main():
    import torch
    import torch.nn as nn
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "logs").mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)
    A = ev.load_assets()
    splits = {t: A["splits"][A["TASK_I"][t] * 2] for t in A["tasks"]}
    tr_all = [t for t in A["tasks"] if splits[t] == "train"]
    dv_all = [t for t in A["tasks"] if splits[t] == "dev"]
    rng = np.random.default_rng(SEED)
    tr_tasks = sorted(rng.choice(tr_all, size=min(N_TASKS, len(tr_all)), replace=False).tolist(),
                      key=lambda t: A["TASK_I"][t])
    dv_tasks = dv_all[:100]
    members = [m for ss in s1.SERIES for m in s1.SERIES[ss]]
    fit_members = [m for m in members if m != HOLDOUT_MEMBER]
    log(f"[h3] smoke subset: train={len(tr_tasks)} tasks, dev={len(dv_tasks)}; members={len(fit_members)}(+1 holdout)")

    Xtr = unit_u(A, fit_members, tr_tasks).astype(np.float32)
    Xdv = unit_u(A, fit_members + [HOLDOUT_MEMBER], dv_tasks).astype(np.float32)
    n_fit = len(fit_members) * len(tr_tasks)
    # 单元标签：member 序号（用于 pair 构造）；检测对 = 同一单元的 [S;D] vs [S;-D]（模式翻转）
    m_index = np.repeat(np.arange(len(fit_members)), len(tr_tasks))

    series_id = {m: i for i, m in enumerate(s1.SERIES)}
    mem_series = np.array([series_id[s1.SERIES_OF[m]] for m in fit_members])

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = nn.Sequential(nn.Linear(1536, 256), nn.ReLU())
            self.head_d = nn.Linear(256, 1)
            self.head_f = nn.Sequential(nn.Linear(256 * 4, 64), nn.ReLU(), nn.Linear(64, 1))
        def z(self, x):
            return self.enc(x)
        def d_score(self, z):
            return self.head_d(z).squeeze(-1)
        def f_score(self, za, zb):
            q = torch.cat([za, zb, (za - zb).abs(), za * zb], -1)
            return self.head_f(q).squeeze(-1)

    def eval_reader(net):
        net.eval()
        from sklearn.metrics import roc_auc_score
        with torch.inference_mode():
            X = torch.tensor(Xdv, device=device)
            Xp = Xdv.copy(); Xp[:, 768:] *= -1  # 模式翻转（D -> -D）
            g = net.d_score(net.z(X)).cpu().numpy()
            gp = net.d_score(net.z(torch.tensor(Xp, device=device))).cpu().numpy()
            det = float(roc_auc_score(np.r_[np.ones(len(g)), np.zeros(len(gp))], np.r_[g, gp]))
            z = net.z(X)
            nt = len(dv_tasks)
            n_m = len(fit_members) + 1
            z = z.reshape(n_m, nt, -1)
            Hm = s1.SERIES_OF[HOLDOUT_MEMBER]
            seen = [m for m in s1.SERIES[Hm] if m != HOLDOUT_MEMBER]
            h_idx = len(fit_members)  # holdout 在 Xdv 的最后位置
            pos_s, neg_s = [], []
            for m in seen:
                mi_ = fit_members.index(m)
                for ti_ in range(nt):
                    pos_s.append(float(net.f_score(z[h_idx, ti_].unsqueeze(0), z[mi_, ti_].unsqueeze(0))))
            for m in members:
                if s1.SERIES_OF[m] != Hm:
                    mi_ = fit_members.index(m)
                    for ti_ in range(nt):
                        neg_s.append(float(net.f_score(z[h_idx, ti_].unsqueeze(0), z[mi_, ti_].unsqueeze(0))))
            rel_auc = float(roc_auc_score(np.r_[np.ones(len(pos_s)), np.zeros(len(neg_s))], np.r_[pos_s, neg_s]))
        net.train()
        return det, rel_auc

    def run_config(name, wd, wf, lambda_cos=0.0, m_cos=0.3, epochs=3, lr=2e-3):
        torch.manual_seed(SEED)
        net = Model().to(device)
        opt = torch.optim.Adam(net.parameters(), lr=lr)
        rng_ = np.random.default_rng(SEED)
        log_hist = []
        cos_hist = []
        X = torch.tensor(Xtr, device=device)
        n_t = len(tr_tasks)
        n_m = len(fit_members)
        for ep in range(epochs):
            perm = rng_.permutation(n_fit)
            bs = 512
            tot_d, tot_f, nb = 0.0, 0.0, 0
            for i in range(0, len(perm), bs):
                sel = perm[i:i + bs]
                idx = torch.tensor(sel, device=device)
                Xb = X[idx]
                Xp = torch.cat([Xb[:, :768], -Xb[:, 768:]], dim=1)  # 模式翻转对（out-of-place）
                loss_d = torch.nn.functional.softplus(-(net.d_score(net.z(Xb)) - net.d_score(net.z(Xp)))).mean()
                # 关系 batch：同 task 内抽两个不同成员（同系列=1 正 / 跨系列=0 负）
                t_sel = rng_.integers(0, n_t, size=len(sel))
                a_m = rng_.integers(0, n_m, size=len(sel))
                b_m = rng_.integers(0, n_m, size=len(sel))
                ok = a_m != b_m
                t_sel, a_m, b_m = t_sel[ok], a_m[ok], b_m[ok]
                if len(a_m) < 8:
                    loss = wd * loss_d
                    opt.zero_grad(); loss.backward(); opt.step()
                    tot_d += float(loss_d); nb += 1
                    continue
                a_idx = a_m * n_t + t_sel
                b_idx = b_m * n_t + t_sel
                y_rel = (mem_series[a_m] == mem_series[b_m]).astype(np.float32)
                za = net.z(X[torch.tensor(a_idx, device=device)])
                zb = net.z(X[torch.tensor(b_idx, device=device)])
                loss_f = torch.nn.functional.binary_cross_entropy_with_logits(
                    net.f_score(za, zb), torch.tensor(y_rel, device=device))
                loss = wd * loss_d + wf * loss_f
                if lambda_cos > 0 and wd > 0 and wf > 0:
                    # 梯度余弦（共享编码器；create_graph 供 cos 惩罚回传）
                    try:
                        gd = torch.autograd.grad(wd * loss_d, net.enc.parameters(),
                                                 retain_graph=True, allow_unused=True, create_graph=True)
                        gf = torch.autograd.grad(wf * loss_f, net.enc.parameters(),
                                                 retain_graph=True, allow_unused=True, create_graph=True)
                        gd_v = torch.cat([g.ravel() for g in gd if g is not None])
                        gf_v = torch.cat([g.ravel() for g in gf if g is not None])
                        cos_t = torch.nn.functional.cosine_similarity(gd_v, gf_v, dim=0)
                        cos_hist.append(float(cos_t.detach()))
                        loss = loss + lambda_cos * torch.relu(cos_t - m_cos)
                    except RuntimeError as e:
                        cos_hist.append(float("nan"))
                        log(f"    [warn] cos penalty skipped: {e}")
                opt.zero_grad(); loss.backward(); opt.step()
                tot_d += float(loss_d); tot_f += float(loss_f); nb += 1
            log_hist.append({"epoch": ep + 1, "loss_d": tot_d / max(1, nb), "loss_f": tot_f / max(1, nb)})
        det, rel = eval_reader(net)
        n_params = sum(p.numel() for p in net.parameters())
        out = {"config": name, "loss_weights": {"d": wd, "f": wf},
               "lambda_cos": lambda_cos, "m_cos": m_cos,
               "loss_hist": log_hist, "grad_cos_mean": float(np.mean(cos_hist)) if cos_hist else None,
               "det_auroc_dev_smoke": det, "holdout_member_rel_auroc_smoke": rel,
               "n_params": n_params}
        log(f"[h3] {name}: det={det:.4f} rel={rel:.4f} cos={out['grad_cos_mean']} params={n_params}")
        return out

    configs = []
    configs.append(run_config("c0_L_D", wd=1.0, wf=0.0))
    configs.append(run_config("c1_L_F", wd=0.0, wf=1.0))
    configs.append(run_config("c2_L_D+L_F", wd=1.0, wf=1.0))
    configs.append(run_config("c3_L_D+L_F+cos", wd=1.0, wf=1.0, lambda_cos=0.1, m_cos=0.3))

    params_ok = len(set(c["n_params"] for c in configs)) == 1
    results = {"schema": "h3_smoke_v1", "smoke_only": True, "paper_main_result": False,
               "generated_utc": datetime.now(timezone.utc).isoformat(),
               "report_first_line": "smoke only — 工程验证（严禁作主结果引用）",
               "subset": {"train_tasks": len(tr_tasks), "dev_tasks": len(dv_tasks),
                          "members_fit": len(fit_members), "holdout_member": HOLDOUT_MEMBER},
               "switches": {"training_allowed": True, "generation_allowed": False,
                            "test_read_allowed": False, "old_test_reuse": False},
               "configs": configs, "param_count_equal_all_configs": bool(params_ok),
               "runtime_seconds": time.time() - t0}
    (OUT / "metrics_h3_smoke.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")

    L = ["# H3 代码 smoke（smoke only；不产出论文主结果）", "",
         f"- 子集：{len(tr_tasks)} train tasks × {len(fit_members)} members；holdout member={HOLDOUT_MEMBER}",
         f"- 参数量一致性（4 配置）：{'PASS' if params_ok else 'FAIL'}", "",
         "| 配置 | det AUROC(smoke) | holdout rel AUROC(smoke) | grad cos |", "|---|---|---|---|"]
    for c in configs:
        gc = c["grad_cos_mean"]
        L.append(f"| {c['config']} | {c['det_auroc_dev_smoke']:.4f} | {c['holdout_member_rel_auroc_smoke']:.4f} | "
                 f"{'—' if gc is None else f'{gc:.3f}'} |")
    L += ["", "> smoke_only=true；数据为固定种子子集；一切数字仅供管线验证；H3 主实验须待 member-heldout 归因闸门通过。"]
    (OUT / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (OUT / "logs" / "h3_smoke.log").write_text("\n".join(LOG) + "\n", encoding="utf-8")
    log(f"[h3] done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
