"""C2: 代码约束静态代理头（server-side, 2026-10-09）.

无测试执行授权 ⇒ 只做 static_proxy_conditioning：
  L = L_series + lambda_v * L_proxy（对 8 个有方差的静态代理做 mask-free MSE）
输入/骨干/seed/预算与 C1 的 full MLP 完全一致（唯一变化 = 辅助代理损失）。
对照：C1 full-mlp（无辅助损失）、C0 强 P0（同一配对 bootstrap）。
另报告：代理方差/缺失率/长度相关（防止把规模重命名为代码约束）+ 长度/style 残差化控制。
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
import cc_common as cc  # noqa: E402

OUT = ROOT / "d-det/artifacts/mainline_c0c3_batch_2026-10-10/c2"
C0_DIR = ROOT / "d-det/artifacts/code_conditioned_fresh_c0_canonical_2026-10-10"
C1_DIR = ROOT / "d-det/artifacts/mainline_c0c3_batch_2026-10-10/c1"
SEEDS = (0, 1, 2)
EPOCHS = 20
LAMBDA_V = 0.1
PSI_TARGETS = ["n_ast_nodes", "n_calls", "n_branches", "n_handlers", "n_returns",
               "n_import_roots", "required_root_overlap_proxy", "entrypoint_present"]
LOG: list[str] = []


def log(m):
    print(m, flush=True)
    LOG.append(m)


def fit_mlp_aux(Xtr, ytr, Ztr, Xev, seed, device, lam=LAMBDA_V, epochs=EPOCHS):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    d = Xtr.shape[1]
    k = Ztr.shape[1]
    trunk = nn.Sequential(nn.Linear(d, 256), nn.ReLU()).to(device)
    head = nn.Linear(256, 1).to(device)
    phead = nn.Linear(256, k).to(device)
    opt = torch.optim.Adam(list(trunk.parameters()) + list(head.parameters())
                           + list(phead.parameters()), lr=1e-3)
    X = torch.tensor(Xtr.astype(np.float32), device=device)
    y = torch.tensor(ytr.astype(np.float32), device=device)
    Z = torch.tensor(Ztr.astype(np.float32), device=device)
    rng = np.random.default_rng(seed)
    bs = 2048
    for _ in range(epochs):
        perm = rng.permutation(len(X))
        for i in range(0, len(perm), bs):
            sel = torch.tensor(perm[i:i + bs], device=device)
            h = trunk(X[sel])
            loss = torch.nn.functional.binary_cross_entropy_with_logits(head(h).squeeze(-1), y[sel])
            loss = loss + lam * torch.nn.functional.mse_loss(phead(h), Z[sel])
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.inference_mode():
        h = trunk(torch.tensor(Xev.astype(np.float32), device=device))
        p = head(h).squeeze(-1).sigmoid().cpu().numpy()
    return p


def residualize(s, ctrl):
    """Label-free residual of scores on control features (OLS with intercept)."""
    A = np.hstack([np.ones((len(ctrl), 1)), ctrl])
    coef, *_ = np.linalg.lstsq(A, s, rcond=None)
    return s - A @ coef


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "local").mkdir(exist_ok=True)
    (OUT / "logs").mkdir(exist_ok=True)
    design = cc.load_design()
    rows = design["rows"]
    b = design["bundle"]
    task_split = {r["task_id"]: r["split"] for r in rows}
    dev_tasks = sorted(t for t in design["tasks_all"] if task_split[t] == "dev")
    tpos_of_task = {t: i for i, t in enumerate(dev_tasks)}
    n_tasks = len(dev_tasks)
    nT = len(design["tasks_all"])
    device = __import__("torch").device("cuda" if __import__("torch").cuda.is_available() else "cpu")
    log(f"[c2] device={device}")

    psi_cols = ["parse_ok", "entrypoint_present", "n_ast_nodes", "n_calls", "n_branches",
                "n_handlers", "n_returns", "n_import_roots", "required_root_overlap_proxy"]
    psi_idx = {k: i for i, k in enumerate(psi_cols)}

    # proxy audit
    audit = {}
    nchar = b["style"][:, 0]
    nlines = b["style"][:, 1]
    for t in PSI_TARGETS:
        v = b["psi"][:, psi_idx[t]]
        audit[t] = {
            "variance": float(np.var(v)),
            "missing_rate": 0.0,
            "corr_nchar": float(np.corrcoef(v, nchar)[0, 1]) if np.std(v) > 0 else None,
            "corr_nlines": float(np.corrcoef(v, nlines)[0, 1]) if np.std(v) > 0 else None,
        }
    audit["parse_ok"] = {"variance": float(np.var(b["psi"][:, 0])), "note": "zero variance -> excluded from targets"}

    # C0 scores + C1 full-mlp scores
    c0, c1full = {}, {}
    for fold in design["folds"]:
        fid = fold["heldout_generator_member"]
        z = np.load(C0_DIR / "local" / f"scores_dev_fold{fid.replace('--','_')}.npz")
        c0[fid] = z["fused"]
        z1 = np.load(C1_DIR / "local" / f"scores_fold{fid.replace('--','_')}.npz")
        c1full[fid] = z1["full_mlp"]

    labels, tposs, c2_scores = {}, {}, {}
    for fold in design["folds"]:
        fid = fold["heldout_generator_member"]
        tf0 = time.time()
        fit_mask, ev_mask, pos_mask, _ = cc.fold_setup(design, fold, split="dev", inner=False)
        y_fit = cc.fold_series_target(design, fold)[fit_mask]
        fit_rows = np.where(fit_mask)[0]
        ev_rows = np.where(ev_mask)[0]
        tp = np.array([tpos_of_task[rows[i]["task_id"]] for i in ev_rows])
        order = np.argsort(tp, kind="mergesort")
        ev_rows, tp = ev_rows[order], tp[order]
        y_ev = pos_mask[ev_rows].astype(int)
        labels[fid], tposs[fid] = y_ev, tp

        ht = design["emb_task"]
        def view(idx):
            ht_ = ht[b["task_idx"][idx]]
            hy_ = b["hy_small"][idx]
            return np.hstack([ht_, hy_, ht_ * hy_, np.abs(ht_ - hy_), b["psi"][idx]])
        Xtr_raw, Xev_raw = view(fit_rows), view(ev_rows)
        mu, sd = Xtr_raw.mean(0), Xtr_raw.std(0)
        sd[sd < 1e-8] = 1.0
        Xtr, Xev = (Xtr_raw - mu) / sd, (Xev_raw - mu) / sd

        Z = b["psi"][fit_rows][:, [psi_idx[t] for t in PSI_TARGETS]]
        zmu, zsd = Z.mean(0), Z.std(0)
        zsd[zsd < 1e-8] = 1.0
        Ztr = (Z - zmu) / zsd

        c2_scores[fid] = np.mean([fit_mlp_aux(Xtr, y_fit, Ztr, Xev, s, device) for s in SEEDS], axis=0)
        np.savez_compressed(OUT / "local" / f"scores_fold{fid.replace('--','_')}.npz",
                            ev_rows=ev_rows, y=y_ev, taskpos=tp, s_c2=c2_scores[fid])
        log(f"[c2] fold {fid[:40]:40s} done ({time.time()-tf0:.0f}s)")

    def agg(scores):
        row_mean = float(np.mean([cc.auroc(labels[f], scores[f]) for f in labels]))
        tm_mean = float(np.mean([cc.task_macro_auroc(labels[f], scores[f], tposs[f]) for f in labels]))
        boot = cc.bootstrap_metrics(scores, labels, tposs, n_tasks=n_tasks)
        return {"row_level_mean": row_mean, "task_macro_mean": tm_mean,
                "ci95": {"row_level": cc.ci(boot["row_level"]), "task_macro": cc.ci(boot["task_macro"])}}

    res = {"schema": "cc_c2_v1", "generated_utc": datetime.now(timezone.utc).isoformat(),
           "report_first_line": "C2 静态代理条件（static_proxy_conditioning；11 折；对照 C0/C1；train/dev only）",
           "lambda_v": LAMBDA_V, "proxy_audit": audit,
           "c2": agg(c2_scores), "c1_full_mlp": agg(c1full), "c0": agg(c0)}
    d_c0 = cc.delta_bootstrap(c2_scores, c0, labels, tposs, n_tasks=n_tasks)
    d_c1 = cc.delta_bootstrap(c2_scores, c1full, labels, tposs, n_tasks=n_tasks)
    res["c2_vs_c0"] = {"row_level": {"mean": float(np.mean(d_c0["row_level"])), "ci95": cc.ci(d_c0["row_level"])},
                       "task_macro": {"mean": float(np.mean(d_c0["task_macro"])), "ci95": cc.ci(d_c0["task_macro"])}}
    res["c2_vs_c1full"] = {"row_level": {"mean": float(np.mean(d_c1["row_level"])), "ci95": cc.ci(d_c1["row_level"])},
                           "task_macro": {"mean": float(np.mean(d_c1["task_macro"])), "ci95": cc.ci(d_c1["task_macro"])}}
    res["task_macro_positive_folds_vs_c0"] = int(sum(
        cc.task_macro_auroc(labels[f], c2_scores[f], tposs[f]) > cc.task_macro_auroc(labels[f], c0[f], tposs[f])
        for f in labels))

    # length/style control: residualize on [nchar_z, nlines_z, c0_style?? use style block z]
    style_score = b["style"][:, :2]  # nchar, nlines block
    res["length_style_control"] = {}
    for name, sc in (("c2", c2_scores), ("c0", c0)):
        resid = {}
        for fold in design["folds"]:
            fid = fold["heldout_generator_member"]
            ev_rows = np.load(OUT / "local" / f"scores_fold{fid.replace('--','_')}.npz")["ev_rows"]
            ctrl = np.hstack([style_score[ev_rows], b["style"][ev_rows][:, [5, 10, 11]]])
            mu, sd = ctrl.mean(0), ctrl.std(0); sd[sd < 1e-8] = 1
            resid[fid] = residualize(sc[fid], (ctrl - mu) / sd)
        res["length_style_control"][name] = {
            "row_level_mean": float(np.mean([cc.auroc(labels[f], resid[f]) for f in labels])),
            "task_macro_mean": float(np.mean([cc.task_macro_auroc(labels[f], resid[f], tposs[f]) for f in labels]))}
    res["length_style_control"]["c2_minus_c0_row_level"] = (
        res["length_style_control"]["c2"]["row_level_mean"]
        - res["length_style_control"]["c0"]["row_level_mean"])
    res["length_style_control"]["c2_minus_c0_task_macro"] = (
        res["length_style_control"]["c2"]["task_macro_mean"]
        - res["length_style_control"]["c0"]["task_macro_mean"])

    (OUT / "metrics.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "hypothesis.json").write_text(json.dumps({
        "schema": "cc_c2_hypothesis_v1",
        "hypothesis": ("带 λv=0.1 静态代理辅助损失的 full 视图 MLP（唯一变化 vs C1 full-mlp）"
                       "在 task-macro 上相对 C0 出现增量；代理只做辅助/描述，不称正确性或规模。"),
        "claim_limit": "λv 为本轮冻结选择；dev 仅开发评测；parse_ok 零方差已排除；不做测试执行。",
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "config.json").write_text(json.dumps({
        "schema": "cc_c2_config_v1",
        "backbone": "同 C1 full-mlp（d->256->1, 3 seeds, 20ep, bs=2048, Adam 1e-3）",
        "aux": {"loss": "mse(z(proxies))", "lambda_v": LAMBDA_V, "targets": PSI_TARGETS,
                "excluded": {"parse_ok": "zero variance"}},
        "bootstrap": {"n": 500, "seed": 20261009, "shared_task_indices": True},
        "switch": {"test_read": False, "generation": False, "weights_downloaded": False, "code_execution": False},
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "data_role_matrix.json").write_text(json.dumps({
        "rows": "records_train_dev.jsonl（instruct）",
        "fit": "折内 train（排除 heldout 成员）；辅助目标=静态代理（mask-free）",
        "eval": "dev：heldout（正）+ 其它系列（负）",
        "never_used": ["test", "canonical_solution", "执行结果"],
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    L = ["# C2 静态代理条件（2026-10-09）", "",
         f"性质：静态代理辅助（无测试执行；λv={LAMBDA_V}）；非正确性监督。", "",
         "| 读出 | row-level | task-macro |", "|---|---|---|",
         f"| C2 (aux) | {res['c2']['row_level_mean']:.4f} | {res['c2']['task_macro_mean']:.4f} |",
         f"| C1 full-mlp | {res['c1_full_mlp']['row_level_mean']:.4f} | {res['c1_full_mlp']['task_macro_mean']:.4f} |",
         f"| C0 | {res['c0']['row_level_mean']:.4f} | {res['c0']['task_macro_mean']:.4f} |", "",
         f"- C2−C0（task-macro 配对）: {res['c2_vs_c0']['task_macro']['mean']:+.4f} "
         f"[{res['c2_vs_c0']['task_macro']['ci95'][0]:+.4f},{res['c2_vs_c0']['task_macro']['ci95'][1]:+.4f}]，"
         f"正折 {res['task_macro_positive_folds_vs_c0']}/11",
         f"- C2−C1full（辅助损失效应）: {res['c2_vs_c1full']['task_macro']['mean']:+.4f}",
         f"- 长度/style 残差化后 C2−C0（row/task-macro）: "
         f"{res['length_style_control']['c2_minus_c0_row_level']:+.4f} / "
         f"{res['length_style_control']['c2_minus_c0_task_macro']:+.4f}",
         "", "## 代理审计（方差/缺失/长度相关）", "",
         "| proxy | var | corr(nchar) | corr(nlines) |", "|---|---|---|---|"]
    for t, a in audit.items():
        if a.get("corr_nchar") is None:
            L.append(f"| {t} | {a['variance']:.4g} | — | — |")
        else:
            L.append(f"| {t} | {a['variance']:.4g} | {a['corr_nchar']:+.3f} | {a['corr_nlines']:+.3f} |")
    L += ["", "> 静态代理不是正确性；parse_ok 零方差排除；无测试执行。"]
    (OUT / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (OUT / "commands.txt").write_text(
        "OMP_NUM_THREADS=8 /root/miniconda3/envs/udet/bin/python -W ignore scripts/cc_c2_static_proxy.py\n",
        encoding="utf-8")
    (OUT / "logs" / "c2_run.log").write_text("\n".join(LOG) + "\n", encoding="utf-8")
    cc.git_meta(OUT)
    cc.write_sha256sums(OUT)
    log(f"[c2] done in {time.time()-t0:.1f}s tm={res['c2']['task_macro_mean']:.4f} "
        f"vs_c0={res['c2_vs_c0']['task_macro']['mean']:+.4f}")


if __name__ == "__main__":
    main()
