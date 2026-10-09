"""C1: 题面条件化代码归因（server-side, 2026-10-09）.

11 个 member-heldout 折；5 个输入视图共用折/标准化/seed/容量：
  code_only   : h_y (512)
  prompt_only : h_t (512)
  ht_hy       : [h_t; h_y] (1024)
  full        : [h_t; h_y; h_t*h_y; |h_t-h_y|; psi] (2057)
  psi_only    : psi (9)
两个头：linear (LR C=1) 与 small MLP (d->256->1, 3 seeds 平均, 20 epochs)。
对照：C0 强 P0（读取 C0 保存的逐折 eval 分数，同一任务索引配对 bootstrap）。
回答"C1 是否改变任务内排序"：full vs code_only 的 task-macro 差值。
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

OUT = ROOT / "d-det/artifacts/code_conditioned_c1_prompt_conditioned_2026-10-09"
C0_DIR = ROOT / "d-det/artifacts/code_conditioned_c0_strong_p0_2026-10-09"
VIEWS = ["code_only", "prompt_only", "ht_hy", "full", "psi_only"]
SEEDS = (0, 1, 2)
EPOCHS = 20
LOG: list[str] = []


def log(m):
    print(m, flush=True)
    LOG.append(m)


def build_view(design, name, row_idx):
    b = design["bundle"]
    ht = design["emb_task"][b["task_idx"][row_idx]]
    hy = b["hy_small"][row_idx]
    psi = b["psi"][row_idx]
    if name == "code_only":
        return hy
    if name == "prompt_only":
        return ht
    if name == "ht_hy":
        return np.hstack([ht, hy])
    if name == "full":
        return np.hstack([ht, hy, ht * hy, np.abs(ht - hy), psi])
    if name == "psi_only":
        return psi
    raise ValueError(name)


def fit_mlp(Xtr, ytr, Xev, seed, device, epochs=EPOCHS):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    d = Xtr.shape[1]
    net = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Linear(256, 1)).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    X = torch.tensor(Xtr.astype(np.float32), device=device)
    y = torch.tensor(ytr.astype(np.float32), device=device)
    rng = np.random.default_rng(seed)
    bs = 2048
    for _ in range(epochs):
        perm = rng.permutation(len(X))
        for i in range(0, len(perm), bs):
            sel = torch.tensor(perm[i:i + bs], device=device)
            logits = net(X[sel]).squeeze(-1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y[sel])
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.inference_mode():
        p = net(torch.tensor(Xev.astype(np.float32), device=device)).squeeze(-1).sigmoid().cpu().numpy()
    return p


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "local").mkdir(exist_ok=True)
    design = cc.load_design()
    rows = design["rows"]
    b = design["bundle"]
    task_split = {r["task_id"]: r["split"] for r in rows}
    dev_tasks = sorted(t for t in design["tasks_all"] if task_split[t] == "dev")
    tpos_of_task = {t: i for i, t in enumerate(dev_tasks)}
    n_tasks = len(dev_tasks)
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"[c1] device={device}")

    # load C0 frozen per-fold eval scores (pair by fold filename, not by row inference)
    c0 = {}
    for fold in design["folds"]:
        fid = fold["heldout_generator_member"]
        p = C0_DIR / "local" / f"scores_dev_fold{fid.replace('--','_')}.npz"
        z = np.load(p)
        c0[fid] = {"ev_rows": z["ev_rows"], "y": z["y"], "taskpos": z["taskpos"],
                   "s_fused": z["fused"]}

    per_view_scores = {v: {"linear": {}, "mlp": {}} for v in VIEWS}
    labels, tposs = {}, {}
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
        # verify C0 alignment
        z0 = c0[fid]
        assert np.array_equal(z0["ev_rows"], ev_rows), "C0/C1 eval row order mismatch"
        assert np.array_equal(z0["y"], y_ev)
        assert np.array_equal(z0["taskpos"], tp)
        for view in VIEWS:
            Xtr = build_view(design, view, fit_rows)
            Xev = build_view(design, view, ev_rows)
            mu = Xtr.mean(0)
            sd = Xtr.std(0)
            sd[sd < 1e-8] = 1.0
            Xtr = (Xtr - mu) / sd
            Xev = (Xev - mu) / sd
            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression(max_iter=3000, C=1.0).fit(Xtr, y_fit)
            per_view_scores[view]["linear"][fid] = clf.decision_function(Xev)
            ps = np.mean([fit_mlp(Xtr, y_fit, Xev, s, device) for s in SEEDS], axis=0)
            per_view_scores[view]["mlp"][fid] = ps
        np.savez_compressed(
            OUT / "local" / f"scores_fold{fid.replace('--','_')}.npz",
            ev_rows=ev_rows, y=y_ev, taskpos=tp,
            **{f"{v}_{h}": per_view_scores[v][h][fid] for v in VIEWS for h in ("linear", "mlp")})
        log(f"[c1] fold {fid[:40]:40s} done ({time.time()-tf0:.0f}s)")

    # metrics + deltas vs C0
    def agg(scores):
        row_mean = float(np.mean([cc.auroc(labels[f], scores[f]) for f in labels]))
        tm_mean = float(np.mean([cc.task_macro_auroc(labels[f], scores[f], tposs[f])
                                 for f in labels]))
        boot = cc.bootstrap_metrics(scores, labels, tposs, n_tasks=n_tasks)
        return {"row_level_mean": row_mean, "task_macro_mean": tm_mean,
                "ci95": {"row_level": cc.ci(boot["row_level"]),
                         "task_macro": cc.ci(boot["task_macro"])}}

    c0_scores = {fid: c0[fid]["s_fused"] for fid in c0}
    res = {"schema": "cc_c1_v1", "generated_utc": datetime.now(timezone.utc).isoformat(),
           "report_first_line": "C1 题面条件化代码归因（11 折；5 视图×2 头；对照 C0 强 P0；train/dev only）",
           "views": {}, "c0_recomputed_on_same_indices": agg(c0_scores)}
    for view in VIEWS:
        res["views"][view] = {}
        for head in ("linear", "mlp"):
            sc = per_view_scores[view][head]
            m = agg(sc)
            d = cc.delta_bootstrap(sc, c0_scores, labels, tposs, n_tasks=n_tasks)
            m["delta_vs_c0"] = {
                "row_level": {"mean": float(np.mean(d["row_level"])), "ci95": cc.ci(d["row_level"])},
                "task_macro": {"mean": float(np.mean(d["task_macro"])), "ci95": cc.ci(d["task_macro"])}}
            pos_folds = int(sum(cc.task_macro_auroc(labels[f], sc[f], tposs[f])
                                > cc.task_macro_auroc(labels[f], c0_scores[f], tposs[f])
                                for f in labels))
            m["task_macro_positive_folds_vs_c0"] = pos_folds
            res["views"][view][head] = m
            log(f"[c1] {view:12s} {head:6s} row={m['row_level_mean']:.4f} tm={m['task_macro_mean']:.4f} "
                f"dTM={m['delta_vs_c0']['task_macro']['mean']:+.4f} pos={pos_folds}/11")

    # task-conditioning within-task ranking: full vs code_only
    for head in ("linear", "mlp"):
        d = cc.delta_bootstrap(per_view_scores["full"][head], per_view_scores["code_only"][head],
                               labels, tposs, n_tasks=n_tasks)
        res[f"full_vs_code_only_{head}"] = {
            "task_macro": {"mean": float(np.mean(d["task_macro"])), "ci95": cc.ci(d["task_macro"])},
            "row_level": {"mean": float(np.mean(d["row_level"])), "ci95": cc.ci(d["row_level"])}}

    (OUT / "metrics.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "hypothesis.json").write_text(json.dumps({
        "schema": "cc_c1_hypothesis_v1",
        "hypothesis": ("题面条件（h_t 及其与 h_y 的交互）相对 code-only/h_t-only 改变任务内排序，"
                       "并在 task-macro 上相对 C0 强 P0 产生可检验增量（阈值：mean≥1pt、≥8/11 折为正、CI>0）。"),
        "claim_limit": "训练/选择仅用折内 train；dev 仅开发评测；不称后训练因果；family_is_confirmed=false。",
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "config.json").write_text(json.dumps({
        "schema": "cc_c1_config_v1",
        "views": {"code_only": "h_y(512)", "prompt_only": "h_t(512)", "ht_hy": "[h_t;h_y](1024)",
                  "full": "[h_t;h_y;h_t*h_y;|h_t-h_y|;psi](2057)", "psi_only": "psi(9)"},
        "heads": {"linear": "LogisticRegression(C=1,max_iter=3000, l2)",
                  "mlp": "torch d->256->1 ReLU, Adam 1e-3, 20 epochs, bs=2048, 3 seeds avg"},
        "standardization": "train-fold mean/std per fold (zero-std dims scale=1)",
        "bootstrap": {"n": 500, "seed": 20261009, "shared_task_indices": True, "paired_vs": "C0"},
        "switches": {"train_dev_only": True, "test_read": False, "generation": False,
                     "weights_downloaded": False, "code_execution": False},
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "data_role_matrix.json").write_text(json.dumps({
        "rows": "records_train_dev.jsonl（10,659 行；instruct）",
        "fit": "折内 train split（排除 heldout 成员），目标=series==F",
        "eval": "dev split：heldout 成员（正）+ 其它系列（负）",
        "c0_reference": "C0 冻结逐折分数（同一 eval 行序断言一致）",
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    L = ["# C1 题面条件化代码归因（2026-10-09）", "",
         "性质：开发集机制诊断（对照 C0 强 P0）；非后训练因果。", "",
         "| view | head | row-level | task-macro | ΔTM vs C0 [CI] | +折 |", "|---|---|---|---|---|---|"]
    for view in VIEWS:
        for head in ("linear", "mlp"):
            m = res["views"][view][head]
            dm = m["delta_vs_c0"]["task_macro"]
            L.append(f"| {view} | {head} | {m['row_level_mean']:.4f} | {m['task_macro_mean']:.4f} | "
                     f"{dm['mean']:+.4f} [{dm['ci95'][0]:+.4f},{dm['ci95'][1]:+.4f}] | {m['task_macro_positive_folds_vs_c0']}/11 |")
    L += ["", "## C0 参考（同索引重算）",
          f"- row={res['c0_recomputed_on_same_indices']['row_level_mean']:.4f} "
          f"tm={res['c0_recomputed_on_same_indices']['task_macro_mean']:.4f}",
          "", "## full vs code_only（任务内排序）"]
    for head in ("linear", "mlp"):
        d = res[f"full_vs_code_only_{head}"]["task_macro"]
        L.append(f"- {head}: ΔTM = {d['mean']:+.4f} [{d['ci95'][0]:+.4f},{d['ci95'][1]:+.4f}]")
    L += ["", "> 全部拟合 fit-only；dev 仅开发评测；无 test/生成/权重下载。"]
    (OUT / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (OUT / "commands.txt").write_text(
        "OMP_NUM_THREADS=8 /root/miniconda3/envs/udet/bin/python -W ignore scripts/cc_c1_prompt_conditioned.py\n",
        encoding="utf-8")
    (OUT / "logs" / "c1_run.log").write_text("\n".join(LOG) + "\n", encoding="utf-8")
    cc.git_meta(OUT)
    cc.write_sha256sums(OUT)
    log(f"[c1] done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
