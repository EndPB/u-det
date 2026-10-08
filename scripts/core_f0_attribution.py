"""F0：exact-unit attribution（§5）。

F0-A：115-way observed model_id 分类（配对单元；视图 h_c/h_i/Δ/u/P0；torch linear）。
       task-heldout（fit=798→eval=171）+ model-heldout 2 折 open-set（max-softmax rejection AUC）。
F0-B：官方系列 observed-member 迁移（11 折 heldout member；member vs 负集二分类；LR）。
       报告名按指导：observed_series/member transfer（family_is_confirmed=false）。

输出：artifacts/f0_exact_and_series_attribution_2026-10-08/
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
import variant_transfer_stage1_execute as s1  # noqa: E402

OUT = ROOT / "d-det/artifacts/f0_exact_and_series_attribution_2026-10-08"
SEED = 20261008
LOG: list[str] = []


def log(m):
    print(m, flush=True)
    LOG.append(m)


def unit_views(A, models, tasks):
    """配对单元视图：返回 dict name -> X_units；mi_u/ti_u 为单元级标签轴。"""
    mi = np.array([A["MODEL_I"][m] for m in models])
    ti = np.array([A["TASK_I"][t] for t in tasks])
    base = (mi[:, None] * A["nT"] * 2 + ti[None, :] * 2).ravel()
    Rc, Ri = base, base + 1
    H = A["emb_base"]
    hc, hi = H[Rc], H[Ri]
    S = (hc + hi) / 2.0; D = (hi - hc) / 2.0
    V = {
        "h_complete": hc,
        "h_instruct": hi,
        "delta": D,
        "u": np.hstack([S, D]),
        "p0": np.hstack([A["style"][Rc], A["meta"][Rc], A["sizelen"][Rc]]),
    }
    mi_u = np.repeat(mi, len(tasks))
    ti_u = np.tile(ti, len(models))
    return V, mi_u, ti_u


def fit_softmax(Xtr, ytr, n_class, epochs=10, lr=1e-3, bs=4096, seed=SEED):
    import torch
    import torch.nn as nn
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = nn.Linear(Xtr.shape[1], n_class).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    Xt = torch.tensor(Xtr, dtype=torch.float32, device=device)
    yt = torch.tensor(ytr, dtype=torch.long, device=device)
    rng = np.random.default_rng(seed)
    for ep in range(epochs):
        perm = rng.permutation(len(Xt))
        tot = 0.0
        for i in range(0, len(perm), bs):
            sel = torch.tensor(perm[i:i + bs], device=device)
            loss = nn.functional.cross_entropy(net(Xt[sel]), yt[sel])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.detach())
    return net


def infer_softmax(net, X):
    import torch
    device = next(net.parameters()).device
    with torch.inference_mode():
        return net(torch.tensor(np.asarray(X), dtype=torch.float32, device=device)).cpu().numpy()


def train_softmax(Xtr, ytr, n_class, Xdv, epochs=10, lr=1e-3, bs=4096, seed=SEED):
    net = fit_softmax(Xtr, ytr, n_class, epochs=epochs, lr=lr, bs=bs, seed=seed)
    return infer_softmax(net, Xdv) if len(Xdv) else np.zeros((0, n_class))


def metrics_topk(logits, y, k=(1, 5)):
    out = {}
    for kk in k:
        top = np.argsort(-logits, axis=1)[:, :kk]
        hit = (top == y[:, None]).any(1)
        out[f"top{kk}"] = float(np.mean(hit))
    pred = logits.argmax(1)
    # balanced acc / macro F1
    accs = []; f1s = []
    for c in np.unique(y):
        sel = y == c
        accs.append(float(np.mean(pred[sel] == c)))
        tp = np.sum((pred == c) & (y == c)); fp = np.sum((pred == c) & (y != c)); fn = np.sum((pred != c) & (y == c))
        pr = tp / max(1, tp + fp); rc = tp / max(1, tp + fn)
        f1s.append(2 * pr * rc / max(1e-9, pr + rc))
    out["balanced_acc"] = float(np.mean(accs))
    out["macro_f1"] = float(np.mean(f1s))
    out["top1"] = out["top1"]
    return out


def f0a(A, results):
    models = A["models"]
    split_task = {}
    for t in A["tasks"]:
        split_task[t] = A["splits"][A["TASK_I"][t] * 2]
    tr_tasks = [t for t in A["tasks"] if split_task[t] == "train"]
    dv_tasks = [t for t in A["tasks"] if split_task[t] == "dev"]
    Vtr, mi_tr, ti_tr = unit_views(A, models, tr_tasks)
    ytr = mi_tr  # 115 类
    Vdv, mi_dv, ti_dv = unit_views(A, models, dv_tasks)
    ydv = mi_dv
    res = {}
    for name, Xtr in Vtr.items():
        t = time.time()
        Xdv = Vdv[name]
        logits = train_softmax(Xtr, ytr, len(models), Xdv)
        m = metrics_topk(logits, ydv)
        # task-macro top1
        accs = []
        for q in np.unique(ti_dv):
            sel = ti_dv == q
            accs.append(float(np.mean(logits[sel].argmax(1) == ydv[sel])))
        m["task_macro_top1"] = float(np.mean(accs))
        # task-cluster CI (top1)
        uniq_t, inv_t = np.unique(ti_dv, return_inverse=True)
        rng = np.random.default_rng(SEED)
        idx_by = {q: np.where(inv_t == q)[0] for q in range(len(uniq_t))}
        vals = []
        for k in range(300):
            pick = rng.choice(len(uniq_t), size=len(uniq_t), replace=True)
            idx = np.concatenate([idx_by[q] for q in pick])
            vals.append(float(np.mean(logits[idx].argmax(1) == ydv[idx])))
        m["top1_ci95_task_cluster"] = [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]
        m["chance_top1"] = 1.0 / len(models)
        m["runtime_s"] = time.time() - t
        res[name] = m
        log(f"[f0a][{name}] top1={m['top1']:.4f} top5={m['top5']:.4f} "
            f"bal={m['balanced_acc']:.4f} macroF1={m['macro_f1']:.4f} (chance {m['chance_top1']:.4f})")
    results["f0a"] = res


def f0a_openset(A, results, folds=(0, 3)):
    """model-heldout open-set：fit=其余 model 全部类 → 留出类做 rejection（描述性口径）。"""
    from sklearn.metrics import roc_auc_score
    models = A["models"]
    split_task = {}
    for t in A["tasks"]:
        split_task[t] = A["splits"][A["TASK_I"][t] * 2]
    dv_tasks = [t for t in A["tasks"] if split_task[t] == "dev"]
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(models))
    mfolds = np.array_split(perm, 5)
    out = {}
    for f in folds:
        hold = [models[i] for i in mfolds[f]]
        fitm = [m for m in models if m not in set(hold)]
        Vtr, mi_tr, _ = unit_views(A, fitm, dv_tasks)
        # 重映射到 0..len(fitm)-1
        remap = {A["MODEL_I"][m]: i for i, m in enumerate(fitm)}
        mi_rel = np.array([remap[int(x)] for x in mi_tr])
        net = fit_softmax(Vtr["u"], mi_rel, len(fitm), epochs=6)
        s_known = infer_softmax(net, Vtr["u"]).max(1)
        Vhold, _, _ = unit_views(A, hold, dv_tasks)
        s_unk = infer_softmax(net, Vhold["u"]).max(1)
        y = np.concatenate([np.zeros(len(s_known)), np.ones(len(s_unk))])
        auc = float(roc_auc_score(y, np.concatenate([s_known, s_unk])))
        out[f"fold{f}"] = {"rejection_auc_maxsoftmax": auc, "n_known": int(len(s_known)),
                           "n_unknown": int(len(s_unk)), "n_fit_models": len(fitm), "n_hold_models": len(hold)}
        log(f"[f0a-openset] fold{f}: AUC={auc:.4f} (fit {len(fitm)} 类, hold {len(hold)})")
    results["f0a_openset"] = out


def f0b(A, results):
    """observed_series/member transfer：11 折 heldout member,member vs 负集二分类。"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, average_precision_score
    split_task = {}
    for t in A["tasks"]:
        split_task[t] = A["splits"][A["TASK_I"][t] * 2]
    tr_tasks = [t for t in A["tasks"] if split_task[t] == "train"]
    dv_tasks = [t for t in A["tasks"] if split_task[t] == "dev"]
    neg_all = [m for ss in s1.SERIES if ss != "CodeLlama-Instruct" for m in s1.SERIES[ss]]
    per_member = {}
    views = ("h_instruct", "delta", "u", "p0", "h_complete")
    for series in s1.SERIES:
        for h in s1.SERIES[series]:
            neg = [m for ss in s1.SERIES if ss != series for m in s1.SERIES[ss]]
            pos = [m for m in s1.SERIES[series] if m != h]
            units = pos + neg
            Vtr, mi_tr, ti_tr = unit_views(A, units, tr_tasks)
            Vdv, mi_dv, ti_dv = unit_views(A, units, dv_tasks)
            ytr = np.isin(mi_tr, [A["MODEL_I"][m] for m in pos]).astype(int)
            ydv = np.isin(mi_dv, [A["MODEL_I"][m] for m in pos]).astype(int)
            # model-balanced 权重
            cnt = {}
            for m in mi_tr:
                cnt[m] = cnt.get(m, 0) + 1
            w = np.array([1.0 / cnt[m] for m in mi_tr])
            entry = {}
            for vname in views:
                clf = LogisticRegression(max_iter=300, C=1.0).fit(Vtr[vname], ytr, sample_weight=w)
                s = clf.decision_function(Vdv[vname])
                entry[vname] = {"auroc": float(roc_auc_score(ydv, s)),
                                "ap": float(average_precision_score(ydv, s))}
            per_member[f"{series}::{h}"] = entry
            log(f"[f0b] {series}::{h} u_auroc={entry['u']['auroc']:.4f} delta={entry['delta']['auroc']:.4f}")
    # 汇总
    agg = {}
    for vname in views:
        vals = [v[vname]["auroc"] for v in per_member.values()]
        agg[vname] = {"mean_auroc": float(np.mean(vals)), "min": float(np.min(vals)), "max": float(np.max(vals))}
    results["f0b"] = {"report_name": "observed_series/member transfer（family_is_confirmed=false）",
                      "per_member": per_member, "aggregate": agg,
                      "note": "11 折 heldout member；负集=其他系列全体；train/dev 仅"}


def main():
    t0 = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["a", "open", "b", "all"])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "logs").mkdir(exist_ok=True)
    mpath = OUT / "metrics_f0.json"
    results = json.loads(mpath.read_text(encoding="utf-8")) if mpath.exists() else \
        {"schema": "f0_metrics_v1", "generated_utc": datetime.now(timezone.utc).isoformat(),
         "switches": {"training_allowed": True, "generation_allowed": False,
                      "test_read_allowed": False, "old_test_reuse": False}}
    A = ev.load_assets()
    if args.stage in ("a", "all"):
        log("[f0] F0-A 115-way")
        f0a(A, results)
        mpath.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    if args.stage in ("open", "all"):
        log("[f0] F0-A open-set (2 folds)")
        f0a_openset(A, results)
        mpath.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    if args.stage in ("b", "all"):
        log("[f0] F0-B series member (11 folds)")
        f0b(A, results)
        mpath.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "logs" / "f0_run.log").write_text("\n".join(LOG) + "\n", encoding="utf-8")
    # 报告（若三部分齐）
    L = ["# F0：exact-unit 与 observed-series 归因（train/dev）", ""]
    if "f0a" in results:
        L += ["## F0-A 115-way observed model_id（task-heldout）", "",
              "| 视图 | top1 | top5 | bal-acc | macro-F1 | task-macro |", "|---|---|---|---|---|---|"]
        for name, m in results["f0a"].items():
            L.append(f"| {name} | {m['top1']:.4f} | {m['top5']:.4f} | {m['balanced_acc']:.4f} | "
                     f"{m['macro_f1']:.4f} | {m['task_macro_top1']:.4f} |")
        L.append(f"（chance top1={1/115:.4f}）")
    if "f0a_openset" in results:
        L += ["", "## F0-A open-set（model-heldout, max-softmax rejection）"]
        for k, v in results["f0a_openset"].items():
            L.append(f"- {k}: AUC={v['rejection_auc_maxsoftmax']:.4f}")
    if "f0b" in results:
        L += ["", "## F0-B observed_series/member transfer（11 折 heldout member）", "",
              "| 视图 | 平均 AUROC | min | max |", "|---|---|---|---|"]
        for vname, v in results["f0b"]["aggregate"].items():
            L.append(f"| {vname} | {v['mean_auroc']:.4f} | {v['min']:.4f} | {v['max']:.4f} |")
        L.append("")
        L.append("> 报告名使用 `observed_series/member transfer`（family_is_confirmed=false）。")
    (OUT / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    log(f"[f0] done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
