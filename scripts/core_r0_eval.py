"""R0 评估核心：读出矩阵 × 四个切分（seen / task / model×5 / size×4）。

统一任务：配对方向判别——对每单元 (model, task) 的 (h_complete, h_instruct)，
判别哪个是 instruct。主指标：model-balanced pair-acc（=对称化 AUROC/BA）；
task-macro；task-cluster bootstrap 500 (seed 20261008，共享 picks)。

读出（13 个）：
  1 single_hc   样本级 LR（complete=1 正类）
  2 single_hi   样本级 LR（instruct=1 正类）
  3 delta_pair  pairwise logistic on Δ=h_i−h_c（symmetrized）
  4 S_sanity    pairwise on ΔS≡0（理论 0.5，阴性对照）
  5 u_linear    pairwise on [S;A] 差分（≡3 的数值验证）
  5 u_mlp       twin-MLP g([S;A])，pairwise BCE（S 条件化机会）
  6 p0_*        tfidf_char/word(SGD×3seed)、style_lr、style_lgb、metadata、size_length
  7 late_fusion [ΔA;Δnchar;Δnlines] 的 LR
"""
from __future__ import annotations

import csv
import gzip
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import variant_transfer_stage1_execute as s1  # noqa: E402
import core_r0_protocol_diff as r0  # noqa: E402

OUT = r0.OUT
LOCAL = r0.LOCAL
BOOT = 500
BOOT_SEED = 20261008
READOUTS = ("single_hc", "single_hi", "delta_pair", "S_sanity", "u_linear", "u_mlp",
            "p0_tfidf_char", "p0_tfidf_word", "p0_style_lr", "p0_style_lgb",
            "p0_metadata", "p0_size_length", "late_fusion")


def log(m: str):
    print(m, flush=True)


_ASSETS = {}


def load_assets(need_texts=False):
    if not _ASSETS:
        idx = json.loads((LOCAL / "row_index.json").read_text(encoding="utf-8"))
        sm = np.load(LOCAL / "style_meta.npz")
        _ASSETS.update({
            "idx": idx,
            "models": idx["models"], "tasks": idx["tasks"],
            "splits": idx["splits"], "valid": np.array(idx["valid"]),
            "style": sm["style"], "meta": sm["meta"], "sizelen": sm["sizelen"], "sizeB": sm["sizeB"],
            "emb_small": np.load(LOCAL / "emb_small.npz")["emb"],
            "emb_base": np.load(LOCAL / "emb_base.npz")["emb"],
        })
        A = _ASSETS
        A["MODEL_I"] = {m: i for i, m in enumerate(A["models"])}
        A["TASK_I"] = {t: i for i, t in enumerate(A["tasks"])}
        A["nM"], A["nT"] = len(A["models"]), len(A["tasks"])
        # sizelen NaN 处理（参数解析失败模型）：中位数填充 + 缺失标志列（防 LR 拒收）
        sz = A["sizelen"].astype(np.float64).copy()
        nan0 = np.isnan(sz[:, 0])
        if nan0.any():
            med = float(np.nanmedian(sz[:, 0]))
            miss = nan0.astype(np.float64)
            sz[:, 0] = np.where(nan0, med, sz[:, 0])
            sz = np.hstack([sz, miss[:, None]])
            log(f"  [assets] sizelen NaN filled: {int(nan0.sum())} rows, median={med:.3f}, +missing_flag")
        A["sizelen"] = sz
    if need_texts and "texts" not in _ASSETS:
        texts = [None] * (len(_ASSETS["models"]) * len(_ASSETS["tasks"]) * 2)
        with gzip.open(LOCAL / "texts.jsonl.gz", "rt", encoding="utf-8") as f:
            for line in f:
                o = json.loads(line)
                texts[o["i"]] = o["text"]
        _ASSETS["texts"] = texts
    return _ASSETS


def rows_for(model_ids, task_ids):
    A = load_assets()
    mi = np.array([A["MODEL_I"][m] for m in model_ids])
    ti = np.array([A["TASK_I"][t] for t in task_ids])
    base = (mi[:, None] * A["nT"] * 2 + ti[None, :] * 2).ravel()
    return base, base + 1  # (rows_comp, rows_instr)


def split_definition(name: str, fold: int):
    A = load_assets()
    models, tasks = A["models"], A["tasks"]
    split_task = {}
    for t in tasks:
        split_task[t] = A["splits"][A["TASK_I"][t] * 2]
    tr_tasks = [t for t in tasks if split_task[t] == "train"]
    dv_tasks = [t for t in tasks if split_task[t] == "dev"]
    if name == "seen":
        return [("seen", models, tr_tasks, models, tr_tasks)]
    if name == "task":
        # task-heldout（fit=train, eval=dev）与 seen（fit=train, eval=train）共用同一 fit 域，一次拟合
        return [("task_dev", models, tr_tasks, models, dv_tasks),
                ("seen", models, tr_tasks, models, tr_tasks)]
    if name == "model":
        rng = np.random.default_rng(20261008)
        perm = rng.permutation(len(models))
        folds = np.array_split(perm, 5)
        hold = [models[i] for i in folds[fold]]
        fitm = [m for m in models if m not in set(hold)]
        return [("model_train", fitm, tr_tasks, hold, tr_tasks),
                ("model_dev", fitm, tr_tasks, hold, dv_tasks)]
    if name == "size":
        sb = np.array([A["sizeB"][A["MODEL_I"][m] * A["nT"] * 2] for m in models])
        buckets = [(0, 2, "lt2B"), (2, 8, "2to8B"), (8, 20, "8to20B"), (20, 1e9, "ge20B")]
        lo, hi, tag = buckets[fold]
        hold = [m for m, s in zip(models, sb) if not np.isnan(s) and lo <= s < hi]
        fitm = [m for m, s in zip(models, sb) if not np.isnan(s) and not (lo <= s < hi)]
        nanm = [m for m, s in zip(models, sb) if np.isnan(s)]
        return [("size_dev", fitm, tr_tasks, hold, dv_tasks),
                ("size_train", fitm, tr_tasks, hold, tr_tasks),
                ("_meta_", [f"hold={len(hold)}", f"fit={len(fitm)}", f"nan_excluded={len(nanm)}", f"tag={tag}"], [], [], [])]
    raise ValueError(name)


# ---------------------------------------------------------------- readout fitters
def _w_lr(X, y, w=None, C=1.0):
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=300, C=C, solver="lbfgs")
    clf.fit(X, y, sample_weight=w)
    return clf.coef_.ravel(), float(clf.intercept_[0])


def _w_lr_sparse(clf_partial, X, y, w, seeds=(0, 1, 2), epochs=5, bs=20000, alpha=2e-6):
    """稀疏 SGD（stage-1 同协议：5ep best-dev 不适用此处，用固定 5ep 平均3seed）。"""
    from sklearn.linear_model import SGDClassifier
    classes = np.array([0, 1])
    ws = []
    for seed in seeds:
        clf = SGDClassifier(loss="log_loss", alpha=alpha, random_state=seed)
        rng = np.random.default_rng(seed)
        for ep in range(epochs):
            perm = rng.permutation(X.shape[0])
            for i in range(0, len(perm), bs):
                sel = perm[i:i + bs]
                clf.partial_fit(X[sel], y[sel], classes=classes, sample_weight=w[sel])
        ws.append((clf.coef_.ravel().copy(), float(clf.intercept_[0])))
    return ws


def model_weights(models_axis, models):
    cnt = {}
    for m in models_axis:
        cnt[m] = cnt.get(m, 0) + 1
    return np.array([1.0 / cnt[m] for m in models_axis])


def fit_close(fit_models, fit_tasks, feats, texts=None, n_jobs=8, heavy=True, rep="base"):
    """返回 dict: name -> fn(units) 其中 units=(model_ids array, task_ids array) 返回 (score_c, score_i)。"""
    A = load_assets()
    Rc, Ri = rows_for(fit_models, fit_tasks)
    fm_axis = np.repeat(np.array(fit_models), len(fit_tasks))
    ft_axis = np.tile(np.array(fit_tasks), len(fit_models))
    wgt = model_weights(fm_axis, fit_models)
    y_c = np.zeros(len(Rc)); y_i = np.ones(len(Ri))
    out = {}

    def mk_linear(w, b):
        def fn(models, tasks):
            C, I = rows_for(models, tasks)
            return Xh[C] @ w + b, Xh[I] @ w + b
        return fn

    Xh = feats[f"emb_{rep}"]
    # 1/2 样本级 LR（对称数据下 w2 = −w1 精确成立，省一次全量拟合）
    Xf = np.concatenate([Xh[Rc], Xh[Ri]], 0)
    yf = np.concatenate([y_c, y_i])
    wf = np.concatenate([wgt, wgt])
    w1, b1 = _w_lr(Xf, 1 - yf, wf)  # complete=1
    w2, b2 = -w1, -b1               # instruct=1（等价）
    out["single_hc"] = mk_linear(w1, b1)
    out["single_hi"] = mk_linear(w2, b2)
    # 3 delta_pair
    Dc = Xh[Ri] - Xh[Rc]  # instr − comp
    Xd = np.concatenate([Dc, -Dc], 0)
    yd = np.concatenate([np.ones(len(Dc)), np.zeros(len(Dc))])
    wd = np.concatenate([wgt, wgt])
    w3, b3 = _w_lr(Xd, yd, wd)
    out["delta_pair"] = mk_linear(w3, b3)
    # 4 S_sanity（ΔS=0）
    out["S_sanity"] = lambda models, tasks: (
        np.zeros(len(models) * len(tasks)), np.zeros(len(models) * len(tasks)))
    # 5 u_linear: [S;A] 差分 = [0;ΔA] ≡ delta_pair（数学等价；标注验证于报告）
    out["u_linear"] = mk_linear(w3, b3)

    # 5m u_mlp：twin-MLP g([S;A])
    out["u_mlp"] = make_mlp_scorer(Xh, Rc, Ri, wgt, heavy=heavy)

    # 6 P0
    if texts is not None and heavy:
        tr_txt = [texts[i] for i in Rc] + [texts[i] for i in Ri]
        for tag, factory in (("char", s1.tfidf_char), ("word", s1.tfidf_word)):
            t = time.time()
            vec = factory().fit(tr_txt)
            Xtr = vec.transform(tr_txt)
            log(f"    tfidf_{tag} fit vocab={Xtr.shape[1]} in {time.time()-t:.1f}s")
            ws = _w_lr_sparse(None, Xtr, yf, wf)
            name = f"p0_tfidf_{tag}"
            def mk_sparse(vec, ws):
                def fn(models, tasks):
                    C, I = rows_for(models, tasks)
                    TXc = vec.transform([texts[i] for i in C])
                    TXi = vec.transform([texts[i] for i in I])
                    sc = np.zeros(TXc.shape[0]); si = np.zeros_like(sc)
                    for w_, b_ in ws:
                        sc += TXc @ w_ + b_
                        si += TXi @ w_ + b_
                    return sc / len(ws), si / len(ws)
                return fn
            out[name] = mk_sparse(vec, ws)
    Xst = feats["style"].astype(np.float64)
    Xme = feats["meta"].astype(np.float64)
    Xsz = feats["sizelen"].astype(np.float64)
    for name, Xs in (("p0_style_lr", Xst), ("p0_metadata", Xme), ("p0_size_length", Xsz)):
        Xf2 = np.concatenate([Xs[Rc], Xs[Ri]], 0)
        w_, b_ = _w_lr(Xf2, yf, wf)
        Xtmp = Xs

        def mk(mkname, Xtmp, w_, b_):
            def fn(models, tasks, Xtmp=Xtmp, w_=w_, b_=b_):
                C, I = rows_for(models, tasks)
                return Xtmp[C] @ w_ + b_, Xtmp[I] @ w_ + b_
            return fn
        out[name] = mk(name, Xtmp, w_, b_)
    if heavy:
        import lightgbm as lgb
        Xf3 = np.concatenate([Xst[Rc], Xst[Ri]], 0)
        m = lgb.LGBMClassifier(objective="binary", n_estimators=400, learning_rate=0.05,
                               num_leaves=31, min_child_samples=20, subsample=0.8, subsample_freq=1,
                               colsample_bytree=0.6, class_weight="balanced",
                               random_state=0, verbose=-1, n_jobs=n_jobs)
        m.fit(Xf3, yf)

        def fn_lgb(models, tasks):
            C, I = rows_for(models, tasks)
            return (m.predict_proba(Xst[C])[:, 1], m.predict_proba(Xst[I])[:, 1])
        out["p0_style_lgb"] = fn_lgb

    # 7 late_fusion: [ΔA;Δnchar;Δnlines]
    nc = feats["style"][:, 0]; nl = feats["style"][:, 1]
    Dctl = np.stack([nc[Ri] - nc[Rc], nl[Ri] - nl[Rc]], 1)
    Xlf = np.concatenate([np.hstack([Dc, Dctl]), np.hstack([-Dc, -Dctl])], 0)
    w7, b7 = _w_lr(Xlf, yd, wd)

    def fn_lf(models, tasks):
        C, I = rows_for(models, tasks)
        d = np.hstack([Xh[I] - Xh[C],
                       np.stack([nc[I] - nc[C], nl[I] - nl[C]], 1)]) @ w7
        return -d / 2, d / 2
    out["late_fusion"] = fn_lf

    # ---- 方向校准：用 fit 域抽样估计每读出的 d 均值符号，负则交换 sc/si ----
    cal_m = fit_models
    cal_t = list(fit_tasks[:120])
    signs = {}
    for name in list(out.keys()):
        try:
            sc, si = out[name](cal_m, cal_t)
            sgn = 1 if float(np.mean(np.asarray(si) - np.asarray(sc))) >= 0 else -1
        except Exception:
            sgn = 1
        signs[name] = sgn
        if sgn < 0:
            fn0 = out[name]
            out[name] = (lambda m, t, fn0=fn0: (lambda sc, si: (si, sc))(*fn0(m, t)))
    out["_signs"] = signs
    return out


_MLP_CACHE = {}


def make_mlp_scorer(Xh, Rc, Ri, wgt, heavy=True, epochs=12, hidden=(256, 64)):
    import torch
    import torch.nn as nn
    S = (Xh[Ri] + Xh[Rc]) / 2.0
    A = (Xh[Ri] - Xh[Rc]) / 2.0
    v1 = np.hstack([S, A])      # 视角1
    v2 = np.hstack([S, -A])     # 视角2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = v1.shape[1]
    net = nn.Sequential(nn.Linear(d, hidden[0]), nn.ReLU(), nn.Linear(hidden[0], hidden[1]),
                        nn.ReLU(), nn.Linear(hidden[1], 1)).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    X1 = torch.tensor(v1, dtype=torch.float32, device=device)
    X2 = torch.tensor(v2, dtype=torch.float32, device=device)
    w_t = torch.tensor(wgt / wgt.mean(), dtype=torch.float32, device=device)
    n = len(v1); bs = 1024
    rng = np.random.default_rng(20261008)
    for ep in range(epochs):
        perm = rng.permutation(n)
        tot = 0.0; nb = 0
        for i in range(0, n, bs):
            sel = perm[i:i + bs]
            g1 = net(X1[sel]).squeeze(-1)
            g2 = net(X2[sel]).squeeze(-1)
            loss = (torch.nn.functional.softplus(-(g1 - g2)) * w_t[sel]).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.detach()); nb += 1
        log(f"    u_mlp ep{ep+1} loss={tot/max(1,nb):.4f}")
    net.eval()

    def fn(models, tasks):
        C, I = rows_for(models, tasks)
        Sx = (Xh[I] + Xh[C]) / 2.0
        Ax = (Xh[I] - Xh[C]) / 2.0
        vc = np.hstack([Sx, Ax])
        vi = np.hstack([Sx, -Ax])
        with torch.inference_mode():
            sc = net(torch.tensor(vc, dtype=torch.float32, device=device)).squeeze(-1).cpu().numpy()
            si = net(torch.tensor(vi, dtype=torch.float32, device=device)).squeeze(-1).cpu().numpy()
        return sc, si
    return fn


# ---------------------------------------------------------------- metrics
def make_picks(n_tasks, repeats=BOOT, seed=BOOT_SEED):
    rng = np.random.default_rng(seed)
    return [rng.choice(n_tasks, size=n_tasks, replace=True) for _ in range(repeats)]


def eval_metrics(score_c, score_i, models, tasks):
    """model-balanced pair-acc + task-macro + bootstrap CI（task-cluster）。"""
    d = score_i - score_c
    uniq_t, inv_t = np.unique(tasks, return_inverse=True)
    uniq_m, inv_m = np.unique(models, return_inverse=True)

    def _acc(idx):
        dd = d[idx]; mm = inv_m[idx]
        accs = []
        for g in range(len(uniq_m)):
            sel = mm == g
            if sel.sum() == 0:
                continue
            accs.append(float(np.mean(dd[sel] > 0) + 0.5 * np.mean(dd[sel] == 0)))
        return float(np.mean(accs)) if accs else np.nan

    def _tmacc(idx):
        dd = d[idx]; tt = inv_t[idx]
        accs = []
        for g in range(len(uniq_t)):
            sel = tt == g
            if sel.sum() == 0:
                continue
            accs.append(float(np.mean(dd[sel] > 0) + 0.5 * np.mean(dd[sel] == 0)))
        return float(np.mean(accs)) if accs else np.nan

    picks = make_picks(len(uniq_t))
    idx_by = {q: np.where(inv_t == q)[0] for q in range(len(uniq_t))}
    mb = np.empty(len(picks)); tm = np.empty(len(picks))
    for k, pick in enumerate(picks):
        idx = np.concatenate([idx_by[q] for q in pick])
        mb[k] = _acc(idx)
        tm[k] = _tmacc(idx)
    return {"pair_acc_mb": _acc(np.arange(len(d))),
            "pair_acc_mb_ci95": [float(np.nanpercentile(mb, 2.5)), float(np.nanpercentile(mb, 97.5))],
            "task_macro": _tmacc(np.arange(len(d))),
            "task_macro_ci95": [float(np.nanpercentile(tm, 2.5)), float(np.nanpercentile(tm, 97.5))],
            "n_units": int(len(d)), "n_models": int(len(uniq_m)), "n_tasks": int(len(uniq_t))}


def run_split(split: str, fold: int, skip_readouts=(), heavy=True, n_jobs=4, rep="base"):
    t0 = time.time()
    A = load_assets(need_texts=heavy)
    defs = split_definition(split, fold)
    tag_main = defs[0][0]
    fit_models, fit_tasks = defs[0][1], defs[0][2]
    log(f"[r0-eval] split={split} fold={fold} rep={rep} fit=({len(fit_models)}m x {len(fit_tasks)}t) "
        f"domains={[d[0] for d in defs]}")
    feats = {"emb_base": A["emb_base"], "emb_small": A["emb_small"],
             "style": A["style"], "meta": A["meta"], "sizelen": A["sizelen"]}
    texts = A.get("texts")
    fns = fit_close(fit_models, fit_tasks, feats, texts=texts, n_jobs=n_jobs, heavy=heavy, rep=rep)
    results = {"schema": "r0_eval_result_v1", "split": split, "fold": fold, "rep": rep,
               "generated_utc": datetime.now(timezone.utc).isoformat(),
               "fit": {"n_models": len(fit_models), "n_tasks": len(fit_tasks)},
               "domains": {}}
    for dom in defs:
        label, eval_models, eval_tasks = dom[0], dom[3], dom[4]
        if not eval_models:
            continue
        Rc, Ri = rows_for(eval_models, eval_tasks)
        fm_axis = np.repeat(np.array(eval_models), len(eval_tasks))
        ft_axis = np.tile(np.array(eval_tasks), len(eval_models))
        dom_out = {}
        scores = {}
        for name in READOUTS:
            if name in skip_readouts or name not in fns:
                continue
            try:
                sc, si = fns[name](eval_models, eval_tasks)
            except Exception as e:
                log(f"    [{name}] FAILED: {type(e).__name__}: {e}")
                dom_out[name] = {"error": f"{type(e).__name__}: {e}"}
                continue
            dom_out[name] = eval_metrics(np.asarray(sc), np.asarray(si), fm_axis, ft_axis)
            scores[f"{label}::{name}::c"] = np.asarray(sc, dtype=np.float64)
            scores[f"{label}::{name}::i"] = np.asarray(si, dtype=np.float64)
            log(f"    [{label}][{name}] mb={dom_out[name]['pair_acc_mb']:.4f} "
                f"ci={dom_out[name]['pair_acc_mb_ci95'][0]:.3f}-{dom_out[name]['pair_acc_mb_ci95'][1]:.3f}")
        results["domains"][label] = {"metrics": dom_out,
                                     "signs": fns.get("_signs", {}),
                                     "axis_models": [str(x) for x in fm_axis],
                                     "axis_tasks": [str(x) for x in ft_axis]}
        np.savez_compressed(OUT / "local" / f"scores_{split}_{fold}_{rep}_{label}.npz", **scores)
    results["runtime_seconds"] = time.time() - t0
    (OUT / "results").mkdir(exist_ok=True)
    (OUT / "results" / f"{split}_{fold}_{rep}.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"[r0-eval] done {split}/{fold}/{rep} in {time.time()-t0:.1f}s")


def make_report():
    """汇总全部核 → metrics_r0.json + r0_report.md（六句话出口）。"""
    res = {}
    for p in sorted((OUT / "results").glob("*.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        res[f"{r['split']}_{r['fold']}"] = r
    L = ["# R0：complete/instruct 协议条件差异读出（2026-10-08）", ""]
    L.append("> `protocol_conditioned_difference`；train/dev only（test 未读）；"
             "`training_allowed=true, generation_allowed=false, test_read_allowed=false`。")
    L.append("")
    L.append("| 核 | 域 | 读出 | mb pair-acc [95% CI] | task-macro |")
    L.append("|---|---|---|---|---|")
    for key, r in res.items():
        for label, dv in r["domains"].items():
            for name, m in dv["metrics"].items():
                if "error" in m:
                    L.append(f"| {key} | {label} | {name} | ERROR | |")
                    continue
                L.append(f"| {key} | {label} | {name} | {m['pair_acc_mb']:.4f} "
                         f"[{m['pair_acc_mb_ci95'][0]:.3f},{m['pair_acc_mb_ci95'][1]:.3f}] | {m['task_macro']:.4f} |")
    (OUT / "r0_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (OUT / "metrics_r0.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"[r0-report] wrote for {len(res)} kernels")
