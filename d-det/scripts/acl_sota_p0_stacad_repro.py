#!/usr/bin/env python
"""ACL SOTA P0：STACAD 官方协议复现（公开基线复现；纯 CPU 阶段）。

复现对象 = STACAD v2 replication package（data/stacad_v2/extracted/STACAD-v2）
随包发布的 journal_v2 结果；口径与官方 scripts/v2_pipeline.py 完全一致：

- classic ：LogReg(StandardScaler, C=1, balanced) / RF(500, min_samples_leaf=2,
             balanced_subsample, seed 42) —— 官方 exp_classic
- learners：XGBoost（protocol.json 调参后配置，ES=训练划分 10% 文件，ES 50）/
             LightGBM（同参映射）/ MLP(256,128)；CatBoost 未安装 → 跳过并引用官方
- stack   ：官方 exp_stack 的 meta 层公式（LR C=1 交叉拟合）。本脚本用随包发布的
             learner_oof_proba.npz / views_oof.npz / codebert_oof_proba.npz 重跑 meta
             （meta-only 复现），另用"本机复现的 xgb/lgb OOF + 官方 cb OOF"跑
             self-consistent 变体，检验端到端一致性

协议（与官方一致）：SEED=42；5 折确定文件级 GroupKFold（assign_folds，=sklearn
GroupKFold 稳定排序等价实现）；早停集 = 训练划分文件级 10%（default_rng(42+f)）；
指标 = 逐折 macro-F1 的 mean±std + ECE(15 bins)/NLL/Brier。

不写入 package 目录；输出到 artifacts/acl_sota_p0/stacad_official_repro[_smoke]/。

用法：
  OMP_NUM_THREADS=8 python scripts/acl_sota_p0_stacad_repro.py --stages classic learners stack
  python scripts/acl_sota_p0_stacad_repro.py --smoke   # 单折小预算冒烟
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "data" / "stacad_v2" / "extracted" / "STACAD-v2"
FEAT = PKG / "cache" / "corpus_v2" / "features_v2.npz"
OFFICIAL = PKG / "cache" / "results" / "journal_v2"
OUT_BASE = ROOT / "artifacts" / "acl_sota_p0"

SEED = 42
K_FOLDS = 5
CLASS_NAMES = ["gemini_3_flash", "gpt5_nano", "claude_3_haiku", "qwen3_coder_30b",
               "deepseek_v3_2", "grok_4_fast", "devstral_2512"]
LANGS_OFFICIAL = ["py", "c", "cpp", "java", "cs", "go", "php"]


def log(msg: str) -> None:
    print(f"[stacad-repro] {msg}", flush=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------ 官方口径复刻
def load_data():
    z = np.load(FEAT, allow_pickle=True)
    return (z["X"].astype(np.float32), z["y"].astype(int),
            z["groups"].astype(str), z["langs"].astype(str), z["folds"].astype(int))


def assign_folds(groups, k=K_FOLDS):
    """官方 corpus_v2.assign_folds：文件按 (对数 desc, 名称 desc) 排序后依次给最轻折。"""
    sizes = Counter(groups.tolist())
    load = [0] * k
    fold = {}
    for name in sorted(sizes, key=lambda f: (sizes[f], f))[::-1]:
        j = min(range(k), key=lambda i: load[i])
        load[j] += sizes[name]
        fold[name] = j
    return np.array([fold[g] for g in groups])


def protocol():
    return json.load(open(OFFICIAL / "protocol.json"))


def xgb_params(num_class, proto):
    base = {"objective": "multi:softprob", "num_class": num_class, "max_depth": 8,
            "learning_rate": 0.05, "n_estimators": 3000, "subsample": 0.8,
            "colsample_bytree": 0.7, "min_child_weight": 10, "reg_alpha": 0.1,
            "reg_lambda": 1.0, "tree_method": "hist", "random_state": SEED,
            "verbosity": 0, "n_jobs": 8, "eval_metric": "mlogloss",
            "early_stopping_rounds": 50}
    base.update(proto["xgb"])
    return base


def sample_weights(y, scheme):
    y = np.asarray(y).astype(int)
    if scheme == "none":
        return np.ones(len(y), dtype=np.float32)
    counts = np.bincount(y, minlength=int(y.max()) + 1).astype(float)
    w = len(y) / (np.count_nonzero(counts) * np.maximum(counts, 1))
    return w[y].astype(np.float32)


def es_split(n, groups_train, seed, frac=0.1):
    rng = np.random.default_rng(seed)
    files = np.unique(np.asarray(groups_train).astype(str))
    rng.shuffle(files)
    val_files = set(files[: int(frac * len(files))].tolist())
    mask = np.isin(np.asarray(groups_train).astype(str), list(val_files))
    return np.where(~mask)[0], np.where(mask)[0]


def metrics(prob, y, fold):
    from sklearn.metrics import f1_score
    pred = prob.argmax(1)
    conf = prob.max(1)
    correct = (pred == y).astype(float)
    bins = np.clip(np.digitize(conf, np.linspace(0, 1, 16)) - 1, 0, 14)
    ece = sum(np.mean(bins == b) * abs(correct[bins == b].mean() - conf[bins == b].mean())
              for b in range(15) if (bins == b).any())
    fa = [float((pred[fold == f] == y[fold == f]).mean()) for f in range(K_FOLDS)]
    ff = [float(f1_score(y[fold == f], pred[fold == f], average="macro")) for f in range(K_FOLDS)]
    return {"accuracy_mean": float(np.mean(fa)), "accuracy_std": float(np.std(fa)),
            "macro_f1_mean": float(np.mean(ff)), "macro_f1_std": float(np.std(ff)),
            "fold_accuracy": fa, "fold_f1_macro": ff, "ece": float(ece),
            "nll": float(-np.mean(np.log(np.clip(prob[np.arange(len(y)), y], 1e-12, 1)))),
            "brier": float(np.mean(np.sum((prob - np.eye(7)[y]) ** 2, axis=1)))}


# ------------------------------------------------------------------ stages
def stage_classic(X, y, groups, fold, out_dir, smoke=False):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    P = {"logreg": np.zeros((len(y), 7)), "rf": np.zeros((len(y), 7))}
    n_folds = 1 if smoke else K_FOLDS
    for f in range(n_folds):
        tr = np.where(fold != f)[0]
        te = np.where(fold == f)[0]
        if smoke:
            tr = tr[: 20000]
        t0 = time.time()
        sc = StandardScaler().fit(X[tr])
        lr = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced").fit(
            sc.transform(X[tr]), y[tr])
        P["logreg"][te] = lr.predict_proba(sc.transform(X[te]))
        log(f"classic fold {f}: logreg {time.time() - t0:.0f}s")
        t0 = time.time()
        rf = RandomForestClassifier(n_estimators=50 if smoke else 500, min_samples_leaf=2,
                                    class_weight="balanced_subsample", n_jobs=8,
                                    random_state=SEED).fit(X[tr], y[tr])
        P["rf"][te] = rf.predict_proba(X[te])
        log(f"classic fold {f}: rf {time.time() - t0:.0f}s")
    np.savez_compressed(out_dir / "classic_oof_repro.npz", **P, y=y, fold=fold)
    out = {k: metrics(P[k], y, fold) for k in P}
    return out


def stage_learners(X, y, groups, fold, out_dir, smoke=False, skip_mlp=False):
    import lightgbm as lgb
    import xgboost as xgb
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    proto = protocol()
    xp = xgb_params(7, proto)
    sw = sample_weights(y, proto["class_weights"])
    keys = ["xgb", "lgb"] + ([] if skip_mlp else ["mlp"])
    P = {k: np.zeros((len(y), 7)) for k in keys}
    rounds = {k: [] for k in ("xgb", "lgb")}
    n_folds = 1 if smoke else K_FOLDS
    for f in range(n_folds):
        tr = np.where(fold != f)[0]
        te = np.where(fold == f)[0]
        if smoke:
            tr = tr[: 20000]
        a_, v_ = es_split(len(tr), groups[tr], seed=SEED + f)
        a, v = tr[a_], tr[v_]
        log(f"learners fold {f}: fit {len(a):,} es {len(v):,} test {len(te):,}")
        t0 = time.time()
        p = dict(xp)
        if smoke:
            p["n_estimators"] = 200
        m1 = xgb.XGBClassifier(**p)
        m1.fit(X[a], y[a], sample_weight=None if proto["class_weights"] == "none" else sw[a],
               eval_set=[(X[v], y[v])], verbose=False)
        P["xgb"][te] = m1.predict_proba(X[te])
        rounds["xgb"].append(int(m1.best_iteration))
        log(f"learners fold {f}: xgb {time.time() - t0:.0f}s rounds={rounds['xgb'][-1]}")
        t0 = time.time()
        m2 = lgb.LGBMClassifier(objective="multiclass", num_class=7,
                                max_depth=max(10, xp["max_depth"]),
                                num_leaves=2 ** min(xp["max_depth"], 8) - 1,
                                learning_rate=xp["learning_rate"],
                                n_estimators=200 if smoke else 3000,
                                subsample=xp["subsample"], subsample_freq=1,
                                colsample_bytree=xp["colsample_bytree"],
                                min_child_samples=20, reg_alpha=xp["reg_alpha"],
                                reg_lambda=xp["reg_lambda"], random_state=SEED,
                                verbose=-1, n_jobs=8)
        m2.fit(X[a], y[a], sample_weight=None if proto["class_weights"] == "none" else sw[a],
               eval_set=[(X[v], y[v])],
               callbacks=[lgb.early_stopping(50 if not smoke else 20, verbose=False)])
        P["lgb"][te] = m2.predict_proba(X[te])
        rounds["lgb"].append(int(m2.best_iteration_))
        log(f"learners fold {f}: lgb {time.time() - t0:.0f}s rounds={rounds['lgb'][-1]}")
        if "mlp" in P:
            t0 = time.time()
            sc = StandardScaler().fit(X[tr])
            m4 = MLPClassifier(hidden_layer_sizes=(256, 128), early_stopping=True,
                               validation_fraction=0.1, max_iter=20 if smoke else 200,
                               random_state=SEED)
            m4.fit(sc.transform(X[tr]), y[tr])
            P["mlp"][te] = m4.predict_proba(sc.transform(X[te]))
            log(f"learners fold {f}: mlp {time.time() - t0:.0f}s")
    np.savez_compressed(out_dir / "learner_oof_repro.npz", **P, y=y, fold=fold)
    out = {k: metrics(P[k], y, fold) for k in P}
    out["rounds"] = rounds
    return out


def stage_stack(y, fold, langs, out_dir, smoke=False):
    """官方 exp_stack 的 meta 层复现（不训练任何基学习器）。

    - meta_only   ：基学习器 OOF 全部来自随包发布文件（xgb/lgb/cb；views；codebert）
    - self_consist：xgb/lgb 用本机复现 OOF，cb 保持官方 OOF
    """
    from sklearn.linear_model import LogisticRegression
    zl = np.load(OFFICIAL / "learner_oof_proba.npz")
    zv = np.load(OFFICIAL / "views_oof.npz")
    zc = np.load(OFFICIAL / "codebert_oof_proba.npz")["proba"]
    assert np.array_equal(zl["y"], y) and np.array_equal(zl["fold"], fold)
    L = np.stack([(langs == l).astype(float) for l in LANGS_OFFICIAL], 1)

    def logit(p):
        return np.log(np.clip(p, 1e-6, 1))

    results = {}
    for mode in (["meta_only", "self_consist"] if (out_dir / "learner_oof_repro.npz").exists()
                 else ["meta_only"]):
        if mode == "self_consist":
            rz = np.load(out_dir / "learner_oof_repro.npz")
            P = {"xgb": np.asarray(rz["xgb"], dtype=np.float64),
                 "lgb": np.asarray(rz["lgb"], dtype=np.float64),
                 "cb": zl["cb"]}
        else:
            P = {k: zl[k] for k in ("xgb", "lgb", "cb")}
        trees = [logit(P["xgb"]), logit(P["lgb"]), logit(P["cb"])]
        cands = {"vote_0.50_0.30_0.20": ("vote", None),
                 "vote_equal": ("vote_eq", None),
                 "stack_trees": ("lr", np.hstack(trees)),
                 "stack_trees_tfidf": ("lr", np.hstack(trees + [zv["diff"], zv["output"]])),
                 "stack_trees_tfidf_lang": ("lr", np.hstack(trees + [zv["diff"], zv["output"], L])),
                 "stack_trees_codebert": ("lr", np.hstack(trees + [logit(zc)])),
                 "stack_trees_tfidf_codebert": ("lr", np.hstack(trees + [zv["diff"], zv["output"], logit(zc)])),
                 "stack_trees_tfidf_lang_codebert": ("lr", np.hstack(trees + [zv["diff"], zv["output"], L, logit(zc)]))}
        probs, out = {}, {}
        n_folds = 1 if smoke else K_FOLDS
        for name, (kind, Z) in cands.items():
            if kind == "vote":
                prob = 0.5 * P["xgb"] + 0.3 * P["lgb"] + 0.2 * P["cb"]
            elif kind == "vote_eq":
                prob = (P["xgb"] + P["lgb"] + P["cb"]) / 3
            else:
                prob = np.zeros((len(y), 7))
                for f in range(n_folds):
                    sel = fold != f
                    hold = fold == f
                    meta = LogisticRegression(max_iter=3000, C=1.0).fit(Z[sel], y[sel])
                    prob[hold] = meta.predict_proba(Z[hold])
            probs[name] = prob
            out[name] = metrics(prob, y, fold)
            log(f"stack[{mode}] {name}: F1 {out[name]['macro_f1_mean']:.4f}")
        order = ["vote_0.50_0.30_0.20", "vote_equal", "stack_trees",
                 "stack_trees_tfidf", "stack_trees_tfidf_lang"]
        best = max(out[n]["macro_f1_mean"] for n in order)
        main_name = next(n for n in order if out[n]["macro_f1_mean"] >= best - 0.001)
        out["canonical_cpu"] = main_name
        order_cb = ["stack_trees_codebert", "stack_trees_tfidf_codebert",
                    "stack_trees_tfidf_lang_codebert"]
        best_cb = max(out[n]["macro_f1_mean"] for n in order_cb)
        full = next(n for n in order_cb if out[n]["macro_f1_mean"] >= best_cb - 0.001)
        out["canonical"] = full
        np.savez_compressed(out_dir / f"stack_probs_{mode}.npz",
                            **{k: v.astype(np.float16) for k, v in probs.items()}, y=y, fold=fold)
        results[mode] = out
    return results


def stage_merge(out_dir):
    """把分阶段运行（classic/learners/stack）的缓存 npz 合并成完整的 metrics_repro.json。

    分阶段运行会互相覆盖 metrics_repro.json，本阶段从 OOF 缓存重建全部行（秒级）。
    """
    import re
    res = {}
    cf = out_dir / "classic_oof_repro.npz"
    if cf.exists():
        z = np.load(cf)
        y, fold = z["y"], z["fold"]
        res["classic"] = {k: metrics(np.asarray(z[k], dtype=np.float64), y, fold)
                          for k in ("logreg", "rf") if k in z.files}
    lf = out_dir / "learner_oof_repro.npz"
    if lf.exists():
        z = np.load(lf)
        y, fold = z["y"], z["fold"]
        res["learners"] = {k: metrics(np.asarray(z[k], dtype=np.float64), y, fold)
                           for k in z.files if k not in ("y", "fold")}
        rounds = {"xgb": [], "lgb": []}
        logp = Path("/tmp/p0_stacad_learners.log")
        if logp.exists():
            for m in re.finditer(r"fold \d+: (xgb|lgb) \d+s rounds=(\d+)", logp.read_text()):
                rounds[m.group(1)].append(int(m.group(2)))
        if rounds["xgb"] or rounds["lgb"]:
            res["learners"]["rounds"] = rounds
    res["stack"] = {}
    for mode in ("meta_only", "self_consist"):
        f2 = out_dir / f"stack_probs_{mode}.npz"
        if not f2.exists():
            continue
        z2 = np.load(f2)
        y2, fold2 = z2["y"], z2["fold"]
        for k in z2.files:
            if k in ("y", "fold"):
                continue
            key = k if mode == "meta_only" else f"self_consist::{k}"
            res["stack"][key] = metrics(np.asarray(z2[k], dtype=np.float64), y2, fold2)
    return res


def stage_compare(out_dir, repro, smoke=False):
    cmp = {}
    for name, off_file in [("classic", "classic.json"), ("learners", "learners.json"),
                           ("stack", "stack.json")]:
        if name not in repro:
            continue
        off = json.load(open(OFFICIAL / off_file))
        rows = {}
        for k, v in repro[name].items():
            if name == "stack":
                offv = off.get(k)
            else:
                offv = off.get(k)
            if not isinstance(v, dict) or "macro_f1_mean" not in v or not isinstance(offv, dict):
                continue
            rows[k] = {"repro_f1": v["macro_f1_mean"], "official_f1": offv.get("macro_f1_mean"),
                       "delta": (v["macro_f1_mean"] - offv["macro_f1_mean"]) if offv.get("macro_f1_mean") is not None else None}
        cmp[name] = rows
    json.dump(cmp, open(out_dir / "compare_vs_official.json", "w"), indent=2)
    for stage, rows in cmp.items():
        for k, r in rows.items():
            d = f"{r['delta']:+.4f}" if r["delta"] is not None else "n/a"
            log(f"compare[{stage}] {k}: repro {r['repro_f1']:.4f} vs official {r['official_f1']} (Δ {d})")
    return cmp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", nargs="+", default=["classic", "learners", "stack"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--skip-mlp", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    tag = "stacad_official_repro_smoke" if args.smoke else "stacad_official_repro"
    out_dir = OUT_BASE / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    assert FEAT.exists(), f"missing {FEAT}"
    X, y, groups, langs, folds_shipped = load_data()
    fold = assign_folds(groups)
    assert np.array_equal(fold, folds_shipped), "assign_folds 与随包 folds 不一致"
    log(f"data {X.shape}; fold sizes {Counter(fold.tolist())}")
    log(f"classes {Counter(y.tolist())}; langs {Counter(langs.tolist())}")

    cfg = {"script": "scripts/acl_sota_p0_stacad_repro.py", "stages": args.stages,
           "smoke": args.smoke, "seed": SEED, "k_folds": K_FOLDS, "class_names": CLASS_NAMES,
           "official_numbers_source": "STACAD-v2/cache/results/journal_v2/*.json",
           "protocol": protocol(), "catboost": "未安装：cb 行与含 cb 的 stack 基学习器引用官方 OOF 文件",
           "features_sha256": sha256(FEAT), "protocol_sha256": sha256(OFFICIAL / "protocol.json")}
    json.dump(cfg, open(out_dir / "config.json", "w"), indent=2)
    env = {"python": sys.version.split()[0], "platform": platform.platform(),
           "numpy": np.__version__, "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
           "note": "纯 CPU 阶段；OMP 放宽到 8 与官方 8 核条件一致（神经实验的 OMP=2 规则不适用于此）"}
    try:
        import sklearn
        env["sklearn"] = sklearn.__version__
        import lightgbm
        env["lightgbm"] = lightgbm.__version__
        import xgboost
        env["xgboost"] = xgboost.__version__
    except Exception as e:  # pragma: no cover
        env["import_error"] = repr(e)
    json.dump(env, open(out_dir / "env.json", "w"), indent=2)

    repro = {}
    for stage in args.stages:
        t1 = time.time()
        if stage == "classic":
            repro["classic"] = stage_classic(X, y, groups, fold, out_dir, args.smoke)
        elif stage == "learners":
            repro["learners"] = stage_learners(X, y, groups, fold, out_dir, args.smoke,
                                               skip_mlp=args.skip_mlp)
        elif stage == "stack":
            repro["stack"] = stage_stack(y, fold, langs, out_dir, args.smoke)
        elif stage == "merge":
            repro.update(stage_merge(out_dir))
        else:
            raise SystemExit(f"unknown stage {stage}")
        log(f"stage {stage} done in {(time.time() - t1) / 60:.1f} min")

    mpath = out_dir / f"metrics_repro{'_smoke' if args.smoke else ''}.json"
    existing = json.load(open(mpath)) if mpath.exists() else {}
    existing.update(repro)
    json.dump(existing, open(mpath, "w"), indent=2)
    stage_compare(out_dir, existing, args.smoke)
    log(f"ALL DONE in {(time.time() - t0) / 60:.1f} min -> {out_dir}")


if __name__ == "__main__":
    main()
