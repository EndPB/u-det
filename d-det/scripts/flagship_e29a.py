#!/usr/bin/env python
"""E29-A：生成器留出审计（GroupKFold by generator；只读 b_train/b_val）。

依据 `docx/d-det_E29核心方向指导_2026-09-30.md` §3：
  · AI 样本按 generator 分组 5 折 GroupKFold（同 generator 同折）；Human 随机分折（仅检测）；
  · 固定表示 R0=m_raw、R1=μ_E24；每折训练折标准化 (5)（std 下限 1e-2）；
    inner 80/20（家族按 family 分层 / 检测按 y 分层）选 C（{0.01..10} 家族 BalAcc / 检测 AUROC），
    验证折只评估一次；
  · 密度诊断 (8)(9)：λ=20 收缩对角高斯 + log π，与线性读出逐折比较；
  · 对照：样本随机 5 折（同协议；仅参照、不能触发晋级）；
  · 出口预注册（操作化定义见 config.json）：signal_ok ∧ density_win → 触发 E29-B；否则停流形/对比线。

只读 b_train/b_val；不加载 test_seen/unseen 或 E25 四划分特征；test_accessed=false。
产物：artifacts/flagship_e29_a/{config.json, manifest.json, fold_metrics.json,
density_metrics.json, solver.log}（report.md 由报告复制）；smoke → artifacts/flagship_e29_a_smoke。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
import warnings
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (balanced_accuracy_score, f1_score, roc_auc_score)
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from flagship_round1 import SEEN_LABS  # noqa: E402
from flagship_e28_probe import (load_train_val, ids_md5, fit_scaler,  # noqa: E402
                                apply_scaler, export_pass, default_args)

C_GRID = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
STD_FLOOR = 1e-2
MAX_ITER = 10000
TOL = 1e-6
LAM = 20            # 收缩强度（规范 §3.2 固定）
INNER_FRAC = 0.8
LEN_EDGES = [(8, 128), (129, 256), (257, 512), (513, 1024), (1025, 2048)]
R28_FEATS = ROOT / "runs/flagship_e28/features_train_val.npz"


def fit_family_c(Ztr, ytr, itr, iva, log):
    rows = []
    for C in C_GRID:
        t0 = time.perf_counter()
        with warnings.catch_warnings(record=True) as wl:
            warnings.simplefilter("always")
            clf = LogisticRegression(max_iter=MAX_ITER, tol=TOL, C=C).fit(Ztr[itr], ytr[itr])
        bal = float(balanced_accuracy_score(ytr[iva], clf.predict(Ztr[iva])))
        rows.append({"C": C, "inner_bal": round(bal, 4),
                     "converged": not any(issubclass(w.category, ConvergenceWarning)
                                          for w in wl),
                     "sec": round(time.perf_counter() - t0, 1)})
        log.append(f"family inner C={C:<5} inner_bal={bal:.4f} "
                   f"({rows[-1]['sec']}s)")
    best = max(rows, key=lambda r: (r["inner_bal"], -r["C"]))
    return best["C"], rows


def fit_det_c(Ztr, ytr, itr, iva, log):
    rows = []
    for C in C_GRID:
        t0 = time.perf_counter()
        with warnings.catch_warnings(record=True) as wl:
            warnings.simplefilter("always")
            clf = LogisticRegression(max_iter=MAX_ITER, tol=TOL, C=C).fit(Ztr[itr], ytr[itr])
        au = float(roc_auc_score(ytr[iva], clf.predict_proba(Ztr[iva])[:, 1]))
        rows.append({"C": C, "inner_auroc": round(au, 4),
                     "converged": not any(issubclass(w.category, ConvergenceWarning)
                                          for w in wl),
                     "sec": round(time.perf_counter() - t0, 1)})
        log.append(f"det    inner C={C:<5} inner_auroc={au:.4f} "
                   f"({rows[-1]['sec']}s)")
    best = max(rows, key=lambda r: (r["inner_auroc"], -r["C"]))
    return best["C"], rows


def stratified_inner(y, frac, rs):
    itr, iva = [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rs.shuffle(idx)
        cut = max(1, int(round(frac * len(idx)))) if len(idx) > 1 else len(idx)
        itr += idx[:cut].tolist()
        iva += idx[cut:].tolist()
    return np.array(sorted(itr)), np.array(sorted(iva))


def fam_eval(pred, y):
    rec = {}
    for c, lab in enumerate(SEEN_LABS):
        m = y == c
        if int(m.sum()) > 0:
            rec[lab] = round(float((pred[m] == c).mean()), 4)
    return {"acc": round(float((pred == y).mean()), 4),
            "bal": round(float(balanced_accuracy_score(y, pred)), 4),
            "n": int(len(y)), "recall": rec}


def count_map(arr):
    out = {}
    for v in arr:
        out[str(v)] = out.get(str(v), 0) + 1
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    args = default_args(smoke=a.smoke)
    OUT = ROOT / "artifacts" / ("flagship_e29_a_smoke" if a.smoke else "flagship_e29_a")
    if OUT.exists() and any(OUT.iterdir()) and not a.force:
        print(f"[e29a] 已存在产物，默认拒绝覆盖：{OUT}（--force 可覆盖）", flush=True)
        return 2
    OUT.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    solver_log: list[str] = []

    def say(msg):
        print(msg, flush=True)
        solver_log.append(msg)

    commit = subprocess.run(["git", "-C", str(ROOT.parent), "rev-parse", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    say(f"[e29a] commit {commit} | smoke={a.smoke} | device={device}")

    # ---- 语料（只读 b_train/b_val）与特征 ----
    corpus = load_train_val(args)
    docs_all = corpus["train"] + corpus["val"]
    n = len(docs_all)
    y_ai = np.array([d.y_ai for d in docs_all], bool)
    fam = np.array([SEEN_LABS.index(d.fam) if d.fam in SEEN_LABS else -1
                    for d in docs_all], np.int8)
    gen = np.array([d.gen for d in docs_all])
    lang = np.array([d.lang for d in docs_all])
    ntok = np.array([len(d.ids) for d in docs_all], np.int32)
    say(f"[e29a] 语料（合并 b_train+b_val）：n={n}；AI={int(y_ai.sum())}；"
        f"Human={int((~y_ai).sum())}；generators={len(set(gen[y_ai].tolist()))}")

    if a.smoke:
        # 冒烟：自建小特征（E28 导出管线；不校验全量缓存）
        from encoders import build_encoder
        from models import build_model
        import yaml
        from flagship_round1 import SetPool
        ms = torch.load(ROOT / "runs/flagship_e27/model_state.pt", map_location="cpu",
                        weights_only=False)
        with open(ROOT / "configs/ddet_v041.yaml", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        ec = dict(cfg["encoder"]); ec["path"] = str(ROOT / ec["path"])
        dual = build_model("dual", encoder=build_encoder(**ec), dim=768,
                           pool=cfg["model"].get("pool", "mean"),
                           s2_rank=cfg["model"].get("s2_rank", 1))
        dual.load_state_dict(torch.load(ROOT / "runs/v0.4.1_covreg/last.pt",
                                        map_location="cpu",
                                        weights_only=False)["state"], strict=False)
        enc = dual.encoder.to(device).eval()
        head = SetPool(768, 512).to(device)
        head.load_state_dict(torch.load(ROOT / "runs/flagship_r1/head.pt",
                                        map_location=device)["head"])
        head.eval()
        wq = torch.as_tensor(ms["wq"], device=device)
        bq = torch.as_tensor(ms["bq"], device=device)
        m_tT = torch.as_tensor(ms["m_t"], dtype=torch.float32, device=device)
        s_tT = torch.as_tensor(ms["s_t"], dtype=torch.float32, device=device)
        raw_tr, mu_tr, _ = export_pass(enc, head, corpus["train"], args.batch, device,
                                       wq, bq, m_tT, s_tT)
        raw_va, mu_va, _ = export_pass(enc, head, corpus["val"], args.batch, device,
                                       wq, bq, m_tT, s_tT)
        raw = np.vstack([raw_tr, raw_va]); mu = np.vstack([mu_tr, mu_va])
        feats_sha = None
    else:
        feats_sha = hashlib.sha256(R28_FEATS.read_bytes()).hexdigest()
        fz = dict(np.load(R28_FEATS, allow_pickle=True))
        m_tr = np.array([ids_md5(d.ids) for d in corpus["train"]])
        m_va = np.array([ids_md5(d.ids) for d in corpus["val"]])
        assert np.array_equal(m_tr, fz["md5_train"]), "train 清单与 E28 特征不符"
        assert np.array_equal(m_va, fz["md5_val"]), "val 清单与 E28 特征不符"
        assert len(corpus["train"]) == 17600 and len(corpus["val"]) == 5372
        raw = np.vstack([fz["raw_train"], fz["raw_val"]])
        mu = np.vstack([fz["mu_train"], fz["mu_val"]])
        say(f"[e29a] 特征来源：runs/flagship_e28/features_train_val.npz"
            f"（sha256 {feats_sha[:16]}…；md5 清单校验通过）")

    reps = {"R0_mraw": raw, "R1_mu": mu}
    ai_idx = np.where(y_ai)[0]
    hum_idx = np.where(~y_ai)[0]

    # ---- 折划分 ----
    def folds_for(mode):
        folds = np.full(n, -1, np.int64)
        if mode == "generator_holdout":
            gkf = GroupKFold(n_splits=5)
            for k, (_, va) in enumerate(gkf.split(np.zeros(len(ai_idx)),
                                                  groups=gen[ai_idx])):
                folds[ai_idx[va]] = k
        else:
            perm = np.random.RandomState(0).permutation(len(ai_idx))
            for k in range(5):
                folds[ai_idx[perm[k::5]]] = k
        ph = np.random.RandomState(1).permutation(len(hum_idx))   # Human 两种划分共用
        for k in range(5):
            folds[hum_idx[ph[k::5]]] = k
        assert (folds >= 0).all()
        return folds

    folds_modes = {m: folds_for(m) for m in ("generator_holdout", "random_split")}
    # generator 留出性自检
    fh = folds_modes["generator_holdout"]
    gen_leak = 0
    for k in range(5):
        va_g = set(gen[ai_idx[fh[ai_idx] == k]].tolist())
        tr_g = set(gen[ai_idx[fh[ai_idx] != k]].tolist())
        gen_leak += len(va_g & tr_g)
    say(f"[e29a] generator 留出性自检：验证折与训练折 generator 交集总数 = {gen_leak}")
    assert gen_leak == 0

    fam_gen_n = {SEEN_LABS[c]: len({g for g, f in zip(gen[ai_idx], fam[ai_idx])
                                    if f == c}) for c in range(8)}
    say(f"[e29a] 每家族 generator 数：{json.dumps(fam_gen_n, ensure_ascii=False)}")

    # ---- 主循环 ----
    fold_metrics = {"modes": {}, "protocol": {
        "C_GRID": C_GRID, "STD_FLOOR": STD_FLOOR, "LAM": LAM,
        "inner_frac": INNER_FRAC,
        "inner_split": "家族按 family 分层 / 检测按 y 分层；seed=100+f(+200 随机模式)",
        "human_folds": "随机（两种划分共用）", "eval": "验证折只评估一次"}}
    density_metrics = {"modes": {}, "LAM": LAM,
                       "note": "密度 (8)(9)：λ=20 收缩对角高斯；只作线性判别对照"}
    t0 = time.time()
    for mode, folds in folds_modes.items():
        fm = {"folds": []}
        dm = {"folds": []}
        for k in range(5):
            tr = np.where(folds != k)[0]
            va = np.where(folds == k)[0]
            tr_ai = tr[y_ai[tr]]; va_ai = va[y_ai[va]]
            seed_base = (100 if mode == "generator_holdout" else 200) + k
            val_fam_cnt = count_map(fam[va_ai])
            fold_rec = {
                "fold": k,
                "n_train": int(len(tr)), "n_val": int(len(va)),
                "n_ai_train": int(len(tr_ai)), "n_ai_val": int(len(va_ai)),
                "n_human_train": int((~y_ai[tr]).sum()),
                "n_human_val": int((~y_ai[va]).sum()),
                "counts_val": {
                    "family": val_fam_cnt,
                    "generator": count_map(gen[va_ai]),
                    "language": count_map(lang[va_ai]),
                    "length": {f"{lo}-{hi}": int(((ntok[va_ai] >= lo) &
                                                  (ntok[va_ai] <= hi)).sum())
                               for lo, hi in LEN_EDGES}},
                "generators_train": len(set(gen[tr_ai].tolist())),
                "generators_val": len(set(gen[va_ai].tolist())),
                "arms": {}}
            dens_rec = {"fold": k, "arms": {}}
            for arm, X in reps.items():
                X_ai = X[ai_idx]
                # 家族：训练折 AI 标准化 + inner 选 C
                m_f, s_f = fit_scaler(X[tr_ai])
                Ztr_f = apply_scaler(X[tr_ai], m_f, s_f)
                Zva_f = apply_scaler(X[va_ai], m_f, s_f)
                ytr_f, yva_f = fam[tr_ai], fam[va_ai]
                itr, iva = stratified_inner(ytr_f, INNER_FRAC,
                                            np.random.RandomState(seed_base))
                C_F, grid_F = fit_family_c(Ztr_f, ytr_f, itr, iva, solver_log)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    clf_F = LogisticRegression(max_iter=MAX_ITER, tol=TOL,
                                               C=C_F).fit(Ztr_f, ytr_f)
                fam_res = fam_eval(clf_F.predict(Zva_f), yva_f)
                # 检测：训练折全部标准化 + inner 选 C
                m_d, s_d = fit_scaler(X[tr])
                Ztr_d = apply_scaler(X[tr], m_d, s_d)
                Zva_d = apply_scaler(X[va], m_d, s_d)
                ytr_d, yva_d = y_ai[tr], y_ai[va]
                itr2, iva2 = stratified_inner(ytr_d, INNER_FRAC,
                                              np.random.RandomState(seed_base + 50))
                C_D, grid_D = fit_det_c(Ztr_d, ytr_d, itr2, iva2, solver_log)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    clf_D = LogisticRegression(max_iter=MAX_ITER, tol=TOL,
                                               C=C_D).fit(Ztr_d, ytr_d)
                det_p = clf_D.predict_proba(Zva_d)[:, 1]
                det_res = {"auroc": round(float(roc_auc_score(yva_d, det_p)), 4),
                           "f1@.5": round(float(f1_score(yva_d, det_p > 0.5,
                                                         average="macro")), 4),
                           "n": int(len(va))}
                fold_rec["arms"][arm] = {
                    "family": {"C_star": C_F, "inner_grid": grid_F,
                               "val": fam_res},
                    "detection": {"C_star": C_D, "inner_grid": grid_D,
                                  "val": det_res}}
                # 密度诊断（家族标准化空间，训练折 AI）
                v0 = Ztr_f.var(0)
                mus = np.zeros((8, Ztr_f.shape[1])); vhat = np.zeros_like(mus)
                logpi = np.zeros(8)
                for c in range(8):
                    Xc = Ztr_f[ytr_f == c]
                    nc = len(Xc)
                    if nc == 0:
                        vhat[c] = v0; continue
                    mu_c = Xc.mean(0); v_c = Xc.var(0)
                    vhat[c] = np.maximum(nc / (nc + LAM) * v_c
                                         + LAM / (nc + LAM) * v0, 1e-8)
                    mus[c] = mu_c
                    logpi[c] = np.log(nc / len(Ztr_f))
                d2 = (Zva_f[:, None, :] - mus[None]) ** 2 / vhat[None]
                score = -0.5 * (d2.sum(-1) + np.log(vhat).sum(1)[None]) + logpi[None]
                dens_res = fam_eval(score.argmax(1), yva_f)
                dens_rec["arms"][arm] = {
                    "density_bal": dens_res["bal"], "linear_bal": fam_res["bal"],
                    "density_beats_linear": bool(dens_res["bal"] > fam_res["bal"]),
                    "recall": dens_res["recall"]}
                say(f"[e29a] {mode} fold{k} {arm}：fam bal {fam_res['bal']}"
                    f"（C*={C_F}）| det {det_res['auroc']}（C*={C_D}）"
                    f"| dens {dens_res['bal']}"
                    f"{' > lin' if dens_res['bal'] > fam_res['bal'] else ' <= lin'}")
            fm["folds"].append(fold_rec)
            dm["folds"].append(dens_rec)
        # 汇总
        summ = {}
        for arm in reps:
            fb = [f["arms"][arm]["family"]["val"]["bal"] for f in fm["folds"]]
            fa = [f["arms"][arm]["detection"]["val"]["auroc"] for f in fm["folds"]]
            db = [f["arms"][arm]["density_bal"] for f in dm["folds"]]
            wins = sum(1 for f in dm["folds"] if f["arms"][arm]["density_beats_linear"])
            summ[arm] = {"fam_bal_mean": round(float(np.mean(fb)), 4),
                         "fam_bal_std": round(float(np.std(fb)), 4),
                         "fam_bal_folds": fb,
                         "det_auroc_mean": round(float(np.mean(fa)), 4),
                         "det_auroc_std": round(float(np.std(fa)), 4),
                         "det_auroc_folds": fa,
                         "density_bal_mean": round(float(np.mean(db)), 4),
                         "density_beats_linear_folds": wins}
        fm["summary"] = summ
        density_metrics["modes"][mode] = {"folds": dm["folds"], "summary": summ}
        fold_metrics["modes"][mode] = fm

    t_solver = (time.time() - t0) / 60
    say(f"[e29a] 求解耗时 {t_solver:.1f} min")

    # ---- 出口判定（操作化定义，写入 config） ----
    g = fold_metrics["modes"]["generator_holdout"]["summary"]
    r = fold_metrics["modes"]["random_split"]["summary"]
    sig_ok = bool(min(g["R0_mraw"]["fam_bal_mean"],
                      g["R1_mu"]["fam_bal_mean"]) >= 0.18)
    dens_win = {arm: g[arm]["density_beats_linear_folds"] >= 3 for arm in reps}
    dens_ok = bool(any(dens_win.values()))
    e29b = bool(sig_ok and dens_ok)
    exit_info = {
        "signal_ok_gen_bal>=0.18(两臂取小)": sig_ok,
        "density_majority_win(folds>=3)": dens_win,
        "density_ok": dens_ok, "e29b_triggered": e29b,
        "gen_vs_random": {arm: {
            "gen_fam_bal": g[arm]["fam_bal_mean"],
            "rand_fam_bal": r[arm]["fam_bal_mean"],
            "gen_det": g[arm]["det_auroc_mean"],
            "rand_det": r[arm]["det_auroc_mean"]} for arm in reps},
        "rule_note": "操作化：signal_ok=min(gen 家族 bal 均值)≥0.18；density_ok=任一臂密度多数折"
                     "(≥3/5)胜线性；两者同时→触发 E29-B；只依据 generator-held-out，不因随机切分触发。",
    }
    say(f"[e29a] 出口：signal_ok={sig_ok} density_ok={dens_ok} → E29-B 触发={e29b}")

    # ---- 产物 ----
    key_lines = "\n".join(f"{d.split}:{d.row}:{ids_md5(d.ids)}" for d in docs_all)
    manifest = {"commit": commit, "smoke": a.smoke,
                "features_source": str(R28_FEATS) if not a.smoke else "smoke-export",
                "features_sha256": feats_sha,
                "input_manifest_sha256": hashlib.sha256(key_lines.encode()).hexdigest(),
                "counts": {"n": n, "ai": int(y_ai.sum()), "human": int((~y_ai).sum()),
                           "generators_ai": len(set(gen[y_ai].tolist())),
                           "families": {SEEN_LABS[c]: int((fam == c).sum())
                                        for c in range(8)}},
                "family_generator_counts": fam_gen_n,
                "folds": {mode: {"val_generators": {
                    str(f["fold"]): sorted(set(gen[ai_idx[folds_modes[mode][ai_idx]
                                                    == f["fold"]]].tolist()))
                    for f in fold_metrics["modes"][mode]["folds"]}}
                    for mode in folds_modes},
                "dims": {"R0": 768, "R1": 768}, "C_GRID": C_GRID, "LAM": LAM,
                "tau": None, "seed": {"ai_random": 0, "human": 1},
                "test_accessed": False, "unseen_accessed": False}
    (OUT / "fold_metrics.json").write_text(json.dumps(fold_metrics, ensure_ascii=False,
                                                      indent=1))
    (OUT / "density_metrics.json").write_text(json.dumps(density_metrics,
                                                         ensure_ascii=False, indent=1))
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    (OUT / "config.json").write_text(json.dumps({
        "args": vars(args), "protocol": fold_metrics["protocol"],
        "exit_rule": exit_info, "commit": commit,
        "notes": ["只读 b_train/b_val；Human 仅用于检测；留出家族过滤",
                  "R0=m_raw、R1=μ（E24 ht 均值）；不引入 s_top/logv/rF/能量",
                  "密度=λ20 收缩对角高斯 (8)(9)，非完整 DMHM"]}, ensure_ascii=False,
        indent=1))
    (OUT / "solver.log").write_text("\n".join(solver_log) + "\n")
    say(f"[e29a] 完成 {time.time()-t_start:.1f}s；产物 {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
