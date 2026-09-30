#!/usr/bin/env python
"""E30：H1 最小实验——family-conditional 两折 generator 留出（只读 b_train/b_val）。

依据 `docx/d-det_ACL核心假设与E30协议修正_2026-09-30.md` §4：
  · 闭集=有 ≥2 generator 的七家族（排除 OpenAI 单 generator）；"七家族"=实验室集合；
  · 每家族固定种子打乱 generator：2-gen→A1/B1；3-gen→A1/B2；5-gen→A2/B3；
    fold0：val=A / train=B；fold1 互补；每家族两侧均有 generator 且完全不相交（断言）；
  · 表示 R0=raw（主）、R1=μ（次要，标注：使用 E24 监督头，不能宣称完整 pipeline 的新
    generator 泛化）；标准化仅训练折（std 下限 1e-2）；分类器固定 **C=0.1**（不做 inner tuning）；
  · 随机控制：同一七家族、每家族与 gen 折相同样本数、同一分类器（家族样本池随机二堆）；
  · 指标：BA_F（家族宏平均 recall）、BA_G（家族内 generator 宏平均再对家族平均）、检测 AUROC
    （同折；Human 随机二折，两模式共用）+ 与随机控制差值；机会水平 1/7；
  · 出口：BA_F 与 BA_G 在 R0、R1 上都 > 随机控制且两折方向一致；
  · 密度诊断（λ=20 对角收缩）：缺失类 score=−∞、方差下限 1e-8，仅诊断、不参与出口。

产物：artifacts/flagship_e30/{config,manifest,metrics,density_metrics,predictions.npz,
solver.log}（report.md 由报告复制）；smoke → artifacts/flagship_e30_smoke；默认拒绝覆盖。
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
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from flagship_round1 import SEEN_LABS  # noqa: E402
from flagship_e28_probe import (load_train_val, ids_md5, fit_scaler,  # noqa: E402
                                apply_scaler, export_pass, default_args)

C_FIXED = 0.1
STD_FLOOR = 1e-2
MAX_ITER = 10000
TOL = 1e-6
LAM = 20
CHANCE = 1.0 / 7.0
R28_FEATS = ROOT / "runs/flagship_e28/features_train_val.npz"
FAM7 = [f for f in SEEN_LABS if f != "OpenAI"]          # 七家族（闭集候选）
FAM7_IDX = np.array([SEEN_LABS.index(f) for f in FAM7])  # 原 8 类索引
LUT7 = {int(SEEN_LABS.index(f)): k for k, f in enumerate(FAM7)}  # 8类索引→7类索引


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    args = default_args(smoke=a.smoke)
    OUT = ROOT / "artifacts" / ("flagship_e30_smoke" if a.smoke else "flagship_e30")
    if OUT.exists() and any(OUT.iterdir()) and not a.force:
        print(f"[e30] 已存在产物，默认拒绝覆盖：{OUT}（--force 可覆盖）", flush=True)
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
    say(f"[e30] commit {commit} | smoke={a.smoke} | device={device}")

    # ---- 语料（只读 b_train/b_val）与特征 ----
    corpus = load_train_val(args)
    docs_all = corpus["train"] + corpus["val"]
    n = len(docs_all)
    y_ai = np.array([d.y_ai for d in docs_all], bool)
    fam = np.array([SEEN_LABS.index(d.fam) if d.fam in SEEN_LABS else -1
                    for d in docs_all], np.int16)
    gen = np.array([d.gen for d in docs_all])
    lang = np.array([d.lang for d in docs_all])
    ntok = np.array([len(d.ids) for d in docs_all], np.int32)
    sp = np.array([d.split for d in docs_all])
    rows = np.array([d.row for d in docs_all], np.int64)

    if a.smoke:
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
        say(f"[e30] smoke：自建特征 AI={int(y_ai.sum())} / Human={int((~y_ai).sum())}")
    else:
        feats_sha = hashlib.sha256(R28_FEATS.read_bytes()).hexdigest()
        fz = dict(np.load(R28_FEATS, allow_pickle=True))
        m_tr = np.array([ids_md5(d.ids) for d in corpus["train"]])
        m_va = np.array([ids_md5(d.ids) for d in corpus["val"]])
        assert np.array_equal(m_tr, fz["md5_train"]) and \
            np.array_equal(m_va, fz["md5_val"]), "清单与 E28 特征不符"
        assert len(corpus["train"]) == 17600 and len(corpus["val"]) == 5372
        raw = np.vstack([fz["raw_train"], fz["raw_val"]])
        mu = np.vstack([fz["mu_train"], fz["mu_val"]])
        say(f"[e30] 特征来源：runs/flagship_e28/features_train_val.npz（sha256 "
            f"{feats_sha[:16]}…；md5 校验通过）")

    reps = {"R0_mraw": raw, "R1_mu": mu}
    ai7_idx = np.array([i for i in np.where(y_ai)[0] if fam[i] in LUT7])
    hum_idx = np.where(~y_ai)[0]
    say(f"[e30] 七家族 AI 样本 {len(ai7_idx)}；Human {len(hum_idx)}；"
        f"generators（七家族）={len(set(gen[ai7_idx].tolist()))}")

    # ---- 家族 generator 分配（A/B） ----
    gen_split = {}
    for f in FAM7:
        fi = SEEN_LABS.index(f)
        gens = sorted({str(gen[i]) for i in ai7_idx if fam[i] == fi})
        rs = np.random.RandomState(1000 + fi)
        rs.shuffle(gens)
        m = len(gens)
        if a.smoke and m < 2:
            say(f"[e30] smoke：家族 {f} 仅 {m} 个 generator，跳过（全量不会发生）")
            continue
        if m == 2:
            nA = 1
        elif m == 3:
            nA = 1
        elif m == 5:
            nA = 2
        else:
            nA = max(1, m // 2)
        A, B = gens[:nA], gens[nA:]
        assert len(A) >= 1 and len(B) >= 1, f"{f} 分配后空侧"
        gen_split[f] = {"A": A, "B": B}
        say(f"[e30] {f}: {m} gens → fold0-val A={A} / fold1-val B={B}")
    fams_used = list(gen_split.keys())
    assert len(fams_used) == 7 or a.smoke

    def fold_ai_masks(fold):        # fold0: val=A；fold1: val=B
        sel = {f: set(gen_split[f]["A" if fold == 0 else "B"]) for f in fams_used}
        is_val = np.array([str(gen[i]) in sel[SEEN_LABS[int(fam[i])]]
                           for i in ai7_idx])
        return (~is_val), is_val    # (train_mask, val_mask)

    # ---- Human 二折（RandomState(1)；两模式共用） ----
    ph = np.random.RandomState(1).permutation(len(hum_idx))
    half = len(hum_idx) // 2
    hum_fold_val = {0: ph[:half], 1: ph[half:]}
    hum_fold_tr = {0: ph[half:], 1: ph[:half]}

    # ---- 划分构建（gen / random） ----
    def build_partition(mode, fold, tr_gen, va_gen):
        """返回 (tr_idx, va_idx)（全样本索引；AI 部分为七家族）。"""
        if mode == "gen":
            atr = ai7_idx[tr_gen]; ava = ai7_idx[va_gen]
        else:  # random：每家族按 gen 折相同的样本数随机二堆
            atr_l, ava_l = [], []
            for f in fams_used:
                fi = SEEN_LABS.index(f)
                fidx = np.array([i for i in ai7_idx if fam[i] == fi])
                n_va = int(va_gen[np.isin(ai7_idx, fidx)].sum())
                rs = np.random.RandomState(200 + fold * 10 + fi)
                perm = rs.permutation(len(fidx))
                ava_l += fidx[perm[:n_va]].tolist()
                atr_l += fidx[perm[n_va:]].tolist()
            atr, ava = np.array(sorted(atr_l)), np.array(sorted(ava_l))
        htr = hum_idx[hum_fold_tr[fold]]; hva = hum_idx[hum_fold_val[fold]]
        return np.sort(np.concatenate([atr, htr])), np.sort(np.concatenate([ava, hva])), atr, ava

    # ---- 主循环 ----
    metrics = {"commit": commit, "smoke": a.smoke, "test_accessed": False,
               "chance": round(CHANCE, 4), "C_fixed": C_FIXED,
               "families": fams_used,
               "excluded": ["OpenAI（单 generator，不进入 H1 归因折）"],
               "splits": {}, "arms": {}, "density": {}, "exit": {}}
    preds_out = {}
    t0 = time.time()
    for fold in (0, 1):
        tr_m, va_m = fold_ai_masks(fold)
        fold_info = {"fold_generators": {f: {"val": sorted(gen_split[f]["A" if fold == 0
                                                                  else "B"]),
                                             "train": sorted(gen_split[f]["B" if fold == 0
                                                                          else "A"])}
                                         for f in fams_used},
                     "n_ai_train": int(tr_m.sum()), "n_ai_val": int(va_m.sum()),
                     "per_family_n_val": {}, "per_family_n_train": {}}
        for f in fams_used:
            fi = SEEN_LABS.index(f)
            fold_info["per_family_n_val"][f] = int(((fam[ai7_idx] == fi) & va_m).sum())
            fold_info["per_family_n_train"][f] = int(((fam[ai7_idx] == fi) & tr_m).sum())
        # 断言：每家族两侧非空、generator 不相交
        for f in fams_used:
            vi = fam[ai7_idx][va_m]; ti = fam[ai7_idx][tr_m]
            fi = SEEN_LABS.index(f)
            assert int(((vi == fi).sum())) > 0 and int(((ti == fi).sum())) > 0
            gv = set(gen[ai7_idx[va_m]][vi == fi].tolist())
            gt = set(gen[ai7_idx[tr_m]][ti == fi].tolist())
            assert len(gv & gt) == 0, f"{f} 两侧 generator 相交"
        metrics["splits"][f"fold{fold}"] = fold_info
        say(f"[e30] fold{fold}: AI train {int(tr_m.sum())} / val {int(va_m.sum())}"
            f"；逐族 n_val {fold_info['per_family_n_val']}")

        for mode in ("gen", "random"):
            for arm, X in reps.items():
                key = f"{arm}|{mode}|f{fold}"
                tr_all, va_all, atr, ava = build_partition(mode, fold, tr_m, va_m)
                # 家族数字标签（7 类）
                ytr = np.array([LUT7[int(fam[i])] for i in atr], np.int64)
                yva = np.array([LUT7[int(fam[i])] for i in ava], np.int64)
                m_f, s_f = fit_scaler(X[atr])
                Ztr = apply_scaler(X[atr], m_f, s_f)
                Zva = apply_scaler(X[ava], m_f, s_f)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    clf = LogisticRegression(max_iter=MAX_ITER, tol=TOL,
                                             C=C_FIXED).fit(Ztr, ytr)
                pred = clf.predict(Zva)
                # BA_F / BA_G
                rec_f, rec_g = {}, {}
                for k, f in enumerate(fams_used):
                    m = yva == k
                    rec_f[f] = round(float((pred[m] == k).mean()), 4)
                    gv = defaultdict(list)
                    for j in np.where(m)[0]:
                        gv[str(gen[ava[j]])].append(float(pred[j] == k))
                    rec_g[f] = {g: round(float(np.mean(v)), 4) for g, v in gv.items()}
                BA_F = float(np.mean([rec_f[f] for f in fams_used]))
                BA_G = float(np.mean([np.mean(list(rec_g[f].values()))
                                      for f in fams_used]))
                # 检测（同折）
                m_d, s_d = fit_scaler(X[tr_all])
                Zd_tr = apply_scaler(X[tr_all], m_d, s_d)
                Zd_va = apply_scaler(X[va_all], m_d, s_d)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    clf_d = LogisticRegression(max_iter=MAX_ITER, tol=TOL,
                                               C=C_FIXED).fit(Zd_tr, y_ai[tr_all])
                p_d = clf_d.predict_proba(Zd_va)[:, 1]
                auroc = float(roc_auc_score(y_ai[va_all], p_d))
                f1 = float(f1_score(y_ai[va_all], p_d > 0.5, average="macro"))
                metrics["arms"].setdefault(arm, {}).setdefault(mode, {})[f"fold{fold}"] = {
                    "BA_F": round(BA_F, 4), "BA_G": round(BA_G, 4),
                    "per_family_recall": rec_f, "per_generator_recall": rec_g,
                    "det_auroc": round(auroc, 4), "det_f1@.5": round(f1, 4),
                    "n_ai_train": int(len(atr)), "n_ai_val": int(len(ava)),
                    "n_train_all": int(len(tr_all)), "n_val_all": int(len(va_all))}
                # 预测保存（AI7 的 val 侧）
                full_pred = np.full(len(ai7_idx), -1, np.int8)
                pos = np.searchsorted(ai7_idx, ava)
                full_pred[pos] = pred.astype(np.int8)
                preds_out[f"pred_{arm}_{mode}_f{fold}"] = full_pred
                # 密度诊断
                v0 = Ztr.var(0)
                mus = np.zeros((7, Ztr.shape[1])); vhat = np.zeros_like(mus)
                logpi = np.zeros(7); missing = []
                for k in range(7):
                    Xk = Ztr[ytr == k]
                    nk = len(Xk)
                    if nk == 0:
                        missing.append(fams_used[k])
                        continue
                    mus[k] = Xk.mean(0)
                    vhat[k] = np.maximum(nk / (nk + LAM) * Xk.var(0)
                                         + LAM / (nk + LAM) * v0, 1e-8)
                    logpi[k] = np.log(nk / len(Ztr))
                d2 = (Zva[:, None, :] - mus[None]) ** 2 / np.where(vhat > 0,
                                                                   vhat, 1.0)[None]
                logv = np.log(np.where(vhat > 0, vhat, 1.0))
                score = -0.5 * (d2.sum(-1) + logv.sum(1)[None]) + logpi[None]
                if missing:  # 缺失类 −∞（规范 §4）
                    score[:, [fams_used.index(m) for m in missing]] = -np.inf
                pden = score.argmax(1)
                rec_fd, rec_gd = {}, {}
                for k, f in enumerate(fams_used):
                    m = yva == k
                    rec_fd[f] = round(float((pden[m] == k).mean()), 4)
                    gv = defaultdict(list)
                    for j in np.where(m)[0]:
                        gv[str(gen[ava[j]])].append(float(pden[j] == k))
                    rec_gd[f] = {g: round(float(np.mean(v)), 4) for g, v in gv.items()}
                metrics["density"].setdefault(arm, {}).setdefault(mode, {})[
                    f"fold{fold}"] = {
                    "BA_F": round(float(np.mean(list(rec_fd.values()))), 4),
                    "BA_G": round(float(np.mean([np.mean(list(rec_gd[f].values()))
                                                 for f in fams_used])), 4),
                    "missing_classes": missing}
                say(f"[e30] {key}: BA_F {BA_F:.4f} BA_G {BA_G:.4f} | det {auroc:.4f}"
                    f" | dens BA_F {metrics['density'][arm][mode][f'fold{fold}']['BA_F']}")

    t_solver = (time.time() - t0) / 60
    say(f"[e30] 求解耗时 {t_solver:.1f} min")

    # ---- 出口判定（规范 §4） ----
    exit_tbl = {}
    all_pass = True
    for arm in reps:
        exit_tbl[arm] = {}
        for metric in ("BA_F", "BA_G"):
            g0 = (metrics["arms"][arm]["gen"]["fold0"][metric]
                  - metrics["arms"][arm]["random"]["fold0"][metric])
            g1 = (metrics["arms"][arm]["gen"]["fold1"][metric]
                  - metrics["arms"][arm]["random"]["fold1"][metric])
            ok = (g0 > 0) and (g1 > 0)
            all_pass &= ok
            exit_tbl[arm][metric] = {"diff_fold0": round(g0, 4),
                                     "diff_fold1": round(g1, 4), "pass": bool(ok)}
    det_diffs = {arm: {f"fold{f}": round(
        metrics["arms"][arm]["gen"][f"fold{f}"]["det_auroc"]
        - metrics["arms"][arm]["random"][f"fold{f}"]["det_auroc"], 4)
        for f in (0, 1)} for arm in reps}
    metrics["exit"] = {
        "rule": "BA_F 与 BA_G 在 R0、R1 上都 > 随机控制且两折方向一致",
        "checks": exit_tbl, "det_gen_minus_random": det_diffs,
        "passed": bool(all_pass),
        "conclusion": ("观察到稳定跨 generator 家族信号（初始证据）" if all_pass
                       else "当前协议下未观察到稳定跨 generator 家族信号")}
    say(f"[e30] 出口：passed={all_pass} | {exit_tbl}")

    # ---- 产物 ----
    key_lines = "\n".join(f"{d.split}:{d.row}:{ids_md5(d.ids)}" for d in docs_all)
    manifest = {"commit": commit, "smoke": a.smoke,
                "features_source": str(R28_FEATS) if not a.smoke else "smoke-export",
                "features_sha256": feats_sha,
                "input_manifest_sha256": hashlib.sha256(key_lines.encode()).hexdigest(),
                "families_closed_set": fams_used,
                "excluded_families": ["OpenAI（单 generator）"],
                "family_generator_counts": {f: len(gen_split[f]["A"]) +
                                            len(gen_split[f]["B"]) for f in fams_used},
                "fold_generators": {f"fold{fold}": metrics["splits"][f"fold{fold}"][
                    "fold_generators"] for fold in (0, 1)},
                "counts": {"n": n, "ai7": int(len(ai7_idx)),
                           "human": int(len(hum_idx))},
                "dims": {"R0": 768, "R1": 768}, "C_fixed": C_FIXED, "LAM": LAM,
                "chance": CHANCE, "seed": {"family_gen_shuffle": "1000+family_idx",
                                           "human": 1, "random_ctrl": "200+10*fold+family_idx"},
                "test_accessed": False, "unseen_accessed": False}
    np.savez_compressed(OUT / "predictions.npz",
                        ai7_rows=rows[ai7_idx], ai7_split=sp[ai7_idx],
                        ai7_family=np.array([LUT7[int(f)] for f in fam[ai7_idx]],
                                            np.int8),
                        ai7_generator=gen[ai7_idx], ai7_language=lang[ai7_idx],
                        ai7_length=ntok[ai7_idx], classes_=np.array(fams_used),
                        **preds_out)
    (OUT / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=1))
    dens = {arm: metrics["density"][arm] for arm in metrics["density"]}
    (OUT / "density_metrics.json").write_text(json.dumps(
        {"LAM": LAM, "note": "诊断用（缺失类 −∞、方差下限 1e-8）；不参与 H1 出口",
         "results": dens}, ensure_ascii=False, indent=1))
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    (OUT / "config.json").write_text(json.dumps({
        "args": vars(args), "C_fixed": C_FIXED, "STD_FLOOR": STD_FLOOR, "LAM": LAM,
        "protocol": "family-conditional 两折（每家族两侧均有 generator；2→1/1、3→1/2、5→2/3）",
        "representations": {"R0_mraw": "主", "R1_mu": "次要；使用 E24 监督头得到，"
                                                  "不能宣称完整 pipeline 的新 generator 泛化"},
        "exit_rule": metrics["exit"]["rule"], "commit": commit}, ensure_ascii=False,
        indent=1))
    (OUT / "solver.log").write_text("\n".join(solver_log) + "\n")
    say(f"[e30] 完成 {time.time()-t_start:.1f}s；产物 {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
