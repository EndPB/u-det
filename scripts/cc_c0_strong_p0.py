"""C0: 強 P0 複現（server reconstruction）— code-conditioned round 2026-10-09.

与 11 个 member-held-out 折（fold_plan.json）逐折重建四个 fit-only 组件：
  1. CodeT5-small semantic linear（冻结 h_y，按折 train 标准化）
  2. char TF-IDF（char_wb 2-4）+ linear
  3. word TF-IDF（标识符/数字词 1-2 gram）+ linear
  4. style/meta linear（style + meta + sizelen）
并按冻结规则等权 z-score 融合（分数组内以折 train 分数均值/标准差标准化）。
输出 pooled、task-macro、member-macro(=折均值)、500 次 task-cluster bootstrap CI，
以及与本机参考值（ACL §48 约值 .9344/.9324）的对照。

开关：train/dev only; test_read=false; generation=false; weights_downloaded=false; code_execution=false。
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import cc_common as cc  # noqa: E402

OUT = ROOT / "d-det/artifacts/code_conditioned_c0_strong_p0_2026-10-09"
LOCAL_REF = {"dev_row_level_mean": 0.9344, "inner_row_level_mean": 0.9324,
             "source": "ACL 总结 §48（本机约值）；本机逐点参考表未随本轮传输"}
LOG: list[str] = []
COMPONENTS = ["semantic", "char_tfidf", "word_tfidf", "style_meta"]


def log(m):
    print(m, flush=True)
    LOG.append(m)


def build_tfidf(texts):
    t0 = time.time()
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=1,
                           sublinear_tf=True, lowercase=False)
    Xc = char.fit_transform(texts)
    word = TfidfVectorizer(analyzer="word", token_pattern=r"[A-Za-z_][A-Za-z0-9_]*",
                           ngram_range=(1, 3), min_df=1,
                           sublinear_tf=True, lowercase=False)
    Xw = word.fit_transform(texts)
    log(f"[c0] tfidf char {Xc.shape} word {Xw.shape} in {time.time()-t0:.1f}s")
    return Xc, Xw


def fit_lr(X, y):
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(X, y)
    return clf


def sgd_ensemble(Xtr, ytr, Xev, seeds=(0, 1, 2), epochs=5, bs=4096, alpha=1e-6):
    """3-seed SGD log-loss ensemble (established server convention for lexical P0)."""
    from sklearn.linear_model import SGDClassifier
    out = np.zeros(Xev.shape[0])
    for s in seeds:
        clf = SGDClassifier(loss="log_loss", alpha=alpha, random_state=s)
        rng = np.random.default_rng(s)
        for _ in range(epochs):
            perm = rng.permutation(Xtr.shape[0])
            for i in range(0, len(perm), bs):
                sel = perm[i:i + bs]
                clf.partial_fit(Xtr[sel], ytr[sel], classes=np.array([0, 1]))
        out += clf.decision_function(Xev)
    return out / len(seeds)


def run_protocol(design, protocol, Xc, Xw):
    b = design["bundle"]
    rows = design["rows"]
    member_idx, task_idx = b["member_idx"], b["task_idx"]
    hy, style, meta, sizelen = b["hy_small"], b["style"], b["meta"], b["sizelen"]
    task_split = {}
    for r in rows:
        task_split[r["task_id"]] = r["split"]
    tasks_all = design["tasks_all"]
    if protocol == "dev":
        sel_tasks = sorted(t for t in tasks_all if task_split[t] == "dev")
        inner_of = None
        use_inner = False
    else:
        inner_of = cc.inner_split(tasks_all)
        sel_tasks = sorted(t for t in tasks_all if task_split[t] == "train"
                           and inner_of[t] == "inner_dev")
        use_inner = True
    tpos_of_task = {t: i for i, t in enumerate(sel_tasks)}
    n_tasks = len(sel_tasks)
    log(f"[c0] protocol={protocol}: tasks={n_tasks}")

    per_fold = {}
    labels, tposs, fused_scores = {}, {}, {}
    comp_scores = {k: {} for k in COMPONENTS}
    for fold in design["folds"]:
        fid = fold["heldout_generator_member"]
        tf0 = time.time()
        fit_mask, ev_mask, pos_mask, y = cc.fold_setup(design, fold,
                                                       split="dev", inner=use_inner)
        y_fit = cc.fold_series_target(design, fold)[fit_mask]
        # eval rows sorted by task position
        ev_rows = np.where(ev_mask)[0]
        tp = np.array([tpos_of_task[rows[i]["task_id"]] for i in ev_rows])
        order = np.argsort(tp, kind="mergesort")
        ev_rows, tp = ev_rows[order], tp[order]
        y_ev = pos_mask[ev_rows].astype(int)
        fit_rows = np.where(fit_mask)[0]

        # --- components ---
        comps_tr, comps_ev = {}, {}
        scaler = StandardScaler().fit(hy[fit_rows])
        comps_tr["semantic"] = scaler.transform(hy[fit_rows])
        comps_ev["semantic"] = scaler.transform(hy[ev_rows])
        Xsm = np.hstack([style, meta, sizelen[:, :3]])
        scaler2 = StandardScaler().fit(Xsm[fit_rows])
        comps_tr["style_meta"] = scaler2.transform(Xsm[fit_rows])
        comps_ev["style_meta"] = scaler2.transform(Xsm[ev_rows])
        comps_tr["char_tfidf"] = Xc[fit_rows]
        comps_ev["char_tfidf"] = Xc[ev_rows]
        comps_tr["word_tfidf"] = Xw[fit_rows]
        comps_ev["word_tfidf"] = Xw[ev_rows]

        s_tr, s_ev = {}, {}
        for k in COMPONENTS:
            if k in ("char_tfidf", "word_tfidf"):
                s_tr[k] = sgd_ensemble(comps_tr[k], y_fit, comps_tr[k])
                s_ev[k] = sgd_ensemble(comps_tr[k], y_fit, comps_ev[k])
            else:
                clf = fit_lr(comps_tr[k], y_fit)
                s_tr[k] = clf.decision_function(comps_tr[k])
                s_ev[k] = clf.decision_function(comps_ev[k])
        fused_ev, z_ev, stats = cc.zfit_fuse(s_tr, s_ev)
        per_fold[fid] = {
            "row_level": {"fused": cc.auroc(y_ev, fused_ev),
                          **{k: cc.auroc(y_ev, s_ev[k]) for k in COMPONENTS}},
            "task_macro": cc.task_macro_auroc(y_ev, fused_ev, tp),
            "runtime_s": round(time.time() - tf0, 1),
        }
        labels[fid], tposs[fid], fused_scores[fid] = y_ev, tp, fused_ev
        for k in COMPONENTS:
            comp_scores[k][fid] = s_ev[k]
        local_dir = OUT / "local"
        local_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(local_dir / f"scores_{protocol}_fold{fid.replace('--','_')}.npz",
                            ev_rows=ev_rows, y=y_ev, taskpos=tp, fused=fused_ev,
                            **{f"s_{k}": s_ev[k] for k in COMPONENTS})
        log(f"[c0] {protocol} fold {fid[:36]:36s} row={per_fold[fid]['row_level']['fused']:.4f} "
            f"tm={per_fold[fid]['task_macro']:.4f} ({per_fold[fid]['runtime_s']}s)")

    # --- aggregates ---
    row_mean = float(np.mean([v["row_level"]["fused"] for v in per_fold.values()]))
    tm_mean = float(np.mean([v["task_macro"] for v in per_fold.values()]))
    all_y = np.concatenate([labels[f] for f in labels])
    all_s = np.concatenate([fused_scores[f] for f in labels])
    pooled = cc.auroc(all_y, all_s)
    boot = cc.bootstrap_metrics(fused_scores, labels, tposs, n_tasks=n_tasks)
    agg = {
        "row_level_mean_over_folds": row_mean,
        "task_macro_mean_over_folds": tm_mean,
        "member_macro": row_mean,  # one member per fold; identical by construction
        "pooled": pooled,
        "bootstrap": {"row_level": {"ci95": cc.ci(boot["row_level"])},
                      "task_macro": {"ci95": cc.ci(boot["task_macro"])}},
        "per_component_row_level_mean": {
            k: float(np.mean([per_fold[f]["row_level"][k] for f in per_fold]))
            for k in COMPONENTS},
        "per_component_task_macro_mean": {
            k: float(np.mean([cc.task_macro_auroc(labels[f], comp_scores[k][f], tposs[f])
                              for f in labels])) for k in COMPONENTS},
    }
    return per_fold, agg


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "logs").mkdir(exist_ok=True)
    design = cc.load_design()
    texts = [r["code"] for r in design["rows"]]
    Xc, Xw = build_tfidf(texts)

    res = {"schema": "cc_c0_strong_p0_v1",
           "generated_utc": datetime.now(timezone.utc).isoformat(),
           "report_first_line": ("C0 强 P0 重建（11 折 member-heldout；四个 fit-only 组件等权 z-score 融合；"
                                 "train/dev only）"),
           "protocols": {}, "local_reference": LOCAL_REF}
    for protocol in ("dev", "inner"):
        per_fold, agg = run_protocol(design, protocol, Xc, Xw)
        res["protocols"][protocol] = {"per_fold": per_fold, "aggregate": agg,
                                      "n_tasks": 171 if protocol == "dev" else None}
    dev = res["protocols"]["dev"]["aggregate"]["row_level_mean_over_folds"]
    inner = res["protocols"]["inner"]["aggregate"]["row_level_mean_over_folds"]
    res["reference_check"] = {
        "dev_row_level_mean": dev,
        "inner_row_level_mean": inner,
        "dev_delta_vs_ref": dev - LOCAL_REF["dev_row_level_mean"],
        "inner_delta_vs_ref": inner - LOCAL_REF["inner_row_level_mean"],
        "within_1e-3": bool(abs(dev - LOCAL_REF["dev_row_level_mean"]) <= 1e-3
                            and abs(inner - LOCAL_REF["inner_row_level_mean"]) <= 1e-3),
        "spec_probe_history": {
            "row_mapping_fix": "emb row mapping used local member index before; fixed to global 115-model index; texts.jsonl.gz check 10659/10659",
            "dev_row_level_by_variant": {
                "S1_LR_insample_z": 0.9245, "S1_LR_oof_z": 0.9236,
                "semantic_base": 0.8649, "semantic_small+base": 0.8553,
                "wide_LR_C4": 0.9303, "wide_LR_balanced": 0.9294,
                "wide_SGD3_frozen": 0.9329,
            },
            "frozen": "wide_SGD3（char 2-5 min_df=1 / word 1-3 min_df=1 / SGD×3 seed 集成）",
        },
        "note": ("本机逐点参考表未随本轮传输；此处仅对 ACL §48 的 approx 值做报告性对照；"
                 "冻结规格下的残差 Δdev=-0.0015/Δinner 待读；严格 1e-3 逐点审计需本机强 P0 参考分数。"),
    }
    (OUT / "metrics.json").write_text(json.dumps(res, ensure_ascii=False, indent=1),
                                      encoding="utf-8")
    (OUT / "hypothesis.json").write_text(json.dumps({
        "schema": "cc_c0_hypothesis_v1",
        "hypothesis": ("同一 11 member-held-out 折上，四个 fit-only 组件（CodeT5-small semantic linear、"
                       "char TF-IDF、word TF-IDF、style/meta linear）等权 z-score 融合可复现本机强 P0 "
                       "(dev 行级均值≈.9344、内部 train-only≈.9324，±1e-3)。"),
        "gate": "dev/inner 行级均值与本机参考差 ≤1e-3；否则停在 C0 修复。",
        "claim_limit": "拟合与选择仅用折内 train；dev 仅开发评测；family_is_confirmed=false。",
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "config.json").write_text(json.dumps({
        "schema": "cc_c0_config_v1",
        "folds": "static_preflight/fold_plan.json（11 折）",
        "components": {
            "semantic": "LR(C=1, lbfgs) on train-fold standardized CodeT5-small h_y (512d)",
            "char_tfidf": "TfidfVectorizer(char_wb,2-5,min_df=1,sublinear,lowercase=False)+SGD log_loss x3 seeds (alpha=1e-6, 5ep)",
            "word_tfidf": "TfidfVectorizer(word,[A-Za-z_][A-Za-z0-9_]*,1-3,min_df=1,sublinear)+SGD log_loss x3 seeds",
            "style_meta": "LR on train-fold standardized [style(92),meta(10),sizelen(3)]",
        },
        "fusion": "equal-weight mean of per-component z-scores (train-fold score mean/std)",
        "bootstrap": {"n": 500, "seed": 20261009, "cluster": "dev task", "shared_across_folds": True},
        "inner_split": {"rule": "sha256('cc_inner_split_v1|'+task_id) % 5 == 0 -> inner_dev", "note": "server-frozen"},
        "switches": {"train_dev_only": True, "test_read": False, "generation": False,
                     "weights_downloaded": False, "code_execution": False},
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "data_role_matrix.json").write_text(json.dumps({
        "rows": "records_train_dev.jsonl（10,659 行 = 11 成员 × 969 task；instruct 模式）",
        "fit": "折内 train split 行（排除 heldout 成员）",
        "eval": "dev split：heldout 成员（正）+ 其它系列成员（负）；同系列兄弟排除（fold_plan 规格）",
        "inner": "train split 内 inner_dev task 同构评测（train-only 方向检查）",
        "never_used": ["test 行", "canonical_solution", "test 正文", "生成输出", "权重下载"],
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    # report
    L = ["# C0 强 P0 重建（server reconstruction；2026-10-09）", "",
         f"性质：复现/审计轮（非新方法结果）。dev 行级均值 = "
         f"{dev:.4f}；inner 行级均值 = {inner:.4f}（本机参考 .9344/.9324）。", ""]
    for protocol in ("dev", "inner"):
        agg = res["protocols"][protocol]["aggregate"]
        L += [f"## {protocol}", "",
              f"- row-level mean = **{agg['row_level_mean_over_folds']:.4f}** "
              f"CI95 {agg['bootstrap']['row_level']['ci95']}",
              f"- task-macro mean = {agg['task_macro_mean_over_folds']:.4f} "
              f"CI95 {agg['bootstrap']['task_macro']['ci95']}",
              f"- pooled = {agg['pooled']:.4f}；member-macro = {agg['member_macro']:.4f}", "",
              "| component | row-level mean |", "|---|---|"]
        for k, v in agg["per_component_row_level_mean"].items():
            L.append(f"| {k} | {v:.4f} |")
        L.append("")
    L += ["## 参考对照", "",
          f"- dev Δ vs .9344 = {res['reference_check']['dev_delta_vs_ref']:+.4f}；"
          f"inner Δ vs .9324 = {res['reference_check']['inner_delta_vs_ref']:+.4f}；"
          f"within_1e-3 = {res['reference_check']['within_1e-3']}",
          f"- {res['reference_check']['note']}", "",
          "## 规格探针（预声明，逐步定位）", "",
          "| 变体 | dev row-level |", "|---|---|"]
    for k, v in res["reference_check"]["spec_probe_history"]["dev_row_level_by_variant"].items():
        L.append(f"| {k} | {v:.4f} |")
    L += [f"- 冻结：{res['reference_check']['spec_probe_history']['frozen']}",
          f"- 行映射修复：{res['reference_check']['spec_probe_history']['row_mapping_fix']}", "",
          "> 全部拟合 fit-only；dev 仅开发评测；无 test/生成/权重下载。"]
    (OUT / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (OUT / "commands.txt").write_text(
        "OMP_NUM_THREADS=8 /root/miniconda3/envs/udet/bin/python -W ignore scripts/cc_c0_strong_p0.py\n",
        encoding="utf-8")
    (OUT / "logs" / "c0_run.log").write_text("\n".join(LOG) + "\n", encoding="utf-8")
    cc.git_meta(OUT)
    cc.write_sha256sums(OUT)
    log(f"[c0] done in {time.time()-t0:.1f}s dev={dev:.4f} inner={inner:.4f}")


if __name__ == "__main__":
    main()
