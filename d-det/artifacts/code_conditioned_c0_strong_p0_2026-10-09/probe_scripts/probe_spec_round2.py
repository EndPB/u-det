"""C0 spec probes round 2 (fixed row mapping). Pre-declared variants (dev protocol):

A : S1 vectorizers, LR C=1, fusion z from in-sample train scores (baseline, should ~.9245)
B : A but fusion z from 5-fold task-grouped OOF train scores
C : B + wide vectorizers (char_wb 2-5 min_df=1, word 1-3 min_df=1) + LR C=4
D : C + class_weight='balanced'
E : D but semantic = CodeT5-base (768d)  [only if A-D don't close the gap]
F : D + semantic = [small;base] concat    [only if needed]

All variants share folds/eval/targets; only listed knobs change. Results saved for
the C0 report; C0 freeze will pick the most defensible spec closest to the local
reference (recorded transparently).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict
from sklearn.base import clone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import cc_common as cc  # noqa: E402

COMPONENTS = ["semantic", "char_tfidf", "word_tfidf", "style_meta"]


def task_grouped_folds(taskpos_full, n_splits=5):
    """5-fold splits grouped by task (rows of a task stay together)."""
    uniq, inv = np.unique(taskpos_full, return_inverse=True)
    rng = np.random.default_rng(20261009)
    perm = rng.permutation(len(uniq))
    groups = np.array_split(perm, n_splits)
    task_grp = np.empty(len(uniq), dtype=int)
    for g, members in enumerate(groups):
        task_grp[members] = g
    return task_grp[inv]


def main():
    t0 = time.time()
    design = cc.load_design()
    bundle = design["bundle"]
    rows = design["rows"]
    nT = len(design["tasks_all"])
    r0_rows = bundle["gmodel_idx"] * nT * 2 + bundle["task_idx"] * 2 + 1
    emb_base_sub = np.load(cc.R0 / "emb_base.npz")["emb"][r0_rows]
    texts = [r["code"] for r in rows]

    print("tfidf building...", flush=True)
    vec = {}
    vec["S1"] = (
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2,
                        max_features=200000, sublinear_tf=True, lowercase=False).fit_transform(texts),
        TfidfVectorizer(analyzer="word", token_pattern=r"[A-Za-z_][A-Za-z0-9_]*",
                        ngram_range=(1, 2), min_df=2, max_features=200000,
                        sublinear_tf=True, lowercase=False).fit_transform(texts))
    vec["S3"] = (
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=1,
                        sublinear_tf=True, lowercase=False).fit_transform(texts),
        TfidfVectorizer(analyzer="word", token_pattern=r"[A-Za-z_][A-Za-z0-9_]*",
                        ngram_range=(1, 3), min_df=1, sublinear_tf=True,
                        lowercase=False).fit_transform(texts))
    print("tfidf:", {k: (a.shape, b.shape) for k, (a, b) in vec.items()},
          f"{time.time()-t0:.0f}s", flush=True)

    task_split = {r["task_id"]: r["split"] for r in rows}
    tpos_all = {t: i for i, t in enumerate(sorted(design["tasks_all"]))}

    def run(tag, vec_key, sem_key, balanced, C, z_mode):
        per_fold = {}
        for fold in design["folds"]:
            fit_mask, ev_mask, pos_mask, _ = cc.fold_setup(design, fold, split="dev", inner=False)
            y_fit = cc.fold_series_target(design, fold)[fit_mask]
            ev_rows = np.where(ev_mask)[0]
            fit_rows = np.where(fit_mask)[0]
            Xc, Xw = vec[vec_key]
            if sem_key == "small":
                sem = bundle["hy_small"]
            elif sem_key == "base":
                sem = emb_base_sub
            else:
                sem = np.hstack([bundle["hy_small"], emb_base_sub])
            sm = np.hstack([bundle["style"], bundle["meta"], bundle["sizelen"][:, :3]])
            sc, sc2 = StandardScaler().fit(sem[fit_rows]), StandardScaler().fit(sm[fit_rows])
            mats_tr = {"semantic": sc.transform(sem[fit_rows]),
                       "char_tfidf": Xc[fit_rows], "word_tfidf": Xw[fit_rows],
                       "style_meta": sc2.transform(sm[fit_rows])}
            mats_ev = {"semantic": sc.transform(sem[ev_rows]),
                       "char_tfidf": Xc[ev_rows], "word_tfidf": Xw[ev_rows],
                       "style_meta": sc2.transform(sm[ev_rows])}
            s_tr, s_ev, s_oof = {}, {}, {}
            tp_fit = np.array([tpos_all[rows[i]["task_id"]] for i in fit_rows])
            grp = task_grouped_folds(tp_fit)
            for k in COMPONENTS:
                clf = LogisticRegression(max_iter=3000, C=C,
                                         class_weight=("balanced" if balanced else None))
                clf.fit(mats_tr[k], y_fit)
                s_tr[k] = clf.decision_function(mats_tr[k])
                s_ev[k] = clf.decision_function(mats_ev[k])
                if z_mode == "oof":
                    oof = np.empty(len(fit_rows))
                    for g in range(5):
                        trm, dvm = grp != g, grp == g
                        c2 = clone(clf).fit(mats_tr[k][trm], y_fit[trm])
                        oof[dvm] = c2.decision_function(mats_tr[k][dvm])
                    s_oof[k] = oof
            stats_source = s_oof if z_mode == "oof" else s_tr
            f_ev, _, _ = cc.zfit_fuse(stats_source, s_ev)
            fid = fold["heldout_generator_member"]
            y_ev = pos_mask[ev_rows].astype(int)
            per_fold[fid] = {"row": cc.auroc(y_ev, f_ev),
                             **{k: cc.auroc(y_ev, s_ev[k]) for k in COMPONENTS}}
        mean_row = float(np.mean([v["row"] for v in per_fold.values()]))
        comps = {k: round(float(np.mean([v[k] for v in per_fold.values()])), 4) for k in COMPONENTS}
        print(f"[probe2] {tag}: fused={mean_row:.4f} comps={comps} ({time.time()-t0:.0f}s)",
              flush=True)
        return {"per_fold_row": {k: v["row"] for k, v in per_fold.items()},
                "mean_row": mean_row, "components": comps}

    res = {}
    res["A_baseline"] = run("A_baseline", "S1", "small", False, 1.0, "insample")
    res["B_oof_z"] = run("B_oof_z", "S1", "small", False, 1.0, "oof")
    res["C_wide_oof"] = run("C_wide_oof", "S3", "small", False, 4.0, "oof")
    res["D_wide_bal_oof"] = run("D_wide_bal_oof", "S3", "small", True, 4.0, "oof")
    out = cc.DESIGN / "local"
    (out / "spec_probes_v2.json").write_text(json.dumps(res, ensure_ascii=False, indent=1),
                                             encoding="utf-8")
    print("saved probes v2", flush=True)


if __name__ == "__main__":
    main()
