#!/usr/bin/env python3
"""Apply the frozen fitted coefficients to reproduce canonical C0 component scores.

Reproduces the exact sklearn dtype chain by rebuilding StandardScaler /
LogisticRegression objects and assigning the frozen fitted attributes, then
calling transform()/decision_function(). Semantic scores are float32 (hy is
float32) — this is expected and matches the canonical run.

Usage (self-test on the server, expect max|delta| == 0.0):
  python apply_frozen_coeffs_template.py \
      --bundle  <.../features/bundle.npz> \
      --coeffs  coeffs_dev_foldcodellama_CodeLlama-13b-Instruct-hf.npz \
      --scores  <.../local/scores_dev_foldcodellama_CodeLlama-13b-Instruct-hf.npz>

--scores is optional; when given, the script also reconstructs the fused score
with the frozen z-stats (using the char/word arrays found in --scores) and
prints the comparison. Nothing is written.
"""
import argparse
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

COMPS = ["semantic", "char_tfidf", "word_tfidf", "style_meta"]


def rebuild_lr(mean, scale, coef, intercept):
    sc = StandardScaler()
    sc.mean_ = np.asarray(mean)
    sc.scale_ = np.asarray(scale)
    sc.var_ = sc.scale_ ** 2
    sc.n_features_in_ = len(sc.mean_)
    lr = LogisticRegression()
    # keep the STORED native dtype (semantic coef is float32) so that
    # decision_function() reproduces the canonical dtype chain exactly
    lr.coef_ = np.asarray(coef).reshape(1, -1)
    lr.intercept_ = np.asarray(intercept).ravel()
    lr.classes_ = np.array([0, 1])
    lr.n_features_in_ = lr.coef_.shape[1]
    return sc, lr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--coeffs", type=Path, required=True)
    ap.add_argument("--scores", type=Path, default=None)
    a = ap.parse_args()

    b = dict(np.load(a.bundle))
    c = dict(np.load(a.coeffs))
    hy, style, meta, sizelen = b["hy_small"], b["style"], b["meta"], b["sizelen"]
    Xsm = np.hstack([style, meta, sizelen[:, :3]])

    sc_s, lr_s = rebuild_lr(c["sem_mean"], c["sem_scale"], c["sem_coef"], c["sem_intercept"])
    sc_m, lr_m = rebuild_lr(c["sm_mean"], c["sm_scale"], c["sm_coef"], c["sm_intercept"])

    ev = np.asarray(c["ev_rows"])
    sem_ev = lr_s.decision_function(sc_s.transform(hy[ev]))
    sm_ev = lr_m.decision_function(sc_m.transform(Xsm[ev]))
    print(f"semantic  ev scores dtype={sem_ev.dtype}  head={sem_ev[:3]}")
    print(f"style_meta ev scores dtype={sm_ev.dtype} head={sm_ev[:3]}")

    if a.scores:
        z = dict(np.load(a.scores))
        d_sem = float(np.abs(sem_ev - np.asarray(z["s_semantic"])).max())
        d_sm = float(np.abs(sm_ev - np.asarray(z["s_style_meta"])).max())
        print(f"vs server: max|d_semantic|={d_sem:.3e}  max|d_style_meta|={d_sm:.3e}")
        # fused reconstruction (needs char/word arrays; take from --scores here)
        comp = {"semantic": sem_ev, "style_meta": sm_ev}
        ok = True
        for k in ("char_tfidf", "word_tfidf"):
            if f"s_{k}" in z:
                comp[k] = np.asarray(z[f"s_{k}"])
            else:
                ok = False
        if ok:
            fused = np.mean([(comp[k] - float(c[f"mu_{k}"])) / float(c[f"sd_{k}"])
                             for k in sorted(COMPS)], axis=0)
            d_f = float(np.abs(fused - np.asarray(z["fused"])).max())
            print(f"fused reconstruction max|delta|={d_f:.3e}")
    print("gate reminder: row_score_max_abs <= 1e-3 AND metric_abs <= 1e-3 (guidance §13/§15)")


if __name__ == "__main__":
    main()
