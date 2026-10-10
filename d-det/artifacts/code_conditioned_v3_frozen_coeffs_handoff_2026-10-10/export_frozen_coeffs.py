"""Export frozen fitted coefficients of the canonical C0 LR-based components.

Per guidance §15 / ACL §57 strict-reproduction path (b): "冻结并交换拟合系数".
Covers dev+inner x 11 folds:
  - semantic : StandardScaler(mean_, scale_) + LogisticRegression(coef_, intercept_)
  - style_meta: same on concat(style 92, meta 10, sizelen[:3])
  - z-stats (mu/sd) for all four components, recomputed from the delivered
    train-side score arrays (t_s_*) with the exact zfit_fuse convention
    (mu=mean, sd=max(std,1e-8), population ddof=0).

Verification per fold/protocol (hard asserts):
  1. refit semantic/style_meta eval & train scores == delivered s_*/t_s_* arrays
     (max|delta| reported; expected 0.0 by machine determinism)
  2. fused reconstructed from delivered per-component arrays + frozen z-stats
     == delivered fused
Failures stop the export.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cc_common as cc

ROOT = cc.ROOT
RUN = ROOT / "d-det/artifacts/code_conditioned_fresh_c0_canonical_2026-10-10"
OUT = ROOT / "d-det/artifacts/code_conditioned_v3_frozen_coeffs_handoff_2026-10-10"
HELDOUT_ORDER = [
    "codellama--CodeLlama-13b-Instruct-hf", "codellama--CodeLlama-34b-Instruct-hf",
    "codellama--CodeLlama-70b-Instruct-hf", "codellama--CodeLlama-7b-Instruct-hf",
    "Qwen--Qwen2.5-Coder-1.5B-Instruct", "Qwen--Qwen2.5-Coder-14B-Instruct",
    "Qwen--Qwen2.5-Coder-32B-Instruct", "Qwen--Qwen2.5-Coder-7B-Instruct",
    "deepseek-ai--deepseek-coder-1.3b-instruct",
    "deepseek-ai--deepseek-coder-33b-instruct",
    "deepseek-ai--deepseek-coder-6.7b-instruct",
]
COMPS = ["semantic", "char_tfidf", "word_tfidf", "style_meta"]


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    design = cc.load_design()
    b = design["bundle"]
    hy, style, meta, sizelen = b["hy_small"], b["style"], b["meta"], b["sizelen"]
    Xsm = np.hstack([style, meta, sizelen[:, :3]])
    fold_by_h = {f["heldout_generator_member"]: f for f in design["folds"]}
    assert design["members_order"] == HELDOUT_ORDER

    checks, files = {}, {}
    for protocol in ("dev", "inner"):
        use_inner = protocol == "inner"
        checks[protocol] = {}
        for h in HELDOUT_ORDER:
            fold = fold_by_h[h]
            fit_mask, _, _, _ = cc.fold_setup(design, fold, split="dev", inner=use_inner)
            y_fit = cc.fold_series_target(design, fold)[fit_mask]
            fit_rows = np.where(fit_mask)[0]
            token = h.replace("--", "_")
            z = dict(np.load(RUN / "local" / f"scores_{protocol}_fold{token}.npz"))

            sc_s = StandardScaler().fit(hy[fit_rows])
            lr_s = LogisticRegression(max_iter=2000, C=1.0).fit(sc_s.transform(hy[fit_rows]), y_fit)
            sc_m = StandardScaler().fit(Xsm[fit_rows])
            lr_m = LogisticRegression(max_iter=2000, C=1.0).fit(sc_m.transform(Xsm[fit_rows]), y_fit)

            ev_rows = np.asarray(z["ev_rows"])
            d = {
                "sem_eval": float(np.abs(lr_s.decision_function(sc_s.transform(hy[ev_rows]))
                                         - np.asarray(z["s_semantic"])).max()),
                "sem_train": float(np.abs(lr_s.decision_function(sc_s.transform(hy[fit_rows]))
                                          - np.asarray(z["t_s_semantic"])).max()),
                "sm_eval": float(np.abs(lr_m.decision_function(sc_m.transform(Xsm[ev_rows]))
                                        - np.asarray(z["s_style_meta"])).max()),
                "sm_train": float(np.abs(lr_m.decision_function(sc_m.transform(Xsm[fit_rows]))
                                         - np.asarray(z["t_s_style_meta"])).max()),
            }
            assert max(d.values()) <= 1e-9, (protocol, h, d)

            mu, sd = {}, {}
            for c in COMPS:
                t = z[f"t_s_{c}"]  # NATIVE stored dtype (semantic is float32!)
                mu[c] = float(np.mean(t))
                sd[c] = max(float(np.std(t)), 1e-8)
            fused_re = np.mean([(z[f"s_{c}"] - mu[c]) / sd[c]
                                for c in sorted(COMPS)], axis=0)
            d["fused_recon"] = float(np.abs(fused_re - np.asarray(z["fused"])).max())
            assert d["fused_recon"] <= 1e-12, (protocol, h, d)
            d["stored_dtypes"] = {c: str(z[f"s_{c}"].dtype) for c in COMPS}
            checks[protocol][h] = d

            out = {
                "sem_mean": sc_s.mean_, "sem_scale": sc_s.scale_,
                "sem_coef": lr_s.coef_.ravel(), "sem_intercept": np.array(lr_s.intercept_).ravel(),
                "sm_mean": sc_m.mean_, "sm_scale": sc_m.scale_,
                "sm_coef": lr_m.coef_.ravel(), "sm_intercept": np.array(lr_m.intercept_).ravel(),
                "ev_rows": ev_rows, "t_rows": np.asarray(z["t_rows"]),
                **{f"mu_{c}": np.float64(mu[c]) for c in COMPS},
                **{f"sd_{c}": np.float64(sd[c]) for c in COMPS},
            }
            fp = OUT / f"coeffs_{protocol}_fold{token}.npz"
            # NATIVE dtypes are preserved on purpose: semantic (float32) must stay
            # float32 so decision_function reproduces the canonical float32 scores.
            np.savez_compressed(fp, **{k: (np.asarray(v) if k != "t_rows" else v.astype(np.int64))
                                       for k, v in out.items() if k != "ev_rows"},
                                ev_rows=ev_rows.astype(np.int64))
            d["coef_dtypes"] = {"sem_coef": str(out["sem_coef"].dtype),
                                "sm_coef": str(out["sm_coef"].dtype)}
            files[fp.name] = {"sha256": sha256_file(fp), "bytes": fp.stat().st_size}
            print(f"[{protocol}] {h.split('--')[-1][:26]:28s} "
                  f"semE={d['sem_eval']:.1e} smE={d['sm_eval']:.1e} fusedRecon={d['fused_recon']:.1e} "
                  f"({time.time()-t0:.0f}s)")

    man = {
        "schema": "code_conditioned_v3_frozen_coeffs_handoff_v1",
        "date": "2026-10-10",
        "guidance_ref": ("d-det/docx/d-det_AutoDL_代码条件后训练_已见家族未见生成器指导_2026-10-09.md "
                         "§15; ACL §57 严格复现路径(b) 冻结并交换拟合系数"),
        "source_run": "code_conditioned_fresh_c0_canonical_2026-10-10 (v3 spec)",
        "source_commit": "43c5b1e3beaed3453568e9fbcb66eb108b186d47",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "files": files,
        "verification": {
            "per_fold": checks,
            "rule": ("refit eval/train scores per fold must equal the delivered canonical s_*/t_s_* arrays "
                     "(<=1e-9, observed 0.0); fused reconstruction from delivered component arrays + "
                     "frozen z-stats must equal delivered fused (<=1e-12)"),
        },
        "conventions": {
            "semantic_score": "z = (hy - sem_mean)/sem_scale ; score = z @ sem_coef + sem_intercept",
            "style_meta_score": "x = concat(style 92, meta 10, sizelen[:3]) ; z = (x - sm_mean)/sm_scale ; "
                                "score = z @ sm_coef + sm_intercept",
            "recommended_apply": "rebuild StandardScaler/LogisticRegression objects by assigning the "
                                 "fitted attributes (mean_, scale_, n_features_in_, coef_, intercept_) and "
                                 "call transform()/decision_function(); this reproduces the canonical "
                                 "dtype chain exactly (semantic scores are float32 because hy is float32)",
            "z_stats": "mu = mean(train scores), sd = max(std(train scores), 1e-8); computed on the "
                       "STORED array dtype exactly as the canonical run (semantic: float32 arrays)",
            "fusion": "fused = mean over the four components of (s_c - mu_c)/sd_c (equal weights)",
        },
        "scope": ("frozen fitted coefficients of the LR-based components (semantic, style_meta) + z-stats; "
                  "train/dev only; no test, no weights, no raw corpus, no generation assets"),
        "notes": ("char/word SGD ensembles are not exported (dense 195k/345k-dim x 3 seeds ~13 MB per "
                  "fold/protocol); their cross-side deviation was reported small. If needed, a float32 "
                  "selective export can be produced on request."),
        "switches": {"training_allowed": True, "test_read_allowed": False,
                     "generation_allowed": False, "weights_downloaded": False,
                     "code_execution": False},
    }
    (OUT / "frozen_coeffs_manifest.json").write_text(
        json.dumps(man, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"ok": True, "files": len(files), "s": round(time.time() - t0, 1)}))


if __name__ == "__main__":
    main()
