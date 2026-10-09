"""Canonical-order fresh train/dev C0 (2026-10-10, per guidance §12).

Fix vs `code_conditioned_fresh_c0_2026-10-09` (blocked: spec_violation_global_lexical_fit):
  - char/word TfidfVectorizer are now fit INSIDE each fold on `fit_rows` texts only,
    then used to `transform` eval rows (no dev/heldout text in vocabulary or IDF).
  - per-fold manifest gains `lexical_fit_spec` (pre-fit) and `lexical_fit_attestation`
    (post-fit: vectorizer params, fit-row hash, vocab size, IDF digest).
  - recompute C0 metrics + score digests; report member-order impact vs the
    permuted corrected run and the historical-package diagnostic (not the gate).

Switches: training_allowed=true (train/dev only); test_read=false; generation=false;
weights_downloaded=false; code_execution=false.

The fold-fit-only lexical fix is retained from `cc_fresh_c0_corrected.py`; the
only new change is canonicalizing the feature member order (guidance §12).
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cc_common as cc

ROOT = cc.ROOT
OUT = ROOT / "d-det/artifacts/code_conditioned_fresh_c0_canonical_2026-10-10"
PREV = ROOT / "d-det/artifacts/code_conditioned_fresh_c0_corrected_2026-10-10"
REF = ROOT / "d-det/artifacts/code_conditioned_p0_reference_2026-10-09/folds"
ADJUDICATION = "d-det/artifacts/code_conditioned_fresh_c0_spec_adjudication_2026-10-09/"
CROSS_SIDE = "d-det/artifacts/code_conditioned_fresh_c0_cross_side_local_reproduction_2026-10-10/"
OFFICIAL_ADMISSION_SHA = "cc4a77851702e0bbf57e50636744ad25b572da9a665148b2b76179236b0bdfae"
HELDOUT_ORDER = [
    "codellama--CodeLlama-13b-Instruct-hf", "codellama--CodeLlama-34b-Instruct-hf",
    "codellama--CodeLlama-70b-Instruct-hf", "codellama--CodeLlama-7b-Instruct-hf",
    "Qwen--Qwen2.5-Coder-1.5B-Instruct", "Qwen--Qwen2.5-Coder-14B-Instruct",
    "Qwen--Qwen2.5-Coder-32B-Instruct", "Qwen--Qwen2.5-Coder-7B-Instruct",
    "deepseek-ai--deepseek-coder-1.3b-instruct",
    "deepseek-ai--deepseek-coder-33b-instruct",
    "deepseek-ai--deepseek-coder-6.7b-instruct",
]
CHAR_PARAMS = dict(analyzer="char_wb", ngram_range=(2, 5), min_df=1, sublinear_tf=True,
                   lowercase=False)
WORD_PARAMS = dict(analyzer="word", token_pattern=r"[A-Za-z_][A-Za-z0-9_]*",
                   ngram_range=(1, 3), min_df=1, sublinear_tf=True, lowercase=False)

LOG: list[str] = []


def log(m):
    line = f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)
    LOG.append(line)


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_str(s: str) -> str:
    return sha256_bytes(s.encode("utf-8"))


def rows_digest(keys) -> str:
    return sha256_str("\n".join(keys) + "\n")


def texts_digest(texts) -> str:
    h = hashlib.sha256()
    for t in texts:
        h.update(t.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def vocab_digest(vocab: dict) -> str:
    return sha256_str("\n".join(sorted(vocab.keys())) + "\n")


def idf_digest(idf: np.ndarray) -> str:
    return sha256_bytes(np.ascontiguousarray(idf, dtype=np.float64).tobytes())


def array_digest(a: np.ndarray) -> str:
    a = np.ascontiguousarray(np.asarray(a, dtype=np.float64))
    return sha256_bytes(a.tobytes())


def component_specs() -> dict:
    return {
        "member_indexing": ("member_idx = admission members order [CodeLlama, Qwen, "
                            "DeepSeek]; feature bundle canonicalized 2026-10-10 (guidance §12)"),
        "semantic": {
            "input_block": "hy_small = frozen CodeT5-small mean-pooled code embedding (512-d), "
                           "from r0 corpus cache row = gmodel_idx*nT*2 + task_idx*2 + 1 (instruct)",
            "scaler": "StandardScaler fit on fold-fit rows (server sklearn)",
            "model": "LogisticRegression(max_iter=2000, C=1.0)",
            "score": "decision_function",
        },
        "char_tfidf": {
            "vectorizer": CHAR_PARAMS,
            "fit_scope": "fold-fit rows texts only (fold-train-only vocabulary & IDF; CORRECTED 2026-10-10)",
            "model": "SGDClassifier(loss=log_loss, alpha=1e-6) x seeds(0,1,2), 5 epochs of"
                     " shuffled partial_fit, batch=4096, seed-local RNG permutation",
            "score": "mean decision_function over 3 seeds",
        },
        "word_tfidf": {
            "vectorizer": WORD_PARAMS,
            "fit_scope": "fold-fit rows texts only (CORRECTED 2026-10-10)",
            "model": "same SGD ensemble as char_tfidf",
        },
        "style_meta": {
            "input_block": "concat(style 92, meta 10, sizelen 3) = 105-d server static features",
            "scaler": "StandardScaler fit on fold-fit rows",
            "model": "LogisticRegression(max_iter=2000, C=1.0)",
        },
        "fusion": {
            "z_stats": "per-component mean/std (population, ddof=0) of fold-fit-row scores",
            "weights": "equal (mean over 4 components, keys sorted)",
        },
        "target": "y_fit = 1 iff row member's series == heldout member's series",
        "eval_layout": "eval member blocks in ASCII-sorted member order; rows within block in "
                       "sorted dev task order",
        "inner_split_rule": "sha256('cc_inner_split_v1|'+task_id) % 5 == 0 -> inner_dev",
        "metrics": "AUROC ties=0.5; task-macro = mean of per-task AUROC; member-macro = mean over "
                   "folds; task-cluster bootstrap B=500 seed 20261009 (shared indices)",
    }


def sgd_ensemble(Xtr, ytr, Xev, seeds=(0, 1, 2), epochs=5, bs=4096, alpha=1e-6):
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


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "manifests").mkdir(exist_ok=True)
    (OUT / "local").mkdir(exist_ok=True)
    (OUT / "logs").mkdir(exist_ok=True)
    design = cc.load_design()
    b = design["bundle"]
    rows = design["rows"]
    members = design["members_order"]
    series_map = {s: sorted(ms) for s, ms in design["series_of"].items()}
    canonical_map = json.dumps(series_map, sort_keys=True, ensure_ascii=False,
                               separators=(",", ":"))
    map_sha = sha256_str(canonical_map)
    admission_file = ROOT / "d-det/artifacts/code_conditioned_design_2026-10-09/inputs/family_series_admission.json"
    admission_local_sha = sha256_file(admission_file)

    # ---- canonical member-order evidence (guidance §12) ----
    assert members == HELDOUT_ORDER, "design members_order != canonical admission order"
    feats_dir = ROOT / "d-det/artifacts/code_conditioned_design_2026-10-09/features"
    order_check_file = feats_dir / "feature_member_order_check.json"
    order_check = json.loads(order_check_file.read_text(encoding="utf-8"))
    assert order_check["members_order_equals_admission"] and \
        order_check["member_idx_recomputed_equal"], "feature member-order check failed"
    feature_mapping = {
        "members_order": members,
        "members_order_sha256": order_check["members_order_sha256"],
        "row_mapping_sha256": order_check["row_mapping_sha256"],
        "row_mapping_convention": order_check["row_mapping_convention"],
        "feature_member_order_check_sha256": sha256_file(order_check_file),
        "feature_bundle_sha256": sha256_file(feats_dir / "bundle.npz"),
        "features_manifest_sha256": sha256_file(feats_dir / "features_manifest.json"),
        "bundle_member_ids": design["member_ids"],
    }
    script_sha256 = {"cc_fresh_c0_canonical.py": sha256_file(Path(__file__)),
                     "cc_common.py": sha256_file(ROOT / "scripts" / "cc_common.py")}
    runtime = {"python": sys.version.split()[0], "numpy": np.__version__,
               "sklearn": sklearn.__version__, "platform": platform.platform(),
               "omp_num_threads": os.environ.get("OMP_NUM_THREADS", "")}

    import subprocess
    code_commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                 text=True, cwd=str(ROOT)).stdout.strip()
    spec = component_specs()
    spec_sha = sha256_str(json.dumps(spec, sort_keys=True, ensure_ascii=False))
    texts = [r["code"] for r in rows]

    # ---------- phase 1: pre-fit manifests (with lexical_fit_spec) ----------
    dev_tasks = sorted({r["task_id"] for r in rows if r["split"] == "dev"})
    manifests = {}
    for hi, h in enumerate(HELDOUT_ORDER):
        h_series = design["member_series"][h]
        fit_members = sorted(m for m in members if m != h)
        eval_members = sorted([h] + [m for m in members
                                     if design["member_series"][m] != h_series])
        pos_keys, neg_keys, fit_keys = [], [], []
        for m in eval_members:
            for t in dev_tasks:
                key = f"{m}|{t}"
                (pos_keys if m == h else neg_keys).append(key)
        for m in fit_members:
            for r in rows:
                if r["model_id"] == m and r["split"] == "train":
                    fit_keys.append(f"{m}|{r['task_id']}")
        fit_rows_idx = np.array([i for i, r in enumerate(rows)
                                 if r["model_id"] in set(fit_members) and r["split"] == "train"])
        man = {
            "schema": "code_conditioned_fresh_c0_fold_manifest_v3_canonical",
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "source_status": "server_reconstruction_only",
            "original_bundle_verified": False,
            "correction_note": "supersedes code_conditioned_fresh_c0_corrected_2026-10-10 (member-order canonicalization, guidance §12; fold-fit-only lexical retained)",
            "switches": {"training_allowed": True, "test_read_allowed": False,
                         "generation_allowed": False, "weights_downloaded": False,
                         "code_execution": False},
            "effective_series_map_sha256": map_sha,
            "effective_series_map_canonical": series_map,
            "official_expected_admission_sha256": OFFICIAL_ADMISSION_SHA,
            "server_admission_file": str(admission_file.relative_to(ROOT)),
            "server_admission_file_sha256": admission_local_sha,
            "heldout_member": h,
            "heldout_series": h_series,
            "fit_member_ids": fit_members,
            "eval_member_ids": eval_members,
            "positive_row_sha256": rows_digest(pos_keys),
            "negative_row_sha256": rows_digest(neg_keys),
            "train_row_sha256": rows_digest(fit_keys),
            "eval_task_sha256": rows_digest(dev_tasks),
            "component_specs": spec,
            "component_specs_sha256": spec_sha,
            "code_commit": code_commit,
            "feature_mapping": feature_mapping,
            "script_sha256": script_sha256,
            "runtime_attestation": runtime,
            "lexical_fit_spec": {
                "fit_scope": "fold-fit rows only (fit_rows texts); eval rows transform-only",
                "char_vectorizer_params": CHAR_PARAMS,
                "word_vectorizer_params": WORD_PARAMS,
                "fit_rows_key_sha256": rows_digest(fit_keys),
                "fit_texts_sha256": texts_digest([texts[i] for i in fit_rows_idx]),
                "fit_texts_digest_convention": "sha256 over utf-8 bytes of fit texts in fit-row "
                                               "index order, each followed by 0x00",
                "n_fit_rows": len(fit_rows_idx),
            },
            "counts": {"n_fit_rows": len(fit_keys), "n_eval_rows": len(pos_keys) + len(neg_keys),
                       "n_pos": len(pos_keys), "n_neg": len(neg_keys),
                       "n_eval_tasks": len(dev_tasks)},
        }
        man["manifest_sha256_pre_fit"] = sha256_str(json.dumps(man, sort_keys=True,
                                                               ensure_ascii=False))
        mp = OUT / "manifests" / f"{hi:02d}_{h.replace('--','_')}.json"
        mp.write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
        manifests[h] = man
        log(f"manifest[pre-fit] {h.split('--')[-1][:26]:28s} fit_texts={man['lexical_fit_spec']['fit_texts_sha256'][:12]}…")
    (OUT / "manifests_index.json").write_text(json.dumps(
        {"schema": "code_conditioned_fresh_c0_manifests_index_v3_canonical",
         "written_before_fitting": True, "writing_time_utc": datetime.now(timezone.utc).isoformat(),
         "effective_series_map_sha256": map_sha, "code_commit": code_commit,
         "feature_mapping": feature_mapping,
         "pre_fit_manifest_sha256": {h: manifests[h]["manifest_sha256_pre_fit"] for h in HELDOUT_ORDER},
         "folds": {h: {"heldout_member": h,
                       "manifest_path": str((OUT / "manifests" / f"{HELDOUT_ORDER.index(h):02d}_{h.replace('--','_')}.json").relative_to(ROOT)),
                       **{k: manifests[h][k] for k in
                          ("fit_member_ids", "eval_member_ids", "counts", "lexical_fit_spec")}}
                   for h in HELDOUT_ORDER}},
        ensure_ascii=False, indent=1), encoding="utf-8")
    log("all pre-fit manifests written & hashed BEFORE any vectorizer fitting")

    # ---------- phase 2: fit canonical C0 (fold-fit-only lexical, admission order) ----------
    hy, style, meta, sizelen = b["hy_small"], b["style"], b["meta"], b["sizelen"]
    Xsm = np.hstack([style, meta, sizelen[:, :3]])
    fold_by_h = {f["heldout_generator_member"]: f for f in design["folds"]}
    comp_names = ["semantic", "char_tfidf", "word_tfidf", "style_meta"]
    labels, tposs = {"dev": {}, "inner": {}}, {"dev": {}, "inner": {}}
    fused = {"dev": {}, "inner": {}}
    digests = {"dev": {}, "inner": {}}
    attest = {"dev": {}, "inner": {}}
    member_names_c = np.array([members[i] for i in b["member_idx"]])

    def run_protocol(protocol):
        use_inner = protocol == "inner"
        splitmap = cc.inner_split(design["tasks_all"]) if use_inner else None
        perfold = {}
        for h in HELDOUT_ORDER:
            fold = fold_by_h[h]
            if protocol == "dev":
                sel_tasks = dev_tasks
            else:
                sel_tasks = sorted(t for t in {r["task_id"] for r in rows if r["split"] == "train"}
                                   if splitmap[t] == "inner_dev")
            tpos_of = {t: i for i, t in enumerate(sel_tasks)}
            fit_mask, ev_mask, pos_mask, y = cc.fold_setup(design, fold, split="dev", inner=use_inner)
            y_fit = cc.fold_series_target(design, fold)[fit_mask]
            ev_rows = np.where(ev_mask)[0]
            tp = np.array([tpos_of[rows[i]["task_id"]] for i in ev_rows])
            order = np.argsort(tp, kind="mergesort")
            ev_rows, tp = ev_rows[order], tp[order]
            y_ev = pos_mask[ev_rows].astype(int)
            # --- canonical member-order assertion (guidance §12) ---
            exp_neg = {m for m in members
                       if design["member_series"][m] != design["member_series"][h]}
            assert set(member_names_c[ev_rows][y_ev == 1]) == {h}, \
                "positive rows are not heldout-member rows"
            assert set(member_names_c[ev_rows][y_ev == 0]) == exp_neg, \
                "negative rows are not cross-family member rows"
            fit_rows = np.where(fit_mask)[0]

            # --- CORRECTED lexical: fold-fit-only vectorizers ---
            tr_texts = [texts[i] for i in fit_rows]
            ev_texts = [texts[i] for i in ev_rows]
            vc = TfidfVectorizer(**CHAR_PARAMS)
            Xc_tr = vc.fit_transform(tr_texts)
            Xc_ev = vc.transform(ev_texts)
            vw = TfidfVectorizer(**WORD_PARAMS)
            Xw_tr = vw.fit_transform(tr_texts)
            Xw_ev = vw.transform(ev_texts)
            attest[protocol][h] = {
                "char": {"n_features": int(len(vc.vocabulary_)),
                         "vocab_sha256": vocab_digest(vc.vocabulary_),
                         "idf_sha256": idf_digest(vc.idf_),
                         "idf_min": float(vc.idf_.min()), "idf_max": float(vc.idf_.max())},
                "word": {"n_features": int(len(vw.vocabulary_)),
                         "vocab_sha256": vocab_digest(vw.vocabulary_),
                         "idf_sha256": idf_digest(vw.idf_),
                         "idf_min": float(vw.idf_.min()), "idf_max": float(vw.idf_.max())},
                "fit_scope": "fit_rows only (corrected)",
            }

            comps_tr, comps_ev = {}, {}
            scaler = StandardScaler().fit(hy[fit_rows])
            comps_tr["semantic"] = scaler.transform(hy[fit_rows])
            comps_ev["semantic"] = scaler.transform(hy[ev_rows])
            scaler2 = StandardScaler().fit(Xsm[fit_rows])
            comps_tr["style_meta"] = scaler2.transform(Xsm[fit_rows])
            comps_ev["style_meta"] = scaler2.transform(Xsm[ev_rows])
            comps_tr["char_tfidf"], comps_ev["char_tfidf"] = Xc_tr, Xc_ev
            comps_tr["word_tfidf"], comps_ev["word_tfidf"] = Xw_tr, Xw_ev

            s_tr, s_ev = {}, {}
            for k in comp_names:
                if k in ("char_tfidf", "word_tfidf"):
                    s_tr[k] = sgd_ensemble(comps_tr[k], y_fit, comps_tr[k])
                    s_ev[k] = sgd_ensemble(comps_tr[k], y_fit, comps_ev[k])
                else:
                    clf = LogisticRegression(max_iter=2000, C=1.0).fit(comps_tr[k], y_fit)
                    s_tr[k] = clf.decision_function(comps_tr[k])
                    s_ev[k] = clf.decision_function(comps_ev[k])
            fused_ev, _, stats = cc.zfit_fuse(s_tr, s_ev)
            perfold[h] = {
                "row_level": {"fused": cc.auroc(y_ev, fused_ev),
                              **{k: cc.auroc(y_ev, s_ev[k]) for k in comp_names}},
                "task_macro": cc.task_macro_auroc(y_ev, fused_ev, tp),
                "n_tasks": len(sel_tasks),
            }
            labels[protocol][h], tposs[protocol][h] = y_ev, tp
            fused[protocol][h] = fused_ev
            np.savez_compressed(
                OUT / "local" / f"scores_{protocol}_fold{h.replace('--','_')}.npz",
                ev_rows=ev_rows, y=y_ev, taskpos=tp, fused=fused_ev,
                **{f"s_{k}": s_ev[k] for k in comp_names},
                t_rows=fit_rows, t_fused=np.mean(
                    [((s_tr[k] - stats[k]["mu"]) / stats[k]["sd"]) for k in sorted(s_tr)], axis=0),
                **{f"t_s_{k}": s_tr[k] for k in comp_names})
            digests[protocol][h] = {
                "fused": array_digest(fused_ev),
                "fused_stats": {"min": float(fused_ev.min()), "max": float(fused_ev.max()),
                                "mean": float(fused_ev.mean()), "std": float(fused_ev.std())},
                "n_eval_rows": int(len(ev_rows)), "n_tasks": len(sel_tasks),
                **{f"s_{k}": array_digest(s_ev[k]) for k in comp_names},
                **{f"t_s_{k}": array_digest(s_tr[k]) for k in comp_names},
                "order": "eval rows in taskpos-sorted order (blocks = ASCII-sorted eval members); "
                         "train arrays in fit-row index order",
            }
            log(f"[{protocol}] {h.split('--')[-1][:26]:28s} row={perfold[h]['row_level']['fused']:.4f} "
                f"tm={perfold[h]['task_macro']:.4f} "
                f"charF={attest[protocol][h]['char']['n_features']} "
                f"wordF={attest[protocol][h]['word']['n_features']} ({time.time()-t0:.0f}s)")
        return perfold

    per_fold_dev = run_protocol("dev")
    per_fold_inner = run_protocol("inner")

    # ---------- phase 3: append post-fit attestations to each manifest ----------
    post_hashes = {}
    for hi, h in enumerate(HELDOUT_ORDER):
        mp = OUT / "manifests" / f"{hi:02d}_{h.replace('--','_')}.json"
        man = json.loads(mp.read_text(encoding="utf-8"))
        man["lexical_fit_attestation"] = {
            "written_after_fitting": True,
            "dev_fold": attest["dev"][h],
            "inner_fold": attest["inner"][h],
        }
        man["manifest_sha256_post_fit"] = sha256_str(json.dumps(man, sort_keys=True,
                                                                ensure_ascii=False))
        mp.write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
        post_hashes[h] = man["manifest_sha256_post_fit"]
    log("post-fit lexical attestations appended to all 11 manifests")

    def aggregate(perfold, protocol):
        row_mean = float(np.mean([v["row_level"]["fused"] for v in perfold.values()]))
        tm_mean = float(np.mean([v["task_macro"] for v in perfold.values()]))
        all_y = np.concatenate([labels[protocol][h] for h in HELDOUT_ORDER])
        all_s = np.concatenate([fused[protocol][h] for h in HELDOUT_ORDER])
        boot = cc.bootstrap_metrics(fused[protocol], labels[protocol], tposs[protocol],
                                    n_tasks=perfold[HELDOUT_ORDER[0]]["n_tasks"])
        return {
            "row_level_mean_over_folds": row_mean, "member_macro": row_mean,
            "task_macro_mean_over_folds": tm_mean, "pooled": cc.auroc(all_y, all_s),
            "bootstrap": {"row_level": {"ci95": cc.ci(boot["row_level"])},
                          "task_macro": {"ci95": cc.ci(boot["task_macro"])}},
            "per_component_row_level_mean": {
                k: float(np.mean([v["row_level"][k] for v in perfold.values()])) for k in comp_names},
            "per_fold": perfold,
        }

    dev_agg, inner_agg = aggregate(per_fold_dev, "dev"), aggregate(per_fold_inner, "inner")

    # ---------- phase 4: determinism (refit fold Q-1.5B, corrected path) ----------
    h4 = HELDOUT_ORDER[4]
    fold = fold_by_h[h4]
    fit_mask, ev_mask, pos_mask, y = cc.fold_setup(design, fold, split="dev")
    y_fit = cc.fold_series_target(design, fold)[fit_mask]
    ev_rows = np.where(ev_mask)[0]
    tp = np.array([dev_tasks.index(rows[i]["task_id"]) for i in ev_rows])
    order = np.argsort(tp, kind="mergesort")
    ev_rows = ev_rows[order]
    fit_rows = np.where(fit_mask)[0]
    vc2 = TfidfVectorizer(**CHAR_PARAMS)
    Xc2_tr = vc2.fit_transform([texts[i] for i in fit_rows])
    Xc2_ev = vc2.transform([texts[i] for i in ev_rows])
    s_char2 = sgd_ensemble(Xc2_tr, y_fit, Xc2_ev)
    sc = StandardScaler().fit(hy[fit_rows])
    lr = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(hy[fit_rows]), y_fit)
    s_sem2 = lr.decision_function(sc.transform(hy[ev_rows]))
    z1 = np.load(OUT / "local" / f"scores_dev_fold{h4.replace('--','_')}.npz")
    det = {"semantic": float(np.abs(s_sem2 - z1["s_semantic"]).max()),
           "char_tfidf": float(np.abs(s_char2 - z1["s_char_tfidf"]).max()),
           "vocab_sha256_match": bool(vocab_digest(vc2.vocabulary_) ==
                                      attest["dev"][h4]["char"]["vocab_sha256"])}
    log(f"determinism refit max|delta|: {det}")

    # ---------- phase 5: member-order impact vs permuted corrected run ----------
    # Controlled change vs PREV: only the member_idx canonicalization differs.
    # CodeLlama folds (indices 0-3) have identical fit/eval row sets in both
    # runs -> scores must be identical (invariance evidence). Qwen/DeepSeek
    # folds changed semantics -> report old/new metrics (not a row-mapped diff).
    invariance, perm_impact = {}, {}
    for h in HELDOUT_ORDER:
        zp = np.load(PREV / "local" / f"scores_dev_fold{h.replace('--','_')}.npz")
        zf = np.load(OUT / "local" / f"scores_dev_fold{h.replace('--','_')}.npz")
        prev_pos_members = sorted(set(member_names_c[zp["ev_rows"]][np.asarray(zp["y"]) == 1]))
        entry = {
            "prev_effective_pos_members": prev_pos_members,
            "prev_row_level": cc.auroc(zp["y"], zp["fused"]),
            "prev_task_macro": cc.task_macro_auroc(zp["y"], zp["fused"], zp["taskpos"]),
            "new_row_level": cc.auroc(zf["y"], zf["fused"]),
            "new_task_macro": cc.task_macro_auroc(zf["y"], zf["fused"], zf["taskpos"]),
        }
        if np.array_equal(np.asarray(zp["ev_rows"]), np.asarray(zf["ev_rows"])):
            d = np.abs(np.asarray(zf["fused"]) - np.asarray(zp["fused"]))
            entry.update({"same_eval_rows": True,
                          "same_y": bool(np.array_equal(np.asarray(zp["y"]), np.asarray(zf["y"]))),
                          "max_abs_fused_diff": float(d.max()),
                          "r_fused": float(np.corrcoef(zf["fused"], zp["fused"])[0, 1])})
            invariance[h] = entry
        else:
            entry.update({"same_eval_rows": False})
            perm_impact[h] = entry
    log(f"member-order impact vs permuted corrected run: invariant folds "
        f"{len(invariance)}/11, redefined folds {len(perm_impact)}/11")

    # ---------- phase 6: historical-package diagnostic (canonical mapping) ----------
    diag = {}
    member_names = np.array([members[i] for i in b["member_idx"]])
    task_names = np.array([design["tasks_all"][i] for i in b["task_idx"]])
    for h in HELDOUT_ORDER:
        zz = np.load(REF / h / "p0_scores.npz")
        zf = np.load(OUT / "local" / f"scores_dev_fold{h.replace('--','_')}.npz")
        ev = zf["ev_rows"]
        my_member = [member_names[i] for i in ev]
        my_task = [task_names[i] for i in ev]
        dev_loc = sorted(set(my_task))
        taskrank = {t: i for i, t in enumerate(dev_loc)}
        em = sorted([h] + [m for m in members
                           if design["member_series"][m] != design["member_series"][h]])
        blockrank = {m: i for i, m in enumerate(em)}
        ref_idx = np.array([blockrank[m] * 171 + taskrank[t] for m, t in zip(my_member, my_task)])
        perm = np.argsort(ref_idx)
        mine_p = np.asarray(zf["fused"])[perm]
        hist_p = np.asarray(zz["p0"])[ref_idx[perm]]
        y_p = np.asarray(zf["y"])[perm]
        tp_p = np.array([taskrank[t] for t in my_task])[perm]
        diag[h] = {
            "n": int(len(mine_p)),
            "r": float(np.corrcoef(mine_p, hist_p)[0, 1]),
            "max_abs_diff": float(np.abs(mine_p - hist_p).max()),
            "row_mine": cc.auroc(y_p, mine_p), "row_hist": cc.auroc(y_p, hist_p),
            "tm_mine": cc.task_macro_auroc(y_p, mine_p, tp_p),
            "tm_hist": cc.task_macro_auroc(y_p, hist_p, tp_p),
        }
    log("historical diagnostic complete (canonical member mapping; not the gate)")

    metrics = {
        "schema": "code_conditioned_fresh_c0_canonical_metrics_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "report_first_line": "Canonical-order fresh train/dev C0 (admission-order member_idx + "
                             "fold-fit-only lexical; supersedes code_conditioned_fresh_c0_corrected_2026-10-10)",
        "source_status": "server_reconstruction_only", "original_bundle_verified": False,
        "correction": {
            "reason": "member-order mismatch (cross-side reproduction 2026-10-10; guidance §12)",
            "change": "feature bundle member_idx rebuilt in admission order; members_order_sha256 + "
                      "row_mapping_sha256 recorded in features manifest / order-check file / every C0 manifest",
            "retained": "fold-fit-only lexical fix from code_conditioned_fresh_c0_corrected_2026-10-10",
            "supersedes": "code_conditioned_fresh_c0_corrected_2026-10-10"},
        "effective_series_map_sha256": map_sha,
        "official_expected_admission_sha256": OFFICIAL_ADMISSION_SHA,
        "server_admission_file_sha256": admission_local_sha, "code_commit": code_commit,
        "feature_mapping": feature_mapping,
        "script_sha256": script_sha256,
        "runtime_attestation": runtime,
        "component_specs_sha256": spec_sha,
        "manifest_sha256_pre_fit": {h: manifests[h]["manifest_sha256_pre_fit"] for h in HELDOUT_ORDER},
        "manifest_sha256_post_fit": post_hashes,
        "protocols": {"dev": dev_agg, "inner": inner_agg},
        "lexical_fit_attestations": attest,
        "score_digests": digests,
        "determinism_refit_fold4": det,
        "invariance_cl_folds_vs_permuted_run": invariance,
        "member_order_impact_vs_permuted_run": perm_impact,
        "diagnostic_vs_historical_official_reading": diag,
        "alignment_gate": {
            "row_score_max_abs": 0.001, "metric_abs": 0.001,
            "all_fold_effective_map_logged": True,
            "all_fold_lexical_fit_attested": True,
            "member_order_canonicalized": True,
            "members_order_sha256": feature_mapping["members_order_sha256"],
            "status": "prepared (canonical member order); cross-side score comparison pending",
            "note": ("per guidance §12: bundle rebuilt + hashes recorded; next step is the "
                     "local score-digest comparison under matched runtime or server-attested digests"),
        },
        "switches": {"training_allowed": True, "test_read_allowed": False,
                     "generation_allowed": False, "weights_downloaded": False,
                     "code_execution": False},
        "c1_c3_status": "stopped; prior values remain server-internal diagnostics "
                        "(do_not_rerun_or_upgrade until the canonical C0 passes the "
                        "1e-3 cross-side gate; guidance §12)",
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "score_digests.json").write_text(json.dumps(
        {"schema": "code_conditioned_fresh_c0_canonical_digests_v1", "code_commit": code_commit,
         "digest_convention": "sha256 over C-contiguous float64 bytes of the canonical arrays "
                              "(eval: taskpos-sorted order; train: fit-row order)",
         "members_order_sha256": feature_mapping["members_order_sha256"],
         "row_mapping_sha256": feature_mapping["row_mapping_sha256"],
         "runtime": runtime,
         "digests": digests}, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "spec_sheet.json").write_text(json.dumps(
        {"schema": "code_conditioned_fresh_c0_spec_v3_canonical", "component_specs": spec,
         "component_specs_sha256": spec_sha,
         "canonicalization_note": ("member_idx rebuilt in admission members order "
                                   "(guidance §12, 2026-10-10); fold-fit-only lexical "
                                   "fix retained from v2"),
         "feature_mapping": feature_mapping,
         "environment": {"python": sys.version.split()[0], "numpy": np.__version__,
                         "sklearn": __import__("sklearn").__version__},
         "replication_note": ("run this exact spec on the local side against its own rebuilt "
                              "bundle; compare per-fold vocab hashes (expect 11/11) and score "
                              "digests under matched runtime or server runtime attestation "
                              "(guidance §12)")},
        ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "runtime_attestation.json").write_text(json.dumps(
        {"schema": "code_conditioned_runtime_attestation_v1", "code_commit": code_commit,
         "runtime": runtime, "script_sha256": script_sha256,
         "feature_mapping": feature_mapping,
         "switches": {"training_allowed": True, "test_read_allowed": False,
                      "generation_allowed": False, "weights_downloaded": False,
                      "code_execution": False}}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    (OUT / "logs" / "cc_fresh_c0_canonical.log").write_text("\n".join(LOG) + "\n", encoding="utf-8")
    log(f"done in {time.time()-t0:.0f}s; canonical dev {dev_agg['row_level_mean_over_folds']:.4f}/"
        f"{dev_agg['task_macro_mean_over_folds']:.4f}; inner {inner_agg['row_level_mean_over_folds']:.4f}")


if __name__ == "__main__":
    main()
