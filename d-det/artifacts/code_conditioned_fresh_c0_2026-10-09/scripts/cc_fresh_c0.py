"""Fresh train/dev C0 reconstruction (2026-10-09, §9 mandate).

Per `code_conditioned_p0_provenance_adjudication_2026-10-09/fresh_c0_protocol.json`:
  - write per-fold manifests (effective map / heldout / fit+eval member ids / row &
    task hashes / component specs / code commit) BEFORE fitting, then hash them;
  - fit the frozen four-component C0 (identical spec to the 2968ff9 C0) on the
    user-provided official admission map; train/dev only;
  - emit per-fold score digests (sha256 over canonical float64 arrays) for future
    cross-side alignment;
  - determinism self-check: re-fit one fold and compare scores bit-wise;
  - diagnostic comparison vs the historical reference package under the official
    reading (labelled diagnostic; provenance incomplete per adjudication).

Switches: training_allowed=true (train/dev only); test_read=false; generation=false;
weights_downloaded=false; code_execution=false.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cc_common as cc

ROOT = cc.ROOT
OUT = ROOT / "d-det/artifacts/code_conditioned_fresh_c0_2026-10-09"
REF = ROOT / "d-det/artifacts/code_conditioned_p0_reference_2026-10-09/folds"
OLD_C0 = ROOT / "d-det/artifacts/code_conditioned_c0_strong_p0_2026-10-09/local"
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


def array_digest(a: np.ndarray) -> str:
    a = np.ascontiguousarray(np.asarray(a, dtype=np.float64))
    return sha256_bytes(a.tobytes())


def component_specs() -> dict:
    return {
        "semantic": {
            "input_block": "hy_small = frozen CodeT5-small mean-pooled code embedding (512-d), "
                           "from r0 corpus cache row = gmodel_idx*nT*2 + task_idx*2 + 1 (instruct)",
            "scaler": "StandardScaler fit on fold-fit rows (server sklearn)",
            "model": "LogisticRegression(max_iter=2000, C=1.0)",
            "score": "decision_function",
        },
        "char_tfidf": {
            "vectorizer": "TfidfVectorizer(analyzer=char_wb, ngram_range=(2,5), min_df=1, "
                          "sublinear_tf=True, lowercase=False)",
            "fit_scope": "fold-fit rows texts only (fold-train-only vocabulary & IDF)",
            "model": "SGDClassifier(loss=log_loss, alpha=1e-6) x seeds(0,1,2), 5 epochs of"
                     " shuffled partial_fit, batch=4096, seed-local RNG permutation",
            "score": "mean decision_function over 3 seeds",
        },
        "word_tfidf": {
            "vectorizer": "TfidfVectorizer(analyzer=word, token_pattern=[A-Za-z_][A-Za-z0-9_]*, "
                          "ngram_range=(1,3), min_df=1, sublinear_tf=True, lowercase=False)",
            "fit_scope": "fold-fit rows texts only",
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
    import subprocess
    code_commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                 text=True, cwd=str(ROOT)).stdout.strip()
    spec = component_specs()
    spec_sha = sha256_str(json.dumps(spec, sort_keys=True, ensure_ascii=False))

    # ---------- phase 1: manifests BEFORE fitting ----------
    dev_tasks = sorted({r["task_id"] for r in rows if r["split"] == "dev"})
    idx_of = {(r["model_id"], r["task_id"]): i for i, r in enumerate(rows)}
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
        man = {
            "schema": "code_conditioned_fresh_c0_fold_manifest_v1",
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "source_status": "server_reconstruction_only",
            "original_bundle_verified": False,
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
            "counts": {"n_fit_rows": len(fit_keys), "n_eval_rows": len(pos_keys) + len(neg_keys),
                       "n_pos": len(pos_keys), "n_neg": len(neg_keys),
                       "n_eval_tasks": len(dev_tasks)},
        }
        man["manifest_sha256"] = sha256_str(json.dumps(man, sort_keys=True, ensure_ascii=False))
        mp = OUT / "manifests" / f"{hi:02d}_{h.replace('--','_')}.json"
        mp.write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
        manifests[h] = {**{k: man[k] for k in man if k != "component_specs"},
                        "manifest_file_sha256": sha256_file(mp),
                        "manifest_path": str(mp.relative_to(ROOT))}
        log(f"manifest[before-fit] {h.split('--')[-1][:26]:28s} pos {man['counts']['n_pos']} "
            f"neg {man['counts']['n_neg']} fit {man['counts']['n_fit_rows']}")
    (OUT / "manifests_index.json").write_text(json.dumps(
        {"schema": "code_conditioned_fresh_c0_manifests_index_v1",
         "written_before_fitting": True, "writing_time_utc": datetime.now(timezone.utc).isoformat(),
         "effective_series_map_sha256": map_sha, "code_commit": code_commit,
         "folds": manifests}, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"all manifests written & hashed BEFORE fitting; map_sha={map_sha[:16]}…")

    # ---------- phase 2: fit C0 (identical spec to 2968ff9 round) ----------
    texts = [r["code"] for r in rows]
    vc = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=1,
                         sublinear_tf=True, lowercase=False)
    Xc = vc.fit_transform(texts)
    vw = TfidfVectorizer(analyzer="word", token_pattern=r"[A-Za-z_][A-Za-z0-9_]*",
                         ngram_range=(1, 3), min_df=1, sublinear_tf=True, lowercase=False)
    Xw = vw.fit_transform(texts)
    log(f"tfidf char {Xc.shape} word {Xw.shape} ({time.time()-t0:.0f}s)")
    hy, style, meta, sizelen = b["hy_small"], b["style"], b["meta"], b["sizelen"]
    Xsm = np.hstack([style, meta, sizelen[:, :3]])

    fold_by_h = {f["heldout_generator_member"]: f for f in design["folds"]}
    per_fold_dev, per_fold_inner = {}, {}
    labels, tposs = {"dev": {}, "inner": {}}, {"dev": {}, "inner": {}}
    fused = {"dev": {}, "inner": {}}
    digests = {"dev": {}, "inner": {}}
    comp_names = ["semantic", "char_tfidf", "word_tfidf", "style_meta"]

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
            fit_rows = np.where(fit_mask)[0]

            comps_tr, comps_ev = {}, {}
            scaler = StandardScaler().fit(hy[fit_rows])
            comps_tr["semantic"] = scaler.transform(hy[fit_rows])
            comps_ev["semantic"] = scaler.transform(hy[ev_rows])
            scaler2 = StandardScaler().fit(Xsm[fit_rows])
            comps_tr["style_meta"] = scaler2.transform(Xsm[fit_rows])
            comps_ev["style_meta"] = scaler2.transform(Xsm[ev_rows])
            comps_tr["char_tfidf"], comps_ev["char_tfidf"] = Xc[fit_rows], Xc[ev_rows]
            comps_tr["word_tfidf"], comps_ev["word_tfidf"] = Xw[fit_rows], Xw[ev_rows]

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
                "n_eval_rows": int(len(ev_rows)), "n_tasks": len(sel_tasks),
                **{f"s_{k}": array_digest(s_ev[k]) for k in comp_names},
                **{f"t_s_{k}": array_digest(s_tr[k]) for k in comp_names},
                "order": "eval rows in taskpos-sorted order (blocks = ASCII-sorted eval members); "
                         "train arrays in fit-row index order",
            }
            log(f"[{protocol}] {h.split('--')[-1][:26]:28s} row={perfold[h]['row_level']['fused']:.4f} "
                f"tm={perfold[h]['task_macro']:.4f} ({time.time()-t0:.0f}s)")
        return perfold

    per_fold_dev = run_protocol("dev")
    per_fold_inner = run_protocol("inner")

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

    # ---------- phase 3: determinism self-check (refit fold 4 = Q-1.5B) ----------
    h4 = HELDOUT_ORDER[4]
    fold = fold_by_h[h4]
    fit_mask, ev_mask, pos_mask, y = cc.fold_setup(design, fold, split="dev")
    y_fit = cc.fold_series_target(design, fold)[fit_mask]
    ev_rows = np.where(ev_mask)[0]
    tp = np.array([dev_tasks.index(rows[i]["task_id"]) for i in ev_rows])
    order = np.argsort(tp, kind="mergesort")
    ev_rows = ev_rows[order]
    fit_rows = np.where(fit_mask)[0]
    sc = StandardScaler().fit(hy[fit_rows])
    lr = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(hy[fit_rows]), y_fit)
    s_sem2 = lr.decision_function(sc.transform(hy[ev_rows]))
    s_char2 = sgd_ensemble(Xc[fit_rows], y_fit, Xc[ev_rows])
    z1 = np.load(OUT / "local" / f"scores_dev_fold{h4.replace('--','_')}.npz")
    det = {"semantic": float(np.abs(s_sem2 - z1["s_semantic"]).max()),
           "char_tfidf": float(np.abs(s_char2 - z1["s_char_tfidf"]).max())}
    log(f"determinism refit max|delta|: {det}")

    # ---------- phase 4: diagnostic vs historical package (official reading) ----------
    # 行身份对齐：本新 npz 为任务序；历史包为成员块序（ASCII 排序的 eval 成员 × 排序任务）。
    # 用 (member, task) 恒等把两侧逐行对齐后再比较（官方读法；供记录，非闸门）。
    member_names = np.array([members[i] for i in b["member_idx"]])
    task_names = np.array([design["tasks_all"][i] for i in b["task_idx"]])
    diag = {}
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
        perm = np.argsort(ref_idx)                      # my rows reindexed by ref position
        mine_p = np.asarray(zf["fused"])[perm]
        hist_p = np.asarray(zz["p0"])[ref_idx[perm]]
        y_p = np.asarray(zf["y"])[perm]
        tp_p = np.array([taskrank[t] for t in my_task])[perm]
        y_ok = bool(np.array_equal(y_p, np.asarray(zz["y"])[ref_idx[perm]]))
        diag[h] = {
            "n": int(len(mine_p)), "y_consistent": y_ok,
            "r": float(np.corrcoef(mine_p, hist_p)[0, 1]),
            "max_abs_diff": float(np.abs(mine_p - hist_p).max()),
            "row_mine": cc.auroc(y_p, mine_p), "row_hist": cc.auroc(y_p, hist_p),
            "tm_mine": cc.task_macro_auroc(y_p, mine_p, tp_p),
            "tm_hist": cc.task_macro_auroc(y_p, hist_p, tp_p),
        }
    # fresh vs my previous C0 (2968ff9) bit-comparison
    prior = {}
    for h in HELDOUT_ORDER:
        p = OLD_C0 / f"scores_dev_fold{h.replace('--','_')}.npz"
        zf = np.load(OUT / "local" / f"scores_dev_fold{h.replace('--','_')}.npz")
        zp = np.load(p)
        prior[h] = {"max_abs_diff_fused": float(np.abs(np.asarray(zf["fused"]) -
                                                       np.asarray(zp["fused"])).max()),
                    "ev_rows_equal": bool(np.array_equal(zf["ev_rows"], zp["ev_rows"]))}
    log("diagnostic vs historical & vs 2968ff9 C0 complete")

    metrics = {
        "schema": "code_conditioned_fresh_c0_metrics_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "report_first_line": "Fresh train/dev C0 (per §9; manifests written before fitting)",
        "source_status": "server_reconstruction_only", "original_bundle_verified": False,
        "effective_series_map_sha256": map_sha, "official_expected_admission_sha256": OFFICIAL_ADMISSION_SHA,
        "server_admission_file_sha256": admission_local_sha, "code_commit": code_commit,
        "component_specs_sha256": spec_sha,
        "protocols": {"dev": dev_agg, "inner": inner_agg},
        "score_digests": digests,
        "determinism_refit_fold4": det,
        "self_check_vs_2968ff9_c0": prior,
        "diagnostic_vs_historical_official_reading": diag,
        "alignment_gate": {
            "row_score_max_abs": 0.001, "metric_abs": 0.001,
            "all_fold_effective_map_logged": True,
            "status": "prepared; cross-side counterpart required (fresh local C0 under the "
                      "logged spec, or adoption of the logged spec on the local side)",
            "note": "historical package is map_consistent_but_spec_and_per_fold_provenance_incomplete; "
                    "diagnostic numbers are NOT the gate result",
        },
        "switches": {"training_allowed": True, "test_read_allowed": False,
                     "generation_allowed": False, "weights_downloaded": False,
                     "code_execution": False},
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "score_digests.json").write_text(json.dumps(
        {"schema": "code_conditioned_fresh_c0_digests_v1", "code_commit": code_commit,
         "digest_convention": "sha256 over C-contiguous float64 bytes of the canonical arrays "
                              "(eval: taskpos-sorted order; train: fit-row order)",
         "digests": digests}, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "spec_sheet.json").write_text(json.dumps(
        {"schema": "code_conditioned_fresh_c0_spec_v1", "component_specs": spec,
         "component_specs_sha256": spec_sha,
         "environment": {"python": sys.version.split()[0], "numpy": np.__version__,
                         "sklearn": __import__("sklearn").__version__},
         "replication_note": "run this exact spec on the local side; compare score_digests.json "
                             "hashes / max-abs-diff to satisfy the 1e-3 alignment gate"},
        ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "logs" / "cc_fresh_c0.log").write_text("\n".join(LOG) + "\n", encoding="utf-8")
    log(f"done in {time.time()-t0:.0f}s; dev {dev_agg['row_level_mean_over_folds']:.4f}/"
        f"{dev_agg['task_macro_mean_over_folds']:.4f}; inner {inner_agg['row_level_mean_over_folds']:.4f}")


if __name__ == "__main__":
    main()
