"""Shared utilities for the code-conditioned round (C0-C3), server-side.

Frozen protocol pieces (do not tweak without recording a config change):
  - folds: static_preflight/fold_plan.json (11 member-held-out folds)
  - eval set per fold: dev rows of the heldout member (positives) + dev rows of
    other-series members (negatives); same-series siblings excluded (pilot spec)
  - fit set per fold: train rows with model_id != heldout member
  - metrics: row-level AUROC, task-macro AUROC (within-task, averaged over tasks),
    member-macro = mean over folds; task-cluster bootstrap with shared resample
    indices across folds; paired deltas via the same indices
  - z-score fusion: per-component train-fold score standardization, equal weights
  - inner split (train-only direction check): deterministic task hash rule below
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

DESIGN = ROOT / "d-det/artifacts/code_conditioned_design_2026-10-09"
INPUTS = DESIGN / "inputs"
PREFLIGHT = DESIGN / "static_preflight"
FEATS = DESIGN / "features"
R0 = ROOT / "d-det/artifacts/r0_protocol_diff_2026-10-08/local"
BOOT_SEED = 20261009
BOOT_N = 500
INNER_TAG = "cc_inner_split_v1"


def load_design():
    rows = [json.loads(s) for s in (INPUTS / "records_train_dev.jsonl").read_text(
        encoding="utf-8").splitlines() if s.strip()]
    bundle = dict(np.load(FEATS / "bundle.npz"))
    emb_task = np.load(FEATS / "emb_task_small.npz")["emb"]
    folds = json.loads((PREFLIGHT / "fold_plan.json").read_text(encoding="utf-8"))["folds"]
    series_map = json.loads((INPUTS / "family_series_admission.json").read_text(
        encoding="utf-8"))
    series_of = {s["series"]: [m["model_id"] for m in s["members"]] for s in series_map["series"]}
    member_series = {m: s for s, ms in series_of.items() for m in ms}
    members_order = [m["model_id"] for s in series_map["series"] for m in s["members"]]
    tasks_all = json.loads((R0 / "row_index.json").read_text(encoding="utf-8"))["tasks"]
    return {"rows": rows, "bundle": bundle, "emb_task": emb_task, "folds": folds,
            "series_of": series_of, "member_series": member_series,
            "members_order": members_order, "tasks_all": tasks_all}


def inner_split(tasks):
    """Deterministic inner split of train tasks: ~1/5 to inner-dev."""
    out = {}
    for t in tasks:
        h = int(hashlib.sha256(f"{INNER_TAG}|{t}".encode()).hexdigest(), 16)
        out[t] = "inner_dev" if h % 5 == 0 else "inner_train"
    return out


def auroc(y, s):
    """Rank AUROC with ties=0.5 (vectorized, validated in core_fix_metric_audit)."""
    y = np.asarray(y, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    n_pos = float(y.sum())
    n_neg = float(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    s_sorted = s[order]
    rank = np.empty(len(s), dtype=np.float64)
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        rank[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((rank[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def task_macro_auroc(y, s, task_ids):
    vals = []
    for t in np.unique(task_ids):
        m = task_ids == t
        a = auroc(y[m], s[m])
        if not np.isnan(a):
            vals.append(a)
    return float(np.mean(vals)) if vals else float("nan")


def zfit_fuse(train_scores: dict, eval_scores: dict):
    """Equal-weight z-score fusion; stats from train scores. Returns (fused_eval, per_eval_z, stats)."""
    z_eval = {}
    stats = {}
    for k, s_tr in train_scores.items():
        mu, sd = float(np.mean(s_tr)), float(np.std(s_tr))
        sd = max(sd, 1e-8)
        stats[k] = {"mu": mu, "sd": sd}
        z_eval[k] = (np.asarray(eval_scores[k]) - mu) / sd
    keys = sorted(train_scores)
    fused = np.mean([z_eval[k] for k in keys], axis=0)
    return fused, z_eval, stats


def fold_setup(design, fold, split="dev", inner=False):
    """Return (fit_mask, eval_mask, pos_mask) boolean masks over the 10659 rows."""
    b = design["bundle"]
    rows = design["rows"]
    member_idx = b["member_idx"]
    task_idx = b["task_idx"]
    is_train = b["is_train"].astype(bool)
    members = design["members_order"]
    h = fold["heldout_generator_member"]
    hi = members.index(h)
    family = design["member_series"][h]
    ms = np.array([design["member_series"][m] for m in members])
    if not inner:
        fit = is_train & (member_idx != hi)
        ev = (~is_train) & ((member_idx == hi) | (ms[member_idx] != family))
        pos = (~is_train) & (member_idx == hi)
    else:
        tasks = design["tasks_all"]
        inner_of = inner_split(tasks)
        task_in = np.array([inner_of[t] for t in tasks])
        fit = is_train & (member_idx != hi) & (task_in[task_idx] == "inner_train")
        ev = is_train & (task_in[task_idx] == "inner_dev") & (
            (member_idx == hi) | (ms[member_idx] != family))
        pos = is_train & (task_in[task_idx] == "inner_dev") & (member_idx == hi)
    y = pos[ev].astype(int)
    return fit, ev, pos, y


def fold_series_target(design, fold):
    """Binary target for fitting: 1 if row's series == heldout member's series."""
    b = design["bundle"]
    members = design["members_order"]
    h = fold["heldout_generator_member"]
    family = design["member_series"][h]
    ms = np.array([design["member_series"][m] for m in members])
    return (ms[b["member_idx"]] == family).astype(int)


def bootstrap_indices(n_tasks=171, n=BOOT_N, seed=BOOT_SEED):
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_tasks, size=(n, n_tasks))


def resample_rows(ev_mask, task_pos, boot_row, n_tasks=171):
    """rows selected by bootstrap task positions: boot_row = array of task positions (len n_tasks)."""
    idx_by = np.where(ev_mask)[0]
    tp = task_pos[idx_by]
    order = np.argsort(tp, kind="mergesort")
    idx_sorted = idx_by[order]
    tp_sorted = tp[order]
    starts = np.searchsorted(tp_sorted, np.arange(n_tasks), side="left")
    ends = np.searchsorted(tp_sorted, np.arange(n_tasks), side="right")
    out = []
    for q in boot_row:
        out.append(idx_sorted[starts[q]:ends[q]])
    return np.concatenate(out)


def per_task_auroc(y, s, taskpos, n_tasks):
    out = np.full(n_tasks, np.nan)
    for q in range(n_tasks):
        m = taskpos == q
        if m.any():
            out[q] = auroc(y[m], s[m])
    return out


def _rows_for_tasks(taskpos, ev_idx, ntasks_sel, n_tasks):
    starts = np.searchsorted(taskpos, np.arange(n_tasks), side="left")
    ends = np.searchsorted(taskpos, np.arange(n_tasks), side="right")
    return np.concatenate([ev_idx[starts[q]:ends[q]] for q in ntasks_sel])


def bootstrap_metrics(scores_per_fold, labels, taskpos, n_tasks=171):
    """Shared-index task-cluster bootstrap of row-level mean and task-macro mean.

    scores_per_fold: {fid: np.ndarray}; labels/taskpos: {fid: np.ndarray}.
    taskpos must be sorted within fold (positions 0..n_tasks-1 occurring in order).
    """
    boot = bootstrap_indices(n_tasks=n_tasks)
    tm = {fid: per_task_auroc(labels[fid], scores_per_fold[fid], taskpos[fid], n_tasks)
          for fid in labels}
    ev_idx = {fid: np.arange(len(labels[fid])) for fid in labels}
    row_dist, macro_dist = [], []
    for b in range(len(boot)):
        rv, mv = [], []
        for fid in labels:
            sel = _rows_for_tasks(taskpos[fid], ev_idx[fid], boot[b], n_tasks)
            rv.append(auroc(labels[fid][sel], scores_per_fold[fid][sel]))
            mv.append(float(np.nanmean(tm[fid][boot[b]])))
        row_dist.append(float(np.mean(rv)))
        macro_dist.append(float(np.mean(mv)))
    return {"row_level": row_dist, "task_macro": macro_dist, "per_task": tm}


def delta_bootstrap(scores_a, scores_b, labels, taskpos, n_tasks=171):
    """Paired delta bootstrap (a - b) of row-level and task-macro means, shared indices."""
    boot = bootstrap_indices(n_tasks=n_tasks)
    tm_d, ev_idx = {}, {}
    for fid in labels:
        ta = per_task_auroc(labels[fid], scores_a[fid], taskpos[fid], n_tasks)
        tb = per_task_auroc(labels[fid], scores_b[fid], taskpos[fid], n_tasks)
        tm_d[fid] = ta - tb
        ev_idx[fid] = np.arange(len(labels[fid]))
    dr, dm = [], []
    for b in range(len(boot)):
        rv, mv = [], []
        for fid in labels:
            sel = _rows_for_tasks(taskpos[fid], ev_idx[fid], boot[b], n_tasks)
            rv.append(auroc(labels[fid][sel], scores_a[fid][sel])
                      - auroc(labels[fid][sel], scores_b[fid][sel]))
            mv.append(float(np.nanmean(tm_d[fid][boot[b]])))
        dr.append(float(np.mean(rv)))
        dm.append(float(np.mean(mv)))
    return {"row_level": dr, "task_macro": dm}


def ci(v):
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def git_meta(out_dir: Path):
    import subprocess
    def run(cmd):
        return subprocess.run(cmd, capture_output=True, text=True,
                              cwd=str(ROOT)).stdout
    (out_dir / "git_head.txt").write_text(run(["git", "rev-parse", "HEAD"]), encoding="utf-8")
    (out_dir / "git_status.txt").write_text(
        run(["git", "status", "--short"]) + run(["git", "diff", "--stat"]), encoding="utf-8")


def write_sha256sums(out_dir: Path):
    h = hashlib.sha256
    files = sorted(p for p in out_dir.rglob("*") if p.is_file()
                   and p.name != "SHA256SUMS.txt" and "local" not in p.parts)
    lines = []
    for p in files:
        hp = h()
        with p.open("rb") as f:
            for b in iter(lambda: f.read(1 << 20), b""):
                hp.update(b)
        lines.append(f"{hp.hexdigest()}  {p.relative_to(out_dir)}")
    (out_dir / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
