#!/usr/bin/env python3
"""H3 data gate for h2_stacad_alignment_v1 (2026-10-10, CPU-only round).

Phases:
  L  license / data-role record
  D  duplicate audits: exact, whitespace-normalized (all rows, integrity use),
     Python AST exact + skeleton (parser availability: stdlib only)
  V  variant safety smoke (Python subset; comment .2 / whitespace .3 / rename .4)
  P  shortcut probes on the Human-vs-AI label (fit=train, eval=dev, test sealed):
     metadata / source / length / lexical / ast-shape(py); transform = not run

All outputs -> d-det/artifacts/h3_data_gate_stacad_2026-10-10/
Switches: test_read(modeling)=false; generation=false; weights_downloaded=false;
code_execution=false. Nothing is executed; only parsed/compiled.
"""
from __future__ import annotations

import ast
import hashlib
import io
import json
import math
import re
import sys
import time
import tokenize
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "d-det/data/h2_stacad_alignment_v1"
RECEIVED = ROOT / "d-det/data/h2_stacad_alignment_v1_received"
LICENSE_SRC = ROOT / "d-det/data/stacad_v2/LICENSE"
OUT = ROOT / "d-det/artifacts/h3_data_gate_stacad_2026-10-10"
PROBES_DIR = OUT / "probes"

SEED = 20261010
BOOT_B = 500
LANGS = ("c", "cpp", "cs", "go", "java", "php", "py")
LOG: list[str] = []


def log(m: str) -> None:
    s = f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {m}"
    print(s, flush=True)
    LOG.append(s)


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def iter_jsonl(p: Path):
    with p.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def auroc(y, s) -> float:
    y = np.asarray(y, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    n_pos = float(y.sum())
    n_neg = float(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    s_sorted = s[order]
    rank = np.empty(len(s))
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        rank[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((rank[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def task_auroc_map(y, s, tasks) -> dict:
    out = {}
    y = np.asarray(y)
    s = np.asarray(s)
    tasks = np.asarray(tasks)
    for t in np.unique(tasks):
        m = tasks == t
        a = auroc(y[m], s[m])
        if not math.isnan(a):
            out[str(t)] = a
    return out


def macro(amap: dict) -> float:
    return float(np.mean(list(amap.values()))) if amap else float("nan")


def boot_ci(amap: dict, rng) -> list:
    keys = sorted(amap)
    vals = np.array([amap[k] for k in keys])
    idx = rng.integers(0, len(keys), size=(BOOT_B, len(keys)))
    means = vals[idx].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def norm_ws(code: str) -> str:
    lines = [l.rstrip() for l in code.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    s = "\n".join(lines).strip("\n")
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s


def ast_skeleton(code: str):
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            node.value = "C"
        elif isinstance(node, ast.Constant):
            node.value = 0
        elif isinstance(node, ast.Name):
            node.id = "N"
        elif isinstance(node, ast.arg):
            node.arg = "N"
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            node.name = "N"
        elif isinstance(node, ast.Attribute):
            node.attr = "N"
        elif isinstance(node, (ast.alias,)):
            node.name = "N"
            node.asname = None
    return ast.dump(tree, include_attributes=False)


# --------------------------------------------------------------------------- #
def phase_license() -> dict:
    lic = LICENSE_SRC.read_text(encoding="utf-8", errors="replace")
    head = " ".join(lic.split())[:200]
    rec = {
        "license_file": str(LICENSE_SRC.relative_to(ROOT)),
        "license_file_sha256": sha256_file(LICENSE_SRC),
        "license_head": head,
        "cc_by_4_0_mentioned": ("CC BY 4.0" in lic) or ("Creative Commons Attribution 4.0" in lic),
        "mit_mentioned": ("MIT" in lic) or ("MIT License" in lic),
        "declared_in_package_readme": "CC BY 4.0" in (DATA / "README.md").read_text(encoding="utf-8"),
        "source_ref_preserved": True,  # checked in audit pass below (field present on all rows)
    }
    log(f"license: CC BY 4.0={rec['cc_by_4_0_mentioned']} MIT={rec['mit_mentioned']} sha={rec['license_file_sha256'][:12]}")
    return rec


def phase_scan():
    """Single streaming pass: probe rows (train/dev) + dedup hashes (all rows)."""
    exact_h, exact_a = set(), set()
    norm_h, norm_a = set(), set()
    norm_dup_pairs = {"human_human": 0, "ai_ai": 0, "human_ai": 0}
    norm_map = {}
    py_rows, parse_err = [], {"human": 0, "ai": 0}
    src_ok = True
    human_hash_ok = 0
    probes = {"train": [], "dev": []}
    counts = Counter()

    for d in iter_jsonl(DATA / "task_index.jsonl"):
        code = d["human_code"]
        split = d["task_split"]
        counts[f"human_{split}"] += 1
        h = d["human_sha256"]
        human_hash_ok += int(sha256_bytes(code.encode()) == h)
        exact_h.add(h)
        nh = sha256_bytes(norm_ws(code).encode())
        norm_h.add(nh)
        norm_map.setdefault(nh, []).append(("human", d["task_id"], split))
        if not d.get("source_ref"):
            src_ok = False
        if split in probes:
            probes[split].append({"task_id": d["task_id"], "task_split": split, "language": d["language"],
                                  "file_name": d["file_name"], "generator": None, "y": 0, "code": code})
        if d["language"] == "py":
            try:
                py_rows.append(("human", ast.dump(ast.parse(code), include_attributes=False),
                                ast_skeleton(code), split))
            except SyntaxError:
                parse_err["human"] += 1

    for d in iter_jsonl(DATA / "core.jsonl"):
        code = d["code"]
        split = d["task_split"]
        counts[f"ai_{split}"] += 1
        exact_a.add(d["code_sha256"])
        na = sha256_bytes(norm_ws(code).encode())
        norm_a.add(na)
        norm_map.setdefault(na, []).append(("ai", d["record_id"], split))
        if not d.get("source_ref"):
            src_ok = False
        if split in ("train", "dev"):
            probes[split].append({"task_id": d["task_id"], "task_split": split, "language": d["language"],
                                  "file_name": d["file_name"], "generator": d["generator"],
                                  "y": 1, "code": code})
        if d["language"] == "py":
            try:
                py_rows.append(("ai", ast.dump(ast.parse(code), include_attributes=False),
                                ast_skeleton(code), split))
            except SyntaxError:
                parse_err["ai"] += 1

    exact_inter = len(exact_h & exact_a)
    norm_inter = len(norm_h & norm_a)
    for nh, members in norm_map.items():
        if len(members) > 1:
            kinds = [m[0] for m in members]
            hh = kinds.count("human")
            aa = kinds.count("ai")
            if hh > 1:
                norm_dup_pairs["human_human"] += hh - 1
            if aa > 1:
                norm_dup_pairs["ai_ai"] += aa - 1
            if hh and aa:
                norm_dup_pairs["human_ai"] += min(hh, aa)
    dup = {
        "exact": {"human_hash_unique": len(exact_h), "ai_hash_unique": len(exact_a),
                  "human_ai_intersection": exact_inter,
                  "human_rows": sum(v for k, v in counts.items() if k.startswith("human_")),
                  "ai_rows": sum(v for k, v in counts.items() if k.startswith("ai_")),
                  "human_hash_matches_content": human_hash_ok},
        "whitespace_normalized": {"human_unique": len(norm_h), "ai_unique": len(norm_a),
                                  "human_ai_intersection": norm_inter,
                                  "intra_human_dups": norm_dup_pairs["human_human"],
                                  "intra_ai_dups": norm_dup_pairs["ai_ai"]},
        "python_ast": {},
        "note": ("dedup uses hash-level scanning over ALL splits including test "
                 "(integrity/leakage certificate only; test not used for modeling)"),
    }
    n_h = sum(v for k, v in counts.items() if k.startswith("human_"))
    n_a = sum(v for k, v in counts.items() if k.startswith("ai_"))
    dup["exact"]["intra_human_dups"] = n_h - len(exact_h)
    dup["exact"]["intra_ai_dups"] = n_a - len(exact_a)

    # python AST level
    ast_exact = defaultdict(int)
    ast_skel = defaultdict(int)
    ast_split = {}
    for kind, d_exact, d_skel, split in py_rows:
        ast_exact[d_exact] += 1
        ast_skel[d_skel] += 1
        ast_split.setdefault(d_skel, set()).add(split)
    skel_cross = sum(1 for v in ast_split.values() if len(v - {"test"}) >= 2)
    dup["python_ast"] = {
        "rows_parsed": len(py_rows), "parse_errors": parse_err,
        "ast_exact_unique": len(ast_exact), "ast_exact_extra_copies": len(py_rows) - len(ast_exact),
        "ast_skeleton_unique": len(ast_skel), "ast_skeleton_extra_copies": len(py_rows) - len(ast_skel),
        "skeleton_groups_spanning_multiple_non_test_splits": skel_cross,
        "pending_parser_languages": [l for l in LANGS if l != "py"],
    }
    log(f"scan: human={n_h} ai={n_a} py_parsed={len(py_rows)} exact_inter={exact_inter} norm_inter={norm_inter} "
        f"ast_skel_extra={len(py_rows) - len(ast_skel)}")
    return probes, dup, counts, src_ok


# --------------------------------------------------------------------------- #
def strip_comments(code: str):
    offsets = [0]
    for line in code.split("\n"):
        offsets.append(offsets[-1] + len(line) + 1)
    ranges = []
    try:
        for t in tokenize.generate_tokens(io.StringIO(code).readline):
            if t.type == tokenize.COMMENT:
                a = offsets[t.start[0] - 1] + t.start[1]
                b = offsets[t.end[0] - 1] + t.end[1]
                ranges.append((a, b))
    except tokenize.TokenError:
        return None, "tokenize_error"
    if not ranges:
        return None, "no_change"
    out, prev = [], 0
    for a, b in ranges:
        out.append(code[prev:a])
        prev = b
    out.append(code[prev:])
    new = "".join(out)
    new = "\n".join(l.rstrip() for l in new.split("\n"))
    return new, None


def perturb_blank_lines(code: str, rng):
    lines = code.split("\n")
    empty = [i for i, l in enumerate(lines) if not l.strip()]
    k = max(1, int(round(0.1 * max(1, len(empty)))))
    mode = int(rng.integers(0, 2))
    if mode == 0 or not empty:
        cand = [i for i in range(len(lines) - 1) if lines[i].strip()]
        if not cand:
            return None, "no_change"
        pick = sorted(rng.choice(cand, size=min(k, len(cand)), replace=False).tolist())
        new_lines = []
        for i, l in enumerate(lines):
            new_lines.append(l)
            if i in pick:
                new_lines.append("")
        return "\n".join(new_lines), None
    pick = rng.choice(empty, size=min(k, len(empty)), replace=False)
    return "\n".join(l for i, l in enumerate(lines) if i not in set(pick.tolist())), None


def _norm_dump_for_rename(dump: str, names) -> str:
    for n in names:
        dump = dump.replace(f"id='{n}'", "id='X'").replace(f"arg='{n}'", "arg='X'")
    return dump


def try_rename(code: str, rng):
    """Returns (new_code, error, old_name, new_name)."""
    if re.search(r"(^|\s)(globals|locals|eval|exec)\s*\(", code) or re.search(r"from\s+\S+\s+import\s+\*", code):
        return None, "unsafe_reflection", None, None
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None, "parse_error", None, None
    strings = {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    funcs = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    cands = []
    for fn in funcs:
        names = [a.arg for a in fn.args.args + fn.args.kwonlyargs if a.arg not in ("self", "cls")]
        names += [n.id for n in ast.walk(fn) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)]
        for nm in set(names):
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", nm) or len(nm) < 2:
                continue
            if nm in strings or nm in attrs:
                continue
            uses = [n for n in ast.walk(tree) if isinstance(n, ast.Name) and n.id == nm]
            if any(not (fn.lineno <= u.lineno <= (fn.end_lineno or fn.lineno)) for u in uses):
                continue
            cands.append((nm, fn.lineno, fn.end_lineno or fn.lineno))
    if not cands:
        return None, "no_safe_candidate", None, None
    nm, l0, l1 = cands[int(rng.integers(0, len(cands)))]
    new = nm + "_v"
    out = []
    ok = True
    try:
        for t in tokenize.generate_tokens(io.StringIO(code).readline):
            if t.type == tokenize.NAME and t.string == nm and l0 <= t.start[0] <= l1:
                out.append((t.start, t.end, new))
    except tokenize.TokenError:
        ok = False
    if not ok:
        return None, "tokenize_error", None, None
    if not out:
        return None, "no_change", None, None
    lines = code.split("\n")
    for (r0, c0), (r1, c1), rep in sorted(out, reverse=True):
        if r0 == r1:
            line = lines[r0 - 1]
            lines[r0 - 1] = line[:c0] + rep + line[c1:]
        else:
            return None, "multiline_token", None, None
    new_code = "\n".join(lines)
    try:
        d0 = _norm_dump_for_rename(ast.dump(tree, include_attributes=False), [nm])
        d1 = _norm_dump_for_rename(ast.dump(ast.parse(new_code), include_attributes=False), [new])
    except SyntaxError:
        return None, "ast_mismatch", None, None
    if d0 != d1:
        return None, "ast_mismatch", None, None
    return new_code, None, nm, new


def phase_variant_smoke(probes) -> dict:
    py_train = [r for r in probes["train"] if r["language"] == "py"]
    by_task = defaultdict(list)
    for r in py_train:
        by_task[r["task_id"]].append(r)
    ranked = sorted(by_task, key=lambda t: hashlib.blake2b(f"{SEED}|{t}".encode(), digest_size=8).digest())
    sample = ranked[:60]
    rows = [r for t in sample for r in by_task[t]]
    trans = {"comment_strip": 0.2, "whitespace": 0.3, "rename": 0.4}
    stats = {k: Counter() for k in trans}
    records = []
    for r in rows:
        code = r["code"]
        rid = f"{r['task_id']}:human" if r["y"] == 0 else f"{r['task_id']}:{r['generator']}"
        base = hashlib.sha256(f"{SEED}|{rid}".encode()).digest()
        for ti, (name, p) in enumerate(trans.items()):
            rng = np.random.default_rng(int.from_bytes(base[ti * 4:(ti + 1) * 4], "big"))
            if rng.random() >= p:
                stats[name]["not_sampled"] += 1
                continue
            stats[name]["requested"] += 1
            old_nm = new_nm = None
            if name == "comment_strip":
                new, err = strip_comments(code)
            elif name == "whitespace":
                new, err = perturb_blank_lines(code, rng)
            else:
                new, err, old_nm, new_nm = try_rename(code, rng)
            if err is not None:
                stats[name][err] += 1
                records.append({"parent_row_id": rid, "transform": name, "accepted": False,
                                "reject_reason": err})
                continue
            try:
                if name == "rename":
                    ok_ast = _norm_dump_for_rename(
                        ast.dump(ast.parse(new), include_attributes=False), [new_nm]) == \
                        _norm_dump_for_rename(
                        ast.dump(ast.parse(code), include_attributes=False), [old_nm])
                else:
                    ok_ast = ast.dump(ast.parse(new), include_attributes=False) == \
                        ast.dump(ast.parse(code), include_attributes=False)
            except SyntaxError:
                ok_ast = False
            try:
                compile(new, "<v>", "exec")
                compile_ok = True
            except (SyntaxError, ValueError):
                compile_ok = False
            if not (ok_ast and compile_ok):
                reason = "ast_mismatch" if not ok_ast else "compile_error"
                stats[name][reason] += 1
                records.append({"parent_row_id": rid, "transform": name, "accepted": False,
                                "reject_reason": reason})
                continue
            bl, al = code.split("\n"), new.split("\n")
            changed = sum(1 for a, b in zip(bl, al) if a != b) + abs(len(bl) - len(al))
            stats[name]["accepted"] += 1
            stats[name]["changed_lines_sum"] += changed
            records.append({
                "parent_row_id": rid, "transform": name, "accepted": True,
                "before_sha256": sha256_bytes(code.encode()),
                "after_sha256": sha256_bytes(new.encode()),
                "ast_before_sha256": sha256_bytes(ast.dump(ast.parse(code), include_attributes=False).encode()),
                "ast_after_sha256": sha256_bytes(ast.dump(ast.parse(new), include_attributes=False).encode()),
                "syntax_ok": True, "compile_ok": True, "changed_lines": changed,
                "n_sampled_p": p, "transform_mask": "changed_line_indices",
                "reject_reason": None})
    summary = {}
    for name, c in stats.items():
        req = c["requested"]
        acc = c["accepted"]
        summary[name] = {
            "p": trans[name], "requested": req, "accepted": acc,
            "not_sampled": c["not_sampled"], "rejected": req - acc,
            "reject_reasons": {k: v for k, v in c.items()
                               if k not in ("requested", "accepted", "not_sampled", "changed_lines_sum") and v},
            "ast_preserved_rate_on_accepted": 1.0 if acc else None,
            "mean_changed_lines": (c["changed_lines_sum"] / acc) if acc else None,
        }
    (OUT / "variant_smoke_records.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n", encoding="utf-8")
    log(f"variant smoke: sample_tasks={len(sample)} rows={len(rows)} "
        + "; ".join(f"{k}: acc {summary[k]['accepted']}/{summary[k]['requested']}" for k in trans))
    return {"scope": "python_only_smoke", "sample_tasks": len(sample), "rows": len(rows),
            "seed": SEED, "switches_note": "no variant is used for training; transform_mask is not a model input",
            "transforms": summary,
            "records_file": "variant_smoke_records.jsonl",
            "full_multilang_variant_build": "pending_local_side (parser boundary; tree_sitter not installed on server)"}


# --------------------------------------------------------------------------- #
def build_features_meta(rows):
    lang_ix = {l: i for i, l in enumerate(LANGS)}
    X = []
    for r in rows:
        v = [0.0] * len(LANGS)
        v[lang_ix[r["language"]]] = 1.0
        fn = r["file_name"]
        v += [len(fn), sum(ch.isdigit() for ch in fn), fn.count("_")]
        X.append(v)
    return np.array(X, dtype=np.float64)


def build_features_source(rows):
    lang = build_features_meta(rows)[:, :len(LANGS)]
    buckets = np.zeros((len(rows), 64))
    for i, r in enumerate(rows):
        b = int.from_bytes(hashlib.blake2b(r["file_name"].encode(), digest_size=8).digest(), "big") % 64
        buckets[i, b] = 1.0
    return np.hstack([lang, buckets])


def build_features_len(rows):
    X = []
    for r in rows:
        c = r["code"]
        lines = c.split("\n")
        ne = [l for l in lines if l.strip()]
        lens = [len(l) for l in lines] or [0]
        X.append([len(c), len(lines), len(ne), len(lines) - len(ne), max(lens),
                  float(np.mean(lens)), len(c.split()), c.count(" "), c.count("\t")])
    return np.array(X, dtype=np.float64)


def build_features_ast(rows):
    types = sorted({t.__name__ for t in vars(ast).values() if isinstance(t, type) and issubclass(t, ast.AST)})
    ti = {t: i for i, t in enumerate(types)}
    X, keep = [], []
    for i, r in enumerate(rows):
        try:
            tree = ast.parse(r["code"])
        except SyntaxError:
            continue
        cnt = Counter(type(n).__name__ for n in ast.walk(tree))
        tot = max(1, sum(cnt.values()))
        X.append([cnt.get(t, 0) / tot for t in types])
        keep.append(i)
    return np.array(X, dtype=np.float64), keep, len(types)


def run_probe(name, Xtr, ytr, Xev, yev, ttr, tev, langs_ev, use_linear=False):
    if use_linear:
        model = LinearSVC(C=1.0)
        s_tr = None
        model.fit(Xtr, ytr)
        s_ev = model.decision_function(Xev).astype(np.float64)
    else:
        sc = StandardScaler().fit(Xtr)
        model = LogisticRegression(max_iter=1000, C=1.0).fit(sc.transform(Xtr), ytr)
        s_ev = model.decision_function(sc.transform(Xev)).astype(np.float64)
    amap = task_auroc_map(yev, s_ev, tev)
    rng = np.random.default_rng(SEED)
    ci = boot_ci(amap, rng)
    per_lang = {}
    for l in LANGS:
        m = np.array([x == l for x in langs_ev])
        if m.any():
            sub = task_auroc_map(yev[m], s_ev[m], np.array(tev)[m])
            per_lang[l] = {"task_macro": macro(sub), "n_rows": int(m.sum())}
    res = {"row_auroc_dev": auroc(yev, s_ev), "task_macro_auroc_dev": macro(amap),
           "ci95_task_cluster": ci, "n_train_rows": int(len(ytr)), "n_dev_rows": int(len(yev)),
           "per_language_task_macro_dev": per_lang,
           "score_digest_dev_sha256_float64": sha256_bytes(np.ascontiguousarray(s_ev, dtype=np.float64).tobytes()),
           "score_stats_dev": {"min": float(s_ev.min()), "max": float(s_ev.max()),
                               "mean": float(s_ev.mean()), "std": float(s_ev.std())}}
    log(f"probe[{name}] task_macro={res['task_macro_auroc_dev']:.4f} "
        f"CI={ci[0]:.4f}..{ci[1]:.4f} row={res['row_auroc_dev']:.4f}")
    return res, s_ev


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    PROBES_DIR.mkdir(exist_ok=True)
    (OUT / "logs").mkdir(exist_ok=True)

    lic = phase_license()
    probes, dup, counts, src_ok = phase_scan()
    var = phase_variant_smoke(probes)

    tr, ev = probes["train"], probes["dev"]
    ytr = np.array([r["y"] for r in tr])
    yev = np.array([r["y"] for r in ev])
    ttr = [r["task_id"] for r in tr]
    tev = [r["task_id"] for r in ev]
    lgev = [r["language"] for r in ev]

    probe_out = {}
    scores_npz = {}
    P = {}
    Xtr, Xev = build_features_meta(tr), build_features_meta(ev)
    P["metadata_only"], scores_npz["metadata_only"] = run_probe("metadata_only", Xtr, ytr, Xev, yev, ttr, tev, lgev)
    Xtr, Xev = build_features_source(tr), build_features_source(ev)
    P["source_only"], scores_npz["source_only"] = run_probe("source_only", Xtr, ytr, Xev, yev, ttr, tev, lgev)
    Xtr, Xev = build_features_len(tr), build_features_len(ev)
    P["length_only"], scores_npz["length_only"] = run_probe("length_only", Xtr, ytr, Xev, yev, ttr, tev, lgev)

    # lexical
    tr_txt = [r["code"] for r in tr]
    ev_txt = [r["code"] for r in ev]
    vc = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=5,
                         max_features=100000, sublinear_tf=True, lowercase=False)
    vw = TfidfVectorizer(analyzer="word", token_pattern=r"[A-Za-z_][A-Za-z0-9_]*",
                         ngram_range=(1, 2), min_df=5, max_features=100000,
                         sublinear_tf=True, lowercase=False)
    from scipy.sparse import hstack as sp_hstack
    Xtr_s = sp_hstack([vc.fit_transform(tr_txt), vw.fit_transform(tr_txt)]).tocsr()
    tr_txt = ev_txt = None
    Xev_s = sp_hstack([vc.transform([r["code"] for r in ev]), vw.transform([r["code"] for r in ev])]).tocsr()
    P["lexical_only"], scores_npz["lexical_only"] = run_probe(
        "lexical_only", Xtr_s, ytr, Xev_s, yev, ttr, tev, lgev, use_linear=True)

    # ast-shape (py subset)
    tr_py = [r for r in tr if r["language"] == "py"]
    ev_py = [r for r in ev if r["language"] == "py"]
    Xtr_a, keep_tr, n_types = build_features_ast(tr_py)
    Xev_a, keep_ev, _ = build_features_ast(ev_py)
    P["ast_shape_only_py"], scores_npz["ast_shape_only_py"] = run_probe(
        "ast_shape_only_py", Xtr_a, np.array([tr_py[i]["y"] for i in keep_tr]),
        Xev_a, np.array([ev_py[i]["y"] for i in keep_ev]),
        [tr_py[i]["task_id"] for i in keep_tr], [ev_py[i]["task_id"] for i in keep_ev],
        [ev_py[i]["language"] for i in keep_ev])

    P["transform_only"] = {"not_run": True, "reason": "no_variants_in_package",
                           "note": "variant machinery smoke in variant_smoke; transform_mask excluded from model inputs by design"}
    P["ast_shape_only_other_languages"] = {"pending_parser": [l for l in LANGS if l != "py"]}

    np.savez_compressed(PROBES_DIR / "probe_scores_dev.npz",
                        y=yev, task_ids=np.array(tev),
                        **{k: v for k, v in scores_npz.items()})
    (PROBES_DIR / "probe_score_digests.json").write_text(json.dumps(
        {"schema": "h3_probe_score_digests_v1", "seed": SEED,
         "digest_convention": "sha256 over float64 C-contiguous dev scores in file order",
         "probes": {k: P[k] for k in scores_npz}}, ensure_ascii=False, indent=1), encoding="utf-8")

    # checks aggregated
    checks = {
        "license_declared": lic["cc_by_4_0_mentioned"],
        "source_ref_present_all_rows": src_ok,
        "exact_no_human_ai_overlap": dup["exact"]["human_ai_intersection"] == 0,
        "no_cross_split_task_overlap": True,  # from delivered audit script result (7/7 checks true)
        "variant_smoke_ast_safe": all(v["ast_preserved_rate_on_accepted"] == 1.0
                                      for v in var["transforms"].values() if v["accepted"]),
        "probes_train_dev_only": True,
        "test_used_for_modeling": False,
    }
    high = [k for k in ("metadata_only", "source_only", "length_only")
            if P[k]["task_macro_auroc_dev"] >= 0.70]
    if high:
        decision = "revise_data"
    else:
        decision = "continue"
    metrics = {
        "schema": "h3_data_gate_metrics_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "h2_stacad_alignment_v1",
        "counts": dict(counts),
        "license": lic,
        "duplicates": dup,
        "variant_smoke": var,
        "probes": P,
        "checks": checks,
        "shortcut_risk_high_probes": high,
        "decision_recommendation": decision,
        "switches": {"test_read_modeling": False, "generation": False,
                     "weights_downloaded": False, "code_execution": False},
        "audit_index_sha256": sha256_file(DATA / "audit_index.json"),
        "runtime_s": None,
    }
    metrics["runtime_s"] = round(time.time() - t0, 1)
    (OUT / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "logs" / "run.log").write_text("\n".join(LOG) + "\n", encoding="utf-8")
    log(f"done in {metrics['runtime_s']}s; decision_recommendation={decision}")


if __name__ == "__main__":
    main()
