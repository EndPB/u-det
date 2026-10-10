#!/usr/bin/env python3
"""CPU-first H3 data revision and gate.

This script never reads test payloads. It audits seven language parsers, removes
the previously adjudicated duplicate task clusters, builds a length-balanced
training view, creates conservative train-only syntax-preserving variants, and
re-runs the pre-registered shortcut probes on untouched clean-dev.
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
import re
import subprocess
import sys
import copy
import importlib.metadata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.feature_extraction import DictVectorizer
from scipy.sparse import hstack

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "d-det/data/h2_stacad_alignment_v1"
OUT = ROOT / "d-det/artifacts/h3_local_revision_v1_2026-10-10"
DERIVED = ROOT / "d-det/data/h3_stacad_revision_v1"
SEED = 20261010
BOOT_B = 500
LANGS = ("c", "cpp", "cs", "go", "java", "php", "py")

PARSER_MODULES = {
    "c": "tree_sitter_c",
    "cpp": "tree_sitter_cpp",
    "cs": "tree_sitter_c_sharp",
    "go": "tree_sitter_go",
    "java": "tree_sitter_java",
    "php": "tree_sitter_php",
    "py": "tree_sitter_python",
}
IDENT_TYPES = {
    "identifier", "type_identifier", "field_identifier", "namespace_identifier",
    "property_identifier", "shorthand_property_identifier_pattern", "variable_name",
}
COMMENT_TYPES = {"comment", "line_comment", "block_comment", "documentation_comment"}
LOG = []


def log(msg: str) -> None:
    line = f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG.append(line)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def raw_test_line(line: str) -> bool:
    return bool(re.search(r'"task_split"\s*:\s*"test"', line))


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip() or raw_test_line(line):
                continue
            yield json.loads(line)


def load_rows():
    humans = []
    ai = []
    for row in iter_jsonl(DATA / "task_index.jsonl"):
        humans.append({
            "row_id": row["task_id"] + ":human",
            "record_id": row["task_id"] + ":human",
            "task_id": row["task_id"], "task_split": row["task_split"],
            "language": row["language"], "file_name": row["file_name"],
            "generator": "human", "family": "human", "label": 0,
            "code": row["human_code"], "code_sha256": row["human_sha256"],
            "source_ref": row.get("source_ref"),
        })
    for row in iter_jsonl(DATA / "core.jsonl"):
        ai.append({
            "row_id": row["record_id"], "record_id": row["record_id"],
            "task_id": row["task_id"], "task_split": row["task_split"],
            "language": row["language"], "file_name": row["file_name"],
            "generator": row["generator"], "family": row.get("family"),
            "label": 1, "code": row["code"], "code_sha256": row["code_sha256"],
            "source_ref": row.get("source_ref"),
        })
    return humans, ai


def duplicate_exclusion_tasks() -> set[str]:
    path = ROOT / "d-det/artifacts/h3_data_gate_stacad_2026-10-10/python_ast_dedup_detail.json"
    excluded = set()
    if path.exists():
        obj = json.loads(path.read_text(encoding="utf-8"))
        for group in obj.get("cross_task_or_cross_split_examples", []):
            excluded.update(group.get("tasks", []))
    return excluded


def walk_nodes(root):
    stack = [root]
    while stack:
        node = stack.pop()
        yield node
        stack.extend(reversed(node.children))


def node_shape(node):
    if node.type in COMMENT_TYPES:
        return None
    return (node.type, tuple(node_shape(ch) for ch in node.named_children if ch.type not in COMMENT_TYPES))


def tokens(root, code, mapping=None, skeleton=False):
    b = code.encode()
    out = []
    for node in walk_nodes(root):
        if node.child_count or node.type in COMMENT_TYPES:
            continue
        value = b[node.start_byte:node.end_byte].decode()
        if mapping and value in mapping and node.type in ("identifier", "variable_name", "name"):
            value = mapping[value]
        if mapping and node.parent is not None and node.parent.type == "variable_name":
            parent_value = b[node.parent.start_byte:node.parent.end_byte].decode()
            if parent_value in mapping and node.type == "name":
                mapped_parent = mapping[parent_value]
                value = mapped_parent[1:] if mapped_parent.startswith("$") else mapped_parent
        if skeleton and ("identifier" in node.type or "literal" in node.type
                         or node.type in ("variable_name", "string_content", "integer", "float")):
            value = "<" + node.type + ">"
        out.append((node.type, value))
    return out


def parser_for(lang: str):
    module = __import__(PARSER_MODULES[lang])
    from tree_sitter import Language, Parser
    fn = getattr(module, "language", None)
    if fn is None:
        fn = getattr(module, "language_php_only")
    return Parser(Language(fn()))


def parse_info(code: str, parser):
    tree = parser.parse(code.encode("utf-8", errors="replace"))
    errors = 0
    missing = 0
    for node in walk_nodes(tree.root_node):
        errors += int(node.type == "ERROR")
        missing += int(node.is_missing)
    return tree, errors, missing


def code_lengths(code: str) -> np.ndarray:
    lines = code.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    nonempty = [x for x in lines if x.strip()]
    toks = re.findall(r"\S+", code)
    return np.array([
        len(code), len(lines), len(nonempty), len(lines) - len(nonempty),
        max((len(x) for x in lines), default=0),
        float(np.mean([len(x) for x in lines])) if lines else 0.0,
        len(toks), code.count(" "), code.count("\t"),
    ], dtype=np.float64)


def json_dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def balance_by_task(rows, scaler):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["task_id"]].append(row)
    out = []
    manifest = {}
    for task_id in sorted(grouped):
        group = grouped[task_id]
        human = [r for r in group if r["label"] == 0]
        ais = [r for r in group if r["label"] == 1]
        if not human or not ais:
            continue
        h = sorted(human, key=lambda r: r["row_id"])[0]
        hv = scaler.transform(np.log1p(code_lengths(h["code"]))[None, :])[0]
        distances = {r["row_id"]: float(np.linalg.norm(
            scaler.transform(np.log1p(code_lengths(r["code"]))[None, :])[0] - hv)) for r in ais}
        chosen = sorted(ais, key=lambda r: (distances[r["row_id"]], r["generator"], r["record_id"]))[:3]
        out.extend([h] + chosen)
        manifest[task_id] = {
            "human": h["row_id"],
            "ai": [r["row_id"] for r in chosen],
            "distance": [distances[r["row_id"]] for r in chosen],
        }
    return out, manifest


FUNCTION_TYPES = {
    "function_definition", "function_declaration", "method_declaration",
}
DYNAMIC = re.compile(r"\b(eval|exec|globals|locals|vars|dir|setattr|getattr|"
                     r"Reflection|reflect|dynamic|compact|extract|get_defined_vars|"
                     r"getframe|currentframe|stack|nameof|NameOf)\b")


def ancestor(node, types):
    cur = node.parent
    while cur is not None:
        if cur.type in types:
            return cur
        cur = cur.parent
    return None


def safe_rename(code, lang, tree):
    """Single, explicit local binding; no interface, member or reflective use.

    Only a conservative subset is admitted. Tree/token/stdlib AST checks run
    separately after the mapped replacements. No execution correctness claim.
    """
    if DYNAMIC.search(code) or re.search(r"^\s*#\s*(define|if|include)", code, re.M):
        return None, "unsafe_dynamic_or_macro", None
    b = code.encode()
    nodes = list(walk_nodes(tree.root_node))
    ident_nodes = [n for n in nodes if n.type in ("identifier", "variable_name")]
    if lang == "php":
        for n in ident_nodes:
            if n.type != "variable_name" or n.parent is None or n.parent.type != "assignment_expression":
                continue
            if n.parent.child_by_field_name("left") != n:
                continue
            fn = ancestor(n, FUNCTION_TYPES)
            body = fn.child_by_field_name("body") if fn is not None else None
            name = n.text.decode()
            occurrences = [x for x in ident_nodes if x.type == "variable_name"
                           and x.text.decode() == name and body is not None
                           and body.start_byte <= x.start_byte < body.end_byte]
            if len(occurrences) < 2:
                continue
            spans = {(m.start(), m.end()) for m in re.finditer(
                rb"(?<![A-Za-z0-9_])" + re.escape(name.encode()) + rb"(?![A-Za-z0-9_])", b)}
            if spans != {(x.start_byte, x.end_byte) for x in occurrences}:
                continue
            replacement = "$h3_local_" + name[1:] if name.startswith("$") else "h3_local_" + name
            out = b
            for x in sorted(occurrences, key=lambda q: q.start_byte, reverse=True):
                out = out[:x.start_byte] + replacement.encode() + out[x.end_byte:]
            return out.decode(), "candidate", {name: replacement}
    candidates = []
    for n in ident_nodes:
        p = n.parent
        if p is None:
            continue
        declared = False
        if lang == "php":
            declared = (n.type == "variable_name" and p.type == "assignment_expression"
                        and p.child_by_field_name("left") == n)
        elif lang in ("c", "cpp"):
            declared = ((p.type == "init_declarator" and p.child_by_field_name("declarator") == n)
                        or (p.type == "declaration" and p.child_by_field_name("declarator") == n))
        elif lang in ("cs", "java"):
            declared = p.type == "variable_declarator" and p.child_by_field_name("name") == n
            if lang == "cs" and p.type == "variable_declarator":
                declared = p.named_children[0] == n
        elif lang == "go":
            declared = (p.type == "expression_list" and len(p.named_children) == 1
                        and p.parent is not None and p.parent.type == "short_var_declaration"
                        and p.parent.child_by_field_name("left") == p)
        elif lang == "py":
            declared = p.type == "assignment" and p.child_by_field_name("left") == n
        if declared:
            candidates.append(n)
    for candidate in candidates:
        name = b[candidate.start_byte:candidate.end_byte].decode()
        if len(name) < 2 or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name):
            continue
        fn = ancestor(candidate, FUNCTION_TYPES)
        if fn is None:
            continue
        body = fn.child_by_field_name("body")
        if body is None or not (body.start_byte <= candidate.start_byte < body.end_byte):
            continue
        fn_nodes = list(walk_nodes(fn))
        if any(n.type in FUNCTION_TYPES or n.type in {
                "lambda", "lambda_expression", "anonymous_function_creation_expression",
                "arrow_function", "func_literal", "class_definition", "class_declaration",
                "list_comprehension", "dictionary_comprehension", "set_comprehension",
                "generator_expression", "global_statement", "nonlocal_statement",
                "yield", "yield_expression", "using_directive", "import_statement",
                "import_from_statement", "unset_statement"} for n in fn_nodes if n != fn):
            continue
        # Exactly one local declaration with this name; no same-name token
        # anywhere outside the chosen function, including strings and members.
        declarations = [n for n in candidates if b[n.start_byte:n.end_byte].decode() == name]
        if len(declarations) != 1:
            continue
        occurrences = [n for n in nodes if n.type == "identifier"
                       and b[n.start_byte:n.end_byte].decode() == name]
        if len(occurrences) < 2:
            continue
        if any(not (body.start_byte <= n.start_byte < body.end_byte) for n in occurrences):
            continue
        if any(n.parent is not None and n.parent.type in {
                "attribute", "field_expression", "member_access_expression",
                "member_access", "scoped_identifier", "qualified_name",
                "keyword_argument", "named_argument", "pair", "labeled_statement",
                "property_element", "member_call_expression", "scope_resolution"}
               for n in occurrences):
            continue
        # All text occurrences must be exactly the parser's identifier spans:
        # this excludes strings, comments, fields, external aliases and macros.
        text_spans = {(m.start(), m.end()) for m in re.finditer(
            rb"(?<![A-Za-z0-9_])" + re.escape(name.encode()) + rb"(?![A-Za-z0-9_])", b)}
        if text_spans != {(n.start_byte, n.end_byte) for n in occurrences}:
            continue
        # Avoid address/alias escapes for non-Python bindings.
        fn_text = b[fn.start_byte:fn.end_byte].decode()
        if lang != "py" and re.search(r"(&\s*" + re.escape(name) + r"\b)|\b(ref|out|inout)\s+" + re.escape(name), fn_text):
            continue
        replacement = ("$h3_local_" + name[1:]) if lang == "php" and name.startswith("$") else "h3_local_" + name
        if replacement in code:
            continue
        result = b
        for n in sorted(occurrences, key=lambda x: x.start_byte, reverse=True):
            result = result[:n.start_byte] + replacement.encode() + result[n.end_byte:]
        return result.decode(), "candidate", {name: replacement}
    return None, "no_scope_safe_candidate", None


def python_dump(code, mapping=None):
    obj = ast.parse(code)
    if mapping:
        for n in ast.walk(obj):
            if isinstance(n, ast.Name) and n.id in mapping:
                n.id = mapping[n.id]
    return ast.dump(obj, include_attributes=False)


def transformed_code(row, transform, parser):
    code = row["code"]
    original, e0, m0 = parse_info(code, parser)
    if e0 or m0:
        return None, "parent_parse_error", {}
    original_shape = node_shape(original.root_node)
    mapping = None
    if transform == "comment":
        spans = []
        for node in walk_nodes(original.root_node):
            if node.type in COMMENT_TYPES:
                spans.append((node.start_byte, node.end_byte))
        if not spans:
            return None, "no_safe_candidate", {}
        b = code.encode("utf-8")
        for start, end in reversed(spans):
            replacement = b" " + b"\n" * b[start:end].count(b"\n")
            b = b[:start] + replacement + b[end:]
        new_code = b.decode("utf-8", errors="replace")
    elif transform == "whitespace":
        lines = code.splitlines(True)
        if len(lines) < 2:
            return None, "no_safe_candidate", {}
        new_code = lines[0] + "\n" + "".join(lines[1:])
        if new_code == code:
            return None, "no_change", {}
    else:
        new_code, reason, mapping = safe_rename(code, row["language"], original)
        if new_code is None:
            return None, reason, {}
    if new_code == code:
        return None, "no_change", {}
    try:
        new_tree, errors, missing = parse_info(new_code, parser)
        if errors or missing or node_shape(new_tree.root_node) != original_shape:
            return None, "ast_mismatch", {}
        if tokens(original.root_node, code, mapping) != tokens(new_tree.root_node, new_code):
            return None, "token_mismatch", {}
        if row["language"] == "py" and python_dump(code, mapping) != python_dump(new_code):
            return None, "python_ast_mismatch", {}
        proof = {"token_alignment": True, "tree_alignment": True,
                 "python_ast_alignment": True if row["language"] == "py" else None,
                 "rename_mapping": mapping, "syntax_errors": 0, "missing_nodes": 0}
        return new_code, "accepted", proof
    except (ValueError, SyntaxError, RecursionError, UnicodeError):
        return None, "parse_error", {}


def variant_audit(train, parsers):
    rng = np.random.default_rng(SEED)
    records = []
    counts = Counter()
    variant_audit.children = []
    for row in train:
        for transform, prob in (("comment", 0.2), ("whitespace", 0.3), ("rename", 0.4)):
            if rng.random() >= prob:
                continue
            new_code, reason, proof = transformed_code(row, transform, parsers[row["language"]])
            counts[f"{transform}:{reason}"] += 1
            rec = {
                "parent_row_id": row["row_id"], "task_id": row["task_id"],
                "language": row["language"], "label": row["label"], "generator": row["generator"],
                "transform": transform, "status": reason, "proof": proof,
                "before_sha256": row["code_sha256"],
            }
            if new_code is not None:
                rec["after_sha256"] = sha256_bytes(new_code.encode())
                rec["changed_bytes"] = len(new_code.encode()) - len(row["code"].encode())
                child = dict(row)
                child.update({"parent_row_id": row["row_id"], "row_id": row["row_id"] + ":" + transform,
                              "code": new_code, "code_sha256": rec["after_sha256"], "transform": transform})
                variant_audit.children.append(child)
            records.append(rec)
    return records, counts


def auroc(y, score):
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    pos = y == 1
    neg = ~pos
    if not pos.any() or not neg.any():
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    sorted_scores = score[order]
    ranks = np.empty(len(score), dtype=float)
    i = 0
    while i < len(score):
        j = i
        while j + 1 < len(score) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * neg.sum()))


def task_macro(y, score, tasks):
    vals = []
    for task in sorted(set(tasks)):
        mask = np.asarray(tasks) == task
        value = auroc(np.asarray(y)[mask], np.asarray(score)[mask])
        if not math.isnan(value):
            vals.append(value)
    return float(np.mean(vals)) if vals else float("nan")


def task_ci(y, score, tasks):
    values = []
    for task in sorted(set(tasks)):
        mask = np.asarray(tasks) == task
        value = auroc(np.asarray(y)[mask], np.asarray(score)[mask])
        if not math.isnan(value):
            values.append(value)
    if not values:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(SEED + 1)
    vals = np.asarray(values)
    draws = vals[rng.integers(0, len(vals), size=(BOOT_B, len(vals)))].mean(axis=1)
    return [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]


def feature_rows(rows, kind):
    if kind == "length":
        return np.log1p(np.vstack([code_lengths(r["code"]) for r in rows]))
    if kind == "length_historical_raw":
        return np.vstack([code_lengths(r["code"]) for r in rows])
    if kind == "metadata":
        langs = {x: i for i, x in enumerate(LANGS)}
        out = []
        for r in rows:
            f = [0.0] * len(LANGS)
            f[langs[r["language"]]] = 1.0
            name = r["file_name"]
            f.extend([len(name), sum(c.isdigit() for c in name), name.count("_")])
            out.append(f)
        return np.asarray(out, dtype=float)
    if kind == "source":
        langs = {x: i for i, x in enumerate(LANGS)}
        out = []
        for r in rows:
            f = [0.0] * len(LANGS)
            f[langs[r["language"]]] = 1.0
            buckets = [0.0] * 64
            buckets[int.from_bytes(hashlib.blake2b(r["file_name"].encode(), digest_size=8).digest(), "big") % 64] = 1.0
            f.extend(buckets)
            out.append(f)
        return np.asarray(out, dtype=float)
    raise ValueError(kind)


def fit_numeric(train, dev, kind):
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=1000, random_state=SEED))
    clf.fit(feature_rows(train, kind), [r["label"] for r in train])
    return clf.decision_function(feature_rows(dev, kind))


def fit_lexical(train, dev):
    texts = [r["code"] for r in train]
    dev_texts = [r["code"] for r in dev]
    matrices_train, matrices_dev = [], []
    for analyzer, ngrams in (("char_wb", (2, 5)), ("word", (1, 2))):
        vectorizer = TfidfVectorizer(analyzer=analyzer, ngram_range=ngrams, min_df=5,
                                     max_features=100000, lowercase=False, sublinear_tf=True,
                                     token_pattern=r"[A-Za-z_][A-Za-z0-9_]*")
        x_train = vectorizer.fit_transform(texts)
        x_dev = vectorizer.transform(dev_texts)
        matrices_train.append(x_train)
        matrices_dev.append(x_dev)
    clf = LinearSVC(C=1, random_state=SEED, max_iter=3000)
    clf.fit(hstack(matrices_train).tocsr(), [r["label"] for r in train])
    return clf.decision_function(hstack(matrices_dev).tocsr())


def fit_ast(train, dev, hist):
    vectorizer = DictVectorizer(sparse=False)
    xt = vectorizer.fit_transform([hist[r["row_id"]] for r in train])
    xe = vectorizer.transform([hist[r["row_id"]] for r in dev])
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=1000, random_state=SEED))
    clf.fit(xt, [r["label"] for r in train])
    return clf.decision_function(xe)


def score_metrics(rows, score):
    y = [r["label"] for r in rows]
    tasks = [r["task_id"] for r in rows]
    return {"task_macro_auroc": task_macro(y, score, tasks), "ci95": task_ci(y, score, tasks),
            "row_auroc": auroc(y, score),
            "per_language_task_macro": {lang: task_macro(
                [r["label"] for r in rows if r["language"] == lang],
                score[np.array([r["language"] == lang for r in rows])],
                [r["task_id"] for r in rows if r["language"] == lang]) for lang in LANGS}}


def run_probes(train, dev, matched, tag, hist):
    combined = dev + matched
    y = [r["label"] for r in dev]
    tasks = [r["task_id"] for r in dev]
    result, sensitivity, scores = {}, {}, {}
    for kind in ("length", "length_historical_raw", "metadata", "source", "ast_shape", "lexical"):
        log(f"probe start: {tag}/{kind}")
        if kind == "lexical":
            score = fit_lexical(train, combined)
        elif kind == "ast_shape":
            score = fit_ast(train, combined, hist)
        else:
            score = fit_numeric(train, combined, kind)
        result[kind] = score_metrics(dev, score[:len(dev)])
        sensitivity[kind] = score_metrics(matched, score[len(dev):])
        scores[kind] = score
        log(f"probe {tag}/{kind}: clean-dev={result[kind]['task_macro_auroc']:.4f}, matched={sensitivity[kind]['task_macro_auroc']:.4f}")
    np.savez_compressed(OUT / (tag + "_scores.npz"),
                        row_ids=np.array([r["row_id"] for r in combined]),
                        task_ids=np.array([r["task_id"] for r in combined]),
                        labels=np.array([r["label"] for r in combined]),
                        primary_n=len(dev), **scores)
    return result, sensitivity


def run_transform_probe(records, train):
    by_parent = defaultdict(dict)
    for rec in records:
        by_parent[rec["parent_row_id"]][rec["transform"]] = int(rec["status"] == "accepted")
    rows = [r for r in train if r["row_id"] in by_parent]
    feats = np.asarray([[by_parent[r["row_id"]].get("comment", 0),
                         by_parent[r["row_id"]].get("whitespace", 0),
                         by_parent[r["row_id"]].get("rename", 0)] for r in rows], dtype=float)
    hold = np.asarray([int(int.from_bytes(hashlib.blake2b(
        f"{SEED}|{r['task_id']}".encode(), digest_size=8).digest(), "big") % 5 == 0) for r in rows], dtype=bool)
    fit = ~hold
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=1000, random_state=SEED))
    clf.fit(feats[fit], [r["label"] for i, r in enumerate(rows) if fit[i]])
    score = clf.decision_function(feats[hold])
    ev = [r for i, r in enumerate(rows) if hold[i]]
    result = score_metrics(ev, score)
    result.update({"fit_rows": int(fit.sum()), "eval_rows": int(hold.sum()),
                   "eval_tasks": len(set(r["task_id"] for r in ev)),
                   "features": ["comment_accepted", "whitespace_accepted", "rename_accepted"],
                   "split": "task hash holdout 1/5"})
    json_dump(OUT / "transform_probe.json", result)
    return result


def parser_and_duplicate_audit(rows, parsers):
    counts = Counter()
    examples = []
    hist = {}
    hashes = {kind: defaultdict(list) for kind in ("exact", "normalized_ws", "syntax_tokens", "skeleton")}
    for row in rows:
        b = row["code"].encode()
        if sha256_bytes(b) != row["code_sha256"]:
            raise RuntimeError("input integrity failure: " + row["row_id"])
        counts[row["language"] + ":rows"] += 1
        hashes["exact"][sha256_bytes(b)].append(row)
        normalized = re.sub(r"\s+", "", row["code"])
        hashes["normalized_ws"][sha256_bytes(normalized.encode())].append(row)
        tree, errors, missing = parse_info(row["code"], parsers[row["language"]])
        counts[row["language"] + ":parse_error"] += int(bool(errors or missing))
        if errors or missing:
            examples.append({"row_id": row["row_id"], "errors": errors, "missing": missing})
        nodes = list(walk_nodes(tree.root_node))
        ncounts = Counter(n.type for n in nodes if n.type not in COMMENT_TYPES)
        total = max(1, sum(ncounts.values()))
        hist[row["row_id"]] = {k: v / total for k, v in ncounts.items()}
        hist[row["row_id"]]["PARSER_ERROR"] = float(bool(errors or missing))
        if not errors and not missing:
            for kind, skel in (("syntax_tokens", False), ("skeleton", True)):
                signature = (row["language"], tokens(tree.root_node, row["code"], skeleton=skel))
                hashes[kind][sha256_bytes(repr(signature).encode())].append(row)
    cross = []
    summary = {}
    excluded_train = set()
    for kind, mapping in hashes.items():
        groups = [group for group in mapping.values() if len(group) > 1]
        cross_task = [g for g in groups if len({r["task_id"] for r in g}) > 1]
        cross_split = [g for g in cross_task if len({r["task_split"] for r in g}) > 1]
        summary[kind] = {"duplicate_groups": len(groups), "cross_task_groups": len(cross_task),
                         "cross_split_groups_before_exclusion": len(cross_split)}
        for group in cross_split:
            excluded_train.update(r["task_id"] for r in group if r["task_split"] == "train")
            cross.append({"kind": kind, "members": [
                {k: r[k] for k in ("row_id", "task_id", "task_split", "language")} for r in group]})
    # Check every certificate after dropping all marked training task clusters.
    for kind, mapping in hashes.items():
        retained = [[r for r in g if not (r["task_split"] == "train" and r["task_id"] in excluded_train)]
                    for g in mapping.values()]
        summary[kind]["cross_split_groups_retained"] = sum(
            len({r["task_split"] for r in g}) > 1 for g in retained)
    json_dump(OUT / "parser_audit.json", {"counts": dict(counts), "invalid_examples": examples,
                                         "rows_audited": len(rows), "test_deserialized": False,
                                         "invalid_rows_retained_for_modeling": True})
    json_dump(OUT / "multilang_duplicate_audit.json", {"summary": summary, "cross_split_groups": cross,
                "excluded_train_tasks": sorted(excluded_train),
                "policy": "exclude all train tasks touching cross-split exact/ws/token/skeleton groups; same-split skeleton matches are diagnostic"})
    return hist, excluded_train, summary, counts


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    DERIVED.mkdir(parents=True, exist_ok=True)
    (DERIVED / ".gitignore").write_text("*\n!.gitignore\n", encoding="utf-8")
    log("loading train/dev rows; test lines skipped before JSON parsing")
    humans, ai = load_rows()
    excluded = duplicate_exclusion_tasks()
    all_rows = [r for r in humans + ai if r["task_id"] not in excluded]
    train_all = [r for r in all_rows if r["task_split"] == "train"]
    dev_clean = [r for r in all_rows if r["task_split"] == "dev"]
    parsers = {lang: parser_for(lang) for lang in LANGS}
    hist, extra_excluded, duplicate_summary, parser_counts = parser_and_duplicate_audit(all_rows, parsers)
    train_all = [r for r in train_all if r["task_id"] not in extra_excluded]
    log(f"duplicate audit: old_excluded={len(excluded)} additional_train_excluded={len(extra_excluded)}")
    scaler = StandardScaler().fit(np.log1p(np.vstack([code_lengths(r["code"]) for r in train_all])))
    train_balanced, balance_manifest = balance_by_task(train_all, scaler)
    dev_matched, dev_match_manifest = balance_by_task(dev_clean, scaler)
    write_jsonl(DERIVED / "train_all.jsonl", train_all)
    write_jsonl(DERIVED / "train_balanced.jsonl", train_balanced)
    write_jsonl(DERIVED / "dev_clean.jsonl", dev_clean)
    write_jsonl(DERIVED / "dev_matched_sensitivity.jsonl", dev_matched)
    json_dump(OUT / "duplicate_adjudication.json", {
        "excluded_task_ids": sorted(excluded),
        "excluded_count": len(excluded),
        "additional_train_excluded": sorted(extra_excluded),
        "policy": "remove all previously adjudicated cross-task/cross-split groups from derived train/dev",
        "input_rows": {"human": len(humans), "ai": len(ai), "train_all": len(train_all), "dev_clean": len(dev_clean)},
    })
    json_dump(OUT / "balance_manifest.json", {
        "train_balanced_tasks": len(balance_manifest),
        "dev_matched_tasks": len(dev_match_manifest),
        "train_rows": len(train_balanced), "dev_rows": len(dev_matched),
        "train": balance_manifest, "dev_matched_sensitivity": dev_match_manifest,
        "matching": "9D log1p Euclidean distance; StandardScaler fitted on retained train only",
        "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(),
    })
    log(f"rows: train_all={len(train_all)} train_balanced={len(train_balanced)} dev_clean={len(dev_clean)}")

    log(f"parser audit complete: {len(all_rows)} rows")

    variants, variant_counts = variant_audit(train_all, parsers)
    write_jsonl(OUT / "variant_records.jsonl", variants)
    write_jsonl(DERIVED / "train_variants.jsonl", variant_audit.children)
    json_dump(OUT / "variant_summary.json", {
        "seed": SEED, "probabilities": {"comment": 0.2, "whitespace": 0.3, "rename": 0.4},
        "counts": dict(variant_counts), "records": len(variants),
        "scope": "train only; tree-shape safety proxy; no code execution",
        "per_language_transform_status": dict(Counter(
            (r["language"] + ":" + r["transform"] + ":" + r["status"]) for r in variants)),
        "per_label_transform_status": dict(Counter(
            (str(r["label"]) + ":" + r["transform"] + ":" + r["status"]) for r in variants)),
    })
    log(f"variant audit complete: {len(variants)} sampled")
    transform_probe = run_transform_probe(variants, train_all)
    log(f"transform-only probe: clean task-heldout={transform_probe['task_macro_auroc']:.4f}")

    all_metrics, all_sensitivity = run_probes(train_all, dev_clean, dev_matched, "all_train", hist)
    balanced_metrics, balanced_sensitivity = run_probes(train_balanced, dev_clean, dev_matched, "balanced_train", hist)
    metrics = {
        "primary_eval": "dev_clean_untouched",
        "matched_eval": "dev_matched_sensitivity_only",
        "all_train": all_metrics,
        "balanced_train": balanced_metrics,
        "matched_dev_sensitivity": {"all_train": all_sensitivity, "balanced_train": balanced_sensitivity},
        "transform_only_train_task_holdout": transform_probe,
    }
    json_dump(OUT / "metrics.json", metrics)
    log("probes complete")
    config = {
        "script": "scripts/h3_local_revision_v1.py", "seed": SEED, "bootstrap": BOOT_B,
        "languages": LANGS, "test_read_modeling": False, "generation": False,
        "weights": False, "code_execution": False, "primary_clean_dev": True,
        "parser_packages": PARSER_MODULES, "excluded_tasks": sorted(excluded),
    }
    json_dump(OUT / "config.json", config)
    json_dump(OUT / "data_role.json", {
        "train": "derived train_all and train_balanced; safe for fitting",
        "dev": "clean_dev primary; matched_dev sensitivity only",
        "test": "sealed; raw payload skipped before JSON deserialization",
        "source": "d-det/data/h2_stacad_alignment_v1",
        "test_streamed_bytes": "container lines necessarily streamed; test JSON/code never deserialized, parsed or modeled",
    })
    (OUT / "git_head.txt").write_text(subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip() + "\n", encoding="utf-8")
    (OUT / "commands.txt").write_text(
        "PYTHONPATH=.codex-local-runtime/h3-parser .codex-local-runtime/c0-v3/Scripts/python.exe scripts/h3_local_revision_v1.py\n",
        encoding="utf-8",
    )
    (OUT / "run.log").write_text("\n".join(LOG) + "\n", encoding="utf-8")
    primary = metrics["balanced_train"]["length"]["task_macro_auroc"]
    length_high = any(metrics[view][kind]["task_macro_auroc"] >= .70
                      for view in ("all_train", "balanced_train")
                      for kind in ("length", "length_historical_raw"))
    decision = "revise_data" if length_high else "pending_provenance_and_transform_gate"
    json_dump(OUT / "gate.json", {
        "decision": decision, "length_shortcut_high": length_high,
        "cross_split_duplicates_retained": {k: v["cross_split_groups_retained"] for k, v in duplicate_summary.items()},
        "accepted_variants": len(variant_audit.children),
        "project_provenance": "not available; not certified project-heldout",
        "transform_probe": {
            "task_macro_auroc": transform_probe["task_macro_auroc"],
            "ci95": transform_probe["ci95"],
            "high_shortcut": bool(transform_probe["task_macro_auroc"] >= 0.60),
        },
        "formal_gpu_ready": False, "test_deserialized": False,
    })
    json_dump(OUT / "data_manifest.json", {
        p.name: {"sha256": sha256_file(p), "bytes": p.stat().st_size}
        for p in sorted(DERIVED.glob("*.jsonl"))})
    json_dump(OUT / "env.json", {
        "python": sys.version, "platform": sys.platform,
        "packages": {name: importlib.metadata.version(name) for name in (
            "numpy", "scipy", "scikit-learn", "tree-sitter", "tree-sitter-c", "tree-sitter-cpp",
            "tree-sitter-c-sharp", "tree-sitter-go", "tree-sitter-java", "tree-sitter-php", "tree-sitter-python")},
        "script_sha256": sha256_file(Path(__file__)),
        "hypothesis_sha256": sha256_file(OUT / "hypothesis.md"),
        "run_utc": datetime.now(timezone.utc).isoformat(),
    })
    report = f"""# H3 本机数据修订 v1 结果

日期：2026-10-10；test payload read = false。

## 数据

- 旧报告裁定的 {len(excluded)} 个任务已从派生 train/dev 排除；新增跨 split 重复涉及的 {len(extra_excluded)} 个 train 任务整簇排除。
- train_all={len(train_all)}，train_balanced={len(train_balanced)}，clean-dev={len(dev_clean)}；matched-dev 只作敏感性分析。
- 七语言 parser 已对 train/dev 派生行执行审计；变体只在 train 采样。

## 预注册判断

balanced-train 在 untouched clean-dev 上的 length-only task-macro AUROC = {primary:.4f}。
固定规则给出的数据侧裁定：{decision}。这不是 H3 方法增量结论；GPU 批次仍需在数据闸门通过后一次性执行。

## 限制

STACAD 每个 observed source 只有一个 generator，因此本包不宣称同家族未见 generator 迁移；source 仅作 task-heldout observed-source 诊断。parser tree-shape 是安全代理，不等价于编译/运行正确性。
项目来源证据缺失，无法认证 project-heldout；语法无效原行保留建模以免引入解析选择偏差，变体对这些行全部拒绝。transform-only task-heldout probe = {transform_probe["task_macro_auroc"]:.4f}（CI {transform_probe["ci95"][0]:.4f}..{transform_probe["ci95"][1]:.4f}）。不能将单项长度下降称为全闸门通过。
"""
    (OUT / "report.md").write_text(report, encoding="utf-8")
    sums = []
    for path in sorted(OUT.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS.txt":
            sums.append(f"{sha256_file(path)}  {path.relative_to(OUT).as_posix()}")
    (OUT / "SHA256SUMS.txt").write_text("\n".join(sums) + "\n", encoding="utf-8")
    log(f"decision={decision}; artifact={OUT}")


if __name__ == "__main__":
    main()
