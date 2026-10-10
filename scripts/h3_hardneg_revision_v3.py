#!/usr/bin/env python3
"""H3 v3 CPU hard-negative construction and gate.

The STACAD alignment exposes a task/file identifier but no repository or
project identifier.  This pass therefore constructs a conservative
same-task hard-negative view and records project provenance as unavailable;
it never labels task IDs as projects and never reads test rows.
"""
from __future__ import annotations

import hashlib
import json
import platform
import re
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / "d-det/data/h3_stacad_revision_v1"
OUT = ROOT / "d-det/artifacts/h3_hardneg_revision_v3_2026-10-11"
DERIVED = ROOT / "d-det/data/h3_stacad_revision_v3"
SEED = 20261011


def load(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def length(row: dict) -> float:
    return float(np.log1p(len(row.get("code", "").encode("utf-8"))))


def mask_code(code: str) -> str:
    # Deliberately language-agnostic: comments, identifiers, strings and
    # numbers are collapsed while keywords and punctuation remain visible.
    code = re.sub(r"/\*.*?\*/", " ", code, flags=re.S)
    code = re.sub(r"//[^\n]*", " ", code)
    code = re.sub(r"(^|[\s;{}])#[^\n]*", r"\1 ", code)
    pat = re.compile(r"(?:\"(?:\\.|[^\"])*\"|'(?:\\.|[^'])*')|(?:\b\d+(?:\.\d+)?\b)|(?:\b[A-Za-z_][A-Za-z0-9_]*\b)")
    keywords = {
        "if", "else", "for", "while", "return", "class", "struct", "public", "private",
        "protected", "static", "void", "int", "float", "double", "char", "new", "try",
        "catch", "def", "import", "from", "package", "func", "var", "const", "true",
        "false", "null", "nil", "switch", "case", "break", "continue", "this", "self",
    }
    def repl(match: re.Match[str]) -> str:
        token = match.group(0)
        if token.startswith(("\"", "'")):
            return "<STR>"
        if token[0].isdigit():
            return "<NUM>"
        return token if token in keywords else "<ID>"
    return pat.sub(repl, code)


def cosine_candidates(human: dict, ais: list[dict]) -> list[tuple[float, dict]]:
    texts = [mask_code(human["code"])] + [mask_code(r["code"]) for r in ais]
    vec = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b|[^\w\s]", ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    mat = vec.fit_transform(texts)
    h = mat[0]
    # TfidfVectorizer rows are L2-normalized by default, so the dot product is cosine.
    sims = (mat[1:] @ h.T).toarray().ravel()
    hv = length(human)
    scored = []
    for row, sim in zip(ais, sims):
        # Length is a hard control; masked shape similarity breaks ties among
        # same-length candidates without using labels beyond the paired task.
        score = 0.70 * abs(length(row) - hv) + 0.30 * (1.0 - float(sim))
        scored.append((score, row))
    return sorted(scored, key=lambda item: (item[0], item[1]["row_id"]))


def select_hardneg(rows: list[dict], max_ai: int = 3) -> tuple[list[dict], dict]:
    by_task: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_task[row["task_id"]].append(row)
    selected: list[dict] = []
    manifest: dict[str, dict] = {}
    task_ids = sorted(by_task, key=lambda task: hashlib.sha256(task.encode()).hexdigest())
    all_generators = sorted({str(r.get("generator")) for group in by_task.values() for r in group if r["label"] == 1})
    total_slots = len(task_ids) * max_ai
    quotas = {g: total_slots // len(all_generators) + int(i < total_slots % len(all_generators)) for i, g in enumerate(all_generators)}
    used = Counter()
    for task_index, task_id in enumerate(task_ids):
        group = by_task[task_id]
        humans = [r for r in group if r["label"] == 0]
        ais = [r for r in group if r["label"] == 1]
        if len(humans) != 1 or not ais:
            continue
        human = humans[0]
        ranked = cosine_candidates(human, ais)
        ranked_by_generator = {str(row.get("generator")): (score, row) for score, row in ranked}
        generators = sorted(ranked_by_generator)
        selected_generators = [generators[(task_index * max_ai + j) % len(generators)] for j in range(max_ai)]
        chosen = [ranked_by_generator[g] for g in selected_generators]
        for gen in selected_generators:
            used[gen] += 1
        group = by_task[task_id]
        humans = [r for r in group if r["label"] == 0]
        if len(humans) != 1:
            continue
        human = humans[0]
        chosen.sort(key=lambda item: (item[0], item[1]["row_id"]))
        for role, score, row in [("human", 0.0, human)] + [("ai", s, r) for s, r in chosen]:
            copy = dict(row)
            copy["eval_view"] = "hardneg_v3"
            copy["hardneg_score"] = float(score)
            copy["masked_length_log1p_bytes"] = length(row)
            copy["project_provenance_status"] = "unavailable_in_stacad_alignment"
            selected.append(copy)
        manifest[task_id] = {
            "human": human["row_id"],
            "ai": [r["row_id"] for _, r in chosen],
            "generators": [r.get("generator") for _, r in chosen],
            "scores": [float(s) for s, _ in chosen],
            "length_deltas": [abs(length(r) - length(human)) for _, r in chosen],
        }
    return selected, {"tasks": manifest, "generator_quotas": quotas, "generator_counts": dict(used)}


def auc(y, scores) -> float:
    return float(roc_auc_score(y, scores)) if len(set(y)) == 2 else float("nan")


def task_macro(rows: list[dict], scores: np.ndarray) -> float:
    groups: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        groups[row["task_id"]].append(idx)
    vals = []
    for idx in groups.values():
        y = [rows[i]["label"] for i in idx]
        if len(set(y)) == 2:
            vals.append(auc(y, scores[idx]))
    return float(np.mean(vals)) if vals else float("nan")


def lexical_probe(train: list[dict], eval_rows: list[dict], view: str) -> dict:
    transform = (lambda x: x) if view == "raw" else mask_code
    vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b|[^\w\s]", ngram_range=(1, 2), min_df=2, max_features=120000, sublinear_tf=True)
    x_train = vectorizer.fit_transform([transform(r["code"]) for r in train])
    x_eval = vectorizer.transform([transform(r["code"]) for r in eval_rows])
    clf = LogisticRegression(C=2.0, max_iter=100, solver="liblinear", random_state=SEED).fit(x_train, [r["label"] for r in train])
    score = clf.decision_function(x_eval)
    return {"row_auroc": auc([r["label"] for r in eval_rows], score), "task_macro_auroc": task_macro(eval_rows, score), "rows": len(eval_rows), "view": view}


def length_probe(train: list[dict], eval_rows: list[dict]) -> dict:
    langs = sorted({r["language"] for r in train + eval_rows})
    lang_map = {lang: i for i, lang in enumerate(langs)}
    def feats(rows):
        return np.asarray([[length(r), len(r.get("code", "").split()), lang_map[r["language"]]] for r in rows], dtype=float)
    scaler = StandardScaler().fit(feats(train))
    clf = LogisticRegression(C=1.0, max_iter=200, random_state=SEED).fit(scaler.transform(feats(train)), [r["label"] for r in train])
    score = clf.decision_function(scaler.transform(feats(eval_rows)))
    return {"row_auroc": auc([r["label"] for r in eval_rows], score), "task_macro_auroc": task_macro(eval_rows, score), "rows": len(eval_rows)}


def provenance_audit(rows: list[dict]) -> dict:
    keys = sorted({k for row in rows for k in row})
    candidates = [k for k in keys if any(term in k.lower() for term in ("repo", "project", "url", "source"))]
    return {"status": "unavailable", "required_fields": ["repository_id", "project_id", "source_url"], "present_candidate_fields": candidates, "observed_fields": keys, "evidence": "alignment rows contain source_ref to corpus file/line, but no repository/project identity"}


def write_sums(directory: Path) -> None:
    lines = []
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS.txt":
            lines.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(directory).as_posix()}")
    (directory / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    DERIVED.mkdir(parents=True, exist_ok=True)
    train = [r for r in load(V1 / "train_all.jsonl") if r.get("task_split") == "train"]
    dev = [r for r in load(V1 / "dev_clean.jsonl") if r.get("task_split") == "dev"]
    train_hard, train_manifest = select_hardneg(train)
    dev_hard, dev_manifest = select_hardneg(dev)
    train_path = DERIVED / "hardneg_train_v3.jsonl"
    dev_path = DERIVED / "hardneg_dev_v3.jsonl"
    for path, rows in ((train_path, train_hard), (dev_path, dev_hard)):
        path.write_text("".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")
    dump(OUT / "train_manifest.json", {**train_manifest, "rows": len(train_hard)})
    dump(OUT / "dev_manifest.json", {**dev_manifest, "rows": len(dev_hard)})

    metrics = {
        "clean_dev_reference": {"raw": lexical_probe(train, dev, "raw"), "masked": lexical_probe(train, dev, "masked_tokens"), "length": length_probe(train, dev)},
        "hardneg_v3": {"raw": lexical_probe(train_hard, dev_hard, "raw"), "masked": lexical_probe(train_hard, dev_hard, "masked_tokens"), "length": length_probe(train_hard, dev_hard)},
        "counts": {"train_rows": len(train_hard), "train_tasks": len(train_manifest["tasks"]), "dev_rows": len(dev_hard), "dev_tasks": len(dev_manifest["tasks"])},
        "generator_counts": {"train": dict(Counter(r.get("generator") for r in train_hard)), "dev": dict(Counter(r.get("generator") for r in dev_hard))},
    }
    dump(OUT / "metrics.json", metrics)
    dump(OUT / "provenance_audit.json", provenance_audit(train + dev))
    dump(OUT / "config.json", {"seed": SEED, "selection": "one human plus three distinct-generator AI rows ranked by 0.70*log-length delta + 0.30*(1-masked-token cosine)", "train_source": "v1/train_all", "dev_source": "v1/dev_clean", "test_deserialized": False, "gpu_run": False, "project_provenance": "unavailable"})
    (OUT / "hypothesis.md").write_text("""# H3 v3 hard-negative hypothesis\n\nSTACAD alignment exposes task/file identity but no repository or project identity.  This CPU pass therefore uses a conservative same-task hard-negative view: one human and three distinct-generator AI rows selected by a masked-token similarity plus log-length score.  It does not call task identity project provenance.\n\nThe pre-registered readout is the masked and raw lexical probe on the hard-negative train/dev pair view, together with the untouched clean-dev reference.  A future GPU manifest is allowed only if project provenance is supplied or the claim is explicitly limited to task-heldout detection, the clean and hard-negative views are frozen, and the main model has a pre-registered gain over the corresponding lexical control.\n\nThis pass is CPU-only; test rows remain unopened and the closed C0-C3 and H3 diagnostic batches are not rerun.\n""", encoding="utf-8")
    (OUT / "report.md").write_text("""# H3 v3 hard-negative CPU revision\n\n## Decision\n\nThis is a data construction and diagnostic package, not GPU authorization.  `project_provenance=unavailable`; task-heldout is the strongest supported claim until repository/project evidence is added.\n\n## Readout\n\n| view | raw task-macro | masked task-macro | length task-macro |\n|---|---:|---:|---:|\n| clean-dev reference | {cr:.4f} | {cm:.4f} | {cl:.4f} |\n| hard-negative v3 | {hr:.4f} | {hm:.4f} | {hl:.4f} |\n\nThe hard-negative view is selected per task and generator from the existing aligned rows; it does not create synthetic labels.  The clean-dev reference remains the primary untouched check.\n""".format(cr=metrics["clean_dev_reference"]["raw"]["task_macro_auroc"], cm=metrics["clean_dev_reference"]["masked"]["task_macro_auroc"], cl=metrics["clean_dev_reference"]["length"]["task_macro_auroc"], hr=metrics["hardneg_v3"]["raw"]["task_macro_auroc"], hm=metrics["hardneg_v3"]["masked"]["task_macro_auroc"], hl=metrics["hardneg_v3"]["length"]["task_macro_auroc"]) + "\n", encoding="utf-8")
    (OUT / "commands.txt").write_text("python scripts/h3_hardneg_revision_v3.py\n", encoding="utf-8")
    (OUT / "env.json").write_text(json.dumps({"python": sys.version, "platform": platform.platform(), "numpy": np.__version__}, indent=2) + "\n", encoding="utf-8")
    (OUT / "git_head.txt").write_text(subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip() + "\n", encoding="utf-8")
    write_sums(OUT)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
