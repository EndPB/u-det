#!/usr/bin/env python3
"""H3 v2 CPU data revision and controlled-dev gate.

This is deliberately a data-side pass.  It never deserializes test rows and
does not start a GPU job.  It freezes a length-controlled dev view, reports
transform acceptance by label/language/generator/length bin, and stratifies
parser failures before a future GPU manifest can be issued.
"""
from __future__ import annotations

import hashlib
import json
import math
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
V1_ART = ROOT / "d-det/artifacts/h3_local_revision_v1_2026-10-10"
OUT = ROOT / "d-det/artifacts/h3_local_revision_v2_2026-10-11"
DERIVED = ROOT / "d-det/data/h3_stacad_revision_v2"
SEED = 20261011
LOG: list[str] = []


def log(message: str) -> None:
    line = f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {message}"
    print(line, flush=True)
    LOG.append(line)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def json_dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def code_len(row: dict) -> float:
    return float(np.log1p(len(row.get("code", "").encode("utf-8"))))


def length_bins(rows: list[dict]) -> tuple[dict[str, list[float]], dict[str, int]]:
    """Fit label-blind language quartiles on train and apply to all rows."""
    values: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        values[row["language"]].append(code_len(row))
    edges = {lang: np.quantile(vals, [0.25, 0.5, 0.75]).tolist() for lang, vals in values.items()}
    return edges, {lang: len(vals) for lang, vals in values.items()}


def assign_bin(value: float, cuts: list[float]) -> int:
    return int(np.searchsorted(np.asarray(cuts), value, side="right"))


def controlled_dev(train: list[dict], dev: list[dict]) -> tuple[list[dict], dict]:
    """One human plus nearest-length AI outputs per task, preferring generators."""
    edges, train_counts = length_bins(train)
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in dev:
        groups[row["task_id"]].append(row)
    out, manifest = [], {}
    for task_id in sorted(groups):
        group = groups[task_id]
        humans = [r for r in group if r["label"] == 0]
        ais = [r for r in group if r["label"] == 1]
        if not humans or not ais:
            continue
        human = sorted(humans, key=lambda r: r["row_id"])[0]
        hv = code_len(human)
        # Keep distinct generators when possible; then use global nearest length.
        chosen = []
        for generator in sorted({r.get("generator") for r in ais}):
            candidates = [r for r in ais if r.get("generator") == generator]
            chosen.append(min(candidates, key=lambda r: (abs(code_len(r) - hv), r["row_id"])))
        chosen = sorted(chosen, key=lambda r: (abs(code_len(r) - hv), r["row_id"]))[:3]
        selected = [human] + chosen
        for row in selected:
            copy = dict(row)
            copy["eval_view"] = "controlled_dev_v2"
            copy["length_log1p_bytes"] = code_len(row)
            copy["length_bin"] = assign_bin(code_len(row), edges[row["language"]])
            copy["matched_to_human_delta"] = abs(code_len(row) - hv)
            out.append(copy)
        manifest[task_id] = {
            "human": human["row_id"],
            "ai": [r["row_id"] for r in chosen],
            "human_length_log1p_bytes": hv,
            "ai_delta": [abs(code_len(r) - hv) for r in chosen],
        }
    return out, {"edges": edges, "train_counts": train_counts, "tasks": manifest}


def auroc(y, score) -> float:
    y = np.asarray(y)
    score = np.asarray(score)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, score))


def task_macro(rows: list[dict], scores: np.ndarray) -> float:
    by_task: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        by_task[row["task_id"]].append(i)
    vals = []
    for idx in by_task.values():
        y = [rows[i]["label"] for i in idx]
        if len(set(y)) > 1:
            vals.append(auroc(y, scores[idx]))
    return float(np.mean(vals)) if vals else float("nan")


def lexical_probe(train: list[dict], eval_rows: list[dict]) -> dict:
    vec = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=2, max_features=120000, sublinear_tf=True)
    x_train = vec.fit_transform([r["code"] for r in train])
    x_eval = vec.transform([r["code"] for r in eval_rows])
    clf = LogisticRegression(C=2.0, max_iter=100, solver="liblinear", random_state=SEED)
    clf.fit(x_train, [r["label"] for r in train])
    score = clf.decision_function(x_eval)
    return {"row_auroc": auroc([r["label"] for r in eval_rows], score), "task_macro_auroc": task_macro(eval_rows, score), "rows": len(eval_rows)}


def length_probe(train: list[dict], eval_rows: list[dict]) -> dict:
    langs = sorted({r["language"] for r in train + eval_rows})
    lang_map = {x: i for i, x in enumerate(langs)}
    def feat(rows):
        return np.asarray([[code_len(r), len(r.get("code", "").split()), lang_map[r["language"]]] for r in rows], dtype=float)
    scaler = StandardScaler().fit(feat(train))
    clf = LogisticRegression(C=1.0, max_iter=200, random_state=SEED)
    clf.fit(scaler.transform(feat(train)), [r["label"] for r in train])
    score = clf.decision_function(scaler.transform(feat(eval_rows)))
    return {"row_auroc": auroc([r["label"] for r in eval_rows], score), "task_macro_auroc": task_macro(eval_rows, score), "rows": len(eval_rows)}


def view_probe(train_rows: list[dict], variant_rows: list[dict], edges: dict) -> dict:
    accepted = defaultdict(set)
    by_parent = {}
    for r in variant_rows:
        key = (r["parent_row_id"], r["transform"])
        by_parent[key] = r
        if r.get("status") == "accepted":
            accepted[r["parent_row_id"]].add(r["transform"])
    parents = {key[0] for key in by_parent}
    parent = {r["row_id"]: r for r in train_rows}
    features, labels, task_ids = [], [], []
    strata = Counter()
    rates = defaultdict(lambda: [0, 0])
    for row_id, row in parent.items():
        if row_id not in parents or row["task_split"] != "train":
            continue
        vals = tuple(int(t in accepted[row_id]) for t in ("comment", "whitespace", "rename"))
        features.append(vals)
        labels.append(row["label"])
        task_ids.append(row["task_id"])
        b = assign_bin(code_len(row), edges[row["language"]])
        strata[(str(row["label"]), row["language"], row.get("generator"), b)] += 1
        for j, t in enumerate(("comment", "whitespace", "rename")):
            rates[(str(row["label"]), row["language"], row.get("generator"), b, t)][0] += vals[j]
            rates[(str(row["label"]), row["language"], row.get("generator"), b, t)][1] += 1
    features = np.asarray(features, dtype=float)
    labels = np.asarray(labels)
    held = np.asarray([hashlib.sha256(t.encode()).hexdigest()[:8] for t in task_ids])
    held = np.asarray([int(x, 16) % 5 == 0 for x in held], dtype=bool)
    fit, ev = ~held, held
    clf = LogisticRegression(C=1.0, max_iter=200, random_state=SEED)
    clf.fit(features[fit], labels[fit])
    score = clf.decision_function(features[ev])
    eval_rows = [{"task_id": task_ids[i], "label": int(labels[i])} for i in np.where(ev)[0]]
    result = {
        "fit_rows": int(fit.sum()), "eval_rows": int(ev.sum()), "eval_tasks": len({task_ids[i] for i in np.where(ev)[0]}),
        "row_auroc": auroc(labels[ev], score), "task_macro_auroc": task_macro(eval_rows, score),
        "features": ["comment_accepted", "whitespace_accepted", "rename_accepted"],
        "accepted_by_stratum": {"|".join(map(str, k)): {"accepted": v[0], "rows": v[1], "rate": (v[0] / v[1] if v[1] else None)} for k, v in sorted(rates.items())},
        "stratum_count": {"|".join(map(str, k)): v for k, v in sorted(strata.items())},
    }
    return result


def parser_strata(train: list[dict], dev: list[dict]) -> dict:
    invalid = {}
    obj = json.loads((V1_ART / "parser_audit.json").read_text(encoding="utf-8"))
    for item in obj.get("invalid_examples", []):
        invalid[item["row_id"]] = {"errors": item.get("errors", 0), "missing": item.get("missing", 0)}
    out = defaultdict(lambda: [0, 0, 0])
    for row in train + dev:
        key = (row["language"], str(row["label"]), row.get("generator"))
        out[key][0] += 1
        if row["row_id"] in invalid:
            out[key][1] += 1
            out[key][2] += int(invalid[row["row_id"]]["errors"] + invalid[row["row_id"]]["missing"] > 0)
    return {"rows": {"|".join(map(str, k)): {"rows": v[0], "invalid_examples_listed": v[1], "invalid_or_missing": v[2], "rate": v[2] / v[0] if v[0] else None} for k, v in sorted(out.items())}, "source": "v1 parser_audit.json", "invalid_examples_only": True}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    DERIVED.mkdir(parents=True, exist_ok=True)
    train = read_jsonl(V1 / "train_all.jsonl")
    dev = read_jsonl(V1 / "dev_clean.jsonl")
    train = [r for r in train if r.get("task_split") == "train"]
    dev = [r for r in dev if r.get("task_split") == "dev"]
    edges, _ = length_bins(train)
    controlled, manifest = controlled_dev(train, dev)
    controlled_path = DERIVED / "controlled_dev_v2.jsonl"
    controlled_path.write_text("".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n" for r in controlled), encoding="utf-8")
    json_dump(OUT / "controlled_dev_manifest.json", manifest)
    log(f"controlled_dev: {len(controlled)} rows / {len(manifest['tasks'])} tasks")

    variants = read_jsonl(V1_ART / "variant_records.jsonl")
    view = view_probe(train, variants, edges)
    json_dump(OUT / "view_probe_stratified.json", view)
    log(f"view probe: row={view['row_auroc']:.4f} task={view['task_macro_auroc']:.4f}")

    parser = parser_strata(train, dev)
    json_dump(OUT / "parser_failure_strata.json", parser)
    clean_lex = lexical_probe(train, dev)
    control_lex = lexical_probe(train, controlled)
    clean_len = length_probe(train, dev)
    control_len = length_probe(train, controlled)
    metrics = {
        "clean_dev": {"lexical": clean_lex, "length": clean_len},
        "controlled_dev_v2": {"lexical": control_lex, "length": control_len},
        "v1_reference": json.loads((V1_ART / "metrics.json").read_text(encoding="utf-8"))["balanced_train"],
        "view_probe": {k: view[k] for k in ("fit_rows", "eval_rows", "eval_tasks", "row_auroc", "task_macro_auroc")},
    }
    json_dump(OUT / "metrics.json", metrics)
    config = {"seed": SEED, "source": "h3_stacad_revision_v1", "test_deserialized": False, "length_bin_fit": "train-only, per-language quartiles", "controlled_dev": "one human plus up to three distinct-generator nearest-log-length AI rows per task", "gpu_run": False}
    json_dump(OUT / "config.json", config)
    json_dump(OUT / "env.json", {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__})
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    (OUT / "git_head.txt").write_text(head + "\n", encoding="utf-8")
    (OUT / "commands.txt").write_text("python scripts/h3_local_revision_v2.py\n", encoding="utf-8")
    (OUT / "hypothesis.md").write_text("""# H3 v2 本机数据修订假设\n\n若将长度控制冻结为训练拟合的语言内分位桶，并在每个 dev task 选择跨 generator 的最近长度 AI 对照，同时按 label×language×generator×length_bin 检查变体接受率，则可区分长度残余、变体标记和 parser 失败造成的表面信号。只有 clean-dev 与 controlled-dev 的控制读数都下降、view probe 接近机会、且 project provenance 补齐，才允许生成新的 GPU batch_manifest。\n\n本轮只做 CPU 数据审计与轻量 probe；test 未反序列化，旧 GPU 诊断不重跑。\n""", encoding="utf-8")
    reasons = []
    if metrics["clean_dev"]["length"]["task_macro_auroc"] >= 0.70 or metrics["controlled_dev_v2"]["length"]["task_macro_auroc"] >= 0.70:
        reasons.append("length_probe_above_0.70")
    if metrics["clean_dev"]["lexical"]["task_macro_auroc"] >= 0.70 or metrics["controlled_dev_v2"]["lexical"]["task_macro_auroc"] >= 0.70:
        reasons.append("lexical_probe_remains_high")
    if view["task_macro_auroc"] >= 0.60:
        reasons.append("view_probe_above_0.60")
    reasons.append("project_repository_provenance_unavailable")
    json_dump(OUT / "gate.json", {"decision": "revise_data", "formal_gpu_ready": False, "test_deserialized": False, "reasons": reasons, "old_gpu_batch": "closed", "next_gpu_requires_new_manifest": True})
    report = f"""# H3 v2 本机数据修订报告\n\n## 结果\n\n- controlled-dev：{len(controlled)} 行、{len(manifest['tasks'])} 个 task；按 train-only、language 内 log1p(bytes) 四分位冻结，task 内保留 human 与最多 3 个不同 generator 的最近长度 AI。\n- clean-dev lexical task-macro AUROC：{clean_lex['task_macro_auroc']:.4f}；controlled-dev：{control_lex['task_macro_auroc']:.4f}。\n- clean-dev length task-macro AUROC：{clean_len['task_macro_auroc']:.4f}；controlled-dev：{control_len['task_macro_auroc']:.4f}。\n- view-probe task-macro AUROC：{view['task_macro_auroc']:.4f}；结果已按 label、language、generator、length_bin 展开。\n- parser 失败按语言、标签、generator 分层；当前文件来自 v1 invalid_examples 清单，未把解析失败行删除。\n\n## 裁定\n\n`revise_data`，`formal_gpu_ready=false`。长度/词法控制仍高于准入线，project/repository provenance 仍不可用；本轮没有生成新的 GPU 批次，也没有读取 test。旧 H3 GPU 诊断已关闭，不在旧冻结包上重跑。\n\n下一步是补齐项目来源并重新构造 label×language×generator×length_bin 平衡包，随后重新跑全量 train/dev 闸门；只有全部通过才制作一次性 GPU manifest。\n"""
    (OUT / "report.md").write_text(report, encoding="utf-8")
    sums = []
    for path in sorted(OUT.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS.txt":
            sums.append(f"{sha256_file(path)}  {path.relative_to(OUT).as_posix()}")
    (OUT / "SHA256SUMS.txt").write_text("\n".join(sums) + "\n", encoding="utf-8")
    (OUT / "run.log").write_text("\n".join(LOG) + "\n", encoding="utf-8")
    log("decision=revise_data; formal_gpu_ready=false")


if __name__ == "__main__":
    main()
