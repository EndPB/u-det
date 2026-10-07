"""Stage D0: streaming support audit for h2_alignment_v3 fold_plan + pair_index.

只读审计：不训练、不计算任何 score、不修改源数据。按 pair 级成员规则（与
build_h2_pair_benchmark_generator_folds_v1.py 相同）为每个 fold 统计支持度：

- positive（same_family_cross_generator）: pair 的 family == fold.family；
- negative（cross_family）: fold.family ∈ {left_family, right_family}；
- role：train/dev = 行内 task_split 匹配且 heldout 不在任一端点；
        test = task_split==test 且 heldout 在端点中；
- source 隔离（fold.source == row.source）；stacad 无 fold（control-only）。

补充：fold_plan 的 train/dev/test_rows 是构建器口径的“记录级”计数（v1：
全源去重记录；v3 llm：outputs 记录），本审计的 pair 级计数单独报告，两者
不可混用。

输出：<out>/alignment_support.json（+ 控制台摘要）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALIGN = ROOT / "d-det/data/h2_alignment_v3"
AB_CORE = ROOT / "d-det/data/h2_authorbench_dcan/core.jsonl"
LLM_RESP = ROOT / "d-det/data/h2_llm_codegen_v2/response_tasks.jsonl"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def stream_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if line.strip():
                yield line_no, json.loads(line)


def build_ab_line_hashes() -> dict[int, str]:
    out = {}
    for line_no, row in stream_jsonl(AB_CORE):
        out[line_no] = hashlib.sha256((row.get("code") or "").encode()).hexdigest()
    return out


def endpoint_hash(source: str, record_id: str, left_ref: str, ab_hashes: dict[int, str]):
    if source == "llm_codegen_v2":
        suffix = record_id.split(":", 1)[1]
        return suffix if len(suffix) == 64 else None
    if source == "authorbench_dcan":
        # ref 形如 ...core.jsonl#L8808（1-based 行号）
        try:
            ln = int(left_ref.rsplit("#L", 1)[1])
        except Exception:
            return None
        return ab_hashes.get(ln)
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=ROOT / "d-det/artifacts/stage_d_support_2026-10-07")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    folds = json.loads((ALIGN / "fold_plan.json").read_text(encoding="utf-8"))["folds"]
    tasks = {r["task_key"]: r for _, r in stream_jsonl(ALIGN / "task_index.jsonl")}
    ab_hashes = build_ab_line_hashes()

    # 每个 fold 的累计器
    acc = {}
    for f in folds:
        acc[f["fold_id"]] = {
            "fold": f,
            "roles": {r: {"pairs": 0, "pos": 0, "neg": 0, "tasks": set(),
                          "families": set(), "generators": set()} for r in ("train", "dev", "test")},
            "heldout_in_train": 0, "heldout_in_dev": 0,
            "hash_roles": defaultdict(set),
            "unresolved_hash": 0, "unresolved": 0,
        }

    for _, row in stream_jsonl(ALIGN / "pair_index.jsonl"):
        src = row.get("source")
        if src not in ("authorbench_dcan", "llm_codegen_v2"):
            continue  # stacad = control-only（无 fold）
        split = row.get("task_split")
        pos = row.get("positive_for_family_invariance") is True
        lf, rf = row.get("left_family"), row.get("right_family")
        gens = {row.get("left_generator"), row.get("right_generator")}
        for f in folds:
            if f["source"] != src:
                continue
            fam = f["family"]
            if pos and not (lf == rf == fam):
                continue
            if (not pos) and fam not in (lf, rf):
                continue
            held = f["heldout_generator"]
            if split in ("train", "dev") and held not in gens:
                role = split
            elif split == "test" and held in gens:
                role = "test"
            else:
                continue
            a = acc[f["fold_id"]]
            rr = a["roles"][role]
            rr["pairs"] += 1
            rr["pos" if pos else "neg"] += 1
            rr["tasks"].add(row.get("task_id"))
            for x in (lf, rf):
                if x:
                    rr["families"].add(x)
            for x in gens:
                if x:
                    rr["generators"].add(x)
            if role in ("train", "dev") and held in gens:
                a[f"heldout_in_{role}"] += 1
            # endpoint hashes
            for rid, ref in ((row.get("left_record_id"), row.get("left_ref")),
                             (row.get("right_record_id"), row.get("right_ref"))):
                h = endpoint_hash(src, rid or "", ref or "", ab_hashes)
                if h:
                    a["hash_roles"][h].add(role)
                else:
                    a["unresolved_hash"] += 1

    fold_reports = []
    for f in folds:
        a = acc[f["fold_id"]]
        roles = {}
        for r in ("train", "dev", "test"):
            rr = a["roles"][r]
            roles[r] = {"pairs": rr["pairs"], "pos": rr["pos"], "neg": rr["neg"],
                        "tasks": len(rr["tasks"]), "families": sorted(rr["families"]),
                        "generators": sorted(rr["generators"])}
        t = {r: a["roles"][r]["tasks"] for r in ("train", "dev", "test")}
        overlap = {"train_dev": len(t["train"] & t["dev"]),
                   "train_test": len(t["train"] & t["test"]),
                   "dev_test": len(t["dev"] & t["test"])}
        xing = {"train_dev": 0, "train_test": 0, "dev_test": 0, "all_three": 0}
        for h, rs in a["hash_roles"].items():
            if len(rs) >= 3:
                xing["all_three"] += 1
            if {"train", "dev"} <= rs:
                xing["train_dev"] += 1
            if {"train", "test"} <= rs:
                xing["train_test"] += 1
            if {"dev", "test"} <= rs:
                xing["dev_test"] += 1
        # 缺失比例（按 fold 任务，从 task_index 取）
        fold_tasks = {k for r in ("train", "dev", "test") for k in a["roles"][r]["tasks"]}
        lang_missing = sum(1 for k in fold_tasks
                           if (tasks.get(k) or {}).get("language") in (None, ""))
        n_task = max(1, len(fold_tasks))
        fail = []
        if roles["train"]["pos"] == 0:
            fail.append("no_train_positive_support")
        if roles["dev"]["pos"] == 0:
            fail.append("no_dev_positive_support")
        if overlap["train_dev"] or overlap["train_test"] or overlap["dev_test"]:
            fail.append("task_overlap")
        if a["heldout_in_train"] or a["heldout_in_dev"]:
            fail.append("heldout_leak_train_dev")
        fold_reports.append({
            "fold_id": f["fold_id"], "source": f["source"], "family": f["family"],
            "heldout_generator": f["heldout_generator"], "seen_generators": f["seen_generators"],
            "declared_record_counts": {"train": f.get("train_rows"),
                                       "dev": f.get("dev_rows"), "test": f.get("test_rows")},
            "roles": roles,
            "task_overlap": overlap,
            "heldout_in_train_dev": {"train": a["heldout_in_train"], "dev": a["heldout_in_dev"]},
            "train_dev_both_labels": {"train": roles["train"]["pos"] > 0 and roles["train"]["neg"] > 0,
                                      "dev": roles["dev"]["pos"] > 0 and roles["dev"]["neg"] > 0},
            "same_family_cross_generator_positives": {r: roles[r]["pos"] for r in roles},
            "field_missing_ratio": {"language": round(lang_missing / n_task, 6),
                                    "base_model": "absent_in_index",
                                    "model_role": "absent_in_index",
                                    "repository_id": "absent_in_index"},
            "code_hash_crossing_splits": {"distinct_endpoint_hashes": len(a["hash_roles"]),
                                          **xing},
            "unresolved_endpoint_hashes": a["unresolved_hash"],
            "admitted": not fail,
            "fail_reasons": fail,
        })

    admitted = [r["fold_id"] for r in fold_reports if r["admitted"]]
    report = {
        "schema": "stage_d_alignment_support_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "fold_plan": {"path": "d-det/data/h2_alignment_v3/fold_plan.json",
                          "sha256": sha256_file(ALIGN / "fold_plan.json")},
            "pair_index": {"path": "d-det/data/h2_alignment_v3/pair_index.jsonl",
                           "sha256": sha256_file(ALIGN / "pair_index.jsonl")},
            "task_index": {"path": "d-det/data/h2_alignment_v3/task_index.jsonl",
                           "sha256": sha256_file(ALIGN / "task_index.jsonl")},
            "authorbench_core": {"path": "d-det/data/h2_authorbench_dcan/core.jsonl",
                                 "sha256": sha256_file(AB_CORE)},
            "llm_response_tasks": {"path": "d-det/data/h2_llm_codegen_v2/response_tasks.jsonl",
                                   "sha256": sha256_file(LLM_RESP)},
        },
        "membership_rule": ("positive: pair family == fold.family; negative: fold.family in "
                            "{left_family,right_family}; role: train/dev = task_split and heldout not in "
                            "endpoints; test = task_split=='test' and heldout in endpoints; sources isolated."),
        "notes": [
            "declared_record_counts 为构建器口径的记录级计数（v1 全源去重 / v3 llm outputs），与 pair 级计数不可混用。",
            "stacad 为 control-only（无 fold），未纳入本审计。",
            "base_model/model_role/repository_id 在 v3 索引与两源数据中均不存在（family 映射依据见 llm outputs.family_mapping_basis）。",
            "唯一正对来源：same_family_cross_generator（authorbench_dcan + llm_codegen_v2）；不得跨 source 合并平均。",
        ],
        "folds": fold_reports,
        "summary": {
            "fold_count": len(fold_reports),
            "admitted_count": len(admitted),
            "admitted_fold_ids": admitted,
            "diagnostic_fold_ids": [r["fold_id"] for r in fold_reports if not r["admitted"]],
        },
        "test_access": "Metadata/support audit only. No model scores computed; no hyperparameter selection.",
    }
    outp = args.out / "alignment_support.json"
    outp.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print("wrote", outp)


if __name__ == "__main__":
    main()
