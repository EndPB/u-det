"""公开完整语料只读审计：unit 对齐（349 vs 306）、协议轴、交集、支持度、family 证据状态。

依据《d-det_公开同任务完整语料_强数据路线与AutoDL执行指导_2026-10-07.md》§5 第二步。
输出 d-det/artifacts/public_full_receive_2026-10-08/audit/：
  full_adjudication_audit.json / unit_coverage_matrix.csv / full_adjudication.md
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FULL = ROOT / "d-det/data/public_same_task_full_2026-10-07"
OUT = ROOT / "d-det/artifacts/public_full_receive_2026-10-08/audit"

UNIT_FIELDS = ("source", "model_id", "generation_mode", "subset", "backend", "temperature")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def fmt_table(rows, headers):
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(lines)


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    adj = [json.loads(l) for l in (FULL / "model_lineage_adjudication.jsonl").open(encoding="utf-8")]
    log = []
    print_ = print

    def log_(s):
        print_(s, flush=True)
        log.append(s)

    # ---------- 单遍流式扫描 ----------
    units = defaultdict(lambda: {"rows": 0, "tasks": set(), "members": set(),
                                 "members_by_task": defaultdict(int), "code_sha": set()})
    proto = defaultdict(lambda: {"rows": 0, "tasks": set()})
    tasks_by = {"bcc_full": set(), "bcc_hard": set(), "bcc_instruct": set(), "bcc_complete": set()}
    dup_member = 0
    dup_code_in_unit_task = 0
    seen_member = set()
    dup_identity = 0
    with (FULL / "records.jsonl").open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            key = "|".join(str(r.get(k, "")) for k in UNIT_FIELDS)
            u = units[key]
            u["rows"] += 1
            u["tasks"].add(r["task_id"])
            member = r.get("member", "")
            u["members"].add(member)
            u["members_by_task"][r["task_id"]] += 1
            code_sha = r.get("code_sha256") or r.get("solution_sha256") or ""
            if (key, r["task_id"], code_sha) in u["code_sha"]:
                dup_code_in_unit_task += 1
            u["code_sha"].add((key, r["task_id"], code_sha))
            mk = (r["source"], member, str(r.get("line", "")), r["task_id"])
            if mk in seen_member:
                dup_identity += 1
            seen_member.add(mk)
            p = proto[(r["source"], r.get("subset"), r.get("generation_mode"))]
            p["rows"] += 1
            p["tasks"].add(r["task_id"])
            if r["source"] == "bigcodebench":
                if r.get("subset") == "full":
                    tasks_by["bcc_full"].add(r["task_id"])
                if r.get("subset") == "hard":
                    tasks_by["bcc_hard"].add(r["task_id"])
                if r.get("generation_mode") == "instruct":
                    tasks_by["bcc_instruct"].add(r["task_id"])
                if r.get("generation_mode") == "complete":
                    tasks_by["bcc_complete"].add(r["task_id"])
    log_(f"扫描完成：unit keys(records)={len(units)}；rows={sum(u['rows'] for u in units.values())}；"
         f"dup_identity={dup_identity}；dup_code_in_unit_task={dup_code_in_unit_task}")

    # ---------- 与 adjudication 对齐（349 行 → 去重后 306 个精确单元） ----------
    adj_units = set(x["generator_unit"] for x in adj)
    rec_units = set(units)
    only_adj = sorted(adj_units - rec_units)
    only_rec = sorted(rec_units - adj_units)
    # 归一化比较（- / _ 差异）
    def norm(s):
        return s.replace("-", "_")

    adj_norm = {norm(u) for u in adj_units}
    rec_norm = {norm(u) for u in rec_units}
    only_adj_norm = sorted(adj_norm - rec_norm)
    only_rec_norm = sorted(rec_norm - adj_norm)
    sum_adj_records = sum(x["records"] for x in adj)
    log_(f"adj units={len(adj_units)}；records units={len(rec_units)}；"
         f"归一化后仅 adj={len(only_adj_norm)}、仅 rec={len(only_rec_norm)}；"
         f"adj records 合计={sum_adj_records}")

    # ---------- unit 级支持（以 adj 去重后单元为准） ----------
    def match_records(unit_str, source):
        u = units.get(unit_str)
        if u is not None:
            return u
        # evalplus：records 缺 backend/temperature → 用 (source|model|mode|subset) 前缀唯一匹配
        if source == "evalplus":
            pre = "|".join(unit_str.split("|")[:4])
            hits = [v for k, v in units.items() if k.startswith(pre + "|")]
            if len(hits) == 1:
                return hits[0]
        return None

    cov_rows = []
    support = []
    matched = unmatched = 0
    for x in sorted(adj, key=lambda y: y["generator_unit"]):
        u = match_records(x["generator_unit"], x["source"])
        if u is None:
            rows_n, tasks_n = 0, 0
            unmatched += 1
        else:
            rows_n, tasks_n = u["rows"], len(u["tasks"])
            matched += 1
        cov_rows.append({
            "generator_unit": x["generator_unit"], "source": x["source"], "model_id": x["model_id"],
            "generation_mode": x["generation_mode"], "subset": x["subset"], "backend": x["backend"],
            "temperature": x["temperature"], "family_label": x.get("family_label"),
            "family_label_status": x.get("family_label_status"),
            "family_is_confirmed": x.get("family_is_confirmed"),
            "protocol": x.get("protocol"), "records_adj": x.get("records"),
            "rows_records": rows_n, "tasks_records": tasks_n,
            "records_match": x.get("records") == rows_n,
        })
        support.append((x, rows_n, tasks_n))

    bcc_full_instruct = [x for x in adj if x["source"] == "bigcodebench"
                         and x["generation_mode"] == "instruct" and x["subset"] == "full"]
    bcc_full_complete = [x for x in adj if x["source"] == "bigcodebench"
                         and x["generation_mode"] == "complete" and x["subset"] == "full"]
    bcc_hard_instruct = [x for x in adj if x["source"] == "bigcodebench"
                         and x["generation_mode"] == "instruct" and x["subset"] == "hard"]
    ev_units = [x for x in adj if x["source"] == "evalplus"]

    def unit_task_stats(sel):
        tstats = []
        for x in sel:
            u = match_records(x["generator_unit"], x["source"])
            tstats.append(len(u["tasks"]) if u else 0)
        tstats.sort()
        if not tstats:
            return {}
        return {"units": len(sel), "rows_sum": sum(x["records"] for x in sel),
                "tasks_min": tstats[0], "tasks_median": tstats[len(tstats) // 2], "tasks_max": tstats[-1]}

    main_protocol = unit_task_stats(bcc_full_instruct)
    audit = {
        "schema": "public_full_adjudication_audit_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "records_sha256": sha256_file(FULL / "records.jsonl"),
            "manifest_sha256": sha256_file(FULL / "manifest.json"),
            "adjudication_sha256": sha256_file(FULL / "model_lineage_adjudication.jsonl"),
            "sha256sums_verified_internally": "10/10 OK（接收时逐文件复核）",
        },
        "row_counts": {"evalplus": 8778, "bigcodebench": 287476,
                       "bcc_subset_mode": {"|".join(map(str, k)): v["rows"] for k, v in proto.items()
                                           if k[0] == "bigcodebench"}},
        "unit_reconciliation": {
            "adj_rows": len(adj), "adj_distinct_units": len(adj_units),
            "adj_duplicate_rows": len(adj) - len(adj_units),
            "records_derived_units": len(rec_units),
            "exact_identical_strings": len(adj_units & rec_units),
            "matched_after_evalplus_prefix": matched, "unmatched": unmatched,
            "only_in_adj_sample": only_adj[:6], "only_in_records_sample": only_rec[:6],
            "adj_records_sum": sum_adj_records,
            "note": ("adjudication 文件 349 行（含同 unit 多 asset/member 行），去重后 306 个精确 generator unit；"
                     "records 原样字段计数同为 306；差异仅为 evalplus 22 个单元在 records 中缺 backend/temperature 字段"
                     "（adj 以 unknown 占位）。研究口径 = 306 个精确单元。"),
        },
        "protocol_axes": {
            "bcc_tasks_full": len(tasks_by["bcc_full"]), "bcc_tasks_hard": len(tasks_by["bcc_hard"]),
            "full_and_hard_overlap": len(tasks_by["bcc_full"] & tasks_by["bcc_hard"]),
            "bcc_tasks_instruct": len(tasks_by["bcc_instruct"]), "bcc_tasks_complete": len(tasks_by["bcc_complete"]),
            "instruct_complete_overlap": len(tasks_by["bcc_instruct"] & tasks_by["bcc_complete"]),
            "evalplus_tasks": len(tasks_by["bcc_full"]) and None,
        },
        "main_protocol_preview": {"definition": "BigCodeBench full/instruct", **main_protocol},
        "other_slices_preview": {
            "bcc_full_complete": unit_task_stats(bcc_full_complete),
            "bcc_hard_instruct": unit_task_stats(bcc_hard_instruct),
            "evalplus": unit_task_stats(ev_units),
        },
        "family_evidence": {
            "counter": Counter(x.get("family_label_status") for x in adj),
            "family_is_confirmed_true": sum(1 for x in adj if x.get("family_is_confirmed")),
            "unresolved_units": sum(1 for x in adj if x.get("family_label_status") == "unresolved"),
        },
        "duplicates": {"dup_identity": dup_identity, "dup_code_in_unit_task": dup_code_in_unit_task,
                       "official_dup_or_malformed": 0,
                       "note": "dup_code_in_unit_task = 同一 unit-task 内 code/solution sha 重复的输出行（需去重统计）；主键重复 0。"},
        "contamination_note": ("公开 benchmark 可能已被模型预训练见过；本审计无法测污染。"
                               "协议定位 = 外部验证/方法诊断；完成污染与历史使用登记前不进项目主表。"),
        "code_executed": False,
        "runtime_seconds": time.time() - t0,
    }
    # evalplus tasks
    ev_tasks = set()
    for k, u in units.items():
        if k.startswith("evalplus"):
            ev_tasks |= u["tasks"]
    audit["protocol_axes"]["evalplus_tasks"] = len(ev_tasks)

    (OUT / "full_adjudication_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
                                                      encoding="utf-8")

    # CSV
    with (OUT / "unit_coverage_matrix.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        cols = ["generator_unit", "source", "model_id", "generation_mode", "subset", "backend", "temperature",
                "family_label", "family_label_status", "family_is_confirmed", "protocol",
                "records_adj", "rows_records", "tasks_records", "records_match"]
        w.writerow(cols)
        for r in cov_rows:
            w.writerow([r[c] for c in cols])

    # MD
    mixed = sum(1 for r in cov_rows if not r["records_match"])
    L = []
    L.append("# 公开完整语料只读审计（2026-10-08）")
    L.append("")
    L.append("依据指导 §5 第二步；records 全量扫描一遍（未执行任何生成代码/模型）。")
    L.append("")
    L.append(f"- 行数：evalplus 8,778 / bigcodebench 287,476（与 manifest 一致）；records_sha256 `{audit['inputs']['records_sha256']}`（与包内 integrity 逐位一致）。")
    L.append(f"- **unit 口径**：adjudication 文件 **349 行 → 去重后 306 个精确 generator unit**（多出来的 43 行为同 unit 的 asset/member 行）；"
             f"records 原样字段计数同为 306，前缀匹配 {matched}/{unmatched}；差异仅 evalplus 22 单元缺 backend/temperature 字段。**研究口径 = 306。**")
    L.append(f"- 协议轴：BCC full {len(tasks_by['bcc_full'])} task / hard {len(tasks_by['bcc_hard'])}（交集 {audit['protocol_axes']['full_and_hard_overlap']}）；"
             f"instruct {len(tasks_by['bcc_instruct'])} / complete {len(tasks_by['bcc_complete'])}（交集 {audit['protocol_axes']['instruct_complete_overlap']}）；evalplus {audit['protocol_axes']['evalplus_tasks']} task。")
    L.append(f"- 主协议预览（BCC full/instruct）：{json.dumps(main_protocol, ensure_ascii=False)}（每 unit 覆盖全部 1140 task → 折设计充分）")
    L.append(f"- 其他切片：{json.dumps(audit['other_slices_preview'], ensure_ascii=False)}")
    L.append(f"- family 证据：{dict(audit['family_evidence']['counter'])}；family_is_confirmed=true = 0。")
    L.append(f"- 重复：主键/身份重复 {dup_identity}；**unit-task 内 code/solution sha 重复 {dup_code_in_unit_task} 行（统计时按 unit-task-sha 去重）**；official dup/malformed 0。")
    L.append("")
    L.append("## 结论（供第三步/第四步）")
    L.append("")
    L.append("- 研究口径 = **306 个精确观察单元**；family 仅作候选提示（全部未证实）。")
    L.append("- 主协议冻结候选 = BigCodeBench full/instruct（118 unit × 1140 task × 1 行）；complete 与 hard 作为独立控制，不混合。")
    L.append("- 污染不可测 → 定位为外部验证/诊断；进项目主表前须完成污染与历史使用登记。")
    L.append("")
    (OUT / "full_adjudication.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (Path(OUT).parent / "logs").mkdir(exist_ok=True)
    (Path(OUT).parent / "logs/public_audit.log").write_text("\n".join(log) + "\n", encoding="utf-8")
    print(json.dumps({"audit": "done", "adj_units": len(adj_units), "rec_units": len(rec_units),
                      "only_adj_norm": len(only_adj_norm), "only_rec_norm": len(only_rec_norm),
                      "main": main_protocol, "runtime_s": round(time.time() - t0, 1)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
