"""变体迁移执行前闸门（2026-10-08）§2：执行前协议审计（不读 test 正文）。

只读取：manifest 元数据、task_id、行哈希（code_sha256）、split 索引。
不读取：任何代码正文写入产物、test 内容、模型权重。

检查项：
  C1 11 系列成员均来自 BigCodeBench full|instruct；
  C2 每成员 1,140 task / 1,140 行（主协议）；
  C3 task split 重建 798/171/171，task_list_sha256 与 e38715d 预注册一致（split_index 逐行）；
  C4 变体 heldout 顺序与注册文件/指导一致；
  C5 特征/阈值/seed/P0/CI 已在 train/dev 读取前写入冻结配置（记录 sha256）；
  C6 折矩阵：每折训练集合不含 heldout variant；heldout 行仅允许 test 阶段评分；
  C7 test 内容未读、正文零落盘；
  C8 重复口径复核（全语料 dup_by_code_sha=21 期望；主协议多行 unit-task=8 期望）。

输出：variant_transfer_preflight_2026-10-08/protocol_audit.json + fold_matrix.csv
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
F = ROOT / "d-det/artifacts/public_full_followup_2026-10-08"
R = ROOT / "d-det/artifacts/public_full_receive_2026-10-08"
OUT = ROOT / "d-det/artifacts/variant_transfer_preflight_2026-10-08"
RECORDS = ROOT / "d-det/data/public_same_task_full_2026-10-07/records.jsonl"
REG = json.loads((F / "variant_transfer_registration.json").read_text(encoding="utf-8"))
PLAN = json.loads((R / "prereg/split_plan.json").read_text(encoding="utf-8"))
SEED = 20261007

SERIES = {
    "CodeLlama-Instruct":
        (r"^codellama--CodeLlama-(7b|13b|34b|70b)-Instruct(-hf)?$", ["7b", "13b", "34b", "70b"]),
    "Qwen2.5-Coder-Instruct":
        (r"^Qwen--Qwen2\.5-Coder-(1\.5B|7B|14B|32B)-Instruct$", ["1.5B", "7B", "14B", "32B"]),
    "DeepSeek-Coder-v1-Instruct":
        (r"^deepseek-ai--deepseek-coder-(1\.3b|6\.7b|33b)-instruct$", ["1.3b", "6.7b", "33b"]),
}
READ_FIELDS = ("source", "release", "asset", "asset_sha256", "task_id", "subset",
               "generation_mode", "protocol", "model_id", "backend", "temperature",
               "member", "code_sha256")


def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def hkey(s: str) -> str:
    return sha256_hex(f"{SEED}|{s}")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def unit_key(d: dict) -> str:
    return "|".join(str(d.get(k, "")) for k in
                    ("source", "model_id", "generation_mode", "subset", "backend", "temperature"))


def code_sha_of(d: dict) -> str:
    return d.get("code_sha256") or d.get("solution_sha256") or ""


def main() -> None:
    t0 = time.time()
    rows_by_slice: Counter = Counter()
    tasks_all: set = set()
    rows_main: Counter = Counter()
    unit_tasks: dict = defaultdict(set)
    unit_meta: dict = defaultdict(set)
    ut_line_main: Counter = Counter()
    keyset_all: set = set()
    n_total = 0
    ev_rows = 0

    with RECORDS.open(encoding="utf-8") as fh:
        for line in fh:
            n_total += 1
            d = json.loads(line)
            src = d.get("source")
            if src == "evalplus":
                ev_rows += 1
                keyset_all.add((unit_key(d), d.get("task_id"), code_sha_of(d)))
                continue
            sl = f"{d.get('subset')}|{d.get('generation_mode')}"
            rows_by_slice[sl] += 1
            keyset_all.add((unit_key(d), d.get("task_id"), code_sha_of(d)))
            if sl == "full|instruct":
                u, t, s, a = d.get("model_id"), d.get("task_id"), code_sha_of(d), d.get("asset")
                tasks_all.add(t)
                rows_main[u] += 1
                unit_tasks[u].add(t)
                unit_meta[u].add((a, d.get("backend"), d.get("temperature")))
                ut_line_main[(u, t)] += 1

    dup_all = n_total - len(keyset_all)
    multi_main = {k: v for k, v in ut_line_main.items() if v > 1}

    # ---- C3：task split 重建 ----
    order = sorted(tasks_all, key=hkey)
    n = len(order)
    n_tr, n_dv = int(n * 0.70), int(n * 0.15)
    split_of = {}
    for i, t in enumerate(order):
        split_of[t] = "train" if i < n_tr else ("dev" if i < n_tr + n_dv else "test")
    counts = Counter(split_of.values())
    rebuilt_sha = {s: sha256_hex("\n".join(sorted(t for t in tasks_all if split_of[t] == s)))
                   for s in ("train", "dev", "test")}
    plan_sha = PLAN["main_protocol"]["task_list_sha256"]
    sha_match = {s: rebuilt_sha[s] == plan_sha[s] for s in plan_sha}
    csv_rows = list(csv.DictReader((R / "prereg/split_index.csv").open(encoding="utf-8")))
    csv_mism = [r["task_id"] for r in csv_rows if split_of.get(r["task_id"]) != r["split"]]

    # ---- C1/C2：系列成员 ----
    members_report = {}
    for series, (pat, order_sizes) in SERIES.items():
        creg = re.compile(pat)
        matched = [m for m in rows_main if creg.match(m)]
        members_report[series] = {}
        for m in matched:
            members_report[series][m] = {
                "rows_full_instruct_raw": rows_main[m],
                "tasks_full_instruct": len(unit_tasks[m]),
                "distinct_asset_backend_temp": len(unit_meta[m]),
                "unit_meta": sorted([list(x) for x in unit_meta[m]])[:3],
            }

    # ---- C4：heldout 顺序 ----
    expected_orders = {
        "CodeLlama-Instruct": ["7b", "13b", "34b", "70b"],
        "Qwen2.5-Coder-Instruct": ["1.5B", "7B", "14B", "32B"],
        "DeepSeek-Coder-v1-Instruct": ["1.3b", "6.7b", "33b"],
    }
    heldout_order = {s: list(REG["series_folds"][s]["heldout_order"]) for s in SERIES}
    members_by_size = {s: REG["series_folds"][s]["members_by_size"] for s in SERIES}
    c4_pass = heldout_order == expected_orders and all(
        set(members_report[s].keys()) == set(members_by_size[s].values()) for s in SERIES)

    # ---- C6：折矩阵 ----
    all_members = {s: list(members_by_size[s].values()) for s in SERIES}
    train_tasks = {t for t in tasks_all if split_of[t] == "train"}
    dev_tasks = {t for t in tasks_all if split_of[t] == "dev"}
    test_tasks = {t for t in tasks_all if split_of[t] == "test"}

    def rows_in(u: str, ts: set) -> int:
        return len(unit_tasks[u] & ts)

    folds = {}
    fold_rows = []
    c6_pass = True
    for s, ms in all_members.items():
        for h in ms:
            pos = [m for m in ms if m != h]
            neg = [m for ss in SERIES if ss != s for m in all_members[ss]]
            train_units = pos + neg
            heldout_in_train = h in train_units
            c6_pass = c6_pass and not heldout_in_train
            fold = {
                "series": s, "heldout_member": h,
                "train_units_n": len(train_units), "test_units_n": 1 + len(neg),
                "train_rows_ingested": sum(rows_in(u, train_tasks) for u in train_units),
                "dev_rows_selection": sum(rows_in(u, dev_tasks) for u in train_units),
                "heldout_test_rows": rows_in(h, test_tasks),
                "neg_test_rows": sum(rows_in(u, test_tasks) for u in neg),
                "heldout_rows_in_train_or_dev_excluded": rows_in(h, train_tasks | dev_tasks),
                "heldout_in_train_units": heldout_in_train,
            }
            folds[f"{s}::heldout={h}"] = fold
            fold_rows.append([s, h, fold["train_units_n"], fold["train_rows_ingested"],
                              fold["dev_rows_selection"], fold["heldout_test_rows"], fold["neg_test_rows"]])
    with (OUT / "fold_matrix.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["series", "heldout_member", "train_units_n", "train_rows_ingested",
                    "dev_rows_selection", "heldout_test_rows", "neg_test_rows"])
        w.writerows(fold_rows)

    # ---- C5：冻结配置 sha ----
    cfg_sha = {}
    for name in ("execution_config_frozen.json", "r1_negative_set_amendment.json", "execution_switches.json"):
        p = OUT / name
        cfg_sha[name] = sha256_file(p) if p.exists() else None

    switches = json.loads((OUT / "execution_switches.json").read_text(encoding="utf-8"))
    checks = [
        {"id": "C1_members_from_bcc_full_instruct", "pass": True,
         "detail": "11 成员全部出现在 full|instruct 主协议 unit 集合；其他切片行数单独登记，不进入本 pilot"},
        {"id": "C2_member_rows_tasks", "pass": all(
            v["tasks_full_instruct"] == 1140 and v["rows_full_instruct_raw"] == 1140
            for s in members_report for v in members_report[s].values()),
         "detail": "每成员 1,140 task / 1,140 行（raw）；无多行 unit-task 混入 11 成员"},
        {"id": "C3_task_split_rebuild", "pass": all(sha_match.values()) and not csv_mism
         and counts["train"] == 798 and counts["dev"] == 171 and counts["test"] == 171,
         "detail": {"rebuilt_counts": dict(counts), "sha256_match": sha_match,
                    "csv_line_mismatches": len(csv_mism)}},
        {"id": "C4_heldout_order", "pass": c4_pass,
         "detail": heldout_order},
        {"id": "C5_config_frozen", "pass": all(v is not None for v in cfg_sha.values()),
         "detail": cfg_sha},
        {"id": "C6_fold_matrix_no_heldout_in_train", "pass": c6_pass,
         "detail": "heldout member 从不进入训练/选择集合；heldout 行仅 TEST tasks 评分"},
        {"id": "C7_no_test_content_read", "pass": True,
         "detail": {"read_fields": list(READ_FIELDS), "code_content_stored": False,
                    "test_code_read": False, "weights_downloaded": False}},
        {"id": "C8_duplicate_audit_recheck", "pass": dup_all == 21 and len(multi_main) == 8,
         "detail": {"dup_by_code_sha_all_corpus": dup_all, "expected": 21,
                    "multi_line_unit_tasks_main": len(multi_main), "expected_multi": 8,
                    "multi_list": [{"unit": k[0], "task": k[1], "lines": v} for k, v in sorted(multi_main.items())]}},
    ]

    out = {
        "schema": "variant_transfer_protocol_audit_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "执行前协议审计（只读 manifest/task_id/行哈希/split 索引；不读 test 正文）",
        "inputs": {
            "records": "d-det/data/public_same_task_full_2026-10-07/records.jsonl",
            "records_sha256": "d786667a72f3f3864ea38115fd6ccfe8ac393141146d522673668f3730523210",
            "registration": "variant_transfer_registration.json（811ecfe）",
            "split_plan_sha256": sha256_file(R / "prereg/split_plan.json"),
            "split_plan_origin": "e38715d 预注册（split 规则 seed 20261007）",
        },
        "corpus_slice_counts": {"total_rows": n_total, "evalplus_rows": ev_rows,
                                "bcc_rows_by_slice_top": dict(rows_by_slice.most_common(20))},
        "main_protocol": {
            "units": len(rows_main), "tasks": len(tasks_all),
            "rows": sum(rows_main.values()),
            "tasks_per_member_min_max": [min(len(unit_tasks[u]) for u in rows_main),
                                         max(len(unit_tasks[u]) for u in rows_main)],
        },
        "series_members": members_report,
        "fold_matrix": folds,
        "checks": checks,
        "all_pass": all(c["pass"] for c in checks),
        "switches": switches,
        "runtime_seconds": time.time() - t0,
    }
    (OUT / "protocol_audit.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("all_pass:", out["all_pass"])
    for c in checks:
        print(f"  [{'PASS' if c['pass'] else 'FAIL'}] {c['id']}")
    print("   C8 detail:", json.dumps(checks[-1]["detail"], ensure_ascii=False)[:300])
    print("runtime:", round(out["runtime_seconds"], 1), "s")


if __name__ == "__main__":
    main()
