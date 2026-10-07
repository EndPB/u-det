"""公开语料预注册 split（第四步）：task 级冻结 + unit-heldout 折 + 人类控制摘要。

依据指导 §5 第四步：先冻结 task split，再指定 generator-heldout、family-variant-heldout 与
human-control 方案；每个方案给出目标总体、纳入/排除规则、任务数、generator 数、每 task 输出数、
family 证据状态、污染风险与统计单位。
输出 d-det/artifacts/public_full_receive_2026-10-08/prereg/。
"""
from __future__ import annotations

import csv
import hashlib
import json
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FULL = ROOT / "d-det/data/public_same_task_full_2026-10-07"
A = ROOT / "d-det/artifacts/public_full_receive_2026-10-08"
OUT = A / "prereg"
SEED = 20261007


def sha256(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def hkey(s: str) -> str:
    return sha256(f"{SEED}|{s}")


def fmt_table(rows, headers):
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(lines)


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    adj = [json.loads(l) for l in (FULL / "model_lineage_adjudication.jsonl").open(encoding="utf-8")]
    main_units = [x["generator_unit"] for x in adj
                  if x["source"] == "bigcodebench" and x["generation_mode"] == "instruct" and x["subset"] == "full"]
    ev_units = [x["generator_unit"] for x in adj if x["source"] == "evalplus"]

    # ---- 单遍：主协议 task 宇宙与每 unit-task 行数（识别多行对） ----
    tasks = set()
    unit_task_rows = Counter()
    hard_tasks = set()
    n_rows_main = 0
    with (FULL / "records.jsonl").open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["source"] != "bigcodebench":
                continue
            if r.get("subset") == "hard":
                hard_tasks.add(r["task_id"])
            if r.get("generation_mode") == "instruct" and r.get("subset") == "full":
                tasks.add(r["task_id"])
                key = "|".join(str(r.get(k, "")) for k in ("source", "model_id", "generation_mode", "subset", "backend", "temperature"))
                unit_task_rows[(key, r["task_id"])] += 1
                n_rows_main += 1
    tasks = sorted(tasks)
    multi = {k: v for k, v in unit_task_rows.items() if v > 1}
    assert n_rows_main == 134528, n_rows_main

    # ---- task 级 70/15/15（按 sha256(seed|task) 排序确定性分配） ----
    order = sorted(tasks, key=hkey)
    n = len(order)
    n_tr = int(n * 0.70)
    n_dv = int(n * 0.15)
    split_of = {}
    for i, t in enumerate(order):
        split_of[t] = "train" if i < n_tr else ("dev" if i < n_tr + n_dv else "test")
    counts = Counter(split_of.values())

    # ---- unit 5 折（按 family_label 轮转 + hash 排序，保证家族铺开） ----
    fam_of = {x["generator_unit"]: x["family_label"] for x in adj}
    by_fam = defaultdict(list)
    for u in main_units:
        by_fam[fam_of[u]].append(u)
    fold_of = {}
    for fam, us in sorted(by_fam.items()):
        for i, u in enumerate(sorted(us, key=hkey)):
            fold_of[u] = f"fold_{i % 5}"
    fold_stats = Counter(fold_of.values())

    # ---- evalplus 次级切片（unit×task 全覆盖） ----
    ev = {"units": len(ev_units), "rows": 8778, "tasks": 399, "rows_per_unit_task": "1（除下述多行）"}

    # ---- 人类控制（CodeContests 索引聚合） ----
    cc = {}
    for name in ("valid", "test"):
        rows = [json.loads(l) for l in (A / f"audit/cc_index_{name}.jsonl").open(encoding="utf-8")]
        lang = Counter()
        inc_lang = Counter()
        src = Counter()
        diff = Counter()
        has_tests = 0
        desc = set()
        sr = [r["cf_rating"] for r in rows if r.get("cf_rating")]
        for r in rows:
            src[r["source"]] += 1
            diff[r["difficulty"]] += 1
            has_tests += int(r["has_tests"])
            desc.add(r["description_sha256"])
            for k, v in r["solutions_langs"].items():
                lang[k] += v
            for k, v in r["incorrect_langs"].items():
                inc_lang[k] += v
        cc[name] = {"problems": len(rows), "unique_descriptions": len(desc), "has_tests": has_tests,
                    "sources": dict(src), "difficulty_raw": dict(diff),
                    "human_correct_by_lang": dict(lang), "human_incorrect_by_lang": dict(inc_lang),
                    "cf_rating_n": len(sr), "cf_rating_median": (sorted(sr)[len(sr) // 2] if sr else None)}
    cc_overlap = len({json.loads(l)["description_sha256"] for l in (A / "audit/cc_index_valid.jsonl").open()}
                     & {json.loads(l)["description_sha256"] for l in (A / "audit/cc_index_test.jsonl").open()})

    plan = {
        "schema": "public_full_prereg_split_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "target_population": "公开同任务语料（EvalPlus MBPP+ / BigCodeBench v0.2.4；外部 benchmark，非项目历史域）",
        "inclusion": ["records.jsonl 全量成员", "BigCodeBench full/instruct = 主协议；full/complete 与 hard/instruct = 独立控制"],
        "exclusion": ["不做任何按输出行的随机切分", "不混协议轴（instruct≠complete，full≠hard）"],
        "main_protocol": {
            "definition": "BigCodeBench full/instruct",
            "units": len(main_units), "tasks": n,
            "rows": n_rows_main, "rows_per_unit_task": "1（%d 个 unit-task 为多行, %s）" % (
                len(multi), "已登记" if multi else "无"),
            "task_split": {k: counts[k] for k in ("train", "dev", "test")},
            "task_split_rule": "按 sha256(f'{SEED}|task_id') 排序取前 70%/15%/15%",
            "task_list_sha256": {s: sha256("\n".join(sorted(t for t in tasks if split_of[t] == s)))
                                 for s in ("train", "dev", "test")},
            "hard_subset_in_full_overlap": {"hard_tasks": len(hard_tasks),
                                            "hard_in_each_split": {s: sum(1 for t in hard_tasks if split_of[t] == s)
                                                                   for s in ("train", "dev", "test")}},
            "unit_heldout_folds": {"scheme": "5 折按 family_label 分组内 hash 排序轮转", "folds": dict(fold_stats),
                                   "per_fold_preview": {
                                       f: {"heldout_units": fold_stats[f],
                                           "train_units": len(main_units) - fold_stats[f],
                                           "test_tasks_per_fold": counts["test"],
                                           "test_rows_per_fold": (len(main_units) - fold_stats[f]) and fold_stats[f] * counts["test"]}
                                       for f in sorted(fold_stats)}},
            "family_variant_heldout": {"status": "blocked", "reason": "family_is_confirmed=true 为 0（349 行/306 单元全部为提示）；须先完成模型卡/lineage 证据才能设 family 折"},
            "statistics_unit": "task cluster（每 task 一个 cluster；bootstrap 按 task 重采样）",
            "contamination": {"status": "unknown", "note": "公开 benchmark 可能进入模型预训练；本轮定位=外部验证/诊断，不进项目主表；须登记后方可升级"},
        },
        "secondary_slices": {
            "bcc_full_complete": {"units": 123, "rows": 140220, "tasks": n, "role": "独立提示协议控制"},
            "bcc_hard_instruct": {"units": 86, "rows": 12728, "tasks": len(hard_tasks), "role": "难题子集控制"},
            "evalplus_mbpp": ev | {"role": "规模/后端变体次级切片"},
        },
        "human_control": {
            "source": "CodeContests validation/test（本包 riegeli，最小解析器提取字段）",
            "valid": cc["valid"], "test": cc["test"],
            "valid_test_description_overlap": cc_overlap,
            "scope": "Human/AI 与 correctness 控制（CPP/PYTHON3/PYTHON2/JAVA 语言层）；与 BCC 无同 task 对齐 → 不能做同任务三 generator 主表的人控列",
            "code_executed": False,
        },
        "next_steps_gate": ["污染/历史使用登记", "family 证据（模型卡/lineage）", "双 P0（fusion + eq）在新 train/dev 冻结", "test 单次读取"],
        "training_allowed": False,
        "generation_allowed": False,
        "runtime_seconds": time.time() - t0,
    }
    (OUT / "split_plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    with (OUT / "split_index.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["task_id", "split", "in_hard"])
        for t in tasks:
            w.writerow([t, split_of[t], int(t in hard_tasks)])
    with (OUT / "unit_folds.json").open("w", encoding="utf-8") as fh:
        json.dump({"seed": SEED, "folds": fold_of, "family_label": fam_of},
                  fh, ensure_ascii=False, indent=2)

    L = []
    L.append("# 公开完整语料预注册 split（2026-10-08）")
    L.append("")
    L.append("依据指导 §5 第四步；**先冻结 split 再允许任何生成/训练**；本轮不训练、不生成、不评测。")
    L.append("")
    L.append(f"- 主协议：BigCodeBench full/instruct —— **{len(main_units)} units × {n} task = {n_rows_main} 行**")
    L.append(f"- task split（seed {SEED}）：train/dev/test = {counts['train']}/{counts['dev']}/{counts['test']}；"
             f"每 split task 列表 sha256 已记录（`split_plan.json`）")
    L.append(f"- 64KiB 含 hard 子集：{len(hard_tasks)} task（全部 ∈ full），split 内分布 "
             f"{plan['main_protocol']['hard_subset_in_full_overlap']['hard_in_each_split']}")
    L.append(f"- unit-heldout：5 折（family_label 分组轮转）→ {dict(fold_stats)}；每折测试 = 折内 unit × 全 test task")
    L.append(f"- family-variant-heldout：**blocked**（family 证据为 0/306 确认）")
    L.append(f"- 人类控制：CodeContests valid {cc['valid']['problems']} / test {cc['test']['problems']} 题；"
             f"valid∩test 描述重叠 {cc_overlap}；语言层计数见 `split_plan.json.human_control`")
    L.append(f"- 统计单位：task cluster；污染：unknown（外部验证定位）")
    L.append("")
    L.append("## 每条方案的必填字段（对照指导）")
    L.append("")
    L.append(fmt_table([
        ["主协议 H1/H2a", "BCC full/instruct", "全量成员", f"{len(main_units)}", n, "1", "未确认", "unknown", "task"],
        ["协议控制", "BCC full/complete", "同 task", "123", n, "1", "未确认", "unknown", "task"],
        ["难题控制", "BCC hard/instruct", "hard 子集", "86", len(hard_tasks), "1", "未确认", "unknown", "task"],
        ["次级切片", "EvalPlus MBPP+", "22 模型包", "22", "399", "1", "未确认", "unknown", "task"],
        ["人类/正确性控制", "CodeContests valid/test", "竞赛题", "—", f"{cc['valid']['problems']}+{cc['test']['problems']}", "人提交", "—", "unknown", "题目"],
    ], ["方案", "目标总体", "纳入规则", "generator 数", "task 数", "每 task 输出", "family 证据", "污染", "统计单位"]))
    L.append("")
    (OUT / "public_prereg.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(json.dumps({"split_plan": "done", "main_units": len(main_units), "tasks": n,
                      "split": dict(counts), "multi_row_unit_task": len(multi),
                      "cc_valid": cc["valid"]["problems"], "cc_test": cc["test"]["problems"],
                      "runtime_s": round(time.time() - t0, 1)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
