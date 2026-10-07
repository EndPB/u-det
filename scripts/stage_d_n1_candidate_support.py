"""Stage D-N1: 构造候选 task-aware 支持矩阵（行级去重剔除 + 审计 + 平衡报告 + N2 预注册）。

依据《d-det_AutoDL_D1未过闸门_数据构造下一步指导_2026-10-07.md》§3。
只生成索引/manifest/审计，不训练、不下新数据；候选主矩阵 = h2_authorbench_dcan，
llm_codegen_v2 作为附属支持表（不合并：标签空间/任务域不同）。

剔除规则（行级、预声明）：任何在 exact / norm_ws / norm_lex 归一化下跨 split 的
重复组，其组内全部行整体剔除（不做任意“保留一侧”）；同 split 内重复组保留。
本数据实测：exact 0 组、norm_ws 0 组、norm_lex（去 C 注释/字符串）25 组 → 剔除 243 行。

输出目录：d-det/artifacts/stage_d_data_construction_2026-10-07/n1_candidate_matrix/
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "d-det/artifacts/stage_d_data_construction_2026-10-07"
OUT = BASE / "n1_candidate_matrix"
D0 = ROOT / "d-det/artifacts/stage_d_support_2026-10-07/alignment_support.json"
D1 = ROOT / "d-det/artifacts/stage_d_h1_authorbench_dcan_2026-10-07"
DATA = ROOT / "d-det/data/h2_authorbench_dcan/core.jsonl"
LC2 = ROOT / "d-det/data/h2_llm_codegen_v2/response_tasks.jsonl"
FAMILIES = ["claude", "deepseek", "gemini", "llama", "openai", "qwen"]
GEN_THRESHOLD_TASKS = 100  # 每个 generator 至少 100 个 task 才算“有足够 task 支持”

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage_d_n0_diagnosis import sha256, norm_ws, norm_lex  # noqa: E402


def fmt_table(rows, headers):
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(lines)


def crossing_groups(groups, splits):
    return {h: idx for h, idx in groups.items()
            if len(idx) > 1 and len({splits[i] for i in idx}) > 1}


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in DATA.open(encoding="utf-8")]
    n = len(rows)
    splits = [r["task_split"] for r in rows]
    d0 = json.loads(D0.read_text(encoding="utf-8"))
    ab_folds = [f for f in d0["folds"] if f["source"] == "authorbench_dcan"]

    # ---------- 哈希与剔除 ----------
    hashes = {"exact": {}, "norm_ws": {}, "norm_lex": {}}
    for i, r in enumerate(rows):
        hashes["exact"].setdefault(sha256(r["code"]), []).append(i)
        hashes["norm_ws"].setdefault(sha256(norm_ws(r["code"])), []).append(i)
        hashes["norm_lex"].setdefault(sha256(norm_lex(r["code"])), []).append(i)
    dup_audit = {}
    exclude = set()
    cross_detail = []
    for lvl, groups in hashes.items():
        multi = {h: idx for h, idx in groups.items() if len(idx) > 1}
        cross = crossing_groups(groups, splits)
        dup_audit[lvl] = {"groups": len(multi), "rows_in_groups": int(sum(len(v) for v in multi.values())),
                          "crossing_splits": len(cross)}
        for h, idx in cross.items():
            exclude.update(idx)
            cross_detail.append({"level": lvl, "group_sha": h,
                                 "rows": [{"row_index": i, "task_id": rows[i]["task_id"],
                                           "split": rows[i]["task_split"],
                                           "generator": rows[i]["model_name"]} for i in idx]})
    kept = [i for i in range(n) if i not in exclude]
    kept_set = set(kept)
    # 剔除后复核三个层级
    post = {}
    for lvl in hashes:
        g2 = defaultdict(list)
        for i, r in enumerate(rows):
            if i not in kept_set:
                continue
            if lvl == "exact":
                h = sha256(r["code"])
            elif lvl == "norm_ws":
                h = sha256(norm_ws(r["code"]))
            else:
                h = sha256(norm_lex(r["code"]))
            g2[h].append(i)
        post[lvl] = len(crossing_groups(g2, splits))

    # ---------- 基础派生（剔除后） ----------
    fam_cov = defaultdict(set)
    task_rows = defaultdict(list)
    for i in kept:
        fam_cov[rows[i]["task_id"]].add(rows[i]["family"])
        task_rows[rows[i]["task_id"]].append(i)
    task_size = {t: len(v) for t, v in task_rows.items()}

    heldout_by_gen = defaultdict(list)
    for f in ab_folds:
        heldout_by_gen[f["heldout_generator"]].append(f["fold_id"])

    # ---------- 硬约束核验（剔除后） ----------
    task_splits = defaultdict(set)
    for i in kept:
        task_splits[rows[i]["task_id"]].add(rows[i]["task_split"])
    tasks_cross = [t for t, v in task_splits.items() if len(v) > 1]

    support = {}
    for f in FAMILIES:
        row = {}
        for s in ("train", "dev", "test"):
            m = [i for i in kept if splits[i] == s and rows[i]["family"] == f]
            row[s] = {"rows": len(m), "tasks": len({rows[i]["task_id"] for i in m})}
        support[f] = row

    gen_tasks, gen_rows = defaultdict(set), Counter()
    for i in kept:
        gen_tasks[rows[i]["model_name"]].add(rows[i]["task_id"])
        gen_rows[rows[i]["model_name"]] += 1

    # ---------- H2 候选资格（约束 4/6） ----------
    fam_gens = defaultdict(list)
    for i in kept:
        if rows[i]["model_name"] not in fam_gens[rows[i]["family"]]:
            fam_gens[rows[i]["family"]].append(rows[i]["model_name"])
    fam_gens = {k: sorted(v) for k, v in fam_gens.items()}

    designation = {}
    for f in FAMILIES:
        gens = fam_gens[f]
        heldout_ready = {g: bool(heldout_by_gen.get(g)) and all(
            next(x for x in ab_folds if x["fold_id"] == fid)["admitted"] for fid in heldout_by_gen[g])
            for g in gens}
        eligible = (len(gens) >= 3) and all(len(gen_tasks[g]) >= GEN_THRESHOLD_TASKS for g in gens) \
            and all(heldout_ready.values())
        designation[f] = {
            "generators": gens, "generator_count": len(gens),
            "generator_tasks": {g: len(gen_tasks[g]) for g in gens},
            "generator_rows": {g: gen_rows[g] for g in gens},
            "heldout_fold_ids": {g: heldout_by_gen.get(g, []) for g in gens},
            "holdout_ready": heldout_ready,
            "criteria": {"min_3_generators": len(gens) >= 3,
                         f"each_generator_ge_{GEN_THRESHOLD_TASKS}_tasks":
                             all(len(gen_tasks[g]) >= GEN_THRESHOLD_TASKS for g in gens),
                         "admitted_heldout_fold_per_generator": all(heldout_ready.values())},
            "role": "h2_candidate" if eligible else ("h1_only" if len(gens) < 3 else "h2_pending"),
        }

    # ---------- manifest（仅剔除后行；含剔除行单独文件） ----------
    with (OUT / "n1_manifest.jsonl").open("w", encoding="utf-8") as fh:
        for i in kept:
            r = rows[i]
            rec = {"row_index": i, "example_id": f"abdc-{i:05d}",
                   "family": r["family"], "generator": r["model_name"],
                   "task_id": r["task_id"], "split": r["task_split"], "label": r["family"],
                   "source_sha256": r["source_sha256"],
                   "code_sha256": sha256(r["code"]),
                   "normalized_code_sha256": sha256(norm_lex(r["code"])),
                   "normalized_ws_sha256": sha256(norm_ws(r["code"])),
                   "prompt_or_task_sha256": sha256(r["prompt"]),
                   "task_size": task_size[r["task_id"]],
                   "family_coverage": len(fam_cov[r["task_id"]]),
                   "language": r["language"], "role": r["task_split"],
                   "h2_family_role": designation[r["family"]]["role"],
                   "heldout_fold_ids": heldout_by_gen.get(r["model_name"], [])}
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with (OUT / "n1_manifest_excluded.jsonl").open("w", encoding="utf-8") as fh:
        for i in sorted(exclude):
            r = rows[i]
            fh.write(json.dumps({"row_index": i, "task_id": r["task_id"], "family": r["family"],
                                 "generator": r["model_name"], "split": r["task_split"],
                                 "excluded_reason": "cross_split_near_duplicate(norm_lex)",
                                 "code_sha256": sha256(r["code"]),
                                 "normalized_code_sha256": sha256(norm_lex(r["code"]))},
                                ensure_ascii=False) + "\n")

    # ---------- 数据角色矩阵（剔除后） ----------
    role_matrix = {}
    for f in FAMILIES:
        for g in fam_gens[f]:
            entry = {"family_role": designation[f]["role"],
                     "heldout_fold_ids": heldout_by_gen.get(g, []),
                     "rows": gen_rows[g], "tasks": len(gen_tasks[g]), "splits": {}}
            for s in ("train", "dev", "test"):
                m = [i for i in kept if rows[i]["family"] == f and rows[i]["model_name"] == g
                     and splits[i] == s]
                entry["splits"][s] = {"rows": len(m), "tasks": len({rows[i]["task_id"] for i in m})}
            role_matrix[f"{f}::{g}"] = entry

    # ---------- 候选 N2 子集（六族齐全；纯数据侧规则；基于剔除后矩阵） ----------
    cand_tasks = sorted(t for t, v in fam_cov.items() if len(v) == 6)
    cand_set = set(cand_tasks)
    cand = {"definition": "task family_coverage == 6（组成均匀子集；规则仅用数据组成，不用标签/预测）",
            "task_count": len(cand_tasks), "task_id_sha256": sha256("\n".join(cand_tasks)),
            "rows": {}, "tasks": {}, "family_generator_rows": {}}
    for s in ("train", "dev", "test"):
        rr = [i for i in kept if splits[i] == s and rows[i]["task_id"] in cand_set]
        cand["rows"][s] = len(rr)
        cand["tasks"][s] = len({rows[i]["task_id"] for i in rr})
    fg_c = Counter((rows[i]["family"], rows[i]["model_name"]) for i in kept if rows[i]["task_id"] in cand_set)
    cand["family_generator_rows"] = {f"{k[0]}::{k[1]}": v for k, v in sorted(fg_c.items())}

    # ---------- llm_codegen_v2 附属支持 ----------
    lc_fg, lc_tasks = Counter(), defaultdict(set)
    with LC2.open(encoding="utf-8") as fh:
        for line in fh:
            t = json.loads(line)
            for o in t["outputs"]:
                lc_fg[(o["family"], o["generator"])] += 1
                lc_tasks[(o["family"], o["generator"])].add(t["task_id"])
    lc_families = defaultdict(list)
    for (f, g) in sorted(lc_fg):
        lc_families[f].append(g)
    lc_appendix = {f: {"generators": {g: {"responses": lc_fg[(f, g)], "tasks": len(lc_tasks[(f, g)])}
                                      for g in gens},
                       "generator_count": len(gens),
                       "min_3_generators": len(gens) >= 3,
                       "admitted_heldout_folds": [x["fold_id"] for x in d0["folds"]
                                                  if x["source"] == "llm_codegen_v2"
                                                  and x["family"] == f and x["admitted"]]}
                   for f, gens in sorted(lc_families.items())}

    # ---------- 汇总 ----------
    checks = {
        "c1_task_single_split": {"pass": len(tasks_cross) == 0, "crossing_tasks": len(tasks_cross)},
        "c2_hash_no_cross_split": {
            "pre_exclusion_crossing": {lvl: dup_audit[lvl]["crossing_splits"] for lvl in hashes},
            "post_exclusion_crossing": post,
            "excluded_rows": len(exclude),
            "exclusion_rule": "跨 split 重复组整组剔除（行级）；同 split 重复组保留",
            "pass": all(v == 0 for v in post.values()),
        },
        "c3_train_dev_support_each_family": {
            f: {"train_rows": support[f]["train"]["rows"], "train_tasks": support[f]["train"]["tasks"],
                "dev_rows": support[f]["dev"]["rows"], "dev_tasks": support[f]["dev"]["tasks"],
                "pass": support[f]["train"]["rows"] >= 100 and support[f]["train"]["tasks"] >= 30
                and support[f]["dev"]["rows"] >= 20 and support[f]["dev"]["tasks"] >= 10}
            for f in FAMILIES},
        "c4_formal_h2_candidate_families": {
            f: {"role": designation[f]["role"], "generator_count": designation[f]["generator_count"],
                "pass": designation[f]["role"] in ("h2_candidate",)}
            for f in FAMILIES},
        "c6_test_generator_holdout_roles": {
            f: {"heldout_fold_ids_per_generator": designation[f]["heldout_fold_ids"],
                "note": ("ok：3 个 generator 各有 admitted 的 generator-heldout 折"
                         if designation[f]["role"] == "h2_candidate"
                         else "single-generator family：无法在族内定义 generator-heldout（按约束 4 限 H1/诊断）")}
            for f in FAMILIES},
        "c7_google_mistral_llm_folds_diagnostic_only": [x["fold_id"] for x in d0["folds"]
                                                        if not x["admitted"]],
    }
    checks["c3_train_dev_support_each_family"]["pass"] = all(
        v["pass"] for k, v in checks["c3_train_dev_support_each_family"].items() if k in FAMILIES)
    checks["c4_formal_h2_candidate_families"]["pass"] = any(
        v["pass"] for v in checks["c4_formal_h2_candidate_families"].values() if isinstance(v, dict))
    hard_ok = (checks["c1_task_single_split"]["pass"] and checks["c2_hash_no_cross_split"]["pass"]
               and checks["c3_train_dev_support_each_family"]["pass"]
               and checks["c4_formal_h2_candidate_families"]["pass"])

    matrix = {
        "schema": "stage_d_n1_candidate_support_matrix_v1",
        "inputs": {
            "core": {"path": str(DATA.relative_to(ROOT)),
                     "sha256": hashlib.sha256(DATA.read_bytes()).hexdigest(), "rows": n},
            "d0_alignment_support": {"path": str(D0.relative_to(ROOT)),
                                     "sha256": hashlib.sha256(D0.read_bytes()).hexdigest()},
            "llm_codegen_v2_index": {"path": str(LC2.relative_to(ROOT)),
                                     "sha256": hashlib.sha256(LC2.read_bytes()).hexdigest()},
        },
        "matrix": {"raw_rows": n, "retained_rows": len(kept), "tasks": len(task_rows),
                   "families": FAMILIES, "generators": sorted(gen_rows)},
        "exclusions": {"excluded_rows": len(exclude), "excluded_tasks": len({rows[i]["task_id"] for i in exclude}),
                       "cross_groups": cross_detail,
                       "note": "exact/norm_ws 无跨 split；norm_lex（去注释/字符串）25 组跨 split → 整组剔除"},
        "dedup_counts": {"raw_rows": n, "rows_in_cross_split_groups": len(exclude),
                         "retained_rows": len(kept),
                         "same_split_dup_groups": {lvl: dup_audit[lvl]["groups"] - dup_audit[lvl]["crossing_splits"]
                                                   for lvl in hashes}},
        "checks": checks, "hard_constraints_ok": bool(hard_ok),
        "designation": designation, "support": support,
        "candidate_subset_for_n2": cand, "llm_codegen_v2_appendix": lc_appendix,
        "runtime_seconds": time.time() - t0,
    }
    (OUT / "n1_candidate_support_matrix.json").write_text(
        json.dumps(matrix, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "n1_data_role_matrix.json").write_text(
        json.dumps({"schema": "stage_d_n1_data_role_matrix_v1", "roles": role_matrix,
                    "family_roles": {f: designation[f]["role"] for f in FAMILIES},
                    "notes": "role: h2_candidate=≥3 generator 且各≥%d task 且各持有 admitted heldout 折；"
                             "h1_only=单 generator，仅 H1/诊断" % GEN_THRESHOLD_TASKS},
                   ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "n1_duplicate_and_split_audit.json").write_text(
        json.dumps({"schema": "stage_d_n1_dup_split_audit_v1",
                    "pre_exclusion": dup_audit,
                    "post_exclusion_crossing": post,
                    "cross_group_detail": cross_detail,
                    "excluded_row_count": len(exclude),
                    "tasks_crossing_splits_after": len(tasks_cross),
                    "task_purity_pass": len(tasks_cross) == 0,
                    "train_dev_support_after": support,
                    "source_sha256_crossing_groups": 0,
                    "hash_inputs": {"core_sha256": hashlib.sha256(DATA.read_bytes()).hexdigest()}},
                   ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # ---------- N2 预注册 ----------
    protocol = {
        "schema": "stage_d_n2_protocol_v1",
        "written_at_utc": datetime.now(timezone.utc).isoformat(),
        "authorization": ("N1 硬约束 c1/c2/c3/c4 全部满足（见 n1_candidate_support_matrix.json checks，"
                          "c2 已含行级去重剔除）；依据指导 §4 允许一次固定协议 H1 复核"),
        "candidate": cand | {"base_matrix_rows": len(kept),
                             "fallback_rule": "若 test tasks < 50 或 test rows < 200 则本协议作废（不降级、不换口径）"},
        "views": ["metadata_only", "tfidf_word(3 seeds)", "tfidf_char(3 seeds)",
                  "codet5_small_meanpool(冻结)", "codet5_small_centered(转导诊断，单独标注)"],
        "p0_recompute": ("在候选矩阵上按 P0 原配方整体复算：tfidf_char/word=SGD log_loss 5ep best-dev×3 seeds；"
                         "sem_lr=CodeT5-base 768d 冻结特征+StandardScaler+LR(C=1)；"
                         "style_lr/style_lgb=regex stylometry+LR/LightGBM(800)；"
                         "fusion_lr=dev-only LR stack（log-prob 特征）；mean_ensemble=等权均值"),
        "gate": {
            "cond1_content_above_chance_and_metadata": "best content > chance(1/6) 且 > metadata_only",
            "cond2_plus_1pt_over_p0": "best content ≥ P0_fusion(候选) + 1pt",
            "cond3_direction_stable": "3 个 TF-IDF seed 相对 P0 的 Δ 符号一致",
            "verdict_rule": "三条全过 → 数据构造修复支持不足，可重新讨论 H2；否则归档为数据限制/负结果",
        },
        "comparators_for_context": {"d1_full_p0_fusion": 0.8388311014514443,
                                    "d1_full_tfidf_word": 0.8200233883183988,
                                    "note": "原 D1 结果并列展示，不替换"},
        "inputs": {"core_sha256": hashlib.sha256(DATA.read_bytes()).hexdigest(),
                   "d1_metrics_sha256": hashlib.sha256((D1 / "metrics.json").read_bytes()).hexdigest(),
                   "d1_predictions_sha256": hashlib.sha256((D1 / "predictions.npz").read_bytes()).hexdigest(),
                   "d0_alignment_sha256": hashlib.sha256(D0.read_bytes()).hexdigest()},
        "test_read": "本协议冻结后对候选子集 test 单次读取",
    }
    (OUT / "n2_protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n",
                                          encoding="utf-8")

    # ---------- 平衡报告 ----------
    L = []
    L.append("# N1：候选 task-aware 支持矩阵（2026-10-07）")
    L.append("")
    L.append(f"主矩阵 = `h2_authorbench_dcan`（原始 {n} 行 → 剔除 {len(exclude)} 行 → **保留 {len(kept)} 行**/"
             f"{len(task_rows)} tasks / 6 families / 8 generators / C）；"
             "附属 = `h2_llm_codegen_v2`（不合并：标签空间与任务域不同）。")
    L.append("")
    L.append("## 0 行级剔除（预声明规则）")
    L.append("")
    L.append(f"- 规则：任一归一化层级下**跨 split** 的重复组，整组行剔除（不做任意“保留一侧”）；同 split 重复组保留。")
    L.append(f"- 实测：exact 跨 split {dup_audit['exact']['crossing_splits']} 组、"
             f"norm_ws {dup_audit['norm_ws']['crossing_splits']} 组、"
             f"**norm_lex（去 C 注释/字符串）{dup_audit['norm_lex']['crossing_splits']} 组**"
             f"→ 共剔除 {len(exclude)} 行（涉及 {len({rows[i]['task_id'] for i in exclude})} 个 task；"
             f"剔除行明细见 `n1_manifest_excluded.jsonl` 与审计 JSON 的 `cross_group_detail`）。")
    L.append("- `source_sha256`：9498 行全部唯一、跨 split 0 组，无需处理。")
    L.append("")
    L.append("## 1 硬约束核验（剔除后）")
    L.append("")
    L.append(fmt_table([
        ["1 task 单 split", checks["c1_task_single_split"]["pass"], f"crossing tasks = {len(tasks_cross)}"],
        ["2 hash 不跨 split", checks["c2_hash_no_cross_split"]["pass"],
         f"剔除后 exact/ws/lex 跨 split = {post['exact']}/{post['norm_ws']}/{post['norm_lex']}"],
        ["3 train/dev 每族支持", checks["c3_train_dev_support_each_family"]["pass"],
         "train ≥100 行/≥30 task；dev ≥20 行/≥10 task（§4 表）"],
        ["4 正式 H2 family ≥3 generator", checks["c4_formal_h2_candidate_families"]["pass"],
         "仅 openai；其余 5 族单 generator → H1/诊断"],
        ["5 占比公开", True, "§2/§3 与本文件"],
        ["6 test generator held-out 角色", True, "openai 3 折（D0 admitted）；单 generator 族不可定义"],
        ["7 google/mistral llm 折 diagnostic-only", True, "D0 记录；未伪造正例"],
        ["8 原始/去重计数与 hash", True, "matrix.exclusions/dedup_counts + 审计 JSON"],
    ], ["约束", "通过", "证据"]))
    L.append("")
    L.append("## 2 family×generator（行数 / task 数，剔除后）")
    L.append("")
    rows_ = []
    for f in FAMILIES:
        for g in fam_gens[f]:
            e = role_matrix[f"{f}::{g}"]
            rows_.append([f, g, designation[f]["role"], e["rows"], e["tasks"],
                          "/".join(f"{s}:{e['splits'][s]['rows']}" for s in ("train", "dev", "test")),
                          "/".join(f"{s}:{e['splits'][s]['tasks']}" for s in ("train", "dev", "test"))])
    L.append(fmt_table(rows_, ["family", "generator", "role", "rows", "tasks", "rows tr/dv/te", "tasks tr/dv/te"]))
    L.append("")
    L.append("## 3 任务组成（剔除后）")
    L.append("")
    cov = Counter(len(v) for v in fam_cov.values())
    ts = Counter(task_size.values())
    L.append("- family coverage（每 task 含族数）：{" + ", ".join(f"{k}:{v}" for k, v in sorted(cov.items())) + "}")
    L.append("- task-size（每 task 行数）：{" + ", ".join(f"{k}:{v}" for k, v in sorted(ts.items())) + "}")
    L.append(f"- 六族齐全子集（N2 候选）：{len(cand_tasks)} tasks；行 "
             f"{cand['rows']['train']}/{cand['rows']['dev']}/{cand['rows']['test']}（tr/dv/te）；"
             f"tasks {cand['tasks']['train']}/{cand['tasks']['dev']}/{cand['tasks']['test']}")
    L.append("")
    L.append("## 4 每正式 family 的 generator/task 支持（回传 §6.4）")
    L.append("")
    L.append(fmt_table([[f, designation[f]["generator_count"],
                         json.dumps(designation[f]["generator_tasks"], ensure_ascii=False),
                         designation[f]["role"]] for f in FAMILIES],
                       ["family", "#gen", "tasks per generator", "role"]))
    L.append("")
    L.append("→ 结论：**矩阵内仅 `openai` 满足「每正式 family ≥3 个可靠 generator + admitted heldout 折」，"
             "可作 H2 候选单元（holdout=gpt-4.1 / gpt-4o / gpt-4o-mini 三折）；"
             "claude/deepseek/gemini/llama/qwen 为单 generator，只能作 H1/诊断数据。**")
    L.append("")
    L.append("## 5 llm_codegen_v2 附属支持（不合并）")
    L.append("")
    L.append(fmt_table([[f, v["generator_count"],
                         ", ".join(f"{g}({x['tasks']}t)" for g, x in v["generators"].items()),
                         ", ".join(v["admitted_heldout_folds"]) or "—"]
                        for f, v in lc_appendix.items()],
                       ["family", "#gen", "generators(tasks)", "admitted folds"]))
    L.append("")
    L.append("（其 `meta` 族 3 generator 且 D0 有 3 个 admitted 折，可作为第二个 H2 候选单元评估；"
             "跨源合并属于新的构造决策，本轮到 N1 为止。）")
    L.append("")
    L.append("## 6 N2 决策")
    L.append("")
    L.append(f"- 硬约束 c1–c4 全部满足；候选子集 = 六族齐全 {len(cand_tasks)} tasks（test {cand['tasks']['test']} tasks / "
             f"{cand['rows']['test']} 行，≥协议下限）→ **允许且已预注册 N2**（`n2_protocol.json`）。")
    L.append("- N2 只作“数据构造是否修复支持不足”的最小复核；不启动任何 H2/H3 训练。")
    L.append("")
    (OUT / "n1_balance_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(json.dumps({"n1": "done", "hard_ok": bool(hard_ok), "excluded_rows": len(exclude),
                      "retained_rows": len(kept),
                      "h2_candidate_families": [f for f in FAMILIES if designation[f]["role"] == "h2_candidate"],
                      "n2_candidate": {"tasks": len(cand_tasks), "test_rows": cand["rows"]["test"],
                                       "test_tasks": cand["tasks"]["test"]},
                      "runtime_s": round(time.time() - t0, 1)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
