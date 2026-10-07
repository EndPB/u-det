"""Stage E2: 独立数据预注册与干跑（只读 metadata；不生成、不训练、不读旧 test 原文做筛选）。

依据《d-det_AutoDL_E0E1完成后_E2独立数据预注册与干跑指导_2026-10-07.md》。
输出：d-det/artifacts/stage_e2_independent_pilot_2026-10-07/{prereg,task_dry_run,audit,logs}
  e2_preregistration.json（status=blocked）/ candidate_tasks.jsonl / excluded_tasks.jsonl /
  collision_groups.json / split_plan.json / license_and_provenance.json /
  dry_run_report.md / pilot_design.md / feasibility_decision.md
（E2a != ready ⇒ 按 §3 门槛不生成 generation_plan.json，原因见 feasibility_decision.md）
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "d-det/artifacts/stage_e2_independent_pilot_2026-10-07"
D1 = ROOT / "d-det/artifacts/stage_d_h1_authorbench_dcan_2026-10-07"
N1M = ROOT / "d-det/artifacts/stage_d_data_construction_2026-10-07/n1_candidate_matrix"
E0 = ROOT / "d-det/artifacts/stage_e_evidence_feasibility_2026-10-07/e0"
LC2 = ROOT / "d-det/data/h2_llm_codegen_v2"
FAMILIES_META = ["llama2", "llama3", "codellama"]

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage_d_n0_diagnosis import sha256, norm_ws, norm_lex  # noqa: E402


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def fmt_table(rows, headers):
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(lines)


def main():
    t0 = time.time()
    for d in ("prereg", "task_dry_run", "audit", "logs"):
        (BASE / d).mkdir(parents=True, exist_ok=True)
    logs = []

    def log(s):
        print(s, flush=True)
        logs.append(s)

    ledger = json.loads((E0 / "test_exposure_ledger.json").read_text(encoding="utf-8"))
    e0_ab_test_tasks = None
    z1 = np.load(D1 / "predictions.npz")
    e0_ab_test_tasks = sorted({str(x) for x in z1["task_id_te"].tolist()})
    ab_prompt_hashes = set()
    with (N1M / "n1_manifest.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            ab_prompt_hashes.add(json.loads(line)["prompt_or_task_sha256"])
    log(f"E0 ledger 已载入（AB test task {len(e0_ab_test_tasks)} 个；AB prompt hash {len(ab_prompt_hashes)} 个）")

    # ---------- 读取 LCv2 任务元数据（候选命名空间；不写正文，只用标量与 prompt 文本做 hash） ----------
    tasks = {}
    with (LC2 / "response_tasks.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            t = json.loads(line)
            prompts = sorted({o["prompt"] for o in t["outputs"]})
            gen_code = {o["generator"]: bool(o["code"]) for o in t["outputs"]}
            tasks[t["task_id"]] = {
                "task_id": t["task_id"], "split": t["task_split"], "group_id": t["group_id"],
                "cwe_id": t["cwe_id"], "language": t["language"],
                "prompt_sha256s": t["prompt_sha256s"], "prompts": prompts,
                "n_prompts": len(prompts), "gen_code": gen_code,
            }
    log(f"LCv2 任务 {len(tasks)} 个（train/dev/test = "
        f"{sum(1 for x in tasks.values() if x['split'] == 'train')}/"
        f"{sum(1 for x in tasks.values() if x['split'] == 'dev')}/"
        f"{sum(1 for x in tasks.values() if x['split'] == 'test')}）")

    # ---------- 候选（保守：整体排除原 test split）与排除 ----------
    eligible = [t for t in tasks.values() if t["split"] != "test"]
    excluded_lc = [t for t in tasks.values() if t["split"] == "test"]
    lc_test_hashes = set()
    for t in excluded_lc:
        lc_test_hashes.update(t["prompt_sha256s"])
    log(f"候选 {len(eligible)}（train+dev）；保守排除原 test {len(excluded_lc)}（含 pair 轮已读 22/24）")

    def task_hashes(t):
        joined = "|".join(sorted(t["prompt_sha256s"]))
        ex = sha256(joined)
        ws = sha256("|".join(sorted(sha256(norm_ws(p)) for p in t["prompts"])))
        lx = sha256("|".join(sorted(sha256(norm_lex(p)) for p in t["prompts"])))
        ts_src = sha256(f"h2_llm_codegen_v2|{t['group_id']}|{ex}|{t['language']}")
        return ex, ws, lx, ts_src

    cand_rows = []
    hit_excluded = []
    for t in sorted(eligible, key=lambda x: x["task_id"]):
        ex, ws, lx, ts_src = task_hashes(t)
        hit = sorted((set(t["prompt_sha256s"]) & ab_prompt_hashes)) + sorted((set(t["prompt_sha256s"]) & lc_test_hashes))
        if hit:
            hit_excluded.append({"task_id_candidate": t["task_id"], "source_dataset": "h2_llm_codegen_v2",
                                 "provenance_group": t["group_id"],
                                 "reason": "prompt_hash_hits_old_sets（AB prompt / LCv2 原 test）"})
            continue
        cand_rows.append({
            "task_source_hash": ts_src, "task_id_candidate": t["task_id"],
            "provenance_group": t["group_id"], "language": t["language"],
            "statement_hash": ex, "split_group": t["split"],
            "existing_overlap_status": "none_vs_E0_ledger",
            "pair_round_usage": "pair_train" if t["split"] == "train" else "pair_dev",
            "exact_hash": ex, "norm_ws_hash": ws, "norm_lex_hash": lx,
            "source_dataset": "h2_llm_codegen_v2",
            "license_status": "missing", "generation_status": "existing_outputs_present(reuse_req_authorization)",
            "gen_nonempty": {g: t["gen_code"][g] for g in sorted(t["gen_code"])},
        })
    log(f"交叉命中剔除 {len(hit_excluded)}（预期 0）")
    with (BASE / "task_dry_run/candidate_tasks.jsonl").open("w", encoding="utf-8") as fh:
        for r in cand_rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    with (BASE / "task_dry_run/excluded_tasks.jsonl").open("w", encoding="utf-8") as fh:
        for t in sorted(excluded_lc, key=lambda x: x["task_id"]):
            fh.write(json.dumps({
                "task_id_candidate": t["task_id"], "source_dataset": "h2_llm_codegen_v2",
                "provenance_group": t["group_id"], "split_group": "test",
                "reason": "original_test_split_conservative_exclusion（24 个中 22 个已在 pair 轮读取 test）",
            }, ensure_ascii=False) + "\n")
        for tid in e0_ab_test_tasks:
            fh.write(json.dumps({
                "task_id_candidate": tid, "source_dataset": "h2_authorbench_dcan",
                "reason": "E0 exposure ledger（AB test，P0/D1/N2 已读）",
            }, ensure_ascii=False) + "\n")
        for r in hit_excluded:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    log(f"excluded_tasks.jsonl：LCv2 {len(excluded_lc)} + AB test {len(e0_ab_test_tasks)} + 交叉命中 {len(hit_excluded)}")

    # ---------- 碰撞审计（三层；候选内 + 对 AB prompt / LCv2 test 的交叉） ----------
    groups = {"exact": defaultdict(list), "norm_ws": defaultdict(list), "norm_lex": defaultdict(list)}
    for r in cand_rows:
        groups["exact"][r["exact_hash"]].append(r["task_id_candidate"])
        groups["norm_ws"][r["norm_ws_hash"]].append(r["task_id_candidate"])
        groups["norm_lex"][r["norm_lex_hash"]].append(r["task_id_candidate"])
    collisions = {"within_candidates": {}, "vs_ab_prompt_exact": 0, "vs_lc_test_exact": 0,
                  "cross_split_groups": {}, "excluded_group_count": 0, "excluded_task_count": 0,
                  "group_detail_truncated": []}
    for lvl, g in groups.items():
        multi = {h: v for h, v in g.items() if len(v) > 1}
        collisions["within_candidates"][lvl] = {"multi_groups": len(multi),
                                                "rows_in_groups": int(sum(len(v) for v in multi.values()))}
    split_of = {r["task_id_candidate"]: r["split_group"] for r in cand_rows}
    for lvl, g in groups.items():
        cross = {h: v for h, v in g.items() if len(v) > 1 and len({split_of[x] for x in v}) > 1}
        collisions["cross_split_groups"][lvl] = len(cross)
        collisions["excluded_group_count"] += len(cross)
        collisions["excluded_task_count"] += int(sum(len(v) for v in cross.values()))
        for h, v in list(cross.items())[:10]:
            collisions["group_detail_truncated"].append({"level": lvl, "hash": h, "tasks": v})
    cand_prompt_shas = set()
    kept_ids = {r["task_id_candidate"] for r in cand_rows}
    for t in eligible:
        if t["task_id"] in kept_ids:
            cand_prompt_shas.update(t["prompt_sha256s"])
    collisions["vs_ab_prompt_exact"] = len(cand_prompt_shas & ab_prompt_hashes)
    collisions["vs_lc_test_exact"] = len(cand_prompt_shas & lc_test_hashes)
    gid_of = {r["task_id_candidate"]: r["provenance_group"] for r in cand_rows}
    multi_lex = {h: v for h, v in groups["norm_lex"].items() if len(v) > 1}
    collisions["multi_groups_intra_provenance_group_only"] = bool(
        all(len({gid_of[x] for x in v}) == 1 for v in multi_lex.values()))
    collisions["multi_group_samples"] = [
        {"level": "norm_lex", "hash": h, "tasks": v,
         "groups": sorted({gid_of[x] for x in v})} for h, v in list(multi_lex.items())[:5]]
    (BASE / "audit/collision_groups.json").write_text(
        json.dumps({"schema": "stage_e2_collision_groups_v1",
                    "levels": {"exact": "sha256(prompt_sha256s join；task 级)",
                               "norm_ws": "sha256(按任务排序的 norm_ws(prompt) 哈希 join)",
                               "norm_lex": f"同上；norm_lex 实现=stage_d_n0_diagnosis.norm_lex（去 C 注释/字符串后折叠空白）"},
                    "norm_lex_impl_sha256": sha256(Path(ROOT / "scripts/stage_d_n0_diagnosis.py").read_text(encoding="utf-8")),
                    "collisions": collisions,
                    "exclusion_rule": "跨 split 组整组 excluded；命中 E0 账本/旧集合的组 excluded（本干跑：跨 split 组见上）",
                    "candidate_count": len(cand_rows)},
                   ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    log(f"碰撞审计：候选内 multi 组 {collisions['within_candidates']}；跨 split {collisions['cross_split_groups']}")

    # ---------- split_plan（group 级、种子 20261007、目标 70/15/15） ----------
    import random
    by_group = defaultdict(list)
    for r in cand_rows:
        by_group[r["provenance_group"]].append(r["task_id_candidate"])
    gids = sorted(by_group)
    rng = random.Random(20261007)
    rng.shuffle(gids)
    target = {"train": int(len(cand_rows) * 0.70), "dev": int(len(cand_rows) * 0.15),
              "test": len(cand_rows) - int(len(cand_rows) * 0.70) - int(len(cand_rows) * 0.15)}
    assign, counts = {}, Counter()
    for gid in gids:
        n = len(by_group[gid])
        order = sorted(target, key=lambda s: (counts[s] + n) / max(target[s], 1))
        pick = order[0]
        assign[gid] = pick
        counts[pick] += n
    split_tasks = defaultdict(list)
    for gid, s in assign.items():
        split_tasks[s].extend(by_group[gid])
    # H2 支持（meta trio：heldout 折的 train/dev 正对与 test 覆盖）
    meta_folds = []
    tinfo = {r["task_id_candidate"]: r for r in cand_rows}
    for held in FAMILIES_META:
        others = [g for g in FAMILIES_META if g != held]
        fold = {"heldout_generator": held,
                "train_tasks_with_ge2_seen_gen_nonempty": sum(
                    1 for t in split_tasks["train"]
                    if sum(1 for g in others if tinfo[t]["gen_nonempty"][g]) >= 2),
                "dev_tasks_with_ge2_seen_gen_nonempty": sum(
                    1 for t in split_tasks["dev"]
                    if sum(1 for g in others if tinfo[t]["gen_nonempty"][g]) >= 2),
                "test_tasks_heldout_nonempty": sum(
                    1 for t in split_tasks["test"] if tinfo[t]["gen_nonempty"][held])}
        meta_folds.append(fold)
    split_plan = {
        "schema": "stage_e2_split_plan_v1",
        "status": "plan_only_not_executed",
        "rule": "CWE group（provenance_group）级随机（seed 20261007）分配；同组 task 不拆分；目标比例 70/15/15",
        "counts": {s: len(split_tasks[s]) for s in ("train", "dev", "test")},
        "group_counts": dict(Counter(assign.values())),
        "task_lists_sha256": {s: sha256("\n".join(sorted(split_tasks[s]))) for s in ("train", "dev", "test")},
        "meta_unit_h2_support_preview": meta_folds,
        "note": "该计划仅对新任务集有效；若复用 LCv2 现有输出（Path A），需先用授权口径确认 task 独立性",
    }
    (BASE / "prereg/split_plan.json").write_text(json.dumps(split_plan, ensure_ascii=False, indent=2) + "\n",
                                                 encoding="utf-8")
    log(f"split_plan：{split_plan['counts']}；meta 折支持 {[(f['heldout_generator'], f['train_tasks_with_ge2_seen_gen_nonempty']) for f in meta_folds]}")

    # ---------- license / provenance ----------
    lic = {
        "schema": "stage_e2_license_and_provenance_v1",
        "source_unit_candidate": "h2_llm_codegen_v2（Meta 单元）/ h2_authorbench_dcan（OpenAI 单元，Path C）",
        "h2_llm_codegen_v2": {
            "provenance_files": [x["path"] for x in json.loads((LC2 / "provenance.json").read_text(encoding="utf-8"))["csv_files"][:3]],
            "csv_files_total": len(json.loads((LC2 / "provenance.json").read_text(encoding="utf-8"))["csv_files"]),
            "license": "missing（README/元数据未提供 license 或使用条款）",
            "family_mapping_basis": "release folder 推导（README 明示）；无 base/SFT/DPO 谱系",
            "limitations": ["CWE group 是相关任务集合，非正式程序语义等价",
                            "词法视图为诊断，非编译/行为等价变换",
                            "v1/v2 分数不可作为方法增量对比"],
        },
        "h2_authorbench_dcan": {
            "source_archive": "LLM-AuthorBench.json.zip（sha256 记于 summary.json）",
            "license": "missing", "namespace_exposure": "已用于 P0/D1/N2（不可作为新 task 来源）",
        },
        "third_generator_availability": {
            "local_weights_found": False,
            "evidence": "服务器仅存 d-det/models/codet5-small 与 d-det/checkpoints/codet5-base；HF 缓存 1.2MB；无 API 授权记录",
            "verdict": "google/mistral 第三 generator 版本/许可/调用均未落实（blocker）",
        },
    }
    (BASE / "audit/license_and_provenance.json").write_text(json.dumps(lic, ensure_ascii=False, indent=2) + "\n",
                                                            encoding="utf-8")
    log("license_and_provenance.json 已写出")

    # ---------- E2a 预注册（blocked） ----------
    prereg = {
        "schema": "stage_e2_preregistration_v1",
        "status": "blocked",
        "status_reason": [
            "不存在能同时满足『未暴露新 task 集』与『≥3 可靠 generator』的现成组合：",
            "  • Path A（meta/LCv2 复用）：三 generator 齐备（D0 3 折 admitted），但其 task 命名空间已在 pair 轮开发使用，"
            "且原 test 24 个中 22 个已被读取——复用需指导端明确接受『命名空间先前开发使用』口径，否则只能算独立复核而非新任务集；",
            "  • Path B（google/mistral + 第三 generator）：本机无任何候选权重/缓存，版本、许可与调用均未落实；",
            "  • Path C（openai/AB 单元 + 新任务）：AB 命名空间已 3 次暴露，新任务需外部 API（未授权）。",
            "按指导 §1：不得为把 status 变成 ready 而猜补字段。",
        ],
        "source_unit": "",
        "family_semantics": "",
        "target_family": "",
        "existing_generators": [],
        "target_third_generator": "",
        "task_source": [],
        "language": "C",
        "task_namespace": "",
        "unexposed_task_rule": "task/provenance 不得命中 E0 暴露账本且不得用于 P0/D1/N2；token 级三层 hash 审计后方可入池",
        "positive_definition": "same task, same family, different generator",
        "negative_definition": "same task, different family, or pre-registered cross-family control",
        "split_rule": "task/provenance group before generation",
        "heldout_generator_rule": "test generator absent from train/dev",
        "generation_allowed": False,
        "training_allowed": False,
        "external_cost_authorized": False,
        "test_read_allowed": False,
        "candidate_paths": {
            "A_meta_lcv2_reuse": {
                "source_unit": "h2_llm_codegen_v2::meta", "family_semantics": "vendor_group（Meta；release-folder 推导）",
                "existing_generators": FAMILIES_META,
                "generation_needed": False,
                "dry_run_pool": f"{len(cand_rows)} 个未作 test 的 task（保守排除原 test 24）",
                "blocker": "task 命名空间在 pair 轮用于 train/dev 开发；需指导端书面裁定『独立复核』是否成立",
                "becomes_ready_if": "指导端接受该口径 + 授权一次评测读取",
            },
            "B_third_generator": {
                "source_unit": "h2_llm_codegen_v2::google 或 ::mistral",
                "existing_generators": {"google": ["Gemni-1.5-pro", "codegemma"], "mistral": ["codestral", "mistral"]},
                "target_third_generator": "未指定（须真实可用版本）",
                "blocker": "本机无权重、无 API 授权、版本/许可未证实",
                "becomes_ready_if": "提供确切版本+许可+调用可行性后重写预注册",
            },
            "C_openai_new_tasks": {
                "source_unit": "h2_authorbench_dcan::openai",
                "existing_generators": ["gpt-4.1", "gpt-4o", "gpt-4o-mini"],
                "blocker": "AB 命名空间已暴露；新任务需外部 API（未授权）",
            },
        },
        "dry_run_artifacts": ["task_dry_run/candidate_tasks.jsonl", "task_dry_run/excluded_tasks.jsonl",
                              "audit/collision_groups.json", "prereg/split_plan.json",
                              "audit/license_and_provenance.json", "task_dry_run/dry_run_report.md"],
        "missing_fields": ["source_unit（Target 未定）", "family_semantics（Target 未定）", "target_family",
                           "existing_generators[]", "target_third_generator（Path B）", "task_source[]",
                           "task_namespace", "license（LCv2/AB 均缺）"],
        "blockers": ["Path A：task 命名空间先前开发使用需裁定", "Path B：第三 generator 不可证实",
                     "Path C：命名空间已暴露 + API 未授权"],
        "generation_plan_produced": False,
        "generation_plan_note": "按 §3：E2a != ready ⇒ 不生成 generation_plan.json（避免无门槛的行动文件）",
        "written_utc": datetime.now(timezone.utc).isoformat(),
    }
    (BASE / "prereg/e2_preregistration.json").write_text(json.dumps(prereg, ensure_ascii=False, indent=2) + "\n",
                                                         encoding="utf-8")
    log("e2_preregistration.json（blocked）已写出")

    # ---------- pilot_design.md / dry_run_report.md / feasibility_decision.md ----------
    pd = []
    pd.append("# pilot_design：为什么该 unit（不）能检验 H2（2026-10-07）")
    pd.append("")
    pd.append("## Path A：Meta / h2_llm_codegen_v2（最接近 ready）")
    pd.append("")
    pd.append(f"- 三 generator 齐备（llama2/llama3/codellama；非空 code task 覆盖 134/152/164 of 168），D0 已有 3 个 admitted heldout 折；")
    pd.append(f"- 能检验：same_task/same_family/different_generator 正对在同族内跨 generator 的判别/迁移（H2 的工程形态）与 H1 族内可读性；"
              f"干跑池 {len(cand_rows)} task（保守排除原 test 24 后）。")
    pd.append("- 不能检验 / 风险：该命名空间的 task 已用于 pair 轮 train/dev 开发——若指导端不接受此口径，则不能称为“新任务集”；"
              "输出为已有生成产物（复用需授权）；无 base/SFT/DPO 元数据 ⇒ 不能检验后训练因果。")
    pd.append("")
    pd.append("## Path B：google/mistral + 第三 generator")
    pd.append("")
    pd.append("- 理论上补齐后同样能检验 H2；但现在**无法给出任何真实可用的第三 generator**（本机无权重/无 API/版本未定）→ 不可检验。")
    pd.append("")
    pd.append("## Path C：OpenAI / AB")
    pd.append("")
    pd.append("- AB 命名空间已暴露，新任务必须全新；需要闭源 API 与新任务集 → 当前不可检验。")
    pd.append("")
    pd.append("## 三个路径共同不能检验的内容")
    pd.append("")
    pd.append("- 跨 dataset 拼接的“普遍 family 几何”；后训练因果；H3 detection/private-adapter（本轮不涉及）。")
    pd.append("")
    (BASE / "pilot_design.md").write_text("\n".join(pd) + "\n", encoding="utf-8")

    dr = []
    dr.append("# E2b 干跑报告（只读；未生成、未训练、未读旧 test 原文）")
    dr.append("")
    dr.append(f"- 候选：{len(cand_rows)}（LCv2 train+dev；三层 hash 建档）；排除：LCv2 原 test {len(excluded_lc)} + AB test {len(e0_ab_test_tasks)}。")
    dr.append("- 碰撞：候选内 multi 组 exact/ws/lex = "
              f"{collisions['within_candidates']['exact']['multi_groups']}/"
              f"{collisions['within_candidates']['norm_ws']['multi_groups']}/"
              f"{collisions['within_candidates']['norm_lex']['multi_groups']}，"
              f"**全部为 group 内 simple/secure 设计对**（同 prompt、同 group、同 split；"
              f"`multi_groups_intra_provenance_group_only={collisions['multi_groups_intra_provenance_group_only']}`，见 `audit/collision_groups.json` 样本）；"
              f"跨 split 组 {collisions['cross_split_groups']}；对 AB prompt 命中 {collisions['vs_ab_prompt_exact']}；"
              f"对 LCv2 原 test 命中 {collisions['vs_lc_test_exact']}。")
    dr.append(f"- split_plan（group 级 seed 20261007）：{split_plan['counts']}；meta 折支持预览见 `prereg/split_plan.json`。")
    dr.append(f"- 磁盘：若走 Path A 复用（无需生成）：新增索引/预测量级 < 5MB；若走 Path B/C 生成 144~300 task × 1–3 gen："
              f"≈0.6–1.8MB 原文 + 索引 < 5MB（E1 的 ≈15MB 估计不变）。")
    dr.append("- 生成计划：**未生成**（E2a=blocked；按 §3 门槛不产出 generation_plan.json）。")
    dr.append("")
    dr.append("## 三层碰撞明细")
    dr.append("")
    dr.append(fmt_table([[lvl, collisions['within_candidates'][lvl]['multi_groups'],
                          collisions['cross_split_groups'][lvl]] for lvl in ("exact", "norm_ws", "norm_lex")],
                        ["level", "candidate 内 multi 组", "跨 split 组"]))
    dr.append("")
    dr.append("（说明：LCv2 的 group = 同一 CWE 的 simple/secure 两个 task，prompt 文本相同；"
              "同名 multi 组均为该设计对，不构成跨任务泄漏；group 级切分保证其不跨 split。）")
    dr.append("")
    dr.append("## meta 单元 H2 折支持预览（split_plan 下）")
    dr.append("")
    dr.append(fmt_table([[f["heldout_generator"], f["train_tasks_with_ge2_seen_gen_nonempty"],
                          f["dev_tasks_with_ge2_seen_gen_nonempty"], f["test_tasks_heldout_nonempty"]]
                         for f in meta_folds],
                        ["heldout", "train(≥2 seen gen 非空)", "dev(同上)", "test(heldout 非空)"]))
    dr.append("")
    (BASE / "task_dry_run/dry_run_report.md").write_text("\n".join(dr) + "\n", encoding="utf-8")

    fd = []
    fd.append("# E2 可行性决定：**blocked**（2026-10-07）")
    fd.append("")
    fd.append("## 唯一缺口（按指导 §6/§7）")
    fd.append("")
    fd.append("1. **任务独立性口径**：Path A（meta/LCv2）需要指导端书面裁定“命名空间曾在 pair 轮开发使用”是否仍算独立复核；"
              "若不接受，则当前服务器上不存在任何满足『未暴露 + ≥3 generator』的 task 集。")
    fd.append("2. **第三 generator 不可证实**（Path B）：版本/许可/权重/API 均缺失（本机核查证据见 `audit/license_and_provenance.json`）。")
    fd.append("")
    fd.append("## 为转成 ready 所需的最小输入")
    fd.append("")
    fd.append("- 二选一：(a) 指导端接受 Path A 口径 + 授权一次评测读取（无需生成）；"
              "或 (b) 提供全新 C 任务集（含 provenance/license）并授权对目标单元生成。")
    fd.append("- 若走 B：(c) 指定第三 generator 的确切版本与调用/许可可行性。")
    fd.append("")
    fd.append("## 停止规则对照（§6）")
    fd.append("")
    fd.append("- 第三 generator 不可复现 ✅ 触发 → 停止扩展；其余规则（重叠、双标签、15MB、随机复制）未触发。")
    fd.append("- 本决定不改变任何既有标签/拆分/对照；不生成任何数据或计划文件。")
    fd.append("")
    (BASE / "prereg/feasibility_decision.md").write_text("\n".join(fd) + "\n", encoding="utf-8")

    (BASE / "logs/e2.log").write_text("\n".join(logs) + "\n", encoding="utf-8")
    print(json.dumps({"e2": "done", "status": "blocked", "candidates": len(cand_rows),
                      "runtime_s": round(time.time() - t0, 1)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
