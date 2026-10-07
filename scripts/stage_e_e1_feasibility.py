"""Stage E1: 独立数据可行性（只扫描已有 manifest/summary/元数据；不下载、不生成、不训练）。

依据《d-det_AutoDL_N2收尾与独立数据可行性指导_2026-10-07.md》§3。
输出：d-det/artifacts/stage_e_evidence_feasibility_2026-10-07/e1/
  dataset_feasibility.json / coverage_matrix.csv / pilot_design.json /
  feasibility_report.md / logs / SHA256SUMS
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
BASE = ROOT / "d-det/artifacts/stage_e_evidence_feasibility_2026-10-07"
OUT = BASE / "e1"
DATA = ROOT / "d-det/data"
ART = ROOT / "d-det/artifacts"

sys.path.insert(0, str(Path(__file__).resolve().parent))


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def fmt_table(rows, headers):
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(lines)


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "logs").mkdir(exist_ok=True)
    logs = []

    def log(s):
        print(s, flush=True)
        logs.append(s)

    # ---------- 输入元数据 ----------
    ab_sum = json.loads((DATA / "h2_authorbench_dcan/summary.json").read_text(encoding="utf-8"))
    lc_sum = json.loads((DATA / "h2_llm_codegen_v2/summary.json").read_text(encoding="utf-8"))
    lc_split = json.loads((DATA / "h2_llm_codegen_v2/split_manifest.json").read_text(encoding="utf-8"))
    lc_prov = json.loads((DATA / "h2_llm_codegen_v2/provenance.json").read_text(encoding="utf-8"))
    d0 = json.loads((ART / "stage_d_support_2026-10-07/alignment_support.json").read_text(encoding="utf-8"))
    n1 = json.loads((ART / "stage_d_data_construction_2026-10-07/n1_candidate_matrix/n1_candidate_support_matrix.json").read_text(encoding="utf-8"))
    n1_roles = json.loads((ART / "stage_d_data_construction_2026-10-07/n1_candidate_matrix/n1_data_role_matrix.json").read_text(encoding="utf-8"))["roles"]
    e0_ledger = json.loads((BASE / "e0/test_exposure_ledger.json").read_text(encoding="utf-8"))
    r1 = json.loads((ART / "acl_dcan_round1/audit_server_2026-10-03.json").read_text(encoding="utf-8"))

    # ---------- LCv2 覆盖统计（index 解析，不取代码正文） ----------
    lc_fg = Counter()
    lc_tasks_by_fg = defaultdict(set)
    lc_splits_by_fg = defaultdict(Counter)
    with (DATA / "h2_llm_codegen_v2/response_tasks.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            t = json.loads(line)
            idx = {o["generator"]: o for o in t["outputs"]}
            for o in t["outputs"]:
                key = (o["family"], o["generator"])
                lc_fg[key] += 1
                lc_tasks_by_fg[key].add(t["task_id"])
                lc_splits_by_fg[key][t["task_split"]] += 1
            _ = idx
    log(f"LCv2: {lc_sum['tasks']} tasks / {sum(lc_fg.values())} outputs / {len(lc_fg)} family-generator 组")

    # ---------- coverage_matrix.csv ----------
    cov_path = OUT / "coverage_matrix.csv"
    with cov_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["source", "family", "generator", "rows", "tasks", "train_rows", "dev_rows", "test_rows",
                    "family_role", "heldout_fold_ids", "test_evaluated", "dup_audit"])
        for fam, info in n1["designation"].items():
            for g in info["generators"]:
                rm = n1_roles[f"{fam}::{g}"]
                w.writerow(["h2_authorbench_dcan", fam, g, info["generator_rows"][g], info["generator_tasks"][g],
                            rm["splits"]["train"]["rows"], rm["splits"]["dev"]["rows"], rm["splits"]["test"]["rows"],
                            info["role"], ";".join(info["heldout_fold_ids"][g]) or "",
                            "yes(D1/N2 exposure)", "exact/ws/lex done"])
        for (fam, g), n_rows in sorted(lc_fg.items()):
            sp = lc_splits_by_fg[(fam, g)]
            w.writerow(["h2_llm_codegen_v2", fam, g, n_rows, len(lc_tasks_by_fg[(fam, g)]),
                        sp.get("train", 0), sp.get("dev", 0), sp.get("test", 0),
                        ("h2_candidate(3gen)" if len({x for (f2, x) in lc_fg if f2 == fam}) >= 3 else
                         ("h1_only(1gen)" if len({x for (f2, x) in lc_fg if f2 == fam}) == 1 else "insufficient(2gen)")),
                        "", "partial(pair rounds；family-test 未做)", "not run"])
    log(f"coverage_matrix.csv 已写出（AB 8 行 + LCv2 {len(lc_fg)} 行）")

    # ---------- dataset_feasibility.json ----------
    ab_dup = {"exact_cross_split": 0, "norm_ws_cross_split": 0, "norm_lex_cross_split": 25,
              "excluded_rows": n1["exclusions"]["excluded_rows"], "status": "done(N0/N1)"}
    sources = [
        {
            "dataset": "h2_authorbench_dcan", "rows": ab_sum["rows"], "tasks": ab_sum["tasks"],
            "language": ab_sum["language"],
            "license_provenance": {"source_archive": ab_sum.get("source_archive"),
                                   "source_archive_sha256": ab_sum.get("source_archive_sha256"),
                                   "license": "missing（summary 未提供）"},
            "family_definition": "vendor group（6 厂商；gpt-4.x 同 vendor 不等同已知共享谱系）",
            "families": ab_sum["families"], "generators": ab_sum["models"],
            "generator_per_family": {f: n1["designation"][f]["generator_count"] for f in n1["designation"]},
            "task_namespace": "prompt 级别 task（task_id），task 单 split",
            "alignable_task_hash": "yes（prompt_or_task_sha256，见 n1_manifest）",
            "split_rows": {"train": 6521, "dev": 1467, "test": 1444},
            "split_support": "N1 c3 全过（每族 train≥801 行 / dev≥182 行）",
            "seen_heldout_role": "openai: 3 个 generator-heldout 折（D0 admitted）；其余 5 族单 generator",
            "dup_audit": ab_dup,
            "evaluated_on_test": "yes：P0(10-05) → D1(08:35Z, n=1457) → N2(09:05Z, 子集 404/51 task)",
            "independently_holdable": "no：同一 dataset 的 test 已三次读取；train/dev 未作 test 但非独立来源",
        },
        {
            "dataset": "h2_llm_codegen_v2", "rows": lc_sum["outputs"], "tasks": lc_sum["tasks"],
            "language": ["C"],
            "license_provenance": {"provenance": "llm_codegen_raw CSV（sha 见 provenance.json）",
                                   "csv_files": len(lc_prov.get("csv_files", [])),
                                   "license": "missing（README/元数据未给 license）"},
            "family_definition": "release-folder 推导的 vendor/族（README 明确：无 base/SFT/DPO 谱系元数据）",
            "families": sorted({f for f, _ in lc_fg}),
            "generators": sorted({g for _, g in lc_fg}),
            "generator_per_family": {f: len({g for f2, g in lc_fg if f2 == f}) for f in sorted({f for f, _ in lc_fg})},
            "task_namespace": "CWE/group_id 任务（同 CWE simple/secure 共组）",
            "alignable_task_hash": "yes（prompt_sha256s / group_id）",
            "split_rows": {"train": lc_sum["tasks_by_split"]["train"], "dev": lc_sum["tasks_by_split"]["dev"],
                           "test": lc_sum["tasks_by_split"]["test"]},
            "split_support": "group/CWE 级 split（84 group：59/13/12）",
            "seen_heldout_role": "meta: 3 折 admitted（D0）；google/mistral: 2 generator（test-only 正支持 0 的 4 折已标 diagnostic）",
            "dup_audit": "not run（exact/ws/lex 未做）",
            "evaluated_on_test": "partial：pair round1/2 读过 test（22 task/196 行）；family 级 task-test 未做",
            "independently_holdable": "partial：146 个非 test task 从未作为评测；但同源已用于开发",
        },
        {
            "dataset": "h2_stacad_alignment_v1 / stacad_v2", "rows": r1.get("stacad", {}).get("pairs"),
            "tasks": r1.get("stacad", {}).get("files"),
            "language": ["py", "java", "c", "cpp", "php", "go", "cs"],
            "license_provenance": {"license": "missing"},
            "family_definition": "generator 集合（7 个模型），非 vendor 分组",
            "families": "7 generators", "generators": 7,
            "generator_per_family": "n/a（单一层）",
            "task_namespace": "file 级 pair（GroupKFold by file）",
            "alignable_task_hash": "file identity",
            "split_rows": "5 折 file 级",
            "seen_heldout_role": "file-heldout 折",
            "dup_audit": "not run",
            "evaluated_on_test": "yes：round1–4 + P0（fold0 .3729 等）",
            "independently_holdable": "no（已评估）",
        },
        {
            "dataset": "h2_droid_full_selected（v2）", "rows": 146718,
            "tasks": "generator-heldout 折（gen 级）",
            "language": "多语言",
            "license_provenance": {"license": "missing"},
            "family_definition": "vendor（Meta/Google/Microsoft/… + human）",
            "families": "32 machine generator（+human）",
            "generators": 32,
            "generator_per_family": "多 generator（具体见表见 v2 audit）",
            "task_namespace": "generator-heldout（无 task 对齐）",
            "alignable_task_hash": "missing（无 task 概念）",
            "split_rows": "fold_plan（E36 同源）",
            "seen_heldout_role": "generator-heldout",
            "dup_audit": "not run（exact 检查有）",
            "evaluated_on_test": "yes：E36 / v2 / P0（.1645）",
            "independently_holdable": "no",
        },
        {
            "dataset": "aicd_t2_numeric_balanced_v1（AICD T2）", "rows": r1.get("aicd", {}).get("dirs", {}).get("T2", {}).get("rows", 1111199),
            "tasks": "missing（无 task/prompt 索引）",
            "language": "多语言",
            "license_provenance": {"license": "missing；12 个 numeric 标签映射未解析"},
            "family_definition": "missing（numeric_id 12 类，未映射到 vendor/谱系）",
            "families": "12 numeric classes（mapping pending）",
            "generators": "missing",
            "generator_per_family": "missing",
            "task_namespace": "missing",
            "alignable_task_hash": "missing",
            "split_rows": "官方 train/validation/test",
            "seen_heldout_role": "n/a",
            "dup_audit": "exact 有（跨 split 泄露哈希 1,848 见 10-03 审计）；ws/lex not run",
            "evaluated_on_test": "yes：numeric baseline 2026-10-07 + P0 char-tfidf .2444",
            "independently_holdable": "no（test 已读）；且 family 映射缺失",
        },
        {
            "dataset": "codet_m4 / balanced_control", "rows": r1.get("codet_m4", {}).get("rows", 500552),
            "tasks": "missing（无 task 概念）",
            "language": "java/python/cpp",
            "license_provenance": {"license": "missing"},
            "family_definition": "model 名（含缺失 13,587 null）",
            "families": "model 级",
            "generators": "多模型（null 需剔除）",
            "generator_per_family": "missing",
            "task_namespace": "missing",
            "alignable_task_hash": "missing",
            "split_rows": "自带 split",
            "seen_heldout_role": "n/a",
            "dup_audit": "not run",
            "evaluated_on_test": "yes：外部控制（codet_m4_external_control_2026-10-07）",
            "independently_holdable": "no",
        },
    ]

    fold_support = []
    for f in d0["folds"]:
        fold_support.append({
            "fold_id": f["fold_id"], "source": f["source"], "family": f["family"],
            "heldout_generator": f["heldout_generator"], "admitted": f["admitted"],
            "fail_reasons": f.get("fail_reasons", []),
            "train_pairs": f["roles"]["train"]["pairs"], "train_pos": f["roles"]["train"]["pos"],
            "train_neg": f["roles"]["train"]["neg"],
            "dev_pairs": f["roles"]["dev"]["pairs"], "dev_pos": f["roles"]["dev"]["pos"],
            "test_pairs": f["roles"]["test"]["pairs"], "test_pos": f["roles"]["test"]["pos"],
            "test_tasks": f["roles"]["test"]["tasks"],
        })

    feasibility = {
        "schema": "stage_e1_dataset_feasibility_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "metadata_only": True,
        "inputs": {
            "ab_summary": {"path": "d-det/data/h2_authorbench_dcan/summary.json",
                           "sha256": sha256_file(DATA / "h2_authorbench_dcan/summary.json")},
            "lc_summary": {"path": "d-det/data/h2_llm_codegen_v2/summary.json",
                           "sha256": sha256_file(DATA / "h2_llm_codegen_v2/summary.json")},
            "lc_provenance_sha256": sha256_file(DATA / "h2_llm_codegen_v2/provenance.json"),
            "d0_alignment": {"path": "d-det/artifacts/stage_d_support_2026-10-07/alignment_support.json",
                             "sha256": sha256_file(ART / "stage_d_support_2026-10-07/alignment_support.json")},
            "n1_candidate_matrix": {"sha256": sha256_file(ART / "stage_d_data_construction_2026-10-07/n1_candidate_matrix/n1_candidate_support_matrix.json")},
            "round1_audit": {"path": "d-det/artifacts/acl_dcan_round1/audit_server_2026-10-03.json",
                             "sha256": sha256_file(ART / "acl_dcan_round1/audit_server_2026-10-03.json")},
        },
        "paper_goal": {
            "main_task": "模型来源/家族归因（见总结文档 §4/§33）",
            "auxiliary": ["detection 单独报告", "unknown-family 拒识"],
            "family_semantics_requirement": "family 必须写明是 vendor group / 已知共享谱系 lineage / generator 集合；"
                                            "GPT 版本同 vendor ≠ 已知共享权重；Llama 系列不能凭名字当作唯一同基座对",
            "post_training_separation": "缺 base/SFT/DPO 配对元数据（LCv2 README 明示）；本轮不把 task 中心化当后训练因果证据",
        },
        "sources": sources,
        "fold_support": fold_support,
        "task_coverage": {
            "ab_tasks_by_family_coverage": {str(k): v for k, v in sorted(Counter(
                json.loads(l)["family_coverage"] for l in
                (ART / "stage_d_data_construction_2026-10-07/n1_candidate_matrix/n1_manifest.jsonl").open(encoding="utf-8")
            ).items())},
            "ab_six_family_subset": n1["candidate_subset_for_n2"],
            "lc_outputs_per_family": {f: sum(v for (f2, g), v in lc_fg.items() if f2 == f)
                                      for f in sorted({f for f, _ in lc_fg})},
        },
        "exposure_note": e0_ledger["relationships"]["note"],
        "runtime_seconds": time.time() - t0,
    }
    (OUT / "dataset_feasibility.json").write_text(json.dumps(feasibility, ensure_ascii=False, indent=2) + "\n",
                                                  encoding="utf-8")
    log("dataset_feasibility.json 已写出")

    # ---------- pilot_design.json ----------
    pilot = {
        "schema": "stage_e1_pilot_design_v1",
        "status": "partial",
        "status_reason": [
            "现成数据中同时满足『≥3 family × 每族 ≥3 generator』的只有 openai(AB) 与 meta(LCv2) 两个单元；"
            "google/mistral 各缺第 3 个 generator，其余族为单 generator。",
            "pilot 要求同一新任务集合上的多族多 generator 采样；现有任务池无此组合，需要新增生成（未授权）。",
        ],
        "family_semantics": "vendor_or_lineage_explicit",
        "target_families": [
            {"family": "openai", "semantics": "vendor（gpt-4.1/gpt-4o/gpt-4o-mini；同 vendor，不等同已知共享谱系）",
             "generators_available": ["gpt-4.1", "gpt-4o", "gpt-4o-mini"], "missing_generators": 0},
            {"family": "meta", "semantics": "vendor/系列（llama2/llama3/codellama；谱系需在协议中写清）",
             "generators_available": ["llama2", "llama3", "codellama"], "missing_generators": 0},
            {"family": "google 或 mistral", "semantics": "vendor",
             "generators_available": {"google": ["Gemni-1.5-pro", "codegemma"], "mistral": ["codestral", "mistral"]},
             "missing_generators": 1},
        ],
        "generators_per_family": {"openai": 3, "meta": 3, "google": 2, "mistral": 2},
        "task_source": [
            {"option": "新 prompt/任务集（推荐；来源委托方指定）", "status": "missing"},
            {"option": "复用 AB prompts 的子集", "status": "exposure：AB test 已读 3 次；train/dev task 可复用但非独立"},
            {"option": "复用 LCv2 CWE 任务", "status": "exposure：22 个 test task 已读；同源开发使用过其余任务"},
        ],
        "unexposed_task_evidence": [
            {"pool": "AB train/dev tasks", "count": 2307, "exposed_as_test": False,
             "note": "同 dataset 内未作 test；可作 pilot 的现实来源但独立性强于全新任务"},
            {"pool": "LCv2 非 test tasks", "count": 146, "exposed_as_test": False,
             "note": "pair 轮仅读 22 个 test task；其余未作评测"},
            {"pool": "aicd/droid/codet_m4", "exposed_as_test": True,
             "note": "均已被既有评测读取；且 family/task 语义缺失，不可作独立 pilot 任务池"},
        ],
        "split_grouping": "task_or_provenance_component（按 task/prompt 或 CWE group 分 split；先 task 后样本）",
        "heldout_generator_plan": [
            "对每个目标 family，每折 hold out 恰好 1 个 generator；该 generator 输出不进该折 train/dev；",
            "test 同时要求未见 task（task/prompt 级隔离）；",
            "正对定义见 positive_definition；每折核对 train/dev 正支持与 test task 数 ≥ 预注册下限。",
        ],
        "positive_definition": "same_task_same_family_different_generator",
        "hard_negative_definition": "same_task_different_family",
        "compute_and_disk_estimate": {
            "pilot_scale_assumption": "≈300 新 task × 3 family × 3 generator ≈ 2,700 输出",
            "raw_outputs": "≈4 MB（按 AB 实测 ≈1.4 KB/行）",
            "light_index": "≈2–5 MB（manifest/coverage/审计 JSON/CSV）",
            "optional_feature_cache": "冻结 CodeT5-small 768d fp32 ≈ 2,700×768×4B ≈ 8 MB（可现算不入库）",
            "reuse": ["d-det/models/codet5-small（已校准）", "d-det/checkpoints/codet5-base", "u-det/scripts/stage_* 审计脚本"],
            "generation_cost": "闭源 API / 未授权（12 模型 × 300 prompt）——仅估计，未请求执行",
            "incremental_disk_total": "≈15 MB 量级（不大；瓶颈是生成与许可而非磁盘）",
        },
        "power_plan": {
            "reference": "N2：51 test tasks → tfidf_word CI 半宽 ≈4.8pt（F1）",
            "scaling_rule": "半宽 ~ 1/sqrt(n_tasks)（粗规则，未含配对增益）",
            "rough_requirement_for_1pt": "≈ (4.8/1)^2 × 51 ≈ 1,200 tasks（未配对、单视图；配对/多视图可显著降低，需在协议中模拟）",
            "status": "power_unknown（最终统计功效需在预注册协议里用模拟/试验性估计冻结）",
        },
        "cost_and_generation_authorization": "not_requested_here",
        "missing_assets": ["google/mistral 第 3 个 generator（生成授权）", "新任务 prompt 集（来源指定）",
                           "闭源 API 访问与预算（如沿用 gpt-4.x）", "各来源 license/许可确认"],
        "training_allowed": False,
    }
    (OUT / "pilot_design.json").write_text(json.dumps(pilot, ensure_ascii=False, indent=2) + "\n",
                                           encoding="utf-8")
    log("pilot_design.json 已写出")

    # ---------- feasibility_report.md ----------
    L = []
    L.append("# E1：独立数据可行性（2026-10-07，仅元数据扫描；未下载/未生成/未训练）")
    L.append("")
    L.append("依据指导 §3；输入为服务器已有 summary/manifest/审计 JSON（hash 见 `dataset_feasibility.json.inputs`）。")
    L.append("")
    L.append("## 1 论文目标与 family 语义")
    L.append("")
    L.append("- 主任务 = 模型来源/家族归因；detection 与拒识另表。family 语义必须在协议中明确为 "
             "vendor group / 共享谱系 lineage / generator 集合；同 vendor 不同版本（gpt-4.x）不等于已知共享权重谱系；"
             "Llama 系列也不能凭名字当作唯一同基座对。")
    L.append("- 缺 base/SFT/DPO 配对元数据（LCv2 README 明示）；本轮不把 task 中心化当作后训练因果证据。")
    L.append("")
    L.append("## 2 现有来源可识别性（要点；全量见 JSON）")
    L.append("")
    L.append(fmt_table([
        ["h2_authorbench_dcan", 9498, 2715, "6 vendor 族 / 8 gen", "exact/ws/lex done", "已评估（3 次 test 读取）", "no"],
        ["h2_llm_codegen_v2", lc_sum["outputs"], lc_sum["tasks"], "5 族 / 9 gen（meta=3，google/mistral=2）",
         "not run", "partial（pair 读 22 task）", "partial"],
        ["h2_stacad_*", r1.get("stacad", {}).get("pairs"), r1.get("stacad", {}).get("files"),
         "7 models（非 vendor 分组）", "not run", "yes", "no"],
        ["h2_droid_full_selected", 146718, "gen-heldout", "32 machine gen", "exact 有", "yes", "no"],
        ["aicd_t2（numeric）", r1.get("aicd", {}).get("dirs", {}).get("T2", {}).get("rows"), "missing",
         "12 numeric（映射缺失）", "exact 有（跨 split 1,848）", "yes", "no"],
        ["codet_m4/balanced", r1.get("codet_m4", {}).get("rows"), "missing", "model 级（含 null）", "not run", "yes", "no"],
    ], ["dataset", "rows", "tasks", "family/generator", "dup audit", "test 暴露", "可独立留出"]))
    L.append("")
    L.append("## 3 覆盖率与折支持（回传 §6.4）")
    L.append("")
    L.append("- AB 每族 generator 数：openai=3；claude/deepseek/gemini/llama/qwen=1（`coverage_matrix.csv`）。")
    L.append("- LCv2：meta=3（三折 admitted）；google=2、mistral=2（4 折 diagnostic-only）；ibm/microsoft=1。")
    L.append("- 10 折 train/dev 正支持与 test task 数：见 `dataset_feasibility.json.fold_support`。")
    L.append("- 同任务覆盖：AB 2,715 task 中六族齐全 285（剔除后 280）；LCv2 每 task 9 输出（5 族）。")
    L.append("")
    L.append("## 4 pilot 蓝图（只规划；training_allowed=false）")
    L.append("")
    L.append(f"- status = **partial**：现成 triple 仅 openai(AB) 与 meta(LCv2)；google/mistral 缺第 3 个 generator；"
             f"任务池无“同一新任务集”的现成组合。")
    L.append(f"- 估计规模：≈300 task × 3 族 × 3 gen ≈ 2,700 输出 ≈ 4MB 原文 + ~10MB 索引/缓存（增量磁盘 ≈15MB 量级）。")
    L.append(f"- 功效：power_unknown（粗规则：~1,200 task 量级才能分辨 1pt；需预注册模拟）。")
    L.append(f"- 缺口：第 3 个 google/mistral generator（生成授权）、新任务提示集、闭源 API/预算、license 确认。")
    L.append("")
    L.append("## 5 出口")
    L.append("")
    L.append("- E1 = **partial**：交付缺口清单与预算；训练/生成/外部调用**未请求、未执行**；不自行删除历史资产。")
    L.append("")
    (OUT / "feasibility_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (OUT / "logs" / "e1.log").write_text("\n".join(logs) + "\n", encoding="utf-8")
    print(json.dumps({"e1": "done", "pilot_status": pilot["status"], "runtime_s": round(time.time() - t0, 1)},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
