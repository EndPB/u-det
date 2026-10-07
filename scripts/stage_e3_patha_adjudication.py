"""Stage E3a: Path A 任务历史角色裁定（只读；不读旧 test 原文、不训练、不生成、不评测）。

依据《d-det_AutoDL_E2阻塞裁定与PathA探索性复核指导_2026-10-07.md》§2/§3/§4。
输出：d-det/artifacts/stage_e3_patha_adjudication_2026-10-07/
  audit/path_a_history_role.json / audit/path_a_adjudication.md /
  audit/fresh_namespace_plan.json / exploratory/status.json / logs / SHA256SUMS
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
BASE = ROOT / "d-det/artifacts/stage_e3_patha_adjudication_2026-10-07"
E2 = ROOT / "d-det/artifacts/stage_e2_independent_pilot_2026-10-07"
LC2 = ROOT / "d-det/data/h2_llm_codegen_v2"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage_d_n0_diagnosis import sha256  # noqa: E402


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def fmt_table(rows, headers):
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(lines)


def main():
    t0 = time.time()
    for d in ("audit", "exploratory", "logs"):
        (BASE / d).mkdir(parents=True, exist_ok=True)
    logs = []

    def log(s):
        print(s, flush=True)
        logs.append(s)

    # 输入
    cand = [json.loads(l) for l in (E2 / "task_dry_run/candidate_tasks.jsonl").open(encoding="utf-8")]
    split_plan = json.loads((E2 / "prereg/split_plan.json").read_text(encoding="utf-8"))
    tasks = {}
    with (LC2 / "response_tasks.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            t = json.loads(line)
            tasks[t["task_id"]] = {"task_id": t["task_id"], "group_id": t["group_id"],
                                   "cwe_id": t["cwe_id"], "task_split": t["task_split"],
                                   "positive_pairs": t.get("positive_pairs", [])}
    pair_rows = [json.loads(l) for l in (LC2 / "positive_pair_index.jsonl").open(encoding="utf-8")]
    pairs_by_task = defaultdict(list)
    for p in pair_rows:
        pairs_by_task[p["task_id"]].append(p)
    groups_with_pairs = {p["group_id"] for p in pair_rows}
    log(f"输入：候选 {len(cand)}；positive pair {len(pair_rows)} 行 / {len(pairs_by_task)} task / "
        f"{len(groups_with_pairs)} group")

    # E2 split_plan → 当前 e3 角色
    # 从 split_plan 的 task 列表 hash 反推角色：重算（同规则）
    # 注：E2 未保存逐 task split 列表文件，但 candidate_tasks + split_plan 规则可复现；这里以
    # split_plan 的计数字段为准，并重建 group→split 以对齐（与 E2 相同 seed/贪心）。
    import random
    by_group = defaultdict(list)
    for r in cand:
        by_group[r["provenance_group"]].append(r["task_id_candidate"])
    gids = sorted(by_group)
    rng = random.Random(20261007)
    rng.shuffle(gids)
    target = {"train": int(len(cand) * 0.70), "dev": int(len(cand) * 0.15),
              "test": len(cand) - int(len(cand) * 0.70) - int(len(cand) * 0.15)}
    assign, counts = {}, Counter()
    for gid in gids:
        n = len(by_group[gid])
        order = sorted(target, key=lambda s: (counts[s] + n) / max(target[s], 1))
        pick = order[0]
        assign[gid] = pick
        counts[pick] += n
    assert dict(counts) == split_plan["counts"], "split plan 复现不一致"
    e3_role = {}
    for gid, s in assign.items():
        for tid in by_group[gid]:
            e3_role[tid] = s
    log(f"E3 新 split 复现一致：{dict(counts)}")

    # 历史角色
    HIST = {
        "historical_train_seen": "task 或其同组数据参与过历史训练（pair-train 命名空间/dev 训练）",
        "historical_dev_seen": "用于阈值/超参/模型选择（pair-dev 或库 split=dev）",
        "historical_test_read": "已被读取或用于报告",
        "historical_unread_but_development_namespace": "未读但来自已开发 namespace",
        "fresh_unseen_namespace": "新的 provenance/task namespace",
    }
    rows = []
    for r in cand:
        tid = r["task_id_candidate"]
        t = tasks[tid]
        pairs = pairs_by_task.get(tid, [])
        pair_splits = sorted({p["task_split"] for p in pairs})
        if t["task_split"] == "train":
            role = "historical_train_seen"
        elif t["task_split"] == "dev":
            role = "historical_dev_seen"
        else:
            role = "historical_test_read"
        endpoint_hash = sha256("|".join(sorted(f"{p['left']}>{p['right']}:{p['family']}" for p in pairs))) if pairs else None
        rows.append({
            "task_id": tid, "group_id": t["group_id"], "cwe": t["cwe_id"],
            "historical_role": role,
            "historical_train_or_dev_seen": t["task_split"] in ("train", "dev"),
            "historical_test_read": False,
            "pair_index_pairs": len(pairs), "pair_index_splits": pair_splits,
            "pair_train_used": "train" in pair_splits,
            "pair_dev_used": "dev" in pair_splits,
            "current_e3_role": e3_role[tid],
            "task_source_hash": r["task_source_hash"],
            "endpoint_hashes": endpoint_hash,
            "related_group_seen": t["group_id"] in groups_with_pairs,
            "admissibility": {
                "decision": "exploratory_reuse_only",
                "main_h2_table": False,
                "appendix_or_diagnostic": "conditional_on_user_acceptance",
                "fresh_test_eligible": False,
                "reason": "来自已开发 namespace（pair train/dev 使用史；原 test 24 已读 22）",
            },
        })
    # 24 个原 test task 附录（信息性；不在 Path A 候选内）
    test_rows = [{
        "task_id": t["task_id"], "group_id": t["group_id"], "historical_role": "historical_test_read",
        "historical_test_read": True,
        "note": "24 中 22 已在 pair 轮读取 test；2 个未读但保守按 test 区处理；整体 excluded（E2）",
    } for t in tasks.values() if t["task_split"] == "test"]

    role_count = Counter(r["historical_role"] for r in rows)
    new_test = [r for r in rows if r["current_e3_role"] == "test"]
    new_test_role = Counter(r["historical_role"] for r in new_test)
    train_dev_pairs = sum(1 for r in rows if r["pair_train_used"] or r["pair_dev_used"])
    summary = {
        "candidates": len(rows),
        "historical_role_counts": dict(role_count),
        "new_test_tasks": len(new_test),
        "new_test_historical_role_counts": dict(new_test_role),
        "tasks_with_historical_pair_usage": train_dev_pairs,
        "fresh_unseen_namespace_tasks": 0,
        "fresh_test_eligible_tasks": 0,
        "adjudication": {
            "confirmatory_independent": False,
            "exploratory_development_namespace_reuse": True,
            "h2_main_table_allowed": False,
            "h2_appendix_diagnostic_allowed": "conditional（需用户明确接受降级标签后才可进行 E3b 只读复核）",
            "training_allowed": False,
            "new_generation_allowed": False,
        },
    }
    role_json = {
        "schema": "stage_e3a_path_a_history_role_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "e2_candidate_tasks": {"path": str((E2 / "task_dry_run/candidate_tasks.jsonl").relative_to(ROOT)),
                                   "sha256": sha256_file(E2 / "task_dry_run/candidate_tasks.jsonl")},
            "e2_split_plan": {"sha256": sha256_file(E2 / "prereg/split_plan.json")},
            "lcv2_positive_pair_index": {"sha256": sha256_file(LC2 / "positive_pair_index.jsonl")},
            "lcv2_response_tasks": {"sha256": sha256_file(LC2 / "response_tasks.jsonl")},
        },
        "historical_role_definitions": HIST,
        "summary": summary,
        "tasks": rows,
        "excluded_original_test_tasks": test_rows,
        "role_rule": ("原 split=train → historical_train_seen；原 split=dev → historical_dev_seen；"
                      "原 split=test → historical_test_read（不在候选内）；fresh_unseen_namespace 在服务器上为 0。"
                      "pair_train_used/pair_dev_used 依据 positive_pair_index 的 task_split。"),
        "runtime_seconds": time.time() - t0,
    }
    (BASE / "audit/path_a_history_role.json").write_text(json.dumps(role_json, ensure_ascii=False, indent=2) + "\n",
                                                         encoding="utf-8")
    log("path_a_history_role.json 已写出")

    # 裁定书
    md = []
    md.append("# E3a：Path A 历史角色裁定（2026-10-07，只读）")
    md.append("")
    md.append("依据《d-det_AutoDL_E2阻塞裁定与PathA探索性复核指导_2026-10-07.md》§2。")
    md.append("**未读任何 test 原文；未训练；未生成；未评测。**")
    md.append("")
    md.append("## 1 结论")
    md.append("")
    md.append("Path A 的 144 个候选 task **全部来自已开发 namespace**：")
    md.append(f"- 历史角色：{json.dumps(summary['historical_role_counts'], ensure_ascii=False)}；")
    md.append(f"- 有新 pair 使用史（train 或 dev）的 task：{train_dev_pairs}/144；")
    md.append(f"- 新 split 的 22 个候选 test task 的历史角色构成：{json.dumps(summary['new_test_historical_role_counts'], ensure_ascii=False)}；")
    md.append(f"- `fresh_unseen_namespace` task 数：0；`fresh_test_eligible` task 数：0。")
    md.append("")
    md.append("**正式标签（与指导 §0 一致）**：")
    md.append("```text")
    for k, v in summary["adjudication"].items():
        md.append(f"{k} = {v}")
    md.append("```")
    md.append("")
    md.append("## 2 规则与证据")
    md.append("")
    md.append("- 历史角色规则：原 split=train → `historical_train_seen`（118）；dev → `historical_dev_seen`（26）；"
              "原 test → `historical_test_read`（24，整体 excluded，见附录）。")
    md.append("- pair 使用证据：`positive_pair_index.jsonl`（695 行/159 task；README 声明 train 正对用于训练、dev 用于开发）。")
    md.append("- E0 暴露账本核对：144 task 无命中（无 AB 交集）；原 test 24 中 22 已读（pair 轮）。")
    md.append("- 新的 102/20/22 只是对已开发 namespace 的重新分组；22 个新 test 中 ")
    md.append(f"  train/dev 历史 task 占 {sum(summary['new_test_historical_role_counts'].values())} 个"
              "——不能因本轮未读而称为 fresh。")
    md.append("")
    md.append("## 3 允许的结论与禁止的写法（与指导 §1/§5 对齐）")
    md.append("")
    md.append("- 允许：`LCv2 development-namespace exploratory re-split` 的工程诊断（appendix/diagnostic；"
              "需用户接受降级标签并授权 E3b）。")
    md.append("- 禁止：独立新任务泛化 / 普遍 family 几何 / 后训练因果 / H3 分离 / "
              "把重切分写成 ACL 独立 H2 证据。")
    md.append("")
    md.append("## 4 出口")
    md.append("")
    md.append("- 默认出口 = 本裁定（E3a 只读）；**Path A 仍不进入 ACL 主 H2 表**。")
    md.append("- E3b（仅冻结输出/冻结读出的探索性复核）**等待用户明确接受降级标签**后执行；"
              "服务器不自行把 Path A 改为 ready。")
    md.append("- E3c（fresh unseen namespace）蓝图见 `audit/fresh_namespace_plan.json`。")
    md.append("")
    (BASE / "audit/path_a_adjudication.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    log("path_a_adjudication.md 已写出")

    # E3b 状态（等待授权；不自行授权）
    status = {
        "schema": "stage_e3b_status_v1",
        "status": "awaiting_user_acceptance",
        "note": "服务器不自行把 Path A 从 blocked 改为 ready；需用户明确接受 development-namespace exploratory 口径。",
        "authorization_template_pending": {
            "accepted_label": "exploratory_reuse_only",
            "confirmatory_independent": False,
            "main_h2_table": False,
            "appendix_or_diagnostic": True,
            "training_allowed": False,
            "new_generation_allowed": False,
            "new_test_read_allowed": True,
            "selection_after_read": False,
            "allowed_question": "performance on a held-out re-split of a historically developed namespace",
            "forbidden_claims": ["independent new-task generalization", "universal cross-family geometry",
                                 "post-training causal effect", "H3 separation"],
        },
        "planned_frozen_readouts_if_authorized": [
            "pair 级判决：复用 h2_pair_round2 的既有冻结模型对（meta 族正对）在新重切分 test 上打分；不重训、不调阈值、不选 seed",
            "冻结 CodeT5-base 嵌入的余弦相似度（零训练读出）",
            "报告 = 每 generator-heldout fold 的 BA/macro-F1/family/generator recall/task-cluster bootstrap CI/"
            "历史角色构成/test exposure；标题 = 'LCv2 development-namespace exploratory re-split'",
        ],
        "forbidden_actions": ["任何训练或重拟合（含 P0 fusion）", "阈值/seed/任务/generator 的选择",
                              "改写 split", "进入主 H2 表"],
    }
    (BASE / "exploratory/status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n",
                                                  encoding="utf-8")
    log("exploratory/status.json 已写出（awaiting_user_acceptance）")

    # E3c fresh namespace 蓝图（只规划）
    fresh = {
        "schema": "stage_e3c_fresh_namespace_plan_v1",
        "status": "plan_only_blocked_on_external_inputs",
        "requirements": [
            "新 task/provenance 不在 LCv2/AB 已开发或已暴露集合内（需提供来源与许可）",
            "先按 task/provenance group 固定 train/dev/test，再生成（generation 前冻结 split）",
            "同一新 task 至少由同一 source unit 的三个可靠 generator 生成",
            "target generator 版本/许可/请求参数/失败重试可复现",
            "冻结 positive/negative 定义、heldout generator 折、强 P0 双列（fusion+eq）、test 单次读取",
            "生成后 exact/ws/lex 对历史全量碰撞 = 0；碰撞组保留解释",
            "未授权前不调用 API、不下载模型",
        ],
        "candidate_source_units": {
            "meta": {"existing_generators": 3, "缺口": "无（在 fresh namespace 上重新生成 3 个既有 generator 输出即可）"},
            "google|mistral": {"existing_generators": 2, "缺口": "第三 generator 的版本/许可/权重（本机无）"},
            "openai(AB)": {"existing_generators": 3, "缺口": "闭源 API 授权 + 全新任务集"},
        },
        "cost_and_disk_estimate": {
            "scale_assumption": "≈150–300 新 task × 3 family × 3 generator",
            "raw_outputs": "≈1–2 MB 原文 + 索引 < 5 MB（E1/E2 一致量级 ≈15 MB 上限）",
            "generation_cost": "本地三 generator 若权重可得（GPU 小时级）；闭源 API 成本未知",
            "missing_assets": ["fresh 任务集（来源+许可）", "生成授权", "（若走 google/mistral）第三 generator 权重/API"],
        },
        "training_allowed": False,
        "new_generation_allowed": False,
        "power": "power_unknown（按真实 task 相关性预注册模拟；当前 51–144 task 的 CI 半宽 2.9–4.8pt 仅作参照）",
    }
    (BASE / "audit/fresh_namespace_plan.json").write_text(json.dumps(fresh, ensure_ascii=False, indent=2) + "\n",
                                                          encoding="utf-8")
    log("fresh_namespace_plan.json（E3c 蓝图）已写出")

    (BASE / "logs/e3a.log").write_text("\n".join(logs) + "\n", encoding="utf-8")
    print(json.dumps({"e3a": "done", "summary": summary["historical_role_counts"],
                      "new_test_role_mix": summary["new_test_historical_role_counts"],
                      "runtime_s": round(time.time() - t0, 1)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
