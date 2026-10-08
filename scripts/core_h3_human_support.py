"""H3 定义恢复与 human 支持表（§4.2）。

任务：
1. 标签重命名记录：旧“检测”分支实为 protocol_direction（complete/instruct 都是 AI）；
   原 H3（Human/AI detection × 来源归因）需要真实 human 数据。
2. human 支持表：盘点本地各候选源的“同题 human×AI”证据（读已有审计文件，不重扫大文件）：
   - public_same_task_full (BCC)：全 machine（complete/instruct）→ 无 human
   - DroidCollection：README 明示无公开 task_id/prompt_id；human 为 controls → 非同题
   - CoDET-M4：audit role="not a same-task semantic-pair corpus" → 非同题
   - STACAD-v2 / h2_stacad_alignment_v1：task_index 含 human_code；每任务 7 AI（7 generator/7 family）；
     pairs=同任务跨 generator → **同题 human×AI 支持成立**
3. 结论：H3_original 在当前 BCC 同题集不可用（H3_original_unavailable）；
   STACAD 为唯一同题 human×AI 支持候选（检测轴=human vs AI；归因轴=generator/family）。

输出：artifacts/h3_definition_and_human_support_2026-10-08/{human_support_table.json, report.md}
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "d-det/artifacts/h3_definition_and_human_support_2026-10-08"


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "logs").mkdir(exist_ok=True)

    # 读已有审计文件
    stacad = json.loads((ROOT / "d-det/data/h2_stacad_alignment_v1/audit_index.json").read_text(encoding="utf-8"))
    stacad_sum = json.loads((ROOT / "d-det/data/h2_stacad_alignment_v1/summary.json").read_text(encoding="utf-8"))
    droid_sum = json.loads((ROOT / "d-det/data/h2_droid_full_selected/summary.json").read_text(encoding="utf-8"))
    codet_audit = json.loads((ROOT / "d-det/data/codet_m4/audit.json").read_text(encoding="utf-8"))

    table = {
        "schema": "h3_human_support_table_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "h3_original_definition": "Human/AI detection（检测轴）与 来源归因（归因轴）的任务分离",
        "label_rename_note": ("已完成四配置的“检测”分支实为 protocol_direction（complete/instruct 均为 AI 生成），"
                              "应称 protocol_direction_plus_relation_multitask；D、F 与协议方向 R0 分开报告。"),
        "sources": [
            {
                "name": "public_same_task_full_2026-10-07 (BCC)",
                "path": "d-det/data/public_same_task_full_2026-10-07",
                "has_human": False,
                "human_evidence": "全部 rows 为机器输出（complete/instruct 双协议）；used records 字段含 solution；无 human 行",
                "same_task_human_ai": False,
                "verdict": "not_applicable_no_human",
            },
            {
                "name": "DroidCollection (h2_droid_full_selected)",
                "path": "d-det/data/h2_droid_full_selected",
                "rows": droid_sum.get("counts", droid_sum),
                "has_human": True,
                "human_evidence": "HUMAN_GENERATED 21,000（controls，每 family/source ≤1000）",
                "same_task_human_ai": False,
                "same_task_evidence": ("README 明示：DroidCollection does not expose a public task_id/prompt_id, "
                                       "so results must not be described as same-task paired generalization"),
                "verdict": "not_same_task",
            },
            {
                "name": "CoDET-M4",
                "path": "d-det/data/codet_m4",
                "rows": codet_audit.get("rows"),
                "has_human": True,
                "human_evidence": f"target=human {codet_audit['counts']['target'].get('human')} 行",
                "same_task_human_ai": False,
                "same_task_evidence": f"audit role={codet_audit.get('role')}",
                "verdict": "not_same_task",
            },
            {
                "name": "STACAD-v2 alignment (h2_stacad_alignment_v1)",
                "path": "d-det/data/h2_stacad_alignment_v1",
                "rows": stacad["core"]["rows"],
                "tasks": stacad["core"]["tasks"],
                "has_human": True,
                "human_evidence": ("task_index 每任务含 human_sha256 + human_code（原作者版本）；"
                                   f"tasks={stacad['core']['tasks']}"),
                "ai_units": {"generators": 7, "families": 7,
                             "per_task": 7, "languages": list(stacad["core"]["languages"].keys())},
                "same_task_human_ai": True,
                "same_task_evidence": ("task_index: 同 file 的 human_code + 7 个模型输出；"
                                       f"pairs: {stacad['pairs']['rows']} same_task_cross_generator；"
                                       "splits 无重叠（no_task_split_overlap=true）"),
                "splits": stacad_sum.get("split_counts"),
                "verdict": "support_candidate",
            },
        ],
        "conclusion": {
            "h3_on_bcc": "H3_original_unavailable（当前同题全 machine 集无 human；不得把协议方向 D 当检测轴）",
            "human_support_candidate": "STACAD-v2（同题 human×AI + 7 generator/7 family 归因轴）",
            "next_requirements": [
                "检测轴（human vs AI）与归因轴（generator）在同一 STACAD 任务集上的分离设计验证（另行立项）",
                "许可与使用条款复核；语言/难度分层；test 轴策略须预先冻结",
                "不得在无同题支持的数据（Droid/CoDET-M4/BCC 拼接）上写“检测轴与归因轴分离”",
            ],
        },
    }
    (OUT / "human_support_table.json").write_text(json.dumps(table, ensure_ascii=False, indent=1), encoding="utf-8")

    L = ["# H3 定义恢复与 human 支持表", "",
         "## 标签重命名（§4.2）", "",
         "- 已完成四配置的“检测”分支实为 **protocol_direction**（complete/instruct 都是 AI）——",
         "  改称 `protocol_direction_plus_relation_multitask`；D、F 与协议方向 R0 分开报告（记录收尾，不重跑补 epoch）。", "",
         "## human 支持表", "",
         "| 源 | human | 同题 human×AI | 依据 | 判定 |", "|---|---|---|---|---|"]
    for s in table["sources"]:
        L.append(f"| {s['name']} | {'有' if s['has_human'] else '无'} | {'✓' if s['same_task_human_ai'] else '✗'} | "
                 f"{s.get('same_task_evidence', s['human_evidence'])[:80]} | {s['verdict']} |")
    L += ["", "## 结论", "",
          f"- **{table['conclusion']['h3_on_bcc']}**",
          f"- 支持候选：**{table['conclusion']['human_support_candidate']}**",
          "- 后续要求：" + "；".join(table["conclusion"]["next_requirements"])]
    (OUT / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"[support] done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
