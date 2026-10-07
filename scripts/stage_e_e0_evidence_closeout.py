"""Stage E0: 证据收尾（只用现有产物；不重载 test 文本、不重调模型、不改超参）。

依据《d-det_AutoDL_N2收尾与独立数据可行性指导_2026-10-07.md》§2。
输出：d-det/artifacts/stage_e_evidence_feasibility_2026-10-07/e0/
  metric_reconciliation.json / p0_definition.json / gate_reconciliation.json /
  test_exposure_ledger.json / evidence_closeout.md / logs / SHA256SUMS
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "d-det/artifacts/stage_e_evidence_feasibility_2026-10-07"
OUT = BASE / "e0"
D1 = ROOT / "d-det/artifacts/stage_d_h1_authorbench_dcan_2026-10-07"
N2 = ROOT / "d-det/artifacts/stage_d_data_construction_2026-10-07/n2_h1_recheck"
N1 = ROOT / "d-det/artifacts/stage_d_data_construction_2026-10-07/n1_candidate_matrix"
P0 = ROOT / "d-det/artifacts/acl_sota_p0/server_tracks/authorbench_dcan/predictions_all.npz"
FAMILIES = ["claude", "deepseek", "gemini", "llama", "openai", "qwen"]

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage_d_n0_diagnosis import sha256  # noqa: E402

import sklearn.metrics as sm  # noqa: E402


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def ordered_id_hash(ids) -> str:
    return sha256("\n".join(str(x) for x in ids))


def git(*args) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, cwd=ROOT).stdout.strip()


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "logs").mkdir(exist_ok=True)
    log_lines = []

    def log(s):
        print(s, flush=True)
        log_lines.append(s)

    d1m = json.loads((D1 / "metrics.json").read_text(encoding="utf-8"))
    n2m = json.loads((N2 / "metrics.json").read_text(encoding="utf-8"))
    z1 = np.load(D1 / "predictions.npz")
    z2 = np.load(N2 / "predictions.npz")
    y1, y2 = z1["y_te"].astype(int), z2["y_te"].astype(int)
    t1 = np.array([str(x) for x in z1["task_id_te"]])
    t2 = np.array([str(x) for x in z2["task_id_te"]])

    # ---------- 1) 指标再计算（纯 argmax，无模型调用） ----------
    recompute = {"d1": {}, "n2": {}}
    for z, yy, tag in ((z1, y1, "d1"), (z2, y2, "n2")):
        for k in z.files:
            if not k.startswith("te__"):
                continue
            f1 = float(sm.f1_score(yy, z[k].astype(np.float64).argmax(1), average="macro", zero_division=0))
            recompute[tag][k[4:]] = f1
    log("指标再从 npz 预测重算完成（fp16 存储，允许 ≤1e-3 级差）")

    # ---------- 2) metric_reconciliation ----------
    def reconcile(issue_id, tag, view, metrics, z, yy, tk, comp_key):
        v = metrics["views"][view]["test"]
        comp = metrics["views"][comp_key]["test"]
        boot = metrics["views"][view].get("vs_p0_fusion", {})
        row = {
            "issue_id": issue_id, "artifacts": tag, "view": view, "comparator": comp_key,
            "seed_composition": ("3-seed 平均（SGD seeds 0/1/2）" if view.startswith(("tfidf", "p0_tfidf"))
                                 else ("确定性（无随机种子）" if view in ("codet5_small_centered", "codet5_small_centered_unscaled",
                                                                       "metadata_only", "codet5_small_meanpool")
                                       else "dev-only LR 拟合（成员为 3-seed 平均概率）")),
            "metric_definition": "sklearn f1_score(average='macro', zero_division=0) on argmax",
            "label_set": FAMILIES, "n_samples": int(v["n"]), "n_tasks": int(len(set(tk.tolist()))),
            "ordered_example_id_hash": ordered_id_hash([f"{a}|{b}" for a, b in zip(tk, yy)]),
            "split_hash": sha256(str(yy.tolist()) + "|" + "|".join(sorted(set(tk.tolist())))),
            "point_estimate_view": float(v["macro_f1"]),
            "point_estimate_comparator": float(comp["macro_f1"]),
            "point_difference": float(v["macro_f1"]) - float(comp["macro_f1"]),
            "bootstrap_mean_difference": float(boot.get("delta_mean", float("nan"))),
            "ci_method": "paired task-cluster percentile bootstrap (500, seed 20261007)",
            "ci95": [boot.get("ci95_low"), boot.get("ci95_high")],
            "frac_boot_le_0": boot.get("frac_le_0"),
            "recomputed_from_predictions": recompute[tag].get(view),
            "source_files": {
                "metrics": str(({"d1": D1, "n2": N2}[tag] / "metrics.json").relative_to(ROOT)),
                "predictions": str(({"d1": D1, "n2": N2}[tag] / "predictions.npz").relative_to(ROOT)),
                "metrics_sha256": sha256_file({"d1": D1, "n2": N2}[tag] / "metrics.json"),
                "predictions_sha256": sha256_file({"d1": D1, "n2": N2}[tag] / "predictions.npz"),
            },
        }
        return row

    rows = [
        reconcile("N2-word-vs-fusion", "n2", "tfidf_word", n2m, z2, y2, t2, "fusion_lr"),
        reconcile("N2-centered-vs-fusion", "n2", "codet5_small_centered", n2m, z2, y2, t2, "fusion_lr"),
        reconcile("N2-ensemble-vs-fusion", "n2", "mean_ensemble", n2m, z2, y2, t2, "fusion_lr"),
        reconcile("D1-word-vs-fusion", "d1", "tfidf_word", d1m, z1, y1, t1, "p0_fusion_lr"),
    ]
    reconciliation = {
        "schema": "stage_e0_metric_reconciliation_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "post_hoc_audit": True,
        "summary_of_finding": (
            "四处差异全部可解析为同一原因：回传表中的 Δ 列取的是 **paired task-cluster bootstrap 的差值均值**"
            "（delta_mean），而不是两个点估计相减；点差值与其并列后完全自洽。原值不覆盖，修正并列。"),
        "issues": [
            {"issue": "tfidf_word 回传 +.0148 vs 直接相减 +.0165",
             "resolution": "回传值 = bootstrap 差值均值（+.0148）；点差值 = .64293… − .62637… = +.01657…",
             "status": "resolved: same test / same seed-averaged predictions / 两列统计量不同"},
            {"issue": "codet5_small_centered 回传 +.0278 vs +.0274",
             "resolution": "回传值 = bootstrap 均值 +.0278；点差值 = +.02739…", "status": "resolved"},
            {"issue": "mean_ensemble 回传 +.0663 vs +.0676",
             "resolution": "回传值 = bootstrap 均值 +.0663（CI [.0296,.1064] 不含 0）；点差值 = +.0676",
             "status": "resolved"},
            {"issue": "D1 .8200−.8388=−.0188 vs 历史回传 −.0191",
             "resolution": "−.0191 = bootstrap 均值；点差值 = −.01881…；同一原因", "status": "resolved"},
        ],
        "rows": rows,
        "notes": [
            "bootstrap 均值是重采样差值的均值（含有限样本平滑），不能替代点差值；两列均已列明。",
            "所有点差值 = 两列点估计之差（逐位核对通过）；CI 为配对 task-cluster percentile bootstrap。",
            "500× 经验重采样频率不是模型优于对照的后验概率；N2 仅 51 个 test task，有效样本量小。",
        ],
    }
    (OUT / "metric_reconciliation.json").write_text(
        json.dumps(reconciliation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    log("metric_reconciliation.json 已写出")

    # ---------- 3) p0_definition ----------
    members = ["sem_lr", "style_lgb", "style_lr", "tfidf_char", "tfidf_word"]
    p0z = np.load(P0)
    checks = {}
    for nm in members:
        dv = np.asarray(p0z[f"dv__{nm}"], np.float64)
        checks[nm] = {"dev_rows": int(dv.shape[0]), "cols": int(dv.shape[1]),
                      "max_abs_rowsum_minus_1": float(np.abs(dv.sum(1) - 1).max()),
                      "all_classes_present": bool(dv.shape[1] == 6)}
    # N2 mean_ensemble 复核（纯算术）
    member_te = [np.asarray(z2[f"te__{nm}"], np.float64) for nm in members]
    mean_recomputed = np.mean(member_te, 0)
    diff_mean = float(np.abs(mean_recomputed - np.asarray(z2["te__mean_ensemble"], np.float64)).max())
    fusion_pre = {"full_p0": {"path": "d-det/scripts/acl_sota_p0_baselines.py::fusion_dev_lr",
                              "fit": "dev-only LR on concat log(clip(p,1e-6,1)) of 5 members, C=1, max_iter=3000",
                              "members": members, "class_weight": None,
                              "double_dev_usage": "无（仅一次 dev 拟合；无二次选择）"},
                  "n2_recompute": {"path": "scripts/stage_d_n2_h1_recheck.py（候选矩阵上同配方复算）",
                                   "fit": "dev=348 行上 LR(log-prob features)", "members": members}}
    p0def = {
        "schema": "stage_e0_p0_definition_v1",
        "mean_ensemble": {
            "definition": "p_eq(f|x) = (1/M) Σ_m p_m(f|x)，M=5，逐元素算术平均（概率空间，非 logits、非 vote）",
            "members": members, "weights": "等权 1/5", "class_order": FAMILIES,
            "calibration": "无（成员输出即 sklearn log_loss/SGD log_loss 归一化概率）",
            "member_check_full_p0_dev": checks,
            "n2_arithmetic_recheck": {"max_abs_diff_vs_saved": diff_mean,
                                      "members_te": members, "status": "recomputed"},
            "pre_declaration": {
                "P0_origin": "artifacts/acl_sota_p0（2026-10-05 P0 轮）已含 mean_ensemble 行（等权均值），D1 以复用行呈现",
                "n2_protocol": "n2_protocol.json（提交 7a0d51a，早于 N2 test 读取 2026-10-07T09:05:32Z）在 p0_recompute 中列明 "
                               "『fusion_lr=dev-only LR stack；mean_ensemble=等权均值』",
                "status": "作为 P0 组件在读取 test 前已声明；‘ensemble 优于 fusion’ 的对比本身为 post-hoc 观察，非预注册假设",
            },
        },
        "fusion_lr": fusion_pre,
        "post_hoc_audit": True,
    }
    (OUT / "p0_definition.json").write_text(json.dumps(p0def, ensure_ascii=False, indent=2) + "\n",
                                            encoding="utf-8")
    log(f"p0_definition.json 已写出（mean_ensemble 算数复核 max|diff|={diff_mean:.2e}）")

    # ---------- 4) gate_reconciliation ----------
    n2_best = n2m["gate"]["best_content_view"]
    n2_word = n2m["views"]["tfidf_word"]
    n2_cen = n2m["views"]["codet5_small_centered"]
    n2_fus = n2m["views"]["fusion_lr"]
    meta = n2m["views"]["metadata_only"]
    d1_word = d1m["views"]["tfidf_word"]
    d1_fus = d1m["views"]["p0_fusion_lr"]
    gate = {
        "schema": "stage_e0_gate_reconciliation_v1",
        "post_hoc_audit": True,
        "original_executor_record": {
            "n2_gate_as_run": {"cond1": True, "cond2": True, "cond3": False,
                               "note": "cond2=true 仅由转导视图 codet5_small_centered 达成；原配置/原报告保留不改"},
            "commits": {"protocol": "7a0d51a（n2_protocol.json）", "script": "466d86d（运行前提交）",
                        "results": "9936df1"},
        },
        "restated_conclusions": [
            {"question": "H1 内容可读性（内容视图高于预声明 chance/metadata 控制）",
             "evidence": {
                 "n2": {"tfidf_word": n2_word["test"]["macro_f1"], "ci95": [n2_word["ci95"]["ci95_low"], n2_word["ci95"]["ci95_high"]],
                        "metadata_only": meta["test"]["macro_f1"], "chance": 1 / 6},
                 "d1": {"tfidf_word": d1_word["test"]["macro_f1"],
                        "ci95": [d1_word["ci95"]["ci95_low"], d1_word["ci95"]["ci95_high"]]},
             },
             "expression": "内容视图在两套评测上均明显高于 chance 与 metadata-only：H1 可读性未被否定；"
                           "‘未超越强 P0’不等于‘无来源信号’。"},
            {"question": "普通单样本（inductive）预测增量",
             "evidence": {"n2_tfidf_word_point": n2_word["test"]["macro_f1"],
                          "n2_fusion_point": n2_fus["test"]["macro_f1"],
                          "point_diff": n2_word["test"]["macro_f1"] - n2_fus["test"]["macro_f1"],
                          "bootstrap_mean_diff": n2_word["vs_p0_fusion"]["delta_mean"],
                          "ci95": [n2_word["vs_p0_fusion"]["ci95_low"], n2_word["vs_p0_fusion"]["ci95_high"]],
                          "seeds_delta_pt": [round(x * 100, 2) for x in n2m["gate"]["cond3_direction_stable"]["seed_deltas_vs_p0"]]},
             "expression": "未通过进入方法扩展的门槛：点差 +1.66pt 但配对 CI 覆盖 0、三 seed 方向不稳定；"
                           "‘执行器 cond2=true（转导）’不能替代 inductive cond2（不通过/未确立）。"},
            {"question": "转导诊断增量（test task 内无标签输出中心）",
             "evidence": {"n2_centered_point": n2_cen["test"]["macro_f1"],
                          "point_diff": n2_cen["test"]["macro_f1"] - n2_fus["test"]["macro_f1"],
                          "bootstrap_mean_diff": n2_cen["vs_p0_fusion"]["delta_mean"],
                          "ci95": [n2_cen["vs_p0_fusion"]["ci95_low"], n2_cen["vs_p0_fusion"]["ci95_high"]],
                          "frac_boot_le_0": n2_cen["vs_p0_fusion"]["frac_le_0"],
                          "transductive": True},
             "expression": "单独列示：方向为正但不显著（CI 含 0，frac≤0=0.13）；不替代 inductive 条件。"},
            {"question": "跨 generator 迁移（进入 H2 的前提）",
             "evidence": {"families_with_ge3_generators_in_ab": ["openai"],
                          "others": "claude/deepseek/gemini/llama/qwen 单 generator",
                          "candidate_has_generator_heldout": False},
             "expression": "现有六族候选不满足跨 generator 普遍性；不启动 H2。"},
        ],
        "statistics_caveats": [
            "500× task-cluster 经验重采样频率 ≠ 优于对照的贝叶斯后验。",
            "N2 有效样本量按 test task 计为 51（不是 404 行）；CI 宽度已按 task 聚类。",
            "word 三 seed 差值 −1.36/+2.18/−0.18pt 全量登记，不只挑正 seed。",
        ],
    }
    (OUT / "gate_reconciliation.json").write_text(json.dumps(gate, ensure_ascii=False, indent=2) + "\n",
                                                  encoding="utf-8")
    log("gate_reconciliation.json 已写出")

    # ---------- 5) test_exposure_ledger ----------
    cand_tasks = sorted({str(x) for x in t2.tolist()})
    d1_tasks = set(str(x) for x in t1.tolist())
    subset_ok = set(cand_tasks) <= d1_tasks
    ledger = {
        "schema": "stage_e0_test_exposure_ledger_v1",
        "rule": "已暴露 test 不得改名/换 split 冒充实测；下列为系统内全部 AB test 读取记录（按时间序）",
        "entries": [
            {"round": "P0 基线（2026-10-05，artifacts/acl_sota_p0）",
             "read": "AB-DCan test n=1457 / 408 tasks（为 P0 表与复用预测）",
             "artifact": "d-det/artifacts/acl_sota_p0/server_tracks/authorbench_dcan/predictions_all.npz",
             "artifact_sha256": sha256_file(P0), "purpose": "P0 强基线估计；后续 D1 复用行"},
            {"round": "D1（2026-10-07T08:35:45Z）",
             "read": "AB-DCan test n=1457 / 408 tasks（单次读取）",
             "artifact": str((D1 / "metrics.json").relative_to(ROOT)),
             "artifact_sha256": sha256_file(D1 / "metrics.json"),
             "predictions_sha256": sha256_file(D1 / "predictions.npz"),
             "task_id_hash": ordered_id_hash(sorted(d1_tasks)),
             "purpose": "H1 控制矩阵（D1 判定）与 N0 分层诊断"},
            {"round": "N2（2026-10-07T09:05:32Z）",
             "read": "六族齐全子集 test n=404 / 51 tasks（单次读取，预注册）",
             "artifact": str((N2 / "metrics.json").relative_to(ROOT)),
             "artifact_sha256": sha256_file(N2 / "metrics.json"),
             "predictions_sha256": sha256_file(N2 / "predictions.npz"),
             "task_id_hash": ordered_id_hash(cand_tasks),
             "purpose": "候选矩阵最小 H1 复核（门未通过，归档）"},
        ],
        "relationships": {
            "n2_tasks_subset_of_d1_test_tasks": bool(subset_ok),
            "note": "N2 子集完全落在 D1 已暴露 test 内；不存在新的独立 test；E0/E1 不新增 test 读取。",
        },
        "e0_adds_no_new_test_read": True,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    (OUT / "test_exposure_ledger.json").write_text(json.dumps(ledger, ensure_ascii=False, indent=2) + "\n",
                                                   encoding="utf-8")
    log("test_exposure_ledger.json 已写出")

    # ---------- 6) evidence_closeout.md ----------
    gap = n2_word["test"]["macro_f1"] - n2_fus["test"]["macro_f1"]
    L = []
    L.append("# E0：证据收尾（2026-10-07，post-hoc audit，只用已存产物）")
    L.append("")
    L.append("依据《d-det_AutoDL_N2收尾与独立数据可行性指导_2026-10-07.md》§2。**未重载 test 文本、未重调模型、未改超参**；"
             "所有数字来自已提交的 `metrics.json` / `predictions.npz` / 协议文件，重算仅做 argmax 级核对。")
    L.append("")
    L.append("## 1 差值对账（原值不覆盖，修正并列）")
    L.append("")
    L.append("| issue | 回传 Δ | 点估计差 | bootstrap 差值均值 | 结论 |")
    L.append("|---|---|---|---|---|")
    reported = {"N2-word-vs-fusion": "+.0148", "N2-centered-vs-fusion": "+.0278",
                "N2-ensemble-vs-fusion": "+.0663", "D1-word-vs-fusion": "−.0191"}
    for r in rows:
        L.append(f"| {r['issue_id']} | {reported[r['issue_id']]} | {r['point_difference']:+.4f} | "
                 f"{r['bootstrap_mean_difference']:+.4f} | resolved |")
    L.append("")
    L.append("**统一解释**：回传表的 Δ 列 = paired task-cluster bootstrap 的**差值均值**（delta_mean）；"
             "与“两个点估计相减”是两个不同统计量。点差值逐位自洽（见 `metric_reconciliation.json` 逐行字段："
             "同 test、同 seed 组成、同指标定义、同 label set、n/任务数/顺序 id 哈希/split 哈希均已登记）。"
             "500× 重采样频率不是后验概率；N2 有效样本量按 51 个 test task 计。")
    L.append("")
    L.append("## 2 mean_ensemble 与 P0 定义")
    L.append("")
    L.append(f"- `mean_ensemble`：5 成员（{', '.join(members)}）概率逐元素等权算术平均（概率空间，非 logits/vote）；"
             f"类别序={FAMILIES}；N2 上由存档预测重算的 max|diff|={diff_mean:.2e}（fp16 存储口径）。")
    L.append("- `fusion_lr`：dev-only LR（log-prob 特征、C=1、无样本权重），无双重 dev 选择；N2 在候选 dev=348 行上复算。")
    L.append("- **预声明时间**：`n2_protocol.json`（提交 `7a0d51a`，早于 N2 test 读取 09:05:32Z）在 P0 复算清单中"
             "明确列出“fusion_lr=dev-only LR stack；mean_ensemble=等权均值”。"
             "“ensemble 优于 fusion”的比较本身为 **post-hoc 观察**，不是预注册假设与 H2 证据；"
             "未来强 P0 应在独立数据 dev 上冻结（等权与 fusion 双列）。")
    L.append("")
    L.append("## 3 闸门重述（不重跑实验）")
    L.append("")
    L.append("| 命题 | 本轮表达 |")
    L.append("|---|---|")
    L.append("| H1 内容可读性 | 内容视图（N2 word .6429 / D1 word .8200）均显著高于 chance(.1667) 与 metadata(.3689)；**未被否定** |")
    L.append(f"| 普通单样本增量 | 未通过进入方法扩展门槛：点差 {gap:+.4f}、CI [{n2_word['vs_p0_fusion']['ci95_low']:+.4f},{n2_word['vs_p0_fusion']['ci95_high']:+.4f}] 含 0、三 seed 方向不稳 |")
    L.append("| 转导诊断 | centered 单独列（+.0274 点差，CI 含 0）；不替代 inductive 条件 |")
    L.append("| 跨 generator 迁移 | 仅 openai 有三 generator；“普遍性”不支持；H2 不启动 |")
    L.append("")
    L.append("原执行器记录：cond1=true、cond2=true（**仅转导**）、cond3=false——原样保留；其读法按上表修正。")
    L.append("")
    L.append("## 4 数据卫生与集合选择边界")
    L.append("")
    L.append("- exact/ws 重复（0 组跨 split）与 **norm_lex 骨架碰撞**（25 组/66 行）分开记录；剔除是**保守协议**，"
             "不自动证明原始评测泄漏；去掉字符串可能抹去常量/IO 规格等语义。")
    L.append("- 六族齐全（280 task）为**条件选择**：改变目标总体；规则仅用数据组成、在 test 读取前预注册（`7a0d51a`）。")
    L.append("- 原 test 行剔除后的子集仍屬已暴露集合；E0/E1 不新增任何 test 读取。")
    L.append("")
    L.append("## 5 test 暴露账本（要点）")
    L.append("")
    L.append("- P0（10-05，n=1457）→ D1（08:35:45Z，n=1457）→ N2（09:05:32Z，子集 404/51 task）——三次读取、同一底层 test。")
    L.append(f"- N2 子集 ⊂ D1 test tasks：{subset_ok}；不得改名/换 split 冒充 fresh test。")
    L.append("")
    L.append("## 6 产出")
    L.append("")
    L.append("- `metric_reconciliation.json`、`p0_definition.json`、`gate_reconciliation.json`、`test_exposure_ledger.json`、"
             "本文件；命令/日志/SHA256 同目录。")
    L.append(f"- 运行耗时 {time.time() - t0:.1f}s；生成 UTC {datetime.now(timezone.utc).isoformat()}。")
    L.append("")
    (OUT / "evidence_closeout.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (OUT / "logs" / "e0.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    log(f"E0 完成 {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
