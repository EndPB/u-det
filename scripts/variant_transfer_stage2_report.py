"""stage-2 报告：从 stage2_metrics.json 生成 stage2_report.md。"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
S2 = ROOT / "d-det/artifacts/variant_transfer_stage2_2026-10-08"


def _f(x, nd=3):
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) else str(x)


def main() -> None:
    m = json.loads((S2 / "stage2_metrics.json").read_text(encoding="utf-8"))
    L = []
    L.append("# 变体迁移 stage-2：test 单次评分汇报（2026-10-08）")
    L.append("")
    L.append("> **source_status=server_reconstruction_only**；original_bundle_verified=false；claims_of_byte_identity=forbidden。")
    L.append("> 一次性 test 读取（§4）。此后 test_read_allowed=false；只能对保存分数重算预声明统计。")
    L.append("")
    L.append(f"- 评分行数：{m['n_rows_scored']}（11 成员 × {m['n_test_tasks']} test task）；读数 {len(m['readouts'])} + R2 {len(m['r2_scored'])}")
    L.append(f"- 读数显示名：metadata_only=`code_layout_control`；size_length_only=`oracle_size_length_control`；arms：size_mix（主）/ member-size-matched（length-unweighted）")
    L.append(f"- 统计：bootstrap 500（seed 20261008，共享抽样序列贯穿所有折/臂）；汇总先系列内折等权、再系列间等权；task-macro 含 multiplicity 修正")
    L.append("")

    L.append("## 1. 汇总（heldout_test 迁移读出；mean [95% CI]，跨任务重采样）")
    L.append("")
    L.append("| 读出 | heldout AUROC | heldout task-macro | seen AUROC | Δ(heldout−seen) AUROC |")
    L.append("|---|---|---|---|---|")
    for r in m["readouts"]:
        s = m["summary"][r]
        L.append(f"| {r} | {_f(s['heldout']['auroc']['mean'])} [{_f(s['heldout']['auroc']['ci95'][0])},{_f(s['heldout']['auroc']['ci95'][1])}] "
                 f"| {_f(s['heldout']['task_macro']['mean'])} [{_f(s['heldout']['task_macro']['ci95'][0])},{_f(s['heldout']['task_macro']['ci95'][1])}] "
                 f"| {_f(s['seen']['auroc']['mean'])} | {_f(s['delta']['auroc']['mean'])} [{_f(s['delta']['auroc']['ci95'][0])},{_f(s['delta']['auroc']['ci95'][1])}] |")
    L.append("")
    L.append("## 2. R2 radial score（positive-standardized；heldout 域）")
    L.append("")
    L.append("| 表示 | AUROC | task-macro |")
    L.append("|---|---|---|")
    for rep in m["r2_scored"]:
        k = list(m["folds"])[0]
        # 汇总（单折示例表在 §3；此处给出跨折 mean）
        vals = [m["folds"][fk]["r2"][rep]["heldout"]["point"]["auroc"] for fk in m["folds"]]
        tms = [m["folds"][fk]["r2"][rep]["heldout"]["point"]["task_macro"] for fk in m["folds"]]
        L.append(f"| {rep}（{len(vals)} 折，等权 mean） | {_f(sum(vals)/len(vals))} | {_f(sum(tms)/len(tms))} |")
    L.append("")
    L.append("## 3. 每折 heldout_test AUROC（P0_fusion / P0_equal / best-member 简表）")
    L.append("")
    L.append("| 折 | P0_fusion | P0_equal | size_length(ctrl) | metadata(ctrl) |")
    L.append("|---|---|---|---|---|")
    for fk, f in m["folds"].items():
        rd = f["readouts"]
        L.append(f"| {fk.replace('::', ' / ')} | {_f(rd['P0_fusion']['heldout']['point']['auroc'])} "
                 f"[{_f(rd['P0_fusion']['heldout']['ci95']['auroc'][0])},{_f(rd['P0_fusion']['heldout']['ci95']['auroc'][1])}] "
                 f"| {_f(rd['P0_equal']['heldout']['point']['auroc'])} "
                 f"| {_f(rd['size_length_only']['heldout']['point']['auroc'])} "
                 f"| {_f(rd['metadata_only']['heldout']['point']['auroc'])} |")
    L.append("")
    L.append("## 4. 控制与配对差（P0_fusion 对照；heldout 域；k=22 折分布）")
    L.append("")
    L.append("| 对照 | ΔAUROC 均值 [min,max] | Δtask-macro 均值 [min,max] | frac≤0（均值） |")
    L.append("|---|---|---|---|")
    for ctrl in ("P0_equal", "size_length_only", "metadata_only"):
        das = [m["folds"][fk]["control_deltas"][ctrl]["auroc"]["delta_mean"] for fk in m["folds"]]
        dms = [m["folds"][fk]["control_deltas"][ctrl]["task_macro"]["delta_mean"] for fk in m["folds"]]
        f0 = [m["folds"][fk]["control_deltas"][ctrl]["auroc"]["frac_le_0"] for fk in m["folds"]]
        L.append(f"| fusion−{ctrl} | {_f(sum(das)/len(das))} [{_f(min(das))},{_f(max(das))}] "
                 f"| {_f(sum(dms)/len(dms))} [{_f(min(dms))},{_f(max(dms))}] | {_f(sum(f0)/len(f0))} |")
    L.append("")
    L.append("## 5. 分层（heldout 域，P0_fusion，size_mix；仅解释）")
    L.append("")
    L.append("| 折 | in_hard n/pos/AUROC | not_hard n/pos/AUROC |")
    L.append("|---|---|---|")
    for fk, layer in m["stratification"].items():
        h = layer["in_hard"]; nh = layer["not_hard"]
        L.append(f"| {fk.replace('::', ' / ')} | {h['n']}/{h['pos']}/{_f(h['auroc'])} | {nh['n']}/{nh['pos']}/{_f(nh['auroc'])} |")
    L.append("")
    L.append("## 6. support_gap（匹配负集 vs heldout 尺寸；解释用）")
    L.append("")
    L.append("| 折 | heldout size(B) | neg min|Δlog10| |")
    L.append("|---|---|---|")
    for fk, sg in m["support_gap"].items():
        if "size_matched" in fk:
            L.append(f"| {fk.replace('::', ' / ')} | {sg['heldout_size_B']} | {sg['neg_min_abs_log10_gap']} |")
    L.append("")
    L.append("## 7. 边界")
    L.append("")
    L.append("- arm=不同负域下的敏感性，不是同样本读出对比；AP 需结合正类率（见 metrics JSON 的 pos_rate）。")
    L.append("- CI 仅反映固定这些模型下的任务变异；bootstrap 频率非后验概率；不挑显著折扩写结论。")
    L.append("- 论文表述上限：固定公开 BigCodeBench 协议下的官方模型系列内尺寸变体迁移；保留模板/清洗/污染未知限制。")
    L.append(f"- 特征哈希：{m['features_hashes']}；预测哈希：{m['predictions_hashes']}")
    (S2 / "stage2_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("wrote stage2_report.md")


if __name__ == "__main__":
    main()
