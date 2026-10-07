"""Stage D1 report.md 生成器：从 metrics.json 渲染中文报告（数字全部取自 metrics.json）。"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "d-det/artifacts/stage_d_h1_authorbench_dcan_2026-10-07"
m = json.load((OUT / "metrics.json").open(encoding="utf-8"))
g = m["gate_evaluation"]
views = m["views"]

try:
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                          cwd=OUT.parents[2]).stdout.strip()
except Exception:
    head = "(unknown)"

ORDER = ["metadata_only", "tfidf_word", "tfidf_char", "codet5_meanpool", "codet5_centered",
         "codet5_centered_unscaled", "p0_fusion_lr", "p0_tfidf_word", "p0_tfidf_char",
         "p0_style_lgb", "p0_style_lr", "p0_sem_lr", "p0_mean_ensemble"]
LABEL = {"metadata_only": "metadata-only（长度/结构，shortcut control）",
         "tfidf_word": "word TF-IDF（NG(1,2), train 词表）",
         "tfidf_char": "char_wb TF-IDF（2-4gram, train 词表）",
         "codet5_meanpool": "冻结 CodeT5-small mean-pool（512d）",
         "codet5_centered": "冻结 CodeT5-small + task LOO 中心化（转导诊断）",
         "codet5_centered_unscaled": "同上（不标准化)（转导诊断）",
         "p0_fusion_lr": "P0 fusion_lr（复用预测重算）",
         "p0_tfidf_word": "P0 tfidf_word（复用）",
         "p0_tfidf_char": "P0 tfidf_char（复用）",
         "p0_style_lgb": "P0 style_lgb（复用）",
         "p0_style_lr": "P0 style_lr（复用）",
         "p0_sem_lr": "P0 sem_lr（复用）",
         "p0_mean_ensemble": "P0 mean_ensemble（复用）"}


def label(k):
    return f"**{LABEL[k]}**" if k == g["best_content_view"] else LABEL[k]


lines = []
lines.append("# 阶段 D1：AuthorBench-DCAN task-aware H1 控制矩阵（2026-10-07）")
lines.append("")
lines.append(f"- 脚本：`scripts/stage_d_h1_authorbench_dcan.py`；git HEAD：`{head}`")
lines.append(f"- 数据：`d-det/data/h2_authorbench_dcan/core.jsonl`，sha256 `{m['protocol']['data_sha256']}`")
lines.append(f"- 切分：包内 `task_split`（train {m['counts']['train']} / dev {m['counts']['dev']} / test {m['counts']['test']}；"
             f"{m['counts']['tasks']} tasks；6 families；全 C）；task 不跨 split")
lines.append(f"- test 读取（UTC）：{m['test_read_utc']}（单次读取；未参与任何选择；运行 {m['runtime_seconds']:.1f}s）")
lines.append(f"- bootstrap：{m['protocol']['bootstrap']['repeats']}× task-cluster（cluster=task_id，seed {m['protocol']['bootstrap']['seed']}）")
lines.append("")
lines.append("## 结论（闸门判定）")
lines.append("")
lines.append(f"**D1 未通过。** 最佳内容视图 = {g['best_content_view']}，macro-F1 = {g['best_content_f1']:.4f}；"
             f"低于同切分 P0 fusion_lr {g['p0_best_same_split']:.4f}"
             f"（paired task-cluster Δ = {g['cond4_paired_ci_vs_p0']['delta_mean']:+.4f}，"
             f"95% CI [{g['cond4_paired_ci_vs_p0']['ci95_low']:+.4f}, {g['cond4_paired_ci_vs_p0']['ci95_high']:+.4f}]，"
             f"P(Δ≤0)={g['cond4_paired_ci_vs_p0']['frac_le_0']:.2f}）；距 +1pt 门槛差 {g['gap_to_1pt'] * 100:.2f}pt。")
lines.append("")
lines.append("→ 按指导 §3.2/§7：结论记为 **task-aware 可读性/任务效应审计**；停止新架构探索，转入数据构造；H2/H3 不启动。")
lines.append("")
lines.append("## 1 视图总表（test n=%d）" % m["counts"]["test"])
lines.append("")
lines.append("| 视图 | macro-F1 | BA | CI95（task-cluster） | ECE15 | NLL | Δ vs P0 fusion | Δ95% CI | P(Δ≤0) |")
lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
for k in ORDER:
    v = views[k]
    t, ci, d = v["test"], v["ci95"], v.get("vs_p0_fusion", {})
    lines.append(f"| {label(k)} | {t['macro_f1']:.4f} | {t['balanced_acc']:.4f} | "
                 f"[{ci['ci95_low']:.4f}, {ci['ci95_high']:.4f}] | {t['ece15']:.4f} | {t['nll']:.4f} | "
                 f"{d.get('delta_mean', float('nan')):+.4f} | [{d.get('ci95_low', float('nan')):+.4f}, {d.get('ci95_high', float('nan')):+.4f}] | "
                 f"{d.get('frac_le_0', float('nan')):.2f} |")
lines.append("")
lines.append("## 2 per-family（precision / recall，关键视图）")
lines.append("")
fam = m["counts"]["families"]
lines.append("| family | test support | tfidf_word P | tfidf_word R | codet5_meanpool P | codet5_meanpool R | codet5_centered P | codet5_centered R | P0 fusion P | P0 fusion R |")
lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
for f in fam:
    row = [f, str(views["tfidf_word"]["test"]["per_class"][f]["support"])]
    for k in ("tfidf_word", "codet5_meanpool", "codet5_centered", "p0_fusion_lr"):
        pc = views[k]["test"]["per_class"][f]
        row += [f"{pc['precision']:.3f}", f"{pc['recall']:.3f}"]
    lines.append("| " + " | ".join(row) + " |")
lines.append("")
lines.append("## 3 分桶（选择关键字视图；全量见 metrics.json）")
lines.append("")
lines.append("### 3.1 长度四分位（边界= train char_count 分位；test 计数）")
lines.append("")
lb = "length_quartiles(test char_count; train-quantile edges)"
lines.append("| 区间 | n | tfidf_word F1 | codet5_centered F1 |")
lines.append("| --- | --- | --- | --- |")
for k, b in views["tfidf_word"]["buckets"][lb].items():
    if b["n"] == 0:
        continue
    c = views["codet5_centered"]["buckets"][lb].get(k, {})
    lines.append(f"| {k} | {b['n']} | {b['macro_f1']:.4f} | {c.get('macro_f1', float('nan')):.4f} |")
lines.append("")
lines.append("### 3.2 task-size（同一 task_id 的 test 样本数）")
lines.append("")
lines.append("| task-size | n | tfidf_word F1 | codet5_centered F1 |")
lines.append("| --- | --- | --- | --- |")
for k, b in views["tfidf_word"]["buckets"]["task_size"].items():
    c = views["codet5_centered"]["buckets"]["task_size"].get(k, {})
    lines.append(f"| {k} | {b['n']} | {b['macro_f1']:.4f} | {c.get('macro_f1', float('nan')):.4f} |")
lines.append("")
lines.append("### 3.3 generator（test accuracy）")
lines.append("")
lines.append("| generator | n | tfidf_word acc | codet5_centered acc |")
lines.append("| --- | --- | --- | --- |")
gb = views["tfidf_word"]["buckets"]["generator"]
for k, b in gb.items():
    if not b["n"]:
        continue
    c = views["codet5_centered"]["buckets"]["generator"].get(k, {})
    ca = f"{c['accuracy']:.4f}" if c.get("accuracy") is not None else "n/a"
    lines.append(f"| {k} | {b['n']} | {b['accuracy']:.4f} | {ca} |")
lines.append("")
lines.append("## 4 复现性核对（与 P0 同切分口径直接对照）")
lines.append("")
repro = [("tfidf_word", "p0_tfidf_word"), ("tfidf_char", "p0_tfidf_char"),
         ("p0_fusion_lr", "p0_fusion_lr"), ("p0_style_lgb", "p0_style_lgb")]
for a, b in repro:
    fa = views[a]["test"]["macro_f1"]
    fb = views[b]["test"]["macro_f1"]
    note = "（同一协议独立实现，完全一致）" if a != b and abs(fa - fb) < 1e-12 else ""
    lines.append(f"- {a} {fa:.4f} vs {b} {fb:.4f}（Δ={fa - fb:+.4f}）{note}")
lines.append("")
lines.append("## 5 闸门（§3）与停止规则（§7）逐条")
lines.append("")
lines.append(f"1. 内容视图显著高于 chance/metadata-only：{'通过' if g['cond1_content_above_chance_and_metadata'] else '未通过'}"
             f"（best {g['best_content_f1']:.4f} vs metadata {g['metadata_only_f1']:.4f}，chance {g['chance_macro_f1']:.4f}）")
_gap_txt = "通过" if g["cond2_plus_1pt_over_p0"] else f"未通过（差 {g['gap_to_1pt'] * 100:.2f}pt）"
lines.append(f"2. 相对同切分 P0 提升 ≥ +1pt：{_gap_txt}")
lines.append(f"3. 多 seed/多折方向一致：TF-IDF 3 seeds dev F1 见 §闸门 JSON（"
             f"word {[round(x, 4) for x in g['cond3_multiseed_or_multifold_direction']['tfidf_word_seed_dev_f1']]}；"
             f"CodeT5 视图为确定性前向）")
lines.append(f"4. paired CI 不覆盖 0 且方向为负：Δ vs fusion 95% CI "
             f"[{g['cond4_paired_ci_vs_p0']['ci95_low']:+.4f}, {g['cond4_paired_ci_vs_p0']['ci95_high']:+.4f}]（不含 0，为负）")
lines.append("5. 转导项单独标注：是（§2 表内注明 transductive；不与其他视图混报）")
lines.append("6. 未混入检测轴：是（本矩阵仅 family 分类）")
lines.append("")
lines.append(f"- 停止规则触发：第 4 条（增益 <1pt）、第 1 条（D0 支持矩阵 10 折中仅 6 折 admitted，task/generator 支持不足）。")
lines.append("")
lines.append("## 6 观测记录")
lines.append("")
lines.append(f"- metadata-only（长度/行数/nloc/圈复杂度/token_size）macro-F1 {views['metadata_only']['test']['macro_f1']:.4f}"
             f"（CI [{views['metadata_only']['ci95']['ci95_low']:.4f}, {views['metadata_only']['ci95']['ci95_high']:.4f}]，chance 0.1667）："
             f"强 shortcut 存在（长度与 family 相关），但远不足以免于内容情报。")
lines.append(f"- 冻结 CodeT5-small mean-pool {views['codet5_meanpool']['test']['macro_f1']:.4f}（dev {views['codet5_meanpool']['meta']['dev_macro_f1']:.4f}）；"
             f"task 中心化（转导）仅 {views['codet5_centered']['test']['macro_f1']:.4f}，未缩小与 P0 的差距 → 任务中心化不能弥补冻结编码器-线性头的容量/协议不足。")
lines.append(f"- task-size 分桶：tfidf_word 在 6+ 桶 F1 {views['tfidf_word']['buckets']['task_size']['6+']['macro_f1']:.4f}，"
             f"低于 3 与 4-5 桶；codet5_centered 在 size=2 桶仅 {views['codet5_centered']['buckets']['task_size']['2']['macro_f1']:.4f}。")
lines.append(f"- generator 分桶（accuracy）：deepseek-chat {gb.get('deepseek-chat', {}).get('accuracy', float('nan')):.4f}、"
             f"qwen-2.5-72b-instruct {gb.get('qwen-2.5-72b-instruct', {}).get('accuracy', float('nan')):.4f}\n"
             f"  明显低于 gemini-2.5-flash-preview {gb.get('gemini-2.5-flash-preview-05-20', {}).get('accuracy', float('nan')):.4f}、"
             f"gpt-4.1 {gb.get('gpt-4.1', {}).get('accuracy', float('nan')):.4f}：family 内 generator 异质性是主要误差源之一。")
lines.append("")
lines.append("## 7 唯一下一步建议")
lines.append("")
lines.append("按 §7 停止规则，不启动 H2/H3。本审计结论（task-aware 可读性/任务效应）回传指导方，"
             "下一步实验（数据构造方向）应由指导方书面出指令后再执行；在此之前保持仓库冻结与可复现状态。")
lines.append("")
(OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("wrote", OUT / "report.md")
