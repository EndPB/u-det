"""variant transfer stage-1：从 train_dev_metrics.json 生成 markdown 报告。

输出：d-det/artifacts/variant_transfer_stage1_2026-10-08/train_dev_report.md
（所有内容携带 source_status=server_reconstruction_only 标注；test 未读取）
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "d-det/artifacts/variant_transfer_stage1_2026-10-08"
READOUTS = ["tfidf_char", "tfidf_word", "sem_base", "sem_small", "style_lr", "style_lgb",
            "metadata_only", "size_length_only", "P0_fusion", "P0_equal"]


def fmt_ci(m):
    return f"{m['point']:.3f} [{m['ci95_low']:.3f}, {m['ci95_high']:.3f}]"


def main() -> None:
    d = json.loads((P / "train_dev_metrics.json").read_text(encoding="utf-8"))
    folds = d["folds"]
    L = []
    L.append("# 变体迁移 stage-1（train/dev）汇报（2026-10-08）")
    L.append("")
    L.append("> **source_status = server_reconstruction_only**；original_bundle_verified = false；")
    L.append("> claims_of_byte_identity = forbidden（依据《服务器重构版最小授权与补传豁免》§1）。")
    L.append("> 本报告全部为 **train/dev** 结果：dev 用于预声明选择与诊断，**test 未读取**。")
    L.append("")
    L.append(f"- 运行：22 折（11 heldout × 2 负集版本），runtime {d['runtime_seconds']:.0f}s")
    L.append(f"- 配置 hash：{json.dumps(d['config_hashes'], ensure_ascii=False)}")
    L.append(f"- 协议：{d['protocol']['dev_selection']}；{d['protocol']['metrics']}")
    L.append("")

    # ---- 表 1：P0 融合/等权 vs 最佳单成员（按负集版本分组）----
    L.append("## 1. 主要读出（dev AUROC [95% task-cluster bootstrap CI]）")
    L.append("")
    L.append("| 折 | 版本 | P0_fusion | P0_equal | best single | best 名称 | metadata_only | size_length_only |")
    L.append("|---|---|---|---|---|---|---|---|")
    for key, f in folds.items():
        m = f["metrics"]
        singles = {k: m[k]["auroc"]["point"] for k in READOUTS
                   if k not in ("P0_fusion", "P0_equal")}
        best_name = max(singles, key=lambda k: (singles[k] if singles[k] == singles[k] else -1))
        L.append(f"| {key.replace('::', ' / ')} | | {fmt_ci(m['P0_fusion']['auroc'])} | "
                 f"{fmt_ci(m['P0_equal']['auroc'])} | {singles[best_name]:.3f} | {best_name} | "
                 f"{fmt_ci(m['metadata_only']['auroc'])} | {fmt_ci(m['size_length_only']['auroc'])} |")
    L.append("")

    # ---- 表 2：size_mix vs size_matched ----
    L.append("## 2. 负集版本敏感性（P0_fusion dev AUROC）")
    L.append("")
    L.append("| 折 | size_mix | size_matched | Δ(matched−mix) |")
    L.append("|---|---|---|---|")
    for key in folds:
        if not key.endswith("::size_mix"):
            continue
        km = key[: -len("::size_mix")] + "::size_matched"
        a = folds[key]["metrics"]["P0_fusion"]["auroc"]["point"]
        b = folds[km]["metrics"]["P0_fusion"]["auroc"]["point"]
        L.append(f"| {key.replace('::', ' / ')} | {a:.3f} | {b:.3f} | {b - a:+.3f} |")
    L.append("")

    # ---- 表 3：R2 ----
    L.append("## 3. R2 表示迁移（dev AUROC；系列中心距离）")
    L.append("")
    L.append("| 折 | base euclid | base cosine | small euclid | small cosine | rank-corr(base euclid, log10 size) |")
    L.append("|---|---|---|---|---|---|")
    for key, f in folds.items():
        r2 = f["r2"]
        L.append(f"| {key.replace('::', ' / ')} | {r2['codet5_base']['euclid']['auroc']['point']:.3f} | "
                 f"{r2['codet5_base']['cosine']['auroc']['point']:.3f} | "
                 f"{r2['codet5_small']['euclid']['auroc']['point']:.3f} | "
                 f"{r2['codet5_small']['cosine']['auroc']['point']:.3f} | "
                 f"{r2['codet5_base']['rank_corr_train_dist_vs_logsize']['euclid']:+.3f} |")
    L.append("")

    # ---- 汇总 ----
    import statistics as st
    fus = [folds[k]["metrics"]["P0_fusion"]["auroc"]["point"] for k in folds]
    eq = [folds[k]["metrics"]["P0_equal"]["auroc"]["point"] for k in folds]
    L.append("## 4. 汇总")
    L.append("")
    L.append(f"- P0_fusion dev AUROC：min {min(fus):.3f} / median {st.median(fus):.3f} / max {max(fus):.3f}（22 折）")
    L.append(f"- P0_equal dev AUROC：min {min(eq):.3f} / median {st.median(eq):.3f} / max {max(eq):.3f}")
    for series in ("CodeLlama-Instruct", "Qwen2.5-Coder-Instruct", "DeepSeek-Coder-v1-Instruct"):
        s_fus = [folds[k]["metrics"]["P0_fusion"]["auroc"]["point"] for k in folds if k.startswith(series)]
        L.append(f"- {series}：P0_fusion 折内范围 {min(s_fus):.3f} – {max(s_fus):.3f}")
    L.append("")
    L.append("## 5. 解释边界（预声明）")
    L.append("")
    L.append("- dev=见尺寸+预声明负集（不含 heldout 变体）；heldout 迁移信号只能由后续 **单独授权的一次性 test read** 评估。")
    L.append("- dev 同时是选择集（C 网格 / SGD best-dev），上述数字为选择内估计，不得当作未见迁移证据。")
    L.append("- 不得表述为：复现本机原件、unseen independent family、后训练因果、无污染确认性 benchmark。")
    (P / "train_dev_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("wrote train_dev_report.md")


if __name__ == "__main__":
    main()
