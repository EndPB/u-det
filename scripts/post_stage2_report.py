"""post-stage2 报告生成：intervention_report.md + intervention_manifest.json。

读取：transforms_audit.json、transforms_rules.json、a0_replay_check.json、
      metrics_readout1_{tid}.json、metrics_readout2_{tid}.json、paired_delta.json、
      backbone_availability.json、execution_switches.json、source_status.json。
输出：intervention_report.md（中文）、intervention_manifest.json（回传清单）。
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import variant_transfer_stage1_execute as s1  # noqa: E402
import post_stage2_eval as ev  # noqa: E402

OUT = ROOT / "d-det/artifacts/post_stage2_intervention_2026-10-08"
TIDS = ("A1_format_norm", "A2_comments_masked", "A3_literals_masked")
READOUTS = ev.READOUTS


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def series_equal_mean(vals_by_fold):
    per = {}
    for fk, v in vals_by_fold.items():
        per.setdefault(fk.split("::")[0], []).append(v)
    return float(np.mean([np.mean(v) for v in per.values()])), {s: float(np.mean(v)) for s, v in per.items()}


def main():
    samples, folds = ev.build_fold_arrays(None)
    ref = np.load(ROOT / "d-det/artifacts/variant_transfer_stage1b_2026-10-08/dev_scores_new.npz")
    audit = load(OUT / "transforms_audit.json")
    rules = load(OUT / "transforms_rules.json")
    a0 = load(OUT / "a0_replay_check.json")
    bb = load(OUT / "backbone_availability.json")
    sw = load(OUT / "execution_switches.json")
    paired = load(OUT / "paired_delta.json")

    # 原始 dev 点值（每折每读出）
    orig = {}
    for fk, fd in folds.items():
        for name in READOUTS:
            s = np.asarray(ref[f"{fk}::{name}"], dtype=np.float64)
            if name in ("sem_base", "sem_small"):
                s = s.astype(np.float32)
            orig.setdefault(name, {})[fk] = s1._auc(fd["y_dv"], s)

    # 读入 r1/r2 metrics
    m1 = {t: load(OUT / f"metrics_readout1_{t}.json") for t in TIDS if (OUT / f"metrics_readout1_{t}.json").exists()}
    m2 = {t: load(OUT / f"metrics_readout2_{t}.json") for t in TIDS if (OUT / f"metrics_readout2_{t}.json").exists()}

    def fold_vals(metrics, name):
        out = {}
        for fk, fv in metrics["folds"].items():
            out[fk] = fv["metrics"][name]["auroc"]["point"]
        return out

    L = []
    L.append("# 变体迁移 post-stage2 Phase A 干预矩阵执行报告（2026-10-08）")
    L.append("")
    L.append("> 《d-det_AutoDL_后续干预与主干探针指导_2026-10-08》执行记录。")
    L.append("> `exploratory_train_dev_only=true`；`test_read=false`；`generation=false`；"
             "`source_status=server_reconstruction_only`；不覆盖 stage1b/stage2 产物。")
    L.append("")
    L.append("## 0 摘要")
    L.append("")
    L.append(f"- 变换：A1 格式规范化 / A2 注释屏蔽 / A3 字面量屏蔽（敏感性）；A4=not_executed（指导 §3）。")
    L.append(f"- 全量变换样本 {audit['n_samples']}（11 成员 train+dev；test 未读取）；失败样本全部保留原文。")
    L.append(f"- A0 对照：22 折重放与 stage-1b dev 分数 **max|Δ|={a0['worst_max_abs_diff']:.3e}**（pass≤1e-9={a0['pass_1e-9']}）。")
    L.append(f"- 读出两组：r1=冻结对象 transform/predict；r2=变换文本重训（原规则）。")
    L.append("")

    # 1 变换审计
    L.append("## 1 变换审计（规则 / 失败率 / hash / token）")
    L.append("")
    L.append("| 变换 | fail | compile 破坏 | token 变化 | 字符变化 | 关键计数 |")
    L.append("|---|---|---|---|---|---|")
    for t in TIDS:
        a = audit["transforms"][t]
        tokr = (a["tok_new"] / max(1, a["tok_orig"]) - 1) * 100
        chr_ = (a["chars_new"] / max(1, a["chars_orig"]) - 1) * 100
        cnt = a["counts"]
        if t == "A1_format_norm":
            kc = f"补尾换行 {cnt.get('end_newline_added',0)}；行尾空白行 {cnt.get('trailing_ws_lines',0)}"
        elif t == "A2_comments_masked":
            kc = f"行注释 {cnt.get('comment_line',0)}；删除字符 {cnt.get('removed_chars',0)}"
        else:
            kc = f"字符串 {cnt.get('strings',0)}；数字 {cnt.get('numbers',0)}；行数变化的样本 {a['samples_line_change']}"
        L.append(f"| {t} | {a['n_fail']}/{audit['n_samples']} | {a['samples_compile_fail_after']} | "
                 f"{tokr:+.2f}% | {chr_:+.2f}% | {kc} |")
    L.append("")
    L.append("hash 联合映射与逐样本映射见 `transforms_audit.json`（joint_hash_map_sha256）与 "
             "`local/hash_maps/*.jsonl.gz`（本地）。")
    L.append("")

    # 2 A0 + 读出汇总
    L.append("## 2 dev 汇总（跨折：系列内折等权 → 系列等权；原始 vs 变换）")
    L.append("")
    L.append("| 读出 | 原始 | A1·r1 | A1·r2 | A2·r1 | A2·r2 | A3·r1 | A3·r2 |")
    L.append("|---|---|---|---|---|---|---|---|")
    for name in READOUTS:
        row = [f"| {name} "]
        o_mean, _ = series_equal_mean(orig[name])
        row.append(f"| {o_mean:.4f} ")
        for t in TIDS:
            for m, tag in ((m1, "r1"), (m2, "r2")):
                if t in m:
                    v, _ = series_equal_mean(fold_vals(m[t], name))
                    row.append(f"| {v:.4f} ")
                else:
                    row.append("| - ")
        row.append("|")
        L.append("".join(row))
    L.append("")
    L.append("（r1=冻结对象仅 transform/predict；r2=变换文本上重训。）")
    L.append("")

    # 3 paired delta
    L.append("## 3 paired delta（变换 − 原始；同一 picks 序列）")
    L.append("")
    L.append("### 3.1 跨折均值（AUROC delta，系列内折等权→系列等权）")
    L.append("")
    L.append("| 读出 | A1·r1 | A1·r2 | A2·r1 | A2·r2 | A3·r1 | A3·r2 |")
    L.append("|---|---|---|---|---|---|---|")
    for name in READOUTS:
        row = [f"| {name} "]
        for t in TIDS:
            for rtag in ("r1", "r2"):
                fam = paired.get("transforms", {}).get(t, {}).get(rtag)
                if fam and name in fam.get("summary", {}):
                    pts = {}
                    for fk, fv in fam["folds"].items():
                        if name in fv:
                            pts[fk] = fv[name]["auroc_delta"]["point"]
                    mval, _ = series_equal_mean(pts)
                    row.append(f"| {mval:+.4f} ")
                else:
                    row.append("| - ")
        row.append("|")
        L.append("".join(row))
    L.append("")

    # 3.2 分臂
    L.append("### 3.2 P0_fusion 分负集版本（size_mix / size_matched）")
    L.append("")
    L.append("| 变换×读出 | size_mix Δ | size_matched Δ |")
    L.append("|---|---|---|")
    for t in TIDS:
        for rtag in ("r1", "r2"):
            fam = paired.get("transforms", {}).get(t, {}).get(rtag)
            if not fam:
                continue
            row = {a: {} for a in ("size_mix", "size_matched")}
            for fk, fv in fam["folds"].items():
                arm = fk.split("::")[-1]
                if "P0_fusion" in fv:
                    row[arm][fk] = fv["P0_fusion"]["auroc_delta"]["point"]
            if row["size_mix"]:
                def sm(vals):
                    per = {}
                    for fk, v in vals.items():
                        per.setdefault(fk.split("::")[0], []).append(v)
                    return float(np.mean([np.mean(x) for x in per.values()]))
                L.append(f"| {t}·{rtag} | {sm(row['size_mix']):+.4f} | {sm(row['size_matched']):+.4f} |")
    L.append("")
    L.append("（全读出×逐折 CI 见 `paired_delta.json`。）")
    L.append("")

    # 4 r2 C 选择
    if m2:
        L.append("## 4 r2 C 选择分布（重训在 dev 按原规则）")
        L.append("")
        for t, mt in m2.items():
            cnt = {}
            for fk, fv in mt["folds"].items():
                for k, v in fv.get("C_selected", {}).items():
                    cnt[f"{k}={v}"] = cnt.get(f"{k}={v}", 0) + 1
            top = ", ".join(f"{k}×{v}" for k, v in sorted(cnt.items(), key=lambda x: -x[1])[:6])
            L.append(f"- {t}: {top}")
        L.append("")

    # 5 判读
    L.append("## 5 判读（对照指导 §5 框架）")
    L.append("")
    L.append("- 原对象（r1）与重训（r2）的差值反映**词表/表示不匹配**分量：r2−r1 越大，说明原分类器对表面干预的脆弱性越主要来自表示不匹配而非信号消失。")
    L.append("- r1 与 r2 同时大幅下降且控制读出下降更小 → 支持**表面信号依赖**（H-format 方向）。")
    L.append("- 两者仍保持高分 → 干预后仍有稳定可读成分（继续排除混杂，不得直接称 H-content 成立）。")
    L.append("- 仅 code-layout / size-length 控制高分 → 归为混杂诊断，不推进主干矩阵。")
    L.append("- 所有数字为 train/dev 探索结果；不得写作迁移或 test 证据；不按掉分大小后选主结果。")
    L.append("")

    # 6 Phase B
    L.append("## 6 Phase B 主干可用性")
    L.append("")
    L.append("- encoder-decoder：CodeT5-small（sha `" + bb["encoder_decoder"][0]["weights_sha256"][:16] + "…`）/ "
             "CodeT5-base（sha `" + bb["encoder_decoder"][1]["weights_sha256"][:16] + "…`）可用（现有基线）。")
    L.append("- encoder-only：**not_executed** —— " + bb["encoder_only"]["reason"] + "。")
    L.append("- decoder-only：**not_executed** —— " + bb["decoder_only"]["reason"] + "。")
    L.append("- 若获单独授权的最小设计已写入 `backbone_availability.json`（2×3：架构 × 固定读出 × 原始/A2）。")
    L.append("")

    # 7 边界
    L.append("## 7 边界与合规")
    L.append("")
    L.append("- `test_read=false`（未读取旧 test、未用 test_scores.npz 做任何选择）；`generation=false`；未下载权重。")
    L.append("- 不覆盖 stage1b/stage2 产物；全部新产物位于本目录。")
    L.append("- `source_status=server_reconstruction_only`、`original_bundle_verified=false`、`claims_of_byte_identity=forbidden` 继续保留。")
    L.append("- A3 仅敏感性分析，不得表述为\"去风格后的纯代码\"；A2 输出不得解读为因果后训练信号。")
    L.append("")

    (OUT / "intervention_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    # manifest
    def sha256_file(p: Path) -> str:
        h = hashlib.sha256()
        with p.open("rb") as f:
            for b in iter(lambda: f.read(1 << 20), b""):
                h.update(b)
        return h.hexdigest()

    files = []
    for p in sorted(OUT.rglob("*")):
        if p.is_file() and "local" not in p.parts and p.name != "SHA256SUMS.txt":
            files.append({"path": str(p.relative_to(OUT)), "bytes": p.stat().st_size,
                          "sha256": sha256_file(p)})
    manifest = {"schema": "post_stage2_intervention_manifest_v1",
                "generated_utc": datetime.now(timezone.utc).isoformat(),
                "exploratory_train_dev_only": True, "test_read": False, "generation": False,
                "weights_downloaded": False,
                "transforms": {t: {"fail": audit["transforms"][t]["n_fail"],
                                   "joint_hash_map_sha256": audit["transforms"][t]["joint_hash_map_sha256"],
                                   "rules": rules["transforms"][t]["rules"]} for t in TIDS},
                "a4_status": audit["a4_status"],
                "readouts": {"r1": list(m1.keys()), "r2": list(m2.keys())},
                "phase_b": {"encoder_decoder": "available (codet5-small/base)",
                            "encoder_only": "not_executed", "decoder_only": "not_executed"},
                "files": files}
    (OUT / "intervention_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1),
                                                    encoding="utf-8")
    print("wrote intervention_report.md +", len(files), "files in manifest")


if __name__ == "__main__":
    main()
