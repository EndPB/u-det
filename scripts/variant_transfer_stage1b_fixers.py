"""stage-1b 修复器（§2.2/§2.3/§2.4/§2.5）。

产出（variant_transfer_stage1b_2026-10-08/）：
- task_macro_ci_fix.json：修正 multiplicity 后的 task-macro CI（用已存 dev 预测重算，不重拟合），
  含 [a1,a1,a2]→(2a1+a2)/3 验证与两种独立实现对照；
- dev_paired_controls.json：P0_fusion vs size_length/metadata 的同样本 paired boot（AUROC/task-macro）；
- r2_correction.json：cosine=invalid_zero_center；euclid 改名 positive_standardized_radial_score；
  base/small euclid 的完整范围（原报告 .536–.65 仅前段，现给全量并保留原表）；
- implementation_deviation.json：§2.1–§2.5 五项偏差与处理；
- test_exposure_ledger_v2.json：分级暴露账本（raw_scan/transient_parse/feature_use/human_inspection/metric_evaluation）；
- prereg_amendment_length_weighting_deferred_2026-10-08.json：长度加权臂 B 方案（时间+理由）；
- readout_display_names.json：显示名/别名变更（旧 key 保留）。
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import variant_transfer_stage1_execute as s1  # noqa: E402

OUT = ROOT / "d-det/artifacts/variant_transfer_stage1b_2026-10-08"
S1 = ROOT / "d-det/artifacts/variant_transfer_stage1_2026-10-08"
BOOT_SEED = 20261008
READOUTS10 = ("tfidf_char", "tfidf_word", "sem_base", "sem_small", "style_lr", "style_lgb",
              "metadata_only", "size_length_only", "P0_fusion", "P0_equal")


def task_terms(y, s, tasks):
    uniq, inv = np.unique(tasks, return_inverse=True)
    a = np.empty(len(uniq))
    for g in range(len(uniq)):
        sel = inv == g
        yy = y[sel]
        a[g] = s1._fast_auc(yy, s[sel]) if 0 < yy.sum() < sel.sum() else np.nan
    return uniq, a


def main() -> None:
    t0 = time.time()
    mj = json.loads((S1 / "train_dev_metrics.json").read_text(encoding="utf-8"))
    dsn = np.load(OUT / "dev_scores_new.npz", allow_pickle=True)
    folds = list(mj["folds"])

    # ---------- §2.3 数值验证 ----------
    # [a1,a1,a2] 抽中 -> (2*a1+a2)/3；与 draw-id 展开实现对照
    a_demo = {"t1": 0.8, "t2": 0.5}
    pick_demo = ["t1", "t1", "t2"]
    m_seq = float(np.mean([a_demo[t] for t in pick_demo]))
    counts = {t: pick_demo.count(t) for t in set(pick_demo)}
    m_drawid = float(sum(counts[t] * a_demo[t] for t in counts) / len(pick_demo))
    assert abs(m_seq - (2 * 0.8 + 0.5) / 3) < 1e-15
    assert abs(m_seq - m_drawid) < 1e-15
    # 去重平均（旧实现）对照
    m_wrong = float(np.mean([a_demo[t] for t in set(pick_demo)]))
    multiplicity_validation = {
        "case": "[a1,a1,a2]",
        "expected": (2 * 0.8 + 0.5) / 3,
        "sequence_impl": m_seq,
        "drawid_weighted_impl": m_drawid,
        "old_dedup_impl": m_wrong,
        "dedup_differs": abs(m_wrong - m_seq) > 1e-12,
        "pass": abs(m_seq - (2 * 0.8 + 0.5) / 3) < 1e-15 and abs(m_seq - m_drawid) < 1e-15,
    }

    # ---------- 修正 task-macro CI（22 折 × 10 读出） ----------
    tm_fix = {}
    cmp_rows = []
    for fold_key in folds:
        meta = dsn[f"{fold_key}::__meta__"]
        keys = [tuple(str(x).split("|")) for x in meta]
        y = np.array([int(k[2]) for k in keys])
        tasks = np.array([k[1] for k in keys])
        uniq, inv = np.unique(tasks, return_inverse=True)
        # 共用同 seed 的原 rng 生成 pick 序列（与原实现相同的调用：每读出单独 rng）
        fold_out = {}
        for readout in READOUTS10:
            s = dsn[f"{fold_key}::{readout}"].astype(np.float64)
            if readout in ("sem_base", "sem_small"):
                s = s.astype(np.float32)  # 沿原 float32 路径（rankdata float32 秩；与 stage-1 一致）
            uniq_t, a_t = task_terms(y, s, tasks)
            rng = np.random.default_rng(BOOT_SEED)
            fixed = np.empty(500)
            old_dedup = np.empty(500)
            for k in range(500):
                pick = rng.choice(len(uniq_t), size=len(uniq_t), replace=True)
                fixed[k] = np.nanmean(a_t[pick])          # multiplicity preserved（§2.3）
                old_dedup[k] = np.nanmean(a_t[np.unique(pick)])  # 旧去重实现（对照）
            old = mj["folds"][fold_key]["metrics"][readout]["task_macro_auroc"]
            fold_out[readout] = {
                "point": old["point"],
                "ci95_fixed": [float(np.nanpercentile(fixed, 2.5)), float(np.nanpercentile(fixed, 97.5))],
                "ci95_old_dedup_bug": [old["ci95_low"], old["ci95_high"]],
                "boot_mean_fixed": float(np.nanmean(fixed)),
            }
            cmp_rows.append({"fold": fold_key, "readout": readout,
                             "low_delta": fold_out[readout]["ci95_fixed"][0] - old["ci95_low"],
                             "high_delta": fold_out[readout]["ci95_fixed"][1] - old["ci95_high"]})
        tm_fix[fold_key] = fold_out
    low_d = [r["low_delta"] for r in cmp_rows]
    high_d = [r["high_delta"] for r in cmp_rows]
    out = {
        "schema": "variant_transfer_task_macro_ci_fix_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "note": "仅修正 task-macro 的 bootstrap 重数（M* = T^-1 Σ a_{t_j*},含重复抽中）；整体 AUROC/AP 本就保留行重数，不变；用已存 dev 预测重算，不重拟合",
        "validation": multiplicity_validation,
        "ci_fix": tm_fix,
        "summary": {"n_cells": len(cmp_rows),
                    "ci_low_delta_mean": float(np.mean(low_d)), "ci_low_delta_max_abs": float(np.max(np.abs(low_d))),
                    "ci_high_delta_mean": float(np.mean(high_d)), "ci_high_delta_max_abs": float(np.max(np.abs(high_d)))},
    }
    (OUT / "task_macro_ci_fix.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("task_macro_ci_fix done; validation:", multiplicity_validation["pass"])

    # ---------- §2.5 同样本 paired 控制（fusion vs size_length / metadata） ----------
    paired = {}
    for fold_key in folds:
        meta = dsn[f"{fold_key}::__meta__"]
        keys = [tuple(str(x).split("|")) for x in meta]
        y = np.array([int(k[2]) for k in keys])
        tasks = np.array([k[1] for k in keys])
        uniq, inv = np.unique(tasks, return_inverse=True)
        idx_by = {t: np.where(tasks == t)[0] for t in uniq}
        s_fus = dsn[f"{fold_key}::P0_fusion"].astype(np.float64)
        fold_p = {}
        for ctrl in ("size_length_only", "metadata_only"):
            s_c = dsn[f"{fold_key}::{ctrl}"].astype(np.float64)
            _, a_fus = task_terms(y, s_fus, tasks)
            _, a_c = task_terms(y, s_c, tasks)
            rng = np.random.default_rng(BOOT_SEED)
            d_auc = np.empty(500)
            d_tm = np.empty(500)
            for k in range(500):
                pick = rng.choice(len(uniq), size=len(uniq), replace=True)
                idx = np.concatenate([idx_by[uniq[q]] for q in pick])
                d_auc[k] = s1._auc(y[idx], s_fus[idx]) - s1._auc(y[idx], s_c[idx])
                d_tm[k] = np.nanmean(a_fus[pick]) - np.nanmean(a_c[pick])
            fold_p[ctrl] = {
                "auroc_delta": {"point_diff": s1._auc(y, s_fus) - s1._auc(y, s_c),
                                "delta_mean": float(np.nanmean(d_auc)),
                                "ci95": [float(np.nanpercentile(d_auc, 2.5)), float(np.nanpercentile(d_auc, 97.5))],
                                "frac_le_0": float((d_auc <= 0).mean())},
                "task_macro_delta": {"delta_mean": float(np.nanmean(d_tm)),
                                     "ci95": [float(np.nanpercentile(d_tm, 2.5)), float(np.nanpercentile(d_tm, 97.5))],
                                     "frac_le_0": float((d_tm <= 0).mean())},
            }
        paired[fold_key] = fold_p
    (OUT / "dev_paired_controls.json").write_text(json.dumps({
        "schema": "variant_transfer_dev_paired_controls_v1",
        "note": "同样本 paired bootstrap（同折同读出、任务抽样一致）；修正版 task-macro multiplicity；"
                "‘内容高于控制’仅相对这两个有限控制器，不能排除全部混杂（§2.5）",
        "paired": paired}, ensure_ascii=False, indent=1), encoding="utf-8")
    print("paired controls done")

    # ---------- §2.2 R2 更正 ----------
    eu, co = [], []
    per_rep = {rep: {"euclid": [], "cosine": []} for rep in ("codet5_base", "codet5_small")}
    for fold_key in folds:
        for rep in ("codet5_base", "codet5_small"):
            v_e = mj["folds"][fold_key]["r2"][rep]["euclid"]["auroc"]["point"]
            v_c = mj["folds"][fold_key]["r2"][rep]["cosine"]["auroc"]["point"]
            eu.append(v_e); co.append(v_c)
            per_rep[rep]["euclid"].append(v_e); per_rep[rep]["cosine"].append(v_c)
    (OUT / "r2_correction.json").write_text(json.dumps({
        "schema": "variant_transfer_r2_correction_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "cosine": {"status": "invalid_zero_center",
                   "reason": "正集 scaler 后取正集均值中心 ⇒ c=0，方向不存在；+1e-12 只放大数值残差",
                   "action": "不进入 stage-2；不得把近 0.5 解读为‘无系列信号’"},
        "euclid": {"renamed": "positive_standardized_radial_score",
                   "scope": "仅描述到正集中心的距离；非同一退化问题"},
        "range_correction": {
            "note": "stage-1 回传正文的 .536–.65 不是全量范围（口述前段）；以下为完整范围（全 22 臂）并保留原表；"
                    "按表示分开列出（指导引用的 .4486–.6393 为 codet5_base euclid 范围）",
            "euclid_all": {"n": len(eu), "min": float(min(eu)), "max": float(max(eu))},
            "cosine_all": {"n": len(co), "min": float(min(co)), "max": float(max(co))},
            "per_representation": {rep: {
                "euclid": {"n": len(v["euclid"]), "min": float(min(v["euclid"])), "max": float(max(v["euclid"]))},
                "cosine": {"n": len(v["cosine"]), "min": float(min(v["cosine"])), "max": float(max(v["cosine"]))}}
                for rep, v in per_rep.items()},
            "euclid_values": eu, "cosine_values": co,
        },
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"r2 correction done: euclid [{min(eu):.4f},{max(eu):.4f}] cosine [{min(co):.4f},{max(co):.4f}]")

    # ---------- §2.4 长度加权：B 方案（deferred） ----------
    (OUT / "prereg_amendment_length_weighting_deferred_2026-10-08.json").write_text(json.dumps({
        "schema": "variant_transfer_prereg_amendment_length_weighting_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "created_before_stage2_test_read": True,
        "amends": "r1_negative_set_amendment.json 的 length_quantile_weighting（registered_but_not_executed）",
        "decision": "deferred",
        "reason": "主实验（22 折 R1/R2/控制）已在 length-unweighted 口径下拟合完成；按 §2.4‘不临时追加到已拟合主实验、"
                  "不伪称已控制全部长度效应’，长度加权作为独立后续协议评估，避免在主实验拟合后引入权重实现差异无对照可比。"
                  "本决定在 stage-2 读取之前写死，不会因 test 结果改变。",
        "consequences": ["现有 size_matched 结果命名 member-size-matched / length-unweighted（原 key 保留）",
                         "报告不得声称已控制全部长度效应", "length_weighting_applied=false 登记于 metrics 释义"],
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print("length weighting deferred amendment done")

    # ---------- §2.1/§2.5 偏差与命名 ----------
    (OUT / "implementation_deviation.json").write_text(json.dumps({
        "schema": "variant_transfer_implementation_deviation_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "deviations": [
            {"id": "D1_objects_not_saved_in_stage1",
             "found": "stage-1 脚本仅返回概率，未序列化 TF-IDF/scaler/分类器/SGD epoch/融合头/R2 统计量",
             "action": "stage-1b 确定性重建并保存（同环境/顺序/seed/预算）；dev 重放对齐（|Δ|≤5.1e-7）与点指标核对（≤1e-10）见 replay_check.json",
             "status": "resolved"},
            {"id": "D2_r2_cosine_degenerate",
             "found": "cosine 中心恒为 0（正集标准化后取正集均值）",
             "action": "标记 invalid_zero_center；stage-2 排除；euclid 改名 positive_standardized_radial_score",
             "status": "resolved"},
            {"id": "D3_task_macro_bootstrap_multiplicity",
             "found": "bootstrap 内按原 task_id 去重求宏平均，丢失重数",
             "action": "以 M*=T^-1Σa_{t_j*}（含重数）修正 CI（task_macro_ci_fix.json）；整体 AUROC/AP 不受影响",
             "status": "resolved"},
            {"id": "D4_length_weighting_not_executed",
             "found": "注册的长度五分位加权未实现",
             "action": "B 方案：prereg amendment（deferred，test 前写死）；命名 member-size-matched / length-unweighted",
             "status": "registered_deferred"},
            {"id": "D5_control_names_and_loader_exposure",
             "found": "metadata_only 为代码派生布局量（非 series 元数据）；size_length_only 含外部参数规模；"
                      "loader 在 split 排除前 json.loads，test 代码字符串短暂进入字典",
             "action": "显示名 code_layout_control / oracle_size_length_control（旧 key 保留）；暴露账本 v2 更正陈述",
             "status": "resolved"},
        ],
    }, ensure_ascii=False, indent=1), encoding="utf-8")

    (OUT / "readout_display_names.json").write_text(json.dumps({
        "schema": "variant_transfer_readout_display_names_v1",
        "readouts": {"metadata_only": {"display": "code_layout_control",
                                       "note": "代码长度/行数/空行/缩进布局等派生量；非 series/model 元数据"},
                     "size_length_only": {"display": "oracle_size_length_control",
                                          "note": "含外部已知参数规模；非未知来源单样本可部署读出"}},
        "arms": {"size_matched": {"display": "member-size-matched / length-unweighted",
                                  "length_weighting_applied": False,
                                  "note": "仅成员级 size 匹配；长度加权 deferred"}},
        "keep_original_keys": True,
    }, ensure_ascii=False, indent=1), encoding="utf-8")

    (OUT / "test_exposure_ledger_v2.json").write_text(json.dumps({
        "schema": "variant_transfer_exposure_ledger_v2",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "d-det/data/public_same_task_full_2026-10-07/records.jsonl（296,254 行）",
        "by_stage": {
            "receiving_scan": {"raw_scan": "接收阶段整文件 sha256/流式行扫描（e38715d 已登记）",
                               "transient_parse": "全部行 json 解析（接收脚本）", "feature_use": False,
                               "human_inspection": False, "metric_evaluation": False},
            "stage1_loader": {"raw_scan": "全部行按字符串前缀过滤",
                              "transient_parse": "11 成员行 json.loads()（含 1,881 test 行 solution 字符串短暂进入字典后按 split 丢弃）",
                              "feature_use": "仅 train/dev 行进入样本表；test 行未用于 fit/transform/预测",
                              "human_inspection": False, "metric_evaluation": "仅 dev"},
            "stage1_audit": {"raw_scan": True, "transient_parse": "仅元数据字段（task_id/行哈希）",
                             "feature_use": False, "human_inspection": False, "metric_evaluation": False},
            "stage1b_rebuild": {"raw_scan": False, "transient_parse": "复合键指认的 train/dev 行", "feature_use": "train（拟合）/dev（重放）",
                                "human_inspection": False, "metric_evaluation": "dev"},
        },
        "statement_correction": {
            "old": "test 正文从未进入内存（不可再称）",
            "new": "test 代码字符串在 stage-1 loader 的 transient_parse 阶段短暂进入进程内存（随即按 split 丢弃）；"
                   "未进入训练/特征/评分样本，未人类阅读，未用于任何指标或选择",
        },
        "finding": "未发现 stage-1 用 test 计算特征或指标的痕迹（feature_use/metric_evaluation 均限于 train/dev）",
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print("deviation/names/exposure done")
    print(f"runtime {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
