"""变体迁移执行前闸门（2026-10-08）§3-§5：R1 负集 amendment + 执行配置冻结 + 开关。

产出（variant_transfer_preflight_2026-10-08/）：
- r1_negative_set_amendment.json：
    原注册版负集定义保留（结果名 series-transfer-with-size-mix）；
    新增预声明 size-matched 匹配控制版（series-transfer-size-matched，敏感性分析）；
    member-level 匹配用穷举最优、纯元数据（参数规模），不看任何数据内容；
    length-quantile 加权规则与 size/length-only 必备对照在此冻结（执行阶段才计算）。
- execution_config_frozen.json：特征/阈值/seed/P0/CI/dev 口径在执行前冻结；
- execution_switches.json：training/generation/test_read 开关。

一致性：脚本 assert 与 811ecfe 已冻结注册文件（variant_transfer_registration.json）不冲突。
不读 test、不读代码正文、不下载权重。
"""
from __future__ import annotations

import itertools
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
F = ROOT / "d-det/artifacts/public_full_followup_2026-10-08"
OUT = ROOT / "d-det/artifacts/variant_transfer_preflight_2026-10-08"
REG = json.loads((F / "variant_transfer_registration.json").read_text(encoding="utf-8"))
ADM = json.loads((F / "server_rebuild/family_series_admission.server_rebuild.json").read_text(encoding="utf-8"))

# 参数规模（十进制 B；来源=官方命名，静态元数据）
PARAM_B = {
    "codellama--CodeLlama-7b-Instruct-hf": 7.0,
    "codellama--CodeLlama-13b-Instruct-hf": 13.0,
    "codellama--CodeLlama-34b-Instruct-hf": 34.0,
    "codellama--CodeLlama-70b-Instruct-hf": 70.0,
    "Qwen--Qwen2.5-Coder-1.5B-Instruct": 1.5,
    "Qwen--Qwen2.5-Coder-7B-Instruct": 7.0,
    "Qwen--Qwen2.5-Coder-14B-Instruct": 14.0,
    "Qwen--Qwen2.5-Coder-32B-Instruct": 32.0,
    "deepseek-ai--deepseek-coder-1.3b-instruct": 1.3,
    "deepseek-ai--deepseek-coder-6.7b-instruct": 6.7,
    "deepseek-ai--deepseek-coder-33b-instruct": 33.0,
}
SERIES = {
    "CodeLlama-Instruct": ["codellama--CodeLlama-7b-Instruct-hf", "codellama--CodeLlama-13b-Instruct-hf",
                           "codellama--CodeLlama-34b-Instruct-hf", "codellama--CodeLlama-70b-Instruct-hf"],
    "Qwen2.5-Coder-Instruct": ["Qwen--Qwen2.5-Coder-1.5B-Instruct", "Qwen--Qwen2.5-Coder-7B-Instruct",
                               "Qwen--Qwen2.5-Coder-14B-Instruct", "Qwen--Qwen2.5-Coder-32B-Instruct"],
    "DeepSeek-Coder-v1-Instruct": ["deepseek-ai--deepseek-coder-1.3b-instruct",
                                   "deepseek-ai--deepseek-coder-6.7b-instruct",
                                   "deepseek-ai--deepseek-coder-33b-instruct"],
}

SIZE_BUCKETS = {
    "A_lt2B": {"lo": None, "hi": 2.0},
    "B_2to10B": {"lo": 2.0, "hi": 10.0},
    "C_10to40B": {"lo": 10.0, "hi": 40.0},
    "D_ge40B": {"lo": 40.0, "hi": None},
}


def bucket_of(size: float) -> str:
    for name, b in SIZE_BUCKETS.items():
        if (b["lo"] is None or size >= b["lo"]) and (b["hi"] is None or size < b["hi"]):
            return name
    raise ValueError(size)


def best_selection(pos: list[str], pool: list[str], series_of: dict[str, str],
                   require_each_series: bool) -> dict:
    """穷举最优：从 pool 选 len(pos) 个成员，与 pos 一一匹配，最小化 Σ|log10 差|。

    平局规则：(总距离, 选中成员名升序元组) 字典序取小。require_each_series 时要求
    每个负系列至少贡献 1 名成员（|pos|>=2 时适用）。
    """
    k = len(pos)
    neg_series = sorted({series_of[m] for m in pool})
    best = None
    for combo in itertools.combinations(pool, k):
        if require_each_series and k >= 2:
            present = {series_of[m] for m in combo}
            if len(present) < len(neg_series):
                continue
        # 最优匹配（k! 极小）
        for perm in itertools.permutations(combo):
            d = sum(abs(math.log10(PARAM_B[p]) - math.log10(PARAM_B[n]))
                    for p, n in zip(pos, perm))
            key = (round(d, 12), tuple(sorted(combo)))
            if best is None or key < best[0]:
                best = (key, perm, d)
    assert best is not None
    _, perm, d = best
    return {
        "selected": list(perm),
        "assignment": [{"positive": p, "negative": n,
                        "abs_log10_size_diff": round(abs(math.log10(PARAM_B[p]) - math.log10(PARAM_B[n])), 4)}
                       for p, n in zip(pos, perm)],
        "total_abs_log10_size_diff": round(d, 4),
    }


def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    series_of = {m: s for s, ms in SERIES.items() for m in ms}
    assert sum(len(v) for v in SERIES.values()) == 11

    # 注册一致性检查（键值不得漂移）
    assert REG["series_folds"]["CodeLlama-Instruct"]["heldout_order"] == ["7b", "13b", "34b", "70b"]
    assert REG["series_folds"]["Qwen2.5-Coder-Instruct"]["heldout_order"] == ["1.5B", "7B", "14B", "32B"]
    assert REG["series_folds"]["DeepSeek-Coder-v1-Instruct"]["heldout_order"] == ["1.3b", "6.7b", "33b"]
    assert REG["seeds"] == [0, 1, 2]
    assert ADM["family_is_confirmed"] is False
    for s, ms in SERIES.items():
        assert set(ms) == set(ADM["series"][s]["members"].keys()), s

    # ---- 逐折 size-matched 负集选择（穷举最优，纯元数据） ----
    per_fold = {}
    for s, ms in SERIES.items():
        m2s = {m: sz for sz, m in REG["series_folds"][s]["members_by_size"].items()}
        for h in ms:
            pos = sorted([m for m in ms if m != h], key=lambda m: PARAM_B[m])
            pool = sorted([m for ss, mms in SERIES.items() if ss != s for m in mms],
                          key=lambda m: PARAM_B[m])
            sel = best_selection(pos, pool, series_of, require_each_series=len(pos) >= 2)
            per_fold[f"{s}::heldout={m2s[h]}"] = {
                "heldout_member": h,
                "heldout_size": PARAM_B[h],
                "positive_seen_members": pos,
                "negative_pool": pool,
                "size_matched_negatives": sel,
                "original_arm_negatives": "负池全体（size-mix 原注册版）",
                "size_bucket_strata": {m: bucket_of(PARAM_B[m]) for m in [h] + pos + pool},
            }

    amendment = {
        "schema": "variant_transfer_r1_negative_set_amendment_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "amends": "variant_transfer_registration.json（811ecfe）readouts.R1_series_membership 的负集选择；原注册版全文保留、不改写",
        "registry_policy": {
            "original_kept": "R1 原注册版=负集为另外两个 series 全体成员；结果命名 series-transfer-with-size-mix",
            "no_silent_rewrite": True,
            "amendment_role": "size-matched 匹配控制版为预注册 amendment 敏感性分析；结果命名 series-transfer-size-matched",
            "main_result_default": "series-transfer-with-size-mix（除非指导端另行把匹配版升级为主结果；升级需新 prereg 版本并保留原注册结果）",
            "decision_timing": "两版本并行报告；不得看完 dev/test 后决定采用哪一个",
            "change_log": ["2026-10-08: 首次写入 amendment（执行前，未读任何 train/dev/test 数据内容）"],
        },
        "size_buckets": SIZE_BUCKETS,
        "member_param_sizes_B": PARAM_B,
        "member_size_buckets": {m: bucket_of(v) for m, v in PARAM_B.items()},
        "size_matched_arm": {
            "member_selection_rule": (
                "每折：正集=目标 series 的已见尺寸成员（按参数升序）；负池=另外两个 series 全体；"
                "穷举选择 |正集| 个负成员并一一匹配，最小化 Σ|log10(size_pos)-log10(size_neg)|；"
                "平局按 (总距离, 成员名升序) 取小；|正集|≥2 时要求每个负 series 至少 1 名成员"
            ),
            "pure_metadata": True,
            "per_fold": per_fold,
            "length_quantile_weighting": {
                "status": "frozen_formula_not_executed",
                "when": "执行阶段（获得 training 授权后）",
                "cluster": "task cluster（每个 task 内）",
                "buckets": "输出字符长度五分位桶（由 TRAIN 正集长度分布定义边界）",
                "weight": "w_b = p_pos(b)/p_neg(b)，截断 [0.1,10]；权重在 train 上估计并冻结，dev/test 复用同一权重",
                "length_field": "records.jsonl 行 output 的字符数（执行阶段读取；本阶段不读正文）",
                "no_test_in_fit": True,
            },
        },
        "required_baselines": {
            "size_length_only": {
                "features": ["log10(参数规模 B)", "输出字符长度", "metadata-only 结构特征（行数/nl/缩进量级）"],
                "model": "同折同 split 的 LogReg（class_weight=balanced）",
                "report": "每折 AUROC/AP 与主读出并列；用于判断系列信号是否超出 size/length 混杂",
            },
            "size_strata_report": "原版（size-mix）结果额外按 size bucket 分层报告（参数规模、代码长度、token 长度、任务难度 in_hard）",
        },
        "note": "R1 标签=目标 series 正域 vs 预声明负域；heldout variant 为正域迁移样本；禁止把未见 variant 当新类别报告 macro-F1",
    }
    (OUT / "r1_negative_set_amendment.json").write_text(
        json.dumps(amendment, ensure_ascii=False, indent=1), encoding="utf-8")

    config = {
        "schema": "variant_transfer_execution_config_v1",
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "supersedes_note": "细化 811ecfe 注册文件的可执行口径；不修改注册文本本身",
        "data": REG["data"],
        "heldout_order": {
            "CodeLlama-Instruct": ["7b", "13b", "34b", "70b"],
            "Qwen2.5-Coder-Instruct": ["1.5B", "7B", "14B", "32B"],
            "DeepSeek-Coder-v1-Instruct": ["1.3b", "6.7b", "33b"],
        },
        "seeds": {"feature_seeds": [0, 1, 2], "bootstrap_resamples": 500, "bootstrap_seed": 20261008},
        "features": REG["features_frozen"],
        "r1": {
            "classifier": "LogReg(class_weight=balanced, max_iter=2000)",
            "C_grid": [0.03, 0.1, 0.3, 1.0],
            "dev_selection": "AUROC 最大；平局取更小 C；仅在预声明网格内",
            "threshold": 0.5,
            "metrics": ["AUROC", "average precision", "task-macro average", "task-cluster 500× bootstrap 95% CI", "每折支持 n", "train→heldout 迁移差"],
        },
        "r2": {
            "estimators": ["series center（cosine / euclidean）", "线性回归迁移（train 拟合）", "预声明相似度函数"],
            "fit_rule": "中心/尺度/回归系数/阈值仅用 train 拟合；dev 仅一次预声明选择；heldout variant 仅在冻结后评分",
            "controls": ["length/char count/lexical stats baseline", "CodeT5-small 冻结表示", "CodeT5-base 冻结表示", "P0-fusion（仅 train 拟合）", "P0-equal（成员概率等权、无校准）"],
            "interpretation": "只回答系列内表示迁移是否可测；不得解释为因果后训练方向",
        },
        "p0": REG["p0"],
        "dev_policy_frozen_reading": (
            "dev=已见尺寸+预声明负集（不含 heldout variant）在 DEV tasks（171）上的一次预声明选择；"
            "heldout variant 仅在冻结后（TEST tasks 单次读取）评分"
        ),
        "metrics_protocol": {"stats_unit": "task cluster", "ci": "task-cluster bootstrap ×500（重采样频率）", "task_macro": "按 task 计算后宏平均"},
        "stop_rules": [
            "任何折发现 heldout variant 行进入训练 → 立即作废该折",
            "负集或匹配规则在 dev 后改变 → 整轮降级为探索性",
            "原件哈希未解决而使用重构版 → 结果只能写 server_reconstruction_only",
            "仅某一系列或某一尺寸有效 → 不得扩写为普遍 family 结论",
        ],
        "paper_language_allow": "在固定 BigCodeBench 任务切分上，官方模型系列的未见尺寸变体保留了可测的系列迁移信号。",
        "paper_language_forbid": ["unseen independent family", "后训练因果效应", "无污染确认性 benchmark", "所有 generator / 模型家族具有同样几何结构"],
    }
    (OUT / "execution_config_frozen.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=1), encoding="utf-8")

    switches = {
        "schema": "variant_transfer_execution_switches_v1",
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "training_allowed": False,
        "generation_allowed": False,
        "test_read_allowed": False,
        "authority": "awaiting_new_execution_authorization",
        "note": "以上三项仅在收到指导端新的明确执行授权后由对应记录开启；本轮全部只读",
    }
    (OUT / "execution_switches.json").write_text(
        json.dumps(switches, ensure_ascii=False, indent=1), encoding="utf-8")

    print("per-fold size-matched selections:")
    for k, v in per_fold.items():
        sel = v["size_matched_negatives"]
        print(f"  {k:<44s} neg={[m.split('--')[-1] for m in sel['selected']]} total_d={sel['total_abs_log10_size_diff']}")
    print("runtime:", round(time.time() - t0, 3), "s")


if __name__ == "__main__":
    main()
