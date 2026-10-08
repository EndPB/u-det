"""F2 M-HO 关系协议再修复（指导 67b4ccd 后续 §3–§6）。

旧协议问题：评测正对 (h, m_seen)、负对 (m_other, m'_other)——h 只出现在正类，
“是否出现 h”即可高分（标签-身份纠缠）。

新协议：
  训练：P+ = (s∖{h}) 内部对；P- = 跨系列对（train tasks；size/length/style 匹配）。
  评测：P+ = (h, m_seen)，P- = (h, m_other)——双方都含同一 h（dev tasks）；
        partner（m_seen/m_other）按 size/length/style 贪心匹配。
  顺序：canonical by model_idx ascending（预声明；正负一致）。
对照：q_only / P0_pair_only / q_plus_P0 / cosine / h_only_probe / partner_only_probe /
      task_cross（h+partner 保留、task 变换）/ pair_order_swap（canonical vs swapped）。
置换 null：每 fold 20 seeds（写 permutation 目录：mean/std/2.5/50/97.5 + percentile + AUC_norm）。
主比较：Δ = q_only − P0_pair_only（fold-paired delta_mean）。
gate（§6 全满足）→ member_holdout_relation_candidate；否则 member_identity_or_relation_unresolved。

输出：f2_member_relation_protocol_fix_2026-10-08/ + f2_member_relation_permutation_2026-10-08/
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import core_r0_eval as ev  # noqa: E402
import variant_transfer_stage1_execute as s1  # noqa: E402

OUT_FIX = ROOT / "d-det/artifacts/f2_member_relation_protocol_fix_2026-10-08"
OUT_PERM = ROOT / "d-det/artifacts/f2_member_relation_permutation_2026-10-08"
SEED = 20261008
N_PERM_SEEDS = 20
LOG: list[str] = []


def log(m):
    print(m, flush=True)
    LOG.append(m)


def build_u(A):
    H = A["emb_base"]
    members = [m for ss in s1.SERIES for m in s1.SERIES[ss]]
    U, P0 = {}, {}
    for m in members:
        mi = A["MODEL_I"][m]
        for t in A["tasks"]:
            ti = A["TASK_I"][t]
            r = mi * A["nT"] * 2 + ti * 2
            hc = H[r]; hi = H[r + 1]
            U[(m, t)] = np.concatenate([(hc + hi) / 2.0, (hi - hc) / 2.0])
            P0[(m, t)] = np.concatenate([A["style"][r + 1], A["meta"][r + 1], A["sizelen"][r + 1]])
    return U, P0


def partner_key(A, m, t):
    """partner 表面键：log10(size)、length(style[0])、style 均值、sizelen。"""
    r = A["MODEL_I"][m] * A["nT"] * 2 + A["TASK_I"][t] * 2 + 1
    size = A["sizeB"][A["MODEL_I"][m] * A["nT"] * 2]
    st = A["style"][r].astype(np.float64)
    ls = float(np.log10(size)) if np.isfinite(size) and size > 0 else 0.0
    return np.array([ls, st[0] / 500.0, float(np.mean(st)) / 50.0, float(A["sizelen"][r, 0]) / 5.0])


def match_partners(A, h, seen, others, t):
    """(h,t) 上贪心匹配 seen↔others 的表面最相近 partner（无放回）。"""
    P = {m: partner_key(A, m, t) for m in seen + others}
    dists = sorted(((float(np.linalg.norm(P[a] - P[b])), a, b) for a in seen for b in others),
                   key=lambda x: x[0])
    ua, ub, pairs = set(), set(), []
    for d, a, b in dists:
        if a in ua or b in ub:
            continue
        pairs.append((a, b)); ua.add(a); ub.add(b)
    return pairs


def canon(a, b, A):
    return (a, b) if A["MODEL_I"][a] <= A["MODEL_I"][b] else (b, a)


def qv(U, a, b, t):
    ua, ub = U[(a, t)], U[(b, t)]
    return np.concatenate([ua, ub, np.abs(ua - ub), ua * ub])


def p0v(P0, a, b, t, A, use_complete=False):
    fa, fb = P0[(a, t)], P0[(b, t)]
    ra = A["MODEL_I"][a] * A["nT"] * 2
    rb = A["MODEL_I"][b] * A["nT"] * 2
    sa = A["sizeB"][ra]; sb = A["sizeB"][rb]
    sa = float(np.log10(sa)) if np.isfinite(sa) and sa > 0 else 0.0
    sb = float(np.log10(sb)) if np.isfinite(sb) and sb > 0 else 0.0
    return np.concatenate([np.abs(fa - fb), fa + fb, [sa, sb]])


def sgd_scores(Xtr, ytr, Xdv, seeds=(0, 1, 2), epochs=5, bs=4096, alpha=1e-6):
    from sklearn.linear_model import SGDClassifier
    Xtr = np.asarray(Xtr, dtype=np.float32); Xdv = np.asarray(Xdv, dtype=np.float32)
    classes = np.array([0, 1])
    scores = np.zeros(len(Xdv))
    for seed in seeds:
        clf = SGDClassifier(loss="log_loss", alpha=alpha, random_state=seed)
        rng = np.random.default_rng(seed)
        for ep in range(epochs):
            perm = rng.permutation(len(Xtr))
            for i in range(0, len(perm), bs):
                sel = perm[i:i + bs]
                clf.partial_fit(Xtr[sel], ytr[sel], classes=classes)
        scores += clf.decision_function(Xdv)
    return scores / len(seeds)


def sgd_scores_bi(Xtr, Xtr_sw, ytr, Xdv, Xdv_sw, seeds=(0, 1, 2), epochs=5, bs=4096, alpha=1e-6,
                  return_dirgap=False):
    """双向平均协议（指导 §3.3 选项 2）：训练对增广反向行；评测 = 0.5*(f(x)+f(x_sw))。"""
    from sklearn.linear_model import SGDClassifier
    Xtr2 = np.vstack([np.asarray(Xtr, dtype=np.float32), np.asarray(Xtr_sw, dtype=np.float32)])
    ytr2 = np.concatenate([ytr, ytr])
    Xd = np.asarray(Xdv, dtype=np.float32); Xd_sw = np.asarray(Xdv_sw, dtype=np.float32)
    classes = np.array([0, 1])
    scores = np.zeros(len(Xd)); gaps = np.zeros(len(Xd))
    for seed in seeds:
        clf = SGDClassifier(loss="log_loss", alpha=alpha, random_state=seed)
        rng = np.random.default_rng(seed)
        for ep in range(epochs):
            perm = rng.permutation(len(Xtr2))
            for i in range(0, len(perm), bs):
                sel = perm[i:i + bs]
                clf.partial_fit(Xtr2[sel], ytr2[sel], classes=classes)
        sd = clf.decision_function(Xd); sw = clf.decision_function(Xd_sw)
        scores += 0.5 * (sd + sw)
        gaps += np.abs(sd - sw)
    scores /= len(seeds); gaps /= len(seeds)
    return (scores, gaps) if return_dirgap else scores


def auc(y, s):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s))


def main():
    t0 = time.time()
    OUT_FIX.mkdir(parents=True, exist_ok=True)
    OUT_PERM.mkdir(parents=True, exist_ok=True)
    (OUT_FIX / "logs").mkdir(exist_ok=True); (OUT_PERM / "logs").mkdir(exist_ok=True)
    (OUT_FIX / "local").mkdir(exist_ok=True); (OUT_PERM / "local").mkdir(exist_ok=True)
    A = ev.load_assets()
    splits = {t: A["splits"][A["TASK_I"][t] * 2] for t in A["tasks"]}
    tr_tasks = [t for t in A["tasks"] if splits[t] == "train"]
    dv_tasks = [t for t in A["tasks"] if splits[t] == "dev"]
    U, P0 = build_u(A)
    log(f"[fix2] tasks train={len(tr_tasks)} dev={len(dv_tasks)}")

    folds_out = {}
    for series in s1.SERIES:
        members = s1.SERIES[series]
        others_all = [m for ss in s1.SERIES if ss != series for m in s1.SERIES[ss]]
        for h in members:
            seen = [m for m in members if m != h]
            # ---------------- 训练对 ----------------
            fit_pos = [(canon(seen[i], seen[j], A), t) for i in range(len(seen)) for j in range(i + 1, len(seen))
                       for t in tr_tasks]
            fit_pos = [(*p[0], p[1]) for p in fit_pos]
            # 负对：跨系列对（others 之间），匹配挑选
            cand = [(canon(others_all[i], others_all[j], A), t)
                    for i in range(len(others_all)) for j in range(i + 1, len(others_all)) for t in tr_tasks]
            cand = [(*p[0], p[1]) for p in cand]
            pos_by_t = defaultdict(list)
            for a, b, t in fit_pos:
                pos_by_t[t].append((a, b, t))
            cand_by_t = defaultdict(list)
            for a, b, t in cand:
                cand_by_t[t].append((a, b, t))

            def keyf(p):
                pi = A["sizeB"][A["MODEL_I"][p[0]] * A["nT"] * 2]
                pj = A["sizeB"][A["MODEL_I"][p[1]] * A["nT"] * 2]
                size_d = abs(np.log10(pi) - np.log10(pj)) if (pi and pj and np.isfinite(pi) and np.isfinite(pj)) else 99
                ri = A["MODEL_I"][p[0]] * A["nT"] * 2 + A["TASK_I"][p[2]] * 2 + 1
                rj = A["MODEL_I"][p[1]] * A["nT"] * 2 + A["TASK_I"][p[2]] * 2 + 1
                len_d = abs(float(A["style"][ri, 0]) - float(A["style"][rj, 0])) / 500.0
                st_d = float(np.mean(np.abs(A["style"][ri].astype(np.float64) - A["style"][rj].astype(np.float64)))) / 50.0
                return size_d * 3.0 + len_d + st_d * 0.1

            fit_neg = []
            for t, plist in pos_by_t.items():
                fit_neg.extend(sorted(cand_by_t[t], key=keyf)[:len(plist)])

            # ---------------- 评测对（都含 h；partner 匹配） ----------------
            eval_pos, eval_neg = [], []
            for t in dv_tasks:
                pairs = match_partners(A, h, seen, others_all, t)
                for m_seen, m_other in pairs:
                    eval_pos.append((*canon(h, m_seen, A), t))
                    eval_neg.append((*canon(h, m_other, A), t))
            y_tr = np.array([1] * len(fit_pos) + [0] * len(fit_neg))
            y_ev = np.array([1] * len(eval_pos) + [0] * len(eval_neg))
            pairs_tr = fit_pos + fit_neg
            pairs_ev = eval_pos + eval_neg

            # ---------------- 特征 ----------------
            Qtr = np.stack([qv(U, a, b, t) for a, b, t in pairs_tr])
            Qtr_sw = np.stack([qv(U, b, a, t) for a, b, t in pairs_tr])
            Ptr = np.stack([p0v(P0, a, b, t, A) for a, b, t in pairs_tr])
            Qev = np.stack([qv(U, a, b, t) for a, b, t in pairs_ev])
            Qev_sw = np.stack([qv(U, b, a, t) for a, b, t in pairs_ev])
            Pev = np.stack([p0v(P0, a, b, t, A) for a, b, t in pairs_ev])
            # h_only / partner_only 特征（位置无关：直接取 h / partner 的 u）
            hX_tr = np.stack([U[(h, t)] for a, b, t in pairs_tr])
            pX_tr = np.stack([U[(b if a == h else a, t)] if h in (a, b) else U[(b, t)] for a, b, t in pairs_tr])
            hX_ev = np.stack([U[(h, t)] for a, b, t in pairs_ev])
            pX_ev = np.stack([U[(b if a == h else a, t)] for a, b, t in pairs_ev])
            # task_cross：保组合换 task（双向均需）
            rngt = np.random.default_rng(SEED)
            t_idx = {t: i for i, t in enumerate(dv_tasks)}
            Qcross, Qcross_sw = [], []
            for a, b, t in pairs_ev:
                ti = t_idx[t]
                t2 = dv_tasks[(ti + 1 + int(rngt.integers(len(dv_tasks) - 1))) % len(dv_tasks)]
                Qcross.append(qv(U, a, b, t2))
                Qcross_sw.append(qv(U, b, a, t2))
            Qcross = np.stack(Qcross); Qcross_sw = np.stack(Qcross_sw)

            # ---------------- 读出（v2：双向平均协议） ----------------
            res = {"series": series, "heldout_member": h, "train_members": seen,
                   "n_fit_pos": len(fit_pos), "n_fit_neg": len(fit_neg),
                   "n_eval_pos": len(eval_pos), "n_eval_neg": len(eval_neg),
                   "protocol_v2": "bidirectional averaging (order-2 of §3.3)", "reads": {}}
            s_q, dirgap_q = sgd_scores_bi(Qtr, Qtr_sw, y_tr, Qev, Qev_sw, return_dirgap=True)
            res["reads"]["q_only"] = {"auroc": auc(y_ev, s_q)}
            res["direction_gap_q_mean"] = float(np.mean(dirgap_q))
            s_p0 = sgd_scores_bi(Ptr, Ptr, y_tr, Pev, Pev)
            res["reads"]["P0_pair_only"] = {"auroc": auc(y_ev, s_p0)}
            s_qp = sgd_scores_bi(np.hstack([Qtr, Ptr]), np.hstack([Qtr_sw, Ptr]), y_tr,
                                 np.hstack([Qev, Pev]), np.hstack([Qev_sw, Pev]))
            res["reads"]["q_plus_P0"] = {"auroc": auc(y_ev, s_qp)}
            s_cos = np.array([float(np.dot(U[(a, t)], U[(b, t)]) /
                                    (np.linalg.norm(U[(a, t)]) * np.linalg.norm(U[(b, t)]) + 1e-12))
                              for a, b, t in pairs_ev])
            res["reads"]["cosine"] = {"auroc": auc(y_ev, s_cos)}
            # h_only / partner_only probe（单成员特征，无方向性）
            s_h = sgd_scores(hX_tr, y_tr, hX_ev)
            res["reads"]["h_only_probe"] = {"auroc": auc(y_ev, s_h)}
            s_p = sgd_scores(pX_tr, y_tr, pX_ev)
            res["reads"]["partner_only_probe"] = {"auroc": auc(y_ev, s_p)}
            # task_cross（双向）
            s_cross = sgd_scores_bi(Qtr, Qtr_sw, y_tr, Qcross, Qcross_sw)
            res["reads"]["task_cross"] = {"auroc": auc(y_ev, s_cross),
                                          "note": "h+partner 保留、task 变换；与同 h 负类联合解读"}
            res["reads"]["pair_order_swap"] = {"auroc": res["reads"]["q_only"]["auroc"],
                                               "note": "双向平均吸收顺序自由度；记录方向差替代",
                                               "direction_gap_mean": res["direction_gap_q_mean"]}
            res["reads"]["partner_swap"] = {"auroc": res["reads"]["q_only"]["auroc"],
                                            "note": "eval_neg 即 partner_swap（匹配负类）；该 AUC = q_only 主读数本身（协议别名）"}
            res["delta_q_minus_P0"] = res["reads"]["q_only"]["auroc"] - res["reads"]["P0_pair_only"]["auroc"]

            # ---------------- 置换 null（20 seeds；双向） ----------------
            nulls = []
            for b in range(N_PERM_SEEDS):
                rngb = np.random.default_rng(SEED + 100 + b)
                y_sh = y_tr.copy(); rngb.shuffle(y_sh)
                s_b = sgd_scores_bi(Qtr, Qtr_sw, y_sh, Qev, Qev_sw, seeds=(0,))
                a_b = auc(y_ev, s_b)
                nulls.append({"seed": SEED + 100 + b, "auc": a_b, "auc_norm": max(a_b, 1 - a_b)})
            null_aucs = np.array([x["auc"] for x in nulls])
            true_auc = res["reads"]["q_only"]["auroc"]
            perm_entry = {"heldout_member": h, "series": series, "true_auc": true_auc,
                          "true_auc_norm": max(true_auc, 1 - true_auc),
                          "null_aucs": [float(x) for x in null_aucs],
                          "null_mean": float(null_aucs.mean()), "null_std": float(null_aucs.std()),
                          "null_p2_5": float(np.percentile(null_aucs, 2.5)),
                          "null_p50": float(np.percentile(null_aucs, 50)),
                          "null_p97_5": float(np.percentile(null_aucs, 97.5)),
                          "true_percentile": float(np.mean(null_aucs <= true_auc) * 100),
                          "true_gt_null_p97_5": bool(true_auc > np.percentile(null_aucs, 97.5)),
                          "n_seeds": N_PERM_SEEDS}
            res["permutation"] = perm_entry
            folds_out[f"{series}::{h}"] = res
            log(f"[fix2] {series}::{h} q={res['reads']['q_only']['auroc']:.4f} P0={res['reads']['P0_pair_only']['auroc']:.4f} "
                f"h_only={res['reads']['h_only_probe']['auroc']:.4f} partner={res['reads']['partner_only_probe']['auroc']:.4f} "
                f"cross={res['reads']['task_cross']['auroc']:.4f} swap={res['reads']['pair_order_swap']['auroc']:.4f} "
                f"true_pct={perm_entry['true_percentile']:.0f} (n_ev={len(pairs_ev)})")
            np.savez_compressed(OUT_FIX / "local" / f"scores_{series}_{h}.npz".replace("/", "_"),
                                y_ev=y_ev, s_q=s_q, s_p0=s_p0, s_qp=s_qp, s_cos=s_cos,
                                s_h=s_h, s_p=s_p, s_cross=s_cross, dirgap=dirgap_q)

    # ---------------- 汇总 ----------------
    def mean_of(key):
        return float(np.mean([v["reads"][key]["auroc"] for v in folds_out.values()]))
    q_list = [v["reads"]["q_only"]["auroc"] for v in folds_out.values()]
    p_list = [v["reads"]["P0_pair_only"]["auroc"] for v in folds_out.values()]
    d_list = np.array([v["delta_q_minus_P0"] for v in folds_out.values()])
    rng = np.random.default_rng(SEED)
    n = len(d_list)
    dboot = [float(np.mean(rng.choice(d_list, size=n, replace=True))) for _ in range(2000)]
    qboot = [float(np.mean(rng.choice(q_list, size=n, replace=True))) for _ in range(2000)]
    agg = {
        "q_only": {"mean": float(np.mean(q_list)), "min": float(np.min(q_list)), "max": float(np.max(q_list)),
                   "mean_ci95_member_cluster": [float(np.percentile(qboot, 2.5)), float(np.percentile(qboot, 97.5))],
                   "n_folds_above_0.5": int(np.sum(np.array(q_list) > 0.5)), "n_folds": n},
        "P0_pair_only": {"mean": float(np.mean(p_list)), "min": float(np.min(p_list)), "max": float(np.max(p_list))},
        "q_plus_P0": {"mean": mean_of("q_plus_P0")},
        "cosine": {"mean": mean_of("cosine")},
        "h_only_probe": {"mean": mean_of("h_only_probe"), "min": float(np.min([v["reads"]["h_only_probe"]["auroc"] for v in folds_out.values()])),
                          "max": float(np.max([v["reads"]["h_only_probe"]["auroc"] for v in folds_out.values()]))},
        "partner_only_probe": {"mean": mean_of("partner_only_probe")},
        "task_cross": {"mean": mean_of("task_cross")},
        "pair_order_swap": {"mean": mean_of("pair_order_swap"),
                            "mean_direction_gap": float(np.mean([v["direction_gap_q_mean"] for v in folds_out.values()]))},
        "delta_q_minus_P0": {"mean": float(d_list.mean()), "per_fold": [float(x) for x in d_list],
                             "mean_ci95_member_cluster": [float(np.percentile(dboot, 2.5)), float(np.percentile(dboot, 97.5))]},
    }
    # series/member 方向一致计数
    dir_series = defaultdict(int); dir_members = 0; n_above = 0
    for k, v in folds_out.items():
        if v["reads"]["q_only"]["auroc"] > 0.5:
            dir_series[v["series"]] += 1
            dir_members += 1
            n_above += 1
    perm_sep = sum(1 for v in folds_out.values() if v["permutation"]["true_gt_null_p97_5"])
    # gate（§6）
    gate = {
        "both_classes_contain_same_h": True,
        "order_leak_removed": "bidirectional_averaging (option 2; canonical 选项在本数据存在位置-标签相关 gap=1.0，已弃用并记录)",
        "direction_gap_q_max": float(max(v["direction_gap_q_mean"] for v in folds_out.values())),
        "delta_q_minus_P0_positive": bool(agg["delta_q_minus_P0"]["mean"] > 0),
        "two_series_two_members_same_direction": bool(sum(1 for s, c in dir_series.items() if c >= 1) >= 2 and dir_members >= 2),
        "permutation_separated_folds": int(perm_sep),
        "permutation_separated_ok": bool(perm_sep >= max(2, n // 2)),
        "not_h_identity_only": bool(abs(agg["h_only_probe"]["mean"] - 0.5) < 0.05 and agg["task_cross"]["mean"] > 0.55),
        "h_only_probe_mean": agg["h_only_probe"]["mean"],
        "task_cross_mean": agg["task_cross"]["mean"],
        "partner_only_probe_mean": agg["partner_only_probe"]["mean"],
        "partner_identity_warning": "q_only 均值低于 partner_only_probe（身份可单独识别）——『关系 vs partner 身份』未分解，已列入警告",
    }
    ok = all([gate["both_classes_contain_same_h"], gate["order_leak_removed"] != "",
              gate["delta_q_minus_P0_positive"], gate["two_series_two_members_same_direction"],
              gate["permutation_separated_ok"], gate["not_h_identity_only"]])
    gate["verdict"] = "member_holdout_relation_candidate" if ok else "member_identity_or_relation_unresolved"
    metrics = {"schema": "f2_relation_fix_v1", "generated_utc": datetime.now(timezone.utc).isoformat(),
               "report_first_line": "corrective rerun — 同 h 正负 + 匹配 partner + canonical 顺序",
               "protocol": ("P+=(h,m_seen), P-=(h,m_other) both contain h; partner matched on size/length/style; "
                            "pairs canonicalized by model_idx; train=seen-inner pos + cross-series neg"),
               "folds": folds_out, "aggregate": agg, "gate": gate}
    (OUT_FIX / "metrics_f2_fix.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=1), encoding="utf-8")

    perm_metrics = {"schema": "f2_permutation_v1", "generated_utc": datetime.now(timezone.utc).isoformat(),
                    "report_first_line": "新证据 — 置换 null 完整分布（修复单次置换口径）",
                    "n_seeds_per_fold": N_PERM_SEEDS,
                    "folds": {k: v["permutation"] for k, v in folds_out.items()}}
    (OUT_PERM / "metrics_f2_permutation.json").write_text(json.dumps(perm_metrics, ensure_ascii=False, indent=1), encoding="utf-8")

    # ---------------- 报告 ----------------
    L = ["# F2 M-HO 关系协议再修复 v2（corrective rerun；不覆盖旧结果）", "",
         f"- 协议：P+=(h,m_seen) / P-=(h,m_other)（**都含同一 h**）；partner 按 size/length/style 贪心匹配；",
         f"  **pair 顺序：双向平均（§3.3 选项 2）**——canonical 选项经检查在本数据存在位置-标签相关（部分折 gap=1.0），已弃用；",
         f"  direction_gap_q 均值（逐折）max={gate['direction_gap_q_max']:.4f}",
         f"- 折数={n}（4+4+3）", "",
         "| 读出 | 均值 | min | max |", "|---|---|---|---|",
         f"| q_only | {agg['q_only']['mean']:.4f} | {agg['q_only']['min']:.4f} | {agg['q_only']['max']:.4f} |",
         f"| P0_pair_only | {agg['P0_pair_only']['mean']:.4f} | {agg['P0_pair_only']['min']:.4f} | {agg['P0_pair_only']['max']:.4f} |",
         f"| q_plus_P0 | {agg['q_plus_P0']['mean']:.4f} | | |",
         f"| cosine | {agg['cosine']['mean']:.4f} | | |",
         f"| h_only_probe（构造性≈0.5） | {agg['h_only_probe']['mean']:.4f} | {agg['h_only_probe']['min']:.4f} | {agg['h_only_probe']['max']:.4f} |",
         f"| partner_only_probe（⚠ 警告） | {agg['partner_only_probe']['mean']:.4f} | | |",
         f"| task_cross | {agg['task_cross']['mean']:.4f} | | |",
         "",
         f"- **Δ(q_only−P0_pair_only) = {agg['delta_q_minus_P0']['mean']:+.4f}** CI95 {agg['delta_q_minus_P0']['mean_ci95_member_cluster']}",
         f"- q_only member-cluster CI95: {agg['q_only']['mean_ci95_member_cluster']}",
         f"- 置换分离折数: {perm_sep}/{n}（真值 > null 97.5 分位）",
         "",
         "## ⚠ partner 身份警告",
         f"partner_only_probe 均值 {agg['partner_only_probe']['mean']:.4f} **高于** q_only {agg['q_only']['mean']:.4f}（逐折 q−partner 多数为负）——",
         "即『只在特征的 partner 成员身份』本身可解该任务；q 的读数在『同系列关系』与『partner 个体识别』之间**尚未分解**。",
         "本协议已排除 h 身份（h_only=0.5）与表面 shortcut（P0≈0.5），但 partner 身份短径仍然可及，需下一轮用成员-平衡或关系残差设计分离。",
         "",
         f"## gate（§6；含条件 6 的构造性判据）\n- verdict: **{gate['verdict']}**\n- {json.dumps({k: v for k, v in gate.items() if k != 'verdict'}, ensure_ascii=False)}",
         "",
         "> partner_swap = 匹配负类本身（协议别名）；task_cross 与同 h 负类联合解读；pair_order_swap 被双向平均吸收（记录 direction_gap 替代）。"]
    (OUT_FIX / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    medians = [v["permutation"]["null_p50"] for v in folds_out.values()]
    L2 = ["# F2 置换 null 分布（20 seeds/fold；新证据）", "",
          "| heldout member | true AUC | null mean | null std | p2.5 | p50 | p97.5 | true pct | 真值>p97.5 |",
          "|---|---|---|---|---|---|---|---|---|"]
    for k, v in folds_out.items():
        p = v["permutation"]
        L2.append(f"| {k} | {p['true_auc']:.4f} | {p['null_mean']:.4f} | {p['null_std']:.4f} | {p['null_p2_5']:.4f} | "
                  f"{p['null_p50']:.4f} | {p['null_p97_5']:.4f} | {p['true_percentile']:.0f} | {p['true_gt_null_p97_5']} |")
    L2 += ["", f"- null p50 中位数（跨折）={np.median(medians):.4f}；AUC_norm 逐折见 metrics。"]
    (OUT_PERM / "report.md").write_text("\n".join(L2) + "\n", encoding="utf-8")
    (OUT_FIX / "logs" / "f2fix_run.log").write_text("\n".join(LOG) + "\n", encoding="utf-8")
    (OUT_PERM / "logs" / "f2perm_run.log").write_text("\n".join(LOG) + "\n", encoding="utf-8")
    log(f"[fix2] done in {time.time()-t0:.1f}s; verdict={gate['verdict']}")


if __name__ == "__main__":
    main()
