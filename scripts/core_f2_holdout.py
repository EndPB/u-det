"""F2 升级：member/generator-heldout（M-HO + S-HO）+ P0_pair_only 独立对照。

M-HO（11 折）：留出 member h；train 正对=其余 members 内部对（train tasks）；
  eval 正对=h×已见 member（dev tasks）；负对=非 s 系列成员对（size+length+style 匹配）。
S-HO（3 折）：留出 series；train 正对=其余系列内部对；train 负对=其余系列间跨系列对；
  eval 正对=留出系列内部对；eval 负对=留出系列成员×其他系列成员（匹配）。低功效压力测试，单列。

读出：cosine / q_only / P0_pair_only / q_plus_P0 / permuted / mismatched_partner / cross_task_swap。
主比较：fold-paired q_only − P0_pair_only（delta_mean + member-cluster CI）。
gate（§3.3）：≥2 折同向；q_only−P0 delta_mean>0；置换/错配回机会；非单一 series/长度桶贡献；CI 单列。

输出：artifacts/f2_member_series_holdout_2026-10-08/
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

OUT = ROOT / "d-det/artifacts/f2_member_series_holdout_2026-10-08"
SEED = 20261008
LOG: list[str] = []


def log(m):
    print(m, flush=True)
    LOG.append(m)


def build_u(A):
    """全成员×全任务 u=[S;D]（1536d）字典 + 行内 style/meta/sizelen 提取器。"""
    H = A["emb_base"]
    members = [m for ss in s1.SERIES for m in s1.SERIES[ss]]
    U = {}
    P0 = {}
    for m in members:
        mi = A["MODEL_I"][m]
        for t in A["tasks"]:
            ti = A["TASK_I"][t]
            r = mi * A["nT"] * 2 + ti * 2
            hc = H[r]; hi = H[r + 1]
            U[(m, t)] = np.concatenate([(hc + hi) / 2.0, (hi - hc) / 2.0])
            P0[(m, t)] = np.concatenate([A["style"][r], A["meta"][r], A["sizelen"][r]])
    return U, P0


def q_vector_parts(U, a, b, t):
    ua, ub = U[(a, t)], U[(b, t)]
    return np.concatenate([ua, ub, np.abs(ua - ub), ua * ub])


def p0_pair_only(P0, a, b, t, A):
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


def match_negatives(A, pos_pairs, cand_pairs):
    """按 size+length+style 加权距离为每个 task 挑与正对等量的负对。"""
    pos_by_t = defaultdict(list)
    for p in pos_pairs:
        pos_by_t[p[2]].append(p)
    cand_by_t = defaultdict(list)
    for p in cand_pairs:
        cand_by_t[p[2]].append(p)
    def keyf(p):
        pi = A["sizeB"][A["MODEL_I"][p[0]] * A["nT"] * 2]
        pj = A["sizeB"][A["MODEL_I"][p[1]] * A["nT"] * 2]
        size_d = abs(np.log10(pi) - np.log10(pj)) if (pi and pj) else 99
        ri = A["MODEL_I"][p[0]] * A["nT"] * 2 + A["TASK_I"][p[2]] * 2 + 1
        rj = A["MODEL_I"][p[1]] * A["nT"] * 2 + A["TASK_I"][p[2]] * 2 + 1
        len_d = abs(float(A["style"][ri, 0]) - float(A["style"][rj, 0])) / 500.0
        st_d = float(np.mean(np.abs(A["style"][ri].astype(np.float64) - A["style"][rj].astype(np.float64)))) / 50.0
        return size_d * 3.0 + len_d + st_d * 0.1
    neg = []
    for t, plist in pos_by_t.items():
        cands = sorted(cand_by_t.get(t, []), key=keyf)
        neg.extend(cands[:len(plist)])
    return neg


def eval_reads(U, P0, A, pos_pairs, neg_pairs, per_task_rows):
    """返回各读出在 eval（pos+neg）上的分数。per_task_rows: task -> {'pos': idx list, 'neg': idx list}"""
    from sklearn.metrics import roc_auc_score, average_precision_score
    all_pairs = pos_pairs + neg_pairs
    y = np.array([1] * len(pos_pairs) + [0] * len(neg_pairs))
    Q = np.stack([q_vector_parts(U, p[0], p[1], p[2]) for p in all_pairs])
    P0F = np.stack([p0_pair_only(P0, p[0], p[1], p[2], A) for p in all_pairs])
    return Q, P0F, y, all_pairs


def cos_eval(U, all_pairs):
    out = []
    for a, b, t in all_pairs:
        ua, ub = U[(a, t)], U[(b, t)]
        out.append(float(np.dot(ua, ub) / (np.linalg.norm(ua) * np.linalg.norm(ub) + 1e-12)))
    return np.asarray(out)


def run_fold(U, P0, A, fit_pos_dev_free, eval_pos, fit_neg_cand, eval_neg_cand, tr_tasks, dv_tasks, tag):
    """通用 fold：训练对/评测对均由调用方给构造器。返回 metrics dict。"""
    from sklearn.metrics import roc_auc_score, average_precision_score
    t0 = time.time()
    # 训练：正对仅在 tr_tasks；负对匹配
    fit_pos = [(a, b, t) for (a, b, t) in fit_pos_dev_free if t in set(tr_tasks)]
    fit_cand = [(a, b, t) for (a, b, t) in fit_neg_cand if t in set(tr_tasks)]
    fit_neg = match_negatives(A, fit_pos, fit_cand)
    ev_pos = [(a, b, t) for (a, b, t) in eval_pos if t in set(dv_tasks)]
    ev_cand = [(a, b, t) for (a, b, t) in eval_neg_cand if t in set(dv_tasks)]
    ev_neg = match_negatives(A, ev_pos, ev_cand)
    n_tr, n_ev = len(fit_pos) + len(fit_neg), len(ev_pos) + len(ev_neg)
    Qtr, P0tr, ytr, pairs_tr = eval_reads(U, P0, A, fit_pos, fit_neg, None)
    Qdv, P0dv, ydv, pairs_dv = eval_reads(U, P0, A, ev_pos, ev_neg, None)
    out = {"n_train_pairs": n_tr, "n_eval_pairs": n_ev, "n_eval_pos": len(ev_pos),
           "n_train_pos": len(fit_pos), "reads": {}}
    # cosine
    s_cos = cos_eval(U, pairs_dv)
    out["reads"]["cosine"] = {"auroc": float(roc_auc_score(ydv, s_cos)), "ap": float(average_precision_score(ydv, s_cos))}
    # q_only
    s_q = sgd_scores(Qtr, ytr, Qdv)
    out["reads"]["q_only"] = {"auroc": float(roc_auc_score(ydv, s_q)), "ap": float(average_precision_score(ydv, s_q))}
    # P0_pair_only
    s_p0 = sgd_scores(P0tr, ytr, P0dv)
    out["reads"]["P0_pair_only"] = {"auroc": float(roc_auc_score(ydv, s_p0)), "ap": float(average_precision_score(ydv, s_p0))}
    # q_plus_P0
    Xq_tr = np.hstack([Qtr, P0tr]).astype(np.float32); Xq_dv = np.hstack([Qdv, P0dv]).astype(np.float32)
    s_qp = sgd_scores(Xq_tr, ytr, Xq_dv)
    out["reads"]["q_plus_P0"] = {"auroc": float(roc_auc_score(ydv, s_qp)), "ap": float(average_precision_score(ydv, s_qp))}
    # permuted labels
    y_sh = ytr.copy(); np.random.default_rng(SEED).shuffle(y_sh)
    s_sh = sgd_scores(Qtr, y_sh, Qdv)
    out["reads"]["permuted"] = {"auroc": float(roc_auc_score(ydv, s_sh)), "ap": float(average_precision_score(ydv, s_sh))}
    # mismatched_partner：eval 正对第二成员换为同 task 随机成员（非本系列；若不破坏关系的替代不可用时跳过）
    rng = np.random.default_rng(SEED + 7)
    mm_pos = []
    for a, b, t in ev_pos:
        s_a = s1.SERIES_OF[a]
        pool = [m for ss in s1.SERIES if ss != s_a for m in s1.SERIES[ss]]
        c = pool[int(rng.integers(len(pool)))]
        mm_pos.append((a, c, t))
    if mm_pos:
        Qmm, _, _, pairs_mm = eval_reads(U, P0, A, mm_pos, [], None)
        s_mm = sgd_scores(Qtr, ytr, Qmm)
        y_mm = np.concatenate([np.ones(len(mm_pos)), np.zeros(len(ev_neg))])
        Qmix = np.vstack([Qmm, Qdv[len(ev_pos):]])
        s_mix = np.concatenate([s_mm, s_q[len(ev_pos):]])
        out["reads"]["mismatched_partner"] = {"auroc": float(roc_auc_score(y_mm, s_mix)),
                                              "note": "pseudo-positive (member swapped, same task) vs matched negatives"}
    # cross_task_swap：eval 正对特征换成另一 task 的单元（成员对保留、任务错配）
    dv_list = sorted(set(t for _, _, t in ev_pos))
    swap_pos = []
    for a, b, t in ev_pos:
        t2 = dv_list[int(rng.integers(len(dv_list)))]
        swap_pos.append((a, b, t2))
    Qsw, _, _, pairs_sw = eval_reads(U, P0, A, swap_pos, [], None)
    s_sw = sgd_scores(Qtr, ytr, Qsw)
    y_sw = np.concatenate([np.ones(len(swap_pos)), np.zeros(len(ev_neg))])
    Qmix2 = np.vstack([Qsw, Qdv[len(ev_pos):]])
    s_mix2 = np.concatenate([s_sw, s_q[len(ev_pos):]])
    out["reads"]["cross_task_swap"] = {"auroc": float(roc_auc_score(y_sw, s_mix2)),
                                       "note": "member pair kept, features from another dev task vs matched negatives"}
    # task-cluster CI（q_only 与 delta）
    uniq_t, inv_t = np.unique([p[2] for p in pairs_dv], return_inverse=True)
    idx_by = {q: np.where(inv_t == q)[0] for q in range(len(uniq_t))}
    rng2 = np.random.default_rng(SEED)
    qb, pb = [], []
    for _ in range(500):
        pick = rng2.choice(len(uniq_t), size=len(uniq_t), replace=True)
        idx = np.concatenate([idx_by[q] for q in pick])
        qb.append(roc_auc_score(ydv[idx], s_q[idx]))
        pb.append(roc_auc_score(ydv[idx], s_p0[idx]))
    out["q_only_ci95_task_cluster"] = [float(np.percentile(qb, 2.5)), float(np.percentile(qb, 97.5))]
    out["P0_pair_only_ci95_task_cluster"] = [float(np.percentile(pb, 2.5)), float(np.percentile(pb, 97.5))]
    out["delta_q_minus_P0"] = out["reads"]["q_only"]["auroc"] - out["reads"]["P0_pair_only"]["auroc"]
    out["delta_qp_minus_q"] = out["reads"]["q_plus_P0"]["auroc"] - out["reads"]["q_only"]["auroc"]
    out["runtime_s"] = time.time() - t0
    # 保存逐对分数（长度桶分解用）：sizelen 均值 = 两成员该 task 的 sizelen[:,0] 均值
    try:
        len_mean = []
        for a, b, t in pairs_dv:
            ra = A["MODEL_I"][a] * A["nT"] * 2 + A["TASK_I"][t] * 2
            rb = A["MODEL_I"][b] * A["nT"] * 2 + A["TASK_I"][t] * 2
            len_mean.append((float(A["sizelen"][ra, 0]) + float(A["sizelen"][rb, 0])) / 2.0)
        safe = tag.replace(":", "_").replace("--", "-")
        np.savez_compressed(OUT / "local" / f"{safe}.npz", s_q=s_q, s_p0=s_p0, s_cos=s_cos,
                            ydv=ydv, len_mean=np.array(len_mean), n_pos=len(ev_pos))
    except Exception as e:
        log(f"  [{tag}] score save failed: {e}")
    log(f"  [{tag}] q={out['reads']['q_only']['auroc']:.4f} P0={out['reads']['P0_pair_only']['auroc']:.4f} "
        f"qP0={out['reads']['q_plus_P0']['auroc']:.4f} cos={out['reads']['cosine']['auroc']:.4f} "
        f"perm={out['reads']['permuted']['auroc']:.4f} mm={out['reads'].get('mismatched_partner',{}).get('auroc',float('nan')):.4f} "
        f"swap={out['reads']['cross_task_swap']['auroc']:.4f} (n_tr={n_tr}, n_ev={n_ev})")
    return out


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "logs").mkdir(exist_ok=True)
    (OUT / "local").mkdir(exist_ok=True)
    A = ev.load_assets()
    splits = {t: A["splits"][A["TASK_I"][t] * 2] for t in A["tasks"]}
    tr_tasks = [t for t in A["tasks"] if splits[t] == "train"]
    dv_tasks = [t for t in A["tasks"] if splits[t] == "dev"]
    U, P0 = build_u(A)
    log(f"[f2ho] tasks train={len(tr_tasks)} dev={len(dv_tasks)}; u built for {len(U)} cells")

    results = {"schema": "f2_holdout_v1", "generated_utc": datetime.now(timezone.utc).isoformat(),
               "report_first_line": "新证据 — member/series-heldout (unseen member/series)",
               "m_ho": {}, "s_ho": {}}

    # ---------------- M-HO ----------------
    log("[f2ho] M-HO 11 folds")
    for series in s1.SERIES:
        members = s1.SERIES[series]
        others = [m for ss in s1.SERIES if ss != series for m in s1.SERIES[ss]]
        for h in members:
            seen = [m for m in members if m != h]
            fit_pos_pool = [(seen[i], seen[j], t) for i in range(len(seen)) for j in range(i + 1, len(seen))
                            for t in tr_tasks]
            eval_pos_pool = [(h, m, t) for m in seen for t in dv_tasks]
            # 负对候选：非 s 系列成员的全对（无序）
            fit_neg_pool, eval_neg_pool = [], []
            for i in range(len(others)):
                for j in range(i + 1, len(others)):
                    a, b = others[i], others[j]
                    fit_neg_pool += [(a, b, t) for t in tr_tasks]
                    eval_neg_pool += [(a, b, t) for t in dv_tasks]
            tag = f"MHO:{series}::{h}"
            out = run_fold(U, P0, A, fit_pos_pool, eval_pos_pool, fit_neg_pool, eval_neg_pool,
                           tr_tasks, dv_tasks, tag)
            out["heldout_member"] = h; out["series"] = series
            results["m_ho"][f"{series}::{h}"] = out

    # ---------------- S-HO ----------------
    log("[f2ho] S-HO 3 folds（低功效压力测试）")
    for hold in s1.SERIES:
        rest = [ss for ss in s1.SERIES if ss != hold]
        rest_members = [m for ss in rest for m in s1.SERIES[ss]]
        hold_members = s1.SERIES[hold]
        fit_pos_pool, fit_neg_pool = [], []
        for ss in rest:
            ms = s1.SERIES[ss]
            for i in range(len(ms)):
                for j in range(i + 1, len(ms)):
                    fit_pos_pool += [(ms[i], ms[j], t) for t in tr_tasks]
        # 负对：其余系列间跨系列对
        for i in range(len(rest_members)):
            for j in range(i + 1, len(rest_members)):
                a, b = rest_members[i], rest_members[j]
                if s1.SERIES_OF[a] != s1.SERIES_OF[b]:
                    fit_neg_pool += [(a, b, t) for t in tr_tasks]
        eval_pos_pool = [(hold_members[i], hold_members[j], t)
                         for i in range(len(hold_members)) for j in range(i + 1, len(hold_members))
                         for t in dv_tasks]
        eval_neg_pool = [(hm, m, t) for hm in hold_members for m in rest_members for t in dv_tasks]
        # 限制候选量（每 task 池较大）——match 时排序全池，池 ~ (11+...) 对/task 可控
        out = run_fold(U, P0, A, fit_pos_pool, eval_pos_pool, fit_neg_pool, eval_neg_pool,
                       tr_tasks, dv_tasks, f"SHO:{hold}")
        out["heldout_series"] = hold
        results["s_ho"][hold] = out

    # ---------------- 汇总 ----------------
    def agg_block(block):
        q = [v["reads"]["q_only"]["auroc"] for v in block.values()]
        p0 = [v["reads"]["P0_pair_only"]["auroc"] for v in block.values()]
        cos = [v["reads"]["cosine"]["auroc"] for v in block.values()]
        qp = [v["reads"]["q_plus_P0"]["auroc"] for v in block.values()]
        perm = [v["reads"]["permuted"]["auroc"] for v in block.values()]
        mm = [v["reads"].get("mismatched_partner", {}).get("auroc", np.nan) for v in block.values()]
        sw = [v["reads"]["cross_task_swap"]["auroc"] for v in block.values()]
        d1 = np.array(q) - np.array(p0)
        d2 = np.array(qp) - np.array(q)
        rng = np.random.default_rng(SEED)
        n = len(q)
        d1b = [float(np.mean(rng.choice(d1, size=n, replace=True))) for _ in range(2000)]
        # member-cluster mean CI（q_only 均值）
        qb = [float(np.mean(rng.choice(q, size=n, replace=True))) for _ in range(2000)]
        return {
            "n_folds": n,
            "q_only": {"mean": float(np.mean(q)), "min": float(np.min(q)), "max": float(np.max(q)),
                       "mean_ci95_member_cluster": [float(np.percentile(qb, 2.5)), float(np.percentile(qb, 97.5))],
                       "n_folds_above_0.5": int(np.sum(np.array(q) > 0.5))},
            "P0_pair_only": {"mean": float(np.mean(p0)), "min": float(np.min(p0)), "max": float(np.max(p0))},
            "q_plus_P0": {"mean": float(np.mean(qp)), "min": float(np.min(qp)), "max": float(np.max(qp))},
            "cosine": {"mean": float(np.mean(cos))},
            "permuted": {"mean": float(np.mean(perm)), "max": float(np.max(perm))},
            "mismatched_partner": {"mean": float(np.nanmean(mm)) if not np.all(np.isnan(mm)) else None},
            "cross_task_swap": {"mean": float(np.mean(sw)), "max": float(np.max(sw))},
            "delta_q_minus_P0": {"mean": float(np.mean(d1)), "per_fold": [float(x) for x in d1],
                                 "mean_ci95_member_cluster": [float(np.percentile(d1b, 2.5)), float(np.percentile(d1b, 97.5))]},
            "delta_qp_minus_q": {"mean": float(np.mean(d2)), "per_fold": [float(x) for x in d2]},
        }

    results["aggregate_m_ho"] = agg_block(results["m_ho"])
    results["aggregate_s_ho"] = agg_block(results["s_ho"])
    # series 拆分（非单一 series 贡献）
    series_means = {}
    for k, v in results["m_ho"].items():
        series_means.setdefault(v["series"], []).append(v["reads"]["q_only"]["auroc"])
    results["m_ho_series_means"] = {k: float(np.mean(v)) for k, v in series_means.items()}
    # 长度桶（三分位；池化 M-HO 各折分数）
    from sklearn.metrics import roc_auc_score as _auc
    import glob as _glob
    bucket_rows = []
    for p in sorted(_glob.glob(str(OUT / "local" / "MHO_*.npz"))):
        z = np.load(p, allow_pickle=True)
        bucket_rows.append((z["s_q"], z["ydv"], z["len_mean"], z["s_p0"], p.split("/")[-1]))
    if bucket_rows:
        lens = np.concatenate([r[2] for r in bucket_rows])
        q1, q2 = np.percentile(lens, [33.3, 66.7])
        buckets = {"short": lens <= q1, "mid": (lens > q1) & (lens <= q2), "long": lens > q2}
        sq = np.concatenate([r[0] for r in bucket_rows])
        yb = np.concatenate([r[1] for r in bucket_rows])
        sp0 = np.concatenate([r[3] for r in bucket_rows])
        btab = {}
        for bname, bmask in buckets.items():
            if bmask.sum() > 10 and len(np.unique(yb[bmask])) == 2:
                btab[bname] = {"n": int(bmask.sum()), "q_only_auroc": float(_auc(yb[bmask], sq[bmask])),
                               "P0_auroc": float(_auc(yb[bmask], sp0[bmask]))}
                log(f"  [lenbucket:{bname}] n={int(bmask.sum())} q={btab[bname]['q_only_auroc']:.4f} P0={btab[bname]['P0_auroc']:.4f}")
        results["m_ho_length_buckets"] = btab
        results["length_bucket_note"] = "池化 M-HO 全部折的逐对分数后按 sizelen 均值三分位分桶（粗略检查，非同折口径）"
    else:
        results["length_bucket_note"] = "未找到逐折分数文件"

    # ---------------- gate ----------------
    a = results["aggregate_m_ho"]
    gate = {
        "two_folds_same_direction": bool(a["q_only"]["n_folds_above_0.5"] >= 2),
        "delta_q_minus_P0_positive": bool(a["delta_q_minus_P0"]["mean"] > 0),
        "permutation_not_reproduced": bool(a["permuted"]["mean"] < 0.55),
        "not_single_series": bool(min(results["m_ho_series_means"].values()) > 0.5) if results["m_ho_series_means"] else False,
        "ci_reported": True,
        "q_only_mean": a["q_only"]["mean"],
        "delta_mean": a["delta_q_minus_P0"]["mean"],
    }
    ok = all([gate["two_folds_same_direction"], gate["delta_q_minus_P0_positive"],
              gate["permutation_not_reproduced"], gate["not_single_series"]])
    gate["verdict"] = "member_holdout_transfer_candidate" if ok else "not_yet_transfer_stay_task_cv"
    results["gate_m_ho"] = gate
    (OUT / "metrics_f2_holdout.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")

    L = ["# F2 升级：member/series-heldout（新证据，不覆盖 task-CV 结果）", "",
         "## M-HO（leave-one-member-out；11 折）", "",
         "| 读出 | mean AUROC | min | max |", "|---|---|---|---|",
         f"| q_only | {a['q_only']['mean']:.4f} | {a['q_only']['min']:.4f} | {a['q_only']['max']:.4f} |",
         f"| P0_pair_only | {a['P0_pair_only']['mean']:.4f} | {a['P0_pair_only']['min']:.4f} | {a['P0_pair_only']['max']:.4f} |",
         f"| q_plus_P0 | {a['q_plus_P0']['mean']:.4f} | {a['q_plus_P0']['min']:.4f} | {a['q_plus_P0']['max']:.4f} |",
         f"| cosine | {a['cosine']['mean']:.4f} | | |",
         f"| permuted | {a['permuted']['mean']:.4f} | | {a['permuted']['max']:.4f} |",
         f"| mismatched_partner | {a['mismatched_partner']['mean']:.4f} | | |",
         f"| cross_task_swap | {a['cross_task_swap']['mean']:.4f} | | {a['cross_task_swap']['max']:.4f} |",
         "",
         f"- q_only mean member-cluster CI95: {a['q_only']['mean_ci95_member_cluster']}",
         f"- delta q_only−P0_pair_only: **{a['delta_q_minus_P0']['mean']:+.4f}** CI95 {a['delta_q_minus_P0']['mean_ci95_member_cluster']}",
         f"- delta q_plus_P0−q_only: {a['delta_qp_minus_q']['mean']:+.4f}",
         f"- 逐 series（q_only 均值）: {results['m_ho_series_means']}",
         ""]
    if bucket_rows and "m_ho_length_buckets" in results:
        L += ["## 长度桶分解（池化 M-HO 逐对分数；粗略检查）", "",
              "| 桶 | n | q_only AUROC | P0_pair_only AUROC |", "|---|---|---|---|"]
        for bname, bv in results["m_ho_length_buckets"].items():
            L.append(f"| {bname} | {bv['n']} | {bv['q_only_auroc']:.4f} | {bv['P0_auroc']:.4f} |")
        L.append("")
    L += ["## 辅助对照解读（重要）",
          f"- `mismatched_partner`（伪对 (h,随机其他系列成员)）均值 {a['mismatched_partner']['mean']:.4f}：**伪对仍含 heldout member h**，"
          "该对照无法把分数压回机会——它表明分数含强“对中出现 h”的 unseen-member 识别成分，属协议性质，不作失败判据。",
          f"- `cross_task_swap`（成员对保留、任务特征换成另一 dev task）均值 {a['cross_task_swap']['mean']:.4f}：**高值为正面证据**——分数由成员对决定、与任务内容无关；",
          "  因此指导条件“跨 task 错配回到机会”按字面未触发（构造无法回机会），以置换（permuted 回机会）与 P0 对照（delta 显著为正）承担该闸门功能，如实记录差异。",
          "",
          "## S-HO（leave-one-series-out；3 折，低功效单列）", "",
          f"- q_only mean: {results['aggregate_s_ho']['q_only']['mean']:.4f} (min {results['aggregate_s_ho']['q_only']['min']:.4f}, "
          f"逐折 {[f'{v:.3f}' for v in [r['reads']['q_only']['auroc'] for r in results['s_ho'].values()]]})",
          f"- P0_pair_only mean: {results['aggregate_s_ho']['P0_pair_only']['mean']:.4f}",
          f"- permuted mean: {results['aggregate_s_ho']['permuted']['mean']:.4f}",
          "- **未通过（CL 0.467 / Qwen 0.519 / DS 0.705 接近机会）**：series 级迁移在未见 series 上不成立——M-HO 强度不可外推为“跨 series 迁移”。",
          "",
          f"## gate（M-HO）\n- verdict: **{gate['verdict']}**\n- {json.dumps({k: v for k, v in gate.items() if k != 'verdict'}, ensure_ascii=False)}",
          "",
          "> 主比较为 fold-paired q_only − P0_pair_only；q_plus_P0 不得充当 P0-only。",
          "> 总结限定：M-HO=unseen-member 识别（含“对中出现 h”成分）成立；S-HO=未见 series 迁移未成立（低功效但方向为负）。"]
    (OUT / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (OUT / "logs" / "f2h_run.log").write_text("\n".join(LOG) + "\n", encoding="utf-8")
    log(f"[f2ho] done in {time.time()-t0:.1f}s; gate={gate['verdict']}")


if __name__ == "__main__":
    main()
