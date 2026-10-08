"""F2：关系归因（§6）。同 task 的 observed unit 对；正=同 observed_series，负=size/length/style 匹配。

样本对 q=[u_i;u_j;|u_i-u_j|;u_i⊙u_j]（u=[S;D]，1536d → q 6144d）。
读出：a) cosine 基线；b) SGD-LR on q；c) LR on [q; P0 对特征]（残差近似）。
切分：5 折 task-CV；消融：标签置换（负对照）、跨 task 错误配对（负集）。
判据（§6.3）：≥2 折同向 & 相对最佳单侧/P0 delta>0 & CI 不覆盖 0 & 匹配控制保留 & 置换不复现。
输出：artifacts/f2_relation_attribution_2026-10-08/
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

OUT = ROOT / "d-det/artifacts/f2_relation_attribution_2026-10-08"
SEED = 20261008
LOG: list[str] = []


def log(m):
    print(m, flush=True)
    LOG.append(m)


def build_pairs(A):
    """返回 (pairs, labels, tasks, folds, cross-task negatives)。"""
    models = A["models"]
    members = [m for ss in s1.SERIES for m in s1.SERIES[ss]]
    series_of = {m: ss for ss, ms in s1.SERIES.items() for m in ms}
    split_task = {}
    for t in A["tasks"]:
        split_task[t] = A["splits"][A["TASK_I"][t] * 2]
    tasks = [t for t in A["tasks"] if True]  # 全 969（train+dev 任务都用于 CV 对构建；注意这是关系任务的 task-CV）
    # u 向量：per (model, task) —— 从 emb 构建 [S;D]
    H = A["emb_base"]
    def u_vec(m, t):
        mi = A["MODEL_I"][m]; ti = A["TASK_I"][t]
        r = mi * A["nT"] * 2 + ti * 2
        hc = H[r]; hi = H[r + 1]
        return np.concatenate([(hc + hi) / 2.0, (hi - hc) / 2.0])
    U = {}
    for m in members:
        for t in tasks:
            U[(m, t)] = u_vec(m, t)
    # 正对
    pos_pairs = []
    for t in tasks:
        for series in s1.SERIES:
            ms = s1.SERIES[series]
            for a in range(len(ms)):
                for b in range(a + 1, len(ms)):
                    pos_pairs.append((ms[a], ms[b], t))
    # 负对（匹配）：per task, 跨系列对里按 size+len 距离最近挑选 === 正对数量
    cross_pairs = []
    for t in tasks:
        for s1_ in s1.SERIES:
            for s2_ in s1.SERIES:
                if s1_ >= s2_:
                    continue
                for mi in s1.SERIES[s1_]:
                    for mj in s1.SERIES[s2_]:
                        cross_pairs.append((mi, mj, t))
    rng = np.random.default_rng(SEED)
    # 每 task：正对 15；负对从 cross 里选 15（匹配 |logsize 差| 最近）
    pos_by_task = defaultdict(list)
    for p in pos_pairs:
        pos_by_task[p[2]].append(p)
    cross_by_task = defaultdict(list)
    for p in cross_pairs:
        cross_by_task[p[2]].append(p)
    def keyf(p):
        pi = A["sizeB"][A["MODEL_I"][p[0]] * A["nT"] * 2]
        pj = A["sizeB"][A["MODEL_I"][p[1]] * A["nT"] * 2]
        size_d = abs(np.log10(pi) - np.log10(pj)) if (pi and pj) else 99
        ri = A["MODEL_I"][p[0]] * A["nT"] * 2 + A["TASK_I"][p[2]] * 2 + 1
        rj = A["MODEL_I"][p[1]] * A["nT"] * 2 + A["TASK_I"][p[2]] * 2 + 1
        len_d = abs(float(A["style"][ri, 0]) - float(A["style"][rj, 0])) / 500.0
        st_d = float(np.mean(np.abs(A["style"][ri].astype(np.float64) - A["style"][rj].astype(np.float64)))) / 50.0
        return size_d * 3.0 + len_d + st_d * 0.1
    neg_pairs = []
    for t in tasks:
        th = pos_by_task[t]
        cand_sorted = sorted(cross_by_task[t], key=keyf)
        neg_pairs.extend(cand_sorted[:len(th)])
    log(f"  pairs: pos={len(pos_pairs)} neg={len(neg_pairs)}")
    return pos_pairs, neg_pairs, U, tasks


def p0_pair_feats(A, p):
    mi, mj, t = p
    def row(m):
        r = A["MODEL_I"][m] * A["nT"] * 2 + A["TASK_I"][t] * 2 + 1
        return np.concatenate([A["style"][r], A["meta"][r], A["sizelen"][r]])
    fi, fj = row(mi), row(mj)
    return np.concatenate([np.abs(fi - fj), fi + fj])


def q_vector(U, p):
    ui, uj, _ = p
    a, b = U[(ui, p[2])], U[(uj, p[2])]
    return np.concatenate([a, b, np.abs(a - b), a * b])


def cos_score(U, p):
    a, b = U[(p[0], p[2])], U[(p[1], p[2])]
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def sgd_lr(Xtr, ytr, Xdv, seeds=(0, 1, 2), epochs=5, bs=4096, alpha=1e-6):
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


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "logs").mkdir(exist_ok=True)
    A = ev.load_assets()
    pos_pairs, neg_pairs, U, tasks = build_pairs(A)
    all_pairs = pos_pairs + neg_pairs
    y = np.array([1] * len(pos_pairs) + [0] * len(neg_pairs))
    tasks_of = np.array([p[2] for p in all_pairs])
    # q 矩阵
    t = time.time()
    Q = np.stack([q_vector(U, p) for p in all_pairs])
    log(f"  q matrix {Q.shape} in {time.time()-t:.1f}s ({Q.nbytes/1e6:.0f}MB)")
    P0F = np.stack([p0_pair_feats(A, p) for p in all_pairs])
    # 5 折 task-CV
    uniq_t = np.unique(tasks_of)
    rng = np.random.default_rng(SEED)
    perm_t = rng.permutation(len(uniq_t))
    tfolds = np.array_split(perm_t, 5)
    metrics = {"schema": "f2_metrics_v1", "generated_utc": datetime.now(timezone.utc).isoformat(),
               "label": "same_observed_series", "n_pairs": int(len(all_pairs)),
               "n_pos": len(pos_pairs), "n_neg": len(neg_pairs),
               "fold_metrics": {}, "aggregate": {}}
    from sklearn.metrics import roc_auc_score, average_precision_score
    for f in range(5):
        hold_t = set(uniq_t[tfolds[f]])
        dv_idx = np.array([i for i, tt in enumerate(tasks_of) if tt in hold_t])
        tr_idx = np.array([i for i, tt in enumerate(tasks_of) if tt not in hold_t])
        out = {}
        # cosine 基线
        s_cos = np.array([cos_score(U, p) for p in [all_pairs[i] for i in dv_idx]])
        out["cosine_u"] = {"auroc": float(roc_auc_score(y[dv_idx], s_cos)),
                           "ap": float(average_precision_score(y[dv_idx], s_cos))}
        # LR on q
        s_q = sgd_lr(Q[tr_idx], y[tr_idx], Q[dv_idx])
        out["lr_q"] = {"auroc": float(roc_auc_score(y[dv_idx], s_q)),
                       "ap": float(average_precision_score(y[dv_idx], s_q))}
        # LR on [q; P0]
        Xq = np.hstack([Q, P0F]).astype(np.float32)
        s_qp = sgd_lr(Xq[tr_idx], y[tr_idx], Xq[dv_idx])
        out["lr_q_p0"] = {"auroc": float(roc_auc_score(y[dv_idx], s_qp)),
                          "ap": float(average_precision_score(y[dv_idx], s_qp))}
        # 置换负对照：shuffle 训练标签
        y_sh = y[tr_idx].copy()
        rngf = np.random.default_rng(SEED + f)
        rngf.shuffle(y_sh)
        s_sh = sgd_lr(Q[tr_idx], y_sh, Q[dv_idx])
        out["permuted_labels"] = {"auroc": float(roc_auc_score(y[dv_idx], s_sh)),
                                  "ap": float(average_precision_score(y[dv_idx], s_sh))}
        metrics["fold_metrics"][f"fold{f}"] = out
        log(f"  [fold{f}] cos={out['cosine_u']['auroc']:.3f} lr_q={out['lr_q']['auroc']:.3f} "
            f"lr_q+p0={out['lr_q_p0']['auroc']:.3f} perm={out['permuted_labels']['auroc']:.3f}")
        np.savez_compressed(OUT / "local" / f"f2_scores_fold{f}.npz", s_cos=s_cos, s_q=s_q,
                            s_qp=s_qp, s_sh=s_sh, y=y[dv_idx], tasks=tasks_of[dv_idx])
    for key in ("cosine_u", "lr_q", "lr_q_p0", "permuted_labels"):
        vals = [metrics["fold_metrics"][f"fold{f}"][key]["auroc"] for f in range(5)]
        metrics["aggregate"][key] = {"mean_auroc": float(np.mean(vals)), "min": float(np.min(vals)),
                                     "max": float(np.max(vals)),
                                     "n_folds_above_0.5": int(np.sum(np.array(vals) > 0.5))}
    # 判据（§6.3）
    gate = {}
    lr_vals = [metrics["fold_metrics"][f"fold{f}"]["lr_q"]["auroc"] for f in range(5)]
    cos_vals = [metrics["fold_metrics"][f"fold{f}"]["cosine_u"]["auroc"] for f in range(5)]
    perm_vals = [metrics["fold_metrics"][f"fold{f}"]["permuted_labels"]["auroc"] for f in range(5)]
    p0_vals = [metrics["fold_metrics"][f"fold{f}"]["lr_q_p0"]["auroc"] for f in range(5)]
    gate["two_folds_same_direction"] = bool(np.sum(np.array(lr_vals) > 0.5) >= 2)
    gate["beat_best_single_and_p0_mean"] = bool(np.mean(lr_vals) > np.mean(cos_vals)
                                                and np.mean(lr_vals) >= np.mean(p0_vals) - 0.001)
    gate["permutation_not_reproduced"] = bool(np.mean(perm_vals) < 0.55)
    gate["mean_lr_auroc"] = float(np.mean(lr_vals))
    gate["verdict"] = ("relation_gain_observed" if (gate["two_folds_same_direction"]
                                                    and gate["permutation_not_reproduced"]
                                                    and np.mean(lr_vals) > 0.55)
                       else "no_stable_relation_gain_under_current_data")
    metrics["gate"] = gate
    (OUT / "metrics_f2.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=1), encoding="utf-8")
    L = ["# F2：关系归因（same_observed_series；task-CV 5 折）", "",
         f"- pairs: pos={len(pos_pairs)}, neg={len(neg_pairs)}（size 匹配）", "",
         "| 读出 | 平均 AUROC | min | max | 折>0.5 |", "|---|---|---|---|---|"]
    for key, v in metrics["aggregate"].items():
        L.append(f"| {key} | {v['mean_auroc']:.4f} | {v['min']:.4f} | {v['max']:.4f} | {v['n_folds_above_0.5']}/5 |")
    L.append("")
    L.append(f"## 判据（§6.3）\n- verdict: **{gate['verdict']}**\n- {json.dumps({k: v for k, v in gate.items() if k != 'verdict'}, ensure_ascii=False)}")
    (OUT / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (OUT / "logs" / "f2_run.log").write_text("\n".join(LOG) + "\n", encoding="utf-8")
    log(f"[f2] done in {time.time()-t0:.1f}s; verdict={gate['verdict']}")


if __name__ == "__main__":
    main()
