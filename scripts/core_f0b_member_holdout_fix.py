"""F0-B corrective rerun：真正的 series member-heldout（unseen member 评测）。

修复 81c68d2 版 F0-B 的实现错误：旧实现循环 heldout member h 时 units=pos+neg 而 pos 已排除 h，
h 从未进入评测（所谓 heldout member 实际仍在训练中出现的成员集合内）。

正确协议（指导 §2.1）：对 series s 的 heldout member h：
  P_train = s minus {h}（train tasks）；P_eval = {h}（dev tasks）；N = 其他 series 全体
  train units = (P_train ∪ N) on train tasks；eval units = (P_eval ∪ N) on dev tasks
  eval positive（h）与 train model 集合交集必须为 0。
任务：member vs 负集二分类（LR, model-balanced）；视图 h_instruct/delta/u/p0/h_complete。
回传：每 member AUROC/AP + task-cluster CI；与旧（错误）结果并列不覆盖。

输出：artifacts/f0_series_member_holdout_fix_2026-10-08/
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import core_r0_eval as ev  # noqa: E402
import variant_transfer_stage1_execute as s1  # noqa: E402

OUT = ROOT / "d-det/artifacts/f0_series_member_holdout_fix_2026-10-08"
OLD = ROOT / "d-det/artifacts/f0_exact_and_series_attribution_2026-10-08"
SEED = 20261008
VIEWS = ("u", "h_instruct", "delta", "p0", "h_complete")
LOG: list[str] = []


def log(m):
    print(m, flush=True)
    LOG.append(m)


def rows_for(A, models, tasks, mode):
    mi = np.array([A["MODEL_I"][m] for m in models])
    ti = np.array([A["TASK_I"][t] for t in tasks])
    return (mi[:, None] * A["nT"] * 2 + ti[None, :] * 2 + mode).ravel()


def views_for(A, models, tasks):
    H = A["emb_base"]
    Rc = rows_for(A, models, tasks, 0)
    Ri = rows_for(A, models, tasks, 1)
    hc, hi = H[Rc], H[Ri]
    S = (hc + hi) / 2.0
    D = (hi - hc) / 2.0
    return {
        "h_complete": hc, "h_instruct": hi, "delta": D,
        "u": np.hstack([S, D]),
        "p0": np.hstack([A["style"][Rc], A["meta"][Rc], A["sizelen"][Rc]]),
    }


def auroc(y, s):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s))


def task_cluster_ci(A, models, tasks, y, s, nb=500, seed=SEED):
    """按 dev task 簇 bootstrap AUROC。单元布局=model-major（每模型块含全部 tasks）。"""
    ti_ids = np.tile(np.array([A["TASK_I"][t] for t in tasks]), len(models))
    idx_by = {}
    for i, ti in enumerate(ti_ids):
        idx_by.setdefault(int(ti), []).append(i)
    keys = sorted(idx_by)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(nb):
        pick = rng.choice(len(keys), size=len(keys), replace=True)
        idx = np.concatenate([idx_by[keys[q]] for q in pick])
        vals.append(auroc(y[idx], s[idx]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "logs").mkdir(exist_ok=True)
    (OUT / "local").mkdir(exist_ok=True)
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score

    A = ev.load_assets()
    splits = {t: A["splits"][A["TASK_I"][t] * 2] for t in A["tasks"]}
    tr_tasks = [t for t in A["tasks"] if splits[t] == "train"]
    dv_tasks = [t for t in A["tasks"] if splits[t] == "dev"]
    log(f"[fix] tasks train={len(tr_tasks)} dev={len(dv_tasks)}")

    per_member = {}
    for series in s1.SERIES:
        members = s1.SERIES[series]
        for h in members:
            others = [m for ss in s1.SERIES for m in s1.SERIES[ss]]
            P_train = [m for m in members if m != h]
            N = [m for ss in s1.SERIES if ss != series for m in s1.SERIES[ss]]
            train_models = P_train + N
            eval_models = [h] + N
            assert h not in set(train_models), "heldout member leaked into train"
            assert len(set(train_models) & {h}) == 0
            Vtr = views_for(A, train_models, tr_tasks)
            Vdv = views_for(A, eval_models, dv_tasks)
            ytr = np.isin([A["MODEL_I"][m] for m in train_models],
                          [A["MODEL_I"][m] for m in P_train]).astype(int)
            ytr = np.repeat(ytr, len(tr_tasks))
            ydv = np.isin([A["MODEL_I"][m] for m in eval_models],
                          [A["MODEL_I"][h]]).astype(int)
            ydv = np.repeat(ydv, len(dv_tasks))
            # model-balanced 权重
            m_axis = np.repeat(np.array([A["MODEL_I"][m] for m in train_models]), len(tr_tasks))
            cnt = {}
            for m in m_axis:
                cnt[m] = cnt.get(m, 0) + 1
            w = np.array([1.0 / cnt[m] for m in m_axis])
            entry = {"series": series, "heldout_member": h,
                     "train_positive_members": P_train, "eval_positive_member": h,
                     "n_train_units": int(len(ytr)), "n_eval_units": int(len(ydv)),
                     "n_eval_pos": int(ydv.sum()), "n_eval_neg": int((1 - ydv).sum()),
                     "support_ok": bool(len(P_train) >= 1),
                     "views": {}}
            for vname in VIEWS:
                clf = LogisticRegression(max_iter=3000, C=1.0).fit(Vtr[vname], ytr, sample_weight=w)
                s_dv = clf.decision_function(Vdv[vname])
                ent = {"auroc": auroc(ydv, s_dv), "ap": float(average_precision_score(ydv, s_dv))}
                if vname in ("u", "h_instruct", "delta"):
                    ent["auroc_ci95_task_cluster"] = task_cluster_ci(A, eval_models, dv_tasks, ydv, s_dv)
                entry["views"][vname] = ent
            per_member[f"{series}::{h}"] = entry
            log(f"[fix] {series}::{h} u={entry['views']['u']['auroc']:.4f} "
                f"ci={entry['views']['u']['auroc_ci95_task_cluster']}")

    agg = {}
    for vname in VIEWS:
        vals = [v[ "views"][vname]["auroc"] for v in per_member.values()]
        agg[vname] = {"mean_auroc": float(np.mean(vals)), "min": float(np.min(vals)),
                      "max": float(np.max(vals)), "n_folds_above_0.5": int(np.sum(np.array(vals) > 0.5)),
                      "n_folds": len(vals)}
    # 与旧（错误）结果并列
    old_ref = {}
    oldp = OLD / "metrics_f0.json"
    if oldp.exists():
        om = json.loads(oldp.read_text(encoding="utf-8"))
        if "f0b" in om:
            old_ref = om["f0b"]["aggregate"]
    results = {"schema": "f0b_member_holdout_fix_v1",
               "generated_utc": datetime.now(timezone.utc).isoformat(),
               "report_first_line": "corrective rerun — true member-heldout (unseen member)",
               "protocol": ("P_train=s\\{h} on train tasks; P_eval={h} on dev tasks; N=other series; "
                            "eval positive disjoint from train model set (asserted)"),
               "per_member": per_member, "aggregate": agg,
               "previous_errored_result": old_ref,
               "previous_errored_note": ("旧 81c68d2 F0-B: units=pos+neg，heldout member 从未进入评测；"
                                         "该结果仅作 same-member task-heldout 诊断，不得作 unseen-member transfer 证据。"),
               "runtime_seconds": None}
    results["runtime_seconds"] = time.time() - t0
    (OUT / "metrics_f0b_fix.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")

    L = ["# F0-B 修复：true member-heldout（corrective rerun，不覆盖旧结果）", "",
         f"- eval positive 与 train model 集合交集=0（断言通过）；折数={len(per_member)}（4+4+3）", "",
         "## 修复后 aggregate（11 折）", "", "| 视图 | mean AUROC | min | max | 折>0.5 |", "|---|---|---|---|---|"]
    for vname, v in agg.items():
        L.append(f"| {vname} | {v['mean_auroc']:.4f} | {v['min']:.4f} | {v['max']:.4f} | {v['n_folds_above_0.5']}/{v['n_folds']} |")
    if old_ref:
        L += ["", "## 旧（错误）结果并列（仅作 same-member task-heldout 诊断）", "",
              "| 视图 | 旧 mean AUROC（错误口径） |", "|---|---|"]
        for vname, v in old_ref.items():
            L.append(f"| {vname} | {v['mean_auroc']:.4f} |")
    L += ["", "## 逐 member（u 视图）", "", "| member | AUROC | task-cluster CI95 | AP |", "|---|---|---|---|"]
    for k, v in per_member.items():
        u = v["views"]["u"]
        L.append(f"| {k} | {u['auroc']:.4f} | [{u['auroc_ci95_task_cluster'][0]:.3f}, {u['auroc_ci95_task_cluster'][1]:.3f}] | {u['ap']:.4f} |")
    L += ["", "> 报告名：observed series / unseen-member transfer（family_is_confirmed=false）。折支持度不足者照留不删。"]
    (OUT / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (OUT / "logs" / "f0b_fix_run.log").write_text("\n".join(LOG) + "\n", encoding="utf-8")
    log(f"[fix] done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
