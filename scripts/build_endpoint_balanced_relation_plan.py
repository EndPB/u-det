"""端点平衡关系计划（48 折模板 + construction audit）。

由 metrics_f0b_fix.json 的 11 成员（3 系列）生成全部 4x4x3=48 折完整模板：
  每折 = (h_CL, h_Q, h_DS) heldout；训练成员 M_s = series - {h_s}。
训练权重（每类总权 1；每成员端点权两侧相等 = 2/(3n_s)）：
  正对 (a,b)⊂M_s: w=1/(3*C(n_s,2))；负对 a∈M_s,b∈M_r,r≠s: w=1/(3*n_s*n_r)。
评测权重（每 anchor 两侧 = 1/3）：
  正 (h_s,b), b∈M_s: w=1/(3n_s)；负 (h_s,b), b∈M_r: w=1/(6n_r)。
audit：heldout 排除、训练/评测端点边际全检查（含"任意 partner 正权=两个其他 anchor 负权之和"）。

用法：
  python scripts/build_endpoint_balanced_relation_plan.py \
    --members-json d-det/artifacts/f0_series_member_holdout_fix_2026-10-08/metrics_f0b_fix.json \
    --out d-det/artifacts/endpoint_balanced_relation_plan_2026-10-08
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import time
from datetime import datetime, timezone
from fractions import Fraction
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--members-json", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    t0 = time.time()
    src = ROOT / args.members_json
    OUT = ROOT / args.out
    OUT.mkdir(parents=True, exist_ok=True)
    data = json.loads(src.read_text(encoding="utf-8"))
    per_member = data["per_member"]
    series_members = {}
    for key in per_member:
        s, m = key.split("::")
        series_members.setdefault(s, []).append(m)
    series_list = sorted(series_members)
    assert len(series_list) == 3, series_list
    for s in series_list:
        series_members[s] = sorted(series_members[s])
    counts = {s: len(ms) for s, ms in series_members.items()}
    assert sorted(counts.values()) == [3, 4, 4], counts

    folds = []
    combos = itertools.product(*[series_members[s] for s in series_list])
    for idx, held in enumerate(combos):
        hold = dict(zip(series_list, held))
        M = {s: [m for m in series_members[s] if m != hold[s]] for s in series_list}
        n = {s: len(M[s]) for s in series_list}
        fold = {"fold_id": idx, "heldout": hold, "train_members": M, "n_train_members": n,
                "train_pos": [], "train_neg": [], "eval_pos": [], "eval_neg": []}
        for s in series_list:
            w = 1.0 / (3 * comb(n[s], 2))
            for a, b in itertools.combinations(M[s], 2):
                fold["train_pos"].append({"a": a, "b": b, "weight": w,
                                          "series": s})
        for s, r in itertools.permutations(series_list, 2):
            if s >= r:
                continue
            w = 1.0 / (3 * n[s] * n[r])
            for a in M[s]:
                for b in M[r]:
                    fold["train_neg"].append({"a": a, "b": b, "weight": w, "series_pair": [s, r]})
        for s in series_list:
            w = 1.0 / (3 * n[s])
            for b in M[s]:
                fold["eval_pos"].append({"anchor": hold[s], "partner": b, "weight": w, "anchor_series": s})
        for s in series_list:
            for r in series_list:
                if s == r:
                    continue
                w = 1.0 / (6 * n[r])
                for b in M[r]:
                    fold["eval_neg"].append({"anchor": hold[s], "partner": b, "weight": w,
                                             "anchor_series": s, "partner_series": r})
        folds.append(fold)

    # ---------------- construction audit ----------------
    tol = 1e-9
    audits = []
    ok_all = True
    for fold in folds:
        hold = fold["heldout"]; M = fold["train_members"]
        checks = {"heldout_excluded": all(hold[s] not in M[s] for s in series_list),
                  "train_pos_endpoint": True, "train_neg_endpoint": True,
                  "eval_anchor_balanced": True, "eval_partner_balanced": True,
                  "partner_pos_equals_other_anchor_negs": True}
        n = fold["n_train_members"]
        # 训练端点权
        for s in series_list:
            pos_w = {}
            for p in fold["train_pos"]:
                if p["series"] != s:
                    continue
                pos_w[p["a"]] = pos_w.get(p["a"], 0.0) + p["weight"]
                pos_w[p["b"]] = pos_w.get(p["b"], 0.0) + p["weight"]
            neg_w = {}
            for p in fold["train_neg"]:
                if s in p["series_pair"]:
                    neg_w[p["a"]] = neg_w.get(p["a"], 0.0) + p["weight"]
                    neg_w[p["b"]] = neg_w.get(p["b"], 0.0) + p["weight"]
            target = 2.0 / (3 * n[s])
            for m in M[s]:
                if abs(pos_w.get(m, 0.0) - target) > tol or abs(neg_w.get(m, 0.0) - target) > tol:
                    checks["train_pos_endpoint"] = False
                    checks["train_neg_endpoint"] = False
        # 评测 anchor 平衡
        for s in series_list:
            pw = sum(p["weight"] for p in fold["eval_pos"] if p["anchor_series"] == s)
            nw = sum(p["weight"] for p in fold["eval_neg"] if p["anchor_series"] == s)
            if abs(pw - 1/3) > tol or abs(nw - 1/3) > tol:
                checks["eval_anchor_balanced"] = False
        # 评测 partner 平衡：b∈M_r 的总正权（作为 anchor h_r 的 partner）与总负权（作为其他 anchor 的负 partner）
        for r in series_list:
            pos_b = sum(p["weight"] for p in fold["eval_pos"]
                        if p["anchor_series"] == r and p["anchor"] == hold[r])
            for b in M[r]:
                p_w = sum(p["weight"] for p in fold["eval_pos"]
                          if p["anchor_series"] == r and p["anchor"] == hold[r] and p["partner"] == b)
                n_w = sum(p["weight"] for p in fold["eval_neg"]
                          if p["partner_series"] == r and p["partner"] == b)
                if abs(p_w - 1.0 / (3 * n[r])) > tol or abs(n_w - p_w) > tol:
                    checks["eval_partner_balanced"] = False
                # 正权 = 两个其他 anchor 的负权之和
                if abs(n_w - p_w) > tol:
                    checks["partner_pos_equals_other_anchor_negs"] = False
        ok = all(checks.values())
        ok_all = ok_all and ok
        audits.append({"fold_id": fold["fold_id"], "checks": checks, "passed": ok})
    plan = {"schema": "endpoint_balanced_plan_v1",
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "source": {"members_json": args.members_json, "sha256": sha256_of(src)},
            "series": series_list, "series_members": series_members,
            "counts": counts, "n_folds": len(folds),
            "weights_spec": {
                "train_pos": "1/(3*C(n_s,2))", "train_neg": "1/(3*n_s*n_r)",
                "eval_pos": "1/(3*n_s)", "eval_neg": "1/(6*n_r)"},
            "usage_note": ("模板折在展开到 train/dev tasks 时生成特征行：训练权重再除以 train task 数并"
                           "整体缩放至均值 1；评测用加权 AUROC/AP（unweighted 另列）。"),
            "folds": folds}
    (OUT / "endpoint_balanced_plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=1), encoding="utf-8")
    audit = {"schema": "endpoint_balanced_construction_audit_v1",
             "generated_utc": datetime.now(timezone.utc).isoformat(),
             "n_folds": len(folds), "all_passed": bool(ok_all), "folds": audits}
    (OUT / "construction_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[plan] folds={len(folds)} all_passed={ok_all} in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
