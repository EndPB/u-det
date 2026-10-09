"""C0 逐点对账：服务器 C0 vs 本机强 P0 逐行参考（按预注册 hypothesis.md 执行）。

产物：p0_alignment.json / p0_alignment_per_fold.csv / unified_p0_deltas.json
用法：cd /root/autodl-tmp/u-det && OMP_NUM_THREADS=8 python scripts/cc_c0_align.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cc_common as cc

ROOT = cc.ROOT
OUT = ROOT / "d-det/artifacts/code_conditioned_c0_alignment_2026-10-09"
REF_DIR = ROOT / "d-det/artifacts/code_conditioned_p0_reference_2026-10-09/folds"
C0_LOCAL = ROOT / "d-det/artifacts/code_conditioned_c0_strong_p0_2026-10-09/local"
C1_LOCAL = ROOT / "d-det/artifacts/code_conditioned_c1_prompt_conditioned_2026-10-09/local"
C2_LOCAL = ROOT / "d-det/artifacts/code_conditioned_c2_static_proxy_2026-10-09/local"
C3_LOCAL = ROOT / "d-det/artifacts/code_conditioned_c3_invariance_2026-10-09/local"
SP_LOCAL = ROOT / "d-det/artifacts/code_conditioned_static_preflight_reference_2026-10-09/static_proxies.jsonl"
SP_MINE = ROOT / "d-det/artifacts/code_conditioned_design_2026-10-09/static_preflight/static_proxies.jsonl"


def npz_name(member: str, prefix: str = "scores_dev_fold") -> str:
    return f"{prefix}{member.replace('--', '_')}.npz"


def rank_map(keys):
    return {k: i for i, k in enumerate(keys)}


def sort_by_task(y, s, tp):
    order = np.argsort(tp, kind="mergesort")
    return y[order], s[order], tp[order]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "logs").mkdir(exist_ok=True)
    design = cc.load_design()
    b = design["bundle"]
    members = design["members_order"]
    tasks_all = design["tasks_all"]
    member_idx, task_idx = b["member_idx"], b["task_idx"]
    member_names = [members[i] for i in member_idx]
    task_names = [tasks_all[i] for i in task_idx]

    per_fold = {}
    metrics_mine, metrics_ref = {}, {}
    scores_mine, scores_ref = {}, {}
    labels, taskpos = {}, {}
    order_t_store = {}

    for fold in design["folds"]:
        h = fold["heldout_generator_member"]
        family = design["member_series"][h]
        # 参考 npz 块顺序 = 成员名 ASCII 排序（经三折 y 模式反推；不是 members_order 过滤）
        eval_members = sorted(m for m in members
                              if m == h or design["member_series"][m] != family)
        n_blocks = len(eval_members)
        n_eval_exp = fold["eval_rows"]
        assert n_blocks * 171 == n_eval_exp, (h, n_blocks)

        z_my = np.load(C0_LOCAL / npz_name(h))
        ev_rows = z_my["ev_rows"]
        assert len(ev_rows) == n_eval_exp, (h, len(ev_rows))
        my_member = [member_names[i] for i in ev_rows]
        my_task = [task_names[i] for i in ev_rows]
        my_y = z_my["y"].astype(int)

        z_ref = np.load(REF_DIR / h / "p0_scores.npz")
        ref_task, ref_y = z_ref["task"], z_ref["y"].astype(int)
        assert len(ref_task) == n_eval_exp and len(ref_y) == n_eval_exp

        dev_tasks = sorted(set(my_task))
        assert len(dev_tasks) == 171
        for k, m in enumerate(eval_members):
            blk = slice(k * 171, (k + 1) * 171)
            assert list(ref_task[blk]) == dev_tasks, (h, k)
            expect_pos = 1 if m == h else 0
            assert set(ref_y[blk]) == {expect_pos}, (h, k, m)

        # 行映射：my (member, task) -> ref 行号
        taskrank = rank_map(dev_tasks)
        blockrank = rank_map(eval_members)
        ref_idx = np.array([blockrank[m] * 171 + taskrank[t]
                            for m, t in zip(my_member, my_task)])
        assert sorted(ref_idx.tolist()) == list(range(n_eval_exp)), "ref_idx not bijective"
        y_match = bool(np.array_equal(my_y, ref_y[ref_idx]))
        assert y_match, f"y mismatch fold {h}"

        # 行对齐：以服务器行序为准，参考分数经 ref_idx 取数（ref_idx 是双射）
        tp_sorted = np.array([taskrank[t] for t in my_task])
        mine_f = z_my["fused"]
        ref_f = z_ref["p0"][ref_idx]
        mine_comp = {k: z_my[k] for k in
                     ("s_semantic", "s_char_tfidf", "s_word_tfidf", "s_style_meta")}
        ref_comp = {k: z_ref[k][ref_idx] for k in
                    ("component_linear", "component_char", "component_word", "component_style")}

        # 按 (taskpos) 排序以便 bootstrap（同一个 taskrank 连续）
        order_t = np.argsort(tp_sorted, kind="mergesort")
        y_t = my_y[order_t]
        tp_t = tp_sorted[order_t]
        mf_t = mine_f[order_t]
        rf_t = ref_f[order_t]
        order_t_store[h] = order_t

        d = mine_f - ref_f  # 行对齐后的行差
        pear = float(np.corrcoef(mine_f, ref_f)[0, 1])

        def spearman(a, bx):
            ra = np.argsort(np.argsort(a))
            rb = np.argsort(np.argsort(bx))
            return float(np.corrcoef(ra, rb)[0, 1])

        comp_pairs = {
            "semantic_vs_linear": ("s_semantic", "component_linear"),
            "char_vs_char": ("s_char_tfidf", "component_char"),
            "word_vs_word": ("s_word_tfidf", "component_word"),
            "style_meta_vs_style": ("s_style_meta", "component_style"),
        }
        comps = {}
        for name, (ka, kb) in comp_pairs.items():
            a, bx = mine_comp[ka], ref_comp[kb]
            comps[name] = {
                "pearson": float(np.corrcoef(a, bx)[0, 1]),
                "spearman": spearman(a, bx),
                "auroc_mine": cc.auroc(my_y, a),
                "auroc_ref": cc.auroc(ref_y, bx),
            }

        m_row = cc.auroc(y_t, mf_t)
        r_row = cc.auroc(y_t, rf_t)
        m_tm = cc.task_macro_auroc(y_t, mf_t, tp_t)
        r_tm = cc.task_macro_auroc(y_t, rf_t, tp_t)

        per_fold[h] = {
            "eval_members": eval_members,
            "n_eval": int(n_eval_exp),
            "rows_mapped": int(len(ref_idx)),
            "y_match": y_match,
            "fused": {
                "max_abs_diff": float(np.abs(d).max()),
                "mean_abs_diff": float(np.abs(d).mean()),
                "pearson": pear,
                "spearman": spearman(mine_f, ref_f),
            },
            "components": comps,
            "metrics": {
                "row_auroc_mine": m_row, "row_auroc_ref": r_row,
                "delta_row": m_row - r_row,
                "task_macro_mine": m_tm, "task_macro_ref": r_tm,
                "delta_tm": m_tm - r_tm,
            },
        }
        metrics_mine[h] = (m_row, m_tm)
        metrics_ref[h] = (r_row, r_tm)
        scores_mine[h] = mf_t
        scores_ref[h] = rf_t
        labels[h] = y_t
        taskpos[h] = tp_t
        print(f"[{h.split('--')[-1][:34]:36s}] max|dfused|={np.abs(d).max():.4g} "
              f"r={pear:.6f} row:{m_row:.4f}/{r_row:.4f} tm:{m_tm:.4f}/{r_tm:.4f}", flush=True)

    # 汇总
    mean_row_mine = float(np.mean([v[0] for v in metrics_mine.values()]))
    mean_tm_mine = float(np.mean([v[1] for v in metrics_mine.values()]))
    mean_row_ref = float(np.mean([v[0] for v in metrics_ref.values()]))
    mean_tm_ref = float(np.mean([v[1] for v in metrics_ref.values()]))

    delta = cc.delta_bootstrap(scores_mine, scores_ref, labels, taskpos)
    d_row = float(np.mean(delta["row_level"]))
    d_row_ci = cc.ci(delta["row_level"])
    d_tm = float(np.mean(delta["task_macro"]))
    d_tm_ci = cc.ci(delta["task_macro"])

    # 预注册判定
    all_max = max(v["fused"]["max_abs_diff"] for v in per_fold.values())
    all_drow = max(abs(v["metrics"]["delta_row"]) for v in per_fold.values())
    all_dtm = max(abs(v["metrics"]["delta_tm"]) for v in per_fold.values())
    min_pear = min(v["fused"]["pearson"] for v in per_fold.values())
    if all_max <= 1e-3 and all_drow <= 1e-3 and all_dtm <= 1e-3:
        verdict = "aligned_strict"
    elif all_drow <= 1e-3 and all_dtm <= 1e-3 and min_pear >= 0.999:
        verdict = "aligned_at_metric_level"
    else:
        verdict = "not_aligned"

    align = {
        "schema": "code_conditioned_c0_alignment_v1",
        "reference": "code_conditioned_p0_reference_2026-10-09 (11 folds, fit-only)",
        "server": "code_conditioned_c0_strong_p0_2026-10-09 (frozen W3 spec)",
        "per_fold": per_fold,
        "aggregate": {
            "mean_row_auroc_mine": mean_row_mine,
            "mean_row_auroc_ref": mean_row_ref,
            "delta_row_mean": d_row, "delta_row_ci95": d_row_ci,
            "mean_task_macro_mine": mean_tm_mine,
            "mean_task_macro_ref": mean_tm_ref,
            "delta_tm_mean": d_tm, "delta_tm_ci95": d_tm_ci,
            "max_max_abs_diff": all_max,
            "max_abs_delta_row": all_drow,
            "max_abs_delta_tm": all_dtm,
            "min_pearson_fused": min_pear,
        },
        "verdict": verdict,
        "tolerance_prereg": {
            "aligned_strict": "max|Δfused|<=1e-3 and |Δrow|<=1e-3 and |Δtm|<=1e-3 all folds",
            "aligned_at_metric_level": "|Δrow|<=1e-3 and |Δtm|<=1e-3 and pearson>=0.999 all folds",
        },
    }
    (OUT / "p0_alignment.json").write_text(json.dumps(align, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"VERDICT: {verdict}; mean row {mean_row_mine:.4f} vs ref {mean_row_ref:.4f}; "
          f"mean tm {mean_tm_mine:.4f} vs ref {mean_tm_ref:.4f}")

    with (OUT / "p0_alignment_per_fold.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["heldout", "max_abs_diff_fused", "pearson_fused", "row_mine", "row_ref",
                    "delta_row", "tm_mine", "tm_ref", "delta_tm",
                    "semantic_vs_linear_pearson", "char_pearson", "word_pearson", "style_pearson"])
        for h, v in per_fold.items():
            w.writerow([h, f"{v['fused']['max_abs_diff']:.6g}", f"{v['fused']['pearson']:.6f}",
                        f"{v['metrics']['row_auroc_mine']:.6f}", f"{v['metrics']['row_auroc_ref']:.6f}",
                        f"{v['metrics']['delta_row']:.6f}", f"{v['metrics']['task_macro_mine']:.6f}",
                        f"{v['metrics']['task_macro_ref']:.6f}", f"{v['metrics']['delta_tm']:.6f}",
                        f"{v['components']['semantic_vs_linear']['pearson']:.6f}",
                        f"{v['components']['char_vs_char']['pearson']:.6f}",
                        f"{v['components']['word_vs_word']['pearson']:.6f}",
                        f"{v['components']['style_meta_vs_style']['pearson']:.6f}"])

    # ---------- 统一 P0 下重算 C1/C2/C3（以及 对服务器C0 的一致性校验） ----------
    candidates = {}
    for key, dirp, npzpref, keys in [
        ("c1", C1_LOCAL, "scores_fold", ["code_only_mlp", "full_mlp", "ht_hy_mlp",
                                          "code_only_linear", "full_linear", "psi_only_mlp",
                                          "prompt_only_mlp"]),
        ("c2", C2_LOCAL, "scores_fold", ["s_c2"]),
        ("c3", C3_LOCAL, "scores_fold", ["s_orig"]),
    ]:
        for ck in keys:
            cand_scores = {}
            ok = True
            for fold in design["folds"]:
                h = fold["heldout_generator_member"]
                z = np.load(dirp / f"{npzpref}{h.replace('--', '_')}.npz")
                ev = z["ev_rows"]
                z0 = np.load(C0_LOCAL / npz_name(h))
                if not np.array_equal(ev, z0["ev_rows"]):
                    ok = False
                    break
                cand = np.asarray(z[ck])
                assert len(cand) == len(labels[h]), (h, ck)
                assert np.array_equal(np.asarray(z["y"])[order_t_store[h]], labels[h]), (h, ck)
                cand_scores[h] = cand[order_t_store[h]]
            if not ok:
                print("skip (ev_rows mismatch)", key, ck)
                continue
            dn = cc.delta_bootstrap(cand_scores, scores_ref, labels, taskpos)
            dc = cc.delta_bootstrap(cand_scores, scores_mine, labels, taskpos)
            per_fold_tm = {h: per_fold_tm_delta(cand_scores[h], scores_ref[h], labels[h], taskpos[h])
                           for h in cand_scores}
            per_fold_row = {h: cc.auroc(labels[h], cand_scores[h]) - cc.auroc(labels[h], scores_ref[h])
                            for h in cand_scores}
            candidates[f"{key}.{ck}"] = {
                "vs_unified_p0": {
                    "delta_row_mean": float(np.mean(dn["row_level"])),
                    "delta_row_ci95": cc.ci(dn["row_level"]),
                    "delta_tm_mean": float(np.mean(dn["task_macro"])),
                    "delta_tm_ci95": cc.ci(dn["task_macro"]),
                    "folds_positive_tm": int(sum(v > 0 for v in per_fold_tm.values())),
                    "per_fold_delta_tm": per_fold_tm,
                    "per_fold_delta_row": per_fold_row,
                },
                "vs_server_c0": {
                    "delta_row_mean": float(np.mean(dc["row_level"])),
                    "delta_row_ci95": cc.ci(dc["row_level"]),
                    "delta_tm_mean": float(np.mean(dc["task_macro"])),
                    "delta_tm_ci95": cc.ci(dc["task_macro"]),
                },
            }
            u = candidates[f"{key}.{ck}"]["vs_unified_p0"]
            c = candidates[f"{key}.{ck}"]["vs_server_c0"]
            print(f"{key}.{ck:18s} vs P0(ref): dRow {u['delta_row_mean']:+.4f} dTM {u['delta_tm_mean']:+.4f} "
                  f"({u['folds_positive_tm']}/11) | vs C0: dTM {c['delta_tm_mean']:+.4f}", flush=True)

    (OUT / "unified_p0_deltas.json").write_text(json.dumps({
        "schema": "code_conditioned_unified_p0_deltas_v1",
        "note": "candidates re-scored against the local fit-only strong P0 reference rows; "
                "paired task-cluster bootstrap with shared indices (seed 20261009, B=500)",
        "c0_vs_unified_p0": {
            "delta_row_mean": d_row, "delta_row_ci95": d_row_ci,
            "delta_tm_mean": d_tm, "delta_tm_ci95": d_tm_ci,
        },
        "candidates": candidates,
    }, ensure_ascii=False, indent=1), encoding="utf-8")

    # ---------- static proxies 对比 ----------
    def load_sp(p):
        out = {}
        with p.open(encoding="utf-8") as f:
            for line in f:
                o = json.loads(line)
                out[(o["model_id"], o["task_id"])] = o
        return out

    sp_l, sp_m = load_sp(SP_LOCAL), load_sp(SP_MINE)
    keys_l, keys_m = set(sp_l), set(sp_m)
    same_sha = sum(1 for k in keys_l & keys_m if sp_l[k].get("solution_sha256") == sp_m[k].get("solution_sha256"))
    same_split = sum(1 for k in keys_l & keys_m if sp_l[k].get("split") == sp_m[k].get("split"))
    proxy_equal = sum(1 for k in keys_l & keys_m if sp_l[k].get("static_proxy") == sp_m[k].get("static_proxy"))
    n_common = len(keys_l & keys_m)
    sp_diff = {"n_local": len(sp_l), "n_mine": len(sp_m), "n_common": n_common,
               "only_local": len(keys_l - keys_m), "only_mine": len(keys_m - keys_l),
               "same_solution_sha256": same_sha, "same_split": same_split,
               "static_proxy_deep_equal": proxy_equal}
    print("static proxies:", sp_diff, flush=True)
    (OUT / "static_proxies_compare.json").write_text(json.dumps(sp_diff, ensure_ascii=False, indent=1), encoding="utf-8")

    print("DONE")


def per_fold_tm_delta(sa, sb, y, tp):
    ta = cc.per_task_auroc(y, sa, tp, 171)
    tb = cc.per_task_auroc(y, sb, tp, 171)
    return float(np.nanmean(ta - tb))


if __name__ == "__main__":
    main()
