"""C0 对账 v2（结构性修正版）：本机 public_series_member_ho_v2 与服务器 C0 的
正确对齐比较。

关键发现（v1 对账暴露）：本机包的 11 个 p0_scores.npz 是同一 11 成员的 LOO 折，
但（a）目录名↔heldout 成员存在置换；（b）其"同系列排除"划分与服务器不同：
    本机: {CL4} / {Q-1.5B,Q-14B,Q-32B} / {Q-7B,DS-1.3b,DS-33b,DS-6.7b}
    服务器: {CL4} / {Q4} / {DS3}
因此仅 CL 四折为同设计；其余折的 eval 集不同。本脚本：
  1. 用 train 侧成员识别（4 模型 char 网格）复推每文件 heldout；与法证表核对；
  2. 用全部 11 折 npz 的 eval 块识别每文件的 slot→成员顺序；
  3. 在"共同成员块"（8 块 × 171，逐折 1368 行）上做逐块相关/分数差与两边指标；
  4. C1/C2/C3 候选在共同行上对 本机 P0 重算 Δ（paired task-cluster bootstrap）。
输出：p0_alignment_v2.json / p0_alignment_v2_per_fold.csv /
      unified_p0_deltas_v2.json / structure_findings.json
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cc_common as cc

ROOT = cc.ROOT
OUT = ROOT / "d-det/artifacts/code_conditioned_c0_alignment_2026-10-09"
REF = ROOT / "d-det/artifacts/code_conditioned_p0_reference_2026-10-09/folds"
C0 = ROOT / "d-det/artifacts/code_conditioned_c0_strong_p0_2026-10-09/local"
C1 = ROOT / "d-det/artifacts/code_conditioned_c1_prompt_conditioned_2026-10-09/local"
C2 = ROOT / "d-det/artifacts/code_conditioned_c2_static_proxy_2026-10-09/local"
C3 = ROOT / "d-det/artifacts/code_conditioned_c3_invariance_2026-10-09/local"

# 法证得到的期望映射（file 目录名 -> heldout 成员）与系列划分（复核用）
EXPECTED_MAP = {
    "Qwen--Qwen2.5-Coder-1.5B-Instruct": "Qwen--Qwen2.5-Coder-7B-Instruct",
    "Qwen--Qwen2.5-Coder-14B-Instruct": "deepseek-ai--deepseek-coder-1.3b-instruct",
    "Qwen--Qwen2.5-Coder-32B-Instruct": "deepseek-ai--deepseek-coder-33b-instruct",
    "Qwen--Qwen2.5-Coder-7B-Instruct": "deepseek-ai--deepseek-coder-6.7b-instruct",
    "codellama--CodeLlama-13b-Instruct-hf": "codellama--CodeLlama-13b-Instruct-hf",
    "codellama--CodeLlama-34b-Instruct-hf": "codellama--CodeLlama-34b-Instruct-hf",
    "codellama--CodeLlama-70b-Instruct-hf": "codellama--CodeLlama-70b-Instruct-hf",
    "codellama--CodeLlama-7b-Instruct-hf": "codellama--CodeLlama-7b-Instruct-hf",
    "deepseek-ai--deepseek-coder-1.3b-instruct": "Qwen--Qwen2.5-Coder-1.5B-Instruct",
    "deepseek-ai--deepseek-coder-33b-instruct": "Qwen--Qwen2.5-Coder-14B-Instruct",
    "deepseek-ai--deepseek-coder-6.7b-instruct": "Qwen--Qwen2.5-Coder-32B-Instruct",
}
LOCAL_PARTITION = {
    "P1": ["codellama--CodeLlama-13b-Instruct-hf", "codellama--CodeLlama-34b-Instruct-hf",
           "codellama--CodeLlama-70b-Instruct-hf", "codellama--CodeLlama-7b-Instruct-hf"],
    "P2": ["Qwen--Qwen2.5-Coder-1.5B-Instruct", "Qwen--Qwen2.5-Coder-14B-Instruct",
           "Qwen--Qwen2.5-Coder-32B-Instruct"],
    "P3": ["Qwen--Qwen2.5-Coder-7B-Instruct", "deepseek-ai--deepseek-coder-1.3b-instruct",
           "deepseek-ai--deepseek-coder-33b-instruct", "deepseek-ai--deepseek-coder-6.7b-instruct"],
}


def tag(m):
    return m.split("--")[-1][:28]


def sgd_ens(Xtr, ytr, Xev, seeds=(0, 1, 2), epochs=5, bs=4096, alpha=1e-6):
    out = np.zeros(Xev.shape[0])
    for s in seeds:
        clf = SGDClassifier(loss="log_loss", alpha=alpha, random_state=s)
        rng = np.random.default_rng(s)
        for _ in range(epochs):
            perm = rng.permutation(Xtr.shape[0])
            for i in range(0, len(perm), bs):
                sel = perm[i:i + bs]
                clf.partial_fit(Xtr[sel], ytr[sel], classes=np.array([0, 1]))
        out += clf.decision_function(Xev)
    return out / len(seeds)


def main():
    t_start = time.time()
    design = cc.load_design()
    b = design["bundle"]
    members = design["members_order"]
    tasks_all = design["tasks_all"]
    rows = design["rows"]
    member_names = np.array([members[i] for i in b["member_idx"]])
    task_names = np.array([tasks_all[i] for i in b["task_idx"]])
    is_train = b["is_train"].astype(bool)
    dev_tasks = sorted(set(task_names[~is_train].tolist()))
    tr_tasks = sorted(set(task_names[is_train].tolist()))
    assert len(dev_tasks) == 171 and len(tr_tasks) == 798
    dv_map = {t: i for i, t in enumerate(dev_tasks)}
    tr_map = {t: i for i, t in enumerate(tr_tasks)}

    # ---------- 1) train 侧 4 模型 char 网格（识别每文件 heldout） ----------
    texts = [r["code"] for r in rows]
    vc = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=1,
                         sublinear_tf=True, lowercase=False)
    X = vc.fit_transform(texts)
    print(f"[align2] tfidf {X.shape} {time.time()-t_start:.0f}s", flush=True)
    grid_models = ["Qwen--Qwen2.5-Coder-7B-Instruct", "Qwen--Qwen2.5-Coder-1.5B-Instruct",
                   "deepseek-ai--deepseek-coder-1.3b-instruct", "codellama--CodeLlama-13b-Instruct-hf"]
    S = {}
    for fold in design["folds"]:
        h = fold["heldout_generator_member"]
        if h not in grid_models:
            continue
        fit_mask, _, _, _ = cc.fold_setup(design, fold, split="dev")
        y_fit = cc.fold_series_target(design, fold)[fit_mask]
        S[h] = sgd_ens(X[fit_mask], y_fit, X)
        print(f"[align2] fit {tag(h)} {time.time()-t_start:.0f}s", flush=True)

    def grid_block(h, m, mask, order_map):
        sel = np.where((member_names == m) & mask)[0]
        order = np.argsort([order_map[t] for t in task_names[sel]])
        return S[h][sel[order]]

    derived_map = {}
    per_file_struct = {}
    for f in sorted(p.name for p in REF.iterdir() if p.is_dir()):
        z = np.load(REF / f / "p0_scores.npz")
        ntr_blk = len(z["train_task"]) // 798
        used = []
        blk_fit = []
        for k in range(ntr_blk):
            blk = z["component_train_char"][k * 798:(k + 1) * 798]
            best = max(((np.corrcoef(blk, grid_block(h, m, is_train, tr_map))[0, 1], h, m)
                        for h in S for m in members), key=lambda x: x[0])
            blk_fit.append((float(best[0]), best[1], best[2]))
            used.append(best[2])
        missing = [m for m in members if m not in used]
        assert len(missing) == 1, (f, missing)
        derived_map[f] = missing[0]
        per_file_struct[f] = {"train_missing": missing[0],
                              "train_block_fit": [[r, tag(h), tag(m)] for r, h, m in blk_fit]}
    assert derived_map == EXPECTED_MAP, "mapping reproduction failed"
    print("[align2] heldout mapping reproduced exactly (11/11)", flush=True)

    # ---------- 2) eval slot→成员识别（用全 11 折 npz 块） ----------
    my_blocks = {}  # (fold_h, member) -> (block, taskpos_sorted) from my npz
    for fold in design["folds"]:
        h = fold["heldout_generator_member"]
        z = np.load(C0 / f"scores_dev_fold{h.replace('--','_')}.npz")
        ev = z["ev_rows"]
        mm, tt = member_names[ev], task_names[ev]
        for m in set(mm.tolist()):
            sel = np.where(mm == m)[0]
            order = np.argsort([dv_map[t] for t in tt[sel]])
            my_blocks[(h, m)] = z["s_char_tfidf"][sel[order]]

    slot_map = {}       # file -> [member per eval slot]
    slot_evidence = {}
    for f in sorted(p.name for p in REF.iterdir() if p.is_dir()):
        z = np.load(REF / f / "p0_scores.npz")
        nb = len(z["y"]) // 171
        ids, evid = [], []
        for k in range(nb):
            blk = z["component_char"][k * 171:(k + 1) * 171]
            cands = sorted(((np.corrcoef(blk, v)[0, 1], h, m) for (h, m), v in my_blocks.items()),
                           key=lambda x: -x[0])[:3]
            ids.append(cands[0][2])
            evid.append([[round(r, 3), tag(h), tag(m)] for r, h, m in cands])
        slot_map[f] = ids
        slot_evidence[f] = evid
        ok = (len(set(ids)) == len(ids))
        print(f"[align2] {tag(f):20s} slots: {[tag(m) for m in ids]} unique={ok}", flush=True)

    # ---------- 3) 共同块对比 ----------
    zl = {f: dict(np.load(REF / f / "p0_scores.npz")) for f in slot_map}
    comp_keys = [("s_semantic", "component_linear"), ("s_char_tfidf", "component_char"),
                 ("s_word_tfidf", "component_word"), ("s_style_meta", "component_style")]
    per_fold = {}
    metric_rows = {}
    for fold in design["folds"]:
        h = fold["heldout_generator_member"]
        f = next(k for k, v in derived_map.items() if v == h)
        slots = slot_map[f]
        z = zl[f]
        # my npz
        zmy = np.load(C0 / f"scores_dev_fold{h.replace('--','_')}.npz")
        ev = zmy["ev_rows"]
        mm, tt = member_names[ev], task_names[ev]
        my_map = {}
        for m in set(mm.tolist()):
            sel = np.where(mm == m)[0]
            order = np.argsort([dv_map[t] for t in tt[sel]])
            my_map[m] = sel[order]
        my_ev_members = set(mm.tolist())
        loc_ev_members = set(slots)
        common = [m for m in slots if m in my_ev_members]  # 以 local 顺序
        assert len(common) >= 5, (h, len(common))  # Q-7B 折交集最小（5 块）
        loc_ok = all(set(z["y"][k * 171:(k + 1) * 171].tolist()) ==
                     ({1} if slots[k] == derived_map[f] else {0}) for k in range(len(slots)))
        assert loc_ok, f"local y pattern mismatch {f}"

        y_blocks, my_blocks_c, ref_blocks_c = [], [], []
        comp_stats = {}
        for m in common:
            k = slots.index(m)
            ref_blk = z["p0"][k * 171:(k + 1) * 171]
            my_blk = zmy["fused"][my_map[m]]
            assert np.array_equal(np.asarray(zmy["y"])[my_map[m]],
                                  z["y"][k * 171:(k + 1) * 171]), (h, m, "y")
            my_blocks_c.append(my_blk)
            ref_blocks_c.append(ref_blk)
            comp_stats[tag(m)] = {"pearson_fused": float(np.corrcoef(my_blk, ref_blk)[0, 1]),
                                  "max_abs_diff_fused": float(np.abs(my_blk - ref_blk).max())}
        my_all = np.concatenate(my_blocks_c)
        ref_all = np.concatenate(ref_blocks_c)
        y_all = np.concatenate([np.asarray(zmy["y"])[my_map[m]] for m in common])
        tp_all = np.concatenate([np.arange(171) for _ in common])  # 任务序已排序
        # 全部共同行的融合统计
        d = my_all - ref_all
        m_row, r_row = cc.auroc(y_all, my_all), cc.auroc(y_all, ref_all)
        m_tm, r_tm = cc.task_macro_auroc(y_all, my_all, tp_all), cc.task_macro_auroc(y_all, ref_all, tp_all)
        per_fold[h] = {
            "local_file": f, "common_members": [tag(m) for m in common],
            "n_common_rows": int(len(my_all)),
            "fused": {"max_abs_diff": float(np.abs(d).max()),
                      "mean_abs_diff": float(np.abs(d).mean()),
                      "pearson": float(np.corrcoef(my_all, ref_all)[0, 1]),
                      "per_member": comp_stats},
            "metrics": {"row_auroc_mine": m_row, "row_auroc_ref": r_row, "delta_row": m_row - r_row,
                        "task_macro_mine": m_tm, "task_macro_ref": r_tm, "delta_tm": m_tm - r_tm},
        }
        metric_rows[h] = (y_all, tp_all)
        print(f"[align2] {tag(h):26s} common={len(my_all)} max|d|={np.abs(d).max():.3f} "
              f"r={np.corrcoef(my_all, ref_all)[0,1]:+.3f} row {m_row:.4f}/{r_row:.4f} "
              f"tm {m_tm:.4f}/{r_tm:.4f}", flush=True)

    mean_row_mine = float(np.mean([v["metrics"]["row_auroc_mine"] for v in per_fold.values()]))
    mean_row_ref = float(np.mean([v["metrics"]["row_auroc_ref"] for v in per_fold.values()]))
    mean_tm_mine = float(np.mean([v["metrics"]["task_macro_mine"] for v in per_fold.values()]))
    mean_tm_ref = float(np.mean([v["metrics"]["task_macro_ref"] for v in per_fold.values()]))
    all_max = max(v["fused"]["max_abs_diff"] for v in per_fold.values())
    min_pear = min(v["fused"]["pearson"] for v in per_fold.values())
    max_drow = max(abs(v["metrics"]["delta_row"]) for v in per_fold.values())
    max_dtm = max(abs(v["metrics"]["delta_tm"]) for v in per_fold.values())
    verdict = ("aligned_strict" if all_max <= 1e-3 and max_drow <= 1e-3 and max_dtm <= 1e-3
               else "aligned_at_metric_level" if min_pear >= 0.999 and max_drow <= 1e-3 and max_dtm <= 1e-3
               else "not_aligned")
    cl_only = {h: v for h, v in per_fold.items() if "CodeLlama" in h}
    structure = {
        "schema": "code_conditioned_c0_alignment_v2_structure",
        "finding": "local public_series_member_ho_v2 uses a different same-series partition "
                   "than the server C0; dir-name->heldout is permuted for non-CL members",
        "local_partition": LOCAL_PARTITION,
        "server_partition": {
            "P1": LOCAL_PARTITION["P1"],
            "P2": ["Qwen--Qwen2.5-Coder-1.5B-Instruct", "Qwen--Qwen2.5-Coder-14B-Instruct",
                   "Qwen--Qwen2.5-Coder-32B-Instruct", "Qwen--Qwen2.5-Coder-7B-Instruct"],
            "P3": ["deepseek-ai--deepseek-coder-1.3b-instruct",
                   "deepseek-ai--deepseek-coder-33b-instruct",
                   "deepseek-ai--deepseek-coder-6.7b-instruct"],
        },
        "dir_to_heldout_reproduced": derived_map,
        "slot_members": {tag(f): [tag(m) for m in slot_map[f]] for f in slot_map},
        "slot_evidence_top3": {tag(f): slot_evidence[f] for f in slot_evidence},
    }
    (OUT / "structure_findings.json").write_text(json.dumps(structure, ensure_ascii=False, indent=1), encoding="utf-8")

    align = {
        "schema": "code_conditioned_c0_alignment_v2",
        "reference": "code_conditioned_p0_reference_2026-10-09 (local v2; matched by true heldout)",
        "server": "code_conditioned_c0_strong_p0_2026-10-09",
        "method": "compare on common member blocks (8 blocks x 171 rows = 1368 rows/fold); "
                  "CL folds are fully comparable (same design); non-CL folds differ in design",
        "per_fold": per_fold,
        "aggregate": {"mean_row_auroc_mine": mean_row_mine, "mean_row_auroc_ref": mean_row_ref,
                      "mean_task_macro_mine": mean_tm_mine, "mean_task_macro_ref": mean_tm_ref,
                      "max_abs_diff_fused": all_max, "min_pearson_fused": min_pear,
                      "max_abs_delta_row": max_drow, "max_abs_delta_tm": max_dtm,
                      "cl_folds_only_row": {
                          "mine": float(np.mean([v["metrics"]["row_auroc_mine"] for v in cl_only.values()])),
                          "ref": float(np.mean([v["metrics"]["row_auroc_ref"] for v in cl_only.values()]))}},
        "verdict": verdict,
        "verdict_note": "per pre-registration; even CL folds exceed 1e-3 per-row tolerance -> "
                        "spec/implementation gap beyond floating error; non-CL folds additionally "
                        "have design-level mismatch (different eval sets).",
    }
    (OUT / "p0_alignment_v2.json").write_text(json.dumps(align, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[align2] VERDICT {verdict}; mean row {mean_row_mine:.4f} vs ref {mean_row_ref:.4f} "
          f"(CL-only {align['aggregate']['cl_folds_only_row']})", flush=True)

    with (OUT / "p0_alignment_v2_per_fold.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["heldout", "local_file", "n_common", "max_abs_diff_fused", "pearson_fused",
                    "row_mine", "row_ref", "delta_row", "tm_mine", "tm_ref", "delta_tm"])
        for h, v in per_fold.items():
            w.writerow([h, v["local_file"], v["n_common_rows"], f"{v['fused']['max_abs_diff']:.6g}",
                        f"{v['fused']['pearson']:.6f}", f"{v['metrics']['row_auroc_mine']:.6f}",
                        f"{v['metrics']['row_auroc_ref']:.6f}", f"{v['metrics']['delta_row']:.6f}",
                        f"{v['metrics']['task_macro_mine']:.6f}", f"{v['metrics']['task_macro_ref']:.6f}",
                        f"{v['metrics']['delta_tm']:.6f}"])

    # ---------- 4) C1–C3 在共同行上对 本机P0 重算 ----------
    def build_ref_and_mine_blocks(h):
        f = derived_map_inv[h]
        slots = slot_map[f]
        z = zl[f]
        zmy = np.load(C0 / f"scores_dev_fold{h.replace('--','_')}.npz")
        ev = zmy["ev_rows"]
        mm, tt = member_names[ev], task_names[ev]
        my_map = {}
        for m in set(mm.tolist()):
            sel = np.where(mm == m)[0]
            order = np.argsort([dv_map[t] for t in tt[sel]])
            my_map[m] = sel[order]
        common = [m for m in slots if m in set(mm.tolist())]
        ref_all = np.concatenate([z["p0"][slots.index(m) * 171:(slots.index(m) + 1) * 171] for m in common])
        ref_idx_rows = np.concatenate([my_map[m] for m in common])  # 我 npz 行号（local 顺序块）
        return f, ref_all, ref_idx_rows, common, zmy

    derived_map_inv = {v: k for k, v in derived_map.items()}

    def order_rows(fold_h, vec_by_row, ref_idx_rows):
        """把候选/我的全行分数组按 (local 块顺序) 抽出并与 ref 对齐"""
        return np.asarray(vec_by_row)[ref_idx_rows]

    candidates = {}
    for key, dirp, keys in [("c1", C1, ["code_only_mlp", "full_mlp", "ht_hy_mlp",
                                        "code_only_linear", "full_linear", "psi_only_mlp"]),
                            ("c2", C2, ["s_c2"]),
                            ("c3", C3, ["s_orig"])]:
        for ck in keys:
            cand = {}
            mine = {}
            labels, tposs = {}, {}
            for fold in design["folds"]:
                h = fold["heldout_generator_member"]
                f, ref_all, ref_idx_rows, common, zmy = build_ref_and_mine_blocks(h)
                cand[h] = ref_all
                z = np.load(dirp / f"scores_fold{h.replace('--','_')}.npz")
                assert np.array_equal(z["ev_rows"], zmy["ev_rows"]), (h, ck)
                mine[h] = np.asarray(z[ck])[ref_idx_rows]
                labels[h] = np.asarray(zmy["y"])[ref_idx_rows]
                tposs[h] = np.concatenate([np.arange(171) for _ in common])
                # bootstrap 需要 taskpos 有序：按 taskpos 稳定排序（行内顺序不影响指标）
                order = np.argsort(tposs[h], kind="mergesort")
                mine[h], cand[h] = mine[h][order], cand[h][order]
                labels[h], tposs[h] = labels[h][order], tposs[h][order]
            dn = cc.delta_bootstrap(mine, cand, labels, tposs)
            my_c0_common = {}
            for fold in design["folds"]:
                h = fold["heldout_generator_member"]
                f, ref_all, ref_idx_rows, common, zmy = build_ref_and_mine_blocks(h)
                v = np.asarray(zmy["fused"])[ref_idx_rows]
                order = np.argsort(np.concatenate([np.arange(171) for _ in common]),
                                   kind="mergesort")
                my_c0_common[h] = v[order]
            dc0 = cc.delta_bootstrap(mine, my_c0_common, labels, tposs)
            candidates[f"{key}.{ck}"] = {
                "vs_local_p0_common_rows": {
                    "delta_row_mean": float(np.mean(dn["row_level"])),
                    "delta_row_ci95": cc.ci(dn["row_level"]),
                    "delta_tm_mean": float(np.mean(dn["task_macro"])),
                    "delta_tm_ci95": cc.ci(dn["task_macro"]),
                },
                "vs_server_c0_common_rows": {
                    "delta_row_mean": float(np.mean(dc0["row_level"])),
                    "delta_tm_mean": float(np.mean(dc0["task_macro"])),
                }}
            print(f"[align2] {key}.{ck:16s} vs localP0(common): dRow {np.mean(dn['row_level']):+.4f} "
                  f"dTM {np.mean(dn['task_macro']):+.4f} | vs C0: dTM {np.mean(dc0['task_macro']):+.4f}",
                  flush=True)

    (OUT / "unified_p0_deltas_v2.json").write_text(json.dumps({
        "schema": "code_conditioned_unified_p0_deltas_v2",
        "note": "candidates vs local v2 P0 on common blocks (8 blocks/fold, heldout-matched); "
                "paired task-cluster bootstrap (seed 20261009, B=500)",
        "c0_vs_local_p0": {h: v["metrics"] for h, v in per_fold.items()},
        "candidates": candidates,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[align2] done in {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    main()
