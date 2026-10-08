"""R1 交换接口修正（指导 §4.1）：真正 child/base 交换只应改 D 不改 S。

错误（旧 core_r1_edges.py / core_r1_update.py）：SD 读出的正负差用 [S;D] 与 −[S;D]（S 也被反号）
——标为 sign-construction diagnostic。正确接口：pos=[S;D] vs neg=[S;-D]（B 只有 D 变号）。

本轮（train/dev；模式分开；7 边）：
  swap_direction：正确接口方向判别（逐边逐 mode 判对率 + task-cluster CI）
  old_sd_diagnostic：旧接口重算（对照）
  side：单侧 child vs base；p0：surface 差方向（partial）
  edge_heldout：LOO 正确接口（描述性）
  版本/类型表：引 core_lineage_audit lineage_adjudication.json（level + base relation 证据）

输出：artifacts/r1_swap_validated_2026-10-08/
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

OUT = ROOT / "d-det/artifacts/r1_swap_validated_2026-10-08"
LINEAGE = ROOT / "d-det/artifacts/core_lineage_audit_2026-10-08/lineage_adjudication.json"
SEED = 20261008
LOG: list[str] = []

EDGES = [
    ("E1_Athene70B_Llama3", "Nexusflow--Athene-70B", "meta-llama--Meta-Llama-3-70B-Instruct"),
    ("E2_AtheneV2Agent_Qwen72B", "Nexusflow--Athene-V2-Agent", "Qwen--Qwen2.5-72B-Instruct"),
    ("E3_AtheneV2Chat_Qwen72B", "Nexusflow--Athene-V2-Chat", "Qwen--Qwen2.5-72B-Instruct"),
    ("E4_SkyT1Flash_Qwen32B", "NovaSky-AI--Sky-T1-32B-Flash", "Qwen--Qwen2.5-32B-Instruct"),
    ("E5_SkyT1Flash_SkyT1Preview", "NovaSky-AI--Sky-T1-32B-Flash", "NovaSky-AI--Sky-T1-32B-Preview"),
    ("E6_SkyT1Preview_Qwen32B", "NovaSky-AI--Sky-T1-32B-Preview", "Qwen--Qwen2.5-32B-Instruct"),
    ("E7_QwQ32B_Qwen32B", "Qwen--QwQ-32B-Preview", "Qwen--Qwen2.5-32B-Instruct"),
]


def log(m):
    print(m, flush=True)
    LOG.append(m)


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "logs").mkdir(exist_ok=True)
    from sklearn.linear_model import LogisticRegression
    A = ev.load_assets()
    splits = {t: A["splits"][A["TASK_I"][t] * 2] for t in A["tasks"]}
    tr_tasks = [t for t in A["tasks"] if splits[t] == "train"]
    dv_tasks = [t for t in A["tasks"] if splits[t] == "dev"]
    H = A["emb_base"]

    def rows2(model, tasks, mode):
        mi = A["MODEL_I"][model]
        ti = np.array([A["TASK_I"][t] for t in tasks])
        return mi * A["nT"] * 2 + ti * 2 + mode

    def sd_block(e, tasks, mode):
        hc = H[rows2(e[1], tasks, mode)]; hb = H[rows2(e[2], tasks, mode)]
        S = (hc + hb) / 2.0; D2 = hc - hb  # 注意：D=child-base；交换只反此处
        return S, D2

    # fit：正确交换接口 [S;D] pos vs [S;-D] neg（7 边两组 mode 同进）
    X_pos, X_neg = [], []
    X_old_pos, X_old_neg = [], []
    for e in EDGES:
        for mode in (0, 1):
            S, D2 = sd_block(e, tr_tasks, mode)
            X_pos.append(np.hstack([S, D2]))
            X_neg.append(np.hstack([S, -D2]))          # 正确：只反 D
            X_old_pos.append(np.hstack([S, D2]))
            X_old_neg.append(np.hstack([-S, -D2]))      # 旧：S 也被反（错误接口）
    Xp = np.concatenate(X_pos, 0); Xn = np.concatenate(X_neg, 0)
    Xop = np.concatenate(X_old_pos, 0); Xon = np.concatenate(X_old_neg, 0)
    y = np.concatenate([np.ones(len(Xp)), np.zeros(len(Xn))])
    clf_swap = LogisticRegression(max_iter=2000, C=1.0).fit(np.concatenate([Xp, Xn], 0), y)
    clf_old = LogisticRegression(max_iter=2000, C=1.0).fit(np.concatenate([Xop, Xon], 0), y)
    # side
    Xc = np.concatenate([H[rows2(e[1], tr_tasks, m)] for e in EDGES for m in (0, 1)], 0)
    Xb = np.concatenate([H[rows2(e[2], tr_tasks, m)] for e in EDGES for m in (0, 1)], 0)
    clf_side = LogisticRegression(max_iter=2000, C=1.0).fit(
        np.concatenate([Xc, Xb], 0), np.concatenate([np.ones(len(Xc)), np.zeros(len(Xb))]))
    # p0（surface 差）
    def p0d(e, tasks, mode):
        rc = rows2(e[1], tasks, mode); rb = rows2(e[2], tasks, mode)
        f = lambda r: np.hstack([A["style"][r], A["meta"][r], A["sizelen"][r]])
        return f(rc) - f(rb)
    Xp0 = np.concatenate([p0d(e, tr_tasks, m) for e in EDGES for m in (0, 1)], 0)
    clf_p0 = LogisticRegression(max_iter=2000, C=1.0).fit(np.concatenate([Xp0, -Xp0], 0), y)
    log("[r1s] readouts fitted (swap / old-diagnostic / side / p0)")

    def edge_eval(e, clf_s, clf_o):
        out = {}
        for mode, mt in ((0, "complete"), (1, "instruct")):
            S, D2 = sd_block(e, dv_tasks, mode)
            Xs = np.hstack([S, D2]); Xn_ = np.hstack([S, -D2])
            # 判对率：真方向 [S;D] 分数 > 交换 [S;-D]
            score_true = clf_s.decision_function(Xs)
            score_swap = clf_s.decision_function(Xn_)
            acc = float(np.mean(score_true > score_swap))
            # 旧接口诊断（同构造逻辑）
            Xo1 = np.hstack([S, D2]); Xo2 = np.hstack([-S, -D2])
            so1 = clf_o.decision_function(Xo1); so2 = clf_o.decision_function(Xo2)
            old = float(np.mean(so1 > so2))
            side = float(np.mean(clf_side.decision_function(H[rows2(e[1], dv_tasks, mode)])
                                 > clf_side.decision_function(H[rows2(e[2], dv_tasks, mode)])))
            p0 = float(np.mean(clf_p0.decision_function(p0d(e, dv_tasks, mode)) > 0))
            # task-cluster CI（swap）
            rng = np.random.default_rng(SEED)
            n = len(dv_tasks)
            d = (score_true > score_swap).astype(float)
            vals = [float(np.mean(d[rng.choice(n, size=n, replace=True)])) for _ in range(300)]
            out[mt] = {"swap_direction": acc, "old_sd_diagnostic": old, "side": side, "p0": p0,
                       "swap_ci95": [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]}
        return out

    # edge-heldout（LOO swap 判别器）
    edge_ho = {}
    for ehold in EDGES:
        Xp2_list, Xn2_list = [], []
        for e in EDGES:
            if e[0] == ehold[0]:
                continue
            for mode in (0, 1):
                S, D2 = sd_block(e, tr_tasks, mode)
                Xp2_list.append(np.hstack([S, D2])); Xn2_list.append(np.hstack([S, -D2]))
        Xp2 = np.concatenate(Xp2_list, 0); Xn2 = np.concatenate(Xn2_list, 0)
        y2 = np.concatenate([np.ones(len(Xp2)), np.zeros(len(Xn2))])
        c2 = LogisticRegression(max_iter=2000, C=1.0).fit(np.concatenate([Xp2, Xn2], 0), y2)
        r = edge_eval(ehold, c2, clf_old)
        edge_ho[ehold[0]] = {"instruct_swap": r["instruct"]["swap_direction"]}
    eh_mean = float(np.mean([v["instruct_swap"] for v in edge_ho.values()]))

    # 版本/类型表
    lineage = json.loads(LINEAGE.read_text(encoding="utf-8"))
    ev_table = {}
    for e in EDGES:
        child = lineage["levels"].get(e[1], {})
        base = lineage["levels"].get(e[2], {})
        ev_table[e[0]] = {
            "child": e[1], "base": e[2],
            "child_level": child.get("level", "unknown"),
            "documented_base_relation": child.get("field_evidence", {}).get("documented_base_relation", {}),
            "relation_type_note": ("same-repo iteration (Sky-T1 Preview -> Flash)" if e[0].startswith("E5")
                                   else "post-training or further training on documented base; specifics unverified"),
            "version_evidence": "partial (HF cardData.base_model; training steps/data unverified)",
        }

    per_edge = {}
    for e in EDGES:
        per_edge[e[0]] = edge_eval(e, clf_swap, clf_old)
        i = per_edge[e[0]]["instruct"]
        log(f"[r1s] {e[0]}: swap={i['swap_direction']:.3f} (old={i['old_sd_diagnostic']:.3f}) "
            f"side={i['side']:.3f} p0={i['p0']:.3f}")
    results = {"schema": "r1_swap_validated_v1", "generated_utc": datetime.now(timezone.utc).isoformat(),
               "report_first_line": "更正性重跑 — [S;D]↔[S;-D] 交换修正",
               "protocol": "pos=[S;D] vs neg=[S;-D]（只反 D）; old 接口以 sign-construction diagnostic 并列",
               "per_edge": per_edge, "edge_heldout_swap": edge_ho, "edge_heldout_mean": eh_mean,
               "evidence_table": ev_table,
               "note": ("旧 [S;D]/-[S;D] 的 ~1.0 SD 数字降级为 sign-construction diagnostic；"
                        "所有边仍 partial；不得写 verified post-training effect。")}
    (OUT / "metrics_r1_swap.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    L = ["# R1 交换接口修正（corrective rerun；不覆盖旧结果）", "",
         "正确接口：pos=[S;D] vs neg=[S;−D]（交换只反 D）。旧 [S;D] vs −[S;D] 列为 sign-construction diagnostic。", "",
         "| edge | swap_direction | old_diagnostic | side | p0 |", "|---|---|---|---|---|"]
    for k, v in per_edge.items():
        i = v["instruct"]
        L.append(f"| {k} | {i['swap_direction']:.3f} | {i['old_sd_diagnostic']:.3f} | {i['side']:.3f} | {i['p0']:.3f} |")
    L += ["", f"- edge-heldout（LOO，instruct）均值: {eh_mean:.3f}",
          "", "## 版本/类型表（partial）"]
    for k, v in ev_table.items():
        L.append(f"- {k}: level={v['child_level']}; {v['relation_type_note']}")
    (OUT / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (OUT / "logs" / "r1s_run.log").write_text("\n".join(LOG) + "\n", encoding="utf-8")
    log(f"[r1s] done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
