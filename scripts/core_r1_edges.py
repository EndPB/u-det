"""R1：六条 lineage 候选的严格 train/dev 评估（§4）。

edges（冻结自 E0 lineage_adjudication，不按结果删选）：
 E1 Athene-70B <- Meta-Llama-3-70B-Instruct
 E2 Athene-V2-Agent <- Qwen2.5-72B-Instruct
 E3 Athene-V2-Chat  <- Qwen2.5-72B-Instruct
 E4 Sky-T1-32B-Flash <- Qwen2.5-32B-Instruct
 E5 Sky-T1-32B-Flash <- Sky-T1-32B-Preview
 E6 Sky-T1-32B-Preview <- Qwen2.5-32B-Instruct
 E7 QwQ-32B-Preview <- Qwen2.5-32B-Instruct

任务：a) pair 方向判别（child vs base；全局方向 + 逐 edge 判对率）；b) edge 识别（7-way，同 task 的 (child,base) 对属于哪条边）。
视图：single_child / single_base / Delta / [S;Delta] / P0(style+meta+size)。
切分：task-heldout（fit=798 train → eval=dev 171）；CI=task-cluster（edge 数不足,不做 model-cluster）。
输出：artifacts/r1_partial_lineage_train_dev_2026-10-08/
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

OUT = ROOT / "d-det/artifacts/r1_partial_lineage_train_dev_2026-10-08"
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
    A = ev.load_assets()
    # 任务切分
    split_task = {}
    for t in A["tasks"]:
        split_task[t] = A["splits"][A["TASK_I"][t] * 2]
    tr_tasks = [t for t in A["tasks"] if split_task[t] == "train"]
    dv_tasks = [t for t in A["tasks"] if split_task[t] == "dev"]
    log(f"[r1] models ok; tasks train={len(tr_tasks)} dev={len(dv_tasks)}; edges={len(EDGES)}")

    for e in EDGES:
        for m in e[1:]:
            assert m in A["MODEL_I"], m
    H = A["emb_base"]

    def rows2(model, tasks, mode):
        """mode: 0=complete,1=instruct; 返回行索引。"""
        mi = A["MODEL_I"][model]
        ti = np.array([A["TASK_I"][t] for t in tasks])
        return mi * A["nT"] * 2 + ti * 2 + mode

    def feats_for(edge, tasks, mode):
        _, ch, ba = edge
        hc = H[rows2(ch, tasks, mode)]
        hb = H[rows2(ba, tasks, mode)]
        S = (hc + hb) / 2.0
        D = hc - hb
        return hc, hb, S, D

    # ---------- 训练（fit 域,双 mode 都进 train）----------
    fit_rows = []
    for e in EDGES:
        for mode in (0, 1):
            hc = H[rows2(e[1], tr_tasks, mode)]
            hb = H[rows2(e[2], tr_tasks, mode)]
            fit_rows.append((e[0], mode, hc, hb))
    # 全局方向判别（symmetrized on Delta）：D=child-base 为正
    D_all = np.concatenate([hc - hb for (_, _, hc, hb) in fit_rows], 0)
    from sklearn.linear_model import LogisticRegression
    X = np.concatenate([D_all, -D_all], 0)
    y = np.concatenate([np.ones(len(D_all)), np.zeros(len(D_all))])
    clf_d = LogisticRegression(max_iter=300, C=1.0).fit(X, y)
    # 单侧（样本级）：child vs base
    Xc = np.concatenate([hc for (_, _, hc, hb) in fit_rows], 0)
    Xb = np.concatenate([hb for (_, _, hc, hb) in fit_rows], 0)
    clf_side = LogisticRegression(max_iter=300, C=1.0).fit(np.concatenate([Xc, Xb], 0),
                                                           np.concatenate([np.ones(len(Xc)), np.zeros(len(Xb))]))
    # [S;D] 联合
    Xs = np.concatenate([np.hstack([(hc + hb) / 2.0, hc - hb]) for (_, _, hc, hb) in fit_rows], 0)
    clf_sd = LogisticRegression(max_iter=300, C=1.0).fit(np.concatenate([Xs, -Xs], 0), y)
    # P0（style+meta+size 拼接差）
    def p0_pair(edge, tasks, mode):
        _, ch, ba = edge
        r_c = rows2(ch, tasks, mode); r_b = rows2(ba, tasks, mode)
        f = lambda r: np.hstack([A["style"][r], A["meta"][r], A["sizelen"][r]])
        return f(r_c) - f(r_b)
    P0D_all = np.concatenate([p0_pair(e, tr_tasks, m) for e in EDGES for m in (0, 1)], 0)
    clf_p0 = LogisticRegression(max_iter=300, C=1.0).fit(np.concatenate([P0D_all, -P0D_all], 0), y)
    log("[r1] direction readouts fitted (delta/side/SD/P0)")

    # ---------- 评测（dev 域,逐 edge × mode）----------
    metrics = {"schema": "r1_edge_metrics_v1",
               "generated_utc": datetime.now(timezone.utc).isoformat(),
               "label": "documented post-training candidate relation (observational; partial)",
               "edges": {}, "fit": {"n_tasks": len(tr_tasks), "n_edges": len(EDGES)}}
    preds = {}
    for e in EDGES:
        eid = e[0]
        per = {}
        for mode, mt in ((0, "complete"), (1, "instruct")):
            hc = H[rows2(e[1], dv_tasks, mode)]
            hb = H[rows2(e[2], dv_tasks, mode)]
            D = hc - hb
            Sd = (hc + hb) / 2.0
            p0 = p0_pair(e, dv_tasks, mode)
            sc_d = clf_d.decision_function(D)
            sc_side = clf_side.decision_function(hc)
            sc_side_b = clf_side.decision_function(hb)
            sc_sd = clf_sd.decision_function(np.hstack([Sd, D]))
            sc_p0 = clf_p0.decision_function(p0)
            # 判对率（child 应得高分）：直接算 child>base 率
            def acc(sc_c, sc_b):
                d = sc_c - sc_b
                return float(np.mean(d > 0) + 0.5 * np.mean(d == 0))
            res = {
                "delta": float(np.mean(sc_d > 0) + 0.5 * np.mean(sc_d == 0)),
                "side_child": acc(sc_side, sc_side_b),
                "SD": float(np.mean(sc_sd > 0) + 0.5 * np.mean(sc_sd == 0)),
                "p0": float(np.mean(sc_p0 > 0) + 0.5 * np.mean(sc_p0 == 0)),
            }
            # task 聚类 bootstrap（简单：对 task 重采样）
            n = len(dv_tasks)
            rng = np.random.default_rng(SEED)
            boots = {"delta": [], "side_child": [], "SD": [], "p0": []}
            pos_map = {"delta": sc_d > 0, "side_child": sc_side > sc_side_b, "SD": sc_sd > 0, "p0": sc_p0 > 0}
            for k in range(300):
                idx = rng.choice(n, size=n, replace=True)
                for key, arr in pos_map.items():
                    boots[key].append(float(np.mean(arr[idx])))
            ci = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in boots.items()}
            per[mt] = {"acc": res, "ci95": ci, "n_tasks": n}
            preds[f"{eid}::{mt}::delta"] = sc_d
            preds[f"{eid}::{mt}::side_c"] = sc_side
            preds[f"{eid}::{mt}::side_b"] = sc_side_b
            preds[f"{eid}::{mt}::SD"] = sc_sd
            preds[f"{eid}::{mt}::p0"] = sc_p0
        metrics["edges"][eid] = {"child": e[1], "base": e[2], "by_mode": per,
                                 "evidence": "partial (HF cardData.base_model; version 级证据缺)"}
        log(f"[r1] {eid}: delta={per['instruct']['acc']['delta']:.3f} "
            f"side={per['instruct']['acc']['side_child']:.3f} SD={per['instruct']['acc']['SD']:.3f} "
            f"p0={per['instruct']['acc']['p0']:.3f}")

    # ---------- edge 识别（7-way）----------
    log("[r1] edge identification (7-way)")
    import torch
    import torch.nn as nn
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def edge_features(e, tasks, mode):
        hc = H[rows2(e[1], tasks, mode)]; hb = H[rows2(e[2], tasks, mode)]
        S = (hc + hb) / 2.0; D = hc - hb
        return np.hstack([S, D])
    Xtr = np.concatenate([edge_features(e, tr_tasks, m) for e in EDGES for m in (0, 1)], 0)
    ytr = np.concatenate([np.full(len(tr_tasks), i) for i in range(len(EDGES)) for m in (0, 1)])
    Xdv = np.concatenate([edge_features(e, dv_tasks, m) for e in EDGES for m in (0, 1)], 0)
    ydv = np.concatenate([np.full(len(dv_tasks), i) for i in range(len(EDGES)) for m in (0, 1)])
    net = nn.Linear(Xtr.shape[1], len(EDGES)).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    Xt = torch.tensor(Xtr, dtype=torch.float32, device=device)
    yt = torch.tensor(ytr, dtype=torch.long, device=device)
    rng = np.random.default_rng(SEED)
    for ep in range(10):
        perm = rng.permutation(len(Xt))
        tot = 0.0
        for i in range(0, len(perm), 4096):
            sel = torch.tensor(perm[i:i + 4096], device=device)
            loss = nn.functional.cross_entropy(net(Xt[sel]), yt[sel])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.detach())
    with torch.inference_mode():
        logits = net(torch.tensor(Xdv, dtype=torch.float32, device=device)).cpu().numpy()
    lab = logits.argmax(1)
    acc7 = float(np.mean(lab == ydv))
    f1s = []
    for c in range(len(EDGES)):
        tp = np.sum((lab == c) & (ydv == c)); fp = np.sum((lab == c) & (ydv != c)); fn = np.sum((lab != c) & (ydv == c))
        pr = tp / max(1, tp + fp); rc = tp / max(1, tp + fn)
        f1s.append(2 * pr * rc / max(1e-9, pr + rc))
    per_edge_acc = {EDGES[c][0]: float(np.mean(lab[ydv == c] == c)) for c in range(len(EDGES))}
    metrics["edge_identification"] = {"acc_7way": acc7, "macro_f1": float(np.mean(f1s)),
                                      "chance": 1.0 / len(EDGES), "per_edge_acc": per_edge_acc,
                                      "note": "dev 任务 heldout；fit=train 任务"}
    log(f"[r1] 7-way acc={acc7:.3f} macroF1={np.mean(f1s):.3f} (chance {1/len(EDGES):.3f})")

    np.savez_compressed(OUT / "local" / "r1_preds.npz", **preds)
    (OUT / "metrics_r1.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=1), encoding="utf-8")

    # 报告
    L = ["# R1：lineage 候选 train/dev 评估（partial）", "",
         "> 所有边 `partial`（HF cardData.base_model 文档；version 级证据缺）；observational，非随机化；",
         "> task-heldout（fit=798 train → eval=171 dev）；CI=task-cluster (300, seed 20261008)。", "",
         "| edge | child | base | Δ判对 | single | [S;Δ] | P0 |", "|---|---|---|---|---|---|---|"]
    for e in EDGES:
        eid = e[0]; p = metrics["edges"][eid]["by_mode"]["instruct"]["acc"]
        L.append(f"| {eid} | {e[1]} | {e[2]} | {p['delta']:.3f} | {p['side_child']:.3f} | {p['SD']:.3f} | {p['p0']:.3f} |")
    L.append("")
    L.append(f"## edge 识别（7-way）")
    L.append(f"- dev acc={acc7:.3f}（chance {1/len(EDGES):.3f}），macro-F1={np.mean(f1s):.3f}")
    L.append(f"- per-edge acc: " + ", ".join(f"{k}={v:.2f}" for k, v in per_edge_acc.items()))
    L.append("")
    L.append("## 合法表述（§4.3）")
    L.append("- 全部 edge 为 partial → 只能写“模型卡记录的 base relation 候选”；不得进入论文标题/摘要的 post-training effect。")
    L.append("- 八条硬条件中 documented relation ✓；same task/protocol ✓；independent provenance ✓；version 独立证据缺 → partial。")
    (OUT / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (OUT / "logs" / "r1_run.log").write_text("\n".join(LOG) + "\n", encoding="utf-8")
    log(f"[r1] done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
