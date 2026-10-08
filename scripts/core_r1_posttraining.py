"""R1 后训练比较（7ee36fb 修复裁定 §3）。

逐边（7 边）× 模式（complete/instruct 分开）读出：
  delta_only（A/Delta 线性方向）、raw 联合 swap（[S;D] vs [S;-D] LR）、
  小 MLP swap（256,64；sym 训练 10ep，3 seeds）、单侧来源组合（side clf）、完整 P0（surface 差）。
逐边 task-cluster paired CI（300）；不做 max(p,1-p)、不依 dev 翻转方向。
数学说明：线性 swap 分数差 f([S,D])−f([S,−D])=2·w_D·D（S 项与截距消去）——不构成 S-A 非线性交互证据。
endpoint/component 隔离支持表：对每条边，移除其共享端点 checkpoint 的所有 incident 边后的可评情况
（不足记 insufficient；E2/E3 共享 parent、E4/E5/E6 共享 checkpoint——留一条边非 checkpoint-heldout）。

输出：artifacts/r1_posttraining_comparison_2026-10-08/
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

OUT = ROOT / "d-det/artifacts/r1_posttraining_comparison_2026-10-08"
SEED = 20261008
SEEDS = (0, 1, 2)
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
    import torch
    import torch.nn as nn
    from sklearn.linear_model import LogisticRegression
    A = ev.load_assets()
    splits = {t: A["splits"][A["TASK_I"][t] * 2] for t in A["tasks"]}
    tr_tasks = [t for t in A["tasks"] if splits[t] == "train"]
    dv_tasks = [t for t in A["tasks"] if splits[t] == "dev"]
    H = A["emb_base"]

    def rows2(m, tasks, mode):
        mi = A["MODEL_I"][m]
        ti = np.array([A["TASK_I"][t] for t in tasks])
        return mi * A["nT"] * 2 + ti * 2 + mode

    def blocks(e, tasks, mode):
        rc = rows2(e[1], tasks, mode); rb = rows2(e[2], tasks, mode)
        S = (H[rc] + H[rb]) / 2.0; D = H[rc] - H[rb]
        f = lambda r: np.hstack([A["style"][r], A["meta"][r], A["sizelen"][r]])
        return S, D, f(rc) - f(rb), H[rc], H[rb]

    # ---------- fit 读出（全部 7 边 × 两 mode 同时）----------
    fit_rows = [blocks(e, tr_tasks, m) for e in EDGES for m in (0, 1)]
    S_all = np.concatenate([r[0] for r in fit_rows], 0)
    D_all = np.concatenate([r[1] for r in fit_rows], 0)
    P0_all = np.concatenate([r[2] for r in fit_rows], 0)
    Hc_all = np.concatenate([r[3] for r in fit_rows], 0)
    Hb_all = np.concatenate([r[4] for r in fit_rows], 0)
    y_sym = np.concatenate([np.ones(len(D_all)), np.zeros(len(D_all))])
    # delta_only：LR on [D,−D]
    clf_d = LogisticRegression(max_iter=1000, C=1.0).fit(np.concatenate([D_all, -D_all], 0), y_sym)
    # raw 联合 swap：[S,D] vs [S,−D]
    Xp = np.hstack([S_all, D_all]); Xn = np.hstack([S_all, -D_all])
    clf_sw = LogisticRegression(max_iter=1000, C=1.0).fit(np.concatenate([Xp, Xn], 0), y_sym)
    # side
    clf_side = LogisticRegression(max_iter=1000, C=1.0).fit(
        np.concatenate([Hc_all, Hb_all], 0), np.concatenate([np.ones(len(Hc_all)), np.zeros(len(Hb_all))]))
    # P0
    clf_p0 = LogisticRegression(max_iter=1000, C=1.0).fit(np.concatenate([P0_all, -P0_all], 0), y_sym)
    # 小 MLP swap
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def train_mlp(seed):
        torch.manual_seed(seed)
        net = nn.Sequential(nn.Linear(1536, 256), nn.ReLU(), nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 1)).to(device)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        X = torch.tensor(np.concatenate([Xp, Xn], 0).astype(np.float32), device=device)
        yt = torch.tensor(y_sym.astype(np.float32), device=device)
        rng = np.random.default_rng(seed)
        for ep in range(10):
            perm = rng.permutation(len(X))
            for i in range(0, len(perm), 4096):
                sel = torch.tensor(perm[i:i + 4096], device=device)
                loss = nn.functional.binary_cross_entropy_with_logits(net(X[sel]).squeeze(-1), yt[sel])
                opt.zero_grad(); loss.backward(); opt.step()
        def sc(V):
            with torch.inference_mode():
                return net(torch.tensor(np.asarray(V, dtype=np.float32), device=device)).squeeze(-1).cpu().numpy()
        return sc
    mlp_scorers = [train_mlp(s) for s in SEEDS]
    log("[r1p] readouts fitted")

    # ---------- 逐边评测 ----------
    per_edge = {}
    for e in EDGES:
        entry = {}
        for mode, mt in ((0, "complete"), (1, "instruct")):
            S_, D_, P0_, Hc_, Hb_ = blocks(e, dv_tasks, mode)
            d_del = clf_d.decision_function(D_) > 0
            d_sw = clf_sw.decision_function(np.hstack([S_, D_])) > clf_sw.decision_function(np.hstack([S_, -D_]))
            X1 = np.hstack([S_, D_]); X2 = np.hstack([S_, -D_])
            mlp_ok_arr = np.mean([(sc(X1) > sc(X2)).astype(float) for sc in mlp_scorers], axis=0)
            d_side = clf_side.decision_function(Hc_) > clf_side.decision_function(Hb_)
            d_p0 = clf_p0.decision_function(P0_) > 0
            n = len(dv_tasks)
            rng = np.random.default_rng(SEED)
            mats = {"delta_only": d_del, "raw_joint_lr": d_sw, "mlp_swap": mlp_ok_arr,
                    "side_combo": d_side, "full_P0": d_p0}
            boots = {k: [] for k in mats}
            for k in range(300):
                idx = rng.choice(n, size=n, replace=True)
                for key, arr in mats.items():
                    boots[key].append(float(np.mean(arr[idx])))
            entry[mt] = {k: {"acc": float(np.mean(v)), "ci95": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]}
                         for k, v in boots.items()}
            entry[mt]["mlp_swap"]["n_seeds"] = len(SEEDS)
        per_edge[e[0]] = entry
        i = entry["instruct"]
        log(f"[r1p] {e[0]}: delta={i['delta_only']['acc']:.3f} rawLR={i['raw_joint_lr']['acc']:.3f} "
            f"mlp={i['mlp_swap']['acc']:.3f} side={i['side_combo']['acc']:.3f} p0={i['full_P0']['acc']:.3f}")

    # ---------- checkpoint/component 隔离支持表 ----------
    endpoints = {}
    for e in EDGES:
        for m in e[1:]:
            endpoints.setdefault(m, []).append(e[0])
    shared = {m: es for m, es in endpoints.items() if len(es) >= 2}
    iso_table = {}

    def fit_eval_subset(remain_names, e_eval):
        rows_sub = [blocks(e2, tr_tasks, m) for e2 in EDGES if e2[0] in remain_names for m in (0, 1)]
        S_s = np.concatenate([r[0] for r in rows_sub], 0); D_s = np.concatenate([r[1] for r in rows_sub], 0)
        y_s = np.concatenate([np.ones(len(D_s)), np.zeros(len(D_s))])
        clf = LogisticRegression(max_iter=1000, C=1.0).fit(
            np.concatenate([np.hstack([S_s, D_s]), np.hstack([S_s, -D_s])], 0), y_s)
        out = {}
        for mode, mt in ((0, "complete"), (1, "instruct")):
            S_, D_, _, _, _ = blocks(e_eval, dv_tasks, mode)
            out[mt] = float(np.mean(clf.decision_function(np.hstack([S_, D_])) > clf.decision_function(np.hstack([S_, -D_]))))
        return out

    for e in EDGES:
        incident = set()
        for m in e[1:]:
            incident.update(endpoints[m])
        held = incident - {e[0]}
        train_edges = [x[0] for x in EDGES if x[0] != e[0] and x[0] not in held]
        iso_table[e[0]] = {"shared_endpoints": [m for m in e[1:] if m in shared],
                           "held_incident_edges": sorted(held),
                           "remaining_train_edges": train_edges,
                           "isolated_rawLR_readout": fit_eval_subset(set(train_edges), e),
                           "status": "evaluable_descriptive" if len(train_edges) >= 1 else "insufficient"}
    results = {"schema": "r1_posttraining_v1", "generated_utc": datetime.now(timezone.utc).isoformat(),
               "report_first_line": "R1 后训练比较（逐边/模式/隔离；不升级因果）",
               "per_edge": per_edge, "checkpoint_isolation_table": iso_table,
               "shared_endpoints": {k: v for k, v in shared.items()},
               "math_note": ("线性 swap 分数差 f([S,D])−f([S,−D])=2·w_D·D（S 项与截距消去）——"
                             "线性 swap 读数不是 S-A 非线性交互证据；小 MLP 版为非线性对照。"),
               "protocol_note": "不取 max(p,1-p)；方向不依 dev 翻转；全部边保留（含 E1/E5 低值）。"}
    (OUT / "metrics_r1_posttraining.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    L = ["# R1 后训练比较（corrective；train/dev）", "",
         "| edge | delta_only(I) | raw_joint_lr(I) | mlp_swap(I) | side(I) | full_P0(I) |", "|---|---|---|---|---|---|"]
    for k, v in per_edge.items():
        i = v["instruct"]
        L.append(f"| {k} | {i['delta_only']['acc']:.3f} | {i['raw_joint_lr']['acc']:.3f} | "
                 f"{i['mlp_swap']['acc']:.3f} | {i['side_combo']['acc']:.3f} | {i['full_P0']['acc']:.3f} |")
    L += ["", "（complete 侧见 metrics；逐项 CI95 同文件）", "",
          "## 数学边界", f"- {results['math_note']}", "",
          "## checkpoint/component 隔离支持表", "", "| edge | 共享端点 | 移除 incident 后训练边 | isolated_rawLR(comp/inst) | 状态 |", "|---|---|---|---|---|"]
    for k, v in iso_table.items():
        ir = v["isolated_rawLR_readout"]
        L.append(f"| {k} | {','.join(v['shared_endpoints']) or '—'} | {len(v['remaining_train_edges'])} 条 | "
                 f"{ir['complete']:.3f}/{ir['instruct']:.3f} | {v['status']} |")
    L += ["", "> 全部边仍 partial；不得写 verified post-training effect 或 checkpoint-heldout 泛化。"]
    (OUT / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (OUT / "logs" / "r1p_run.log").write_text("\n".join(LOG) + "\n", encoding="utf-8")
    log(f"[r1p] done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
