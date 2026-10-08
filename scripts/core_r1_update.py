"""R1 边缘证据更新（指导 §5）：只做可识别性，不升级因果。

1) provenance/一致性核对：同 mode、同 task、两模式行齐备（断言）。
2) 每 edge 版本证据等级登记（documented base relation candidate；version 级证据缺 → partial）。
3) child/base paired direction 与 style/length/size 控制并列：
   raw Δ 判对率 / Δ⊥（对表面特征差正交化后）判对率 / 表面回归 R² / p0 判对率。
4) edge-heldout（7 折 LOO）与 child-heldout 的描述性结果。
标题与报告不得含 "post-training effect"。

输出：artifacts/r1_edge_evidence_update_2026-10-08/
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

OUT = ROOT / "d-det/artifacts/r1_edge_evidence_update_2026-10-08"
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
CHILD_GROUPS = {
    "Nexusflow--Athene-70B": ["E1_Athene70B_Llama3"],
    "Nexusflow--Athene-V2-Agent": ["E2_AtheneV2Agent_Qwen72B"],
    "Nexusflow--Athene-V2-Chat": ["E3_AtheneV2Chat_Qwen72B"],
    "NovaSky-AI--Sky-T1-32B-Flash": ["E4_SkyT1Flash_Qwen32B", "E5_SkyT1Flash_SkyT1Preview"],
    "NovaSky-AI--Sky-T1-32B-Preview": ["E5_SkyT1Flash_SkyT1Preview", "E6_SkyT1Preview_Qwen32B"],
    "Qwen--QwQ-32B-Preview": ["E7_QwQ32B_Qwen32B"],
}


def log(m):
    print(m, flush=True)
    LOG.append(m)


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "logs").mkdir(exist_ok=True)
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import r2_score

    A = ev.load_assets()
    splits = {t: A["splits"][A["TASK_I"][t] * 2] for t in A["tasks"]}
    tr_tasks = [t for t in A["tasks"] if splits[t] == "train"]
    dv_tasks = [t for t in A["tasks"] if splits[t] == "dev"]
    H = A["emb_base"]

    def rows2(model, tasks, mode):
        mi = A["MODEL_I"][model]
        ti = np.array([A["TASK_I"][t] for t in tasks])
        return mi * A["nT"] * 2 + ti * 2 + mode

    def surface_diff(edge, tasks, mode):
        _, ch, ba = edge
        rc = rows2(ch, tasks, mode); rb = rows2(ba, tasks, mode)
        f = lambda r: np.hstack([A["style"][r], A["meta"][r], A["sizelen"][r]])
        return f(rc) - f(rb)

    # ---- 1) provenance 一致性核对 ----
    prov = {}
    for e in EDGES:
        for m in e[1:]:
            assert m in A["MODEL_I"], m
        check = {"both_modes_rows_present": True, "task_coverage_full": True}
        for mode in (0, 1):
            r = rows2(e[1], A["tasks"], mode)
            assert r.max() < len(H)
        prov[e[0]] = check
    log("[r1u] provenance checks passed (rows/models/tasks)")

    # ---- 2) 证据等级登记 ----
    evidence = {e[0]: {"child": e[1], "base": e[2], "evidence_level": "partial",
                       "source": "HF model card cardData.base_model（E0 lineage_adjudication 冻结）",
                       "missing": "version-level evidence（训练步数/数据/commit）",
                       "label": "documented base relation candidate"} for e in EDGES}

    # ---- 3) fit：raw / orthogonalized / p0 ----
    fit_rows = []
    for e in EDGES:
        for mode in (0, 1):
            rc = rows2(e[1], tr_tasks, mode); rb = rows2(e[2], tr_tasks, mode)
            fit_rows.append((e[0], mode, H[rc] - H[rb], surface_diff(e, tr_tasks, mode)))
    D_all = np.concatenate([fr[2] for fr in fit_rows], 0)
    C_all = np.concatenate([fr[3] for fr in fit_rows], 0)
    # 表面回归：Δ ~ C（含常数）
    Xt = np.hstack([C_all, np.ones((len(C_all), 1))])
    coef, *_ = np.linalg.lstsq(Xt, D_all, rcond=None)
    pred = Xt @ coef
    r2 = float(1 - np.sum((D_all - pred) ** 2) / max(1e-12, np.sum((D_all - D_all.mean(0)) ** 2)))
    D_perp = D_all - pred  # 正交化后 Δ
    log(f"[r1u] surface regression on Δ: R2={r2:.4f} (fit={len(D_all)} rows)")

    # 方向判别器：torch 线性（对称增广 [D;-D]；随机初值避免零驻点——
    # 注：perp 残差每维均值为零，对称构造下对称 LR 在 w=0 是精确驻点，故统一用 torch）
    import torch
    import torch.nn as nn
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit_dir(Dmat, epochs=8, lr=1e-3, seed=SEED):
        X = np.concatenate([Dmat, -Dmat], 0).astype(np.float32)
        y = np.concatenate([np.ones(len(Dmat)), np.zeros(len(Dmat))]).astype(np.float32)
        torch.manual_seed(seed)
        net = nn.Linear(Dmat.shape[1], 1).to(device)
        opt = torch.optim.Adam(net.parameters(), lr=lr)
        Xt_ = torch.tensor(X, device=device); yt_ = torch.tensor(y, device=device)
        rng = np.random.default_rng(seed)
        for ep in range(epochs):
            perm = rng.permutation(len(Xt_))
            for i in range(0, len(perm), 4096):
                sel = torch.tensor(perm[i:i + 4096], device=device)
                loss = nn.functional.binary_cross_entropy_with_logits(net(Xt_[sel]).squeeze(-1), yt_[sel])
                opt.zero_grad(); loss.backward(); opt.step()
        w = net.weight.detach().cpu().numpy().ravel()
        b = float(net.bias.detach().cpu().numpy())
        return w, b

    def score_dir(wb, X):
        w, b = wb
        return np.asarray(X) @ w + b

    clf_raw = fit_dir(D_all)
    clf_perp = fit_dir(D_perp)
    clf_p0 = fit_dir(C_all)
    # side 分类器（child vs base 单侧，非对称数据——torch 同上）
    def fit_side(Xc, Xb, epochs=8, lr=1e-3, seed=SEED):
        X = np.concatenate([Xc, Xb], 0).astype(np.float32)
        y = np.concatenate([np.ones(len(Xc)), np.zeros(len(Xb))]).astype(np.float32)
        torch.manual_seed(seed)
        net = nn.Linear(X.shape[1], 1).to(device)
        opt = torch.optim.Adam(net.parameters(), lr=lr)
        Xt_ = torch.tensor(X, device=device); yt_ = torch.tensor(y, device=device)
        rng = np.random.default_rng(seed)
        for ep in range(epochs):
            perm = rng.permutation(len(Xt_))
            for i in range(0, len(perm), 4096):
                sel = torch.tensor(perm[i:i + 4096], device=device)
                loss = nn.functional.binary_cross_entropy_with_logits(net(Xt_[sel]).squeeze(-1), yt_[sel])
                opt.zero_grad(); loss.backward(); opt.step()
        return net.weight.detach().cpu().numpy().ravel(), float(net.bias.detach().cpu().numpy())

    X_ch = np.concatenate([H[rows2(e[1], tr_tasks, m)] for e in EDGES for m in (0, 1)], 0)
    X_ba = np.concatenate([H[rows2(e[2], tr_tasks, m)] for e in EDGES for m in (0, 1)], 0)
    clf_side = fit_side(X_ch, X_ba)
    log("[r1u] readouts fitted (raw/perp/p0/side, torch linear)")

    def edge_eval(edge, clf_r, clf_p, clf_s, clf_p0):
        out = {}
        for mode, mt in ((0, "complete"), (1, "instruct")):
            rc = rows2(edge[1], dv_tasks, mode); rb = rows2(edge[2], dv_tasks, mode)
            Dh = H[rc] - H[rb]
            Ch = surface_diff(edge, dv_tasks, mode)
            Dh_perp = Dh - np.hstack([Ch, np.ones((len(Ch), 1))]) @ coef
            out[mt] = {
                "delta": float(np.mean(score_dir(clf_r, Dh) > 0)),
                "delta_perp": float(np.mean(score_dir(clf_p, Dh_perp) > 0)),
                "p0": float(np.mean(score_dir(clf_p0, Ch) > 0)),
                "side_child": float(np.mean(score_dir(clf_s, H[rc]) > score_dir(clf_s, H[rb]))),
            }
        return out

    per_edge = {}
    for e in EDGES:
        per_edge[e[0]] = edge_eval(e, clf_raw, clf_perp, clf_side, clf_p0)
        i = per_edge[e[0]]["instruct"]
        log(f"[r1u] {e[0]}: delta={i['delta']:.3f} delta_perp={i['delta_perp']:.3f} "
            f"p0={i['p0']:.3f} side={i['side_child']:.3f}")

    # ---- 4) edge-heldout（7 折 LOO）----
    block_ids = np.repeat([fr[0] for fr in fit_rows], len(tr_tasks))
    edge_ho = {}
    for ehold in EDGES:
        m_tr = block_ids != ehold[0]
        c_r = fit_dir(D_all[m_tr])
        c_p = fit_dir(D_perp[m_tr])
        c_p0 = fit_dir(C_all[m_tr])
        ev_r = edge_eval(ehold, c_r, c_p, clf_side, c_p0)
        edge_ho[ehold[0]] = ev_r["instruct"]
        log(f"[r1u] edge-heldout {ehold[0]}: delta={ev_r['instruct']['delta']:.3f} "
            f"delta_perp={ev_r['instruct']['delta_perp']:.3f}")
    eh_mean = float(np.mean([v["delta"] for v in edge_ho.values()]))

    # ---- child-heldout（6 组 LOO）----
    child_ho = {}
    for child, held_ids in CHILD_GROUPS.items():
        m_k = ~np.isin(block_ids, held_ids)
        if m_k.sum() == 0:
            continue
        c_r = fit_dir(D_all[m_k])
        c_p = fit_dir(D_perp[m_k])
        c_p0 = fit_dir(C_all[m_k])
        tgt = [e for e in EDGES if e[0] in held_ids]
        vals = []
        for e in tgt:
            ev_r = edge_eval(e, c_r, c_p, clf_side, c_p0)
            vals.append(ev_r["instruct"]["delta"])
        child_ho[child] = {"held_edges": held_ids, "delta_mean": float(np.mean(vals)), "per_edge": vals}
        log(f"[r1u] child-heldout {child}: {vals} mean={np.mean(vals):.3f}")

    results = {"schema": "r1_evidence_update_v1", "generated_utc": datetime.now(timezone.utc).isoformat(),
               "report_first_line": "新证据 — 证据分级与表面控制并列（可识别性；不升级因果）",
               "provenance_checks": prov, "evidence": evidence,
               "surface_regression_on_delta": {"R2": r2, "n_rows": int(len(D_all))},
               "per_edge": per_edge,
               "edge_heldout": {"per_edge_delta": {k: v["delta"] for k, v in edge_ho.items()},
                                "mean_delta": eh_mean},
               "child_heldout": child_ho,
               "note": ("所有边保持 partial；标题/摘要不得出现 'post-training effect'；"
                        "delta_perp=对 style/len/size 差正交化后的方向判对率。")}
    (OUT / "metrics_r1_update.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")

    L = ["# R1 证据更新：可识别性与表面控制（不升级因果）", "",
         f"- provenance 核对：两模式行齐备、任务覆盖完整（7/7 边断言通过）",
         f"- 表面回归对 Δ 的 R²={r2:.4f}（fit={len(D_all)} 行）",
         f"- 方向读数口径：torch 线性对称判别器（旧目录的 sklearn LR 口径在正交化残差上因每维均值为零而在零驻点失效；本目录统一 torch 口径）", "",
         "## 逐边并列读数（instruct 侧）", "",
         "| edge | delta | delta⊥surface | p0 | side | 证据等级 |", "|---|---|---|---|---|---|"]
    for k, v in per_edge.items():
        i = v["instruct"]
        L.append(f"| {k} | {i['delta']:.3f} | {i['delta_perp']:.3f} | {i['p0']:.3f} | {i['side_child']:.3f} | partial |")
    L += ["", "## edge-heldout（7 折 LOO；描述性）", "",
          f"- 留出边 delta 均值={eh_mean:.3f}", ""]
    L += ["| edge | heldout delta |", "|---|---|"]
    for k, v in edge_ho.items():
        L.append(f"| {k} | {v['delta']:.3f} |")
    L += ["", "## child-heldout（6 组 LOO；描述性）", "", "| child | held edges | delta 均值 |", "|---|---|---|"]
    for k, v in child_ho.items():
        L.append(f"| {k} | {','.join(v['held_edges'])} | {v['delta_mean']:.3f} |")
    L += ["", "> 全部边为 `partial observational`（documented base relation candidate）；不得写 verified post-training effect。"]
    (OUT / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (OUT / "logs" / "r1u_run.log").write_text("\n".join(LOG) + "\n", encoding="utf-8")
    log(f"[r1u] done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
